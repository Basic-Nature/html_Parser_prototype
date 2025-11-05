from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Set, Tuple, cast

import orjson

from ...Context_Integration.Context_Library.constants import (
    BALLOT_TYPES,
    CANDIDATE_KEYWORDS,
    CONTEST_KEYWORDS,
    CONTEST_TITLE_SKIP_PHRASES,
    GROUP_RENAME_MAP,
    LOCATION_KEYWORDS,
    PARTY_KEYWORDS,
)
from ...utils.contest_selector import (
    select_contest_auto_first,
)
from ...utils.location_helpers import (
    attach_precinct_column,
    collect_location_headers,
)
from ...utils.logger_singleton import logger
from ...utils.output_utils import finalize_election_output
from ...utils.pivot import expand_single_rawjson_row
from ...utils.shared_logic import safe_get, safe_slug
from ...utils.table_builder import build_table_noninteractive
from ...utils.table_core import robust_table_extraction

# ============================================================
# 🗳️ Smart Elections: Universal JSON Election Results Parser
# ============================================================

# Add robust regex builder for contest keywords
def _build_contest_regex(keywords: Iterable[str] | None) -> re.Pattern:
    """
    Build a tolerant regex that matches keyword phrases even with dots, hyphens, or extra separators.
    - Treat '.' as optional (e.g., 'u.s.' ~ 'us')
    - Treat '-' or space as interchangeable
    - Allow small separators between tokens
    """
    parts = []
    for phrase in (keywords or []):
        if not isinstance(phrase, str) or not phrase.strip():
            continue
        # token-wise transform
        toks = re.split(r"\s+", phrase.strip().lower())
        xtoks = []
        for t in toks:
            t = re.escape(t)
            t = t.replace(r"\.", r"\.?")     # periods optional (U.S. -> US)
            t = t.replace(r"\-", r"[-\s]?")  # hyphen/space optional
            xtoks.append(t)
        # allow flexible separators between tokens
        pat = r"(?:[\s\-_\/]*?)".join(xtoks)
        # wordish boundaries (letters/digits)
        pat = rf"(?<![A-Za-z0-9]){pat}(?![A-Za-z0-9])"
        parts.append(pat)
    if not parts:
        # fallback that never matches
        return re.compile(r"(?!x)x", re.I)
    return re.compile("|".join(parts), re.I)

# Precompile regex once
_CONTEST_RX: re.Pattern = _build_contest_regex(CONTEST_KEYWORDS)

def find_key_by_keywords(obj: Dict[str, Any] | Any, keywords: Iterable[str]) -> Optional[str]:
    """Find the first key in obj that matches any keyword (case-insensitive, partial match allowed)."""
    if not isinstance(obj, dict):
        return None
    for key in obj.keys():
        try:
            key_s = key.lower()
        except Exception:
            continue
        # First, regex hit for contest keywords if provided
        if _CONTEST_RX.search(key_s):
            return key
        # Fallback to simple substring for provided keywords
        for kw in keywords:
            if isinstance(kw, str) and kw and kw.lower() in key_s:
                return key
    return None

def _is_dict_list(x: Any) -> bool:
    # Ensure we return a strict boolean, not a possibly-empty list via short-circuit behavior
    return isinstance(x, list) and bool(x) and all(isinstance(i, dict) for i in x)

def parse_json_election_results(
    json_path: str,
    session_id: Optional[str] = None,
    coordinator: Any = None,
) -> Tuple[List[str], List[Dict[str, Any]], str, Dict[str, Any]]:
    with open(json_path, "rb") as f:
        data = orjson.loads(f.read())

    # --- Resolve where "ballot items/contests" live ---
    results_obj = data.get("results")
    ballot_items: Any = []
    if isinstance(results_obj, dict):
        ballot_items_key = find_key_by_keywords(
            results_obj,
            set(CONTEST_KEYWORDS) | {"ballotitem", "ballotitems", "contests", "races"}
        )
        ballot_items = results_obj.get(ballot_items_key, []) if ballot_items_key else []
    elif isinstance(results_obj, list):
        ballot_items = results_obj
    elif isinstance(data, dict):
        # Some exports put contests at the top level
        top_key = find_key_by_keywords(data, {"ballotitems", "contests", "races"})
        if top_key:
            ballot_items = data.get(top_key, []) or []

    if not isinstance(ballot_items, list):
        ballot_items = []

    # --- Build contest name set robustly ---
    contests = set()
    if _is_dict_list(ballot_items):
        exemplar = ballot_items[0]
        # prefer name/title keys, but let contest regex help when schema uses office-like keys
        contest_name_key = find_key_by_keywords(
            exemplar,
            set(CONTEST_KEYWORDS) | {"name", "title", "contest"}
        )
        for item in ballot_items:
            # try chosen key; otherwise scan likely name-like fields with regex
            name = (item.get(contest_name_key, "") or "").strip() if contest_name_key else ""
            if not name:
                # scan first stringy fields to find one containing a contest keyword
                for k, v in item.items():
                    if isinstance(v, str) and v.strip() and _CONTEST_RX.search(v.lower()):
                        name = v.strip()
                        break
            if name:
                contests.add(name)
    else:
        # If items are strings, assume contest titles directly
        for item in ballot_items:
            if isinstance(item, str) and item.strip():
                s = item.strip()
                if _CONTEST_RX.search(s.lower()):
                    contests.add(s)
                else:
                    contests.add(s)

    if not contests:
        logger.error({
            "level": "ERROR",
            "type": "input",
            "message": "No contests found in JSON.",
            "session_id": session_id
        })
        return [], [], "", {"error": "No contests found"}

    selection_context = {
        "selector_data": {
            "contests": [{"title": name} for name in sorted(contests)],
            "noisy_patterns": [s.lower() for s in (CONTEST_TITLE_SKIP_PHRASES or set())]
        },
        "input_file": os.path.basename(json_path)
    }

    auto_pick = select_contest_auto_first(
        coordinator=coordinator,
        context=selection_context,
        session_id=session_id,
        allow_multiple=False,
        force_interactive=False
    )
    if not auto_pick:
        logger.error({
            "level": "ERROR",
            "type": "input",
            "message": "No contest selected.",
            "session_id": session_id
        })
        return [], [], "", {"error": "No contest selected"}
    target_contest = safe_get(auto_pick[0], "title") or next(iter(contests))

    logger.info({
        "level": "INFO",
        "type": "input",
        "message": f"\n🔍 Parsing contest: {target_contest}\n",
        "session_id": session_id
    })

    # Locate the chosen contest item (dict if available)
    contest_item = None
    if _is_dict_list(ballot_items):
        for item in ballot_items:
            name_key = find_key_by_keywords(item, set(CONTEST_KEYWORDS) | {"name", "title", "contest"})
            if (item.get(name_key, "") or "").strip() == target_contest:
                contest_item = item
                break

    # --- Extract options/candidates and nested results when schema supports it ---
    rows: List[Dict[str, Any]] = []
    headers: List[str] = []
    normalization_map: Dict[str, str] = {}
    candidate_metadata: List[Dict[str, Any]] = []
    candidate_header_map: DefaultDict[str, Set[str]] = defaultdict(set)
    candidate_id_map: Dict[str, str] = {}
    if isinstance(contest_item, dict):
        ballot_options_key = find_key_by_keywords(
            contest_item,
            {"ballotoption", "ballotoptions", "candidates", "options", "choices"},
        )
        ballot_options = contest_item.get(ballot_options_key, []) if ballot_options_key else []

        if not _is_dict_list(ballot_options):
            # Fall back to the first list-of-dicts payload that looks candidate-like.
            for key, value in contest_item.items():
                if key == ballot_options_key:
                    continue
                if not _is_dict_list(value):
                    continue
                exemplar = value[0]
                if find_key_by_keywords(exemplar, set(CANDIDATE_KEYWORDS) | {"name", "label"}):
                    ballot_options_key = key
                    ballot_options = value
                    break
        if _is_dict_list(ballot_options):
            candidate_name_key = find_key_by_keywords(ballot_options[0], set(CANDIDATE_KEYWORDS) | {"name"})
            party_key = find_key_by_keywords(ballot_options[0], set(PARTY_KEYWORDS) | {"politicalparty"})
            precinct_results_key = find_key_by_keywords(
                ballot_options[0],
                set(LOCATION_KEYWORDS) | {"precinctresult", "precinctresults"}
            )
            group_results_key = find_key_by_keywords(
                ballot_options[0],
                set(BALLOT_TYPES) | {"groupresult", "groupresults"}
            )

            # Build normalization map and candidate metadata entries
            raw_candidates: Dict[str, str] = {}
            candidate_metadata = []
            candidate_id_map = {}
            for opt in ballot_options:
                raw = (opt.get(candidate_name_key, "") or "").strip()
                party = (opt.get(party_key, "") or "") if party_key else ""
                label = f"{raw} ({party})" if party else raw
                if raw:
                    raw_candidates[raw] = label
                meta = {
                    "id": opt.get("id"),
                    "raw_name": raw,
                    "display_label": label,
                    "party": party,
                    "ballot_order": opt.get("ballotOrder"),
                }
                candidate_metadata.append(meta)
                if opt.get("id") is not None and raw:
                    candidate_id_map[str(opt.get("id"))] = label

            normalization_map = {k: v for k, v in raw_candidates.items()}

            # Build nested results -> flat rows
            results_nested: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
            for opt in ballot_options:
                raw_label = (opt.get(candidate_name_key, "") or "").strip()
                if not raw_label:
                    continue
                precinct_results = opt.get(precinct_results_key, []) if precinct_results_key else []
                for precinct in precinct_results or []:
                    if not isinstance(precinct, dict):
                        continue
                    precinct_name_key = find_key_by_keywords(precinct, set(LOCATION_KEYWORDS) | {"name"})
                    p = (precinct.get(precinct_name_key, "") or "").strip()
                    if not p:
                        continue
                    results_nested[p][raw_label]["Total"] = precinct.get("voteCount")
                    group_results = precinct.get(group_results_key, []) if group_results_key else []
                    for grp in group_results or []:
                        if not isinstance(grp, dict):
                            continue
                        group_name_key = find_key_by_keywords(grp, set(BALLOT_TYPES) | {"groupname", "name"})
                        g = (grp.get(group_name_key, "") or "").strip()
                        norm_g = GROUP_RENAME_MAP.get(g.lower(), g) if g else g
                        results_nested[p][raw_label][norm_g or "Subtotal"] = grp.get("voteCount")

            all_keys = set()
            for precinct, cands in results_nested.items():
                row = {"Precinct": precinct}
                for raw_label, method_counts in cands.items():
                    norm_label = normalization_map.get(raw_label, raw_label)
                    for method, count in method_counts.items():
                        col_name = f"{norm_label} - {method}"
                        row[col_name] = count
                        all_keys.add(col_name)
                        candidate_header_map[norm_label].add(method)
                rows.append(row)
            headers = ["Precinct"] + sorted(all_keys)

            # Freeze candidate_header_map into JSON-serializable structure
    # --- Fallback when schema isn't the expected nested structure ---
    if not rows:
        # Try to emit something useful rather than crash
        try:
            raw_blob = contest_item if contest_item is not None else ballot_items
            blob_bytes = orjson.dumps(raw_blob, option=orjson.OPT_INDENT_2)
            blob_str = blob_bytes.decode("utf-8", errors="ignore")
        except Exception:
            blob_str = str(contest_item or ballot_items)
        # Limit very large strings
        if len(blob_str) > 100_000:
            blob_str = blob_str[:100_000] + "\n... [truncated]"
        headers = ["Contest", "RawJSON"]
        rows = [{"Contest": target_contest, "RawJSON": blob_str}]

    location_headers = collect_location_headers(headers)
    headers, rows, precinct_attached = attach_precinct_column(
        headers,
        rows,
        location_headers=location_headers,
    )
    location_diagnostics = {
        "detected_location_headers": location_headers,
        "precinct_attached": precinct_attached,
    }

    # Harmonize/pivot via non-interactive builder
    pre_builder_context = {
        "contest": target_contest,
        "handler": "json_handler",
        "phase": "pre_builder",
        "location_headers": location_headers,
        "precinct_attached": precinct_attached,
    }
    headers, rows = expand_single_rawjson_row(headers, rows, context=pre_builder_context)

    fname = os.path.basename(json_path).lower()
    state = "Unknown"
    county = "Unknown"
    for part in fname.replace(".json", "").split("_"):
        if "county" in part:
            county = part.replace("county", "").strip().title() + " County"
        if len(part) == 2 and part.isalpha():
            state = part.upper()
    m = re.search(r"(19|20)\d{2}", fname)
    year = int(m.group(0)) if m else None

    domain = safe_slug(os.path.basename(json_path))
    candidate_header_map_serializable: Dict[str, List[str]] = {}
    for label, methods in candidate_header_map.items():
        filtered = sorted({(m or "").strip() for m in methods if m})
        candidate_header_map_serializable[label] = filtered

    context = {
        "contest": target_contest,
        "state": state,
        "county": county,
        "year": year,
        "session_id": session_id,
        "handler": "json_handler",
        "contest_slug": safe_slug(target_contest, 80),
        "source_slug": domain,
        "candidate_label_map": normalization_map,
        "candidate_header_map": candidate_header_map_serializable,
        "candidate_metadata": candidate_metadata,
        "candidate_id_to_label": candidate_id_map,
        "location_headers": location_headers,
        "precinct_attached": precinct_attached,
        "location_diagnostics": location_diagnostics,
    }
    headers_final, data_final, _entity_info = build_table_noninteractive(
        domain=domain,
        headers=headers,
        data=rows,
        coordinator=coordinator,
        context=context,
        pivot_to_wide=True,
        debug=False
    )

    # Ensure the output context carries forward candidate metadata while dropping
    # non-serializable helpers (like the coordinator instance) before we hand it
    # to finalize_election_output for JSON serialization.
    export_context = {
        k: v for k, v in context.items()
        if k not in {"coordinator"}
    }
    export_context.update({
        "handler": "json_handler",
        "input_file": os.path.basename(json_path),
        "session_id": session_id,
        "race": target_contest,
        "location_headers": location_headers,
        "precinct_attached": precinct_attached,
    })

    finalized = finalize_election_output(
        headers=headers_final,
        data=data_final,
        coordinator=coordinator,
        contest=target_contest,
        state=state,
        county=county,
        context=export_context,
        enable_user_feedback=False,
        session_id=session_id
    )

    metadata = {
        "race": target_contest,
        "input_file": os.path.basename(json_path),
        "output_file": os.path.basename(finalized.get("csv_path", "")),
        "headers": headers_final,
        "row_count": len(data_final),
        "handler": "json_handler",
        "state": state,
        "county": county,
        "year": year,
        "csv_path": finalized.get("csv_path"),
        "metadata_path": finalized.get("metadata_path"),
        "candidate_label_map": normalization_map,
        "candidate_header_map": candidate_header_map_serializable,
        "candidate_metadata": candidate_metadata,
        "candidate_id_to_label": candidate_id_map,
        "location_headers_detected": location_headers,
        "precinct_attached": precinct_attached,
        "location_diagnostics": location_diagnostics,
    }

    logger.info({
        "level": "INFO",
        "type": "output",
        "message": f"✅ Completed! Output CSV: {finalized.get('csv_path')}, Metadata: {finalized.get('metadata_path')}",
        "session_id": session_id
    })

    return headers_final, data_final, target_contest, metadata

def parse(
    page: Any | None = None,
    coordinator: Any | None = None,
    html_context: Dict[str, Any] | None = None,
    manual_file: str | None = None,
    session_id: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[List[str] | None, List[Dict[str, Any]] | None, str | None, Dict[str, Any]]:
    """
    Universal pipeline entry: Accepts a JSON file path (manual_file) from the format router.
    Returns: headers, data, contest, metadata
    """
    html_context = html_context or {}
    # Parity guard: allow provided_tables + skip_pivot without manual_file
    provided_tables = html_context.get("provided_tables")
    if isinstance(provided_tables, list) and provided_tables:
        ctx = dict(html_context)
        ctx.update({
            "session_id": session_id,
            "coordinator": coordinator,
        })
        merged_headers, merged_rows = robust_table_extraction(page=None, extraction_context=ctx)

        contest = html_context.get("contest") or "Provided Tables"
        state = html_context.get("state") or "Unknown"
        county = html_context.get("county") or "Unknown"
        year = html_context.get("year")
        domain = html_context.get("source_slug") or safe_slug(contest)

        headers_final, data_final, _entity_info = build_table_noninteractive(
            domain=domain,
            headers=merged_headers,
            data=merged_rows,
            coordinator=coordinator,
            context={
                **ctx,
                "contest": contest,
                "state": state,
                "county": county,
                "year": year,
                "handler": "json_handler",
            },
            pivot_to_wide=not bool(html_context.get("skip_pivot")),
            debug=False,
        )

        finalized = finalize_election_output(
            headers=headers_final,
            data=data_final,
            coordinator=coordinator,
            contest=contest,
            state=state,
            county=county,
            context={
                "handler": "json_handler",
                "session_id": session_id,
                "race": contest,
                "provided_tables": True,
                "skip_pivot": bool(html_context.get("skip_pivot")),
            },
            enable_user_feedback=False,
            session_id=session_id,
        )

        metadata = {
            "race": contest,
            "input_file": html_context.get("input_file") or "<provided>",
            "output_file": os.path.basename(finalized.get("csv_path", "")),
            "headers": headers_final,
            "row_count": len(data_final),
            "handler": "json_handler",
            "state": state,
            "county": county,
            "year": year,
            "csv_path": finalized.get("csv_path"),
            "metadata_path": finalized.get("metadata_path"),
        }
        return headers_final, data_final, contest, metadata
    if html_context.get("skip_format") or html_context.get("manual_skip"):
        logger.info({
            "level": "INFO",
            "type": "handler",
            "message": "[SKIP] JSON parsing intentionally skipped via context flag.",
            "session_id": session_id
        })
        return None, None, None, {"skipped": True}

    if not manual_file or not os.path.isfile(manual_file):
        logger.error({
            "level": "ERROR",
            "type": "handler",
            "message": "[ERROR] No JSON file provided to parse().",
            "session_id": session_id
        })
        return None, None, None, {"skipped": True}

    logger.info({
        "level": "INFO",
        "type": "handler",
        "message": f"[INFO] Using JSON file: {manual_file}",
        "session_id": session_id
    })

    result = parse_json_election_results(manual_file, session_id=session_id, coordinator=coordinator)

    result_any = cast(Any, result)
    if not (isinstance(result_any, tuple) and len(result_any) == 4):
        logger.error({
            "level": "ERROR",
            "type": "handler",
            "message": "[ERROR] Invalid result from parse_json_election_results (expected 4-tuple).",
            "session_id": session_id,
            "got_type": type(result).__name__
        })
        return None, None, None, {"error": "Invalid parse result"}
    return cast(Tuple[List[str], List[Dict[str, Any]], str, Dict[str, Any]], result_any)