from __future__ import annotations
# ============================================================
# 🗳️ Smart Elections: Universal JSON Election Results Parser
# ============================================================
import orjson
import os
import csv
import time
import re
from collections import defaultdict
from ...config import (
    OUTPUT_DIR
)
from ...Context_Integration.Context_Library.constants import (
    GROUP_RENAME_MAP, LOCATION_KEYWORDS, CANDIDATE_KEYWORDS, BALLOT_TYPES, PARTY_KEYWORDS, TOTAL_KEYWORDS,
    MISC_FOOTER_KEYWORDS, CONTEST_KEYWORDS, CONTEST_TITLE_SKIP_PHRASES
)
from ...utils.logger_singleton import logger, console, prompt
from ...utils.table_core import harmonize_headers_and_data
from ...utils.contest_selector import select_contest
from ...utils.table_builder import build_table_noninteractive
from ...utils.output_utils import finalize_election_output
from ...utils.shared_logic import safe_slug

# Add robust regex builder for contest keywords
def _build_contest_regex(keywords) -> re.Pattern:
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
_CONTEST_RX = _build_contest_regex(CONTEST_KEYWORDS)

def find_key_by_keywords(obj, keywords):
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

def _is_dict_list(x) -> bool:
    return isinstance(x, list) and x and all(isinstance(i, dict) for i in x)

def parse_json_election_results(json_path, session_id=None, coordinator=None):
    with open(json_path, "rb") as f:
        data = orjson.loads(f.read())

    # --- Resolve where "ballot items/contests" live ---
    results_obj = data.get("results")
    ballot_items = []
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
        return None, None, None, {"error": "No contests found"}

    # Select contest
    if len(contests) == 1:
        target_contest = next(iter(contests))
    else:
        selector_data = {
            "contests": [{"title": name} for name in sorted(contests)],
            "noisy_patterns": [s.lower() for s in (CONTEST_TITLE_SKIP_PHRASES or set())]
        }
        selected = select_contest(
            coordinator=coordinator,
            state=None, county=None, year=None,
            session_id=session_id,
            context={"selector_data": selector_data},
            allow_multiple=False,
            prompt_message="[PROMPT] Select contest (index, text, or 'cancel'): ",
            force_interactive=True,
            disable_ml_verify=False
        )
        if not selected:
            logger.error({
                "level": "ERROR",
                "type": "input",
                "message": "No contest selected.",
                "session_id": session_id
            })
            return None, None, None, {"error": "No contest selected"}
        target_contest = (selected[0] or {}).get("title") or next(iter(contests))

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
    rows = []
    headers = []
    if isinstance(contest_item, dict):
        ballot_options_key = find_key_by_keywords(contest_item, set(CANDIDATE_KEYWORDS) | {"ballotoption", "ballotoptions", "candidates"})
        ballot_options = contest_item.get(ballot_options_key, []) if ballot_options_key else []
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

            # Build normalization map for candidates/parties
            raw_candidates = {}
            for opt in ballot_options:
                raw = (opt.get(candidate_name_key, "") or "").strip()
                party = (opt.get(party_key, "") or "") if party_key else ""
                label = f"{raw} ({party})" if party else raw
                if raw:
                    raw_candidates[raw] = label

            normalization_map = {k: v for k, v in raw_candidates.items()}

            # Build nested results -> flat rows
            from collections import defaultdict
            results_nested = defaultdict(lambda: defaultdict(dict))
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
                for raw_label, methods in cands.items():
                    norm_label = normalization_map.get(raw_label, raw_label)
                    for method, count in methods.items():
                        col_name = f"{norm_label} - {method}"
                        row[col_name] = count
                        all_keys.add(col_name)
                rows.append(row)
            headers = ["Precinct"] + sorted(all_keys)

    # --- Fallback when schema isn't the expected nested structure ---
    if not rows:
        # Try to emit something useful rather than crash
        try:
            raw_blob = contest_item if contest_item is not None else ballot_items
            blob_str = orjson.dumps(raw_blob, option=orjson.OPT_INDENT_2)
            blob_str = blob_str.decode("utf-8", errors="ignore")
        except Exception:
            blob_str = str(contest_item or ballot_items)
        # Limit very large strings
        if len(blob_str) > 100_000:
            blob_str = blob_str[:100_000] + "\n... [truncated]"
        headers = ["Contest", "RawJSON"]
        rows = [{"Contest": target_contest, "RawJSON": blob_str}]

    # Harmonize/pivot via non-interactive builder
    headers, rows = harmonize_headers_and_data(headers, rows)

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
    context = {
        "contest": target_contest,
        "state": state,
        "county": county,
        "year": year,
        "session_id": session_id,
        "handler": "json_handler",
        "contest_slug": safe_slug(target_contest, 80),
        "source_slug": domain
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

    result = finalize_election_output(
        headers=headers_final,
        data=data_final,
        coordinator=coordinator,
        contest=target_contest,
        state=state,
        county=county,
        context={
            "handler": "json_handler",
            "input_file": os.path.basename(json_path),
            "session_id": session_id,
            "race": target_contest
        },
        enable_user_feedback=False,
        session_id=session_id
    )

    metadata = {
        "race": target_contest,
        "input_file": os.path.basename(json_path),
        "output_file": os.path.basename(result.get("csv_path", "")),
        "headers": headers_final,
        "row_count": len(data_final),
        "handler": "json_handler",
        "state": state,
        "county": county,
        "year": year,
        "csv_path": result.get("csv_path"),
        "metadata_path": result.get("metadata_path")
    }

    logger.info({
        "level": "INFO",
        "type": "output",
        "message": f"✅ Completed! Output CSV: {result.get('csv_path')}, Metadata: {result.get('metadata_path')}",
        "session_id": session_id
    })

    return headers_final, data_final, target_contest, metadata

def parse(page=None, coordinator=None, html_context=None, manual_file=None, session_id=None, **kwargs):
    """
    Universal pipeline entry: Accepts a JSON file path (manual_file) from the format router.
    Returns: headers, data, contest, metadata
    """
    html_context = html_context or {}
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

    # Defensive: always return a 4-tuple, never a bool
    if not (isinstance(result, tuple) and len(result) == 4):
        logger.error({
            "level": "ERROR",
            "type": "handler",
            "message": "[ERROR] Invalid result from parse_json_election_results (expected 4-tuple).",
            "session_id": session_id,
            "got_type": type(result).__name__
        })
        return None, None, None, {"error": "Invalid parse result"}
    return result