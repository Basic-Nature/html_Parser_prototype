from __future__ import annotations

import os
import re
from collections import Counter, defaultdict, OrderedDict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Set, Tuple, cast

import orjson

from ...Context_Integration.Context_Library.constants import (
    BALLOT_TYPES,
    BALLOT_TYPES_SORT_ORDER,
    CANDIDATE_KEYWORDS,
    CONTEST_KEYWORDS,
    CONTEST_TITLE_SKIP_PHRASES,
    DEFAULT_TOTAL_RESULT_DISPLAY,
    GROUP_RENAME_MAP,
    KNOWN_STATE_TO_COUNTY_MAP,
    LOCATION_KEYWORDS,
    PARTY_KEYWORDS,
    canonical_ballot_group,
)
from ...utils.salvage import normalize_ballot_column_name
from ...utils.contest_selector import (
    select_contest_auto_first,
)
from ...utils.json_export_loader import _ALL_COUNTIES_LABEL, load_json_export
from ...utils.location_helpers import (
    attach_precinct_column,
    collect_location_headers,
)
from ...utils.logger_singleton import logger
from ...utils.output_utils import finalize_election_output
from ...utils.pivot import expand_single_rawjson_row
from ...utils.shared_logic import (
    format_county_label,
    format_state_label,
    normalize_county_name,
    normalize_state_name,
    safe_get,
    safe_slug,
)
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


def _canonical_contest_key(title: str) -> str:
    if not isinstance(title, str):
        return ""
    normalized = re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()
    return re.sub(r"\s+", " ", normalized)


def _split_primary_title_for_grouping(title: str) -> tuple[str, str]:
    """Split an office title into (office, variant/locality) parts."""
    if not isinstance(title, str):
        return "", ""
    text = title.strip()
    if not text:
        return "", ""

    # Prefer explicit separators first
    for sep in (" - ", " – ", " — ", ":", " –", " —", "-", ","):
        if sep in text:
            head, tail = text.split(sep, 1)
            return head.strip(" ,:-"), tail.strip(" ,:-")

    # Heuristic: extract trailing district/circuit labels
    match = re.search(r"(District\s+[\w\-()#]+.*)$", text, re.IGNORECASE)
    if match and match.start() > 0:
        office = text[:match.start()].strip(" ,:-")
        variant = match.group(1).strip()
        if office:
            return office, variant

    match = re.search(r"(Circuit\s+-?\s+.+)$", text, re.IGNORECASE)
    if match and match.start() > 0:
        office = text[:match.start()].strip(" ,:-")
        variant = match.group(1).strip()
        if office:
            return office, variant

    return text, ""


def _format_county_preview(counties: Iterable[str], limit: int = 3, scope_hint: str | None = None) -> tuple[str, str]:
    """Return (label, preview) for a counties list, truncated for readability."""
    cleaned: List[str] = []
    for county in counties or []:
        if isinstance(county, str):
            county_s = county.strip()
            if county_s:
                cleaned.append(county_s)
    if not cleaned:
        return "", ""
    label = f"{len(cleaned)} county" if len(cleaned) == 1 else f"{len(cleaned)} counties"
    normalized_scope = (scope_hint or "").strip().lower()

    if normalized_scope == "statewide" or len(cleaned) >= 25:
        preview = "Statewide" if normalized_scope == "statewide" else f"{cleaned[0]}, +{len(cleaned) - 1} more"
        return label, preview

    if len(cleaned) <= limit:
        return label, ", ".join(cleaned)

    preview_items = cleaned[:limit]
    preview = ", ".join(preview_items)
    remaining = len(cleaned) - limit
    preview = f"{preview}, +{remaining} more"
    return label, preview


def _format_scope_label(scopes: Iterable[str]) -> str:
    """Convert division scopes into a single human-readable label."""
    cleaned: List[str] = []
    seen: Set[str] = set()
    for scope in scopes or []:
        if not scope:
            continue
        scope_s = scope.replace("_", " ").strip().lower()
        if not scope_s or scope_s == "unknown" or scope_s in seen:
            continue
        seen.add(scope_s)
        if scope_s == "statewide":
            cleaned.append("Statewide")
        elif scope_s == "single-county":
            cleaned.append("Single County")
        else:
            cleaned.append(scope_s.title())
    return " | ".join(cleaned)


def _collect_contest_groups(export) -> List[Dict[str, Any]]:
    groups: Dict[str, Dict[str, Any]] = {}
    for contest_id, coverage in (export.coverage or {}).items():
        name = coverage.contest_name or str(contest_id)
        key = _canonical_contest_key(name)
        bucket = groups.setdefault(key, {
            "title_samples": [],
            "contest_ids": set(),
            "division_scopes": set(),
            "division_identifiers": set(),
            "counties": set(),
            "vote_for": set(),
            "contest_types": set(),
            "ballot_items": [],
            "questions": set(),
            "raw_titles": set(),
        })
        bucket["title_samples"].append(name)
        bucket["contest_ids"].add(str(contest_id))
        if getattr(coverage, "division_scope", None):
            bucket["division_scopes"].add(coverage.division_scope)
        if getattr(coverage, "division_identifier", None):
            bucket["division_identifiers"].add(coverage.division_identifier)
        bucket["counties"].update(getattr(coverage, "counties_in_scope", []) or [])
        if getattr(coverage, "vote_for", None) is not None:
            bucket["vote_for"].add(coverage.vote_for)
        if getattr(coverage, "contest_type", None):
            bucket["contest_types"].add(coverage.contest_type)
        if getattr(coverage, "contest_question", None):
            bucket["questions"].add(coverage.contest_question)
        if getattr(coverage, "contest_name_raw", None):
            bucket["raw_titles"].add(coverage.contest_name_raw)
        ballot_meta = {
            "contest_id": str(contest_id),
            "vote_for": coverage.vote_for,
            "contest_type": coverage.contest_type,
            "division_scope": coverage.division_scope,
            "division_identifier": coverage.division_identifier,
            "counties_in_scope": list(getattr(coverage, "counties_in_scope", []) or []),
        }
        bucket["ballot_items"].append(ballot_meta)

    group_list: List[Dict[str, Any]] = []
    for key, data in groups.items():
        title_counts = Counter(data["title_samples"] or ["Election Results"])
        primary_title = title_counts.most_common(1)[0][0]
        contest_ids = sorted(data["contest_ids"])
        division_scopes = sorted(scope for scope in data["division_scopes"] if scope and scope != "unknown")
        metadata = {
            "contest_ids": contest_ids,
            "primary_title": primary_title,
            "counties": sorted(data["counties"]),
            "division_scopes": division_scopes,
            "division_identifiers": sorted(data["division_identifiers"]),
            "variants": len(contest_ids),
            "vote_for": sorted(data["vote_for"]),
            "contest_types": sorted(data["contest_types"]),
            "ballot_items": data["ballot_items"],
        }
        questions = sorted(q for q in data["questions"] if q)
        if questions:
            metadata["questions"] = questions
            if "question" not in metadata and questions:
                metadata["question"] = questions[0]
        raw_titles = sorted(t for t in data["raw_titles"] if t)
        if raw_titles:
            metadata["raw_titles"] = raw_titles
        summary_parts: List[str] = []
        if metadata["variants"] > 1:
            summary_parts.append(f"{metadata['variants']} variants")
        if metadata["counties"]:
            summary_parts.append(f"{len(metadata['counties'])} counties")
        if division_scopes:
            summary_parts.append("/".join(division_scopes))

        office_title, variant_label = _split_primary_title_for_grouping(primary_title)
        scope_label = _format_scope_label(division_scopes)
        county_label, county_preview = _format_county_preview(metadata["counties"], scope_hint=scope_label)

        metadata["office_title"] = office_title or primary_title
        metadata["variant_label"] = variant_label
        if scope_label:
            metadata["scope_label"] = scope_label
        if county_label:
            metadata["county_label"] = county_label
        if county_preview:
            metadata["county_preview"] = county_preview

        detail_segments: List[str] = []
        if county_label:
            detail_segments.append(county_label + (f": {county_preview}" if county_preview else ""))
        if scope_label and scope_label.lower() not in (variant_label or "").lower():
            detail_segments.append(scope_label)
        if metadata["variants"] > 1:
            detail_segments.append(f"{metadata['variants']} contest ids")

        base_office = office_title or primary_title
        header_label = base_office
        if variant_label:
            header_label = f"{base_office} – {variant_label}"
        elif scope_label and scope_label.lower() != base_office.lower():
            header_label = f"{base_office} – {scope_label}"

        display_title = header_label
        if detail_segments:
            display_title = f"{header_label} [{' | '.join(detail_segments)}]"

        metadata["summary"] = summary_parts
        metadata["display_title"] = display_title
        metadata["display_header"] = header_label
        metadata["display_details"] = detail_segments
        group_list.append({
            "key": key,
            "primary_title": primary_title,
            "display_title": display_title,
            "contest_ids": contest_ids,
            "metadata": metadata,
        })

    group_list.sort(key=lambda g: g["primary_title"].lower())
    return group_list

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


def _state_key_for_county(county: Optional[str]) -> Optional[str]:
    county_norm = normalize_county_name(county) if county else None
    if not county_norm:
        return None
    for state_key, counties in KNOWN_STATE_TO_COUNTY_MAP.items():
        for candidate in counties:
            if normalize_county_name(candidate) == county_norm:
                return state_key
    return None


def _extract_first_str(obj: Dict[str, Any], *keys: str) -> str:
    for key in keys:
        val = obj.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _derive_location_metadata(payload: Dict[str, Any]) -> tuple[str, str]:
    """Infer state and county labels from the JSON payload."""
    state_candidate = ""
    county_candidate = ""
    results_obj = payload.get("results") if isinstance(payload, dict) else None
    if isinstance(results_obj, dict):
        county_candidate = _extract_first_str(
            results_obj,
            "county",
            "countyName",
            "county_name",
            "jurisdictionName",
        ) or _extract_first_str(results_obj, "name")
        state_candidate = _extract_first_str(results_obj, "state", "stateName", "state_name")

    if not county_candidate and isinstance(payload, dict):
        county_candidate = _extract_first_str(payload, "county", "countyName", "county_name")
    if not state_candidate and isinstance(payload, dict):
        state_candidate = _extract_first_str(payload, "state", "stateName", "state_name")

    if not state_candidate and county_candidate:
        state_key = _state_key_for_county(county_candidate)
        if state_key:
            state_candidate = state_key

    return format_state_label(state_candidate), format_county_label(county_candidate, state_candidate)


def _fastpath_county_results(
    json_path: str,
    payload: Dict[str, Any],
    session_id: Optional[str],
    coordinator: Any,
) -> Optional[Tuple[List[str], List[Dict[str, Any]], str, Dict[str, Any]]]:
    if not isinstance(payload, dict) or not payload.get("localResults"):
        return None

    try:
        export = load_json_export(Path(json_path))
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning({
            "level": "WARNING",
            "type": "handler",
            "message": f"Fast-path json_export_loader failed: {exc}",
            "session_id": session_id,
        })
        return None

    if not export.county_rows and not export.statewide_rows:
        return None

    fname = os.path.basename(json_path).lower()
    fallback_state = ""
    fallback_county = ""
    for part in fname.replace(".json", "").split("_"):
        if "county" in part and not fallback_county:
            fallback_county = part.replace("county", "").strip()
        if len(part) == 2 and part.isalpha() and not fallback_state:
            fallback_state = part.upper()

    derived_state, derived_county = _derive_location_metadata(payload)
    state = derived_state or format_state_label(fallback_state)
    county = derived_county or format_county_label(fallback_county, state)
    state = state or "Unknown"
    county = county or "Unknown"

    m = re.search(r"(19|20)\d{2}", fname)
    year = int(m.group(0)) if m else None

    contest_groups = _collect_contest_groups(export)
    if not contest_groups:
        return None

    selected_group = contest_groups[0]
    selected_metadata: Dict[str, Any] = dict(selected_group["metadata"])
    selected_ids: Set[str] = set(selected_group["contest_ids"])

    if len(contest_groups) > 1:
        selector_entries = []
        for group in contest_groups:
            selector_entries.append({
                "title": group["display_title"],
                "primary_title": group["primary_title"],
                "contest_ids": group["contest_ids"],
                "group_metadata": group["metadata"],
                "summary": group["metadata"].get("summary"),
            })
        selection_context = {
            "selector_data": {"contests": selector_entries},
            "handler": "json_handler",
            "input_file": os.path.basename(json_path),
            "state": state,
            "county": county,
            "year": year,
            "webapp": True,
        }
        choice = select_contest_auto_first(
            coordinator=coordinator,
            context=selection_context,
            session_id=session_id,
            allow_multiple=False,
            force_interactive=True,
        )
        if not choice:
            logger.info({
                "level": "INFO",
                "type": "handler",
                "message": "JSON contest selection cancelled; falling back to standard pipeline.",
                "session_id": session_id,
            })
            return None
        picked = choice[0]
        choice_meta: Dict[str, Any] = dict(picked.get("metadata") or {})
        selected_ids = set(choice_meta.get("contest_ids") or [])
        if not selected_ids:
            picked_title = picked.get("title")
            for group in contest_groups:
                if group["display_title"] == picked_title:
                    selected_ids = set(group["contest_ids"])
                    choice_meta.setdefault("contest_ids", group["contest_ids"])
                    choice_meta.setdefault("primary_title", group["primary_title"])
                    choice_meta.setdefault("display_title", group["display_title"])
                    choice_meta.setdefault("summary", group["metadata"].get("summary"))
                    break
        for group in contest_groups:
            if set(group["contest_ids"]) == selected_ids:
                selected_group = group
                break
        selected_metadata = {**selected_group["metadata"], **choice_meta}
        if "contest_ids" not in selected_metadata:
            selected_metadata["contest_ids"] = selected_group["contest_ids"]

    if not selected_ids:
        selected_ids = set(selected_group["contest_ids"])

    contest_name = selected_metadata.get("primary_title") or selected_group["primary_title"] or "Election Results"

    base_headers = [
        "Division",
        "Candidate",
        "Party",
    ]

    selected_county_rows = [row for row in export.county_rows if row.contest_id in selected_ids]
    selected_statewide_rows = [row for row in export.statewide_rows if row.contest_id in selected_ids]

    if not selected_statewide_rows and not any(row.county == _ALL_COUNTIES_LABEL for row in export.county_rows):
        all_ids = {row.contest_id for row in export.statewide_rows}
        if all_ids and selected_ids != all_ids:
            logger.info({
                "level": "INFO",
                "type": "handler",
                "message": "Selected contest lacks statewide totals; retrying with full statewide grouping.",
                "session_id": session_id,
            })
            selected_ids = all_ids
            selected_county_rows = [row for row in export.county_rows if row.contest_id in selected_ids]
            selected_statewide_rows = list(export.statewide_rows)
            for group in contest_groups:
                if set(group["contest_ids"]) == selected_ids:
                    selected_group = group
                    selected_metadata = dict(group["metadata"])
                    break

    if not selected_county_rows and not selected_statewide_rows:
        logger.warning({
            "level": "WARNING",
            "type": "handler",
            "message": "Selected contest did not yield any result rows; aborting fast-path.",
            "session_id": session_id,
        })
        return None

    bundle_audit: Optional[Dict[str, Any]] = None
    bundle_metadata: Optional[Dict[str, Any]] = None
    bundle_summary_text: Optional[str] = None
    if selected_metadata.get("bundle_mode") == "aggregate":
        row_counts = Counter()
        for record in selected_county_rows + selected_statewide_rows:
            if getattr(record, "contest_id", None) is not None:
                row_counts[str(record.contest_id)] += 1

        raw_members = selected_metadata.get("bundle_members") or []
        member_entries: List[Dict[str, Any]] = []
        aggregated_ids: Set[str] = set()
        for member in raw_members:
            if not isinstance(member, dict):
                continue
            meta = dict(member.get("metadata") or {})
            member_ids = meta.get("contest_ids") or meta.get("bundle_contest_ids") or []
            member_ids_list = [str(cid) for cid in member_ids if cid is not None]
            if member_ids_list:
                aggregated_ids.update(member_ids_list)
            entry = {
                "title": member.get("title"),
                "contest_ids": member_ids_list,
                "variant_label": meta.get("variant_label"),
                "scope_label": meta.get("scope_label"),
                "county_label": meta.get("county_label"),
                "county_preview": meta.get("county_preview"),
                "division_scopes": list(meta.get("division_scopes") or []),
                "vote_for": meta.get("vote_for"),
            }
            entry["row_count"] = int(sum(row_counts.get(cid, 0) for cid in member_ids_list)) if member_ids_list else 0
            member_entries.append(entry)

        if not member_entries:
            fallback_ids = aggregated_ids or {str(cid) for cid in selected_ids}
            fallback_row_count = int(sum(row_counts.get(cid, 0) for cid in fallback_ids)) if fallback_ids else 0
            member_entries.append({
                "title": contest_name,
                "contest_ids": sorted(fallback_ids),
                "row_count": fallback_row_count,
            })
            aggregated_ids.update(fallback_ids)

        bundle_size = int(selected_metadata.get("bundle_size") or len(member_entries) or len(selected_ids))
        summary_field = selected_metadata.get("summary")
        if isinstance(summary_field, (list, tuple)):
            summary_text = " | ".join(str(item) for item in summary_field if item)
        elif isinstance(summary_field, str):
            summary_text = summary_field
        else:
            summary_text = ""
        display_details = selected_metadata.get("display_details")
        if isinstance(display_details, (list, tuple)):
            detail_text = " | ".join(str(item) for item in display_details if item)
        elif isinstance(display_details, str):
            detail_text = display_details
        else:
            detail_text = ""
        bundle_summary_text = selected_metadata.get("display_title") or selected_metadata.get("primary_title") or contest_name
        bundle_audit = {
            "bundle_key": selected_metadata.get("bundle_key"),
            "bundle_mode": "aggregate",
            "bundle_size": bundle_size,
            "contest_ids": sorted(aggregated_ids or {str(cid) for cid in selected_ids}),
            "summary": summary_text,
            "details": detail_text,
            "display_title": bundle_summary_text,
            "members": member_entries,
            "row_count_total": int(sum(entry.get("row_count", 0) for entry in member_entries)),
        }
        bundle_metadata = {
            "bundle_key": selected_metadata.get("bundle_key"),
            "bundle_mode": "aggregate",
            "bundle_size": bundle_size,
            "contest_ids": sorted(aggregated_ids or {str(cid) for cid in selected_ids}),
            "summary": summary_text,
            "display_title": bundle_summary_text,
            "details": detail_text,
            "members": member_entries,
        }
        if raw_members:
            bundle_metadata["raw_members_count"] = len(raw_members)
        selected_metadata["bundle_audit"] = bundle_audit

    candidate_header_map_raw: DefaultDict[str, Set[str]] = defaultdict(set)
    candidate_id_to_label: Dict[str, str] = {}
    candidate_metadata_map: Dict[str, Dict[str, Any]] = {}
    candidate_label_map: Dict[str, str] = {}

    for record in selected_county_rows + selected_statewide_rows:
        candidate_header_map_raw[record.candidate_name].add(record.group_display)
        if record.candidate_id:
            candidate_id_to_label.setdefault(record.candidate_id, record.candidate_name)
        candidate_key = record.candidate_id or record.candidate_name.lower()
        if candidate_key not in candidate_metadata_map:
            candidate_metadata_map[candidate_key] = {
                "id": record.candidate_id,
                "raw_name": record.candidate_raw,
                "display_label": record.candidate_name,
                "party": record.party_full,
            }
        candidate_label_map.setdefault(record.candidate_raw, record.candidate_name)

    candidate_metadata = list(candidate_metadata_map.values())

    REPORTED_TOTAL_COLUMN = "Reported Vote Total"
    CALCULATED_TOTAL_COLUMN = "Calculated Vote Total"

    def _coerce_vote_value(value: Any) -> int:
        if value in (None, "", "NA"):
            return 0
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(value)
        try:
            text = str(value).strip()
            if not text:
                return 0
            if text.endswith("%"):
                text = text[:-1].strip()
            text = text.replace(",", "")
            if not text:
                return 0
            return int(float(text))
        except Exception:
            return 0

    order_lookup = {canonical_ballot_group(name).lower(): idx for idx, name in enumerate(BALLOT_TYPES_SORT_ORDER)}
    ballot_display_map: OrderedDict[str, str] = OrderedDict()
    aggregated_rows: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    aggregated_order: List[Tuple[str, str, str]] = []

    for record in selected_county_rows + selected_statewide_rows:
        division = record.county
        candidate = record.candidate_name
        party = record.party_full
        key = (division, candidate, party)
        if key not in aggregated_rows:
            aggregated_rows[key] = {
                "Division": division,
                "Candidate": candidate,
                "Party": party,
            }
            aggregated_order.append(key)

        entry = aggregated_rows[key]
        display_label = (record.group_display or "").strip()
        canonical_label = canonical_ballot_group(display_label) if display_label else ""

        if record.group_name == "total" or canonical_label.lower() == "total":
            entry[REPORTED_TOTAL_COLUMN] = entry.get(REPORTED_TOTAL_COLUMN, 0) + record.vote_count
            continue

        if canonical_label:
            preferred_label = (display_label or canonical_label or "").strip() or canonical_label
            existing = ballot_display_map.get(canonical_label)
            if existing is None:
                ballot_display_map[canonical_label] = preferred_label
            else:
                existing_text = (existing or "").strip().lower()
                canonical_text = (canonical_label or "").strip().lower()
                preferred_text = (preferred_label or "").strip().lower()
                if existing_text == canonical_text and preferred_text and preferred_text != canonical_text:
                    ballot_display_map[canonical_label] = preferred_label
            column_label = ballot_display_map.get(canonical_label) or preferred_label or canonical_label
        else:
            column_label = display_label or canonical_label

        if column_label:
            entry[column_label] = entry.get(column_label, 0) + record.vote_count
        else:
            entry[REPORTED_TOTAL_COLUMN] = entry.get(REPORTED_TOTAL_COLUMN, 0) + record.vote_count

    def _ballot_sort_key(item: Tuple[str, str]) -> Tuple[int, str]:
        canon, label = item
        return (order_lookup.get(canon.lower(), len(order_lookup)), label.lower())

    ballot_headers: List[str] = [label for canon, label in sorted(ballot_display_map.items(), key=_ballot_sort_key)]

    headers = base_headers + [REPORTED_TOTAL_COLUMN] + ballot_headers
    rows: List[Dict[str, Any]] = []
    numeric_columns = set(ballot_headers + [REPORTED_TOTAL_COLUMN])

    for key in aggregated_order:
        entry = aggregated_rows[key]
        if REPORTED_TOTAL_COLUMN not in entry or entry.get(REPORTED_TOTAL_COLUMN) in (None, ""):
            computed_total = sum(entry.get(col, 0) for col in ballot_headers)
            entry[REPORTED_TOTAL_COLUMN] = computed_total
        row = {}
        for column in headers:
            if column in numeric_columns:
                row[column] = entry.get(column, 0)
            else:
                row[column] = entry.get(column, "")
        rows.append(row)

    candidate_header_map: DefaultDict[str, Set[str]] = defaultdict(set)
    for row in rows:
        candidate_name = str(row.get("Candidate") or "").strip()
        if not candidate_name:
            continue
        if row.get(REPORTED_TOTAL_COLUMN) not in (None, "", 0):
            candidate_header_map[candidate_name].add(REPORTED_TOTAL_COLUMN)
        if row.get(CALCULATED_TOTAL_COLUMN) not in (None, "", 0):
            candidate_header_map[candidate_name].add(CALCULATED_TOTAL_COLUMN)
        for ballot_label in ballot_headers:
            if row.get(ballot_label) not in (None, "", 0):
                candidate_header_map[candidate_name].add(ballot_label)

    for label, original_methods in candidate_header_map_raw.items():
        for method in original_methods:
            if not method:
                continue
            if method == DEFAULT_TOTAL_RESULT_DISPLAY:
                candidate_header_map[label].add(REPORTED_TOTAL_COLUMN)
            else:
                candidate_header_map[label].add(method)

    candidate_header_map_serializable = {
        label: sorted({m for m in methods if m})
        for label, methods in candidate_header_map.items()
    }

    location_headers = ["Division"]
    location_diagnostics = {
        "detected_location_headers": location_headers,
        "precinct_attached": False,
    }

    domain = safe_slug(os.path.basename(json_path))

    selected_coverage = {
        cid: export.coverage[cid]
        for cid in selected_ids
        if cid in export.coverage
    }
    contest_question = selected_metadata.get("question")
    if not contest_question:
        for coverage_entry in selected_coverage.values():
            if getattr(coverage_entry, "contest_question", None):
                contest_question = coverage_entry.contest_question
                break
    contest_name_raw = selected_metadata.get("raw_titles")
    if isinstance(contest_name_raw, list):
        contest_name_raw = contest_name_raw[0] if contest_name_raw else None
    if not contest_name_raw:
        for coverage_entry in selected_coverage.values():
            if getattr(coverage_entry, "contest_name_raw", None):
                contest_name_raw = coverage_entry.contest_name_raw
                break
    coverage_serializable = {
        cid: data.__dict__
        for cid, data in selected_coverage.items()
    }
    total_counties = len(selected_metadata.get("counties", [])) or export.total_counties

    context_snapshot = dict(export.context_snapshot or {})
    context_snapshot["selected_contest_ids"] = sorted(selected_ids)
    context_snapshot["selected_contest_title"] = contest_name
    context_snapshot["selected_contest_summary"] = selected_metadata
    if contest_question:
        context_snapshot["selected_contest_question"] = contest_question
    if contest_name_raw and contest_name_raw != contest_name:
        context_snapshot["selected_contest_name_raw"] = contest_name_raw
    if bundle_metadata:
        context_snapshot["bundle_mode"] = "aggregate"
        context_snapshot["bundle_metadata"] = bundle_metadata
        if bundle_audit:
            context_snapshot["bundle_audit"] = bundle_audit

    context = {
        "contest": contest_name,
        "state": state,
        "county": county,
        "year": year,
        "session_id": session_id,
        "handler": "json_handler",
        "contest_slug": safe_slug(contest_name, 80),
        "source_slug": domain,
        "candidate_label_map": candidate_label_map,
        "candidate_header_map": candidate_header_map_serializable,
        "candidate_metadata": candidate_metadata,
        "candidate_id_to_label": candidate_id_to_label,
        "location_headers": location_headers,
        "precinct_attached": False,
        "location_diagnostics": location_diagnostics,
        "coverage": coverage_serializable,
        "context_snapshot": context_snapshot,
        "selected_contest_ids": sorted(selected_ids),
        "selected_contest_summary": selected_metadata,
        "contest_question": contest_question,
        "contest_name_raw": contest_name_raw,
    }
    if bundle_metadata:
        context["bundle_mode"] = "aggregate"
        context["bundle_metadata"] = bundle_metadata
        context["bundle_summary"] = bundle_summary_text
        context["bundle_key"] = bundle_metadata.get("bundle_key")
        context["bundle_size"] = bundle_metadata.get("bundle_size")
        if bundle_audit:
            context["bundle_audit"] = bundle_audit


    headers_final, data_final, _entity_info = build_table_noninteractive(
        domain=domain,
        headers=headers,
        data=rows,
        coordinator=coordinator,
        context=context,
        pivot_to_wide=False,
        debug=False,
    )

    ballot_columns_detected: List[str] = []
    for column in headers_final:
        norm = normalize_ballot_column_name(column)
        if not norm:
            continue
        norm_low = norm.lower()
        if norm_low in {"total vote", "total votes", "grand total", "reported vote total", "votes", CALCULATED_TOTAL_COLUMN.lower()}:
            continue
        if norm in BALLOT_TYPES or norm in BALLOT_TYPES_SORT_ORDER or norm in GROUP_RENAME_MAP.values():
            ballot_columns_detected.append(column)

    if "Total Vote" in headers_final:
        headers_final = [col for col in headers_final if col != "Total Vote"]
        for row in data_final:
            row.pop("Total Vote", None)

    if ballot_columns_detected:
        for row in data_final:
            calculated_total = sum(_coerce_vote_value(row.get(col)) for col in ballot_columns_detected)
            row[CALCULATED_TOTAL_COLUMN] = calculated_total
        if CALCULATED_TOTAL_COLUMN not in headers_final:
            insert_idx = headers_final.index(REPORTED_TOTAL_COLUMN) + 1 if REPORTED_TOTAL_COLUMN in headers_final else len(headers_final)
            headers_final = headers_final[:insert_idx] + [CALCULATED_TOTAL_COLUMN] + headers_final[insert_idx:]
        context.setdefault("ballot_columns_detected", ballot_columns_detected)

    if REPORTED_TOTAL_COLUMN in headers_final and ballot_columns_detected:
        context.setdefault("ballot_columns_tracked", ballot_columns_detected)

    metric_columns = [
        column
        for column in headers_final
        if column not in {"Division", "Precinct", "Candidate", "Party", "County"}
    ]
    candidate_header_map_post: DefaultDict[str, Set[str]] = defaultdict(set)
    for row in data_final:
        candidate_name = str(row.get("Candidate") or "").strip()
        if not candidate_name:
            continue
        for column in metric_columns:
            value = row.get(column)
            if value not in (None, "", 0):
                candidate_header_map_post[candidate_name].add(column)
    if candidate_header_map_post:
        candidate_header_map_serializable = {
            label: sorted(columns)
            for label, columns in candidate_header_map_post.items()
        }

    if REPORTED_TOTAL_COLUMN in headers_final or CALCULATED_TOTAL_COLUMN in headers_final:
        preferred_order: List[str] = []
        for base in ("Division", "Precinct", "County"):
            if base in headers_final and base not in preferred_order:
                preferred_order.append(base)
        for base in ("Candidate", "Party"):
            if base in headers_final and base not in preferred_order:
                preferred_order.append(base)
        appendables = [col for col in (REPORTED_TOTAL_COLUMN, CALCULATED_TOTAL_COLUMN) if col in headers_final]
        preferred_order.extend(col for col in appendables if col not in preferred_order)
        preferred_order.extend(
            col for col in headers_final
            if col not in preferred_order
        )
        if preferred_order != headers_final:
            headers_final = preferred_order
            reordered_rows: List[Dict[str, Any]] = []
            for row in data_final:
                reordered_rows.append({col: row.get(col, "") for col in headers_final})
            data_final = reordered_rows

    export_context = {
        **{k: v for k, v in context.items() if k != "coordinator"},
        "handler": "json_handler",
        "input_file": os.path.basename(json_path),
        "race": contest_name,
        "location_headers": location_headers,
        "precinct_attached": False,
    }
    if bundle_metadata:
        export_context.setdefault("bundle_mode", "aggregate")
        export_context.setdefault("bundle_metadata", bundle_metadata)
        export_context.setdefault("bundle_summary", bundle_summary_text)
        if bundle_audit:
            export_context.setdefault("bundle_audit", bundle_audit)


    finalized = finalize_election_output(
        headers=headers_final,
        data=data_final,
        coordinator=coordinator,
        contest=contest_name,
        state=state,
        county=county,
        context=export_context,
        enable_user_feedback=False,
        session_id=session_id,
    )

    metadata = {
        "race": contest_name,
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
        "candidate_label_map": candidate_label_map,
        "candidate_header_map": candidate_header_map_serializable,
        "candidate_metadata": candidate_metadata,
        "candidate_id_to_label": candidate_id_to_label,
        "location_headers_detected": location_headers,
        "precinct_attached": False,
        "location_diagnostics": location_diagnostics,
        "coverage": coverage_serializable,
        "total_counties": total_counties,
        "context_snapshot": context_snapshot,
        "selected_contest_ids": sorted(selected_ids),
        "selected_contest_summary": selected_metadata,
        "contest_question": contest_question,
        "contest_name_raw": contest_name_raw,
    }
    if bundle_metadata:
        metadata["bundle_mode"] = "aggregate"
        metadata["bundle_metadata"] = bundle_metadata
        metadata["bundle_summary"] = bundle_summary_text
        metadata["bundle_size"] = bundle_metadata.get("bundle_size")
        if bundle_audit:
            metadata["bundle_audit"] = bundle_audit

    logger.info({
        "level": "INFO",
        "type": "output",
        "message": (
            "✅ Completed via json_export_loader fast-path! "
            f"Output CSV: {finalized.get('csv_path')}, Metadata: {finalized.get('metadata_path')}"
        ),
        "session_id": session_id,
    })

    return headers_final, data_final, contest_name, metadata

def parse_json_election_results(
    json_path: str,
    session_id: Optional[str] = None,
    coordinator: Any = None,
) -> Tuple[List[str], List[Dict[str, Any]], str, Dict[str, Any]]:
    with open(json_path, "rb") as f:
        data = orjson.loads(f.read())

    fastpath = _fastpath_county_results(json_path, data, session_id, coordinator)
    if fastpath:
        return fastpath

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
    fallback_state = ""
    fallback_county = ""
    for part in fname.replace(".json", "").split("_"):
        if "county" in part and not fallback_county:
            fallback_county = part.replace("county", "").strip()
        if len(part) == 2 and part.isalpha() and not fallback_state:
            fallback_state = part.upper()

    derived_state, derived_county = _derive_location_metadata(data)
    state = derived_state or format_state_label(fallback_state)
    county = derived_county or format_county_label(fallback_county, state)
    state = state or "Unknown"
    county = county or "Unknown"

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