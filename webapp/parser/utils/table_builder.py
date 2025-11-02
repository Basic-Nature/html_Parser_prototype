from __future__ import annotations

import copy
import os
import re
import time
from collections import OrderedDict
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

# ===================================================================
# table_builder.py
# Election Data Cleaner - Table Extraction and Cleaning Orchestrator
# Centralizes user feedback, ML learning, and structure confirmation.
# ===================================================================
import orjson
from rich.table import Table

from ..config import CACHE_DIR
from ..Context_Integration.Context_Library.constants import (
    BALLOT_TYPES_SORT_ORDER,
    PERCENT_KEYWORDS,
    get_camelot_row_regex,
    get_camelot_title_regex,
    is_pseudo_result_party,
)
from .coordinator_protocol import CoordinatorProtocol
from .detect import (
    emit_metric,
    harmonize_headers_and_data,
    nlp_entity_annotate_table,
    normalize_header,
)
from .logger_singleton import logger
from .merge_utils import merge_table_data
from .pivot import pivot_candidate_groups_from_rawjson
from .pivot import pivot_to_wide as pivot_to_wide_format
from .salvage import collapse_ballot_synonym_columns
from .shared_logic import (
    build_camelot_row_filter_for_context,
    record_noise_suggestion,
    resolve_state_county_from_context,
    safe_append,
    safe_copy,
    safe_get,
    safe_isalnum,
    safe_lower,
    safe_replace,
    safe_strip,
    safe_values,
)
from .structure_cache import cache_table_structure, table_signature

try:
    from .dynamic_table_extractor import dynamic_table_extractor
except Exception:
    # Fallback no-op extractor to keep import chain stable in minimal environments
    def dynamic_table_extractor(page, context, coordinator, table_html=None):  # type: ignore
        return [], []

if TYPE_CHECKING:
    pass


@lru_cache(maxsize=2048)
def _normalize_header_cached(raw: str) -> str:
    """Cache-normalized header lookups to avoid repeated regex work."""
    return normalize_header(raw)


def _norm_header(value: Any) -> str:
    """Normalize a header-like value with caching for hot loops."""
    if value is None:
        return _normalize_header_cached("")
    if isinstance(value, str):
        return _normalize_header_cached(value)
    return _normalize_header_cached(str(value))


@lru_cache(maxsize=1)
def _percent_norms() -> set[str]:
    extras = (
        "% reported",
        "% precincts reporting",
        "percent reported",
        "precincts reporting",
    )
    norms = {_norm_header(term) for term in PERCENT_KEYWORDS}
    norms.update(_norm_header(term) for term in extras)
    norms.add(_norm_header("Percent Reported"))
    return norms


@lru_cache(maxsize=1)
def _percent_reported_norm() -> str:
    return _norm_header("Percent Reported")


_LOCATION_PRIORITY = (
    "Division Name",
    "Precinct",
    "Municipality",
    "Ward",
    "District",
    "County",
    "Division Type",
)

_LOCATION_PRIORITY_NORMS = tuple(_norm_header(label) for label in _LOCATION_PRIORITY)
_LOCATION_TOKENS = (
    "precinct",
    "division",
    "district",
    "ward",
    "municipality",
    "county",
    "borough",
    "township",
    "location",
    "jurisdiction",
)

_CANDIDATE_SUFFIX_BASE = (
    "Party",
    "Total Vote",
    "Total Votes",
    "Vote Total",
    "% Vote",
    "Percent Vote",
    "Cumulative Vote",
    "Cumulative %",
    "Cumulative Percent",
    "Vote Share",
)
_CANDIDATE_SUFFIX_NORMS = {_norm_header(label) for label in _CANDIDATE_SUFFIX_BASE}
_BALLOT_TYPE_NORMS = {_norm_header(bt) for bt in BALLOT_TYPES_SORT_ORDER}


def _looks_like_location_header(header: str) -> bool:
    nh = _norm_header(header)
    if nh in _LOCATION_PRIORITY_NORMS:
        return True
    low = header.lower()
    return any(token in low for token in _LOCATION_TOKENS)


def _location_priority_score(header: str, original_index: int) -> tuple[int, int]:
    nh = _norm_header(header)
    if nh in _LOCATION_PRIORITY_NORMS:
        return (_LOCATION_PRIORITY_NORMS.index(nh), original_index)
    low = header.lower()
    for offset, token in enumerate(_LOCATION_TOKENS):
        if token in low:
            return (len(_LOCATION_PRIORITY_NORMS) + offset, original_index)
    return (len(_LOCATION_PRIORITY_NORMS) + len(_LOCATION_TOKENS), original_index)


def _candidate_header_info(header: str) -> tuple[str, str] | None:
    if " - " not in header:
        return None
    left, right = header.split(" - ", 1)
    left = left.strip()
    right = right.strip()
    if not left or not right:
        return None
    norm_right = _norm_header(right)
    if norm_right in _CANDIDATE_SUFFIX_NORMS:
        return left, right
    if norm_right in _BALLOT_TYPE_NORMS:
        return left, right
    for bt in BALLOT_TYPES_SORT_ORDER:
        if bt.lower() in right.lower():
            return left, right
    return None


def _extract_candidate_blocks(headers: list[str]) -> OrderedDict[str, list[str]]:
    blocks: OrderedDict[str, list[str]] = OrderedDict()
    for h in headers:
        info = _candidate_header_info(h)
        if not info:
            continue
        cand, _ = info
        blocks.setdefault(cand, []).append(h)
    return blocks


def _coerce_int_for_total(val: Any) -> Optional[int]:
    if val in (None, ""):
        return None
    if isinstance(val, bool):
        return None
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val) if val.is_integer() else None
    if isinstance(val, str):
        s = val.replace(",", "").strip()
        if s.endswith("%"):
            s = s[:-1].strip()
        if not s:
            return None
        if s.lstrip("+-").isdigit():
            try:
                return int(s)
            except Exception:
                return None
    return None


def _ensure_division_totals(headers: list[str], rows: list[dict]) -> tuple[list[str], list[dict]]:
    if not headers or not rows:
        return headers, rows
    norm_map = {h: _norm_header(h) for h in headers}
    grand_norm = _norm_header("Grand Total")
    grand_header = next((h for h in headers if norm_map[h] == grand_norm), None)
    percent_norms = _percent_norms()
    candidate_blocks = _extract_candidate_blocks(headers)
    if not candidate_blocks:
        if grand_header is not None:
            for row in rows:
                existing = row.get(grand_header)
                existing_int = _coerce_int_for_total(existing)
                if existing_int is not None:
                    row[grand_header] = existing_int
        return headers, rows
    candidate_total_cols: list[str] = []
    ballot_value_cols: list[str] = []
    for cols in candidate_blocks.values():
        for col in cols:
            info = _candidate_header_info(col)
            if not info:
                continue
            _, suffix = info
            suffix_norm = _norm_header(suffix)
            if suffix_norm == _norm_header("Total Vote") or suffix_norm == _norm_header("Total Votes"):
                candidate_total_cols.append(col)
            elif suffix_norm in _BALLOT_TYPE_NORMS:
                ballot_value_cols.append(col)
            else:
                for bt in BALLOT_TYPES_SORT_ORDER:
                    if bt.lower() in suffix.lower():
                        ballot_value_cols.append(col)
                        break

    totals_needed = grand_header is None
    if totals_needed:
        headers = headers + ["Grand Total"]
        grand_header = "Grand Total"
    assert grand_header is not None

    for row in rows:
        existing = row.get(grand_header)
        existing_int = _coerce_int_for_total(existing)
        if existing_int is not None and existing_int >= 0:
            row[grand_header] = existing_int
            continue
        total_val = None
        if candidate_total_cols:
            total_val = sum(_coerce_int_for_total(row.get(col)) or 0 for col in candidate_total_cols)
        elif ballot_value_cols:
            total_val = sum(_coerce_int_for_total(row.get(col)) or 0 for col in ballot_value_cols)
        else:
            numeric_sum = 0
            numeric_found = False
            for col, value in row.items():
                ncol = _norm_header(col)
                if ncol in _LOCATION_PRIORITY_NORMS or ncol in percent_norms:
                    continue
                iv = _coerce_int_for_total(value)
                if iv is not None:
                    numeric_sum += iv
                    numeric_found = True
            if numeric_found:
                total_val = numeric_sum
        if total_val is not None:
            row[grand_header] = total_val
        elif totals_needed:
            row.setdefault(grand_header, "")

    return headers, rows


def _apply_canonical_order(headers: list[str]) -> list[str]:
    if not headers:
        return headers

    ordered: list[str] = []
    seen: set[str] = set()
    percent_norm_set = _percent_norms()
    candidate_blocks = _extract_candidate_blocks(headers)
    location_candidates = [h for h in headers if _looks_like_location_header(h)]

    primary = None
    if location_candidates:
        primary = min(location_candidates, key=lambda h: _location_priority_score(h, headers.index(h)))
        ordered.append(primary)
        seen.add(primary)
    else:
        primary = None
        first = headers[0]
        ordered.append(first)
        seen.add(first)

    for h in headers:
        if h in seen:
            continue
        if _norm_header(h) in percent_norm_set:
            ordered.append(h)
            seen.add(h)

    if location_candidates:
        for h in sorted(location_candidates, key=lambda h: _location_priority_score(h, headers.index(h))):
            if h in seen:
                continue
            ordered.append(h)
            seen.add(h)

    for cols in candidate_blocks.values():
        for col in cols:
            if col in seen:
                continue
            ordered.append(col)
            seen.add(col)

    total_norms = {
        _norm_header("Grand Total"),
        _norm_header("Total Vote"),
        _norm_header("Total Votes"),
        _norm_header("Total Ballots"),
    }
    if candidate_blocks:
        for h in headers:
            if h in seen:
                continue
            if _norm_header(h) in total_norms:
                ordered.append(h)
                seen.add(h)

    for h in headers:
        if h not in seen:
            ordered.append(h)
            seen.add(h)

    # Ensure percent columns follow the primary location when both exist
    if primary and any(_norm_header(h) in percent_norm_set for h in ordered):
        percent_cols = [h for h in ordered if _norm_header(h) in percent_norm_set]
        for h in percent_cols:
            ordered.remove(h)
        idx = ordered.index(primary) + 1
        for offset, h in enumerate(percent_cols):
            ordered.insert(idx + offset, h)

    return ordered

# ===================================================================
# Helper: Structured logging wrapper
# ===================================================================

def _emit(level: str, msg_type: str, message: str, session_id: Optional[str] = None, **fields):
    """
    Emit a structured log payload using SharedLogger with consistent keys.
    """
    payload = {
        "level": level.upper(),
        "type": msg_type,
        "message": message,
        "session_id": session_id,
    }
    # Attach extra fields only if not None
    for k, v in fields.items():
        if v is not None:
            payload[k] = v
    # Route to appropriate logger method
    level_l = level.lower()
    log_fn = getattr(logger, level_l, logger.info)
    log_fn(payload)

def _salvage_promote_best_row_as_header(rows: List[Any], session_id=None):
    """
    Heuristically choose the best list/tuple row to act as header.
    Score = (non_empty_cells / total) + 0.3 * uniqueness_ratio
    Returns (headers, dict_rows, diagnostics)
    """
    if not rows:
        return [], [], {"row_index": None, "score": 0.0, "columns": 0, "strategy": "none"}
    candidate_indices = list(range(min(6, len(rows))))  # inspect first few
    best = {"score": -1.0, "idx": 0, "headers": []}
    for idx in candidate_indices:
        r = rows[idx]
        if not isinstance(r, (list, tuple)):
            continue
        cells = [str(c).strip() for c in r]
        non = sum(1 for c in cells if c)
        uniq = len({_norm_header(c) for c in cells if c})
        denom = max(1, len(cells))
        score = (non / denom) + 0.3 * (uniq / denom)
        if score > best["score"]:
            best = {"score": score, "idx": idx, "headers": cells}
    if not best["headers"]:
        return [], [], {"row_index": None, "score": 0.0, "columns": 0, "strategy": "none"}

    # Normalize + dedupe headers from the chosen row
    raw_headers = [str(c).strip() for c in best["headers"]]
    final_headers: list[str] = []
    norm_seen: set[str] = set()
    for i, h in enumerate(raw_headers):
        base = h or f"Column {i+1}"
        candidate = base
        suffix = 2
        while _norm_header(candidate) in norm_seen:
            candidate = f"{base}_{suffix}"
            suffix += 1
        final_headers.append(candidate)
        norm_seen.add(_norm_header(candidate))

    # Build dict rows from all rows except the promoted header row
    dict_rows: list[dict] = []
    for idx, r in enumerate(rows):
        if idx == best["idx"]:
            continue
        if isinstance(r, (list, tuple)):
            d = {final_headers[i]: (r[i] if i < len(r) else "") for i in range(len(final_headers))}
        elif isinstance(r, dict):
            d = {h: r.get(h, "") for h in final_headers}
        else:
            d = {final_headers[0]: r}
        if any(v not in ("", None) for v in d.values()):
            dict_rows.append(d)
    emit_metric("builder_salvage_promote_header", rows=len(dict_rows))
    diag = {"row_index": best["idx"], "score": best["score"], "columns": len(final_headers), "strategy": "auto_best_row"}
    return final_headers, dict_rows, diag

def _salvage_promote_first_row_as_header(rows: List[Any]):
    """Fallback: treat the first list/tuple row as header and build dict rows."""
    if not rows or not isinstance(rows[0], (list, tuple)):
        return [], []
    raw = [str(c).strip() for c in rows[0]]
    final_headers: list[str] = []
    norm_seen: set[str] = set()
    for i, h in enumerate(raw):
        base = h or f"Column {i+1}"
        candidate = base
        suffix = 2
        while _norm_header(candidate) in norm_seen:
            candidate = f"{base}_{suffix}"
            suffix += 1
        final_headers.append(candidate)
        norm_seen.add(_norm_header(candidate))
    dict_rows: list[dict] = []
    for r in rows[1:]:
        if isinstance(r, (list, tuple)):
            d = {final_headers[i]: (r[i] if i < len(r) else "") for i in range(len(final_headers))}
        elif isinstance(r, dict):
            d = {h: r.get(h, "") for h in final_headers}
        else:
            d = {final_headers[0]: r}
        if any(v not in ("", None) for v in d.values()):
            dict_rows.append(d)
    emit_metric("builder_salvage_promote_header", rows=len(dict_rows))
    return final_headers, dict_rows

def _sanitize_headers_and_rows(headers: List[Any], rows: List[Any], session_id=None, context: dict | None = None):
    """
    Defensive sanitation & salvage.
    """
    headers = headers or []
    rows = rows or []
    context = context or {}
    norm = _norm_header

    # Salvage detection: many list rows and missing/placeholder headers
    list_like = sum(1 for r in rows if isinstance(r, (list, tuple)))
    salvage_diag = None
    if rows and list_like / max(1, len(rows)) >= 0.8 and (not headers or all(re.match(r"^col(umn)?\s*\d+$", str(h), re.I) for h in headers)):
        try:
            headers, rows, salvage_diag = _salvage_promote_best_row_as_header(rows, session_id=session_id)
        except Exception:
            try:
                headers, rows = _salvage_promote_first_row_as_header(rows)
            except Exception:
                pass
    if salvage_diag:
        context.setdefault("salvage_events", []).append({"type": "promote_row_header", **salvage_diag})

    # Flatten headers
    def _flatten(seq):
        for x in (seq or []):
            if isinstance(x, (list, tuple)):
                for y in _flatten(x):
                    yield y
            else:
                yield x

    flat_headers: list[str] = []
    seen_norm = set()
    col_pattern = re.compile(r"^column\s+(\d+)$", re.I)
    for raw in _flatten(headers):
        h = str(raw) if raw is not None else ""
        h = h.strip()
        if not h:
            continue
        nh = norm(h)
        # collapse duplicate Column N
        m = col_pattern.match(h)
        if m:
            h = f"Column {m.group(1)}"
            nh = norm(h)
        if nh not in seen_norm:
            flat_headers.append(h)
            seen_norm.add(nh)

    # Sanitize rows
    sanitized_rows: list[dict] = []
    changed = False

    def _ensure_columns(count: int):
        while len(flat_headers) < count:
            idx = len(flat_headers) + 1
            name = f"Column {idx}"
            if norm(name) not in seen_norm:
                flat_headers.append(name)
                seen_norm.add(norm(name))

    for r in rows:
        if isinstance(r, dict):
            # coerce keys to strings
            d = {}
            for k, v in r.items():
                ks = k if isinstance(k, str) else str(k)
                if norm(ks) not in seen_norm:
                    flat_headers.append(ks)
                    seen_norm.add(norm(ks))
                d[ks] = v
            sanitized_rows.append(d)
        elif isinstance(r, (list, tuple)):
            _ensure_columns(len(r))
            d = {flat_headers[i]: r[i] if i < len(r) else "" for i in range(len(flat_headers))}
            sanitized_rows.append(d)
            changed = True
        else:
            if not flat_headers:
                flat_headers = ["Value"]
                seen_norm.add(norm("Value"))
            sanitized_rows.append({flat_headers[0]: r})
            changed = True

    if changed:
        emit_metric("builder_rows_sanitized", rows=len(sanitized_rows))
    if any(not isinstance(h, str) for h in flat_headers):
        flat_headers = [str(h) for h in flat_headers]
    return flat_headers, sanitized_rows, context

def _stringify_for_pivot(headers: List[Any], rows: List[Dict[str, Any]]) -> tuple[list[str], list[dict]]:
    """
    Coerce headers and row keys to strings and stringify all scalar values.
    This prevents type errors during header concatenation inside pivot logic.
    """
    sh = [(h if isinstance(h, str) else str(h)) for h in (headers or [])]
    out = []
    for r in (rows or []):
        nr = {}
        if isinstance(r, dict):
            for k, v in r.items():
                ks = k if isinstance(k, str) else str(k)
                # keep numeric types; stringify others
                if isinstance(v, (int, float)) or v in (None, ""):
                    nr[ks] = v
                else:
                    try:
                        nr[ks] = v if isinstance(v, (int, float)) else str(v)
                    except Exception:
                        nr[ks] = str(v)
        out.append(nr)
    return sh, out

def _stringify_entity_info(entity_info: dict | None) -> dict:
    """
    Deep-stringify entity_info keys/labels that can be concatenated into headers by pivot.
    """
    if not isinstance(entity_info, dict):
        return {}
    def _s(x):
        try:
            return x if isinstance(x, (int, float)) else (x if isinstance(x, str) else str(x))
        except Exception:
            return str(x)
    out = {}
    for k, v in entity_info.items():
        if isinstance(v, (list, tuple, set)):
            out[k] = [_s(x) for x in v]
        elif isinstance(v, dict):
            out[k] = { _s(kk): _s(vv) for kk, vv in v.items() }
        else:
            out[k] = _s(v)
    return out

# -------------------------------------------------------------------
# Row noise dropper (title/boilerplate) prior to pivoting
# -------------------------------------------------------------------

def _drop_title_noise_rows(headers: list[str], rows: list[dict], *, context: dict | None = None) -> tuple[list[str], list[dict]]:
    """
    Remove obvious title/boilerplate rows before pivoting.
    Uses centralized noise regexes from constants.py and pseudo-party buckets.

    Heuristics:
    - If any cell in a row matches title/boilerplate patterns, drop the row.
    - If a row's Party field is a pseudo result bucket (Blank/Undervote, etc.), drop the row.
    Conservatively keeps data rows; fails open on errors.
    """
    context = context or {}
    try:
        state, county = resolve_state_county_from_context(context)
        title_re = get_camelot_title_regex(state=state, county=county)
        row_re = get_camelot_row_regex(state=state, county=county)
    except Exception:
        # If regex helpers are unavailable, no-op
        return headers, rows

    kept: list[dict] = []
    dropped = 0
    # Build jurisdiction-aware row filter (uses Candidate/Party keys primarily)
    try:
        row_noise = build_camelot_row_filter_for_context(context)
    except Exception:
        row_noise = None
    for r in rows or []:
        try:
            if not isinstance(r, dict):
                kept.append(r)
                continue
            # Pseudo party buckets
            party_val = r.get("Party", "")
            if party_val and is_pseudo_result_party(str(party_val)):
                dropped += 1
                try:
                    record_noise_suggestion(state, county, f"Party={party_val}", category="pseudo_party")
                except Exception:
                    pass
                continue
            # If a jurisdiction-aware row filter is available, apply it
            if row_noise and row_noise(r):
                dropped += 1
                try:
                    snippet = next((str(v).strip() for v in r.values() if str(v).strip()), "")
                    record_noise_suggestion(state, county, snippet, category="row")
                except Exception:
                    pass
                continue
            # Any cell match to boilerplate/title patterns
            is_noise = False
            matched_cat = None
            for v in r.values():
                s = str(v or "").strip()
                if not s:
                    continue
                if title_re.search(s):
                    matched_cat = "title"
                    is_noise = True
                    break
                if row_re.search(s):
                    matched_cat = "row"
                    is_noise = True
                    break
            if is_noise:
                dropped += 1
                try:
                    record_noise_suggestion(state, county, s, category=matched_cat or "row")
                except Exception:
                    pass
                continue
            kept.append(r)
        except Exception:
            # On error, keep row to avoid data loss
            kept.append(r)

    if dropped:
        try:
            emit_metric("builder_drop_title_noise_rows", dropped=dropped, kept=len(kept))
        except Exception:
            pass
        try:
            _emit("info", "builder", "[TABLE_BUILDER] Dropped title/boilerplate rows", None, dropped=dropped, kept=len(kept))
        except Exception:
            pass
    return headers, kept

# ===================================================================
# MAIN TABLE BUILDING PIPELINE
# ===================================================================

def build_dynamic_table(
    domain: str,
    headers: List[str],
    data: List[Dict[str, Any]],
    coordinator: CoordinatorProtocol | None,
    context: dict = None,
    max_feedback_loops: int = 2,
    learning_mode: bool = True,
    confirm_table_structure_callback=None,
    pivot_to_wide: bool = True,
    debug: bool = False,
) -> Tuple[List[str], List[Dict[str, Any]], dict]:
    """
    Orchestrates robust, multi-source, entity-aware table extraction and harmonization.
    Always merges, harmonizes, and pivots all panel tables before any feedback/confirmation.
    Returns (headers, data, entity_info) for downstream enrichment.
    """
    from ..Context_Integration.context_coordinator import ContextCoordinator

    # Normalize inputs / coordinator / context
    coordinator = coordinator or ContextCoordinator()
    context = context or {}
    data = data or []
    headers = headers or []
    if "coordinator" not in context or context["coordinator"] is None:
        context["coordinator"] = coordinator
    # session_id for logs
    session_id = safe_get(context, "session_id", None)
    if safe_get(context, "panel_heading") and not safe_get(context, "Precinct"):
        context["Precinct"] = safe_get(context, "panel_heading")
    page = safe_get(context, "page", [])

    _emit("info", "builder", "[TABLE_BUILDER] Starting dynamic build", session_id, domain=domain, pivot_to_wide=pivot_to_wide, learning_mode=learning_mode)

    # --- 1. Gather all panel tables if present ---
    all_panel_tables = []
    if safe_get(context, "panels"):
        for panel in context["panels"]:
            for table in safe_get(panel, "tables", []):
                table_context = context.copy()
                table_context["panel_heading"] = safe_get(panel, "panel_heading", [])
                table_context["Precinct"] = safe_get(panel, "Precinct", [])
                table_context["table_html"] = safe_get(table, "table_html", [])
                try:
                    h, d = dynamic_table_extractor(page, table_context, coordinator, table_html=safe_get(table, "table_html"))
                except Exception as e:
                    _emit("warning", "builder", "[TABLE_BUILDER] dynamic_table_extractor failed for panel table", session_id, error=str(e))
                    h, d = [], []
                if h and d:
                    all_panel_tables.append((h, d))
        _emit("debug", "builder", "[TABLE_BUILDER] Collected panel tables", session_id, count=len(all_panel_tables))
    elif headers and data:
        all_panel_tables.append((headers, data))
        _emit("debug", "builder", "[TABLE_BUILDER] Using provided headers/data as sole table", session_id, headers=len(headers), rows=len(data))
    else:
        try:
            h, d = dynamic_table_extractor(page, context, coordinator)
        except Exception as e:
            _emit("warning", "builder", "[TABLE_BUILDER] dynamic_table_extractor failed (no panels path)", session_id, error=str(e))
            h, d = [], []
        if h and d:
            all_panel_tables.append((h, d))
        _emit("debug", "builder", "[TABLE_BUILDER] Fallback extractor path", session_id, found=bool(h and d))

    # Defensive: coerce/validate structure before iterating
    if not isinstance(all_panel_tables, list):
        _emit("warning", "builder", "[TABLE_BUILDER] all_panel_tables was not a list; coercing to empty list", session_id, got_type=str(type(all_panel_tables)))
        all_panel_tables = []
    else:
        # Keep only (headers, rows) tuple pairs
        fixed = []
        for item in all_panel_tables:
            if isinstance(item, tuple) and len(item) == 2:
                fixed.append(item)
            else:
                _emit("warning", "builder", "[TABLE_BUILDER] Dropping invalid table entry", session_id, entry_type=str(type(item)))
        all_panel_tables = fixed

    # --- 2. Merge and harmonize all tables (simple merge) ---
    if all_panel_tables:
        all_headers = []
        all_data = []
        for h, d in all_panel_tables:
            all_headers.append(h)
            all_data.extend(d)
        merged_headers, merged_rows = merge_table_data(all_headers, all_data)
    else:
        merged_headers, merged_rows = headers or [], data or []

    try:
        merged_headers, merged_rows, context = _sanitize_headers_and_rows(merged_headers, merged_rows, session_id=session_id, context=context)
    except Exception as e:
        _emit("warning", "builder", "[TABLE_BUILDER] sanitize failed", session_id, error=str(e))

    try:
        merged_headers, merged_rows = harmonize_headers_and_data(merged_headers, merged_rows, context)
    except Exception as e:
        _emit("warning", "builder", "[TABLE_BUILDER] harmonize failed", session_id, error=str(e))

    # Canonicalize ballot/method headers and merge synonyms
    try:
        merged_headers, merged_rows = collapse_ballot_synonym_columns(merged_headers, merged_rows)
    except Exception as e:
        _emit("warning", "builder", "[TABLE_BUILDER] collapse_ballot_synonym_columns failed", session_id, error=str(e))

    # NEW: drop title/boilerplate rows before any pivoting
    merged_headers, merged_rows = _drop_title_noise_rows(merged_headers, merged_rows, context=context)

    # Define schema validator before first use
    def _validate_stage_schema(stage: str, hdrs: list[str], rows: list[dict]):
        """Lightweight schema checks with logging-only warnings; always emits a structured event.
        - normalized: expect at least one candidate-like column or a total + ballot columns; Precinct optional.
        - wide: expect candidate columns pivoted to headers and at least one numeric vote column.
        """
        status = "unknown"
        details = {}
        try:
            nh = [_norm_header(h) for h in hdrs]
            precinct_norm = _norm_header("Precinct")
            total_norm = _norm_header("Total Vote")
            grand_total_norm = _norm_header("Grand Total")
            percent_norm = _norm_header("Percent Reported")
            has_precinct = precinct_norm in nh
            has_total = total_norm in nh or grand_total_norm in nh
            has_pct = percent_norm in nh
            cand_like = [h for h in hdrs if any(k in h.lower() for k in ("candidate","name"))]
            ballot_like = [h for h in hdrs if any(bt.lower() in h.lower() for bt in BALLOT_TYPES_SORT_ORDER)]
            numeric_cols = set()
            for r in rows[: min(50, len(rows))]:
                for k, v in (r or {}).items():
                    if isinstance(v, (int, float)):
                        numeric_cols.add(k)
                        continue
                    s = str(v or "").replace(",", "").replace("%", "").strip()
                    if s.replace(".", "", 1).isdigit():
                        numeric_cols.add(k)
            if stage == "normalized":
                ok = bool(cand_like or (has_total and ballot_like))
            else:  # wide
                ok = bool(numeric_cols)
            status = "ok" if ok else "weak"
            details = {
                "headers": len(hdrs),
                "rows": len(rows),
                "has_precinct": has_precinct,
                "has_total": has_total,
                "has_percent": has_pct,
                "candidates": len(cand_like),
                "ballots": len(ballot_like),
            }
        except Exception as e:
            status = "error"
            details = {"error": str(e), "headers": len(hdrs), "rows": len(rows)}
        finally:
            _emit(
                "info" if status == "ok" else "warning",
                "builder",
                {"event": "schema_check", "stage": stage, "status": status, **details},
                session_id,
            )

    # Validate normalized stage before pivot
    try:
        _validate_stage_schema("normalized", merged_headers, merged_rows)
    except Exception:
        pass

    # --- Single canonical percent column (de-noise) ---
    percent_norm_set = _percent_norms()
    percent_reported_norm = _percent_reported_norm()
    has_any_percent = any(_norm_header(h) in percent_norm_set for h in merged_headers)
    if has_any_percent and not any(_norm_header(h) == percent_reported_norm for h in merged_headers):
        merged_headers.append("Percent Reported")
    context["has_percent_reported"] = any(_norm_header(h) == percent_reported_norm for h in merged_headers)

    # --- 3. NLP entity annotation (optional, safe) ---
    try:
        eh, ed, entity_info = nlp_entity_annotate_table(merged_headers, merged_rows, context=context, coordinator=coordinator)
        merged_headers, merged_rows = eh, ed
    except Exception as e:
        _emit("warning", "builder", "[TABLE_BUILDER] entity annotate failed", session_id, error=str(e))
        entity_info = {}
    try:
        entity_info = _stringify_entity_info(entity_info)
    except Exception as e:
        _emit("warning", "builder", "[TABLE_BUILDER] stringify entity_info failed", session_id, error=str(e))

    # --- 4. Pivot (RawJSON specialized first, then generic) ---
    def _apply_pivot(hdrs, rows, *, prefer_rawjson: bool = True):
        stage: str | None = None
        if prefer_rawjson:
            try:
                hh, rr = pivot_candidate_groups_from_rawjson(
                    hdrs, rows, context=context, drop_rawjson=True
                )
                if hh and rr:
                    return hh, rr, "rawjson"
            except Exception:
                pass
        try:
            hh, rr = pivot_to_wide_format(hdrs, rows, entity_info, coordinator, context)
            if hh and rr:
                stage = "wide"
                return hh, rr, stage
        except Exception as e:
            _emit("warning", "builder", "[TABLE_BUILDER] pivot_to_wide failed", session_id, error=str(e))
        return hdrs, rows, stage

    pivot_stage: str | None = None
    if pivot_to_wide and not context.get("skip_pivot") and context.get("rawjson_expanded_early"):
        merged_headers, merged_rows, pivot_stage = _apply_pivot(merged_headers, merged_rows)
        _validate_stage_schema("wide", merged_headers, merged_rows)

    if pivot_to_wide and not context.get("skip_pivot") and (pivot_stage is None or pivot_stage == "rawjson"):
        merged_headers, merged_rows, pivot_stage = _apply_pivot(
            merged_headers,
            merged_rows,
            prefer_rawjson=pivot_stage is None,
        )
        _validate_stage_schema("wide", merged_headers, merged_rows)

    # Ensure division-level totals are always available
    try:
        merged_headers, merged_rows = _ensure_division_totals(merged_headers, merged_rows)
    except Exception as e:
        _emit("warning", "builder", "[TABLE_BUILDER] ensure division totals failed", session_id, error=str(e))

    # Apply canonical ordering at the end for consistency
    try:
        merged_headers = _apply_canonical_order(merged_headers)
        merged_rows = [{h: r.get(h, "") for h in merged_headers} for r in merged_rows]
    except Exception:
        pass

    # Attach salvage events to entity_info if any
    if 'salvage_events' in context and isinstance(context['salvage_events'], list):
        entity_info['salvage_events'] = context['salvage_events']

    # --- 5. Optional cache for debugging ---
    if debug:
        persistent_cache = {
            "timestamp": time.time(),
            "domain": domain,
            "headers": merged_headers,
            "rows": merged_rows[:50],
        }
        try:
            _save_table_builder_cache(domain, persistent_cache)
        except Exception:
            pass

    # --- 6. Learning disabled in non-interactive path; keep hook for future ---
    _emit("info", "builder", "[TABLE_BUILDER] Completed dynamic build", session_id, headers=len(merged_headers), rows=len(merged_rows))
    return merged_headers, merged_rows, entity_info

# Non-interactive wrapper for format handlers

def build_table_noninteractive(
    domain: str,
    headers: List[str] | None,
    data: List[Dict[str, Any]] | None,
    coordinator: CoordinatorProtocol | None = None,
    context: dict | None = None,
    pivot_to_wide: bool = True,
    debug: bool = False
) -> Tuple[List[str], List[Dict[str, Any]], dict]:
    """
    Convenience wrapper around build_dynamic_table with learning_mode disabled.
    Use this in format handlers (CSV/JSON/PDF) after contest selection to
    harmonize, annotate, and optionally pivot tables without any prompts.
    """
    context = context or {}
    session_id = safe_get(context, "session_id", None)
    _emit("info", "builder", "[TABLE_BUILDER] build_table_noninteractive called", session_id, domain=domain)
    return build_dynamic_table(
        domain=domain,
        headers=headers or [],
        data=data or [],
        coordinator=coordinator,
        context=context,
        max_feedback_loops=0,
        learning_mode=False,
        confirm_table_structure_callback=None,
        pivot_to_wide=pivot_to_wide,
        debug=debug
    )

# ===================================================================
# CACHE MANAGEMENT STRATEGY
# ===================================================================

def _get_table_builder_cache_dir():
    """
    Returns the directory for table builder cache files, nested under CACHE_DIR for consistency.
    """
    cache_dir = os.path.join(CACHE_DIR, "table_builder_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

def _save_table_builder_cache(domain, persistent_cache, keep_last_n=5):
    """
    Save the persistent cache for debugging/recovery only.
    Keeps only the last N cache files per domain to avoid stale data buildup.
    """
    cache_dir = _get_table_builder_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    safe_domain = "".join(c for c in domain if safe_isalnum(c) or c in ("-", "_"))
    timestamp = int(safe_get(persistent_cache, "timestamp", time.time()))
    cache_path = os.path.join(cache_dir, f"{safe_domain}_{timestamp}_table.json")
    with open(cache_path, "wb") as f:
        f.write(orjson.dumps(persistent_cache, option=orjson.OPT_INDENT_2))
    # Cleanup
    files = sorted(
        [f for f in os.listdir(cache_dir) if f.startswith(safe_domain)],
        key=lambda fn: os.path.getmtime(os.path.join(cache_dir, fn)),
        reverse=True
    )
    for old_file in files[keep_last_n:]:
        try:
            os.remove(os.path.join(cache_dir, old_file))
        except Exception:
            pass

def _list_table_builder_cache(domain=None):
    """
    List available cache files for a domain (or all if domain is None).
    """
    cache_dir = _get_table_builder_cache_dir()
    if not os.path.exists(cache_dir):
        return []
    files = os.listdir(cache_dir)
    if domain:
        safe_domain = "".join(c for c in domain if safe_isalnum(c) or c in ("-", "_"))
        files = [f for f in files if f.startswith(safe_domain)]
    return sorted(files, key=lambda fn: os.path.getmtime(os.path.join(cache_dir, fn)), reverse=True)

def _load_table_builder_cache(domain, latest=True):
    """
    Load the latest (or all) cache files for a domain.
    """
    files = _list_table_builder_cache(domain)
    if not files:
        return None
    cache_dir = _get_table_builder_cache_dir()
    if latest:
        with open(os.path.join(cache_dir, files[0]), "rb") as f:
            return orjson.loads(f.read())
    else:
        caches = []
        for fn in files:
            with open(os.path.join(cache_dir, fn), "rb") as f:
                caches.append(orjson.loads(f.read()))
        return caches

# ===================================================================
# USER FEEDBACK, CONFIRMATION, AND LEARNING
# ===================================================================

def prompt_user_to_confirm_table_structure(
    headers,
    data,
    domain,
    contest,
    coordinator,
    session_id: Optional[str] = None
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Interactive CLI for user to confirm, correct, or reject table structure.
    Ensures 'Percent Reported' is always included if present in data or context.
    """
    from ..Context_Integration.context_coordinator import ContextCoordinator
    coordinator = coordinator or ContextCoordinator()
    should_log = True
    columns_changed = False
    new_headers = safe_copy(headers)

    # Always include 'Percent Reported' if present in any row
    if any("Percent Reported" in row for row in data) and "Percent Reported" not in new_headers:
        new_headers = safe_append(new_headers, "Percent Reported")

    os.makedirs(CACHE_DIR, exist_ok=True)

    # Denied structures cache
    denied_structures_path = os.path.join(CACHE_DIR, "denied_table_structures.json")
    denied_structures = {}
    if os.path.exists(denied_structures_path):
        with open(denied_structures_path, "rb") as f:
            denied_structures = orjson.loads(f.read())
    sig = f"{domain}:{table_signature(headers)}"
    denied_count = safe_get(denied_structures, sig, 0)

    # Removed columns cache
    removed_columns_log_path = os.path.join(CACHE_DIR, "removed_columns_cache.json")
    removed_columns_log = {}
    if os.path.exists(removed_columns_log_path):
        with open(removed_columns_log_path, "rb") as f:
            removed_columns_log = orjson.loads(f.read())

    # ML/NLP suggestions
    ml_scores = []
    nlp_suggestions = []
    for h in new_headers:
        try:
            score = coordinator.score_header(h, {"contest": contest})
        except Exception:
            score = 0.0
        ml_scores = safe_append(ml_scores, score)
        try:
            ents = coordinator.extract_entities(h)
        except Exception:
            ents = []
        if ents:
            ent, label = ents[0]
            nlp_suggestions = safe_append(nlp_suggestions, (h, ent, label))
        else:
            nlp_suggestions = safe_append(nlp_suggestions, (h, None, None))

    avg_score = sum(ml_scores) / len(ml_scores) if ml_scores else 0.0
    auto_accept_threshold = 0.93  # Accept automatically if ML is very confident

    # If ML confidence is low and NLP suggests better header names, auto-apply those suggestions
    if avg_score < 0.7 and any(ent and ent != h for h, ent, label in nlp_suggestions):
        _emit("info", "builder", "[TABLE_BUILDER] ML confidence low; auto-applying NLP header suggestions", session_id, contest=contest)
        alt = safe_copy(new_headers)
        for idx, (h, ent, label) in enumerate(nlp_suggestions):
            if ent and ent != h and idx < len(alt):
                alt[idx] = ent
                new_headers[idx] = ent
        new_headers, data = harmonize_headers_and_data(new_headers, data)
        try:
            ml_scores = [coordinator.score_header(h, {"contest": contest}) for h in new_headers]
        except Exception:
            ml_scores = [0.0 for _ in new_headers]
        avg_score = sum(ml_scores) / len(ml_scores) if ml_scores else 0.0

    # Multiple structure candidates (if available)
    structure_candidates = [safe_copy(new_headers)]
    alt_headers = []
    for idx, (h, ent, label) in enumerate(nlp_suggestions):
        if ent and ent != h and idx < len(new_headers):
            alt = safe_copy(new_headers)
            alt[idx] = ent
            alt_headers = safe_append(alt_headers, alt)
    if alt_headers:
        structure_candidates += alt_headers

    candidate_idx = 0
    while True:
        candidate_headers = structure_candidates[candidate_idx]

        # Preview log
        preview = {
            "headers": candidate_headers,
            "rows_preview": [
                {h: safe_get(row, h, "") for h in candidate_headers} for row in data[:5]
            ]
        }
        _emit("info", "builder", "[TABLE_BUILDER] Candidate structure preview", session_id, contest=contest, candidate_index=candidate_idx+1, candidates_total=len(structure_candidates), preview=preview, ml_avg_confidence=round(avg_score, 3))

        # Also show in rich table for CLI users (optional)
        try:
            preview_table = Table(show_header=True, header_style="bold magenta")
            for h in candidate_headers:
                preview_table.add_column(h)
            for row in data[:5]:
                preview_table.add_row(*(str(safe_get(row, h, "")) for h in candidate_headers))
            logger.alert(preview_table)
        except Exception:
            pass

        # Auto-accept if ML is very confident
        if avg_score >= auto_accept_threshold:
            _emit("info", "builder", "[TABLE_BUILDER] Auto-accepting structure due to high ML confidence", session_id, confidence=round(avg_score, 3))
            new_headers = candidate_headers
            break

        # Interactive options (CLI)
        logger.info({
            "level": "INFO",
            "type": "builder",
            "message": "[TABLE_BUILDER] Options: [Y] Accept | [N] Reject | [C] Remove columns | [O] Reorder | [R] Rename | [A] Add | [Next]/[Prev]",
            "session_id": session_id
        })

        # Raw input prompt (CLI only)
        resp = input("Accept, Reject, mark Columns, reorder, Rename, Add, Next, or Prev? [Y/n/c/o/r/a/next/prev]: ").strip().lower()
        if resp in ("", "y", "yes"):
            log_structure = getattr(coordinator, "log_table_structure", None)
            if should_log and callable(log_structure):
                try:
                    log_structure(domain, new_headers, data)
                except Exception:
                    pass
            new_headers, data = harmonize_headers_and_data(new_headers, data)
            if columns_changed:
                _emit("info", "builder", "[TABLE_BUILDER] Columns were changed by user before acceptance", session_id)
            return new_headers, data
        elif resp in ("n", "no"):
            denied_structures[sig] = safe_get(denied_structures, sig, 0) + 1
            denied_count = denied_structures[sig]
            with open(denied_structures_path, "wb") as f:
                f.write(orjson.dumps(denied_structures, option=orjson.OPT_INDENT_2))
            _emit("info", "builder", "[TABLE_BUILDER] User declined structure", session_id, contest=contest, denied_count=denied_count)
            retry = input("Would you like to retry correction? [y/N]: ").strip().lower()
            if retry in ("y", "yes"):
                continue
            else:
                return headers, data
        elif resp == "c":
            print("Enter column numbers (comma-separated) that are incorrect (starting from 1):")
            for idx, h in enumerate(candidate_headers):
                print(f"  {idx+1}: {h}")
            wrong_cols = input("Columns to mark as incorrect: ")
            if wrong_cols:
                wrong_idxs = [int(i)-1 for i in wrong_cols.split(",") if i.strip().isdigit()]
                for idx in wrong_idxs:
                    if 0 <= idx < len(candidate_headers):
                        col_name = candidate_headers[idx]
                        _emit("warning", "builder", f"[TABLE_BUILDER] Column marked incorrect: {col_name}", session_id, contest=contest)
                        removed_columns_log.setdefault(contest, {})
                        removed_columns_log[contest][col_name] = safe_get(removed_columns_log[contest], col_name, 0) + 1
                candidate_headers = [h for i, h in enumerate(candidate_headers) if i not in wrong_idxs]
                data = [{h: safe_get(row, h, "") for h in candidate_headers} for row in data]
                columns_changed = True
                structure_candidates[candidate_idx] = candidate_headers
            with open(removed_columns_log_path, "wb") as f:
                f.write(orjson.dumps(removed_columns_log, option=orjson.OPT_INDENT_2))
        elif resp == "o":
            print("Enter new order of columns as space/comma-separated numbers (starting from 1):")
            for idx, h in enumerate(candidate_headers):
                print(f"  {idx+1}: {h}")
            order = input("New order: ").replace(",", " ").split()
            try:
                new_order = [candidate_headers[int(i)-1] for i in order if i.strip().isdigit() and 0 < int(i) <= len(candidate_headers)]
                if new_order:
                    candidate_headers = new_order
                    data = [{h: safe_get(row, h, "") for h in candidate_headers} for row in data]
                    columns_changed = True
                    structure_candidates[candidate_idx] = candidate_headers
                    _emit("info", "builder", "[TABLE_BUILDER] Columns reordered", session_id)
            except Exception as e:
                _emit("error", "builder", "[TABLE_BUILDER] Invalid reorder sequence", session_id, error=str(e))
        elif resp == "r":
            print("Enter column numbers (comma-separated) to rename (starting from 1):")
            for idx, h in enumerate(candidate_headers):
                print(f"  {idx+1}: {h}")
            col_nums = input("Columns to rename: ").strip()
            if col_nums:
                rename_idxs = [int(i)-1 for i in col_nums.split(",") if i.strip().isdigit() and 0 <= int(i)-1 < len(candidate_headers)]
                for idx in rename_idxs:
                    old_name = candidate_headers[idx]
                    new_name = input(f"Rename column '{old_name}' to: ").strip()
                    if new_name:
                        candidate_headers[idx] = new_name
                data = [{h: safe_get(row, h, "") for h in candidate_headers} for row in data]
                columns_changed = True
                structure_candidates[candidate_idx] = candidate_headers
        elif resp == "a":
            print("Enter names of columns to add, separated by commas:")
            add_cols = input("Columns to add: ").split(",")
            for col in add_cols:
                col = col.strip()
                if col and col not in candidate_headers:
                    candidate_headers = safe_append(candidate_headers, col)
                    for row in data:
                        row[col] = safe_get(row, col, "")
                    _emit("info", "builder", f"[TABLE_BUILDER] Added column '{col}'", session_id)
            columns_changed = True
            structure_candidates[candidate_idx] = candidate_headers
            for row in data:
                for col in candidate_headers:
                    if col not in row:
                        row[col] = ""
        elif resp in ("next", "nxt"):
            candidate_idx = (candidate_idx + 1) % len(structure_candidates)
            continue
        elif resp in ("prev", "previous"):
            candidate_idx = (candidate_idx - 1) % len(structure_candidates)
            continue
        else:
            _emit("error", "builder", "[TABLE_BUILDER] Unknown option in confirmation prompt", session_id)

        # Always harmonize after user modification
        candidate_headers, data = harmonize_headers_and_data(candidate_headers, data)

    # Save user-confirmed structure for future ML learning
    log_structure = getattr(coordinator, "log_table_structure", None)
    if should_log and callable(log_structure):
        try:
            log_structure(contest, new_headers, context={"domain": domain})
        except Exception as e:
            _emit("warning", "builder", "[TABLE_BUILDER] Failed to persist table structure logs", session_id, error=str(e))
        else:
            cache_table_structure(domain, new_headers, new_headers)
            _emit("info", "builder", "[TABLE_BUILDER] Logged confirmed table structure", session_id, contest=contest)
            save_structure = getattr(coordinator, "save_table_structure_to_db", None)
            if callable(save_structure):
                try:
                    save_structure(
                        contest=contest,
                        headers=new_headers,
                        context={"domain": domain},
                        ml_confidence=avg_score if 'avg_score' in locals() else None,
                        confirmed_by_user=True
                    )
                except Exception as e:
                    _emit("warning", "builder", "[TABLE_BUILDER] Failed to persist coordinator DB log", session_id, error=str(e))

    # Always harmonize before returning
    new_headers, data = harmonize_headers_and_data(new_headers, data)
    if columns_changed:
        _emit("info", "builder", "[TABLE_BUILDER] Final structure had user column changes", session_id)
    return new_headers, data

# ===================================================================
# OPTIONAL: BATCH OPERATIONS AND SUGGESTIONS
# ===================================================================

def interactive_batch_operations(headers, data) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Allow batch renaming, reordering, or removal of columns in the CLI.
    """
    history = []
    while True:
        print("\n[Batch Ops] [R]ename, [O]rder, [D]elete, [U]ndo, [Q]uit")
        cmd = input("Choose operation: ").strip().lower()
        if cmd == "r":
            print("Enter column numbers (comma-separated) to rename:")
            for idx, h in enumerate(headers):
                print(f"  {idx+1}: {h}")
            col_nums = input("Columns to rename: ").strip()
            if col_nums:
                rename_idxs = [int(i)-1 for i in col_nums.split(",") if i.strip().isdigit() and 0 <= int(i)-1 < len(headers)]
                history.append((copy.deepcopy(headers), copy.deepcopy(data)))
                for idx in rename_idxs:
                    old_name = headers[idx]
                    new_name = input(f"Rename column '{old_name}' to: ").strip()
                    if new_name:
                        headers[idx] = new_name
                data = [{h: row.get(h, "") for h in headers} for row in data]
        elif cmd == "o":
            print("Enter new order of columns as space/comma-separated numbers (starting from 1):")
            for idx, h in enumerate(headers):
                print(f"  {idx+1}: {h}")
            order = input("New order: ").replace(",", " ").split()
            try:
                new_order = [headers[int(i)-1] for i in order if i.strip().isdigit() and 0 < int(i) <= len(headers)]
                if new_order:
                    history.append((copy.deepcopy(headers), copy.deepcopy(data)))
                    headers = new_order
                    data = [{h: row.get(h, "") for h in headers} for row in data]
            except Exception as e:
                _emit("error", "builder", "[TABLE_BUILDER] Invalid order in batch ops", None, error=str(e))
        elif cmd == "d":
            print("Enter column numbers (comma-separated) to delete:")
            for idx, h in enumerate(headers):
                print(f"  {idx+1}: {h}")
            del_nums = input("Columns to delete: ").strip()
            if del_nums:
                del_idxs = [int(i)-1 for i in del_nums.split(",") if i.strip().isdigit() and 0 <= int(i)-1 < len(headers)]
                history.append((copy.deepcopy(headers), copy.deepcopy(data)))
                headers = [h for i, h in enumerate(headers) if i not in del_idxs]
                data = [{h: row.get(h, "") for h in headers} for row in data]
        elif cmd == "u":
            if history:
                headers, data = history.pop()
                print("[Batch Ops] Undo successful.")
            else:
                print("[Batch Ops] Nothing to undo.")
        elif cmd == "q":
            break
        else:
            print("[Batch Ops] Unknown option.")
    return headers, data

def auto_suggest_corrections(headers, data, coordinator: CoordinatorProtocol | None):
    """
    Suggest likely corrections based on previous user feedback or ML confidence.
    Examines both headers and the data content for issues.
    """
    from ..Context_Integration.context_coordinator import ContextCoordinator
    coordinator = coordinator or ContextCoordinator()
    suggestions = []

    # Suggest based on ML confidence for headers
    for h in headers:
        try:
            score = coordinator.score_header(h, {})
        except Exception:
            score = 0.0
        if score < 0.7:
            suggestions.append((h, "Low ML confidence"))

    # Suggest if any header is empty or non-alphanumeric
    for h in headers:
        h_clean = safe_replace(safe_strip(h), " ", "")
        if not h_clean or not safe_isalnum(h_clean):
            suggestions.append((h, "Header is empty or not alphanumeric"))

    # Suggest if any column is mostly empty in the data
    for h in headers:
        empty_count = sum(1 for row in data if not safe_strip(safe_get(row, h, "")))
        if data and empty_count > len(data) * 0.7:
            suggestions.append((h, "Column is mostly empty in data"))

    # Suggest if any row is missing values for required headers
    for idx, row in enumerate(data):
        missing = [h for h in headers if not safe_strip(safe_get(row, h, ""))]
        if missing:
            suggestions.append((f"Row {idx+1}", f"Missing values for columns: {missing}"))

    # Feedback log based suggestions (if available)
    feedback_log: dict = {}
    get_feedback = getattr(coordinator, "get_feedback_log", None)
    if callable(get_feedback):
        try:
            raw_feedback = get_feedback() or {}
            if isinstance(raw_feedback, dict):
                feedback_log = raw_feedback
        except Exception:
            feedback_log = {}
    for h in headers:
        h_norm = safe_lower(safe_strip(h))
        removed = safe_get(feedback_log.get("removed_columns", {}), h_norm)
        if isinstance(removed, int) and removed > 2:
            suggestions.append((h, f"Column '{h}' was removed {removed} times in past feedback"))
        renamed = safe_get(feedback_log.get("renamed_columns", {}), h_norm)
        if renamed:
            suggestions.append((h, f"Column '{h}' was often renamed to '{renamed}'"))

    return suggestions

def dynamic_confidence_threshold(history, coordinator: CoordinatorProtocol | None = None, default=0.93):
    """
    Adjust threshold for auto-accepting structures based on past accuracy and feedback log.
    If a ContextCoordinator is provided, use its feedback analytics for smarter adjustment.
    Uses safe_get and safe_values for robustness.
    """
    if coordinator is None:
        from ..Context_Integration.context_coordinator import ContextCoordinator
        coordinator = ContextCoordinator()
    threshold = default

    if history:
        correct = sum(1 for h in history[-5:] if safe_get(h, "accepted"))
        if correct >= 4:
            threshold = min(0.98, threshold + 0.02)
        elif correct <= 2:
            threshold = max(0.85, threshold - 0.05)

    feedback = {}
    get_feedback = getattr(coordinator, "get_feedback_log", None)
    if callable(get_feedback):
        try:
            raw_feedback = get_feedback() or {}
            if isinstance(raw_feedback, dict):
                feedback = raw_feedback
        except Exception:
            feedback = {}
        denials = sum(safe_values(safe_get(feedback, "structure_denials", {})))
        removals = sum(safe_values(safe_get(feedback, "removed_columns", {})))
        total_feedback = denials + removals
        if total_feedback > 10:
            threshold = min(0.99, threshold + 0.03)
        elif denials > removals and denials > 5:
            threshold = min(0.99, threshold + 0.04)
        elif removals > 10:
            threshold = max(0.85, threshold - 0.03)
    return threshold

def _unify_percent_columns(headers: List[str], rows: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Collapse multiple percent-reporting synonym columns into one 'Percent Reported'.
    Keep first meaningful value per row.
    """
    if not headers:
        return headers, rows

    percent_norms = _percent_norms()
    norm_map = {h: _norm_header(h) for h in headers}
    candidates = [h for h in headers if norm_map[h] in percent_norms]
    if not candidates:
        return headers, rows

    # Decide canonical header
    canonical = None
    for pref in ("Percent Reported", "% Reported", "% Precincts Reporting"):
        if pref in headers:
            canonical = pref
            break
    if not canonical:
        canonical = "Percent Reported"

    # Build new header list (remove others)
    new_headers = []
    inserted = False
    seen_norm = set()
    for h in headers:
        if h in candidates and h != canonical:
            continue
        if h == canonical:
            inserted = True
        norm_value = norm_map.get(h, _norm_header(h))
        if norm_value not in seen_norm:
            new_headers.append(h)
            seen_norm.add(norm_value)
    if not inserted:
        new_headers.insert(1 if "Total Ballots Reported" in new_headers else 0, canonical)
        seen_norm.add(_norm_header(canonical))

    # Row merge
    out_rows = []
    for r in rows:
        val = ""
        for h in candidates:
            v = r.get(h)
            if v not in (None, "", 0):
                val = v
                break
        if val and isinstance(val, (int, float)):
            val = f"{val}"
        new_r = {h: r.get(h, "") for h in new_headers}
        new_r[canonical] = val
        out_rows.append(new_r)
    return new_headers, out_rows

# ===================================================================
# END OF FILE
# ===================================================================