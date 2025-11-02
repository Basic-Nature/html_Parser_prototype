"""
salvage.py
Row/column salvage, merging, RawJSON flatten, footer pruning.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from .detect import parse_numeric

try:
    from .logger_singleton import logger  # type: ignore
except Exception:
    class _DummyLogger:
        def debug(self, *a, **k): pass
    logger = _DummyLogger()

# Use canonical maps from constants (fallback to safe defaults if missing)
try:
    from ..Context_Integration.Context_Library.constants import (
        BALLOT_NAME_CANON_MAP,  # lower -> Canonical ballot group label
        BALLOT_TYPES,  # list of canonical ballot/method labels
        canonical_ballot_group,  # robust normalizer for composite labels
    )
except Exception:
    BALLOT_NAME_CANON_MAP = {}
    BALLOT_TYPES = [
        "Election Day", "Early Voting", "Absentee", "Mail", "Provisional", "Affidavit",
        "Military", "Absentee Military", "Emergency", "Advance Voting", "Advance In-Person",
    ]
    def canonical_ballot_group(x: str) -> str:
        return (x or "").strip()

def _to_int_or_none(v):
    n, _ = parse_numeric(v)
    return n

def normalize_ballot_column_name(h: str) -> str:
    """
    Normalize a ballot/method column name using constants:
      - Direct map via BALLOT_NAME_CANON_MAP (case-insensitive)
      - Fall back to canonical_ballot_group
      - Handle common totals
    """
    if not h:
        return h
    raw = str(h).strip()
    # collapse repeated whitespace for more robust matching (e.g., 'Election  Day' -> 'Election Day')
    low = " ".join(raw.lower().split())

    # Candidate-specific columns (e.g., "Candidate - Total Vote") should retain
    # their original formatting so downstream wide-table detection preserves the
    # "Candidate - Method" pattern.  Rewriting these headers (for example, to
    # use '/' separators) breaks the fast-path detection and causes numeric
    # prefixes to be introduced later in the pivot pipeline.  Guard here by
    # returning the original header when the suffix clearly matches a
    # candidate-centric metric.
    if " - " in raw:
        _, right = raw.split(" - ", 1)
        right_norm = right.strip().lower()
        candidate_suffixes = {
            "total",
            "total vote",
            "total votes",
            "total ballots",
            "% vote",
            "percent vote",
            "percent",
            "cumulative vote",
            "cumulative %",
            "cumulative percent",
            "party",
        }
        ballot_suffixes = {b.lower() for b in BALLOT_TYPES}
        if right_norm in candidate_suffixes or right_norm in ballot_suffixes or right_norm.endswith("%"):
            return raw

    if low in {"total", "total vote"}:
        return "Total Vote"
    if low in {"grand total"}:
        return "Grand Total"

    if low in BALLOT_NAME_CANON_MAP:
        return BALLOT_NAME_CANON_MAP[low]

    # Use canonical_ballot_group on the trimmed original; it already handles punctuation/composites
    cand = canonical_ballot_group(raw)
    return cand or raw

def collapse_ballot_synonym_columns(headers: List[str], rows: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    - Rename ballot/method headers to canonical names using constants.py
    - Merge duplicate/synonym columns by summing numeric values
    - Backfill 'Total Vote' from summed ballot columns if missing
    """
    headers = headers or []
    rows = rows or []

    # 1) Build rename map
    rename: dict[str, str] = {}
    for h in headers:
        canon = normalize_ballot_column_name(h)
        if canon != h:
            rename[h] = canon

    # 2) Build header order with dedupe
    seen = set()
    new_headers: list[str] = []
    for h in headers:
        nh = rename.get(h, h)
        if nh not in seen:
            new_headers.append(nh)
            seen.add(nh)

    # 3) Rewrite rows and merge numeric ballot columns
    out_rows: list[dict] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        acc: dict[str, Any] = {}
        for k, v in r.items():
            nk = rename.get(k, k)
            # Sum numeric for ballot/method and totals
            if nk in ("Total Vote", "Grand Total") or (nk != k and nk in new_headers):
                cur = _to_int_or_none(acc.get(nk))
                nxt = _to_int_or_none(v)
                if cur is None and nxt is None:
                    acc[nk] = acc.get(nk, v)
                elif cur is None:
                    acc[nk] = nxt
                elif nxt is None:
                    acc[nk] = cur
                else:
                    acc[nk] = cur + nxt
            else:
                acc[nk] = v

        # 4) Backfill Total Vote if possible
        if ("Total Vote" not in acc) or (_to_int_or_none(acc.get("Total Vote")) is None):
            vote_sum = 0
            found_any = False
            for key, val in acc.items():
                # Treat headers whose canonical name is a known ballot/method as ballot columns
                canon_key = normalize_ballot_column_name(key)
                if canon_key in BALLOT_TYPES:
                    iv = _to_int_or_none(val)
                    if iv is not None:
                        vote_sum += iv
                        found_any = True
            if found_any:
                acc["Total Vote"] = vote_sum

        out_rows.append(acc)

    # 5) Ensure headers include any new canonical names present in data
    present = set()
    for r in out_rows:
        present.update(r.keys())

    final_headers = [h for h in new_headers if h in present]
    for t in ("Total Vote", "Grand Total"):
        if t in present and t not in final_headers:
            final_headers.append(t)

    logger.debug({"level": "DEBUG", "type": "salvage", "message": "Collapsed ballot/method synonym columns",
                  "headers": len(final_headers), "rows": len(out_rows)})
    return final_headers, out_rows

__all__ = [
    # ...existing exports...
    "normalize_ballot_column_name",
    "collapse_ballot_synonym_columns",
]

# ---------------- Additional Salvage Utilities (lightweight defaults) ---------------- #

def merge_multiline_candidate_rows(headers: List[str], rows: List[Dict[str, Any]]):
    """
    Merge simple two-line candidate rows where the first row has Candidate and the next line
    carries supplemental fields (e.g., Party) with an empty Candidate.

    Conservative behavior: only merges when current row has no Candidate-like value but has
    at least one non-empty other field; fills missing keys on the previous row.
    """
    headers = headers or []
    rows = rows or []
    if not rows:
        return headers, rows
    cand_keys = {k for k in ("Candidate", "Name") if k in headers}
    if not cand_keys:
        return headers, rows
    out: List[Dict[str, Any]] = []
    for r in rows:
        if out:
            prev = out[-1]
        else:
            prev = None
        cand_vals = [str(r.get(k, "") or "").strip() for k in cand_keys]
        has_candidate = any(cand_vals)
        non_empty_other = any(str(v or "").strip() for k, v in r.items() if k not in cand_keys)
        if not has_candidate and non_empty_other and prev is not None:
            # merge fields into prev where missing
            for k, v in r.items():
                if not str(prev.get(k, "") or "").strip() and str(v or "").strip():
                    prev[k] = v
        else:
            out.append(dict(r))
    return headers, out

def combine_panel_tables_by_precinct(tables: List[tuple[List[str], List[Dict[str, Any]]]]):
    """
    Combine multiple (headers, rows) pairs by simple union of headers and concatenation of rows.
    If a row is missing a header present in the union, it will be filled with empty string.
    """
    if not tables:
        return [], []
    union_headers: list[str] = []
    seen = set()
    for h, _ in tables:
        for x in (h or []):
            if x not in seen:
                seen.add(x)
                union_headers.append(x)
    out_rows: list[dict] = []
    for h, rows in tables:
        for r in (rows or []):
            full = {hh: r.get(hh, "") for hh in union_headers}
            out_rows.append(full)
    return union_headers, out_rows

def _salvage_rows_from_rawjson(headers: List[str], rows: List[Dict[str, Any]], context: Dict[str, Any] | None = None):
    """
    Light flattening for inline RawJSON blocks in rows.

        For each row with an inline RawJSON (string/dict) containing common contest keys,
        lift a few safe fields into columns without mutating or dropping the RawJSON itself:
            - Raw Contest Name (from 'name')
            - Raw Precincts Participating (from 'precinctsParticipating')
            - Raw Precincts Reporting (from 'precinctsReporting')
            - Raw Candidate Count (len(ballotOptions))
            - Raw Contest Type (from 'type')

    Does not read external files (RawJSONPath/RawId); only handles inline JSON
    to keep IO minimal and behavior predictable in tests.
    """
    headers = headers or []
    rows = rows or []
    ctx = context or {}
    if not rows:
        return headers, rows

    # Case-insensitive detector for 'RawJSON' column
    raw_col = None
    for h in headers:
        if isinstance(h, str) and h.lower().strip() == "rawjson":
            raw_col = h
            break
    if raw_col is None:
        # try best-effort match
        for h in headers:
            if isinstance(h, str) and "rawjson" in h.lower():
                raw_col = h
                break

    # Nothing to do if no inline RawJSON present
    if raw_col is None:
        return headers, rows

    import orjson as _orjson

    # Determine which fields to lift (allowlist optional via context)
    default_fields = [
        ("name", "Raw Contest Name"),
        ("precinctsParticipating", "Raw Precincts Participating"),
        ("precinctsReporting", "Raw Precincts Reporting"),
        ("ballotOptions", "Raw Candidate Count"),
        ("type", "Raw Contest Type"),
    ]
    allowlist = ctx.get("rawjson_flatten_allowlist")
    if isinstance(allowlist, (list, tuple, set)):
        allow = {str(x) for x in allowlist}
        fields = [(k, col) for (k, col) in default_fields if k in allow]
    else:
        fields = list(default_fields)
    added_cols = [col for _, col in fields]
    # Extend headers with any missing added columns
    for col in added_cols:
        if col not in headers:
            headers.append(col)

    out: list[dict] = []
    for r in rows:
        if not isinstance(r, dict):
            out.append(r)
            continue
        rec = dict(r)
        blob = rec.get(raw_col)
        obj = None
        if isinstance(blob, (bytes, bytearray)):
            try:
                obj = _orjson.loads(blob)
            except Exception:
                obj = None
        elif isinstance(blob, str):
            s = blob.strip()
            if s and (s.startswith("{") or s.startswith("[")):
                try:
                    obj = _orjson.loads(s)
                except Exception:
                    obj = None
        elif isinstance(blob, dict):
            obj = blob

        if isinstance(obj, dict):
            for key, col in fields:
                if key == "ballotOptions":
                    val = len(obj.get("ballotOptions", [])) if isinstance(obj.get("ballotOptions"), list) else None
                else:
                    val = obj.get(key)
                if val is not None and not rec.get(col):
                    rec[col] = val

        out.append(rec)

    return headers, out

def remove_footer_and_summary_rows(rows: List[Dict[str, Any]], headers: List[str]):
    """
    Remove obviously empty rows and simple summary/footer lines.
    Drops rows where all values are blank, or when Candidate/Party equals common summary tokens.
    """
    rows = rows or []
    headers = headers or []
    summary_re = re.compile(r"^(total|totals?|grand\s+total|summary|overall)$", re.I)
    out: list[dict] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        if not any(str(v or "").strip() for v in r.values()):
            continue
        cand = str(r.get("Candidate", "") or "").strip()
        party = str(r.get("Party", "") or "").strip()
        if (cand and summary_re.match(cand)) or (party and summary_re.match(party)):
            continue
        out.append(r)
    return out

def remove_outlier_and_empty_rows(rows: List[Dict[str, Any]]):
    """
    Conservative cleaner: drop rows where all fields are empty. No numeric outlier logic applied.
    """
    rows = rows or []
    return [r for r in rows if isinstance(r, dict) and any(str(v or "").strip() for v in r.values())]

# update exports
for _name in [
    "merge_multiline_candidate_rows",
    "combine_panel_tables_by_precinct",
    "_salvage_rows_from_rawjson",
    "remove_footer_and_summary_rows",
    "remove_outlier_and_empty_rows",
]:
    if _name not in __all__:
        __all__.append(_name)