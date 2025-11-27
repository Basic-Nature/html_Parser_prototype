from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

try:
    import camelot
    _CAMELOT_AVAILABLE = True
except Exception:
    _CAMELOT_AVAILABLE = False

from ..Context_Integration.Context_Library.constants import (
    build_camelot_row_filter,
    get_camelot_title_regex,
)
from .salvage import normalize_ballot_column_name

# Centralized, canonical noise detectors from constants.py
TITLE_NOISE_RE = get_camelot_title_regex()
_ROW_NOISE_FN = build_camelot_row_filter()

def _normalize_headers(raw_headers: List[str]) -> List[str]:
    """
    Normalize PDF table headers with constants-driven ballot/method canon and dedupe.
    """
    out, seen = [], set()
    for i, h in enumerate(raw_headers or []):
        hs = (str(h) or "").strip() or f"Column {i+1}"
        hs = re.sub(r"\s+", " ", hs)
        hs = normalize_ballot_column_name(hs)
        base = hs
        k = 2
        while hs.lower() in seen:
            hs = f"{base}_{k}"
            k += 1
        seen.add(hs.lower())
        out.append(hs)
    return out

def _row_is_title_noise(rec: Dict[str, Any]) -> bool:
    """Backwards-compatible alias for the centralized row noise filter."""
    return _ROW_NOISE_FN(rec)

def _table_to_rows(table, limit_rows: int = 1500) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Convert Camelot table to normalized headers/rows with noise filtering.
    """
    try:
        df = table.df.replace("\n", " ", regex=True)
    except Exception:
        df = table.df
    headers = _normalize_headers(df.columns.tolist())
    rows: List[Dict[str, Any]] = []
    for _, r in df.head(limit_rows).iterrows():
        rec = {}
        empty = True
        for h, v in zip(headers, r.tolist()):
            vs = v.strip() if isinstance(v, str) else v
            rec[h] = vs
            if vs not in ("", None):
                empty = False
        if empty or _row_is_title_noise(rec):
            continue
        rows.append(rec)
    return headers, rows

def _score_table(headers: List[str], rows: List[Dict[str, Any]]) -> float:
    if not headers or not rows:
        return 0.0
    # moderate heuristic: numeric density + presence of vote-ish headers
    total_cells = len(headers) * len(rows)
    nums = 0
    vote_hits = sum(1 for h in headers if any(k in h.lower() for k in ("vote", "absentee", "early", "election day", "provisional", "%")))
    for r in rows:
        for v in r.values():
            s = str(v or "")
            core = s.replace(",", "").replace("%", "").strip()
            if core.isdigit():
                nums += 1
    nd = nums / max(1, total_cells)
    return round(0.6 * nd + 0.4 * (vote_hits / max(1, len(headers))), 4)

def attempt_camelot_extraction(pdf_path: str, session_id: str | None = None, pages: str = "1-end", max_tables: int = 15):
    if not _CAMELOT_AVAILABLE:
        return []
    results = []
    for flavor in ("lattice", "stream"):
        try:
            tables = camelot.read_pdf(
                pdf_path,
                pages=pages,
                flavor=flavor,
                strip_text="\n",
                suppress_stdout=True
            )
        except Exception:
            continue
        for t in list(tables)[:max_tables]:
            try:
                h, r = _table_to_rows(t)
                if not h or not r:
                    continue
                score = _score_table(h, r)
                results.append({
                    "headers": h,
                    "rows": r,
                    "score": score,
                    "flavor": flavor,
                    "page_range": getattr(t, "page", None)
                })
            except Exception:
                continue
        if len(results) >= max_tables:
            break
    results.sort(key=lambda x: (-x["score"], -len(x["rows"])))
    return results

def hybrid_fill_camelot(top_table: dict, ocr_lines: list[str] | None = None):
    # minimal no-op; hook for OCR enrichment if desired
    return

__all__ = [
    "_normalize_headers",
    "_row_is_title_noise",
    "_table_to_rows",
    "TITLE_NOISE_RE",
]