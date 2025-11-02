"""
merge_utils.py
Utility to merge multiple header collections and row dicts into a unified table.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

try:
    from .logger_singleton import logger  # type: ignore
except Exception:
    class _DummyLogger:
        def debug(self, *a, **k): pass
    logger = _DummyLogger()

from .salvage import collapse_ballot_synonym_columns


def merge_table_data(all_headers: List[Any], all_rows: List[Any]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Merge multiple (headers, rows) into a unified rectangular table, then normalize ballot/method columns.
    """
    ordered: List[str] = []
    rows_out: List[Dict[str, Any]] = []

    # Union of headers in appearance order
    for h in (all_headers or []):
        for col in (h or []):
            scol = str(col)
            if scol not in ordered:
                ordered.append(scol)

    # Align rows
    for r in (all_rows or []):
        if isinstance(r, dict):
            d = {h: r.get(h, "") for h in ordered}
        elif isinstance(r, (list, tuple)):
            d = {ordered[i]: (r[i] if i < len(ordered) else "") for i in range(len(ordered))}
        else:
            continue
        if any(v not in ("", None) for v in d.values()):
            rows_out.append(d)

    # Collapse ballot synonyms and backfill totals
    return collapse_ballot_synonym_columns(ordered, rows_out)