"""
merge_utils.py
Utility to merge multiple header collections and row dicts into a unified table.
"""
from __future__ import annotations
from typing import List, Dict, Any, Tuple, Iterable
from .detect import normalize_header
from .logger_singleton import logger

def _iter_flat(items: Iterable):
    for x in items:
        if isinstance(x, (list, tuple)):
            for y in _iter_flat(x):
                yield y
        else:
            yield x

def merge_table_data(all_headers: List[Any], all_rows: List[Any]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Merge heterogeneous header collections and row dict/list objects into unified table.
    - Flattens nested header lists
    - Coerces non-string headers to strings
    - Deduplicates by normalized form (first occurrence wins)
    - Coerces list/tuple rows to dicts (Column N) if needed
    """
    normalized_seen = set()
    ordered: List[str] = []

    # Flatten & sanitize headers
    for raw in _iter_flat(all_headers or []):
        if raw is None:
            continue
        if isinstance(raw, (list, dict)):  # skip structured objects as header tokens
            raw_str = str(raw)
        else:
            raw_str = str(raw)
        nh = normalize_header(raw_str)
        if not nh:
            continue
        if nh not in normalized_seen:
            ordered.append(raw_str)
            normalized_seen.add(nh)

    # Pass 2: discover additional keys from rows
    sanitized_rows: List[Dict[str, Any]] = []
    for entry in all_rows or []:
        if isinstance(entry, dict):
            row_dict = {}
            for k, v in entry.items():
                # Coerce key
                if isinstance(k, (list, dict, tuple)):
                    k_str = str(k)
                else:
                    k_str = str(k)
                nh = normalize_header(k_str)
                if nh not in normalized_seen:
                    ordered.append(k_str)
                    normalized_seen.add(nh)
                row_dict[k_str] = v
            sanitized_rows.append(row_dict)
        elif isinstance(entry, (list, tuple)):
            # Map to existing headers (extend if needed)
            while len(ordered) < len(entry):
                col_name = f"Column {len(ordered)+1}"
                nh = normalize_header(col_name)
                if nh not in normalized_seen:
                    ordered.append(col_name)
                    normalized_seen.add(nh)
            row_dict = {ordered[i]: entry[i] for i in range(min(len(entry), len(ordered)))}
            if any(v not in ("", None) for v in row_dict.values()):
                sanitized_rows.append(row_dict)
        else:
            # Scalar row -> single column
            if "Value" not in ordered:
                ordered.append("Value")
            sanitized_rows.append({"Value": entry})

    # Final alignment
    unified: List[Dict[str, Any]] = []
    for r in sanitized_rows:
        aligned = {h: r.get(h, "") for h in ordered}
        if any(v not in ("", None) for v in aligned.values()):
            unified.append(aligned)

    logger.debug(f"[MERGE_UTILS] merged_headers={len(ordered)} rows={len(unified)}")
    return ordered, unified