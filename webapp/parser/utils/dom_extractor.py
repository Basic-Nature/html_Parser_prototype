"""
dom_extractor.py
Structural DOM table heuristics (modern replacement for former legacy functions).
Focus: discover tabular repetition when no <table> tags exist.
"""
from __future__ import annotations
from typing import List, Dict, Any, Tuple
import statistics
from .browser_utils import (
    safe_locator, safe_nth, safe_count, safe_inner_text
)
from .logger_singleton import logger
from .shared_logic import safe_strip, safe_get
from .detect import is_likely_header, normalize_header

ContainerSelector = "div,section,article,main,ul,ol"

def _row_score(cells: List[str]) -> float:
    if not cells: return 0.0
    non_empty = sum(1 for c in cells if c.strip())
    return non_empty / len(cells)

def _extract_row_cells(row_node, max_cells=40) -> List[str]:
    cells_loc = safe_locator(row_node, "> *", logger)
    cnt = safe_count(cells_loc, logger)
    if cnt == 0:
        # fallback: treat row text as single cell
        txt = safe_inner_text(row_node, logger).strip()
        return [txt] if txt else []
    out=[]
    for i in range(min(cnt, max_cells)):
        c = safe_nth(cells_loc, i, logger)
        out.append(safe_inner_text(c, logger).strip() if c else "")
    return out

def _pick_header(rows_cells: List[List[str]]) -> Tuple[int, List[str]]:
    """
    Heuristic: choose row index that looks like header.
    Priority:
      1. is_likely_header(row)
      2. highest distinctiveness (unique tokens / total)
      3. first row
    """
    best_idx = 0
    best_headers = rows_cells[0] if rows_cells else []
    best_score = -1.0
    for idx, cells in enumerate(rows_cells[:6]):  # only inspect first few
        if not cells: continue
        labelness = 1.0 if is_likely_header(cells) else 0.0
        toks = [normalize_header(c) for c in cells if c]
        uniq_ratio = len(set(toks))/max(1,len(toks))
        score = labelness*2.0 + uniq_ratio
        if score > best_score:
            best_score = score
            best_idx = idx
            best_headers = cells
    # Ensure non-empty placeholders
    headers = []
    seen=set()
    for i,h in enumerate(best_headers):
        hh = h.strip() or f"Column {i+1}"
        # de-dup quickly
        base=hh
        k=2
        while normalize_header(hh) in seen:
            hh=f"{base}_{k}"
            k+=1
        seen.add(normalize_header(hh))
        headers.append(hh)
    return best_idx, headers

def extract_rows_and_headers_from_dom(page, context=None) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Core structural extraction: look for a container whose children form uniform rows.
    """
    context = context or {}
    diagnostics = {"strategy": "dom_structural"}
    try:
        containers = safe_locator(page, ContainerSelector, logger)
        best: Dict[str, Any] = {"score": 0, "headers": [], "rows": []}
        limit = min(400, safe_count(containers, logger))
        for ci in range(limit):
            cont = safe_nth(containers, ci, logger)
            if not cont: continue
            child_rows = safe_locator(cont, "> *", logger)
            row_cnt = safe_count(child_rows, logger)
            if row_cnt < 3 or row_cnt > 500:
                continue
            rows_cells: List[List[str]] = []
            widths=[]
            for ri in range(row_cnt):
                rn = safe_nth(child_rows, ri, logger)
                cells = _extract_row_cells(rn)
                # Skip ultra-short rows
                if len([c for c in cells if c]) < 1:
                    continue
                widths.append(len(cells))
                rows_cells.append(cells)
            if len(rows_cells) < 3:
                continue
            # Column count stability
            try:
                median_w = int(statistics.median(widths))
            except statistics.StatisticsError:
                continue
            if median_w < 2:
                continue
            stable_rows = [r for r in rows_cells if abs(len(r)-median_w) <= 1]
            if len(stable_rows) < max(3, int(0.5*len(rows_cells))):
                continue
            header_idx, headers = _pick_header(stable_rows)
            body_rows = stable_rows[header_idx+1:]
            if len(headers) < 2 or len(body_rows) < 2:
                continue
            dict_rows=[]
            non_empty_rows=0
            for r in body_rows:
                row_dict={}
                for i,h in enumerate(headers):
                    cell = r[i] if i < len(r) else ""
                    row_dict[h]=cell
                if any(v for v in row_dict.values()):
                    non_empty_rows+=1
                    dict_rows.append(row_dict)
            if non_empty_rows < 2:
                continue
            fill_score = sum(_row_score([row.get(h,"") for h in headers]) for row in dict_rows)/len(dict_rows)
            table_score = fill_score * len(dict_rows) * len(headers)
            if table_score > best["score"]:
                best.update({
                    "score": table_score,
                    "headers": headers,
                    "rows": dict_rows,
                    "container_index": ci,
                    "median_width": median_w,
                    "row_count": len(dict_rows)
                })
        if best["headers"] and best["rows"]:
            diagnostics.update({
                "score": best["score"],
                "container_index": best.get("container_index"),
                "median_width": best.get("median_width"),
                "row_count": best.get("row_count")
            })
            return best["headers"], best["rows"], diagnostics
        return [], [], diagnostics
    except Exception as e:
        logger.warning(f"[DOM_EXTRACTOR] failure: {e}")
        return [], [], diagnostics

def guess_headers_from_row(row_locator, context=None):
    cells = _extract_row_cells(row_locator)
    headers=[]
    seen=set()
    for i,c in enumerate(cells):
        base = c.strip() or f"Column {i+1}"
        name=base
        k=2
        while normalize_header(name) in seen:
            name=f"{base}_{k}"
            k+=1
        seen.add(normalize_header(name))
        headers.append(name)
    return headers, {"cells": len(headers)}

__all__ = [
    "extract_rows_and_headers_from_dom",
    "guess_headers_from_row"
]