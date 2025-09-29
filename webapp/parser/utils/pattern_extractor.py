"""
pattern_extractor.py
Structured pattern / rule driven extraction.
Supports loading JSON-defined column patterns or regex row grouping.
"""
from __future__ import annotations
from typing import List, Dict, Any, Tuple
import json, re, os
from .logger_singleton import logger
from .shared_logic import safe_get
from .detect import normalize_header

def load_dom_patterns(path: str | None) -> List[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path,"r",encoding="utf-8") as f:
            data=json.load(f)
        if isinstance(data,list):
            return data
    except Exception as e:
        logger.warning(f"[PATTERN] load fail {e}")
    return []

def extract_with_patterns(page, context=None) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Pattern format (example):
    [
      {
        "name": "candidate_vote_pairs",
        "row_selector": "div.result-row",
        "cells": [
           {"selector": ".cand", "name": "Candidate"},
           {"selector": ".party", "name": "Party"},
           {"selector": ".votes", "name": "Votes"}
        ],
        "min_rows": 3
      }
    ]
    """
    context=context or {}
    pattern_file = safe_get(context,"pattern_file")
    patterns = load_dom_patterns(pattern_file)
    if not patterns:
        return [], [], {"strategy":"patterns","applied":0}
    from .browser_utils import safe_locator, safe_nth, safe_count, safe_inner_text
    best_headers=[]; best_rows=[]; applied=0
    for pat in patterns:
        try:
            row_sel = pat.get("row_selector")
            cell_defs = pat.get("cells", [])
            if not row_sel or not cell_defs:
                continue
            rows_loc = safe_locator(page, row_sel, logger)
            rc = safe_count(rows_loc, logger)
            if rc < pat.get("min_rows",2):
                continue
            tmp_rows=[]
            for i in range(rc):
                rloc = safe_nth(rows_loc,i,logger)
                row_dict={}
                non=0
                for cdef in cell_defs:
                    csel=cdef.get("selector")
                    cname=cdef.get("name") or csel
                    cloc = safe_locator(rloc, csel, logger)
                    val=""
                    if safe_count(cloc, logger):
                        val = safe_inner_text(safe_nth(cloc,0,logger), logger).strip()
                    row_dict[cname]=val
                    if val: non+=1
                if non:
                    tmp_rows.append(row_dict)
            if len(tmp_rows) >= pat.get("min_rows",2):
                hdrs=[]
                seen=set()
                for cdef in cell_defs:
                    nm = cdef.get("name") or cdef.get("selector","col")
                    nn = normalize_header(nm)
                    if nn not in seen:
                        hdrs.append(nm); seen.add(nn)
                score = len(tmp_rows)*len(hdrs)
                if score > len(best_rows)*len(best_headers):
                    best_headers, best_rows = hdrs, tmp_rows
                    applied+=1
        except Exception as e:
            logger.warning(f"[PATTERN] pattern error {pat.get('name')}: {e}")
    return best_headers, best_rows, {"strategy":"patterns","applied":applied}

__all__ = [
    "load_dom_patterns",
    "extract_with_patterns"
]