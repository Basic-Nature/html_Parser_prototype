"""
extraction_strategies.py
Strategy registry + individual extraction strategies.
Each strategy returns a list of (headers, rows, diagnostics) tuples.
"""

from __future__ import annotations
from typing import Callable, List, Tuple, Dict, Any
import re

from .logger_singleton import logger
from .browser_utils import (
    safe_locator, safe_nth, safe_count, safe_inner_text, safe_content
)
from .shared_logic import safe_get, safe_append
from selectolax.parser import HTMLParser
from ..Context_Integration.Context_Library.constants import (
    LOCATION_KEYWORDS, NLP_SKIP_PHRASES, TOTAL_KEYWORDS, MISC_FOOTER_KEYWORDS
)
from .detect import (
    find_best_header,
    extract_percent_reported_from_heading,
    normalize_header,
    is_location_header,
    emit_metric,
    dynamic_detect_location_header,
    is_likely_header
)

from .dom_extractor import extract_rows_and_headers_from_dom
from .pattern_extractor import extract_with_patterns
from .date_utils import is_date_like 
import time
StrategyResult = Tuple[List[str], List[Dict[str, Any]], Dict[str, Any]]
StrategyFunc = Callable[[Any, dict | None], List[StrategyResult]]

STRATEGY_REGISTRY: List[Dict[str, Any]] = []

def register_strategy(fn: StrategyFunc, name: str, priority: int = 50, cost: int = 1, enabled: bool = True):
    STRATEGY_REGISTRY.append({
        "fn": fn,
        "name": name,
        "priority": priority,
        "cost": cost,
        "enabled": enabled
    })

def run_registered_strategies(page, context=None) -> List[StrategyResult]:
    context = context or {}
    collected: List[StrategyResult] = []
    overall_start = time.perf_counter()

    for meta in sorted(STRATEGY_REGISTRY, key=lambda m: m["priority"]):
        if not meta.get("enabled", True):
            continue
        name = meta["name"]
        start = time.perf_counter()
        try:
            results = meta["fn"](page, context) or []
            duration_ms = (time.perf_counter() - start) * 1000.0
            emit_metric("strategy_duration", method=name, ms=round(duration_ms, 2))
            for h, d, diag in results:
                if h and d:
                    diag = diag or {}
                    diag.setdefault("strategy", name)
                    diag["duration_ms"] = round(duration_ms, 2)
                    safe_append(collected, (h, d, diag))
                    emit_metric("strategy_success", method=name, rows=len(d))
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000.0
            logger.warning(f"[STRATEGY] {name} failed: {e}")
            emit_metric("strategy_error", method=name, error=str(e), ms=round(duration_ms, 2))

    # Optional merging of similar header schemas
    if context.get("merge_similar", True):
        collected = _merge_similar_tables(collected, context)

    total_ms = (time.perf_counter() - overall_start) * 1000.0
    emit_metric("strategy_pipeline_total_ms", ms=round(total_ms, 2), tables=len(collected))
    return collected

# ------------------- Individual Strategies -------------------

def strategy_html_tables(page, context=None) -> List[StrategyResult]:
    """Raw <table> extraction (lightweight)."""
    from .detect import extract_table_data
    results: List[StrategyResult] = []
    tables = safe_locator(page, "table", logger)
    for i in range(safe_count(tables, logger)):
        tab = safe_nth(tables, i, logger)
        headers, data, diagnostics = extract_table_data(tab, structure_info={"context": context})
        if headers and data:
            diagnostics["table_index"] = i
            results.append((headers, data, diagnostics))
    return results

def strategy_dom_repetition(page, context=None) -> List[StrategyResult]:
    headers, data, diagnostics = extract_rows_and_headers_from_dom(page, context=context)
    if headers and data:
        return [(headers, data, diagnostics)]
    return []

def strategy_pattern_based(page, context=None) -> List[StrategyResult]:
    h, d, diag = extract_with_patterns(page, context=context)
    return [(h, d, diag)] if h and d else []

def strategy_heading_associated(page, context=None) -> List[StrategyResult]:
    """Tables with nearest heading -> location column enrichment."""
    from .browser_utils import safe_locator, safe_count, safe_nth
    from .detect import extract_table_data
    results: List[StrategyResult] = []
    percent_global = ""
    # quick global percent scan
    spans = safe_locator(page, "span,div,p", logger)
    for i in range(min(40, safe_count(spans, logger))):
        txt = safe_inner_text(safe_nth(spans, i, logger), logger)
        val = extract_percent_reported_from_heading(txt)
        if val:
            percent_global = val
            break
    tables = safe_locator(page, "table", logger)
    for i in range(safe_count(tables, logger)):
        tab = safe_nth(tables, i, logger)
        # heading above
        heading_loc = safe_locator(tab, "xpath=preceding-sibling::*[self::h1 or self::h2 or self::h3 or self::h4 or self::h5][1]", logger)
        heading_txt = ""
        if safe_count(heading_loc, logger):
            heading_el = safe_nth(heading_loc, 0, logger)
            heading_txt = safe_inner_text(heading_el, logger).strip()
        h, d, diag = extract_table_data(tab, structure_info={"context": context})
        if not (h and d):
            continue
        loc_header = find_best_header(h, LOCATION_KEYWORDS)
        if loc_header and loc_header != "Precinct":
            # rename
            h = ["Precinct" if x == loc_header else x for x in h]
            for r in d:
                r["Precinct"] = r.pop(loc_header, r.get("Precinct", ""))
        elif not loc_header:
            if "Precinct" not in h:
                h = ["Precinct"] + h
            for r in d:
                r["Precinct"] = heading_txt
        percent_header = find_best_header(h, {"Percent Reported"})
        pv = extract_percent_reported_from_heading(heading_txt) or percent_global
        if pv:
            if not percent_header:
                h.append("Percent Reported")
                for r in d:
                    r["Percent Reported"] = pv
            else:
                for r in d:
                    if not r.get(percent_header):
                        r[percent_header] = pv
        diag["heading"] = heading_txt
        results.append((h, d, diag))
    return results


def strategy_ml_detection(page, context=None) -> List[StrategyResult]:
    """Use ML detector (already returns tables)."""
    from .ml_table_detector import detect_tables_ml
    html = safe_content(page)
    if not html or len(html) < 80:
        return []
    res = []
    tables = detect_tables_ml(html)
    for idx, t in enumerate(tables):
        headers = t.get("headers") if isinstance(t, dict) else []
        data = t.get("data") if isinstance(t, dict) else []
        if headers and data:
            res.append((headers, data, {"ml_index": idx}))
    return res


def strategy_selectolax_fallback(page, context=None) -> List[StrategyResult]:
    """Parse tables using selectolax only (lowest priority)."""
    html = safe_content(page)
    if not html:
        return []
    tree = HTMLParser(html)
    results: List[StrategyResult] = []
    for idx, tbl in enumerate(tree.css("table")):
        rows = tbl.css("tr")
        if not rows:
            continue
        header_cells = rows[0].css("th") or rows[0].css("td")
        headers = [c.text(strip=True) for c in header_cells]
        data = []
        for row in rows[1:]:
            cells = row.css("td") or row.css("th")
            rd = {headers[i]: cells[i].text(strip=True) if i < len(cells) else "" for i in range(len(headers))}
            data.append(rd)
        if headers and data:
            results.append((headers, data, {"fallback": True, "index": idx}))
    return results


def strategy_nlp_fallback(page, context=None) -> List[StrategyResult]:
    """
    Last resort: scan DOM text for label-number pairs (candidate-like).
    """
    elements = safe_locator(page, "*", logger)
    label_pat = re.compile(r"^[A-Za-z][A-Za-z\s\-']{1,40}$")
    vote_pat = re.compile(r"^\d{1,3}(,\d{3})*$")
    skip = NLP_SKIP_PHRASES
    labels = []
    votes = []
    count = safe_count(elements, logger)
    for i in range(count):
        try:
            el = safe_nth(elements, i, logger)
            txt = safe_inner_text(el, logger).strip()
        except Exception:
            continue
        if not txt or any(s.lower() in txt.lower() for s in skip):
            continue
        if vote_pat.fullmatch(txt.replace(",", "")):
            votes.append((i, txt))
        elif label_pat.match(txt):
            labels.append((i, txt))
    rows = []
    used = set()
    for vi, vv in votes:
        lab = None
        for li, lt in reversed(labels):
            if li < vi and li not in used:
                lab = lt
                used.add(li)
                break
        if lab:
            rows.append({"Label": lab, "Votes": vv})
    if rows:
        return [(["Label", "Votes"], rows, {"nlp_fallback": True})]
    return []

# --- Added helper merging + instrumentation utilities ---

def _normalized_header_tuple(headers: List[str]) -> tuple:
    return tuple(normalize_header(h) for h in headers if h)

def _merge_similar_tables(results: List[StrategyResult], context: dict) -> List[StrategyResult]:
    """
    Merge tables sharing identical normalized header schemas.
    Group size capped to prevent pathological merging.
    """
    if not results:
        return results
    max_merge = context.get("max_merge_group", 12)
    grouped: Dict[tuple, List[StrategyResult]] = {}
    for item in results:
        hdr_key = _normalized_header_tuple(item[0])
        grouped.setdefault(hdr_key, []).append(item)

    merged_out: List[StrategyResult] = []
    merged_groups = 0
    for key, group in grouped.items():
        if len(group) == 1 or len(group) > max_merge:
            merged_out.extend(group)
            continue
        # Merge rows
        base_headers = group[0][0]
        all_rows: List[Dict[str, Any]] = []
        diagnostics_combined: Dict[str, Any] = {
            "merged": True,
            "sources": [],
            "group_size": len(group)
        }
        for h, rows, diag in group:
            # Preserve original diag references
            diagnostics_combined["sources"].append(diag)
            for r in rows:
                all_rows.append(r)
        merged_groups += 1
        merged_out.append((
            base_headers,
            all_rows,
            diagnostics_combined
        ))
    if merged_groups:
        emit_metric("strategy_merged_groups", count=merged_groups)
    return merged_out

# Register default strategies (priority ascending)
register_strategy(strategy_dom_repetition, "dom_repetition", priority=10)
register_strategy(strategy_pattern_based, "patterns", priority=15)
register_strategy(strategy_heading_associated, "heading_tables", priority=20)
register_strategy(strategy_html_tables, "html_tables", priority=30)
register_strategy(strategy_ml_detection, "ml_detection", priority=40)
register_strategy(strategy_selectolax_fallback, "selectolax_fallback", priority=90)
register_strategy(strategy_nlp_fallback, "nlp_fallback", priority=100)