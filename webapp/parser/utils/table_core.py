"""
table_core.py (refactored orchestrator)

Pipeline:
  1. Instantiate Detector (column + entity heuristics)
  2. Run extraction strategies concurrently (DOM + pure HTML groups)
  3. Deduplicate tables (keep largest per normalized header signature)
  4. Merge (if >1) panel-like tables
  5. Multiline candidate row merge
  6. RawJSON salvage to flat rows (if present)
  7. Basic cleaning (footer/outlier removal)
  8. Entity annotation (Detector)
  9. Harmonize headers/data
 10. Pivot to wide format
 11. Optional output (CSV/JSON preview)
 12. Emit metrics

External modules required (added in refactor):
  - detector.py
  - extraction_strategies.py
  - strategy_concurrency.py
  - salvage.py
  - detect.py
  - pivot.py
  - io.py

Expose:
  robust_table_extraction(page, extraction_context)
  build_table_from_page(page, extraction_context)

extraction_context optional keys:
  session_id, contest, percent_reported, location_header, coordinator,
  output_dir, output_basename

Returns:
  (headers: List[str], rows: List[Dict[str, Any]])
"""

from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional
import time

from .logger_singleton import logger
from .shared_logic import safe_get
import inspect

# Concurrency + strategies
from .strategy_concurrency import run_strategies_concurrently, run_strategies_concurrently_async
import asyncio
from .extraction_strategies import (
    strategy_dom_repetition,
    strategy_pattern_based,
    strategy_heading_associated,
    strategy_html_tables,
    strategy_ml_detection,
    strategy_selectolax_fallback,
    strategy_nlp_fallback,
)

# Salvage / cleaning
from .salvage import (
    merge_multiline_candidate_rows,
    combine_panel_tables_by_precinct,
    _salvage_rows_from_rawjson,
    remove_footer_and_summary_rows,
    remove_outlier_and_empty_rows,
)

# Detection / harmonization / pivot / IO
from .detector import Detector
from .detect import (
    harmonize_headers_and_data,
)
from .pivot import pivot_to_wide as pivot_to_wide_unified
from .output_utils import finalize_election_output

# Metric emitter (re-use from detect for consistency)
from .detect import emit_metric, normalize_header

# ------------------ Internal Helpers ------------------ #

def _deduplicate_tables(tables: List[Tuple[List[str], List[Dict[str, Any]], Dict[str, Any]]]):
    """
    Deduplicate by normalized header signature (alphabetical normalized headers).
    Keep the variant having the most data rows for each signature.
    """
    sig_map = {}
    for headers, rows, diag in tables:
        if not headers or not rows:
            continue
        sig = tuple(sorted(h.lower() for h in headers))
        if sig not in sig_map or len(rows) > len(sig_map[sig][1]):
            sig_map[sig] = (headers, rows, diag)
    return list(sig_map.values())

def _log_extraction_summary(extraction_logs: List[Dict[str, Any]], session_id: Optional[str]):
    summary = {
        "strategies_success": sum(1 for e in extraction_logs if e.get("success")),
        "strategies_total": len(extraction_logs),
        "rows_max": max((e.get("rows", 0) for e in extraction_logs), default=0),
        "headers_variants": len(extraction_logs),
    }
    logger.info({"type": "extraction_summary", "session_id": session_id, **summary})

def _annotate_entities_via_detector(detector: Detector, headers: List[str], data: List[Dict[str, Any]]):
    """
    Produce a minimal entity_info dict from Detector annotation for pivot compatibility.
    """
    ann = detector.annotate_entities(headers, data)
    entity_info = {
        "people": list(ann.people),
        "locations": list(ann.locations),
        "ballot_types": list(ann.ballot_types),
        "numbers": list(ann.numbers),
        # Keep placeholders for older interface compatibility
        "location_column": None,
        "percent_column": None,
        "detector": detector,
    }
    return entity_info

# ------------------ Core Orchestrator ------------------ #

def robust_table_extraction(
    page,
    extraction_context: Dict[str, Any] | None = None,
    existing_headers: List[str] | None = None,
    existing_data: List[Dict[str, Any]] | None = None
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Main table build routine (refactored).
    """
    t0 = time.time()
    ctx = extraction_context or {}
    session_id = safe_get(ctx, "session_id")
    coordinator = safe_get(ctx, "coordinator")  # Optional external NER/coordinator

    logger.info({
        "level": "INFO",
        "type": "table_core",
        "message": "[TABLE BUILDER] Start extraction",
        "session_id": session_id
    })

    detector = Detector(coordinator)
    ctx["detector"] = detector  # Expose for downstream / pivot

    collected: List[Tuple[List[str], List[Dict[str, Any]], Dict[str, Any]]] = []
    extraction_logs: List[Dict[str, Any]] = []

    # Include externally supplied data first (if any)
    if existing_headers and existing_data:
        collected.append((existing_headers, existing_data, {"strategy": "provided"}))
        extraction_logs.append({
            "strategy": "provided",
            "rows": len(existing_data),
            "columns": len(existing_headers),
            "success": True
        })

    # Strategy classification:
    # DOM-bound strategies (need real page object)
    dom_strategies = [
        strategy_dom_repetition,
        strategy_pattern_based,
        strategy_heading_associated,
        strategy_html_tables,
    ]
    # Pure HTML / text strategies (can run on snapshot concurrently)
    html_strategies = [
        strategy_ml_detection,
        strategy_selectolax_fallback,
        strategy_nlp_fallback,
    ]

    # Run concurrently
    try:
        strategy_results = run_strategies_concurrently(
            page,
            ctx,
            dom_strategies,
            html_strategies,
            max_workers=4
        )
        for h, d, diag in strategy_results:
            collected.append((h, d, diag))
            extraction_logs.append({
                "strategy": diag.get("strategy", "unknown"),
                "rows": len(d),
                "columns": len(h),
                "success": True
            })
    except Exception as e:
        logger.warning(f"[TABLE BUILDER] Concurrent strategies execution failed: {e}")

    if not collected:
        emit_metric("extraction_empty")
        _log_extraction_summary(extraction_logs, session_id)
        return [], []

    # Deduplicate
    deduped = _deduplicate_tables(collected)

    # If multiple, merge as a panel; else use single
    if len(deduped) > 1:
        headers, data = combine_panel_tables_by_precinct([(h, d) for h, d, _ in deduped])
    else:
        headers, data, _ = deduped[0]

    # Merge multiline candidate rows (e.g., candidate + party on separate lines)
    headers, data = merge_multiline_candidate_rows(headers, data)

    # RawJSON salvage (if any)
    headers, data = _salvage_rows_from_rawjson(headers, data)

    # Basic cleaning
    data = remove_footer_and_summary_rows(data, headers)
    data = remove_outlier_and_empty_rows(data)

    if not data or not headers:
        emit_metric("extraction_no_data_after_clean")
        _log_extraction_summary(extraction_logs, session_id)
        return [], []

    # Entity annotation via Detector (lightweight)
    entity_info = _annotate_entities_via_detector(detector, headers, data)

    # Harmonize before pivot
    headers, data = harmonize_headers_and_data(headers, data, ctx)

    # Provide location/percent hints from context if given
    if safe_get(ctx, "location_header") and "location_column" not in entity_info:
        entity_info["location_column"] = safe_get(ctx, "location_header")
    if safe_get(ctx, "percent_reported") and "percent_column" not in entity_info:
        entity_info["percent_column"] = "Percent Reported"

    # Pivot
    headers, data = pivot_to_wide_unified(headers, data, entity_info, coordinator, ctx)

    headers = _sanitize_headers(headers)

    emit_metric("extraction_success", rows=len(data), cols=len(headers))
    _log_extraction_summary(extraction_logs, session_id)

    logger.info({
        "level": "INFO",
        "type": "table_core",
        "message": "[TABLE BUILDER] Completed",
        "rows": len(data),
        "cols": len(headers),
        "elapsed_sec": round(time.time() - t0, 3),
        "session_id": session_id
    })

    return headers, data

def _sanitize_headers(headers: List[Any]) -> List[str]:
    seen = set()
    cleaned = []
    for h in headers or []:
        if h is None:
            continue
        hs = str(h).strip()
        if not hs:
            continue
        nh = normalize_header(hs)
        if not nh or nh in seen:
            continue
        seen.add(nh)
        cleaned.append(hs)
    return cleaned

def build_table_from_page(page, extraction_context: Dict[str, Any] | None = None):
    """
    Convenience wrapper:
      - Runs robust_table_extraction
      - Optionally writes CSV/JSON if output_dir provided in context
    """
    headers, data = robust_table_extraction(page, extraction_context)
    if headers and data:
        ctx = extraction_context or {}
        out_dir = safe_get(ctx, "output_dir")
        # Use new unified output utility if directory provided
        if out_dir:
            # Build base metadata from context
            meta = {
                "state": safe_get(ctx, "state"),
                "county": safe_get(ctx, "county"),
                "contests": safe_get(ctx, "contest"),
                "year": safe_get(ctx, "year"),
                "election_types": safe_get(ctx, "election_type"),
            }
            # finalize_election_output handles path creation & metadata
            finalize_election_output(
                headers=headers,
                data=data,
                context=ctx,
                contest=meta.get("contests"),
                state=meta.get("state"),
                county=meta.get("county"),
                session_id=safe_get(ctx, "session_id")
            )
    return headers, data

async def robust_table_extraction_async(
    page,
    extraction_context: Dict[str, Any] | None = None,
    existing_headers: List[str] | None = None,
    existing_data: List[Dict[str, Any]] | None = None
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Async variant of robust_table_extraction.
    All core logic mirrors sync path; only strategy execution + outer wrapper are async.
    """
    t0 = time.time()
    ctx = extraction_context or {}
    session_id = safe_get(ctx, "session_id")
    coordinator = safe_get(ctx, "coordinator")

    logger.info({
        "level": "INFO",
        "type": "table_core_async",
        "message": "[TABLE BUILDER][ASYNC] Start extraction",
        "session_id": session_id
    })

    detector = Detector(coordinator)
    ctx["detector"] = detector

    collected: List[Tuple[List[str], List[Dict[str, Any]], Dict[str, Any]]] = []
    extraction_logs: List[Dict[str, Any]] = []

    if existing_headers and existing_data:
        collected.append((existing_headers, existing_data, {"strategy": "provided"}))
        extraction_logs.append({
            "strategy": "provided",
            "rows": len(existing_data),
            "columns": len(existing_headers),
            "success": True
        })

    dom_strategies = [
        strategy_dom_repetition,
        strategy_pattern_based,
        strategy_heading_associated,
        strategy_html_tables,
    ]
    html_strategies = [
        strategy_ml_detection,
        strategy_selectolax_fallback,
        strategy_nlp_fallback,
    ]

    try:
        strategy_results = await run_strategies_concurrently_async(
            page, ctx, dom_strategies, html_strategies, max_workers=4
        )
        for h, d, diag in strategy_results:
            collected.append((h, d, diag))
            extraction_logs.append({
                "strategy": diag.get("strategy", "unknown"),
                "rows": len(d),
                "columns": len(h),
                "success": True
            })
    except Exception as e:
        logger.warning(f"[TABLE BUILDER][ASYNC] Concurrent strategies execution failed: {e}")

    if not collected:
        emit_metric("extraction_empty_async")
        _log_extraction_summary(extraction_logs, session_id)
        return [], []

    deduped = _deduplicate_tables(collected)

    if len(deduped) > 1:
        headers, data = combine_panel_tables_by_precinct([(h, d) for h, d, _ in deduped])
    else:
        headers, data, _ = deduped[0]

    headers, data = merge_multiline_candidate_rows(headers, data)
    headers, data = _salvage_rows_from_rawjson(headers, data)

    data = remove_footer_and_summary_rows(data, headers)
    data = remove_outlier_and_empty_rows(data)

    if not data or not headers:
        emit_metric("extraction_no_data_after_clean_async")
        _log_extraction_summary(extraction_logs, session_id)
        return [], []

    entity_info = _annotate_entities_via_detector(detector, headers, data)
    headers, data = harmonize_headers_and_data(headers, data, ctx)

    if safe_get(ctx, "location_header") and "location_column" not in entity_info:
        entity_info["location_column"] = safe_get(ctx, "location_header")
    if safe_get(ctx, "percent_reported") and "percent_column" not in entity_info:
        entity_info["percent_column"] = "Percent Reported"

    headers, data = pivot_to_wide_unified(headers, data, entity_info, coordinator, ctx)
    headers = _sanitize_headers(headers)

    emit_metric("extraction_success_async", rows=len(data), cols=len(headers))
    _log_extraction_summary(extraction_logs, session_id)

    logger.info({
        "level": "INFO",
        "type": "table_core_async",
        "message": "[TABLE BUILDER][ASYNC] Completed",
        "rows": len(data),
        "cols": len(headers),
        "elapsed_sec": round(time.time() - t0, 3),
        "session_id": session_id
    })
    return headers, data

async def build_table_from_page_async(page, extraction_context: Dict[str, Any] | None = None):
    """
    Async convenience wrapper.
    """
    headers, data = await robust_table_extraction_async(page, extraction_context)
    if headers and data:
        ctx = extraction_context or {}
        out_dir = safe_get(ctx, "output_dir")
        if out_dir:
            meta = {
                "state": safe_get(ctx, "state"),
                "county": safe_get(ctx, "county"),
                "contest": safe_get(ctx, "contest"),
                "year": safe_get(ctx, "year"),
                "election_type": safe_get(ctx, "election_type"),
            }
            finalize_election_output(
                headers=headers,
                data=data,
                context=ctx,
                contest=meta.get("contest"),
                state=meta.get("state"),
                county=meta.get("county"),
                session_id=safe_get(ctx, "session_id")
            )
    return headers, data

def auto_table_build(page, extraction_context=None, async_hint: bool | None = None):
    """
    Auto-detect sync/async environment. If already inside an event loop or async_hint=True,
    uses async variant. Otherwise sync path.
    """
    if async_hint is True or (async_hint is None and asyncio.get_event_loop().is_running()):
        return asyncio.ensure_future(robust_table_extraction_async(page, extraction_context))
    return robust_table_extraction(page, extraction_context)

__all__ = [
    "robust_table_extraction",
    "build_table_from_page",
    "auto_table_build",
    "robust_table_extraction_async",
    "build_table_from_page_async"
]