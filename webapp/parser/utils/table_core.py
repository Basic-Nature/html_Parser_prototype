# webapp/parser/utils/table_core.py
# -----------------------------------------------------------------------------------
# This module provides centralized utilities for table extraction, harmonization,
# annotation, and verification in web scraping and automation tasks.
# It serves as the single source of truth for all table-related operations.
# -----------------------------------------------------------------------------------
"""
table_core.py

Centralized Table Extraction, Harmonization, Annotation, and Verification Utilities

This module is the SINGLE SOURCE OF TRUTH for:
- Robust, multi-strategy table extraction from HTML/DOM (tables, repeated DOM, patterns, NLP fallback)
- Harmonization and cleaning of headers/data
- Entity annotation (NLP/NER) and structure verification
- User feedback and correction loop (interactive/CLI)
- Table structure detection/classification and pivoting

All candidate generation and scoring is handled in dynamic_table_extractor.py.
All high-level orchestration is handled in table_builder.py.

This ensures all table structure learning, harmonization, and feedback are centralized.
"""
from __future__ import annotations
import os
import orjson
import re
import unicodedata
import glob
import re
import string
import difflib
import types
import dateutil.parser
from rich.table import Table as RichTable
from urllib.parse import urlparse
from langdetect import detect, DetectorFactory
from .browser_utils import (
    safe_nth, safe_locator, safe_count, safe_inner_text, safe_content, 
    safe_get_attribute, safe_evaluate
)
from .shared_logic import (
    safe_get, safe_lower, safe_append, safe_pop, safe_add,
    safe_items, safe_values, safe_replace, safe_isalpha,
    safe_extract, safe_scheme, safe_netloc, safe_geturl, safe_strip,
    safe_translate, safe_isdigit, safe_split, safe_startswith, safe_keys
)
from difflib import SequenceMatcher
from collections import Counter
from typing import List, Dict, Any, Tuple, TYPE_CHECKING
import time
from selectolax.parser import HTMLParser
import hashlib
from .logger_singleton import logger, console, prompt
from .ml_table_detector import detect_tables_ml
from ..config import CACHE_DIR, LOG_DIR
from difflib import get_close_matches
from ..Context_Integration.Context_Library.constants import (
    PARTY_KEYWORDS, CANDIDATE_KEYWORDS, BALLOT_TYPES_SORT_ORDER, 
    CANDIDATE_KEYWORDS, PARTY_KEYWORDS, BALLOT_TYPES, CONTEST_TITLE_KEYWORDS,
    KNOWN_COUNTY_TO_PRECINCTS_MAP, TOTAL_KEYWORDS, LOCATION_ABBREVIATIONS,
    LOCATION_KEYWORDS, PERCENT_KEYWORDS, MISC_FOOTER_KEYWORDS, NLP_SKIP_PHRASES,
    LIKELY_ROW_CLASSES, CONTEST_TITLE_TAGS, CONTEST_TITLE_MIN_WORDS, CONTEST_TITLE_SKIP_PHRASES 
)
from ..Context_Integration.librarian import (
    normalize_segment_text, get_safe_log_path
)
if TYPE_CHECKING:
    from ..Context_Integration.context_coordinator import ContextCoordinator
# --- CONSTANTS & GLOBALS ---
TABLE_STRUCTURE_CACHE_PATH = os.path.join(CACHE_DIR, "table_structure_cache.json")

context_cache = {}

# ===================================================================
# MAIN EXTRACTION ENTRY POINT
# ===================================================================

def robust_table_extraction(page, extraction_context=None, existing_headers=None, existing_data=None):
    """
    Unified, persistent table extraction pipeline with robust location detection and forced wide format.
    Now supports ML-driven context, segments, and panels from html_scanner.
    """
    from ..Context_Integration.context_coordinator import ContextCoordinator
    coordinator = ContextCoordinator()
    def safe_json(obj):
        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                if k in ("coordinator", "ContextCoordinator"):
                    continue
                if isinstance(v, (types.FunctionType, types.ModuleType)) or hasattr(v, "__dict__"):
                    continue
                try:
                    orjson.dumps(v, option=orjson.OPT_INDENT_2)
                    result[k] = safe_json(v)
                except Exception:
                    continue
            return result
        elif isinstance(obj, list):
            return [safe_json(v) for v in obj if not hasattr(v, "__dict__")]
        else:
            return obj

    extraction_logs = []
    all_tables = []

    # --- Try to use cached structure if available ---
    domain = None
    if extraction_context:
        domain = safe_get(extraction_context, "domain", None) or safe_get(extraction_context, "url", None)
    cached_headers, cached_data = None, None
    if domain:
        # Try to get cached structure for this domain and headers
        # If existing_headers provided, use those for signature, else try after extraction
        if existing_headers:
            cached = get_cached_table_structure(domain, existing_headers)
            if cached:
                logger.info(f"[TABLE BUILDER] Using cached table structure for domain: {domain}")
                if isinstance(cached, dict):
                    return cached.get("headers", []), cached.get("data", [])
                else:
                    logger.warning(f"[TABLE BUILDER] Cached table structure is not a dict: {type(cached)}")
                    return [], []

    # --- ML context integration ---
    ml_confidence = safe_get(extraction_context, "ml_confidence", None) if extraction_context else None
    association_log = safe_get(extraction_context, "association_log", None) if extraction_context else None
    segments = safe_get(extraction_context, "segments", None) if extraction_context else None
    panels = safe_get(extraction_context, "panels", None) if extraction_context else None

    # 1. DOM structure extraction (divs, lists, etc.)
    try:
        headers_dom, data_dom, diagnostics_dom = extract_rows_and_headers_from_dom(
            page,
            coordinator = safe_get(extraction_context, "coordinator", None) if extraction_context else None,
            context = extraction_context
        )
        if headers_dom and data_dom:
            all_tables.append((headers_dom, data_dom))
            extraction_logs.append({
                "method": "repeated_dom",
                "headers": headers_dom,
                "rows": len(data_dom),
                "columns": len(headers_dom),
                "success": True,
                "context": extraction_context,
                "diagnostics": diagnostics_dom,
            })
    except Exception as e:
        logger.error(f"[TABLE BUILDER] DOM structure extraction failed: {e}")
        extraction_logs.append({
            "method": "repeated_dom",
            "error": str(e),
            "success": False,
            "context": extraction_context,
            "diagnostics": None,
        })

    # 2. Pattern-based extraction (approved DOM patterns)
    try:
        headers_pat, data_pat, diagnostics_pat = extract_with_patterns(page, extraction_context)
        if headers_pat and data_pat:
            all_tables.append((headers_pat, data_pat))
            extraction_logs.append({
                "method": "pattern",
                "headers": headers_pat,
                "rows": len(data_pat),
                "columns": len(headers_pat),
                "success": True,
                "context": extraction_context,
                "diagnostics": diagnostics_pat,
            })
    except Exception as e:
        logger.error(f"[TABLE BUILDER] Pattern extraction failed: {e}")
        extraction_logs.append({
            "method": "pattern",
            "error": str(e),
            "success": False,
            "context": extraction_context,
            "diagnostics": None,
        })

    # 3. Standard HTML table extraction
    try:
        tables = safe_locator(page, "table", logger)
        table_count = safe_count(tables, logger)
        for i in range(table_count):
            table = safe_nth(tables, i, logger)
            if table is not None:
                headers_tab, data_tab, diagnostics_tab = extract_table_data(
                    table,
                    coordinator = safe_get(extraction_context, "coordinator", None) if extraction_context else None,
                    structure_info = {"context": extraction_context} if extraction_context else None
                )
                if headers_tab and data_tab:
                    all_tables.append((headers_tab, data_tab))
                    extraction_logs.append({
                        "method": "table",
                        "headers": headers_tab,
                        "rows": len(data_tab),
                        "columns": len(headers_tab),
                        "success": True,
                        "context": extraction_context,
                        "diagnostics": diagnostics_tab,
                    })
    except Exception as e:
        logger.error(f"[TABLE BUILDER] Table extraction failed: {e}")
        extraction_logs.append({
            "method": "table",
            "error": str(e),
            "success": False,
            "context": extraction_context,
            "diagnostics": None,
        })

    # 4. Table extraction with heading/location context
    try:
        headers_loc, data_loc, diagnostics_loc = extract_all_tables_with_location(
            page, 
            coordinator = safe_get(extraction_context, "coordinator", None) if extraction_context else None,
            context = extraction_context
        )
        if headers_loc and data_loc:
            all_tables.append((headers_loc, data_loc))
            extraction_logs.append({
                "method": "table_with_heading",
                "headers": headers_loc,
                "rows": len(data_loc),
                "columns": len(headers_loc),
                "success": True,
                "context": extraction_context,
                "diagnostics": diagnostics_loc,
            })
    except Exception as e:
        logger.error(f"[TABLE BUILDER] Table-with-heading extraction failed: {e}")
        extraction_logs.append({
            "method": "table_with_heading",
            "error": str(e),
            "success": False,
            "context": extraction_context,
            "diagnostics": None,
        })

    # 5. ML-based table detection (optionally use ML context/segments/panels)
    try:
        ml_tables = ml_based_table_detection(page, extraction_context)
        for idx, (headers_ml, data_ml, diagnostics_ml) in enumerate(ml_tables):
            if headers_ml and data_ml:
                all_tables.append((headers_ml, data_ml))
                extraction_logs.append({
                    "method": "ml_table_detection",
                    "headers": headers_ml,
                    "rows": len(data_ml),
                    "columns": len(headers_ml),
                    "success": True,
                    "context": extraction_context,
                    "diagnostics": diagnostics_ml,
                })
    except Exception as e:
        logger.error(f"[TABLE BUILDER] ML table detection failed: {e}")
        extraction_logs.append({
            "method": "ml_table_detection",
            "error": str(e),
            "success": False,
            "context": extraction_context,
            "diagnostics": None,
        })

    # 6. Nested table extraction
    try:
        nested_tables = nested_table_extraction(page)
        for idx, (headers_nested, data_nested, diagnostics_nested) in enumerate(nested_tables):
            if headers_nested and data_nested:
                all_tables.append((headers_nested, data_nested))
                extraction_logs.append({
                    "method": "nested_table",
                    "headers": headers_nested,
                    "rows": len(data_nested),
                    "columns": len(headers_nested),
                    "success": True,
                    "context": extraction_context,
                    "diagnostics": diagnostics_nested,
                })
    except Exception as e:
        logger.error(f"[TABLE BUILDER] Nested table extraction failed: {e}")
        extraction_logs.append({
            "method": "nested_table",
            "error": str(e),
            "success": False,
            "context": extraction_context,
            "diagnostics": None,
        })

    # 7. Custom plugin extraction
    try:
        plugin_tables = custom_plugin_extraction(page, extraction_context)
        for idx, (headers_plugin, data_plugin, diagnostics_plugin) in enumerate(plugin_tables):
            if headers_plugin and data_plugin:
                all_tables.append((headers_plugin, data_plugin))
                extraction_logs.append({
                    "method": "plugin",
                    "headers": headers_plugin,
                    "rows": len(data_plugin),
                    "columns": len(headers_plugin),
                    "success": True,
                    "context": extraction_context,
                    "diagnostics": diagnostics_plugin,
                })
    except Exception as e:
        logger.error(f"[TABLE BUILDER] Plugin extraction failed: {e}")
        extraction_logs.append({
            "method": "plugin",
            "error": str(e),
            "success": False,
            "context": extraction_context,
            "diagnostics": None,
        })

    # 8. Add any existing headers/data provided
    if existing_headers and existing_data and len(existing_headers) > 0 and len(existing_data) > 0:
        all_tables.append((existing_headers, existing_data))
        extraction_logs.append({
            "method": "existing",
            "headers": existing_headers,
            "rows": len(existing_data),
            "columns": len(existing_headers),
            "success": True,
            "context": extraction_context,
            "diagnostics": None,
        })

    # 9. Robust HTML fallback using selectolax
    try:
        fallback_tables = robust_html_fallback_extraction(page)
        for idx, (headers_fallback, data_fallback, diagnostics_fallback) in enumerate(fallback_tables):
            if headers_fallback and data_fallback:
                all_tables.append((headers_fallback, data_fallback))
                extraction_logs.append({
                    "method": "html_fallback",
                    "headers": headers_fallback,
                    "rows": len(data_fallback),
                    "columns": len(headers_fallback),
                    "success": True,
                    "context": extraction_context,
                    "diagnostics": diagnostics_fallback,
                })
    except Exception as e:
        logger.error(f"[TABLE BUILDER] HTML fallback extraction failed: {e}")
        extraction_logs.append({
            "method": "html_fallback",
            "error": str(e),
            "success": False,
            "context": extraction_context,
            "diagnostics": None,
        })

    logger.info(f"[TABLE BUILDER] Extraction summary: {orjson.dumps(safe_json(extraction_logs), option=orjson.OPT_INDENT_2)}")

    # --- Deduplicate tables by header signature ---
    unique_tables = {}
    for headers, data in all_tables:
        sig = tuple(normalize_header(h) for h in headers)
        if sig not in unique_tables or (len(data) > len(unique_tables[sig][1])):
            unique_tables[sig] = (headers, data)
    all_tables = list(unique_tables.values())

    # --- Combine, merge, annotate, pivot, feedback ---
    if all_tables:
        # 1. Combine all panel tables into one
        combined_headers, combined_data = combine_panel_tables_by_precinct(all_tables)
        # 2. Merge candidate/party fields
        combined_headers, combined_data = merge_multiline_candidate_rows(combined_headers, combined_data)
        # 3. Entity annotation and structure verification (pass ML context)
        coordinator = safe_get(extraction_context, "coordinator", None) if extraction_context else None
        entity_info = {
            "ml_confidence": ml_confidence,
            "association_log": association_log,
            "segments": segments,
            "panels": panels,
        }
        combined_headers, combined_data, entity_info = nlp_entity_annotate_table(
            combined_headers, combined_data, context=extraction_context, coordinator=coordinator
        )
        # 4. Pivot to wide format
        combined_headers, combined_data = pivot_to_wide_format(
            combined_headers, combined_data, entity_info, coordinator, extraction_context
        )
        # 5. Feedback/correction loop (user-in-the-loop)
        combined_headers, combined_data = feedback_correction_loop(combined_headers, combined_data, extraction_context)
        if domain:
            cache_table_structure(domain, combined_headers, {
                "headers": combined_headers,
                "data": combined_data,
                "entity_info": entity_info,
                "context": extraction_context
            })
            logger.info(f"[TABLE BUILDER] Cached table structure for domain: {domain}")

        return combined_headers, combined_data

    # --- Only now try fallback NLP extraction ---
    try:
        headers, data = fallback_nlp_candidate_vote_scan(page)
        extraction_logs.append({
            "method": "nlp_fallback",
            "headers": headers,
            "rows": len(data),
            "columns": len(headers),
            "success": bool(headers and data),
            "context": extraction_context
        })
        if headers and data:
            logger.warning("[TABLE BUILDER] Fallback NLP extraction used. Only candidate/vote pairs extracted.")
            return headers, data
    except Exception as e:
        logger.error(f"[TABLE BUILDER] NLP fallback extraction failed: {e}")
        extraction_logs.append({
            "method": "nlp_fallback",
            "error": str(e),
            "success": False,
            "context": extraction_context
        })

    logger.warning("[TABLE BUILDER] No extraction method succeeded.")
    return [], []

# ===================================================================
# EXTRACTION STRATEGIES (HTML, DOM, PATTERNS, NLP)
# ===================================================================

def extract_percent_reported_from_heading(heading):
    """Extract percent reported or fully reported from heading text."""
    # Look for patterns like '80% Reported', 'Fully Reported', etc.
    percent_pattern = re.compile(r"(\d{1,3})\s*%[\s\-]*reported", re.I)
    match = percent_pattern.search(heading)
    if match:
        return f"{match.group(1)}%"
    try:
        if isinstance(heading, str) and "fully reported" in heading.lower():
            return "100%"
    except Exception:
        pass
    return ""

def extract_percent_reported_from_page(page):
    """Try to extract percent reported from the page outside the table."""
    # Look for common phrases in spans/divs
    for selector in ["span", "div", "p"]:
        elements = safe_locator(page, selector, logger)
        count = safe_count(elements, logger)
        for i in range(count):
            element = safe_nth(elements, i, logger)
            text = safe_inner_text(element, logger).strip() if element else ""
            if not text:
                continue
            percent = extract_percent_reported_from_heading(text)
            if percent:
                return percent
    return ""

def extract_all_tables_with_location(page, coordinator=None, context=None):
    """
    Extract all tables, associating each with the nearest section/district/heading.
    Dynamically chooses between panel-based and section-heading-based extraction,
    scores each, and merges/patches missing information if possible.
    """
    from .dynamic_table_extractor import (
        find_tables_with_panel_headings,
        find_tables_with_section_headings,
    )
    from ..Context_Integration.context_coordinator import ContextCoordinator
    coordinator = coordinator or ContextCoordinator()
    extraction_types = [
        ("panel", find_tables_with_panel_headings(page)),
        ("section", find_tables_with_section_headings(page)),
    ]

    percent_reported_global = extract_percent_reported_from_page(page)
    extraction_results = []

    for method, tables_with_headings in extraction_types:
        all_headers = set()
        all_panel_rows = []
        all_panel_headers = set()
        all_entity_previews = []
        for heading, table in tables_with_headings:
            headers, data, entity_preview = extract_table_data(
                table, coordinator=coordinator, structure_info={"context": context or {}}
            )
            if not headers or not data:
                continue

            # --- Always normalize location column to "Precinct" ---
            location_col = find_best_header(headers, LOCATION_KEYWORDS)
            if not location_col:
                if "Precinct" not in headers:
                    headers = ["Precinct"] + headers
                for row in data:
                    row["Precinct"] = heading
                location_col = "Precinct"
            else:
                if location_col != "Precinct":
                    headers = ["Precinct" if h == location_col else h for h in headers]
                    for row in data:
                        row["Precinct"] = row.pop(location_col)
                for row in data:
                    if not row.get("Precinct", []):
                        row["Precinct"] = heading

            percent_col = find_best_header(headers, PERCENT_KEYWORDS)
            percent_value = extract_percent_reported_from_heading(heading) or percent_reported_global
            if not percent_col:
                percent_col = "Percent Reported"
                if percent_col not in headers:
                    headers.append(percent_col)
                for row in data:
                    row[percent_col] = percent_value
            else:
                for row in data:
                    if not row.get(percent_col, []):
                        row[percent_col] = percent_value

            all_headers.update(headers)
            all_panel_rows.extend(data)
            all_entity_previews.append(entity_preview)
        candidate_cols = [h for h in all_panel_headers if any(k in safe_lower(h) for k in CANDIDATE_KEYWORDS)]
        ballot_types_cols = [h for h in all_panel_headers if any(bt in safe_lower(h) for bt in BALLOT_TYPES)]
        final_headers = ["Precinct"] + sorted(set(candidate_cols + ballot_types_cols)) + [
            h for h in all_panel_headers if h not in candidate_cols + ballot_types_cols + ["Precinct"]
        ]
        all_panel_headers = list(all_headers)
        all_panel_headers, all_panel_rows = harmonize_headers_and_data(final_headers, all_panel_rows)
        extraction_results.append({
            "method": method,
            "headers": all_panel_headers,
            "data": all_panel_rows,
            "entity_previews": all_entity_previews,
            "score": 0
        })

    # --- Score each extraction result using ML/NLP if available ---
    for result in extraction_results:
        score = 0
        if coordinator and hasattr(coordinator, "score_header"):
            scores = [coordinator.score_header(h, {}) for h in safe_get(result, "headers", [])]
            score = sum(scores) / len(scores) if scores else 0
        score += 0.1 * min(len(safe_get(result, "data", [])) / 10.0, 1.0)
        score += 0.1 * min(len(safe_get(result, "headers", [])) / 8.0, 1.0)
        result["score"] = score

    # --- Robust patching logic using context_coordinator ---
    def patch_missing_info(primary, secondary):
        """
        Patch missing headers and their values from secondary into primary.
        Uses sec_headers for robust logic and safe appending.
        """
        patched = False
        sec_headers = set(safe_get(secondary, "headers", []))
        primary_headers = set(safe_get(primary, "headers", []))
        sec_data = safe_get(secondary, "data", [])
        primary_data = safe_get(primary, "data", [])
        for h in sec_headers:
            if h not in primary_headers:
                safe_append(safe_get(primary, "headers", []), h)
                for i, row in enumerate(primary_data):
                    if i < len(sec_data):
                        row[h] = safe_get(sec_data[i], h, "")
                    else:
                        row[h] = ""
                patched = True
        return patched

    # --- Pick the best extraction by score, patch with info from others if possible ---
    extraction_results.sort(key=lambda r: r["score"], reverse=True)
    best = extraction_results[0] if extraction_results else None

    # Patch all others into best, not just the second-best
    if best and len(extraction_results) > 1:
        for other in extraction_results[1:]:
            patch_missing_info(best, other)
            # Optionally, use context_coordinator for more advanced row association in the future

    # --- Combine all panel tables if more than one ---
    all_tables = [(safe_get(r, "headers", []), safe_get(r, "data", [])) for r in extraction_results if safe_get(r, "headers", []) and safe_get(r, "data", [])]
    if len(all_tables) > 1:
        headers, data = combine_panel_tables_by_precinct(all_tables)
        # Optionally, merge entity previews as well
        entity_previews = []
        for r in extraction_results:
            entity_previews.extend(safe_get(r, "entity_previews", []))
        return headers, data, entity_previews

    if best:
        return safe_get(best, "headers", []), safe_get(best, "data", []), safe_get(best, "entity_previews", [])
    return [], [], []

def extract_table_data(table, coordinator=None, structure_info=None) -> Tuple[List[str], List[Dict[str, Any]], dict]:
    """
    Extracts headers and data from a Playwright table locator.
    Uses advanced NLP/NER, ML scoring, fuzzy and value-based matching to robustly detect entity columns.
    Improves detection for location and percent reported columns.
    Returns headers, data, and a meta dict with entity preview and detected location/percent columns.
    Now walks the DOM for best-matching columns and values, scoring all candidates and picking the best.
    """
    from ..Context_Integration.context_coordinator import ContextCoordinator
    coordinator = coordinator or ContextCoordinator()
    if table is None:
        logger.error("[TABLE BUILDER][extract_table_data] Table locator is None.")
        return [], [], {}

    logger.info("[TABLE BUILDER][extract_table_data] Starting table extraction.")
    headers = []
    data = []
    entity_preview = {
        "candidates": set(),
        "ballot_types": set(),
        "numbers": set(),
        "locations": set(),
        "location_column": None,
        "percent_column": None,
    }

    try:
        # --- Extract headers ---
        header_cells = safe_locator(table, "thead tr th", logger)
        if safe_count(header_cells, logger) == 0:
            first_row = safe_nth(safe_locator(table, "tr", logger), 0, logger)
            header_cells = safe_locator(first_row, "th, td", logger) if first_row else None
        for i in range(safe_count(header_cells, logger)):
            text = safe_inner_text(safe_nth(header_cells, i, logger), logger).strip()
            headers.append(text if text else f"Column {i+1}")

        # --- Extract rows ---
        rows = safe_locator(table, "tbody tr", logger)
        if safe_count(rows, logger) == 0:
            all_rows = safe_locator(table, "tr", logger)
            rows = all_rows

        for i in range(safe_count(rows, logger)):
            row = {}
            row_locator = safe_nth(rows, i, logger)
            cells = safe_locator(row_locator, "td, th", logger) if row_locator else None
            if safe_count(cells, logger) == 0:
                continue
            for j in range(safe_count(cells, logger)):
                cell = safe_nth(cells, j, logger)
                if j < len(headers):
                    row[headers[j]] = safe_inner_text(cell, logger).strip()
                else:
                    row[f"Extra_{j+1}"] = safe_inner_text(cell, logger).strip()
            if any(v for v in row.values()):
                data.append(row)

        # --- After extracting headers and data ---
        context = safe_get(structure_info, "context", {}) if structure_info else {}
        panel_heading = (
            safe_get(context, "panel_heading", None)
            or safe_get(context, "Precinct", None)
            or safe_get(context, "district", None)
        )
        location_col = (
            safe_get(entity_preview, "location_column", None)
            or next((h for h in headers if is_location_header(h)), None)
            or "Precinct"
        )
        if location_col and location_col != "Precinct":
            headers = ["Precinct" if h == location_col else h for h in headers]
            for row in data:
                row["Precinct"] = safe_pop(row, location_col, "")
            location_col = "Precinct"
        if not location_col and panel_heading:
            for row in data:
                row["Precinct"] = panel_heading
            if "Precinct" not in headers:
                headers = ["Precinct"] + headers
            location_col = "Precinct"

        unique_locations = sorted(
            set(str(safe_get(row, location_col, "") or "") for row in data if safe_get(row, location_col, ""))
        )
        unique_candidates = sorted(set(safe_get(row, "Candidate", "") for row in data if safe_get(row, "Candidate", "")))
        n_candidates = len(safe_get(entity_preview, "candidates", []))
        n_ballot_types = len(safe_get(entity_preview, "ballot_types", []))
        n_numbers = len(safe_get(entity_preview, "numbers", []))
        n_locations = len(unique_locations)
        loc_col_disp = location_col if location_col else "N/A"
        pct_col_disp = safe_get(entity_preview, "percent_column", "N/A")

        if location_col and len(unique_locations) <= 1 and panel_heading:
            for row in data:
                row[location_col] = panel_heading

        county = safe_lower(safe_get(context, "county", "")) if context else ""
        known_districts = set()
        if coordinator and hasattr(coordinator, "library"):
            county_map = KNOWN_COUNTY_TO_PRECINCTS_MAP
            if county and county_map.get(county.title(), []):
                known_districts = set(d.lower() for d in county_map[county.title()])

        # --- Robust Location & Percent Detection: Score all candidates, don't stop at first ---
        location_candidates = []
        percent_candidates = []
        percent_patterns = set(PERCENT_KEYWORDS)

        for h in headers:
            score = 0
            if coordinator:
                ents = []
                if hasattr(coordinator, "extract_entities"):
                    ents = coordinator.extract_entities(h)
                for ent, label in ents:
                    if label in {"GPE", "LOC", "FAC"} and safe_lower(h) != "candidate":
                        score += 1.0
                if is_location_header(h) and safe_lower(h) != "candidate":
                    score += coordinator.score_header(h, {}) if hasattr(coordinator, "score_header") else 0.5
            if is_location_header(h) and safe_lower(h) != "candidate":
                score += 0.3
            if known_districts:
                col_vals = [safe_lower(str(safe_get(row, h, ""))) for row in data]
                match_count = sum(
                    1 for v in col_vals
                    if v in known_districts or difflib.get_close_matches(v, known_districts, n=1, cutoff=0.8)
                )
                if match_count / max(1, len(col_vals)) > 0.5:
                    score += 0.7
            col_vals = [str(safe_get(row, h, "")) for row in data]
            unique_vals = len(set(col_vals))
            if unique_vals > 3 and not all(v.replace(",", "").isdigit() for v in col_vals if v):
                score += 0.2
            if score > 0:
                location_candidates.append((h, score))

            # Percent detection
            pscore = 0
            if any(kw in safe_lower(h) for kw in percent_patterns):
                pscore += 1.0
            elif "%" in h:
                pscore += 0.8
            if pscore > 0:
                percent_candidates.append((h, pscore))

        dom_location_scores = {}
        dom_percent_scores = {}
        for h in headers:
            col_vals = [str(safe_get(row, h, "")) for row in data]
            if known_districts:
                match_count = sum(
                    1 for v in col_vals
                    if safe_lower(v) in known_districts or difflib.get_close_matches(safe_lower(v), known_districts, n=1, cutoff=0.8)
                )
                dom_location_scores[h] = match_count / max(1, len(col_vals))
            percent_count = sum(1 for v in col_vals if "%" in v)
            dom_percent_scores[h] = percent_count / max(1, len(col_vals))
        for h, v in dom_location_scores.items():
            if v > 0.5:
                location_candidates.append((h, 0.5 + v))
        for h, v in dom_percent_scores.items():
            if v > 0.5:
                percent_candidates.append((h, 0.5 + v))

        location_candidates = sorted(location_candidates, key=lambda x: x[1], reverse=True)
        percent_candidates = sorted(percent_candidates, key=lambda x: x[1], reverse=True)
        location_col = location_candidates[0][0] if location_candidates and location_candidates[0][1] > 0.7 else location_col
        percent_col = percent_candidates[0][0] if percent_candidates and percent_candidates[0][1] > 0.7 else None

        entity_preview["location_column"] = location_col
        entity_preview["percent_column"] = percent_col

        ballot_types_keywords = set(safe_lower(bt) for bt in BALLOT_TYPES)
        number_pattern = re.compile(r"^-?\d{1,3}(?:,\d{3})*(?:\.\d+)?%?$")
        for row in data:
            for h, v in safe_items(row):
                if not v:
                    continue
                if any(ck in safe_lower(h) for ck in CANDIDATE_KEYWORDS):
                    safe_add(entity_preview["candidates"], v)
                if any(bk in safe_lower(h) for bk in ballot_types_keywords):
                    safe_add(entity_preview["ballot_types"], h)
                if number_pattern.match(safe_replace(v, ",", "")):
                    safe_add(entity_preview["numbers"], v)
                if location_col and h == location_col:
                    safe_add(entity_preview["locations"], v)

        if not location_col or len(entity_preview["locations"]) == 0:
            logger.warning(f"[TABLE BUILDER][extract_table_data] No valid location column {location_col} or values detected {entity_preview['locations']}. Consider user/ML feedback.")

        percent_value = ""
        if percent_col:
            for row in data:
                val = safe_get(row, percent_col, "")
                if val and "%" in val:
                    percent_value = val
                    break
        if not percent_value:
            if context and "percent_reported" in context:
                percent_value = context["percent_reported"]
            else:
                for row in data:
                    for v in safe_values(row):
                        if isinstance(v, str) and "%" in v:
                            percent_value = v
                            break
                    if percent_value:
                        break
        if percent_col and percent_value:
            for row in data:
                if not safe_get(row, percent_col, []):
                    row[percent_col] = percent_value

        summary = (
            f"[bold green][NLP PREVIEW][/bold green] "
            f"Candidates: [cyan]{n_candidates}[/cyan], "
            f"Ballot Types: [magenta]{n_ballot_types}[/magenta], "
            f"Numbers: [yellow]{n_numbers}[/yellow], "
            f"Locations: [blue]{n_locations}[/blue], "
            f"LocCol: [bold]{loc_col_disp}[/bold], "
            f"PctCol: [bold]{pct_col_disp}[/bold]"
        )
        logger.alert(summary)
        if unique_locations:
            logger.info(f"[blue]Unique {loc_col_disp}s: {unique_locations}[/blue]")
        else:
            logger.warning(f"[yellow]No unique {loc_col_disp}s detected in data.[/yellow]")
        if unique_candidates:
            logger.info(f"[magenta]Unique Candidates: {unique_candidates}[/magenta]")
        else:
            logger.warning(f"[yellow]No unique candidates detected in data.[/yellow]")
        if data:
            preview_table = RichTable(show_header=True, header_style="bold magenta")
            for h in headers:
                preview_table.add_column(h)
            for row in data[:2]:
                preview_table.add_row(*(str(safe_get(row, h, "")) for h in headers))
            logger.alert(preview_table)
        if not headers and data:
            max_cols = max(len(row) for row in data)
            headers = [f"Column {i+1}" for i in range(max_cols)]
            logger.warning("[TABLE BUILDER][extract_table_data] No headers but there is data. Generating generic headers.")
            new_data = []
            for row in data:
                new_row = {}
                for idx, h in enumerate(headers):
                    vals = safe_values(row)
                    new_row[h] = vals[idx] if idx < len(vals) else ""
                new_data.append(new_row)
            data = new_data

        if not headers and not data:
            logger.warning("[TABLE BUILDER][extract_table_data] Empty table encountered.")

    except Exception as e:
        logger.error(f"[TABLE BUILDER][extract_table_data] Malformed HTML or extraction error: {e}")
        return [], [], {}

    logger.info(f"[TABLE BUILDER][extract_table_data] Finished: {len(data)} rows, {len(headers)} columns.")
    return headers, data, entity_preview

def guess_headers_from_row(row, known_keywords=None, context=None):
    """
    Attempts to guess headers from a row's children using keywords or context.
    Returns (headers, diagnostics)
    """
    diagnostics = {}
    if row is None:
        logger.warning("[TABLE BUILDER][guess_headers_from_row] Row is None, cannot guess headers.")
        diagnostics["error"] = "Row is None"
        return [], diagnostics

    # --- Build robust keyword set ---
    keyword_set = set()
    # 1. Always include constants
    keyword_set.update(CANDIDATE_KEYWORDS)
    keyword_set.update(PARTY_KEYWORDS)
    keyword_set.update(LOCATION_KEYWORDS)
    keyword_set.update(TOTAL_KEYWORDS)
    # 2. Add from context if present
    if context:
        # Add any explicit header keywords
        if "header_keywords" in context and isinstance(context["header_keywords"], (list, set)):
            keyword_set.update(context["header_keywords"])
        # Add contest/field names if present
        for k in ("contest", "field_names", "expected_headers"):
            if k in context and isinstance(context[k], (list, set)):
                keyword_set.update(context[k])
            elif k in context and isinstance(context[k], str):
                keyword_set.add(context[k])
    # 3. Add any provided known_keywords
    if known_keywords:
        keyword_set.update(known_keywords)
    # 4. Normalize all keywords (lowercase, strip)
    normalized_keywords = set(str(kw).strip().lower() for kw in keyword_set if kw)

    # --- Extract cell texts ---
    cells = safe_locator(row, "> *", logger)
    headers = []
    cell_texts = []
    for i in range(safe_count(cells, logger)):
        cell = safe_nth(cells, i, logger)
        text = safe_inner_text(cell, logger)
        text_stripped = text.strip() if isinstance(text, str) else ""
        text_lower = text_stripped.lower()
        cell_texts.append(text_lower)
        header = None
        # Try to match any normalized keyword as substring or exact
        for kw in normalized_keywords:
            if kw and (kw == text_lower or kw in text_lower):
                header = kw.capitalize()
                break
        if not header:
            header = f"Column {i+1}"
        headers.append(header)

    diagnostics["cell_texts"] = cell_texts
    diagnostics["headers"] = headers
    diagnostics["used_keywords"] = sorted(normalized_keywords)
    diagnostics["context_used"] = bool(context)
    return headers, diagnostics

def extract_rows_and_headers_from_dom(page, extra_keywords=None, min_row_count=2, coordinator=None, context=None):
    """
    Attempts to extract tabular data from repeated DOM structures (divs, etc.).
    Returns headers, data, and diagnostics.
    Enhanced: logs and returns what is being removed, and column stats.
    Utilizes heading context and diagnostics from header guessing.
    """
    if coordinator is None:
        from ..Context_Integration.context_coordinator import ContextCoordinator
        coordinator = coordinator or ContextCoordinator()
    logger.info("[TABLE_BUILDER][extract_rows_and_headers_from_dom] Starting DOM structure extraction.")
    repeated_rows = extract_repeated_dom_structures(page, extra_keywords=extra_keywords, min_row_count=min_row_count)
    logger.info(f"[TABLE_BUILDER][extract_rows_and_headers_from_dom] Found {len(repeated_rows)} repeated rows.")
    if not repeated_rows:
        logger.warning("[TABLE_BUILDER][extract_rows_and_headers_from_dom] No repeated rows found.")
        return [], [], {"diagnostics": "No repeated rows found."}

    # --- Heuristic header detection block ---
    headers = None
    header_row_idx = None
    header_diag = None
    for idx, (heading, row) in enumerate(repeated_rows[:10]):
        if row is None:
            logger.warning(f"[TABLE_BUILDER][extract_rows_and_headers_from_dom] Row locator is None at index {idx}. Skipping.")
            continue
        cells = safe_locator(row, "> *", logger)
        cell_texts = []
        for i in range(safe_count(cells, logger)):
            cell = safe_nth(cells, i, logger)
            text = safe_inner_text(cell, logger)
            cell_texts.append(text.strip() if isinstance(text, str) else "")
        logger.info(f"[TABLE_BUILDER][extract_rows_and_headers_from_dom] Checking row {idx} for headers: {cell_texts} (heading: {heading})")
        # Heuristic: header row if at least 2 known fields or all non-numeric
        if is_likely_header(cell_texts) or all(not re.match(r"^\d+([,.]\d+)?$", c) for c in cell_texts):
            headers = cell_texts
            header_row_idx = idx
            header_diag = {"detected_by": "heuristic", "row_idx": idx, "heading": heading, "cell_texts": cell_texts}
            logger.info(f"[TABLE_BUILDER][extract_rows_and_headers_from_dom] Detected header row at index {idx}: {headers} (heading: {heading})")
            break
    if headers is not None:
        repeated_rows = repeated_rows[header_row_idx + 1 :]
    else:
        # Use coordinator logic if available for header guessing
        if coordinator and hasattr(coordinator, "score_header"):
            guessed_headers, diag = guess_headers_from_row(repeated_rows[0][1], context=context or {})
            header_diag = diag
            # Optionally, score and reorder headers by ML score
            header_scores = [(h, coordinator.score_header(h, context or {})) for h in guessed_headers]
            header_scores.sort(key=lambda x: x[1], reverse=True)
            headers = [h for h, _ in header_scores]
        else:
            headers, diag = guess_headers_from_row(repeated_rows[0][1], context=context or {})
            header_diag = diag
        logger.info(f"[TABLE_BUILDER][extract_rows_and_headers_from_dom] Guessed headers from first row: {headers} (heading: {repeated_rows[0][0]})")
        header_diag = header_diag or {}

    # --- Merge split header rows (e.g., two header rows) ---
    if len(repeated_rows) > 1:
        first_row_heading, first_row = repeated_rows[0]
        cells = safe_locator(first_row, "> *", logger)
        first_row_cells = []
        for i in range(safe_count(cells, logger)):
            cell = safe_nth(cells, i, logger)
            text = safe_inner_text(cell, logger)
            first_row_cells.append(text.strip() if isinstance(text, str) else "")
        if all(safe_isalpha(c) or c == "" for c in first_row_cells) and any(c for c in first_row_cells):
            logger.info(f"[TABLE_BUILDER][extract_rows_and_headers_from_dom] Merging split header rows: {headers} + {first_row_cells} (heading: {first_row_heading})")
            headers = [" ".join(filter(None, [h, f])) for h, f in zip(headers, first_row_cells)]
            repeated_rows = repeated_rows[1:]

    # --- Sample rows for stats ---
    sample_rows = []
    sample_headings = []
    for heading, row in repeated_rows[:20]:
        cells = safe_locator(row, "> *", logger)
        cell_texts = []
        for i in range(safe_count(cells, logger)):
            cell = safe_nth(cells, i, logger)
            text = safe_inner_text(cell, logger)
            cell_texts.append(text.strip() if isinstance(text, str) else "")
        sample_rows.append(cell_texts)
        sample_headings.append(heading)
    logger.info(f"[TABLE_BUILDER][extract_rows_and_headers_from_dom] Sample rows for stats: {sample_rows[:3]} (headings: {sample_headings[:3]})")

    # --- Column stats ---
    col_stats = []
    for col in range(len(headers)):
        values = [r[col] for r in sample_rows if len(r) > col]
        num_numeric = sum(1 for v in values if re.match(r"^\d+([,.]\d+)?$", v))
        num_empty = sum(1 for v in values if not v)
        unique_vals = len(set(values))
        col_stats.append({
            "numeric_ratio": num_numeric / len(values) if values else 0,
            "empty_ratio": num_empty / len(values) if values else 1,
            "unique_vals": unique_vals,
            "values": values,
        })
    logger.info(f"[TABLE_BUILDER][extract_rows_and_headers_from_dom] Column stats: {col_stats}")

    # --- Build all data rows before filtering ---
    all_panel_rows = []
    all_panel_headings = []
    for heading, row in repeated_rows:
        cells = safe_locator(row, "> *", logger)
        cell_values = []
        for i in range(len(headers)):
            cell = safe_nth(cells, i, logger)
            text = safe_inner_text(cell, logger)
            cell_values.append(text.strip() if isinstance(text, str) else "")
        row_data = {headers[idx]: cell_values[idx] if idx < len(cell_values) else "" for idx in range(len(headers))}
        # Attach heading context to each row for downstream use
        row_data["_heading"] = heading
        all_panel_rows.append(row_data)
        all_panel_headings.append(heading)

    # --- Remove footer/summary rows, log what is removed ---
    filtered_data = []
    removed_footer_rows = []
    for row in all_panel_rows:
        if row not in remove_footer_and_summary_rows([row], headers):
            removed_footer_rows.append(row)
        else:
            filtered_data.append(row)
    logger.info(f"[TABLE_BUILDER][extract_rows_and_headers_from_dom] Removed {len(removed_footer_rows)} footer/summary rows.")

    # --- Remove outlier/empty rows, log what is removed ---
    final_data = []
    removed_empty_rows = []
    for row in filtered_data:
        if row not in remove_outlier_and_empty_rows([row]):
            removed_empty_rows.append(row)
        else:
            final_data.append(row)
    logger.info(f"[TABLE_BUILDER][extract_rows_and_headers_from_dom] Removed {len(removed_empty_rows)} empty/outlier rows.")

    # --- Diagnostics dictionary ---
    diagnostics = {
        "headers": headers,
        "header_diag": header_diag,
        "all_panel_rows": all_panel_rows,
        "all_panel_headings": all_panel_headings,
        "removed_footer_rows": removed_footer_rows,
        "removed_empty_rows": removed_empty_rows,
        "col_stats": col_stats,
        "sample_rows": sample_rows,
        "sample_headings": sample_headings,
        "final_row_count": len(final_data),
        "final_col_count": len(headers),
    }

    logger.info(f"[TABLE_BUILDER][extract_rows_and_headers_from_dom] Finished: {len(final_data)} rows, {len(headers)} columns.")
    return headers, final_data, diagnostics

def extract_with_patterns(page, context=None, log_path=None):
    """
    Attempts to extract tabular data using approved DOM patterns.
    Returns (headers, data, diagnostics)
    Safeguards all DOM and list operations.
    """
    # Use lambda x: [] as a fallback for any safe_get in this function
    fallback_empty = lambda x: []

    patterns = safe_get(globals(), "load_dom_patterns", fallback_empty)(log_path)
    approved = [p for p in patterns if safe_get(p, "approved", fallback_empty)]
    results = []
    diagnostics = {
        "patterns_tried": len(patterns),
        "patterns_approved": len(approved),
        "matches": [],
    }
    for pat in approved:
        selector = safe_get(pat, "selector", "")
        # Use fallback_empty for all safe_gets that expect a list
        cell_selectors = safe_get(pat, "cell_selectors", fallback_empty) or [safe_get(pat, "cell_selector", "> *")]
        try:
            containers = safe_locator(page, selector, logger)
            container_count = safe_count(containers, logger)
        except Exception:
            continue
        for i in range(container_count):
            container = safe_nth(containers, i, logger)
            if container is None:
                continue
            heading = safe_get(pat, "heading", None) or f"Pattern: {selector} #{i+1}"
            for cell_selector in cell_selectors:
                try:
                    children = safe_locator(container, cell_selector, logger)
                    children_count = safe_count(children, logger)
                except Exception:
                    continue
                if children_count > 0:
                    for j in range(children_count):
                        row = safe_nth(children, j, logger)
                        if row is None:
                            continue
                        try:
                            if "row_tag" in pat:
                                tag = row.evaluate("el => el.tagName.toLowerCase()") if hasattr(row, "evaluate") else ""
                                if tag != safe_get(pat, "row_tag", ""):
                                    continue
                            if "row_class" in pat:
                                classes = row.evaluate("el => el.className") if hasattr(row, "evaluate") else ""
                                if safe_get(pat, "row_class", "") not in classes:
                                    continue
                            if "row_text_contains" in pat:
                                text = safe_inner_text(row, logger).strip()
                                if safe_get(pat, "row_text_contains", "") not in text:
                                    continue
                            results.append((heading, row, pat))
                            safe_append(
                                diagnostics["matches"],
                                {
                                    "heading": heading,
                                    "selector": selector,
                                    "cell_selector": cell_selector,
                                    "row_index": j
                                },
                                logger
                            )
                        except Exception:
                            continue
    # Build headers/data if any matches
    if results:
        # Use fallback_empty for guess_headers_from_row context
        headers, _ = guess_headers_from_row(results[0][1], context=context or fallback_empty)
        data = []
        for heading, row, pat in results:
            try:
                cells = safe_locator(row, "> *", logger)
                cell_count = safe_count(cells, logger)
                row_data = {}
                for idx in range(cell_count):
                    cell = safe_nth(cells, idx, logger)
                    if cell is not None:
                        try:
                            cell_text = safe_inner_text(cell, logger).strip()
                        except Exception:
                            cell_text = ""
                        key = headers[idx] if idx < len(headers) else f"Column {idx+1}"
                        row_data[key] = cell_text
                if row_data:
                    data.append(row_data)
            except Exception:
                continue
        return headers, data, diagnostics
    return [], [], diagnostics

def fallback_nlp_candidate_vote_scan(page):
    """
    Improved fallback: scan for elements with candidate-like, party-like, or location-like names and vote-like numbers nearby.
    Returns headers, data.
    Enhanced: extra input sanitization, more robust Playwright API usage, and stricter skip phrase/label filtering.
    """
    # Accept more flexible candidate/location/party patterns
    label_pattern = re.compile(r"^[A-Za-z][A-Za-z\s\-\']{1,40}$")
    vote_pattern = re.compile(r"^\d{1,3}(,\d{3})*$")
    # Use skip phrases from constants if available, else fallback
    skip_phrases = NLP_SKIP_PHRASES if 'NLP_SKIP_PHRASES' in globals() else [
        "Last Updated", "Vote Method", "Fully Reported", "Search", "Reported", "Total", "Precincts Reporting"
    ]

    elements = safe_locator(page, "*", logger)
    labels = []
    votes = []
    element_count = safe_count(elements, logger)
    for i in range(element_count):
        text = ""
        try:
            el = safe_nth(elements, i, logger)
            text = safe_inner_text(el, logger).strip() if el else ""
            # --- Extra sanitization: remove control chars, excessive whitespace, and dangerous chars ---
            text = re.sub(r"[\x00-\x1F\x7F]", "", text)
            text = text.replace("\r", " ").replace("\n", " ").strip()
            text = re.sub(r"\s+", " ", text)
            # Remove SQL meta-characters (defense-in-depth, even though not used in SQL here)
            text = text.replace(";", "").replace("--", "").replace("'", "").replace('"', "")
        except Exception:
            continue
        if not text or len(text) < 2:
            continue
        # Stricter skip: match whole word or phrase, case-insensitive
        if any(skip.lower() in text.lower() for skip in skip_phrases):
            continue
        # Only allow ASCII printable for fallback
        if not all(32 <= ord(c) < 127 for c in text):
            continue
        # Robust vote and label detection
        if vote_pattern.fullmatch(text.replace(",", "")):
            votes.append((i, text))
        elif label_pattern.match(text):
            # Avoid labels that look like SQL keywords or dangerous input
            if text.upper() in {"SELECT", "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE"}:
                continue
            labels.append((i, text))
    # Pair each vote with the closest preceding label
    data = []
    used_label_idxs = set()
    for vote_idx, vote_val in votes:
        # Find the closest label before this vote
        label = None
        for idx, lbl in reversed(labels):
            if idx < vote_idx and idx not in used_label_idxs:
                label = lbl
                used_label_idxs.add(idx)
                break
        if label is not None:
            # Extra: sanitize output fields
            safe_label = re.sub(r"[^\w\s\-']", "", label).strip()
            safe_vote = re.sub(r"[^\d,]", "", vote_val)
            data.append({"Label": safe_label, "Votes": safe_vote})
    headers = ["Label", "Votes"]
    logger.info(f"[TABLE BUILDER] Robust NLP fallback: {len(data)} rows, {len(headers)} columns.")
    return headers, data

def extract_repeated_dom_structures(page, container_selectors=None, min_row_count=2, extra_keywords=None):
    """
    Scans the DOM for repeated structures (divs, uls, etc.) that look like tabular data.
    Returns a list of (section_heading, row_locator) tuples.
    Dynamically updates likely_row_classes from log analysis.
    Enhanced: input sanitization for selectors, robust Playwright API usage, and defense-in-depth for selector injection.
    """
    log_dir = LOG_DIR
    suggested_classes, suggested_ids = suggest_new_row_classes_from_logs(log_dir)
    likely_row_classes = list(LIKELY_ROW_CLASSES) + suggested_classes if 'LIKELY_ROW_CLASSES' in globals() else [
        "row", "table-row", "ballot-option", "candidate-info", "result-row", "precinct-row"
    ] + suggested_classes
    likely_row_ids = suggested_ids

    # --- Sanitize selectors to prevent selector injection ---
    def sanitize_selector(s):
        # Only allow alphanum, dash, underscore, and no quotes/brackets
        return re.sub(r"[^a-zA-Z0-9_\-]", "", s)

    if container_selectors is None:
        selectors = [f"div.{sanitize_selector(cls)}" for cls in likely_row_classes if cls]
        selectors += [f"div#{sanitize_selector(id_)}" for id_ in likely_row_ids if id_]
        selectors += ["ul > li", "ol > li"]
    else:
        selectors = [sanitize_selector(sel) for sel in container_selectors if sel]

    results = []
    MAX_CONTAINERS = 100
    for selector in selectors:
        try:
            containers = safe_locator(page, selector, logger)
            container_count = safe_count(containers, logger)
        except Exception:
            continue
        for i in range(min(container_count, MAX_CONTAINERS)):
            try:
                container = safe_nth(containers, i, logger)
                children = safe_locator(container, "> *", logger) if container else None
                children_count = safe_count(children, logger) if children else 0
                if children_count >= min_row_count:
                    # Try to find a heading above the container
                    heading = ""
                    heading_loc = safe_locator(container, "xpath=preceding-sibling::*[self::h1 or self::h2 or self::h3 or self::h4 or self::h5 or self::h6][1]", logger) if container else None
                    heading_count = safe_count(heading_loc, logger) if heading_loc else 0
                    if heading_count > 0:
                        heading_el = safe_nth(heading_loc, 0, logger)
                        heading = safe_inner_text(heading_el, logger).strip() if heading_el else ""
                        # Extra sanitization for heading
                        heading = re.sub(r"[\x00-\x1F\x7F]", "", heading)
                        heading = heading.replace("\r", " ").replace("\n", " ").strip()
                        heading = re.sub(r"\s+", " ", heading)
                    else:
                        heading = f"Section {i+1}"
                    for j in range(children_count):
                        row = safe_nth(children, j, logger)
                        if row is not None:
                            results.append((heading, row))
            except Exception as e:
                log_failed_container(page, container, selector, i, str(e))
    return results

def extract_all_candidates_from_data(headers, data, extraction_context=None):
    """
    Extract all unique candidate names from the data, using the provided headers and context.
    Optionally uses extraction_context for more robust candidate column detection.
    Safeguards .lower, .split, .strip, .startswith, and .get usage.
    """
    candidates = set()
    # Try to find the candidate column robustly
    candidate_col = None
    # 1. Use context if available
    if extraction_context and isinstance(extraction_context, dict) and "candidate_column" in extraction_context:
        candidate_col = extraction_context.get("candidate_column")
    # 2. Fallback: look for best header match
    if not candidate_col:
        for h in headers:
            h_safe = h.lower() if isinstance(h, str) else str(h).lower()
            if any(ck in h_safe for ck in CANDIDATE_KEYWORDS):
                candidate_col = h
                break
    # 3. Fallback: use "Candidate" if present
    if not candidate_col and any((isinstance(h, str) and h == "Candidate") for h in headers):
        candidate_col = "Candidate"
    # 4. If still not found, skip extraction
    if not candidate_col:
        logger.warning("[extract_all_candidates_from_data] No candidate column found in headers or context.")
        return candidates

    for row in data:
        val = row.get(candidate_col, "") if isinstance(row, dict) else ""
        # Defensive: ensure val is a string for split
        if not isinstance(val, str):
            val = str(val)
        for part in val.split("\n"):
            part_safe = part.strip() if isinstance(part, str) else str(part).strip()
            # Filter out party-only or generic lines
            if part_safe and not any(
                part_safe.lower().startswith(pk.lower() if isinstance(pk, str) else str(pk).lower())
                for pk in PARTY_KEYWORDS
            ):
                candidates.add(part_safe)
    return candidates

# 1. ML-based table detection (e.g., using a model to find tables in arbitrary HTML)
def ml_based_table_detection(page, extraction_context=None):
    """
    Use a machine learning model to detect and extract tables from arbitrary HTML.
    Returns a list of (headers, data, diagnostics) tuples.
    Each diagnostics dict includes the extraction_context for traceability.
    Enhanced: uses safe_content, robust error handling, and logs more diagnostics.
    """
    from ..Context_Integration.context_coordinator import ContextCoordinator
    coordinator = ContextCoordinator()
    try:
        html = safe_content(page)
        if not html or not isinstance(html, str) or len(html) < 100:
            logger.error("[ML TABLE DETECTION] Empty or invalid HTML content for ML table detection.")
            return []
        ml_tables = detect_tables_ml(html)
        results = []
        for idx, table_dict in enumerate(ml_tables):
            headers = table_dict.get("headers", []) if isinstance(table_dict, dict) else []
            data = table_dict.get("data", []) if isinstance(table_dict, dict) else []
            # Optionally, correlate context to this table (if available)
            context = extraction_context if extraction_context else {}
            if coordinator and hasattr(coordinator, "get_for_table_builder"):
                try:
                    context = coordinator.get_for_table_builder()
                except Exception as e:
                    logger.warning(f"[ML TABLE DETECTION] Could not get context from coordinator: {e}")
            diagnostics = {
                "ml_table_index": idx,
                "row_count": len(data),
                "headers": headers,
                "extraction_context": context,
                "source": "ml_based_table_detection"
            }
            # Optionally, attach context to each row for downstream traceability
            if headers and data and isinstance(headers, list) and isinstance(data, list):
                # Attach context and diagnostics to each row for traceability and debugging
                for row in data:
                    # Only attach if row is a dict (defensive)
                    if isinstance(row, dict):
                        # Attach extraction context (if not already present)
                        if "_extraction_context" not in row:
                            row["_extraction_context"] = context
                        # Attach ML diagnostics index for traceability
                        row["_ml_table_index"] = idx
                        # Optionally, attach header signature for deduplication/debug
                        row["_header_signature"] = tuple(headers)
                        # Optionally, attach row source
                        row["_row_source"] = "ml_based_table_detection"
                results.append((headers, data, diagnostics))
            else:
                logger.warning(f"[ML TABLE DETECTION] Skipping table {idx}: invalid headers or data.")
        if not results:
            logger.warning("[ML TABLE DETECTION] No tables detected by ML model.")
        return results
    except Exception as e:
        logger.error(f"[ML TABLE DETECTION] Error: {e}")
        return []
    
# 2. Nested table extraction (see handle_nested_tables)
def nested_table_extraction(page):
    """
    Extract tables that are nested within other tables or complex DOM structures.
    Returns a list of (headers, data, diagnostics) tuples.
    """
    try:
        results = []
        tables = safe_locator(page, "table table", logger)
        table_count = safe_count(tables, logger)
        for i in range(table_count):
            table = safe_nth(tables, i, logger)
            if table is not None:
                headers, data, diagnostics = extract_table_data(table)
                diagnostics = diagnostics or {}
                diagnostics["nested_table_index"] = i
                if headers and data:
                    results.append((headers, data, diagnostics))
        return results
    except Exception as e:
        logger.error(f"[NESTED TABLE EXTRACTION] Error: {e}")
        return []

# 3. Robust HTML fallback using selectolax (see robust_html_fallback)
def robust_html_fallback_extraction(page):
    """
    Use selectolax to parse HTML and extract tables as a last-resort fallback.
    Returns a list of (headers, data, diagnostics) tuples.
    Enhanced: uses LOCATION_KEYWORDS and constants for header normalization and attaches extraction context.
    """
    try:
        # Use safe_content from browser_utils for robust HTML extraction
        html = safe_content(page)
        if not html or not isinstance(html, str):
            logger.error("[HTML FALLBACK] No HTML content available for fallback extraction.")
            return []
        html_tree = HTMLParser(html)
        tables = html_tree.css("table")
        all_tables = []
        for idx, table in enumerate(tables):
            rows = table.css("tr")
            if not rows:
                continue
            # Extract headers, normalize using LOCATION_KEYWORDS and constants
            header_cells = rows[0].css("th") or rows[0].css("td")
            headers = [cell.text(strip=True) for cell in header_cells]
            # Normalize location headers if possible
            for i, h in enumerate(headers):
                h_norm = h.strip().lower()
                for loc_kw in LOCATION_KEYWORDS:
                    if loc_kw in h_norm and h != "Precinct":
                        headers[i] = "Precinct"
            data = []
            for row in rows[1:]:
                cells = row.css("td") or row.css("th")
                row_dict = {headers[i]: cells[i].text(strip=True) if i < len(cells) else "" for i in range(len(headers))}
                # Attach fallback context for traceability
                row_dict["_extraction_context"] = {"source": "robust_html_fallback", "table_index": idx}
                data.append(row_dict)
            diagnostics = {
                "fallback_table_index": idx,
                "row_count": len(data),
                "headers": headers
            }
            if headers and data:
                all_tables.append((headers, data, diagnostics))
        return all_tables
    except Exception as e:
        logger.error(f"[HTML FALLBACK] Error: {e}")
        return []

# 4. Custom per-county or per-state extraction strategies (plug-in architecture)
def custom_plugin_extraction(page, extraction_context=None):
    """
    Use custom extraction plugins based on county/state or other context.
    Returns a list of (headers, data, diagnostics) tuples.
    """
    try:
        plugins = safe_get(extraction_context, "plugins", []) if extraction_context else []
        results = []
        for idx, plugin in enumerate(plugins):
            plugin_result = safe_extract(plugin, page, extraction_context)
            if plugin_result:
                for headers, data in plugin_result:
                    diagnostics = {
                        "plugin_index": idx,
                        "plugin_name": getattr(plugin, "__name__", str(plugin)),
                        "row_count": len(data)
                    }
                    if headers and data:
                        results.append((headers, data, diagnostics))
        return results
    except Exception as e:
        logger.error(f"[PLUGIN EXTRACTION] Error: {e}")
        return []

# 5. Feedback/correction loop for user-in-the-loop extraction
def feedback_correction_loop(headers, data, extraction_context=None):
    """
    Allow user or operator to review and correct extracted table data.
    Returns possibly corrected (headers, data).
    """
    try:
        if extraction_context and safe_get(extraction_context, "interactive", []):
            logger.info("\n[FEEDBACK] Review extracted headers and data:")
            logger.info("Headers:", headers)
            for i, row in enumerate(data[:5]):
                logger.info(f"Row {i+1}:", row)
            resp = input("Are the headers and data correct? (y/n): ").strip().lower()
            if resp == "n":
                new_headers = input("Enter corrected headers as comma-separated values: ").strip().split(",")
                headers = [h.strip() for h in new_headers]
                # Optionally, allow editing data as well
                # For brevity, only headers are corrected here
        return headers, data
    except Exception as e:
        logger.error(f"[FEEDBACK LOOP] Error: {e}")
        return headers, data

# --- CLIENT-SIDE UNVALIDATED URL REDIRECTION MITIGATION ---
def safe_redirect_url(user_url, allowed_domains=None):
    """
    Prevent unvalidated redirects by checking user-supplied URLs against a whitelist.
    """
    if allowed_domains is None:
        allowed_domains = {"yourdomain.com"}
    try:
        parsed = urlparse(user_url)
        if safe_scheme(parsed) not in {"http", "https"}:
            return "/"
        if safe_netloc(parsed) and safe_netloc(parsed) not in allowed_domains:
            return "/"
        # Optionally, further sanitize the path
        return safe_geturl(parsed)
    except Exception:
        return "/"

def find_best_header(headers, keywords):
    """Find the best matching header from a set of keywords (case-insensitive, fuzzy)."""
    headers_lower = [safe_lower(h) for h in headers]
    # Try substring match for any keyword
    for kw in keywords:
        kw_lower = safe_lower(kw)
        for i, h in enumerate(headers_lower):
            if kw_lower in h:
                return headers[i]
    # Fuzzy match if no substring match
    for kw in keywords:
        kw_lower = safe_lower(kw)
        matches = get_close_matches(kw_lower, headers_lower, n=1, cutoff=0.7)
        if matches:
            return headers[headers_lower.index(matches[0])]
    return None
    
# ===================================================================
# HARMONIZATION & CLEANING
# ===================================================================

def harmonize_headers_and_data(headers: list, data: list, context: dict = None) -> tuple:
    """
    Ensures all rows have the same headers, filling missing fields with empty string.
    Deduplicates rows using a composite key of Location, Candidate, and Ballot Type columns.
    Always includes 'Percent Reported' if present in any row or context.
    Logs unique values in the location column.
    """
    # 1. Collect all headers from input and data rows
    all_headers = set(h for h in headers if h)
    for row in data:
        all_headers.update(safe_keys(row))

    # 2. Ensure 'Percent Reported' is present if in any row or context
    percent_val = None
    if any("Percent Reported" in safe_keys(row) for row in data):
        all_headers.add("Percent Reported")
        for row in data:
            if safe_get(row, "Percent Reported", []):
                percent_val = safe_get(row, "Percent Reported", "")
                break
    if context and safe_get(context, "percent_reported", []):
        all_headers.add("Percent Reported")
        percent_val = safe_get(context, "percent_reported", "")

    # 3. Build ordered headers: preserve input order, then add new ones
    seen = set()
    ordered_headers = [h for h in headers if h in all_headers and not (h in seen or seen.add(h))]
    ordered_headers += [h for h in all_headers if h not in seen and not seen.add(h)]

    # 4. Normalize location column to "Precinct"
    location_col = next((h for h in ordered_headers if is_location_header(h)), None)
    if location_col and location_col != "Precinct":
        ordered_headers = ["Precinct" if h == location_col else h for h in ordered_headers]
        for row in data:
            row["Precinct"] = safe_pop(row, location_col, "")
        location_col = "Precinct"

    # 5. Identify candidate and ballot type columns
    candidate_col = next((h for h in ordered_headers if any(ck in safe_lower(h) for ck in CANDIDATE_KEYWORDS)), None)
    ballot_types_cols = [h for h in ordered_headers if any(bt in safe_lower(h) for bt in BALLOT_TYPES)]

    # 6. Deduplicate rows using composite key (location, candidate, ballot types)
    harmonized = []
    seen_keys = set()
    for row in data:
        full_row = {h: safe_get(row, h, "") for h in ordered_headers}
        # Fill missing Percent Reported from context if needed
        if "Percent Reported" in ordered_headers and not safe_get(full_row, "Percent Reported", []) and percent_val:
            full_row["Percent Reported"] = percent_val
        # Deduplication key
        if location_col and candidate_col and safe_get(full_row, location_col, []) and safe_get(full_row, candidate_col, []):
            key = (
                safe_get(full_row, location_col, ""),
                safe_get(full_row, candidate_col, ""),
                *(safe_get(full_row, bt, "") for bt in ballot_types_cols)
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
        harmonized.append(full_row)

    # 7. Remove columns that are all empty or zero, but always keep columns present in input headers
    keep = [h for h in ordered_headers if (h in headers) or any(safe_get(row, h, "") not in ("", "0") for row in harmonized)]
    if not keep and ordered_headers:
        keep = ordered_headers
    harmonized = [{h: safe_get(row, h, "") for h in keep} for row in harmonized]

    # 8. Log unique locations and warn if only one unique value
    unique_locations = set(safe_get(row, location_col, "") for row in harmonized if location_col and location_col in row)
    logger.info(f"[HARMONIZE] Unique values in location column '{location_col}': {sorted(unique_locations)}")
    logger.info(f"[HARMONIZE] Unique values in location column '{location_col}': {sorted(unique_locations)}")
    if location_col and len(unique_locations) <= 1:
        logger.warning(f"[HARMONIZE] WARNING: Only one unique value found in location column '{location_col}'. Extraction may be incorrect.")

    # 9. Reorder columns: Precinct, candidates, ballot types, then others
    candidate_cols = [h for h in keep if any(k in safe_lower(h) for k in CANDIDATE_KEYWORDS)]
    ballot_types_cols = [h for h in keep if any(bt in safe_lower(h) for bt in BALLOT_TYPES)]
    ordered_final = []
    if "Precinct" in keep:
        ordered_final.append("Precinct")
    ordered_final += sorted(set(candidate_cols + ballot_types_cols))
    ordered_final += [h for h in keep if h not in candidate_cols + ballot_types_cols + ["Precinct"]]
    # Remove duplicates while preserving order
    seen_final = set()
    ordered_final = [h for h in ordered_final if not (h in seen_final or seen_final.add(h))]

    # 10. Return final headers and harmonized data
    return ordered_final, [{h: safe_get(row, h, "") for h in ordered_final} for row in harmonized]

def deduplicate_headers(headers, data) -> tuple[list[str], list[dict]]:
    """Remove duplicate headers by normalized name, keep first occurrence."""
    seen = set()
    new_headers = []
    for h in headers:
        norm = normalize_header(h)
        if norm not in seen:
            new_headers.append(h)
            seen.add(norm)
    new_data = [{h: safe_get(row, h, "") for h in new_headers} for row in data]
    return new_headers, new_data

def remove_low_signal_columns(headers, data, min_unique=2, min_non_empty_ratio=0.05) -> tuple[list[str], list[dict]]:
    """
    Remove columns with low variance or too many repeated values.
    """
    keep = []
    n_rows = len(data)
    for h in headers:
        col_vals = [safe_get(row, h, "") for row in data]
        unique_vals = set(col_vals)
        non_empty = [v for v in col_vals if v not in ("", None)]
        if len(unique_vals) >= min_unique and len(non_empty) / n_rows >= min_non_empty_ratio:
            keep.append(h)
    return keep, [{h: safe_get(row, h, "") for h in keep} for row in data]

def merge_table_data(headers_list, data_list) -> tuple[list[str], list[dict]]:
    """
    Merge multiple (headers, data) pairs into a single (headers, data).
    Later data fills in missing values from earlier data.
    """
    all_headers = []
    for headers in headers_list:
        for h in headers:
            if h not in all_headers:
                all_headers.append(h)
    merged_data = []
    for data in data_list:
        for row in data:
            match = None
            for mrow in merged_data:
                keys = ["Precinct", "Candidate", "Party"]
                if all(
                    k in row and k in mrow and safe_get(row, k, "") == safe_get(mrow, k, "") and safe_get(row, k, "")
                    for k in keys if k in row
                ):
                    match = mrow
                    break
            if match:
                for h in all_headers:
                    if not safe_get(match, h, []) and safe_get(row, h, []):
                        match[h] = safe_get(row, h, "")
            else:
                merged_data.append(row)
    # Only harmonize once at the end
    return harmonize_headers_and_data(all_headers, merged_data)

def merge_multiline_candidate_rows(headers, data) -> tuple[list[str], list[dict]]:
    """
    Merge rows where candidate name and party are split across two rows or within a cell.
    Ensures 'Precinct' and 'Percent Reported' columns are preserved and consistent.
    - If a candidate cell contains a newline, split into candidate and party.
    - If the next row is just a party, merge it.
    - Always ensure 'Precinct' and 'Percent Reported' columns are present if found in any row.
    """
    # --- Detect if we have these columns in headers or data ---
    has_precinct = "Precinct" in headers or any("Precinct" in safe_keys(row) for row in data)
    has_percent_reported = "Percent Reported" in headers or any("Percent Reported" in safe_keys(row) for row in data)
    # If not in headers but present in data, add to headers
    if not has_precinct and any("Precinct" in safe_keys(row) for row in data):
        safe_append(headers, "Precinct")
        has_precinct = True
    if not has_percent_reported and any("Percent Reported" in safe_keys(row) for row in data):
        safe_append(headers, "Percent Reported")
        has_percent_reported = True

    # --- Main merge logic ---
    if "Candidate" not in headers:
        return headers, data
    merged_data = []
    i = 0
    while i < len(data):
        row = data[i]
        candidate_val = safe_get(row, "Candidate", "")
        # Preserve Precinct and Percent Reported if present
        precinct_val = safe_get(row, "Precinct", "")
        percent_reported_val = safe_get(row, "Percent Reported", "")
        party_abbrevs = ["DEM", "REP", "CON", "WOR", "IND", "GRN", "LIB", "Other", "Write-in"]
        # Try to match pattern: [ABBR] Name[PartyName]
        match = re.match(r"^([A-Z]{2,4})\s+(.+?)([A-Z][a-z]+)$", candidate_val)
        if match:
            abbr, name, party = match.groups()
            row["Candidate"] = safe_strip(f"{abbr} {name}")
            row["Party"] = safe_strip(party)
            if has_precinct:
                row["Precinct"] = precinct_val
            if has_percent_reported:
                row["Percent Reported"] = percent_reported_val
            merged_data.append(row)
            i += 1
            continue
        # Try to split by known party abbreviations at start
        for abbr in party_abbrevs:
            if safe_startswith(candidate_val, abbr + " "):
                rest = safe_strip(candidate_val[len(abbr):])
                # Try to split at the last uppercase word (party)
                m = re.match(r"(.+?)([A-Z][a-z]+)$", rest)
                if m:
                    name, party = m.groups()
                    row["Candidate"] = safe_strip(f"{abbr} {name}")
                    row["Party"] = safe_strip(party)
                    if has_precinct:
                        row["Precinct"] = precinct_val
                    if has_percent_reported:
                        row["Percent Reported"] = percent_reported_val
                    merged_data.append(row)
                    i += 1
                    break
        else:
            # If candidate cell has a newline, split into candidate and party
            if "\n" in candidate_val:
                parts = [safe_strip(p) for p in safe_split(candidate_val, "\n") if safe_strip(p)]
                if len(parts) == 2:
                    row["Candidate"], row["Party"] = parts
                elif len(parts) > 2:
                    row["Candidate"] = parts[0]
                    row["Party"] = " ".join(parts[1:])
                else:
                    row["Candidate"] = safe_replace(candidate_val, "\n", " ")
                # Ensure columns are preserved
                if has_precinct:
                    row["Precinct"] = precinct_val
                if has_percent_reported:
                    row["Percent Reported"] = percent_reported_val
                merged_data.append(row)
                i += 1
            # If next row is just a party, merge it
            elif i + 1 < len(data):
                next_row = data[i + 1]
                next_candidate_val = safe_get(next_row, "Candidate", "")
                # Only merge if all other columns are empty in next row
                non_candidate_cols = [k for k in safe_keys(next_row) if k != "Candidate" and safe_get(next_row, k, "")]
                if next_candidate_val and not non_candidate_cols:
                    row["Party"] = safe_strip(next_candidate_val)
                    # Ensure columns are preserved
                    if has_precinct:
                        row["Precinct"] = precinct_val
                    if has_percent_reported:
                        row["Percent Reported"] = percent_reported_val
                    merged_data.append(row)
                    i += 2
                else:
                    if has_precinct:
                        row["Precinct"] = precinct_val
                    if has_percent_reported:
                        row["Percent Reported"] = percent_reported_val
                    merged_data.append(row)
                    i += 1
            else:
                if has_precinct:
                    row["Precinct"] = precinct_val
                if has_percent_reported:
                    row["Percent Reported"] = percent_reported_val
                merged_data.append(row)
                i += 1

    # Add "Party" to headers if not present and any row has it
    if any("Party" in safe_keys(row) for row in merged_data) and "Party" not in headers:
        safe_append(headers, "Party")
    # Ensure 'Precinct' and 'Percent Reported' in headers if present in any row
    if any("Precinct" in safe_keys(row) for row in merged_data) and "Precinct" not in headers:
        safe_append(headers, "Precinct")
    if any("Percent Reported" in safe_keys(row) for row in merged_data) and "Percent Reported" not in headers:
        safe_append(headers, "Percent Reported")
    return headers, merged_data

def combine_panel_tables_by_precinct(all_tables) -> tuple[list[str], list[dict]]:
    """
    Combine all panel tables into one, using 'Precinct' as the unique location column.
    """
    combined_headers = set()
    combined_data = []
    for headers, data in all_tables:
        combined_headers.update(headers)
        combined_data.extend(data)
    combined_headers = list(combined_headers)
    # Harmonize to ensure all rows have all headers
    combined_headers, combined_data = harmonize_headers_and_data(combined_headers, combined_data)
    return combined_headers, combined_data

# ===================================================================
# ENTITY ANNOTATION & STRUCTURE VERIFICATION
# ===================================================================

def nlp_entity_annotate_table(
    headers: List[str],
    data: List[Dict[str, Any]],
    context: dict = None,
    coordinator: "ContextCoordinator" = None
) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Annotate table with detected entities (people, locations, ballot types, numbers).
    Improved: Uses both 'Candidate' and 'Party' fields for entity extraction.
    Integrates segments and panels from context if available.
    """
    from ..Context_Integration.context_coordinator import ContextCoordinator
    coordinator = coordinator or ContextCoordinator()
    logger.info("[TABLE_CORE][nlp_entity_annotate_table] Starting NLP entity annotation.")
    if not coordinator:
        logger.warning("[TABLE_CORE][nlp_entity_annotate_table] No coordinator provided, skipping NLP annotation.")
        return headers, data, {}

    # ML context integration
    ml_confidence = safe_get(context, "ml_confidence", None)
    association_log = safe_get(context, "association_log", None)
    segments = safe_get(context, "segments", None)
    panels = safe_get(context, "panels", None)

    entity_info = {
        "people": set(),
        "locations": set(),
        "ballot_types": set(),
        "numbers": set(),
        "row_entities": [],
        "ml_confidence": ml_confidence,
        "association_log": association_log,
        "segments": segments,
        "panels": panels,
    }
    # Optionally: log ML context for debugging
    if ml_confidence is not None:
        logger.info(f"[TABLE_CORE][nlp_entity_annotate_table] ML confidence: {ml_confidence}")
    if association_log:
        logger.info(f"[TABLE_CORE][nlp_entity_annotate_table] Association log: {association_log}")
    if segments:
        logger.info(f"[TABLE_CORE][nlp_entity_annotate_table] Segments: {segments}")
    if panels:
        logger.info(f"[TABLE_CORE][nlp_entity_annotate_table] Panels: {panels}")

    # Analyze headers for entity types
    header_entities = {}
    for h in headers:
        ents = coordinator.extract_entities(h)
        header_entities[h] = ents
        for ent, label in ents:
            if label == "PERSON":
                safe_add(entity_info["people"], ent)
            elif label in {"GPE", "LOC", "FAC"}:
                safe_add(entity_info["locations"], ent)
            elif any(bt.lower() in safe_lower(h) for bt in BALLOT_TYPES):
                safe_add(entity_info["ballot_types"], h)
    # Analyze each row for entities
    annotated_data = []
    for row in data:
        row_ents = {"people": set(), "locations": set(), "ballot_types": set(), "numbers": set()}
        for field in ["Candidate", "Party"]:
            val = safe_get(row, field, "")
            if val:
                ents = coordinator.extract_entities(val)
                for ent, label in ents:
                    if label == "PERSON":
                        safe_add(row_ents["people"], ent)
                        safe_add(entity_info["people"], ent)
                    elif label in {"GPE", "LOC", "FAC"}:
                        safe_add(row_ents["locations"], ent)
                        safe_add(entity_info["locations"], ent)
        for h in headers:
            val = safe_get(row, h, "")
            if not val:
                continue
            ents = coordinator.extract_entities(val)
            for ent, label in ents:
                if label == "PERSON":
                    safe_add(row_ents["people"], ent)
                    safe_add(entity_info["people"], ent)
                elif label in {"GPE", "LOC", "FAC"}:
                    safe_add(row_ents["locations"], ent)
                    safe_add(entity_info["locations"], ent)
            # Ballot type detection
            for bt in BALLOT_TYPES:
                if bt.lower() in safe_lower(h) or bt.lower() in safe_lower(val):
                    safe_add(row_ents["ballot_types"], bt)
                    safe_add(entity_info["ballot_types"], bt)
            # Number detection
            if isinstance(val, str) and val.replace(",", "").replace(".", "").isdigit():
                safe_add(row_ents["numbers"], val)
                safe_add(entity_info["numbers"], val)
        safe_append(entity_info["row_entities"], row_ents)
        annotated_data.append(row)
    # Convert sets to sorted lists for JSON serializability
    for k in entity_info:
        if isinstance(entity_info[k], set):
            entity_info[k] = sorted(entity_info[k])
    logger.info(f"[TABLE_CORE][nlp_entity_annotate_table] Entity summary: {entity_info}")
    return headers, annotated_data, entity_info

def verify_table_structure(
    headers: List[str],
    data: List[Dict[str, Any]],
    entity_info: Dict[str, Any],
    coordinator: "ContextCoordinator",
    context: dict = None
) -> Tuple[bool, List[str]]:
    """
    Verifies that the table contains required columns/entities:
    - At least one location column
    - At least one candidate/person
    - At least one ballot type
    - At least one numeric column (votes/totals)
    Returns (verified: bool, missing: List[str])
    """
    from ..Context_Integration.context_coordinator import ContextCoordinator
    coordinator = coordinator or ContextCoordinator()
    logger.info("[TABLE_CORE][verify_table_structure] Verifying table structure using NLP and DOM info.")
    missing = []

    # Use context for additional robustness
    context = context or {}

    # Check for location
    has_location = (
        bool(safe_get(entity_info, "locations", []))
        or any(
            any(safe_lower(lk) in safe_lower(h) for lk in LOCATION_KEYWORDS)
            for h in headers
        )
        or bool(safe_get(context, "location_header", None))
        or bool(safe_get(context, "location_value", None))
    )
    if not has_location:
        missing.append("location")

    # Check for candidate/person
    has_candidate = (
        bool(safe_get(entity_info, "people", []))
        or any(
            coordinator and any(label == "PERSON" for ent, label in coordinator.extract_entities(h))
            for h in headers
        )
        or bool(safe_get(context, "candidate_header", None))
        or bool(safe_get(context, "candidate_value", None))
    )
    if not has_candidate:
        missing.append("candidate")

    # Check for ballot type
    has_ballot_type = (
        bool(safe_get(entity_info, "ballot_types", []))
        or any(
            any(safe_lower(bt) in safe_lower(h) for bt in BALLOT_TYPES)
            for h in headers
        )
        or bool(safe_get(context, "ballot_type_header", None))
        or bool(safe_get(context, "ballot_type_value", None))
    )
    if not has_ballot_type:
        missing.append("ballot_type")

    # Check for numbers
    has_numbers = (
        bool(safe_get(entity_info, "numbers", []))
        or any(
            any(safe_isdigit(str(c)) for c in safe_values(row))
            for row in data
        )
        or bool(safe_get(context, "number_header", None))
        or bool(safe_get(context, "number_value", None))
    )
    if not has_numbers:
        missing.append("numbers")

    verified = len(missing) == 0
    logger.info(f"[TABLE_CORE][verify_table_structure] Verified: {verified}, Missing: {missing}")
    return verified, missing

def progressive_table_verification(headers, data, coordinator, context) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Stepwise verification of extracted table structure.
    Logs and verifies each component: location, ballot types, candidates, totals.
    Returns (verified_headers, verified_data, structure_info)
    """
    from ..Context_Integration.context_coordinator import ContextCoordinator
    logger.info("[TABLE BUILDER][progressive_table_verification] Starting verification of extracted table.")
    coordinator = coordinator or ContextCoordinator()
    context = context or {}

    # 1. Detect location column (use context if available)
    location_header = None
    location_patterns = set()
    if coordinator and hasattr(coordinator, "library"):
        location_patterns = set(safe_get(coordinator.library, "location_patterns", [])) | LOCATION_KEYWORDS
    else:
        location_patterns = set(LOCATION_KEYWORDS)
    # Use context for location header if present
    location_header = safe_get(context, "location_header", None)
    if not location_header:
        for h in headers:
            if any(safe_lower(pat) in safe_lower(h) for pat in location_patterns):
                location_header = h
                break
    if not location_header:
        logger.warning("[TABLE BUILDER][progressive_table_verification] No location column detected.")
    else:
        logger.info(f"[TABLE BUILDER][progressive_table_verification] Detected location column: {location_header}")

    # 2. Detect ballot type columns (use context if available)
    ballot_types_headers = []
    context_ballot_type_header = safe_get(context, "ballot_type_header", None)
    if context_ballot_type_header:
        ballot_types_headers.append(context_ballot_type_header)
    ballot_types_headers += [h for h in headers if any(safe_lower(bt) in safe_lower(h) for bt in BALLOT_TYPES)]
    ballot_types_headers = list(dict.fromkeys(ballot_types_headers))  # deduplicate
    if not ballot_types_headers:
        logger.warning("[TABLE BUILDER][progressive_table_verification] No ballot type columns detected.")
    else:
        logger.info(f"[TABLE BUILDER][progressive_table_verification] Detected ballot type columns: {ballot_types_headers}")

    # 3. Detect candidate columns (using NER and context)
    candidate_headers = []
    context_candidate_header = safe_get(context, "candidate_header", None)
    if context_candidate_header:
        candidate_headers.append(context_candidate_header)
    for h in headers:
        ents = coordinator.extract_entities(h)
        # Only add header if a PERSON entity is found and the entity string is not empty
        if any(label == "PERSON" and ent and isinstance(ent, str) and ent.strip() for ent, label in ents):
            candidate_headers.append(h)
    candidate_headers = list(dict.fromkeys(candidate_headers))  # deduplicate
    if not candidate_headers:
        logger.warning("[TABLE BUILDER][progressive_table_verification] No candidate columns detected.")
    else:
        logger.info(f"[TABLE BUILDER][progressive_table_verification] Detected candidate columns: {candidate_headers}")

    # 4. Detect Grand Total column (use context if available)
    total_header = safe_get(context, "total_header", None)
    if not total_header:
        total_header = next((h for h in headers if "total" in safe_lower(h)), None)
    if not total_header:
        logger.warning("[TABLE BUILDER][progressive_table_verification] No Grand Total column detected.")
    else:
        logger.info(f"[TABLE BUILDER][progressive_table_verification] Detected Grand Total column: {total_header}")

    # 5. Verify row structure
    for i, row in enumerate(data[:5]):
        loc_val = safe_get(row, location_header, "")
        ballot_vals = [safe_get(row, h, "") for h in ballot_types_headers]
        candidate_vals = [safe_get(row, h, "") for h in candidate_headers]
        logger.info(f"[TABLE BUILDER][progressive_table_verification] Row {i}: location={loc_val}, ballot_types={ballot_vals}, candidates={candidate_vals}")

    # 6. Structure info summary
    structure_info = {
        "location_header": location_header,
        "ballot_types_headers": ballot_types_headers,
        "candidate_headers": candidate_headers,
        "total_header": total_header,
        "verified": all([location_header, ballot_types_headers, candidate_headers, total_header])
    }
    logger.info(f"[TABLE_BUILDER][progressive_table_verification] Structure summary: {structure_info}")

    # Optionally: prompt for correction or fallback if not verified
    # Optionally: persist structure_info for feedback learning

    return headers, data, structure_info

def rescan_and_verify(
    headers: List[str],
    data: List[Dict[str, Any]],
    coordinator: "ContextCoordinator",
    context: dict,
    threshold: float = 0.85
) -> Tuple[List[str], List[Dict[str, Any]], bool]:
    """
    Rescans headers and data, verifies with ML/NER, and retries if below threshold.
    Returns (headers, data, passed)
    """
    from ..Context_Integration.context_coordinator import ContextCoordinator
    coordinator = coordinator or ContextCoordinator()
    scores = []
    for h in headers:
        score = coordinator.score_header(h, context)
        scores.append(score)
    avg_score = sum(scores) / len(scores) if scores else 0
    passed = avg_score >= threshold
    if not passed:
        # Attempt to re-extract or re-map headers using NER/ML
        new_headers = []
        for h in headers:
            entities = coordinator.extract_entities(h)
            if entities:
                ent, label = entities[0]
                # Use the entity label and value for more robust header naming
                if label and ent and isinstance(ent, str) and ent.strip():
                    new_headers.append(f"{ent} ({label})")
                elif ent:
                    new_headers.append(str(ent))
                else:
                    new_headers.append(h)
            else:
                new_headers.append(h)
        headers = new_headers
        # Optionally, re-harmonize data
        headers, data = harmonize_headers_and_data(headers, data)
    logger.info(f"[TABLE BUILDER] Rescan and verify final table: {len(data)} rows, {len(headers)} columns (learned structure).")
    return headers, data, passed

# ===================================================================
# STRUCTURE DETECTION, CLASSIFICATION, PIVOTING
# ===================================================================

def force_fully_wide_format(headers, data, coordinator: "ContextCoordinator" = None, context=None) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Pivot to fully wide format: one row per location (real or synthetic),
    columns for each candidate/party/ballot type pair, plus special columns like
    Percent Reported and Misc Totals. Preserves all rows.
    """
    from ..Context_Integration.context_coordinator import ContextCoordinator
    coordinator = coordinator or ContextCoordinator()
    # 1. Find or synthesize location column
    location_col = next((h for h in headers if is_location_header(h)), None)
    if not location_col:
        location_col = "Location"
        for idx, row in enumerate(data):
            row[location_col] = (
                safe_get(context, "contest", []) if context and safe_get(context, "contest", []) else f"Row {idx+1}"
            )

    # 2. Find candidate and party columns
    candidate_col = next((h for h in headers if any(ck in safe_lower(h) for ck in CANDIDATE_KEYWORDS)), None)
    party_col = next((h for h in headers if "party" in safe_lower(h)), None)

    # 3. Find ballot type columns (known types, not location/candidate/party)
    ballot_types_cols = [
        h for h in headers
        if h not in (location_col, candidate_col, party_col) and any(bt.lower() in safe_lower(h) for bt in BALLOT_TYPES)
    ]
    # If no ballot type columns, use all except location/candidate/party/total/specials
    if not ballot_types_cols:
        ballot_types_cols = [
            h for h in headers
            if h not in (location_col, candidate_col, party_col)
            and "total" not in safe_lower(h)
            and not any(kw in safe_lower(h) for kw in PERCENT_KEYWORDS)
            and not any(kw in safe_lower(h) for kw in MISC_FOOTER_KEYWORDS)
        ]

    # 4. Find special columns
    percent_cols = [h for h in headers if any(kw in safe_lower(h) for kw in PERCENT_KEYWORDS)]
    misc_total_cols = [h for h in headers if any(kw in safe_lower(h) for kw in (MISC_FOOTER_KEYWORDS | TOTAL_KEYWORDS))]

    # 5. Get all unique locations, candidates, parties, ballot types
    locations = [safe_get(row, location_col, f"Row {i+1}") for i, row in enumerate(data)]
    unique_locations = sorted(set(loc for loc in locations if loc and str(loc).strip()))

    # Robust candidate extraction: strip, deduplicate, ignore empty
    candidates = set()
    parties = set()
    for row in data:
        candidate = safe_get(row, candidate_col, "")
        party = safe_get(row, party_col, "") if party_col else ""
        candidate = candidate.strip() if isinstance(candidate, str) else ""
        party = party.strip() if isinstance(party, str) else ""
        if candidate:
            candidates.add(candidate)
        if party:
            parties.add(party)
        # Supplement party with NER if missing
        if not party and candidate:
            ents = coordinator.extract_entities(candidate)
            for ent, label in ents:
                if label in {"ORG", "NORP"} and ent:
                    parties.add(ent)
    candidates = sorted(candidates)
    parties = sorted(parties)
    ballot_types = sorted(set(ballot_types_cols))

    # 6. Build wide headers
    wide_headers = [location_col]
    wide_headers.extend(percent_cols)
    candidate_party_pairs = []
    for row in data:
        candidate = safe_strip(safe_get(row, candidate_col, ""))
        party = safe_strip(safe_get(row, party_col, "")) if party_col else ""
        # If party is not found, try NER on candidate
        if not party and candidate:
            ents = coordinator.extract_entities(candidate)
            for ent, label in ents:
                if label in {"ORG", "NORP"} and ent:
                    party = ent
                    break
        if not party:
            party = "Other"
        candidate_party_pairs.append((candidate, party))
    for candidate, party in sorted(set(candidate_party_pairs)):
        for bt in ballot_types:
            if party:
                wide_headers.append(f"{candidate} ({party}) - {bt}")
            else:
                wide_headers.append(f"{candidate} - {bt}")
    wide_headers.extend(misc_total_cols)
    wide_headers.append("Grand Total")

    # 7. Build wide data, one row per unique location
    wide_data = []
    for loc in unique_locations:
        out_row = {h: "" for h in wide_headers}
        out_row[location_col] = loc
        grand_total = 0
        # Find all rows for this location
        for row in data:
            if safe_get(row, location_col, "") != loc:
                continue
            # Special columns
            for pcol in percent_cols:
                if pcol in out_row and safe_get(row, pcol, ""):
                    out_row[pcol] = safe_get(row, pcol, "")
            for mcol in misc_total_cols:
                if mcol in out_row and safe_get(row, mcol, ""):
                    out_row[mcol] = safe_get(row, mcol, "")
                    try:
                        grand_total += int(safe_replace(safe_get(row, mcol, "0"), ",", ""))
                    except Exception:
                        pass
            candidate = safe_strip(safe_get(row, candidate_col, ""))
            party = safe_strip(safe_get(row, party_col, "")) if party_col else ""
            # If party is not found, try NER on candidate
            if not party and candidate:
                ents = coordinator.extract_entities(candidate)
                for ent, label in ents:
                    if label in {"ORG", "NORP"} and ent:
                        party = ent
                        break
            if not party:
                party = "Other"
            for bt in ballot_types:
                key = f"{candidate} ({party}) - {bt}" if party else f"{candidate} - {bt}"
                val = safe_get(row, bt, "")
                if val and key in out_row:
                    out_row[key] = val
                    try:
                        grand_total += int(safe_replace(val, ",", ""))
                    except Exception:
                        pass
        out_row["Grand Total"] = str(grand_total)
        wide_data.append(out_row)
    return wide_headers, wide_data

def detect_table_structure(
    headers: List[str],
    data: List[Dict[str, Any]],
    coordinator: "ContextCoordinator",
    entity_info: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Annotates table structure using both NLP, DOM info, and data content.
    Returns a dict with structure type and detected entity columns.
    Never transforms data.
    """
    logger.info("[TABLE_CORE][detect_table_structure] Analyzing table structure.")
    if entity_info is None:
        entity_info = {}

    # Heuristic: If first column is "Candidate" and the rest are ballot types, it's already wide
    if headers and safe_lower(headers[0]) == "candidate" and all(
        any(bt in safe_lower(h) for bt in ["election day", "early voting", "absentee", "mail", "total"]) for h in headers[1:]
    ):
        return {"type_": "already-wide", "candidate_col": 0, "ballot_types_cols": list(range(1, len(headers)))}

    candidate_cols = []
    location_cols = []
    ballot_types_cols = []

    # Use entity_info, header heuristics, and data/coordinator for detection
    for idx, h in enumerate(headers):
        # Entity info from annotation
        if entity_info.get("people", []) and any(p in h for p in entity_info["people"]):
            candidate_cols.append(idx)
        if entity_info.get("locations", []) and any(l in h for l in entity_info["locations"]):
            location_cols.append(idx)
        if entity_info.get("ballot_types", []) and any(bt in h for bt in entity_info["ballot_types"]):
            ballot_types_cols.append(idx)

        # Fallback: heuristics
        if is_location_header(h):
            location_cols.append(idx)
        if any(bt.lower() in safe_lower(h) for bt in BALLOT_TYPES):
            ballot_types_cols.append(idx)

        # Use coordinator NER on header
        if coordinator and hasattr(coordinator, "extract_entities"):
            ents = coordinator.extract_entities(h)
            for ent, label in ents:
                if label == "PERSON" and idx not in candidate_cols:
                    candidate_cols.append(idx)
                if label in {"GPE", "LOC", "FAC"} and idx not in location_cols:
                    location_cols.append(idx)
                if label in {"ORG", "NORP"} and idx not in ballot_types_cols:
                    ballot_types_cols.append(idx)

        # Use data content: if most values in this column look like names, locations, or ballot types
        col_vals = [safe_get(row, h, "") for row in data]
        non_empty_vals = [v for v in col_vals if v]
        # Candidate: many unique, non-numeric, non-empty values
        if len(set(non_empty_vals)) > 3 and all(not safe_isdigit(str(v).replace(",", "")) for v in non_empty_vals):
            if idx not in candidate_cols:
                candidate_cols.append(idx)
        # Location: many unique, some match known locations
        if len(set(non_empty_vals)) > 3 and any(is_location_header(str(v)) for v in non_empty_vals):
            if idx not in location_cols:
                location_cols.append(idx)
        # Ballot type: matches known ballot type keywords
        if any(any(bt in safe_lower(str(v)) for bt in BALLOT_TYPES) for v in non_empty_vals):
            if idx not in ballot_types_cols:
                ballot_types_cols.append(idx)

    # Heuristic: if first col is candidate, columns are ballot types
    if candidate_cols and set(ballot_types_cols) == set(range(1, len(headers))):
        return {"type_": "candidate-major", "candidate_col": candidate_cols[0], "ballot_types_cols": ballot_types_cols}
    if location_cols and set(candidate_cols) == set(range(1, len(headers))):
        return {"type_": "precinct-major", "location_col": location_cols[0], "candidate_cols": candidate_cols}
    return {
        "type_": "ambiguous",
        "candidate_cols": candidate_cols,
        "location_cols": location_cols,
        "ballot_types_cols": ballot_types_cols
    }

def handle_candidate_major(headers, data, coordinator, context) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Handles tables where each row is a candidate, columns are ballot types.
    Uses safe_get, safe_replace, safe_lower, and robust context usage.
    """
    from ..Context_Integration.context_coordinator import ContextCoordinator
    coordinator = coordinator or ContextCoordinator()
    context = context or {}

    # Detect location and percent columns, prefer context if available
    location_header, percent_header = None, None
    if context:
        location_header = safe_get(context, "location_header", None)
        percent_header = safe_get(context, "percent_header", None)
    if not location_header or not percent_header:
        loc, pct, _ = dynamic_detect_location_header(headers, coordinator)
        if not location_header:
            location_header = loc
        if not percent_header:
            percent_header = pct
    if not location_header:
        location_header = "Precinct"
    if not percent_header:
        percent_header = "Percent Reported"

    # Detect candidate, party, and ballot type columns
    structure_info = detect_table_structure(headers, data, coordinator)
    candidate_col = safe_get(structure_info, "candidate_col", 0)
    party_col = safe_get(structure_info, "party_col", None)
    ballot_types_cols = safe_get(structure_info, "ballot_types_cols", list(range(1, len(headers))))

    # Get ballot type names
    ballot_types = [safe_get(headers, idx, "") for idx in ballot_types_cols]

    # Special columns
    percent_cols = [h for h in headers if any(kw in safe_lower(h) for kw in PERCENT_KEYWORDS)]
    misc_total_cols = [h for h in headers if any(kw in safe_lower(h) for kw in (TOTAL_KEYWORDS | MISC_FOOTER_KEYWORDS))]

    output_headers = [percent_header, location_header]
    candidate_party_map = {}
    for row in data:
        candidate = safe_get(row, safe_get(headers, candidate_col, ""), "")
        party = safe_get(row, safe_get(headers, party_col, ""), "") if party_col is not None else ""
        candidate = safe_strip(candidate)
        party = safe_strip(party)
        # If party is not found, try NER on candidate
        if not party and candidate:
            ents = coordinator.extract_entities(candidate)
            for ent, label in ents:
                if label in {"ORG", "NORP"} and ent:
                    party = ent
                    break
        if not party:
            party = "Other"
        candidate_party_map[candidate] = party
    for candidate, party in candidate_party_map.items():
        for bt in ballot_types:
            output_headers.append(f"{candidate} ({party}) - {bt}")
        output_headers.append(f"{candidate} ({party}) - Total")
    output_headers.append("Grand Total")

    # Build output data
    output_data = []
    location_vals = set(safe_get(row, location_header, "All") for row in data)
    for loc in location_vals:
        out_row = {h: "" for h in output_headers}
        out_row[location_header] = loc
        out_row[percent_header] = ""
        grand_total = 0
        for row in data:
            if safe_get(row, location_header, "All") != loc:
                continue
            # Special columns
            for pcol in percent_cols:
                if pcol in out_row and safe_get(row, pcol, ""):
                    out_row[pcol] = safe_get(row, pcol, "")
            for mcol in misc_total_cols:
                if mcol in out_row and safe_get(row, mcol, ""):
                    out_row[mcol] = safe_get(row, mcol, "")
                    try:
                        grand_total += int(safe_replace(safe_get(row, mcol, "0"), ",", ""))
                    except Exception:
                        pass
            candidate = safe_strip(safe_get(row, safe_get(headers, candidate_col, ""), ""))
            party = safe_strip(safe_get(row, safe_get(headers, party_col, ""), "")) if party_col is not None else ""
            if not party and candidate:
                ents = coordinator.extract_entities(candidate)
                for ent, label in ents:
                    if label in {"ORG", "NORP"} and ent:
                        party = ent
                        break
            if not party:
                party = "Other"
            for bt in ballot_types:
                key = f"{candidate} ({party}) - {bt}" if party else f"{candidate} - {bt}"
                val = safe_get(row, bt, "")
                if val and key in out_row:
                    out_row[key] = val
                    try:
                        grand_total += int(safe_replace(val, ",", ""))
                    except Exception:
                        pass
        out_row["Grand Total"] = str(grand_total)
        output_data.append(out_row)
    return harmonize_headers_and_data(output_headers, output_data)

def handle_precinct_major(headers, data, coordinator, context):
    """
    Handles tables where each row is a precinct, columns are candidates.
    Uses context_coordinator for robust detection of location/candidate columns and context-aware scoring.
    """
    from ..Context_Integration.context_coordinator import ContextCoordinator
    # Use provided coordinator if available, else instantiate
    coordinator = coordinator or ContextCoordinator()

    # --- Robust detection: check if table is truly precinct-major using structure detection ---
    structure_info = detect_table_structure(headers, data, coordinator)
    is_precinct_major = False
    # Heuristic: structure_info type or location/candidate columns
    if structure_info.get("type_") == "precinct-major":
        is_precinct_major = True
    elif structure_info.get("location_cols") or any(is_location_header(h) for h in headers):
        is_precinct_major = True
    elif context and safe_get(context, "expected_structure", None) == "precinct-major":
        is_precinct_major = True

    if not is_precinct_major:
        logger.warning("[handle_precinct_major] Table may not be precinct-major. Structure info: %s", structure_info)

    # Optionally, use context_coordinator to further validate or enrich context
    if hasattr(coordinator, "score_header"):
        # Score location/candidate columns for extra validation
        location_scores = [coordinator.score_header(h, context) for h in headers if is_location_header(h)]
        candidate_scores = [coordinator.score_header(h, context) for h in headers if any(ck in h.lower() for ck in CANDIDATE_KEYWORDS)]
        logger.info(f"[handle_precinct_major] Location header scores: {location_scores}, Candidate header scores: {candidate_scores}")

    # Proceed to pivot using robust context and coordinator
    return pivot_precinct_major_to_wide(headers, data, coordinator, context)

def handle_ambiguous(headers, data, coordinator, context) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Handles ambiguous tables by trying both handlers and picking the one with more filled data.
    Uses context_coordinator for additional context-aware scoring if available.
    """
    from ..Context_Integration.context_coordinator import ContextCoordinator
    coordinator = coordinator or ContextCoordinator()

    # Try candidate-major
    cand_headers, cand_data = handle_candidate_major(headers, data, coordinator, context)
    # Try precinct-major
    prec_headers, prec_data = handle_precinct_major(headers, data, coordinator, context)

    # Heuristic: pick the one with more non-empty cells, using safe_values for robustness
    def non_empty_count(data) -> int:
        return sum(1 for row in data for v in safe_values(row) if v not in ("", "0", 0, None))

    cand_score = non_empty_count(cand_data)
    prec_score = non_empty_count(prec_data)

    # If context_coordinator has a scoring method, use it to break ties or further improve selection
    if hasattr(coordinator, "score_header"):
        cand_struct_score = sum(coordinator.score_header(h, context) for h in cand_headers) / max(1, len(cand_headers))
        prec_struct_score = sum(coordinator.score_header(h, context) for h in prec_headers) / max(1, len(prec_headers))
        # Weighted: prefer more filled data, but use structure score as tiebreaker
        if cand_score > prec_score or (cand_score == prec_score and cand_struct_score >= prec_struct_score):
            return cand_headers, cand_data
        else:
            return prec_headers, prec_data
    else:
        if cand_score >= prec_score:
            return cand_headers, cand_data
        else:
            return prec_headers, prec_data

def pivot_to_wide_format(
    headers: List[str],
    data: List[Dict[str, Any]],
    entity_info: Dict[str, Any],
    coordinator: "ContextCoordinator",
    context: dict = None
) -> Tuple[List[str], List[Dict[str, Any]]]:
    logger.info("[TABLE_CORE][pivot_to_wide_format] Pivoting to wide format.")
    # 1. Detect location header robustly and normalize to "Precinct"
    location_header = None
    percent_header = None

    # Use entity_info for robust detection if available
    if entity_info:
        location_header = entity_info.get("location_header", None)
        percent_header = entity_info.get("percent_header", None)
        entity_candidates = set(safe_strip(c) for c in entity_info.get("people", []) if c)
        entity_locations = set(safe_strip(l) for l in entity_info.get("locations", []) if l)
    else:
        entity_candidates = set()
        entity_locations = set()

    # Use coordinator for fallback detection if needed
    if coordinator and (not location_header or not percent_header):
        detected_loc, detected_pct, _ = dynamic_detect_location_header(headers, coordinator)
        if not location_header:
            location_header = detected_loc
        if not percent_header:
            percent_header = detected_pct

    # Fallback to header scan if not found in entity_info or coordinator
    for h in headers:
        if not location_header and is_location_header(h) and safe_lower(h) != "candidate":
            location_header = h
        if not percent_header and (safe_lower(h) in (safe_lower(ph) for ph in PERCENT_KEYWORDS) or "%" in h or "reported" in safe_lower(h)):
            percent_header = h

    # Use context for fallback if still not found
    if not location_header and context:
        location_header = safe_get(context, "location_header", "Precinct")
    if not percent_header and context:
        percent_header = safe_get(context, "percent_header", "Percent Reported")

    if not location_header:
        location_header = "Precinct"
    if location_header != "Precinct":
        headers = ["Precinct" if h == location_header else h for h in headers]
        for row in data:
            row["Precinct"] = row.pop(location_header)
        location_header = "Precinct"

    # 2. Gather all unique candidates and ballot types using canonical normalization
    candidates = set(entity_candidates)
    ballot_types = set()
    for row in data:
        cand = safe_get(row, "Candidate", "")
        if cand:
            candidates.add(safe_strip(cand))
        for h in row.keys():
            norm_h = normalize_segment_text(h)
            if norm_h in [normalize_segment_text(bt) for bt in BALLOT_TYPES_SORT_ORDER] or h in BALLOT_TYPES_SORT_ORDER:
                ballot_types.add(h)
    # Fallback: scan headers if not found in data
    if not ballot_types:
        for h in headers:
            norm_h = normalize_segment_text(h)
            if norm_h in [normalize_segment_text(bt) for bt in BALLOT_TYPES_SORT_ORDER] or h in BALLOT_TYPES_SORT_ORDER:
                ballot_types.add(h)
    # Use canonical sort order for ballot types
    ballot_types_sorted = [bt for bt in BALLOT_TYPES_SORT_ORDER if bt in ballot_types]
    for bt in sorted(ballot_types):
        if bt not in ballot_types_sorted:
            ballot_types_sorted.append(bt)

    # 3. Build wide headers: Precinct, % Reported, [Candidate - BallotType ... Total Vote], Grand Total
    wide_headers = [location_header]
    if percent_header:
        wide_headers.append(percent_header)
    for candidate in sorted(candidates):
        for bt in ballot_types_sorted:
            wide_headers.append(f"{candidate} - {bt}")
        wide_headers.append(f"{candidate} - Total Vote")
    wide_headers.append("Grand Total")

    # 4. Build wide data, one row per unique location
    # Use entity_info locations if available, else extract from data
    if entity_locations:
        location_values = entity_locations
    else:
        location_values = set(safe_get(row, location_header, "") for row in data if safe_get(row, location_header, ""))

    wide_data = []
    for loc in sorted(location_values):
        out_row = {h: "" for h in wide_headers}
        out_row[location_header] = loc
        if percent_header:
            # Use the first found value for this precinct
            for row in data:
                if safe_get(row, location_header, "") == loc and percent_header in row:
                    out_row[percent_header] = row[percent_header]
                    break
        grand_total = 0
        for candidate in sorted(candidates):
            cand_total = 0
            for bt in ballot_types_sorted:
                val = ""
                for row in data:
                    if safe_get(row, location_header, "") == loc and safe_strip(safe_get(row, "Candidate", "")) == candidate:
                        val = row.get(bt, "") or row.get(f"{candidate} - {bt}", "")
                        break
                out_row[f"{candidate} - {bt}"] = val if val not in (None, "") else "-"
                try:
                    if val and str(val).replace(",", "").isdigit():
                        cand_total += int(str(val).replace(",", ""))
                except Exception:
                    pass
            out_row[f"{candidate} - Total Vote"] = str(cand_total)
            grand_total += cand_total
        out_row["Grand Total"] = str(grand_total)
        wide_data.append(out_row)
    logger.info(
        f"[TABLE_CORE][pivot_to_wide_format] Wide format: {len(wide_data)} rows, {len(wide_headers)} columns. "
        f"Used coordinator: {bool(coordinator)}, Used context: {bool(context)}"
    )
    return wide_headers, wide_data

def pivot_precinct_major_to_wide(
    headers: List[str],
    data: List[Dict[str, Any]],
    coordinator: "ContextCoordinator",
    context: dict
) -> Tuple[List[str], List[Dict[str, Any]], dict]:
    """
    Pivot a precinct-major table to wide format:
    Precinct | Percent Reported | [Candidate (Party) - BallotType ... Total Votes] | [Misc Totals] | Grand Total
    Handles variable ballot types and miscellaneous columns.
    Context is used for fallback values and robust header/entity detection.
    Returns (output_headers, output_rows, info_dict) where info_dict includes location_entity_value.
    """
    from ..Context_Integration.context_coordinator import ContextCoordinator
    coordinator = ContextCoordinator()
    # Use context for robust header/entity detection
    location_header, percent_header, location_entity_value = dynamic_detect_location_header(headers, coordinator)
    if not percent_header:
        percent_header = safe_get(context, "percent_header", "Percent Reported")
    if not location_header:
        location_header = safe_get(context, "location_header", "Precinct")

    # Parse headers
    candidate_party_ballot = {}  # (candidate, party) -> {ballot_types: header}
    ballot_types_set = set()
    misc_columns = []
    candidate_party_set = set()

    for h in headers:
        m = re.match(r"(.+?)\s*\((.+?)\)\s*-\s*(.+)", h)
        if m:
            candidate, party, ballot_types = m.groups()
            candidate = candidate.strip()
            party = party.strip()
            ballot_types = ballot_types.strip()
            candidate_party_set.add((candidate, party))
            ballot_types_set.add(ballot_types)
            candidate_party_ballot.setdefault((candidate, party), {})[ballot_types] = h
        else:
            # Try: Candidate - BallotType
            m = re.match(r"(.+?)\s*-\s*(.+)", h)
            if m:
                candidate, ballot_types = m.groups()
                candidate = candidate.strip()
                party = ""
                ballot_types = ballot_types.strip()
                candidate_party_set.add((candidate, party))
                ballot_types_set.add(ballot_types)
                candidate_party_ballot.setdefault((candidate, party), {})[ballot_types] = h
            else:
                # Try: BallotType only (miscellaneous totals)
                ballot_types = h.strip()
                ballot_types_set.add(ballot_types)
                misc_columns.append(h)

    # Remove location and percent headers from ballot_types/misc
    for col in [location_header, percent_header]:
        if col in ballot_types_set:
            ballot_types_set.remove(col)
        if col in misc_columns:
            misc_columns.remove(col)

    # Remove candidate columns from misc_columns
    for (candidate, party), bt_map in candidate_party_ballot.items():
        for bt, h in safe_items(bt_map):
            if h in misc_columns:
                misc_columns.remove(h)

    # Sort ballot types: Election Day, Early Voting, Absentee, ...rest alphabetically
    ballot_types = []
    for bt in BALLOT_TYPES_SORT_ORDER:
        if bt in ballot_types_set:
            ballot_types.append(bt)
    for bt in sorted(ballot_types_set):
        if bt not in ballot_types:
            ballot_types.append(bt)

    # 3. Build output headers
    output_headers = [location_header, percent_header]
    candidate_columns = []
    for candidate, party in sorted(candidate_party_set):
        for bt in ballot_types:
            candidate_columns.append(f"{candidate} ({party}) - {bt}")
        candidate_columns.append(f"{candidate} ({party}) - Total Votes")
    output_headers.extend(candidate_columns)
    output_headers.extend(misc_columns)
    output_headers.append("Grand Total")

    # Build output rows
    output_rows = []
    for row in data:
        if len(row) != len(headers):
            logger.warning(f"[TABLE BUILDER] pivot_precinct_major_to_wide Row length mismatch: {row}")
        out_row = {}
        # Use context fallback for location/percent if missing
        # --- Robustly use location_entity_value if location is missing ---
        location_val = safe_get(row, location_header, None)
        if not location_val and location_entity_value:
            location_val = location_entity_value
        out_row[location_header] = location_val if location_val is not None else safe_get(context, "location_value", "")
        out_row[percent_header] = safe_get(row, percent_header, safe_get(context, "percent_value", "Fully Reported"))
        grand_total = 0
        # Candidate columns
        for candidate, party in sorted(candidate_party_set):
            cand_total = 0
            bt_map = candidate_party_ballot.get((candidate, party), {})
            for bt in ballot_types:
                col = f"{candidate} ({party}) - {bt}"
                val = safe_get(row, safe_get(bt_map, bt, ""), "")
                try:
                    ival = int(safe_replace(val, ",", "")) if val else 0
                except Exception:
                    ival = 0
                out_row[col] = str(ival) if val != "" else ""
                cand_total += ival
            out_row[f"{candidate} ({party}) - Total Votes"] = str(cand_total)
            grand_total += cand_total
        # Misc columns
        for h in misc_columns:
            out_row[h] = safe_get(row, h, "")
            try:
                misc_val = safe_get(row, h, "0")
                misc_val_clean = safe_replace(misc_val, ",", "")
                if misc_val_clean and (safe_isdigit(misc_val_clean) or (safe_startswith(misc_val_clean, "-") and safe_isdigit(misc_val_clean[1:]))):
                    grand_total += int(misc_val_clean)
            except Exception:
                pass
        out_row["Grand Total"] = str(grand_total)
        output_rows.append(out_row)

    # Add a single totals row at the end
    totals_row = {h: "" for h in output_headers}
    totals_row[location_header] = "TOTAL"
    totals_row[percent_header] = ""
    for h in candidate_columns + misc_columns + ["Grand Total"]:
        try:
            values = [safe_replace(safe_get(r, h, "0"), ",", "") for r in output_rows]
            if all(v == "" or safe_isdigit(v) or (safe_startswith(v, "-") and safe_isdigit(v[1:])) for v in values):
                totals_row[h] = str(sum(int(v) for v in values if v != ""))
            else:
                totals_row[h] = ""
        except Exception:
            totals_row[h] = ""
    output_rows.append(totals_row)
    logger.info(
        f"[TABLE BUILDER] Build dynamic tables Final table: {len(output_rows)} rows, {len(output_headers)} columns. "
        f"Location entity value used: {location_entity_value}"
    )
    # Return info dict for downstream use
    info_dict = {
        "location_header": location_header,
        "percent_header": percent_header,
        "location_entity_value": location_entity_value,
    }
    return output_headers, output_rows, info_dict

def dynamic_detect_location_header(headers: List[str], coordinator: "ContextCoordinator") -> Tuple[str, str, str]:
    """
    Dynamically detect the first and second location columns (e.g., precinct, ward, city, district, municipal).
    Uses context, regex, NER, and library.
    Returns (location_header, percent_reported_header, location_entity_value)
    """
    from ..Context_Integration.context_coordinator import ContextCoordinator
    coordinator = ContextCoordinator()
    # Use patterns from the context library if available, else fall back to librarian
    location_patterns = set()
    percent_patterns = set()
    if coordinator and hasattr(coordinator, "library"):
        location_patterns = set()
        percent_patterns = set()
        if coordinator and hasattr(coordinator, "get_dom_parts"):
            dom_parts = coordinator.get_dom_parts()
            location_patterns = set(dom_parts.get("location_patterns", []))
            percent_patterns = set(dom_parts.get("percent_patterns", []))
        if not location_patterns and hasattr(coordinator, "get_known_counties"):
            location_patterns = set(LOCATION_KEYWORDS)
        if not percent_patterns:
            percent_patterns = set(PERCENT_KEYWORDS)
    if not location_patterns:
        location_patterns = set(LOCATION_KEYWORDS)
    if not percent_patterns:
        percent_patterns = set(PERCENT_KEYWORDS)

    norm_headers = [normalize_text(h) for h in headers]
    location_header = None
    percent_header = None
    location_entity_value = None

    # 1. Try exact match (case-insensitive)
    for idx, h in enumerate(norm_headers):
        for pat in location_patterns:
            if normalize_text(pat) == h:
                location_header = headers[idx]
                break
        if location_header:
            break

    # 2. Try substring match
    if not location_header:
        for idx, h in enumerate(norm_headers):
            for pat in location_patterns:
                if normalize_text(pat) in h:
                    location_header = headers[idx]
                    break
            if location_header:
                break

    # 3. Try spaCy NER if available, and store entity value
    if not location_header and coordinator:
        for idx, h in enumerate(headers):
            entities = coordinator.extract_entities(h)
            for ent, label in entities:
                if label in {"GPE", "LOC", "FAC"}:
                    location_header = headers[idx]
                    location_entity_value = ent  # <-- Store the entity value
                    break
            if location_header:
                break

    # 4. Fallback to first column
    if not location_header and headers:
        location_header = headers[0]

    # Percent header: exact match first
    for idx, h in enumerate(norm_headers):
        for pat in percent_patterns:
            if normalize_text(pat) == h:
                percent_header = headers[idx]
                break
        if percent_header:
            break

    # Percent header: substring match
    if not percent_header:
        for idx, h in enumerate(norm_headers):
            for pat in percent_patterns:
                if normalize_text(pat) in h:
                    percent_header = headers[idx]
                    break
            if percent_header:
                break

    # Fallback: any header with '%' in it
    if not percent_header and headers:
        percent_header = next((h for h in headers if "%" in h), None)

    logger.info(f"[TABLE BUILDER] Location header detected: {location_header}, Percent header detected: {percent_header}, Location entity: {location_entity_value}")
    return location_header, percent_header, location_entity_value

def is_likely_header(row) -> bool:
    """
    Heuristically determine if a row is likely a header row.
    Uses robust normalization (safe_lower, safe_strip) and keyword sets.
    """
    # Combine all relevant keywords into a single set for header detection
    known_fields = (
        set(safe_lower(k) for k in CANDIDATE_KEYWORDS)
        | set(safe_lower(k) for k in PARTY_KEYWORDS)
        | set(safe_lower(k) for k in LOCATION_KEYWORDS)
        | set(safe_lower(k) for k in PERCENT_KEYWORDS)
        | set(safe_lower(k) for k in TOTAL_KEYWORDS)
        | {"votes", "percent", "district", "party", "candidate"}
    )
    # Use safe_lower and safe_strip for each cell
    return sum(
        1 for cell in row
        if any(k in safe_lower(safe_strip(cell)) for k in known_fields)
    ) >= 2

# ===================================================================
# ADVANCED/UTILITY FUNCTIONS
# ===================================================================

def normalize_text(text, lang="en", collapse_whitespace=True, translate_func=None) -> str:
    """
    Normalize text for comparison:
    - Converts to string, strips, lowercases, removes accents.
    - Optionally collapses whitespace.
    - Optionally translates if a translation function is provided and lang != 'en'.
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.strip().lower()
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    if collapse_whitespace:
        text = re.sub(r"\s+", " ", text)
    if lang != "en" and translate_func is not None:
        text = translate_func(text, lang)
    return text

def normalize_header(header, lang="en", collapse_whitespace=True, translate_func=None) -> str:
    """
    Normalize header for comparison and deduplication:
    - Converts to string, strips, lowercases, removes accents.
    - Optionally collapses whitespace.
    - Optionally translates if a translation function is provided and lang != 'en'.
    """
    if not isinstance(header, str):
        header = str(header)
    header = header.strip().lower()
    header = unicodedata.normalize('NFKD', header).encode('ascii', 'ignore').decode('ascii')
    if collapse_whitespace:
        header = re.sub(r"\s+", " ", header)
    if lang != "en" and translate_func is not None:
        header = translate_func(header, lang)
    return header

def is_date_like(val) -> bool:
    """
    Robustly determine if a value is date-like.
    Handles strings, numbers, and common date formats. Ignores empty/null values.
    """
    if val is None or (isinstance(val, str) and not val.strip()):
        return False
    # Accept numeric timestamps (e.g., 20230704)
    if isinstance(val, (int, float)) and 1800 < val < 30000000:
        return True
    # Accept ISO and common date formats
    try:
        if isinstance(val, bytes):
            val = val.decode("utf-8", errors="ignore")
        val_str = str(val).strip()
        # Quick reject for very short or non-date-like strings
        if len(val_str) < 6 or not any(c.isdigit() for c in val_str):
            return False
        # Try parsing
        dateutil.parser.parse(val_str, fuzzy=True)
        return True
    except Exception:
        return False

def detect_language(headers) -> str:
    """
    Detect language of headers (robust, supports empty input and fallback).
    Uses langdetect if available, else defaults to 'en'.
    """
    try:
        DetectorFactory.seed = 0  # For deterministic results
        if not headers or not isinstance(headers, (list, tuple)):
            return "en"
        text = " ".join(str(h) for h in headers if h)
        if not text.strip():
            return "en"
        lang = detect(text)
        # Defensive: only return ISO 639-1 codes, fallback to 'en'
        if isinstance(lang, str) and len(lang) == 2:
            return lang
        return "en"
    except Exception:
        return "en"

def dynamic_required_columns(context, default_required=None) -> set:
    """
    Adjust required columns based on context and robust election constants.
    Uses LOCATION_KEYWORDS, PERCENT_KEYWORDS, TOTAL_KEYWORDS from constants.py for flexibility.
    """
    # Start with a robust default set if not provided
    if default_required is None:
        # Use canonical names for location and percent columns
        default_required = set(["Grand Total", "Precinct"])
        # Add a generic location column if not already present
        default_required.update([kw.title() for kw in LOCATION_KEYWORDS if kw.lower() in {"precinct", "district", "ward", "county", "city"}])
        # Add percent reported if relevant
        default_required.update([kw.title() for kw in PERCENT_KEYWORDS])
    # Remove percent columns if context says not present
    if not safe_get(context, "has_percent_reported", True):
        for kw in list(default_required):
            if any(p.lower() in kw.lower() for p in PERCENT_KEYWORDS):
                default_required.discard(kw)
    # Remove location columns if context says not present
    if not safe_get(context, "has_location", True):
        for kw in list(default_required):
            if any(lk.lower() in kw.lower() for lk in LOCATION_KEYWORDS):
                default_required.discard(kw)
    # Remove total columns if context says not present
    if not safe_get(context, "has_totals", True):
        for kw in list(default_required):
            if any(tk.lower() in kw.lower() for tk in TOTAL_KEYWORDS):
                default_required.discard(kw)
    # Add any extra required columns from context
    extra_required = safe_get(context, "extra_required_columns", [])
    if extra_required:
        default_required.update(extra_required)
    return default_required

def log_failed_container(page, container, selector, idx, error_msg) -> None:
    """
    Log details of a failed container extraction, using robust safe_* utilities for all DOM operations.
    The page variable is used to provide additional context if available.
    """
    if container is None:
        logger.error(f"[TABLE BUILDER] log_failed_container: container is None for selector {selector} idx {idx}")
        return
    try:
        # Safely get outer HTML
        html = safe_evaluate(container, "el => el.outerHTML", logger) or ""
        # Safely get parent element and its attributes
        parent = safe_locator(container, "xpath=..", logger)
        parent_class = safe_get_attribute(parent, "class", logger) or ""
        parent_id = safe_get_attribute(parent, "id", logger) or ""
        # Safely get heading above the container
        heading = ""
        heading_loc = safe_locator(container, "xpath=preceding-sibling::*[self::h1 or self::h2 or self::h3 or self::h4 or self::h5 or self::h6][1]", logger)
        if safe_count(heading_loc, logger) > 0:
            heading_el = safe_nth(heading_loc, 0, logger)
            heading = safe_inner_text(heading_el, logger).strip() if heading_el else ""
        # Optionally, get page URL for extra context
        page_url = ""
        if page is not None and hasattr(page, "url"):
            page_url = getattr(page, "url", "")
        log_entry = {
            "selector": selector,
            "container_idx": idx,
            "parent_class": parent_class,
            "parent_id": parent_id,
            "heading": heading,
            "error": error_msg,
            "html": (html[:2000] if html else ""),
            "page_url": page_url
        }
        safe_selector = safe_replace(selector, ".", "_")
        log_path = get_safe_log_path(f"failed_container_{safe_selector}_{idx}.json")
        with open(log_path, "wb") as f:
            f.write(orjson.dumps(log_entry))
        logger.error(f"[TABLE BUILDER] Failed container logged: {log_path}")
    except Exception as e:
        logger.error(f"[TABLE BUILDER] Could not log failed container: {e}")

def suggest_new_row_classes_from_logs(log_dir) -> Tuple[List[str], List[str]]:
    """
    Analyze failed container logs and suggest new likely row classes/IDs.
    """
    class_counter = Counter()
    parent_counter = Counter()
    for path in glob.glob(os.path.join(log_dir, "failed_container_*.json")):
        with open(path, "rb") as f:
            entry = orjson.loads(f.read())
            cls = safe_get(entry, "parent_class", "")
            if cls:
                for c in safe_split(cls):
                    class_counter[c] += 1
            parent_id = safe_get(entry, "parent_id", "")
            if parent_id:
                parent_counter[parent_id] += 1
    # Suggest top classes/IDs as new selectors
    suggested_classes = [c for c, _ in class_counter.most_common(10)]
    suggested_ids = [pid for pid, _ in parent_counter.most_common(5)]
    logger.info("Suggested new row classes:", suggested_classes)
    logger.info("Suggested new row IDs:", suggested_ids)
    return suggested_classes, suggested_ids

def load_dom_patterns(log_path=None) -> list[dict]:
    """
    Loads all DOM patterns, returns a list of dicts.
    """
    if log_path is None:
        log_path = get_safe_log_path("dom_pattern_log.jsonl")
    if not os.path.exists(log_path):
        return []
    with open(log_path, "rb") as f:
        return [orjson.loads(line) for line in f if line.strip()]

def remove_footer_and_summary_rows(data, headers) -> list[dict]:
    """
    Remove rows that are likely summary, totals, or repeated headers.
    --- Only remove if 'total' or 'summary' appears in a column that is a total/summary column.
    Advanced: also skips rows that are all empty or all repeated values.
    """
    filtered = []
    total_cols = [h for h in headers if any(kw in safe_lower(h) for kw in TOTAL_KEYWORDS.union(MISC_FOOTER_KEYWORDS))]
    for row in data:
        values = list(safe_values(row))
        # Advanced: skip if all values are empty or all values are the same (repeated header row)
        if not any(v not in ("", None) for v in values):
            continue
        if len(set(values)) == 1 and len(values) > 1:
            continue
        # --- Only remove if 'total' or 'summary' appears in a total/summary column
        remove = False
        for h in total_cols:
            v = safe_get(row, h, "")
            if any(kw in safe_lower(str(v)) for kw in TOTAL_KEYWORDS.union(MISC_FOOTER_KEYWORDS)):
                remove = True
                break
        # --- Do not remove if header row repeated (keep as is)
        if not remove:
            filtered.append(row)
    return filtered

def remove_outlier_and_empty_rows(data, min_non_empty=2) -> list[dict]:
    """
    Remove rows with too many empty or repeated values.
    --- Only keep rows with at least min_non_empty non-empty values.
    Advanced: also skips rows where all values are the same (likely repeated header or noise).
    """
    filtered = []
    for row in data:
        values = list(safe_values(row))
        non_empty = [v for v in values if v not in ("", None)]
        # Only keep if at least min_non_empty non-empty values
        if len(non_empty) >= min_non_empty:
            # Skip if all values are the same (repeated header or noise)
            if len(set(values)) == 1 and len(values) > 1:
                continue
            filtered.append(row)
    return filtered

def review_learned_table_structures(log_path=None) -> None:
    """
    CLI to review/edit learned table structures.
    """
    # --- Use log directory parent to webapp for default path
    if log_path is None:
        log_path = get_safe_log_path("table_structure_learning_log.jsonl")
    if not os.path.exists(log_path):
        logger.info("No learned table structures found.")
        return

    entries = []
    with open(log_path, "rb") as f:
        for line in f:
            try:
                entry = orjson.loads(line)
                entries.append(entry)
            except Exception:
                continue

    for idx, entry in enumerate(entries):
        logger.info(f"\n[{idx}] Contest: {safe_get(entry, 'contest', [])}")
        logger.info(f"    Headers: {safe_get(entry, 'headers', [])}")
        logger.info(f"    Context: {safe_get(entry, 'context', [])}")
        logger.info(f"    Result: {safe_get(entry, 'result', [])}")
        logger.info("-" * 40)

    while True:
        cmd = input("\nEnter entry number to delete/edit, or 'q' to quit: ").strip()
        if cmd.lower() == "q":
            break
        if cmd.isdigit():
            idx = int(cmd)
            if 0 <= idx < len(entries):
                action = input("Delete (d) or Edit (e) this entry? [d/e]: ").strip().lower()
                if action == "d":
                    entries.pop(idx)
                    logger.info("Entry deleted.")
                elif action == "e":
                    new_headers = input("Enter new headers as comma-separated values: ").strip().split(",")
                    entries[idx]["headers"] = [h.strip() for h in new_headers]
                    logger.info("Headers updated.")
                else:
                    logger.info("Unknown action.")
            else:
                logger.warning("Invalid entry number.")
        # Save changes
        with open(log_path, "wb") as f:
            for entry in entries:
                f.write(orjson.dumps(entry) + b"\n")
        logger.info("Changes saved.")

def table_signature(headers) -> str:
    return hashlib.md5(orjson.dumps(headers, sort_keys=True)).hexdigest()

def load_table_structure_cache() -> dict:
    if os.path.exists(TABLE_STRUCTURE_CACHE_PATH):
        with open(TABLE_STRUCTURE_CACHE_PATH, "rb") as f:
            return orjson.loads(f.read())
    return {}

def save_table_structure_cache(cache) -> None:
    with open(TABLE_STRUCTURE_CACHE_PATH, "wb") as f:
        f.write(orjson.dumps(cache))

def cache_table_structure(domain, headers, structure) -> None:
    cache = load_table_structure_cache()
    sig = f"{domain}:{table_signature(headers)}"
    cache[sig] = structure
    save_table_structure_cache(cache)

def get_cached_table_structure(domain, headers) -> list[dict]:
    cache = load_table_structure_cache()
    sig = f"{domain}:{table_signature(headers)}"
    return safe_get(cache, sig, [])

def guess_contest(table_headers, known_titles) -> str | None:
    """
    Try to match table headers to known contest titles using robust matching.
    Uses CONTEST_TITLE_KEYWORDS and CONTEST_TITLE_SKIP_PHRASES from constants.py.
    Returns the best-matching contest keyword or None.
    """
    # Normalize all keywords and skip phrases for robust matching
    contest_keywords = set(normalize_for_matching(k) for k in CONTEST_TITLE_KEYWORDS)
    skip_phrases = set(normalize_for_matching(k) for k in CONTEST_TITLE_SKIP_PHRASES)
    contest_keywords.update(normalize_for_matching(k) for k in known_titles if k)

    best_match = None
    best_score = 0.0

    for header in table_headers:
        if not header or not isinstance(header, str):
            continue
        header_norm = normalize_for_matching(header)
        # Skip known non-contest/summary/footer phrases
        if any(skip in header_norm for skip in skip_phrases):
            continue
        # Try exact and substring match
        for keyword in contest_keywords:
            if keyword in header_norm or header_norm in keyword:
                return keyword
        # Fuzzy match with score
        matches = difflib.get_close_matches(header_norm, contest_keywords, n=1, cutoff=0.7)
        if matches:
            score = difflib.SequenceMatcher(None, header_norm, matches[0]).ratio()
            if score > best_score:
                best_match = matches[0]
                best_score = score
    return best_match

def extract_title_from_html_near_table(table_idx, dom_nodes, window=5) -> str:
    """
    Scan nearby DOM nodes for likely contest titles.
    Returns the first likely contest title found, or None.
    """
    # Defensive: use constants, fallback to defaults if not present
    title_tags = set(CONTEST_TITLE_TAGS) if 'CONTEST_TITLE_TAGS' in locals() or 'CONTEST_TITLE_TAGS' in globals() else {"h1", "h2", "h3", "caption"}
    min_words = CONTEST_TITLE_MIN_WORDS if 'CONTEST_TITLE_MIN_WORDS' in locals() or 'CONTEST_TITLE_MIN_WORDS' in globals() else 3
    skip_phrases = set(normalize_for_matching(k) for k in CONTEST_TITLE_SKIP_PHRASES) if 'CONTEST_TITLE_SKIP_PHRASES' in locals() or 'CONTEST_TITLE_SKIP_PHRASES' in globals() else set()

    idx_range = range(max(0, table_idx - window), min(len(dom_nodes), table_idx + window + 1))
    for idx in idx_range:
        node = dom_nodes[idx]
        tag = safe_lower(safe_get(node, "tag", ""))
        if tag in title_tags:
            text = safe_strip(safe_get(node, "html", ""))
            if text and len(safe_split(text)) >= min_words:
                text_norm = normalize_for_matching(text)
                if any(skip in text_norm for skip in skip_phrases):
                    continue
                return text
    return None

def merge_multirow_headers(header_rows) -> list[str]:
    """
    Merge multiple header rows (e.g., stacked headers) into a single header list.
    Uses safe_strip and safe_isdigit for robustness.
    """
    merged = []
    for cols in zip(*header_rows):
        merged_col = " ".join([
            c for c in cols
            if c and safe_strip(c) and not safe_isdigit(safe_strip(c))
        ])
        merged.append(safe_strip(merged_col))
    return merged

def fuzzy_merge_headers(headers, threshold=0.85) -> list[str]:
    """
    Merge similar headers using fuzzy matching.
    """
    merged = []
    used = set()
    for i, h in enumerate(headers):
        if i in used:
            continue
        group = [h]
        for j, h2 in enumerate(headers):
            if i != j and j not in used:
                score = difflib.SequenceMatcher(None, normalize_header(h), normalize_header(h2)).ratio()
                if score > threshold:
                    group.append(h2)
                    used.add(j)
        merged.append(group[0])  # Keep the first as canonical
        used.add(i)
    return merged

def profile_extraction_step(func) -> callable:
    """
    Decorator to profile extraction speed.
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"[PROFILE] {func.__name__} took {elapsed:.3f}s")
        return result
    return wrapper

def log_decision(decision, context=None) -> None:
    """
    Log not just errors but also decisions made by heuristics for later review.
    """
    logger.info(f"[DECISION] {decision} | Context: {context}")

def handle_nested_tables(page) -> list[tuple[list[str], list[list[str]], dict]]:
    """
    Handle tables within tables or complex nested DOM structures.
    Returns a list of (headers, data, diagnostics) tuples.
    Enhanced: attaches diagnostics, skips empty tables, logs extraction.
    """
    try:
        results = []
        tables = safe_locator(page, "table table", logger)
        table_count = safe_count(tables, logger)
        for i in range(table_count):
            table = safe_nth(tables, i, logger)
            if table is not None:
                headers, data, diagnostics = extract_table_data(table)
                diagnostics = diagnostics or {}
                diagnostics["nested_table_index"] = i
                if headers and data:
                    results.append((headers, data, diagnostics))
                else:
                    logger.warning(f"[HANDLE NESTED TABLES] Skipping empty or malformed nested table at index {i}.")
        if not results:
            logger.warning("[HANDLE NESTED TABLES] No nested tables extracted.")
        return results
    except Exception as e:
        logger.error(f"[HANDLE NESTED TABLES] Error: {e}")
        return []

def fuzzy_in(word, text, threshold=0.7) -> bool:
    """Return True if word is in text by substring or fuzzy match."""
    word = safe_strip(safe_lower(word))
    text = safe_strip(safe_lower(text))
    if word in text:
        return True
    # Fuzzy match: allow for partials (e.g., "town" in "orangetown")
    ratio = SequenceMatcher(None, word, text).ratio()
    return ratio >= threshold

def normalize_for_matching(text) -> str:
    text = safe_strip(safe_lower(text))
    table = str.maketrans('', '', string.punctuation)
    text = safe_translate(text, table)
    return text

def contains_location_keyword(text, keywords=LOCATION_KEYWORDS) -> bool:
    text_norm = normalize_for_matching(text)
    for kw in keywords:
        # Match as a whole word or as a suffix/prefix (e.g., "orangetown")
        if re.search(rf"\b{re.escape(kw)}\b", text_norm):
            return True
        if kw in text_norm:
            return True
    return False

def is_location_header(header) -> bool:
    """
    Robustly determine if a header is a location column using LOCATION_KEYWORDS and abbreviations.
    This is the SINGLE SOURCE OF TRUTH for location column detection.
    - Uses normalization, substring, fuzzy, and regex matching.
    - Always update LOCATION_KEYWORDS in constants.py for new variants.
    """
    header_norm = normalize_for_matching(header)
    for kw in LOCATION_KEYWORDS:
        if fuzzy_in(kw, header_norm) or contains_location_keyword(header_norm, LOCATION_KEYWORDS):
            return True
    # Also match common abbreviations and variants
    if header_norm in LOCATION_ABBREVIATIONS:
        return True
    return False
