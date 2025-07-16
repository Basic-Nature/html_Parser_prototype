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

import os
import orjson
import re
import unicodedata
import glob
import re
import string
import difflib
from difflib import SequenceMatcher
from collections import Counter
from typing import List, Dict, Any, Tuple, TYPE_CHECKING
import time
from bs4 import BeautifulSoup
import hashlib
from ..utils.shared_logger import SharedLogger
from ..utils.ml_table_detector import detect_tables_ml
from ..config import CACHE_DIR, PROJECT_ROOT, LOG_DIR
from difflib import get_close_matches
from ..bots.librarian import (
    LOCATION_KEYWORDS,
    PERCENT_KEYWORDS,
    BALLOT_TYPES,
    BALLOT_TYPES_SORT_ORDER,
    CANDIDATE_KEYWORDS,
    TOTAL_KEYWORDS,
    MISC_FOOTER_KEYWORDS,
    PARTY_KEYWORDS,
    LOCATION_ABBREVIATIONS,
    KNOWN_COUNTY_TO_PRECINCTS_MAP,
    normalize_segment_text
)
if TYPE_CHECKING:
    from ..Context_Integration.context_coordinator import ContextCoordinator
# --- CONSTANTS & GLOBALS ---
logger = SharedLogger()
TABLE_STRUCTURE_CACHE_PATH = os.path.join(CACHE_DIR, "table_structure_cache.json")

def get_safe_log_path(filename):
    """
    Returns a safe log path inside the PROJECT_ROOT/log directory.
    Prevents path-injection and directory traversal.
    """
    log_dir = LOG_DIR
    os.makedirs(log_dir, exist_ok=True)
    safe_filename = os.path.basename(filename)
    return os.path.join(log_dir, safe_filename)

context_cache = {}

# ===================================================================
# MAIN EXTRACTION ENTRY POINT
# ===================================================================

def robust_table_extraction(page, extraction_context=None, existing_headers=None, existing_data=None):
    """
    Unified, persistent table extraction pipeline with robust location detection and forced wide format.
    Now supports ML-driven context, segments, and panels from html_scanner.
    """
    import types
    if coordinator is None:
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
        domain = extraction_context.get("domain") or extraction_context.get("url")
    cached_headers, cached_data = None, None
    if domain:
        # Try to get cached structure for this domain and headers
        # If existing_headers provided, use those for signature, else try after extraction
        if existing_headers:
            cached = get_cached_table_structure(domain, existing_headers)
            if cached:
                logger.info(f"[TABLE BUILDER] Using cached table structure for domain: {domain}")
                return cached.get("headers", []), cached.get("data", [])

    # --- ML context integration ---
    ml_confidence = extraction_context.get("ml_confidence", []) if extraction_context else None
    association_log = extraction_context.get("association_log", []) if extraction_context else None
    segments = extraction_context.get("segments", []) if extraction_context else None
    panels = extraction_context.get("panels", []) if extraction_context else None

    # 1. DOM structure extraction (divs, lists, etc.)
    try:
        headers_dom, data_dom, diagnostics_dom = extract_rows_and_headers_from_dom(
            page,
            coordinator=extraction_context.get("coordinator", []) if extraction_context else None,
            context=extraction_context
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
        tables = page.locator("table")
        for i in range(tables.count()):
            table = tables.nth(i)
            if table is not None:
                headers_tab, data_tab, diagnostics_tab = extract_table_data(
                    table,
                    coordinator=extraction_context.get("coordinator", []) if extraction_context else None,
                    structure_info={"context": extraction_context} if extraction_context else None
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
            page, coordinator=extraction_context.get("coordinator", []) if extraction_context else None, context=extraction_context
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

    # 9. Robust HTML fallback using BeautifulSoup
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
        sig = tuple(normalize_header_name(h) for h in headers)
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
        coordinator = extraction_context.get("coordinator", []) if extraction_context else None
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
    if "fully reported" in heading.lower():
        return "100%"
    return ""

def extract_percent_reported_from_page(page):
    """Try to extract percent reported from the page outside the table."""
    # Look for common phrases in spans/divs
    for selector in ["span", "div", "p"]:
        elements = page.locator(selector)
        for i in range(elements.count()):
            text = elements.nth(i).inner_text().strip()
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
    from ..utils.dynamic_table_extractor import (
        find_tables_with_panel_headings,
        find_tables_with_section_headings,
    )
    if coordinator is None:
        coordinator = ContextCoordinator()
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
                # Use heading as the location value
                if "Precinct" not in headers:
                    headers = ["Precinct"] + headers
                for row in data:
                    row["Precinct"] = heading
                location_col = "Precinct"
            else:
                # Rename location_col to "Precinct" if needed
                if location_col != "Precinct":
                    headers = ["Precinct" if h == location_col else h for h in headers]
                    for row in data:
                        row["Precinct"] = row.pop(location_col)
                # Fill missing values with heading
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
        candidate_cols = [h for h in all_panel_headers if any(k in h.lower() for k in CANDIDATE_KEYWORDS)]
        ballot_types_cols = [h for h in all_panel_headers if any(bt in h.lower() for bt in BALLOT_TYPES)]
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
            # Average ML score for headers
            scores = [coordinator.score_header(h, {}) for h in result["headers"]]
            score = sum(scores) / len(scores) if scores else 0
        # Bonus for more rows and columns
        score += 0.1 * min(len(result["data"]) / 10.0, 1.0)
        score += 0.1 * min(len(result["headers"]) / 8.0, 1.0)
        result["score"] = score

    # --- Try to merge/patch missing information between extraction types ---
    # If one extraction is missing a location or percent column, but the other has it, fill in
    def patch_missing_info(primary, secondary):
        patched = False
        sec_headers = set(secondary["headers"])
        for h in secondary["headers"]:
            if h not in primary["headers"]:
                # Add missing header and fill with values if possible
                primary["headers"].append(h)
                for i, row in enumerate(primary["data"]):
                    # Try to match by row index (could be improved with NLP/ML row association)
                    if i < len(secondary["data"]):
                        row[h] = secondary["data"][i].get(h, "")
                    else:
                        row[h] = ""
                patched = True
        return patched

    # Pick the best extraction by score, but patch with info from the other if possible
    extraction_results.sort(key=lambda r: r["score"], reverse=True)
    best = extraction_results[0]
    if len(extraction_results) > 1:
        other = extraction_results[1]
        patched = patch_missing_info(best, other)
        # Optionally, use NLP/ML to check if rows are associated (e.g., by location/district/candidate)
        # This can be extended with coordinator.match_rows(row1, row2) if implemented

    # --- Combine all panel tables if more than one ---
    all_tables = [(r["headers"], r["data"]) for r in extraction_results if r["headers"] and r["data"]]
    if len(all_tables) > 1:
        headers, data = combine_panel_tables_by_precinct(all_tables)
        # Optionally, merge entity previews as well
        entity_previews = []
        for r in extraction_results:
            entity_previews.extend(r.get("entity_previews", []))
        return headers, data, entity_previews
    return best["headers"], best["data"], best["entity_previews"]

def extract_table_data(table, coordinator=None, structure_info=None) -> Tuple[List[str], List[Dict[str, Any]], dict]:
    """
    Extracts headers and data from a Playwright table locator.
    Uses advanced NLP/NER, ML scoring, fuzzy and value-based matching to robustly detect entity columns.
    Improves detection for location and percent reported columns.
    Returns headers, data, and a meta dict with entity preview and detected location/percent columns.
    Now walks the DOM for best-matching columns and values, scoring all candidates and picking the best.
    """
    if coordinator is None:
        coordinator = ContextCoordinator()
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
        header_cells = table.locator("thead tr th")
        if header_cells.count() == 0:
            first_row = table.locator("tr").first
            header_cells = first_row.locator("th, td")
        for i in range(header_cells.count()):
            text = header_cells.nth(i).inner_text().strip()
            headers.append(text if text else f"Column {i+1}")

        # --- Extract rows ---
        rows = table.locator("tbody tr")
        if rows.count() == 0:
            all_rows = table.locator("tr")
            rows = all_rows

        for i in range(rows.count()):
            row = {}
            cells = rows.nth(i).locator("td, th")
            if cells.count() == 0:
                continue
            for j in range(cells.count()):
                if j < len(headers):
                    row[headers[j]] = cells.nth(j).inner_text().strip()
                else:
                    row[f"Extra_{j+1}"] = cells.nth(j).inner_text().strip()
            if any(v for v in row.values()):
                data.append(row)

        # --- After extracting headers and data ---
        # If no location column, but context provides one, inject it
        context = structure_info.get("context", []) if structure_info else {}
        panel_heading = context.get("panel_heading", []) or context.get("Precinct", []) or context.get("district", [])
        location_col = entity_preview.get("location_column", []) or (
            next((h for h in headers if is_location_header(h)), None)
        ) or "Precinct"
        if location_col and location_col != "Precinct":
            headers = ["Precinct" if h == location_col else h for h in headers]
            for row in data:
                row["Precinct"] = row.pop(location_col)
            location_col = "Precinct"
        # --- PATCH: Inject location if missing ---
        if not location_col and panel_heading:           
            for row in data:
                row["Precinct"] = panel_heading
            if "Precinct" not in headers:
                headers = ["Precinct"] + headers
            location_col = "Precinct"
        # --- PATCH: If only one unique location, synthesize from context ---
        unique_locations = sorted(
            set(str(row.get(location_col, "") or "") for row in data if row.get(location_col, ""))
        )
        unique_candidates = sorted(set(row.get("Candidate", "") for row in data if row.get("Candidate", "")))
        n_candidates = len(entity_preview.get("candidates") or [])
        n_ballot_types = len(entity_preview.get("ballot_types") or [])
        n_numbers = len(entity_preview.get("numbers") or [])
        n_locations = len(unique_locations)
        loc_col_disp = location_col if location_col else "N/A"
        pct_col_disp = entity_preview.get("percent_column") or "N/A"
        
        if location_col and len(unique_locations) <= 1 and panel_heading:
            for row in data:
                row[location_col] = panel_heading             
        county = context.get("county", "").lower() if context else ""
        known_districts = set()
        if coordinator and hasattr(coordinator, "library"):
            county_map = KNOWN_COUNTY_TO_PRECINCTS_MAP
            if county and county_map.get(county.title(), []):
                known_districts = set(d.lower() for d in county_map[county.title()])

        # --- Robust Location & Percent Detection: Score all candidates, don't stop at first ---
        location_candidates = []
        percent_candidates = []
        percent_patterns = set(PERCENT_KEYWORDS)

        # 1. Score headers using ML/NLP/NER and heuristics
        for h in headers:
            score = 0
            if coordinator:
                ents = []
                if coordinator and hasattr(coordinator, "extract_entities"):
                    ents = coordinator.extract_entities(h)
                for ents, label in ents:
                    if label in {"GPE", "LOC", "FAC"} and h.lower() != "candidate":
                        score += 1.0
                if is_location_header(h) and h.lower() != "candidate":
                    score += coordinator.score_header(h, {}) if hasattr(coordinator, "score_header") else 0.5
            if is_location_header(h) and h.lower() != "candidate":
                score += 0.3
            # Value-based: check if values match known districts
            if known_districts:
                col_vals = [str(row.get(h, "")).lower() for row in data]
                match_count = sum(
                    1 for v in col_vals
                    if v in known_districts or difflib.get_close_matches(v, known_districts, n=1, cutoff=0.8)
                )
                if match_count / max(1, len(col_vals)) > 0.5:
                    score += 0.7
            # Uniqueness/entropy: high unique values, not all numeric
            col_vals = [str(row.get(h, "")) for row in data]
            unique_vals = len(set(col_vals))
            if unique_vals > 3 and not all(v.replace(",", "").isdigit() for v in col_vals if v):
                score += 0.2
            if score > 0:
                location_candidates.append((h, score))

            # Percent detection
            pscore = 0
            if any(kw in h.lower() for kw in percent_patterns):
                pscore += 1.0
            elif "%" in h:
                pscore += 0.8
            if pscore > 0:
                percent_candidates.append((h, pscore))

        # 2. Walk the DOM for additional clues (scan all cells for location/percent-like values)
        # This is a second pass, not just headers
        dom_location_scores = {}
        dom_percent_scores = {}
        for h in headers:
            col_vals = [str(row.get(h, "")) for row in data]
            # Location: match against known districts or location-like patterns
            if known_districts:
                match_count = sum(
                    1 for v in col_vals
                    if v.lower() in known_districts or difflib.get_close_matches(v.lower(), known_districts, n=1, cutoff=0.8)
                )
                dom_location_scores[h] = match_count / max(1, len(col_vals))
            # Percent: look for % in values
            percent_count = sum(1 for v in col_vals if "%" in v)
            dom_percent_scores[h] = percent_count / max(1, len(col_vals))
        # Add to candidates if above threshold
        for h, v in dom_location_scores.items():
            if v > 0.5:
                location_candidates.append((h, 0.5 + v))
        for h, v in dom_percent_scores.items():
            if v > 0.5:
                percent_candidates.append((h, 0.5 + v))

        # 3. Score and pick the best (highest score) for each, require threshold
        location_candidates = sorted(location_candidates, key=lambda x: x[1], reverse=True)
        percent_candidates = sorted(percent_candidates, key=lambda x: x[1], reverse=True)
        location_col = location_candidates[0][0] if location_candidates and location_candidates[0][1] > 0.7 else None
        percent_col = percent_candidates[0][0] if percent_candidates and percent_candidates[0][1] > 0.7 else None

        entity_preview["location_column"] = location_col
        entity_preview["percent_column"] = percent_col

        # --- Scan data for entity types ---
        ballot_types_keywords = set(bt.lower() for bt in BALLOT_TYPES)
        number_pattern = re.compile(r"^-?\d{1,3}(?:,\d{3})*(?:\.\d+)?%?$")
        for row in data:
            for h, v in row.items():
                if not v:
                    continue
                # Candidate detection (robust)
                if any(ck in h.lower() for ck in CANDIDATE_KEYWORDS):
                    entity_preview["candidates"].add(v)
                # Ballot type detection (robust)
                if any(bk in h.lower() for bk in ballot_types_keywords):
                    entity_preview["ballot_types"].add(h)
                # Number detection (improved)
                if number_pattern.match(v.replace(",", "")):
                    entity_preview["numbers"].add(v)
                # Location detection (only if a valid location_col was found)
                if location_col and h == location_col:
                    entity_preview["locations"].add(v)

        # --- Automated feedback/learning: log if location_col is missing or suspect ---
        if not location_col or len(entity_preview["locations"]) == 0:
            logger.warning(f"[TABLE BUILDER][extract_table_data] No valid location column {location_col} or values detected {entity_preview['locations']}. Consider user/ML feedback.")

        # --- Percent Reported Value Extraction ---
        percent_value = ""
        if percent_col:
            # Try to extract a percent value from the first row
            for row in data:
                val = row.get(percent_col, "")
                if val and "%" in val:
                    percent_value = val
                    break
        if not percent_value:
            # Try to extract from context or fallback
            if context and "percent_reported" in context:
                percent_value = context["percent_reported"]
            else:
                # Try to extract from any cell value
                for row in data:
                    for v in row.values():
                        if isinstance(v, str) and "%" in v:
                            percent_value = v
                            break
                    if percent_value:
                        break
        # Optionally, fill percent_col in all rows if found
        if percent_col and percent_value:
            for row in data:
                if not row.get(percent_col, []):
                    row[percent_col] = percent_value

        # Log NLP-style preview
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
        # Optional: show a compact table preview of the first 2 rows
        if data:
            from rich.table import Table as RichTable
            preview_table = RichTable(show_header=True, header_style="bold magenta")
            for h in headers:
                preview_table.add_column(h)
            for row in data[:2]:
                preview_table.add_row(*(str(row.get(h, "")) for h in headers))
            logger.alert(preview_table)
        # If not headers and data, fallback to generic headers
        if not headers and data:
            max_cols = max(len(row) for row in data)
            headers = [f"Column {i+1}" for i in range(max_cols)]
            logger.warning("[TABLE BUILDER][extract_table_data] No headers but there is data. Generating generic headers.")
            new_data = []
            for row in data:
                new_row = {}
                for idx, h in enumerate(headers):
                    new_row[h] = list(row.values())[idx] if idx < len(row) else ""
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
    if known_keywords is None:
        known_keywords = ["candidate", "votes", "party", "precinct", "choice", "option", "response", "total"]
    cells = row.locator("> *")
    headers = []
    cell_texts = []
    for i in range(cells.count()):
        text = cells.nth(i).inner_text().strip().lower()
        cell_texts.append(text)
        header = None
        for kw in known_keywords:
            if kw in text:
                header = kw.capitalize()
                break
        if not header:
            header = f"Column {i+1}"
        headers.append(header)
    diagnostics["cell_texts"] = cell_texts
    diagnostics["headers"] = headers
    return headers, diagnostics

def extract_rows_and_headers_from_dom(page, extra_keywords=None, min_row_count=2, coordinator=None, context=None):
    """
    Attempts to extract tabular data from repeated DOM structures (divs, etc.).
    Returns headers, data, and diagnostics.
    Enhanced: logs and returns what is being removed, and column stats.
    """
    if coordinator is None:
        coordinator = ContextCoordinator()
    logger.info("[TABLE_BUILDER][extract_rows_and_headers_from_dom] Starting DOM structure extraction.")
    repeated_rows = extract_repeated_dom_structures(page, extra_keywords=extra_keywords, min_row_count=min_row_count)
    logger.info(f"[TABLE_BUILDER][extract_rows_and_headers_from_dom] Found {len(repeated_rows)} repeated rows.")
    if not repeated_rows:
        logger.warning("[TABLE_BUILDER][extract_rows_and_headers_from_dom] No repeated rows found.")
        return [], [], {"diagnostics": "No repeated rows found."}

    # --- Heuristic header detection block ---
    headers = None
    header_row_idx = None
    for idx, (heading, row) in enumerate(repeated_rows[:10]):
        if row is None:
            logger.warning(f"[TABLE_BUILDER][extract_rows_and_headers_from_dom] Row locator is None at index {idx}. Skipping.")
            continue
        cells = row.locator("> *")
        cell_texts = [cells.nth(i).inner_text().strip() for i in range(cells.count())]
        logger.info(f"[TABLE_BUILDER][extract_rows_and_headers_from_dom] Checking row {idx} for headers: {cell_texts}")
        # Heuristic: header row if at least 2 known fields or all non-numeric
        if is_likely_header(cell_texts) or all(not re.match(r"^\d+([,.]\d+)?$", c) for c in cell_texts):
            headers = cell_texts
            header_row_idx = idx
            logger.info(f"[TABLE_BUILDER][extract_rows_and_headers_from_dom] Detected header row at index {idx}: {headers}")
            break
    if headers is not None:
        repeated_rows = repeated_rows[header_row_idx + 1 :]
    else:
        headers, _ = guess_headers_from_row(repeated_rows[0][1], context=context or {})
        logger.info(f"[TABLE_BUILDER][extract_rows_and_headers_from_dom] Guessed headers from first row: {headers}")

    # --- Merge split header rows (e.g., two header rows) ---
    if len(repeated_rows) > 1:
        first_row_cells = [repeated_rows[0][1].locator("> *").nth(i).inner_text().strip() for i in range(repeated_rows[0][1].locator("> *").count())]
        if all(c.isalpha() or c == "" for c in first_row_cells) and any(c for c in first_row_cells):
            logger.info(f"[TABLE_BUILDER][extract_rows_and_headers_from_dom] Merging split header rows: {headers} + {first_row_cells}")
            headers = [" ".join(filter(None, [h, f])) for h, f in zip(headers, first_row_cells)]
            repeated_rows = repeated_rows[1:]

    # --- Sample rows for stats ---
    sample_rows = []
    for heading, row in repeated_rows[:20]:
        cells = row.locator("> *")
        cell_texts = [cells.nth(i).inner_text().strip() for i in range(cells.count())]
        sample_rows.append(cell_texts)
    logger.info(f"[TABLE_BUILDER][extract_rows_and_headers_from_dom] Sample rows for stats: {sample_rows[:3]}")

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
    for heading, row in repeated_rows:
        cells = row.locator("> *")
        cell_values = [cells.nth(i).inner_text().strip() for i in range(cells.count())]
        row_data = {headers[idx]: cell_values[idx] if idx < len(cell_values) else "" for idx in range(len(headers))}
        all_panel_rows.append(row_data)

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
        "all_panel_rows": all_panel_rows,
        "removed_footer_rows": removed_footer_rows,
        "removed_empty_rows": removed_empty_rows,
        "col_stats": col_stats,
        "sample_rows": sample_rows,
        "final_row_count": len(final_data),
        "final_col_count": len(headers),
    }

    logger.info(f"[TABLE_BUILDER][extract_rows_and_headers_from_dom] Finished: {len(final_data)} rows, {len(headers)} columns.")
    return headers, final_data, diagnostics

def extract_with_patterns(page, context=None, log_path=None):
    """
    Attempts to extract tabular data using approved DOM patterns.
    Returns (headers, data, diagnostics)
    """
    patterns = load_dom_patterns(log_path)
    approved = [p for p in patterns if p.get("approved", [])]
    results = []
    diagnostics = {
        "patterns_tried": len(patterns),
        "patterns_approved": len(approved),
        "matches": [],
    }
    for pat in approved:
        selector = pat["selector"]
        cell_selectors = pat.get("cell_selectors", []) or [pat.get("cell_selector", "> *")]
        containers = page.locator(selector)
        for i in range(containers.count()):
            container = containers.nth(i)
            heading = pat.get("heading", []) or f"Pattern: {selector} #{i+1}"
            for cell_selector in cell_selectors:
                children = container.locator(cell_selector)
                if children.count() > 0:
                    for j in range(children.count()):
                        row = children.nth(j)
                        if "row_tag" in pat:
                            tag = row.evaluate("el => el.tagName.toLowerCase()")
                            if tag != pat["row_tag"]:
                                continue
                        if "row_class" in pat:
                            classes = row.evaluate("el => el.className")
                            if pat["row_class"] not in classes:
                                continue
                        if "row_text_contains" in pat:
                            text = row.inner_text().strip()
                            if pat["row_text_contains"] not in text:
                                continue
                        if row is not None:
                            results.append((heading, row, pat))
                            diagnostics["matches"].append({
                                "heading": heading,
                                "selector": selector,
                                "cell_selector": cell_selector,
                                "row_index": j
                            })
    # Build headers/data if any matches
    if results:
        headers, _ = guess_headers_from_row(results[0][1], context=context)
        data = []
        for heading, row, pat in results:
            cells = row.locator("> *")
            row_data = {}
            for idx in range(cells.count()):
                row_data[headers[idx] if idx < len(headers) else f"Column {idx+1}"] = cells.nth(idx).inner_text().strip()
            if row_data:
                data.append(row_data)
        return headers, data, diagnostics
    return [], [], diagnostics

def fallback_nlp_candidate_vote_scan(page):
    """
    Improved fallback: scan for elements with candidate-like, party-like, or location-like names and vote-like numbers nearby.
    Returns headers, data.
    """
    import re
    # Accept more flexible candidate/location/party patterns
    label_pattern = re.compile(r"^[A-Za-z][A-Za-z\s\-\']{1,40}$")
    vote_pattern = re.compile(r"^\d{1,3}(,\d{3})*$")
    skip_phrases = [
        "Last Updated", "Vote Method", "Fully Reported", "Search", "Reported", "Total", "Precincts Reporting"
    ]
    elements = page.locator("*")
    labels = []
    votes = []
    for i in range(elements.count()):
        text = elements.nth(i).inner_text().strip()
        if not text or len(text) < 2:
            continue
        if any(skip in text for skip in skip_phrases):
            continue
        if vote_pattern.fullmatch(text.replace(",", "")):
            votes.append((i, text))
        elif label_pattern.match(text):
            labels.append((i, text))
    # Pair each vote with the closest preceding label
    data = []
    for vote_idx, vote_val in votes:
        # Find the closest label before this vote
        label = None
        for idx, lbl in reversed(labels):
            if idx < vote_idx:
                label = lbl
                break
        if label is not None:
            data.append({"Label": label, "Votes": vote_val})
    headers = ["Label", "Votes"]
    logger.info(f"[TABLE BUILDER] Robust NLP fallback: {len(data)} rows, {len(headers)} columns.")
    return headers, data

def extract_repeated_dom_structures(page, container_selectors=None, min_row_count=2, extra_keywords=None):
    """
    Scans the DOM for repeated structures (divs, uls, etc.) that look like tabular data.
    Returns a list of (section_heading, row_locator) tuples.
    Dynamically updates likely_row_classes from log analysis.
    """
    # --- Dynamically update likely_row_classes from logs ---
    log_dir = LOG_DIR
    suggested_classes, suggested_ids = suggest_new_row_classes_from_logs(log_dir)
    likely_row_classes = [
        "row", "table-row", "ballot-option", "candidate-info", "result-row", "precinct-row"
    ] + suggested_classes
    likely_row_ids = suggested_ids

    if container_selectors is None:
        selectors = [f"div.{cls}" for cls in likely_row_classes]
        selectors += [f"div#{id_}" for id_ in likely_row_ids]
        selectors += ["ul > li", "ol > li"]
    else:
        selectors = container_selectors

    results = []
    MAX_CONTAINERS = 100
    for selector in selectors:
        containers = page.locator(selector)
        for i in range(min(containers.count(), MAX_CONTAINERS)):
            try:
                container = containers.nth(i)
                children = container.locator("> *")
                if children.count() >= min_row_count:
                    # Try to find a heading above the container
                    heading = ""
                    heading_loc = container.locator("xpath=preceding-sibling::*[self::h1 or self::h2 or self::h3 or self::h4 or self::h5 or self::h6][1]")
                    if heading_loc.count() > 0:
                        heading = heading_loc.nth(0).inner_text().strip()
                    else:
                        heading = f"Section {i+1}"
                    for j in range(children.count()):
                        row = children.nth(j)
                        if row is not None:
                            results.append((heading, row))
            except Exception as e:
                log_failed_container(page, container, selector, i, str(e))
    return results

def extract_all_candidates_from_data(headers, data, extraction_context=None):
    """
    Extract all unique candidate names from the data, using the provided headers and context.
    Optionally uses extraction_context for more robust candidate column detection.
    """
    candidates = set()
    # Try to find the candidate column robustly
    candidate_col = None
    # 1. Use context if available
    if extraction_context and "candidate_column" in extraction_context:
        candidate_col = extraction_context["candidate_column"]
    # 2. Fallback: look for best header match
    if not candidate_col:
        for h in headers:
            if any(ck in h.lower() for ck in CANDIDATE_KEYWORDS):
                candidate_col = h
                break
    # 3. Fallback: use "Candidate" if present
    if not candidate_col and "Candidate" in headers:
        candidate_col = "Candidate"
    # 4. If still not found, skip extraction
    if not candidate_col:
        logger.warning("[extract_all_candidates_from_data] No candidate column found in headers or context.")
        return candidates

    for row in data:
        val = row.get(candidate_col, "")
        for part in val.split("\n"):
            part = part.strip()
            # Filter out party-only or generic lines
            if part and not any(part.lower().startswith(pk) for pk in PARTY_KEYWORDS):
                candidates.add(part)
    return candidates
# 1. ML-based table detection (e.g., using a model to find tables in arbitrary HTML)
def ml_based_table_detection(page, extraction_context=None):
    """
    Use a machine learning model to detect and extract tables from arbitrary HTML.
    Returns a list of (headers, data, diagnostics) tuples.
    Each diagnostics dict includes the extraction_context for traceability.
    """
    if coordinator is None:
        coordinator = ContextCoordinator()
    try:
        ml_tables = detect_tables_ml(page.content())
        results = []
        for idx, table_dict in enumerate(ml_tables):
            headers = table_dict.get("headers", [])
            data = table_dict.get("data", [])
            # Optionally, correlate context to this table (if available)
            context = extraction_context if extraction_context else {}
            if coordinator and hasattr(coordinator, "get_for_table_builder"):
                context = coordinator.get_for_table_builder()
            diagnostics = {
                "ml_table_index": idx,
                "row_count": len(data),
                "headers": headers,
                "extraction_context": context
            }
            # Optionally, attach context to each row for downstream traceability
            if headers and data:
                # Optionally, add context to each row (comment out if not needed)
                # for row in data:
                #     row["_extraction_context"] = context
                results.append((headers, data, diagnostics))
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
        tables = page.locator("table table")
        for i in range(tables.count()):
            table = tables.nth(i)
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

# 3. Robust HTML fallback using BeautifulSoup (see robust_html_fallback)
def robust_html_fallback_extraction(page):
    """
    Use BeautifulSoup to parse HTML and extract tables as a last-resort fallback.
    Returns a list of (headers, data, diagnostics) tuples.
    """
    try:
        html = page.content()
        soup = BeautifulSoup(html, "html.parser")
        tables = soup.find_all("table")
        all_tables = []
        for idx, table in enumerate(tables):
            rows = table.find_all("tr")
            if not rows:
                continue
            headers = [th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])]
            data = []
            for row in rows[1:]:
                cells = row.find_all(["td", "th"])
                data.append({headers[i]: cells[i].get_text(strip=True) if i < len(cells) else "" for i in range(len(headers))})
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
        plugins = extraction_context.get("plugins", []) if extraction_context else []
        results = []
        for idx, plugin in enumerate(plugins):
            try:
                plugin_result = plugin.extract(page, extraction_context)
                if plugin_result:
                    for headers, data in plugin_result:
                        diagnostics = {
                            "plugin_index": idx,
                            "plugin_name": getattr(plugin, "__name__", str(plugin)),
                            "row_count": len(data)
                        }
                        if headers and data:
                            results.append((headers, data, diagnostics))
            except Exception as e:
                logger.error(f"[PLUGIN EXTRACTION] Plugin {plugin}: {e}")
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
        if extraction_context and extraction_context.get("interactive", []):
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
    from urllib.parse import urlparse
    if allowed_domains is None:
        allowed_domains = {"yourdomain.com"}
    try:
        parsed = urlparse(user_url)
        if parsed.scheme not in {"http", "https"}:
            return "/"
        if parsed.netloc and parsed.netloc not in allowed_domains:
            return "/"
        # Optionally, further sanitize the path
        return parsed.geturl()
    except Exception:
        return "/"

def find_best_header(headers, keywords):
    """Find the best matching header from a set of keywords (case-insensitive, fuzzy)."""
    headers_lower = [h.lower() for h in headers]
    # Try substring match for any keyword
    for kw in keywords:
        for i, h in enumerate(headers_lower):
            if kw in h:
                return headers[i]
    # Fuzzy match if no substring match
    for kw in keywords:
        matches = get_close_matches(kw, headers_lower, n=1, cutoff=0.7)
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
        all_headers.update(row.keys())

    # 2. Ensure 'Percent Reported' is present if in any row or context
    percent_val = None
    if any("Percent Reported" in row for row in data):
        all_headers.add("Percent Reported")
        for row in data:
            if row.get("Percent Reported", []):
                percent_val = row["Percent Reported"]
                break
    if context and context.get("percent_reported", []):
        all_headers.add("Percent Reported")
        percent_val = context["percent_reported"]

    # 3. Build ordered headers: preserve input order, then add new ones
    seen = set()
    ordered_headers = [h for h in headers if h in all_headers and not (h in seen or seen.add(h))]
    ordered_headers += [h for h in all_headers if h not in seen and not seen.add(h)]

    # 4. Normalize location column to "Precinct"
    location_col = next((h for h in ordered_headers if is_location_header(h)), None)
    if location_col and location_col != "Precinct":
        ordered_headers = ["Precinct" if h == location_col else h for h in ordered_headers]
        for row in data:
            row["Precinct"] = row.pop(location_col)
        location_col = "Precinct"

    # 5. Identify candidate and ballot type columns
    candidate_col = next((h for h in ordered_headers if any(ck in h.lower() for ck in CANDIDATE_KEYWORDS)), None)
    ballot_types_cols = [h for h in ordered_headers if any(bt.lower() in h.lower() for bt in BALLOT_TYPES)]

    # 6. Deduplicate rows using composite key (location, candidate, ballot types)
    harmonized = []
    seen_keys = set()
    for row in data:
        full_row = {h: row.get(h, "") for h in ordered_headers}
        # Fill missing Percent Reported from context if needed
        if "Percent Reported" in ordered_headers and not full_row.get("Percent Reported", []) and percent_val:
            full_row["Percent Reported"] = percent_val
        # Deduplication key
        if location_col and candidate_col and full_row.get(location_col, []) and full_row.get(candidate_col, []):
            key = (
                full_row.get(location_col, ""),
                full_row.get(candidate_col, ""),
                *(full_row.get(bt, "") for bt in ballot_types_cols)
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
        harmonized.append(full_row)

    # 7. Remove columns that are all empty or zero, but always keep columns present in input headers
    keep = [h for h in ordered_headers if (h in headers) or any(row.get(h, "") not in ("", "0") for row in harmonized)]
    if not keep and ordered_headers:
        keep = ordered_headers
    harmonized = [{h: row.get(h, "") for h in keep} for row in harmonized]

    # 8. Log unique locations and warn if only one unique value
    unique_locations = set(row.get(location_col, "") for row in harmonized if location_col and location_col in row)
    logger.info(f"[HARMONIZE] Unique values in location column '{location_col}': {sorted(unique_locations)}")
    logger.info(f"[HARMONIZE] Unique values in location column '{location_col}': {sorted(unique_locations)}")
    if location_col and len(unique_locations) <= 1:
        logger.warning(f"[HARMONIZE] WARNING: Only one unique value found in location column '{location_col}'. Extraction may be incorrect.")

    # 9. Reorder columns: Precinct, candidates, ballot types, then others
    candidate_cols = [h for h in keep if any(k in h.lower() for k in CANDIDATE_KEYWORDS)]
    ballot_types_cols = [h for h in keep if any(bt in h.lower() for bt in BALLOT_TYPES)]
    ordered_final = []
    if "Precinct" in keep:
        ordered_final.append("Precinct")
    ordered_final += sorted(set(candidate_cols + ballot_types_cols))
    ordered_final += [h for h in keep if h not in candidate_cols + ballot_types_cols + ["Precinct"]]
    # Remove duplicates while preserving order
    seen_final = set()
    ordered_final = [h for h in ordered_final if not (h in seen_final or seen_final.add(h))]

    # 10. Return final headers and harmonized data
    return ordered_final, [{h: row.get(h, "") for h in ordered_final} for row in harmonized]

def deduplicate_headers(headers, data):
    """Remove duplicate headers by normalized name, keep first occurrence."""
    seen = set()
    new_headers = []
    for h in headers:
        norm = normalize_header_name(h)
        if norm not in seen:
            new_headers.append(h)
            seen.add(norm)
    new_data = [{h: row.get(h, "") for h in new_headers} for row in data]
    return new_headers, new_data

def remove_low_signal_columns(headers, data, min_unique=2, min_non_empty_ratio=0.05):
    """
    Remove columns with low variance or too many repeated values.
    """
    keep = []
    n_rows = len(data)
    for h in headers:
        col_vals = [row.get(h, "") for row in data]
        unique_vals = set(col_vals)
        non_empty = [v for v in col_vals if v not in ("", None)]
        if len(unique_vals) >= min_unique and len(non_empty) / n_rows >= min_non_empty_ratio:
            keep.append(h)
    return keep, [{h: row.get(h, "") for h in keep} for row in data]

def merge_table_data(headers_list, data_list):
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
                if all(k in row and k in mrow and row[k] == mrow[k] and row[k] for k in keys if k in row):
                    match = mrow
                    break
            if match:
                for h in all_headers:
                    if not match.get(h, []) and row.get(h, []):
                        match[h] = row[h]
            else:
                merged_data.append(row)
    # Only harmonize once at the end
    return harmonize_headers_and_data(all_headers, merged_data)

def merge_multiline_candidate_rows(headers, data):
    """
    Merge rows where candidate name and party are split across two rows or within a cell.
    Ensures 'Precinct' and 'Percent Reported' columns are preserved and consistent.
    - If a candidate cell contains a newline, split into candidate and party.
    - If the next row is just a party, merge it.
    - Always ensure 'Precinct' and 'Percent Reported' columns are present if found in any row.
    """
    import re

    # --- Detect if we have these columns in headers or data ---
    has_precinct = "Precinct" in headers or any("Precinct" in row for row in data)
    has_percent_reported = "Percent Reported" in headers or any("Percent Reported" in row for row in data)
    # If not in headers but present in data, add to headers
    if not has_precinct and any("Precinct" in row for row in data):
        headers.append("Precinct")
        has_precinct = True
    if not has_percent_reported and any("Percent Reported" in row for row in data):
        headers.append("Percent Reported")
        has_percent_reported = True

    # --- Main merge logic ---
    if "Candidate" not in headers:
        return headers, data
    merged_data = []
    i = 0
    while i < len(data):
        row = data[i]
        candidate_val = row.get("Candidate", "")
        # Preserve Precinct and Percent Reported if present
        precinct_val = row.get("Precinct", "")
        percent_reported_val = row.get("Percent Reported", "")
        party_abbrevs = ["DEM", "REP", "CON", "WOR", "IND", "GRN", "LIB", "Other", "Write-in"]
        # Try to match pattern: [ABBR] Name[PartyName]
        match = re.match(r"^([A-Z]{2,4})\s+(.+?)([A-Z][a-z]+)$", candidate_val)
        if match:
            abbr, name, party = match.groups()
            row["Candidate"] = f"{abbr} {name}".strip()
            row["Party"] = party
            if has_precinct:
                row["Precinct"] = precinct_val
            if has_percent_reported:
                row["Percent Reported"] = percent_reported_val
            merged_data.append(row)
            i += 1
            continue
        # PATCH: Try to split by known party abbreviations at start
        for abbr in party_abbrevs:
            if candidate_val.startswith(abbr + " "):
                rest = candidate_val[len(abbr):].strip()
                # Try to split at the last uppercase word (party)
                m = re.match(r"(.+?)([A-Z][a-z]+)$", rest)
                if m:
                    name, party = m.groups()
                    row["Candidate"] = f"{abbr} {name}".strip()
                    row["Party"] = party
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
                parts = [p.strip() for p in candidate_val.split("\n") if p.strip()]
                if len(parts) == 2:
                    row["Candidate"], row["Party"] = parts
                elif len(parts) > 2:
                    row["Candidate"] = parts[0]
                    row["Party"] = " ".join(parts[1:])
                else:
                    row["Candidate"] = candidate_val.replace("\n", " ")
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
                next_candidate_val = next_row.get("Candidate", "")
                # Only merge if all other columns are empty in next row
                non_candidate_cols = [k for k in next_row if k != "Candidate" and next_row[k]]
                if next_candidate_val and not non_candidate_cols:
                    row["Party"] = next_candidate_val
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
    if any("Party" in row for row in merged_data) and "Party" not in headers:
        headers.append("Party")
    # Ensure 'Precinct' and 'Percent Reported' in headers if present in any row
    if any("Precinct" in row for row in merged_data) and "Precinct" not in headers:
        headers.append("Precinct")
    if any("Percent Reported" in row for row in merged_data) and "Percent Reported" not in headers:
        headers.append("Percent Reported")
    return headers, merged_data

def combine_panel_tables_by_precinct(all_tables):
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
    """
    if coordinator is None:
        coordinator = ContextCoordinator()
    logger.info("[TABLE_CORE][nlp_entity_annotate_table] Starting NLP entity annotation.")
    if not coordinator:
        logger.warning("[TABLE_CORE][nlp_entity_annotate_table] No coordinator provided, skipping NLP annotation.")
        return headers, data, {}

    # ML context integration
    ml_confidence = context.get("ml_confidence", []) if context else None
    association_log = context.get("association_log", []) if context else None
    segments = context.get("segments", []) if context else None
    panels = context.get("panels", []) if context else None

    entity_info = {
        "people": set(),
        "locations": set(),
        "ballot_types": set(),
        "numbers": set(),
        "row_entities": [],
        "ml_confidence": ml_confidence,
        "association_log": association_log,
    }
    # Optionally: log ML context for debugging
    if ml_confidence is not None:
        logger.info(f"[TABLE_CORE][nlp_entity_annotate_table] ML confidence: {ml_confidence:.2f}")
    if association_log:
        logger.info(f"[TABLE_CORE][nlp_entity_annotate_table] Association log: {association_log}")

    # Analyze headers for entity types
    header_entities = {}
    for h in headers:
        ents = coordinator.extract_entities(h)
        header_entities[h] = ents
        for ent, label in ents:
            if label == "PERSON":
                entity_info["people"].add(ent)
            elif label in {"GPE", "LOC", "FAC"}:
                entity_info["locations"].add(ent)
            elif any(bt.lower() in h.lower() for bt in BALLOT_TYPES):
                entity_info["ballot_types"].add(h)
    # Analyze each row for entities
    annotated_data = []
    for row in data:
        row_ents = {"people": set(), "locations": set(), "ballot_types": set(), "numbers": set()}
        for field in ["Candidate", "Party"]:
            val = row.get(field, "")
            if val:
                ents = coordinator.extract_entities(val)
                for ent, label in ents:
                    if label == "PERSON":
                        row_ents["people"].add(ent)
                        entity_info["people"].add(ent)
                    elif label in {"GPE", "LOC", "FAC"}:
                        row_ents["locations"].add(ent)
                        entity_info["locations"].add(ent)        
        for h in headers:
            val = row.get(h, "")
            if not val:
                continue
            ents = coordinator.extract_entities(val)
            for ent, label in ents:
                if label == "PERSON":
                    row_ents["people"].add(ent)
                    entity_info["people"].add(ent)
                elif label in {"GPE", "LOC", "FAC"}:
                    row_ents["locations"].add(ent)
                    entity_info["locations"].add(ent)
            # Ballot type detection
            for bt in BALLOT_TYPES:
                if bt.lower() in h.lower() or bt.lower() in val.lower():
                    row_ents["ballot_types"].add(bt)
                    entity_info["ballot_types"].add(bt)
            # Number detection
            if isinstance(val, str) and val.replace(",", "").replace(".", "").isdigit():
                row_ents["numbers"].add(val)
                entity_info["numbers"].add(val)
        entity_info["row_entities"].append(row_ents)
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
    if coordinator is None:
        coordinator = ContextCoordinator()
    logger.info("[TABLE_CORE][verify_table_structure] Verifying table structure using NLP and DOM info.")
    missing = []
    # Check for location
    has_location = bool(entity_info.get("locations", [])) or any(
        any(lk in h.lower() for lk in LOCATION_KEYWORDS) for h in headers
    )
    if not has_location:
        missing.append("location")
    # Check for candidate/person
    has_candidate = bool(entity_info.get("people", [])) or any(
        coordinator and any(label == "PERSON" for ent, label in coordinator.extract_entities(h)) for h in headers
    )
    if not has_candidate:
        missing.append("candidate")
    # Check for ballot type
    has_ballot_type = bool(entity_info.get("ballot_types", [])) or any(
        any(bt.lower() in h.lower() for bt in BALLOT_TYPES) for h in headers
    )
    if not has_ballot_type:
        missing.append("ballot_type")
    # Check for numbers
    has_numbers = bool(entity_info.get("numbers", [])) or any(
        any(c.isdigit() for c in row.values()) for row in data
    )
    if not has_numbers:
        missing.append("numbers")
    verified = len(missing) == 0
    logger.info(f"[TABLE_CORE][verify_table_structure] Verified: {verified}, Missing: {missing}")
    return verified, missing

def progressive_table_verification(headers, data, coordinator, context):
    """
    Stepwise verification of extracted table structure.
    Logs and verifies each component: location, ballot types, candidates, totals.
    Returns (verified_headers, verified_data, structure_info)
    """
    logger.info("[TABLE BUILDER][progressive_table_verification] Starting verification of extracted table.")
    if coordinator is None:
        coordinator = ContextCoordinator()
    # 1. Detect location column
    location_header = None
    location_patterns = set(coordinator.library.get("location_patterns", [])) | LOCATION_KEYWORDS
    for h in headers:
        if any(pat in h.lower() for pat in location_patterns):
            location_header = h
            break
    if not location_header:
        logger.warning("[TABLE BUILDER][progressive_table_verification] No location column detected.")
    else:
        logger.info(f"[TABLE BUILDER][progressive_table_verification] Detected location column: {location_header}")

    # 2. Detect ballot type columns
    ballot_types_headers = [h for h in headers if any(bt.lower() in h.lower() for bt in BALLOT_TYPES)]
    if not ballot_types_headers:
        logger.warning("[TABLE BUILDER][progressive_table_verification] No ballot type columns detected.")
    else:
        logger.info(f"[TABLE BUILDER][progressive_table_verification] Detected ballot type columns: {ballot_types_headers}")

    # 3. Detect candidate columns (using NER)
    candidate_headers = []
    for h in headers:
        ents = coordinator.extract_entities(h)
        if any(label == "PERSON" for ent, label in ents):
            candidate_headers.append(h)
    if not candidate_headers:
        logger.warning("[TABLE BUILDER][progressive_table_verification] No candidate columns detected.")
    else:
        logger.info(f"[TABLE BUILDER][progressive_table_verification] Detected candidate columns: {candidate_headers}")

    # 4. Detect Grand Total column
    total_header = next((h for h in headers if "total" in h.lower()), None)
    if not total_header:
        logger.warning("[TABLE BUILDER][progressive_table_verification] No Grand Total column detected.")
    else:
        logger.info(f"[TABLE BUILDER][progressive_table_verification] Detected Grand Total column: {total_header}")

    # 5. Verify row structure
    for i, row in enumerate(data[:5]):
        loc_val = row.get(location_header, "")
        ballot_vals = [row.get(h, "") for h in ballot_types_headers]
        candidate_vals = [row.get(h, "") for h in candidate_headers]
        logger.info(f"[TABLE BUILDER][progressive_table_verification] Row {i}: location={loc_val}, ballot_types={ballot_vals}, candidates={candidate_vals}")

    # 6. Structure info summary
    structure_info = {
        "location_header": location_header,
        "ballot_types_headers": ballot_types_headers,
        "candidate_headers": candidate_headers,
        "total_header": total_header,
        "verified": all([location_header, ballot_types_headers, candidate_headers, total_header])
    }
    logger.info(f"[TABLE BUILDER][progressive_table_verification] Structure summary: {structure_info}")

    # Optionally: prompt for correction or fallback if not verified
    # Optionally: persist structure_info for feedback learning

    return headers, data, structure_info

def rescan_and_verify(headers: List[str], data: List[Dict[str, Any]], coordinator: "ContextCoordinator", context: dict, threshold: float = 0.85) -> Tuple[List[str], List[Dict[str, Any]], bool]:
    """
    Rescans headers and data, verifies with ML/NER, and retries if below threshold.
    Returns (headers, data, passed)
    """
    # Use coordinator's ML/NER to score headers
    if coordinator is None:
        coordinator = ContextCoordinator()
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
                # Use the most likely entity label
                ent, label = entities[0]
                new_headers.append(ent)
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

def force_fully_wide_format(headers, data, coordinator: "ContextCoordinator" = None, context=None):
    """
    Pivot to fully wide format: one row per location (real or synthetic),
    columns for each candidate/party/ballot type pair, plus special columns like
    Percent Reported and Misc Totals. Preserves all rows.
    """
    if coordinator is None:
        coordinator = ContextCoordinator()
    # 1. Find or synthesize location column
    location_col = next((h for h in headers if is_location_header(h)), None)
    if not location_col:
        location_col = "Location"
        for idx, row in enumerate(data):
            row[location_col] = (
                context.get("contest_title", []) if context and context.get("contest_title", []) else f"Row {idx+1}"
            )

    # 2. Find candidate and party columns
    candidate_col = next((h for h in headers if any(ck in h.lower() for ck in CANDIDATE_KEYWORDS)), None)
    party_col = next((h for h in headers if "party" in h.lower()), None)

    # 3. Find ballot type columns (known types, not location/candidate/party)
    ballot_types_cols = [
        h for h in headers
        if h not in (location_col, candidate_col, party_col) and any(bt.lower() in h.lower() for bt in BALLOT_TYPES)
    ]
    # If no ballot type columns, use all except location/candidate/party/total/specials
    if not ballot_types_cols:
        ballot_types_cols = [
            h for h in headers
            if h not in (location_col, candidate_col, party_col)
            and "total" not in h.lower()
            and not any(kw in h.lower() for kw in PERCENT_KEYWORDS)
            and not any(kw in h.lower() for kw in MISC_FOOTER_KEYWORDS)
        ]

    # 4. Find special columns
    percent_cols = [h for h in headers if any(kw in h.lower() for kw in PERCENT_KEYWORDS)]
    misc_total_cols = [h for h in headers if any(kw in h.lower() for kw in MISC_FOOTER_KEYWORDS or TOTAL_KEYWORDS)]

    # 5. Get all unique locations, candidates, parties, ballot types
    locations = [row.get(location_col, f"Row {i+1}") for i, row in enumerate(data)]
    unique_locations = sorted(set(locations))
    candidates = sorted(set(row.get(candidate_col, "") for row in data if candidate_col))
    parties = sorted(set(row.get(party_col, "") for row in data if party_col))
    ballot_types = sorted(set(ballot_types_cols))

    # 6. Build wide headers
    wide_headers = [location_col]
    wide_headers.extend(percent_cols)
    candidate_party_pairs = []
    for row in data:
        candidate = row.get(candidate_col, "")
        party = ""
        ents = coordinator.extract_entities(candidate)
        for ent, label in ents:
            if label in {"ORG", "NORP"}:
                party = ent
        if not party:
            party = "Other"
        candidate_party_pairs.append((candidate, party))
    for candidate, party in sorted(candidate_party_pairs):
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
            if row.get(location_col, "") != loc:
                continue
            # Special columns
            for pcol in percent_cols:
                if pcol in out_row and row.get(pcol, ""):
                    out_row[pcol] = row.get(pcol, "")
            for mcol in misc_total_cols:
                if mcol in out_row and row.get(mcol, ""):
                    out_row[mcol] = row.get(mcol, "")
                    try:
                        grand_total += int(row.get(mcol, "0").replace(",", ""))
                    except Exception:
                        pass
            candidate = row.get(candidate_col, "")
            party = row.get(party_col, "") if party_col else ""
            for bt in ballot_types:
                key = f"{candidate} ({party}) - {bt}" if party else f"{candidate} - {bt}"
                val = row.get(bt, "")
                if val and key in out_row:
                    out_row[key] = val
                    try:
                        grand_total += int(val.replace(",", ""))
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
    Annotates table structure using both NLP and DOM info.
    Returns a dict with structure type and detected entity columns.
    Never transforms data.
    """
    if coordinator is None:
        coordinator = ContextCoordinator()
    logger.info("[TABLE_CORE][detect_table_structure] Analyzing table structure.")
    if entity_info is None:
        entity_info = {}

    # Heuristic: If first column is "Candidate" and the rest are ballot types, it's already wide
    if headers and headers[0].lower() == "candidate" and all(
        any(bt in h.lower() for bt in ["election day", "early voting", "absentee", "mail", "total"]) for h in headers[1:]
    ):
        return {"type_": "already-wide", "candidate_col": 0, "ballot_types_cols": list(range(1, len(headers)))}

    # Use entity_info and header heuristics
    candidate_cols = []
    location_cols = []
    ballot_types_cols = []
    for idx, h in enumerate(headers):
        if entity_info.get("people", []) and any(p in h for p in entity_info["people"]):
            candidate_cols.append(idx)
        if entity_info.get("locations", []) and any(l in h for l in entity_info["locations"]):
            location_cols.append(idx)
        if entity_info.get("ballot_types", []) and any(bt in h for bt in entity_info["ballot_types"]):
            ballot_types_cols.append(idx)
        # Fallback: heuristics
        if is_location_header(h):
            location_cols.append(idx)
        if any(bt.lower() in h.lower() for bt in BALLOT_TYPES):
            ballot_types_cols.append(idx)
    # Heuristic: if first col is candidate, columns are ballot types
    if candidate_cols and set(ballot_types_cols) == set(range(1, len(headers))):
        return {"type_": "candidate-major", "candidate_col": candidate_cols[0], "ballot_types_cols": ballot_types_cols}
    if location_cols and set(candidate_cols) == set(range(1, len(headers))):
        return {"type_": "precinct-major", "location_col": location_cols[0], "candidate_cols": candidate_cols}
    return {"type_": "ambiguous", "candidate_cols": candidate_cols, "location_cols": location_cols, "ballot_types_cols": ballot_types_cols}

def handle_candidate_major(headers, data, coordinator, context):
    """
    Handles tables where each row is a candidate, columns are ballot types.
    """
    if coordinator is None:
        coordinator = ContextCoordinator()
    # Detect location and percent columns
    location_header, percent_header = dynamic_detect_location_header(headers, coordinator)
    if not location_header:
        location_header = "Precinct"
    if not percent_header:
        percent_header = "Percent Reported"
        
    # Detect candidate, party, and ballot type columns    
    structure_info = detect_table_structure(headers, data, coordinator)
    candidate_col = structure_info.get("candidate_col", 0)
    party_col = structure_info.get("party_col", None)
    ballot_types_cols = structure_info.get("ballot_types_cols", list(range(1, len(headers))))
    
    # Get ballot type names
    ballot_types = [headers[idx] for idx in ballot_types_cols]

    # Special columns
    percent_cols = [h for h in headers if any(kw in h.lower() for kw in PERCENT_KEYWORDS)]
    misc_total_cols = [h for h in headers if any(kw in h.lower() for kw in (TOTAL_KEYWORDS | MISC_FOOTER_KEYWORDS))]
    
    output_headers = [percent_header, location_header]
    candidate_party_map = {}
    for row in data:
        candidate = row[headers[candidate_col]]
        party = ""
        ents = coordinator.extract_entities(candidate)
        for ent, label in ents:
            if label in {"ORG", "NORP"}:
                party = ent
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
    location_vals = set(row.get(location_header, "All") for row in data)
    for loc in location_vals:
        out_row = {h: "" for h in output_headers}
        out_row[location_header] = loc
        out_row[percent_header] = ""
        grand_total = 0
        for row in data:
            if row.get(location_header, "All") != loc:
                continue
            # Special columns
            for pcol in percent_cols:
                if pcol in out_row and row.get(pcol, ""):
                    out_row[pcol] = row.get(pcol, "")
            for mcol in misc_total_cols:
                if mcol in out_row and row.get(mcol, ""):
                    out_row[mcol] = row.get(mcol, "")
                    try:
                        grand_total += int(row.get(mcol, "0").replace(",", ""))
                    except Exception:
                        pass
            candidate = row.get(candidate_col, "")
            party = row.get(party_col, "") if party_col else ""
            for bt in ballot_types:
                key = f"{candidate} ({party}) - {bt}" if party else f"{candidate} - {bt}"
                val = row.get(bt, "")
                if val and key in out_row:
                    out_row[key] = val
                    try:
                        grand_total += int(val.replace(",", ""))
                    except Exception:
                        pass
        out_row["Grand Total"] = str(grand_total)
        output_data.append(out_row)
    return harmonize_headers_and_data(output_headers, output_data)

def handle_precinct_major(headers, data, coordinator, context):
    """
    Handles tables where each row is a precinct, columns are candidates.
    """
    if coordinator is None:
        coordinator = ContextCoordinator()
    return pivot_precinct_major_to_wide(headers, data, coordinator, context)

def handle_ambiguous(headers, data, coordinator, context):
    """
    Handles ambiguous tables by trying both handlers and picking the one with more filled data.
    """
    if coordinator is None:
        coordinator = ContextCoordinator()
    # Try candidate-major
    cand_headers, cand_data = handle_candidate_major(headers, data, coordinator, context)
    # Try precinct-major
    prec_headers, prec_data = handle_precinct_major(headers, data, coordinator, context)
    # Heuristic: pick the one with more non-empty cells
    def non_empty_count(data):
        return sum(1 for row in data for v in row.values() if v not in ("", "0", 0, None))
    if non_empty_count(cand_data) >= non_empty_count(prec_data):
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
    if coordinator is None:
        coordinator = ContextCoordinator()
    location_header = None
    percent_header = None
    for h in headers:
        if is_location_header(h) and h.lower() != "candidate":
            location_header = h
        if h.lower() in (ph.lower() for ph in PERCENT_KEYWORDS) or "%" in h or "reported" in h.lower():
            percent_header = h
    if not location_header:
        location_header = "Precinct"
    if location_header != "Precinct":
        headers = ["Precinct" if h == location_header else h for h in headers]
        for row in data:
            row["Precinct"] = row.pop(location_header)
        location_header = "Precinct"
    # 2. Gather all unique candidates and ballot types using canonical normalization
    candidates = set()
    ballot_types = set()
    for row in data:
        cand = row.get("Candidate", "")
        if cand:
            candidates.add(cand.strip())
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
    location_values = set(row.get(location_header, "") for row in data if row.get(location_header, ""))
    wide_data = []
    for loc in sorted(location_values):
        out_row = {h: "" for h in wide_headers}
        out_row[location_header] = loc
        if percent_header:
            # Use the first found value for this precinct
            for row in data:
                if row.get(location_header, "") == loc and percent_header in row:
                    out_row[percent_header] = row[percent_header]
                    break
        grand_total = 0
        for candidate in sorted(candidates):
            cand_total = 0
            for bt in ballot_types_sorted:
                val = ""
                for row in data:
                    if row.get(location_header, "") == loc and row.get("Candidate", "") == candidate:
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
    logger.info(f"[TABLE_CORE][pivot_to_wide_format] Wide format: {len(wide_data)} rows, {len(wide_headers)} columns.")
    return wide_headers, wide_data

def pivot_precinct_major_to_wide(
    headers: List[str],
    data: List[Dict[str, Any]],
    coordinator: "ContextCoordinator",
    context: dict
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Pivot a precinct-major table to wide format:
    Precinct | Percent Reported | [Candidate (Party) - BallotType ... Total Votes] | [Misc Totals] | Grand Total
    Handles variable ballot types and miscellaneous columns.
    """
    if coordinator is None:
        coordinator = ContextCoordinator()
    location_header, percent_header = dynamic_detect_location_header(headers, coordinator)
    if not percent_header:
        percent_header = "Percent Reported"

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
        for bt, h in bt_map.items():
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
        out_row[location_header] = row.get(location_header, "")
        out_row[percent_header] = row.get(percent_header, "Fully Reported")
        grand_total = 0
        # Candidate columns
        for candidate, party in sorted(candidate_party_set):
            cand_total = 0
            bt_map = candidate_party_ballot.get((candidate, party), {})
            for bt in ballot_types:
                col = f"{candidate} ({party}) - {bt}"
                val = row.get(bt_map.get(bt, ""), "")
                try:
                    ival = int(val.replace(",", "")) if val else 0
                except Exception:
                    ival = 0
                out_row[col] = str(ival) if val != "" else ""
                cand_total += ival
            out_row[f"{candidate} ({party}) - Total Votes"] = str(cand_total)
            grand_total += cand_total
        # Misc columns
        for h in misc_columns:
            out_row[h] = row.get(h, "")
            try:
                grand_total += int(row.get(h, "0").replace(",", "")) if row.get(h, "") else 0
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
            values = [r.get(h, "0").replace(",", "") for r in output_rows]
            if all(v == "" or v.isdigit() or (v.startswith('-') and v[1:].isdigit()) for v in values):
                totals_row[h] = str(sum(int(v) for v in values if v != ""))
            else:
                totals_row[h] = ""
        except Exception:
            totals_row[h] = ""
    output_rows.append(totals_row)
    logger.info(f"[TABLE BUILDER] Build dynamic tables Final table: {len(output_rows)} rows, {len(output_headers)} columns.")
    return output_headers, output_rows

def dynamic_detect_location_header(headers: List[str], coordinator: "ContextCoordinator") -> Tuple[str, str]:
    """
    Dynamically detect the first and second location columns (e.g., precinct, ward, city, district, municipal).
    Uses context, regex, NER, and library.
    Returns (location_header, percent_reported_header)
    """
    if coordinator is None:
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

    # 3. Try spaCy NER if available
    if not location_header and coordinator:
        for idx, h in enumerate(headers):
            entities = coordinator.extract_entities(h)
            for ent, label in entities:
                if label in {"GPE", "LOC", "FAC"}:
                    location_header = headers[idx]
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

    logger.info(f"[TABLE BUILDER] Location header detected: {location_header}, Percent header detected: {percent_header}")
    return location_header, percent_header

def is_likely_header(row):
    # Combine all relevant keywords into a single set for header detection
    known_fields = (
        set(k.lower() for k in CANDIDATE_KEYWORDS)
        | set(k.lower() for k in PARTY_KEYWORDS)
        | set(k.lower() for k in LOCATION_KEYWORDS)
        | set(k.lower() for k in PERCENT_KEYWORDS)
        | set(k.lower() for k in TOTAL_KEYWORDS)
        | {"votes", "percent", "district", "party", "candidate"}
    )
    return sum(1 for cell in row if any(k in cell.lower() for k in known_fields)) >= 2

# ===================================================================
# ADVANCED/UTILITY FUNCTIONS
# ===================================================================

def normalize_text(text):
    """
    Normalize text for comparison: lowercase, strip, remove accents.
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.strip().lower()
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    return text

def normalize_header(header, lang="en"):
    """
    Normalize header for comparison: lower, strip, remove accents, and translate if needed.
    """
    header = header.strip().lower()
    header = unicodedata.normalize('NFKD', header).encode('ascii', 'ignore').decode('ascii')
    # Optionally: add translation for non-English headers here using a translation dictionary or service
    # Example: if lang != "en": header = translate(header, lang)
    return header

def normalize_header_name(header):
    """
    Normalize header for deduplication and comparison.
    Lowercase, strip, remove accents, and collapse whitespace.
    """
    if not isinstance(header, str):
        header = str(header)
    header = header.strip().lower()
    header = unicodedata.normalize('NFKD', header).encode('ascii', 'ignore').decode('ascii')
    header = re.sub(r"\s+", " ", header)
    return header

def is_date_like(val):
    import dateutil.parser
    try:
        dateutil.parser.parse(val)
        return True
    except Exception:
        return False

def detect_language(headers):
    """
    Detect language of headers (very basic, can be replaced with langdetect).
    """
    try:
        from langdetect import detect
        text = " ".join(headers)
        return detect(text)
    except Exception:
        return "en"

def dynamic_required_columns(context, default_required=None):
    """
    Adjust required columns based on context.
    """
    if default_required is None:
        default_required = {"Grand Total", "Precinct", "Location"}
    # Example: if context says percent reported is not present, remove it
    if not context.get("has_percent_reported", True):
        default_required.discard("Percent Reported")
    return default_required

def log_failed_container(page, container, selector, idx, error_msg):
    if container is None:
        logger.error(f"[TABLE BUILDER] log_failed_container: container is None for selector {selector} idx {idx}")
        return
    try:
        html = container.evaluate("el => el.outerHTML")
        parent = container.locator("xpath=..")
        parent_class = parent.get_attribute("class", []) or ""
        parent_id = parent.get_attribute("id", []) or ""
        heading = ""
        heading_loc = container.locator("xpath=preceding-sibling::*[self::h1 or self::h2 or self::h3 or self::h4 or self::h5 or self::h6][1]")
        if heading_loc.count() > 0:
            heading = heading_loc.nth(0).inner_text().strip()
        log_entry = {
            "selector": selector,
            "container_idx": idx,
            "parent_class": parent_class,
            "parent_id": parent_id,
            "heading": heading,
            "error": error_msg,
            "html": html[:2000]  # Truncate for log size
        }
        log_path = get_safe_log_path(f"failed_container_{selector.replace('.', '_')}_{idx}.json")
        with open(log_path, "wb") as f:
            f.write(orjson.dumps(log_entry))
        logger.error(f"[TABLE BUILDER] Failed container logged: {log_path}")
    except Exception as e:
        logger.error(f"[TABLE BUILDER] Could not log failed container: {e}")



def suggest_new_row_classes_from_logs(log_dir):
    """
    Analyze failed container logs and suggest new likely row classes/IDs.
    """
    class_counter = Counter()
    parent_counter = Counter()
    for path in glob.glob(os.path.join(log_dir, "failed_container_*.json")):
        with open(path, "rb") as f:
            entry = orjson.loads(f.read())
            cls = entry.get("parent_class", "")
            if cls:
                for c in cls.split():
                    class_counter[c] += 1
            parent_id = entry.get("parent_id", "")
            if parent_id:
                parent_counter[parent_id] += 1
    # Suggest top classes/IDs as new selectors
    suggested_classes = [c for c, _ in class_counter.most_common(10)]
    suggested_ids = [pid for pid, _ in parent_counter.most_common(5)]
    logger.info("Suggested new row classes:", suggested_classes)
    logger.info("Suggested new row IDs:", suggested_ids)
    return suggested_classes, suggested_ids

def load_dom_patterns(log_path=None):
    """
    Loads all DOM patterns, returns a list of dicts.
    """
    if log_path is None:
        log_path = get_safe_log_path("dom_pattern_log.jsonl")
    if not os.path.exists(log_path):
        return []
    with open(log_path, "rb") as f:
        return [orjson.loads(line) for line in f if line.strip()]

def remove_footer_and_summary_rows(data, headers):
    """
    Remove rows that are likely summary, totals, or repeated headers.
    --- Only remove if 'total' or 'summary' appears in a column that is a total/summary column.
    """
    filtered = []
    total_cols = [h for h in headers if any(kw in h.lower() for kw in TOTAL_KEYWORDS.union(MISC_FOOTER_KEYWORDS))]
    for row in data:
        values = list(row.values())
        # --- Only remove if 'total' or 'summary' appears in a total/summary column
        remove = False
        for h in total_cols:
            v = row.get(h, "")
            if any(kw in str(v).lower() for kw in TOTAL_KEYWORDS.union(MISC_FOOTER_KEYWORDS)):
                remove = True
                break
        # --- Do not remove if header row repeated (keep as is)
        if not remove:
            filtered.append(row)
    return filtered

def remove_outlier_and_empty_rows(data, min_non_empty=2):
    """
    Remove rows with too many empty or repeated values.
    --- Only remove if truly all values are empty.
    """
    filtered = []
    for row in data:
        values = list(row.values())
        non_empty = [v for v in values if v not in ("", None)]
        # --- Only remove if all values are empty
        if len(non_empty) > 0:
            filtered.append(row)
    return filtered

def review_learned_table_structures(log_path=None):
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
        logger.info(f"\n[{idx}] Contest: {entry.get('contest_title', [])}")
        logger.info(f"    Headers: {entry.get('headers', [])}")
        logger.info(f"    Context: {entry.get('context', [])}")
        logger.info(f"    Result: {entry.get('result', [])}")
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

def table_signature(headers):
    return hashlib.md5(orjson.dumps(headers, sort_keys=True)).hexdigest()

def load_table_structure_cache():
    if os.path.exists(TABLE_STRUCTURE_CACHE_PATH):
        with open(TABLE_STRUCTURE_CACHE_PATH, "rb") as f:
            return orjson.loads(f.read())
    return {}

def save_table_structure_cache(cache):
    with open(TABLE_STRUCTURE_CACHE_PATH, "wb") as f:
        f.write(orjson.dumps(cache))

def cache_table_structure(domain, headers, structure):
    cache = load_table_structure_cache()
    sig = f"{domain}:{table_signature(headers)}"
    cache[sig] = structure
    save_table_structure_cache(cache)

def get_cached_table_structure(domain, headers):
    cache = load_table_structure_cache()
    sig = f"{domain}:{table_signature(headers)}"
    return cache.get(sig, [])

def guess_contest_title(table_headers, known_titles):
    """
    Try to match table headers to known contest titles using fuzzy matching.
    """
    for header in table_headers:
        matches = difflib.get_close_matches(header, known_titles, n=1, cutoff=0.7)
        if matches:
            return matches[0]
    return None

def extract_title_from_html_near_table(table_idx, dom_nodes, window=5):
    """
    Scan nearby DOM nodes for likely contest titles.
    """
    idx_range = range(max(0, table_idx - window), min(len(dom_nodes), table_idx + window + 1))
    for idx in idx_range:
        node = dom_nodes[idx]
        if node.get("tag", "").lower() in {"h1", "h2", "h3", "caption"}:
            text = node.get("html", "").strip()
            if text and len(text.split()) > 2:
                return text
    return None

def merge_multirow_headers(header_rows):
    """
    Merge multiple header rows (e.g., stacked headers) into a single header list.
    """
    merged = []
    for cols in zip(*header_rows):
        merged_col = " ".join([c for c in cols if c and c.strip() and not c.strip().isdigit()])
        merged.append(merged_col.strip())
    return merged

def fuzzy_merge_headers(headers, threshold=0.85):
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

def profile_extraction_step(func):
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

def log_decision(decision, context=None):
    """
    Log not just errors but also decisions made by heuristics for later review.
    """
    logger.info(f"[DECISION] {decision} | Context: {context}")

def robust_html_fallback(page):
    """
    Add more robust fallbacks for broken or inconsistent markup.
    """
    try:
        html = page.content()
        # Try to parse with BeautifulSoup as a fallback
        soup = BeautifulSoup(html, "html.parser")
        tables = soup.find_all("table")
        all_tables = []
        for table in tables:
            rows = table.find_all("tr")
            headers = [th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])]
            data = []
            for row in rows[1:]:
                cells = row.find_all(["td", "th"])
                data.append({headers[i]: cells[i].get_text(strip=True) if i < len(cells) else "" for i in range(len(headers))})
            all_tables.append((headers, data))
        return all_tables
    except Exception as e:
        logger.error(f"[HTML FALLBACK] Error: {e}")
        return []

def handle_nested_tables(page):
    """
    Handle tables within tables or complex nested DOM structures.
    """
    tables = page.locator("table table")
    results = []
    for i in range(tables.count()):
        table = tables.nth(i)
        if table is not None:
            headers, data, _ = extract_table_data(table)
            results.append((headers, data))
    return results

def fuzzy_in(word, text, threshold=0.7):
    """Return True if word is in text by substring or fuzzy match."""
    word = word.lower()
    text = text.lower()
    if word in text:
        return True
    # Fuzzy match: allow for partials (e.g., "town" in "orangetown")
    ratio = SequenceMatcher(None, word, text).ratio()
    return ratio >= threshold

def normalize_for_matching(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

def contains_location_keyword(text, keywords=LOCATION_KEYWORDS):
    text_norm = normalize_for_matching(text)
    for kw in keywords:
        # Match as a whole word or as a suffix/prefix (e.g., "orangetown")
        if re.search(rf"\b{re.escape(kw)}\b", text_norm):
            return True
        if kw in text_norm:
            return True
    return False

def is_location_header(header):
    """
    Robustly determine if a header is a location column using LOCATION_KEYWORDS and abbreviations.
    This is the SINGLE SOURCE OF TRUTH for location column detection.
    - Uses normalization, substring, fuzzy, and regex matching.
    - Always update LOCATION_KEYWORDS in librarian.py for new variants.
    """
    header_norm = normalize_for_matching(header)
    for kw in LOCATION_KEYWORDS:
        if fuzzy_in(kw, header_norm) or contains_location_keyword(header_norm, LOCATION_KEYWORDS):
            return True
    # Also match common abbreviations and variants
    if header_norm in LOCATION_ABBREVIATIONS:
        return True
    return False
