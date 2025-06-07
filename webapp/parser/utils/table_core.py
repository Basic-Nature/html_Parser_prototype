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
import json
import re
import unicodedata
import glob
import re
import string
from difflib import SequenceMatcher
from collections import Counter
from typing import List, Dict, Any, Tuple, TYPE_CHECKING
import time
import hashlib
from ..utils.shared_logger import logger
from ..utils.ml_table_detector import detect_tables_ml
from ..config import BASE_DIR

if TYPE_CHECKING:
    from ..Context_Integration.context_coordinator import ContextCoordinator

# --- CONSTANTS & GLOBALS ---

TABLE_STRUCTURE_CACHE_PATH = os.path.join(BASE_DIR, "parser", "Context_Integration", "Context_Library", "table_structure_cache.json")
BALLOT_TYPES = [
    "Election Day", "Early Voting", "Absentee", "Mail", "Provisional", "Affidavit", "Other", "Void"
]
LOCATION_KEYWORDS = {
    "precinct", "ward", "district", "location", "area", "city", "municipal", "town",
    "borough", "village", "county", "division", "subdistrict", "polling place", "ed", "municipality"
}
PERCENT_KEYWORDS = {
    "% precincts reporting", "% reported", "percent reported", "fully reported", "precincts reporting"
}
TOTAL_KEYWORDS = {"total", "sum", "votes", "overall", "all", "Percent Reported", "Reporting Status" }
MISC_FOOTER_KEYWORDS = {"undervote", "overvote", "scattering", "write-in", "blank", "void", "spoiled"}
CANDIDATE_KEYWORDS = {
    "candidate", "candidates", "name", "nominee", "person", "individual", "contestant"
}
context_cache = {}

# ===================================================================
# MAIN EXTRACTION ENTRY POINT
# ===================================================================

def robust_table_extraction(page, extraction_context=None, existing_headers=None, existing_data=None):
    """
    Unified, persistent table extraction pipeline:
    - Accumulates all plausible tables/rows from DOM, patterns, standard tables, ML, plugins, and robust fallbacks.
    - Deduplicates tables and rows.
    - Runs entity annotation and structure verification on the unified result.
    - Only uses fallback extraction if all other methods fail.
    - All strategies are integrated here. Do not call individual strategies from handlers.
    """
    import types

    def safe_json(obj):
        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                if k in ("coordinator", "ContextCoordinator"):
                    continue
                if isinstance(v, (types.FunctionType, types.ModuleType)) or hasattr(v, "__dict__"):
                    continue
                try:
                    json.dumps(v)
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

    # 1. DOM structure extraction (divs, lists, etc.)
    try:
        headers_dom, data_dom = extract_rows_and_headers_from_dom(
            page, coordinator=extraction_context.get("coordinator") if extraction_context else None
        )
        if headers_dom and data_dom:
            all_tables.append((headers_dom, data_dom))
            extraction_logs.append({
                "method": "repeated_dom",
                "headers": headers_dom,
                "rows": len(data_dom),
                "columns": len(headers_dom),
                "success": True,
                "context": extraction_context
            })
    except Exception as e:
        logger.error(f"[TABLE BUILDER] DOM structure extraction failed: {e}")
        extraction_logs.append({
            "method": "repeated_dom",
            "error": str(e),
            "success": False,
            "context": extraction_context
        })

    # 2. Pattern-based extraction (approved DOM patterns)
    try:
        pattern_rows = extract_with_patterns(page, extraction_context)
        pattern_rows = [tup for tup in pattern_rows if tup[1] is not None]
        if pattern_rows:
            headers_pat = guess_headers_from_row(pattern_rows[0][1])
            data_pat = []
            for heading, row, pat in pattern_rows:
                if row is None:
                    continue
                cells = row.locator("> *")
                row_data = {}
                for idx in range(cells.count()):
                    row_data[headers_pat[idx] if idx < len(headers_pat) else f"Column {idx+1}"] = cells.nth(idx).inner_text().strip()
                if row_data:
                    data_pat.append(row_data)
            if headers_pat and data_pat:
                all_tables.append((headers_pat, data_pat))
                extraction_logs.append({
                    "method": "pattern",
                    "headers": headers_pat,
                    "rows": len(data_pat),
                    "columns": len(headers_pat),
                    "success": True,
                    "context": extraction_context
                })
    except Exception as e:
        logger.error(f"[TABLE BUILDER] Pattern extraction failed: {e}")
        extraction_logs.append({
            "method": "pattern",
            "error": str(e),
            "success": False,
            "context": extraction_context
        })

    # 3. Standard HTML table extraction
    try:
        tables = page.locator("table")
        for i in range(tables.count()):
            table = tables.nth(i)
            if table is not None:
                headers_tab, data_tab, _ = extract_table_data(table)
                if headers_tab and data_tab:
                    all_tables.append((headers_tab, data_tab))
                    extraction_logs.append({
                        "method": "table",
                        "headers": headers_tab,
                        "rows": len(data_tab),
                        "columns": len(headers_tab),
                        "success": True,
                        "context": extraction_context
                    })
    except Exception as e:
        logger.error(f"[TABLE BUILDER] Table extraction failed: {e}")
        extraction_logs.append({
            "method": "table",
            "error": str(e),
            "success": False,
            "context": extraction_context
        })

    # 4. Table extraction with heading/location context
    try:
        headers_loc, data_loc, _ = extract_all_tables_with_location(
            page, coordinator=extraction_context.get("coordinator") if extraction_context else None
        )
        if headers_loc and data_loc:
            all_tables.append((headers_loc, data_loc))
            extraction_logs.append({
                "method": "table_with_heading",
                "headers": headers_loc,
                "rows": len(data_loc),
                "columns": len(headers_loc),
                "success": True,
                "context": extraction_context
            })
    except Exception as e:
        logger.error(f"[TABLE BUILDER] Table-with-heading extraction failed: {e}")
        extraction_logs.append({
            "method": "table_with_heading",
            "error": str(e),
            "success": False,
            "context": extraction_context
        })

    # 5. ML-based table detection
    try:
        ml_tables = ml_based_table_detection(page, extraction_context)
        for headers_ml, data_ml in ml_tables:
            if headers_ml and data_ml:
                all_tables.append((headers_ml, data_ml))
                extraction_logs.append({
                    "method": "ml_table_detection",
                    "headers": headers_ml,
                    "rows": len(data_ml),
                    "columns": len(headers_ml),
                    "success": True,
                    "context": extraction_context
                })
    except Exception as e:
        logger.error(f"[TABLE BUILDER] ML table detection failed: {e}")
        extraction_logs.append({
            "method": "ml_table_detection",
            "error": str(e),
            "success": False,
            "context": extraction_context
        })

    # 6. Nested table extraction
    try:
        nested_tables = nested_table_extraction(page)
        for headers_nested, data_nested in nested_tables:
            if headers_nested and data_nested:
                all_tables.append((headers_nested, data_nested))
                extraction_logs.append({
                    "method": "nested_table",
                    "headers": headers_nested,
                    "rows": len(data_nested),
                    "columns": len(headers_nested),
                    "success": True,
                    "context": extraction_context
                })
    except Exception as e:
        logger.error(f"[TABLE BUILDER] Nested table extraction failed: {e}")
        extraction_logs.append({
            "method": "nested_table",
            "error": str(e),
            "success": False,
            "context": extraction_context
        })

    # 7. Custom plugin extraction
    try:
        plugin_tables = custom_plugin_extraction(page, extraction_context)
        for headers_plugin, data_plugin in plugin_tables:
            if headers_plugin and data_plugin:
                all_tables.append((headers_plugin, data_plugin))
                extraction_logs.append({
                    "method": "plugin",
                    "headers": headers_plugin,
                    "rows": len(data_plugin),
                    "columns": len(headers_plugin),
                    "success": True,
                    "context": extraction_context
                })
    except Exception as e:
        logger.error(f"[TABLE BUILDER] Plugin extraction failed: {e}")
        extraction_logs.append({
            "method": "plugin",
            "error": str(e),
            "success": False,
            "context": extraction_context
        })

    # 8. Add any existing headers/data provided
    if existing_headers and existing_data and len(existing_headers) > 0 and len(existing_data) > 0:
        all_tables.append((existing_headers, existing_data))

    # 9. Robust HTML fallback using BeautifulSoup
    try:
        fallback_tables = robust_html_fallback_extraction(page)
        for headers_fallback, data_fallback in fallback_tables:
            if headers_fallback and data_fallback:
                all_tables.append((headers_fallback, data_fallback))
                extraction_logs.append({
                    "method": "html_fallback",
                    "headers": headers_fallback,
                    "rows": len(data_fallback),
                    "columns": len(headers_fallback),
                    "success": True,
                    "context": extraction_context
                })
    except Exception as e:
        logger.error(f"[TABLE BUILDER] HTML fallback extraction failed: {e}")
        extraction_logs.append({
            "method": "html_fallback",
            "error": str(e),
            "success": False,
            "context": extraction_context
        })

    logger.info(f"[TABLE BUILDER] Extraction summary: {json.dumps(safe_json(extraction_logs), indent=2)}")

    # --- Deduplicate tables by header signature ---
    unique_tables = {}
    for headers, data in all_tables:
        sig = tuple(normalize_header_name(h) for h in headers)
        if sig not in unique_tables or (len(data) > len(unique_tables[sig][1])):
            unique_tables[sig] = (headers, data)
    all_tables = list(unique_tables.values())

    # --- Merge all unique rows from all sources ---
    if all_tables:
        merged_headers = []
        merged_data = []
        for headers, data in all_tables:
            for h in headers:
                if h not in merged_headers:
                    merged_headers.append(h)
            merged_data.extend(data)
        # Harmonize and deduplicate rows
        merged_headers, merged_data = harmonize_headers_and_data(merged_headers, merged_data)
        logger.info(f"[TABLE BUILDER] Unified extraction: {len(merged_data)} rows, {len(merged_headers)} columns.")

        # Entity annotation and structure verification
        coordinator = extraction_context.get("coordinator") if extraction_context else None
        merged_headers, merged_data, entity_info = nlp_entity_annotate_table(
            merged_headers, merged_data, context=extraction_context, coordinator=coordinator
        )
        merged_headers, merged_data = harmonize_headers_and_data(merged_headers, merged_data)
        merged_headers, merged_data, _ = progressive_table_verification(
            merged_headers, merged_data, coordinator, extraction_context
        )

        # 10. Feedback/correction loop (user-in-the-loop)
        merged_headers, merged_data = feedback_correction_loop(merged_headers, merged_data, extraction_context)

        return merged_headers, merged_data

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

def extract_all_tables_with_location(page, coordinator=None):
    """
    Extract all tables, associating each with the nearest section/district/heading.
    Dynamically chooses between panel-based and section-heading-based extraction,
    scores each, and merges/patches missing information if possible.
    """
    from ..utils.dynamic_table_extractor import (
        find_tables_with_panel_headings,
        find_tables_with_section_headings,
    )

    # Use the global keyword sets for easier expansion
    LOCATION_HEADERS = list(LOCATION_KEYWORDS)
    PERCENT_HEADERS = list(PERCENT_KEYWORDS)

    extraction_types = [
        ("panel", find_tables_with_panel_headings(page)),
        ("section", find_tables_with_section_headings(page)),
    ]

    percent_reported_global = extract_percent_reported_from_page(page)
    extraction_results = []

    for method, tables_with_headings in extraction_types:
        all_headers = set()
        all_data = []
        all_entity_previews = []
        for heading, table in tables_with_headings:
            headers, data, entity_preview = extract_table_data(table, coordinator=coordinator)
            if not headers or not data:
                continue
            
            # --- Find or create a location column name (fuzzy/substring match) ---
            location_col = None
            for h in headers:
                if is_location_header(h):
                    location_col = h
                    break

            # If not found, synthesize
            if location_col is None or location_col.lower() == "candidate":
                # Prefer "District" or "Precinct" as column name
                location_col = "District" if "district" in heading.lower() else "Precinct"
                # Insert as first column if not present
                if location_col not in headers:
                    headers = [location_col] + headers
                # Inject heading value into each row
                for row in data:
                    row[location_col] = heading

            # --- Find or create a percent reported column ---
            percent_col = None
            for h in headers:
                if any(p.lower() in h.lower() for p in PERCENT_HEADERS):
                    percent_col = h
                    break
            if not percent_col:
                percent_col = "Percent Reported"
                if percent_col not in headers:
                    headers.append(percent_col)
                percent_value = extract_percent_reported_from_heading(heading) or percent_reported_global
                for row in data:
                    row[percent_col] = percent_value

            all_headers.update(headers)
            all_data.extend(data)
            all_entity_previews.append(entity_preview)

        # Harmonize headers and data
        all_headers_list = list(all_headers)
        all_headers_list, all_data = harmonize_headers_and_data(all_headers_list, all_data)
        extraction_results.append({
            "method": method,
            "headers": all_headers_list,
            "data": all_data,
            "entity_previews": all_entity_previews,
            "score": 0  # Will be filled below
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

    return best["headers"], best["data"], best["entity_previews"]

def extract_table_data(table, coordinator=None, structure_info=None) -> Tuple[List[str], List[Dict[str, Any]], dict]:
    """
    Extracts headers and data from a Playwright table locator.
    Uses advanced NLP/NER and ML scoring to robustly detect entity columns.
    Never uses "Candidate" as a location column.
    Returns headers, data, and a meta dict with entity preview and detected location column.
    """
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
    }

    try:
        # --- Extract headers ---
        header_cells = table.locator("thead tr th")
        logger.info(f"[TABLE BUILDER][extract_table_data] Found {header_cells.count()} header cells in thead.")
        if header_cells.count() == 0:
            first_row = table.locator("tr").first
            header_cells = first_row.locator("th, td")
            logger.info(f"[TABLE BUILDER][extract_table_data] No thead headers, using first row: {header_cells.count()} cells.")
        for i in range(header_cells.count()):
            text = header_cells.nth(i).inner_text().strip()
            headers.append(text if text else f"Column {i+1}")
        logger.info(f"[TABLE BUILDER][extract_table_data] Extracted headers: {headers}")

        # --- Extract rows ---
        rows = table.locator("tbody tr")
        logger.info(f"[TABLE BUILDER][extract_table_data] Found {rows.count()} rows in tbody.")
        if rows.count() == 0:
            all_rows = table.locator("tr")
            logger.info(f"[TABLE BUILDER][extract_table_data] No tbody rows, using all tr: {all_rows.count()} rows.")
            rows = all_rows

        for i in range(rows.count()):
            row = {}
            cells = rows.nth(i).locator("td, th")
            logger.info(f"[TABLE BUILDER][extract_table_data] Row {i}: {cells.count()} cells.")
            if cells.count() == 0:
                continue
            for j in range(cells.count()):
                if j < len(headers):
                    row[headers[j]] = cells.nth(j).inner_text().strip()
                else:
                    row[f"Extra_{j+1}"] = cells.nth(j).inner_text().strip()
            if any(v for v in row.values()):
                data.append(row)
        logger.info(f"[TABLE BUILDER][extract_table_data] Extracted {len(data)} data rows.")

        # --- Advanced NLP/ML entity detection ---
        # Use centralized, robust keyword sets
        ballot_type_keywords = set(bt.lower() for bt in BALLOT_TYPES)
        # Improved number pattern: allows commas, decimals, percents, and negative numbers
        number_pattern = re.compile(r"^-?\d{1,3}(?:,\d{3})*(?:\.\d+)?%?$")

        # 1. Use NLP/NER to detect likely location columns
        location_col = None
        location_scores = []
        if coordinator:
            for h in headers:
                ents = coordinator.extract_entities(h)
                for ent, label in ents:
                    if label in {"GPE", "LOC", "FAC"} and h.lower() != "candidate":
                        location_scores.append((h, 1.0))
                if is_location_header(h) and h.lower() != "candidate":
                    score = coordinator.score_header(h, {}) if hasattr(coordinator, "score_header") else 0.5
                    location_scores.append((h, score))
            if location_scores:
                location_col = max(location_scores, key=lambda x: x[1])[0]
        # 2. Fallback: use location keyword match, but never "Candidate"
        if not location_col:
            for h in headers:
                if is_location_header(h) and h.lower() != "candidate":
                    location_col = h
                    break
        # 3. If still not found, log a warning (optionally suppress for already-wide)
        entity_preview["location_column"] = location_col
        suppress_warning = structure_info and structure_info.get("type") == "already-wide"
        if not location_col and not suppress_warning:
            logger.warning("[TABLE BUILDER][extract_table_data] No location column detected by NLP/ML. Will not use 'Candidate' as fallback.")

        # --- Scan data for entity types ---
        for row in data:
            for h, v in row.items():
                if not v:
                    continue
                # Candidate detection (robust)
                if any(ck in h.lower() for ck in CANDIDATE_KEYWORDS):
                    entity_preview["candidates"].add(v)
                # Ballot type detection (robust)
                if any(bk in h.lower() for bk in ballot_type_keywords):
                    entity_preview["ballot_types"].add(h)
                # Number detection (improved)
                if number_pattern.match(v.replace(",", "")):
                    entity_preview["numbers"].add(v)
                # Location detection (only if a valid location_col was found)
                if location_col and h == location_col:
                    entity_preview["locations"].add(v)

        # --- Automated feedback/learning: log if location_col is missing or suspect ---
        if not location_col or len(entity_preview["locations"]) == 0:
            logger.warning("[TABLE BUILDER][extract_table_data] No valid location column or values detected. Consider user/ML feedback.")
            # Optionally: trigger feedback_correction_loop or log for learning

        # Log NLP-style preview
        logger.info(f"[NLP PREVIEW][extract_table_data] Candidates: {sorted(entity_preview['candidates'])}")
        logger.info(f"[NLP PREVIEW][extract_table_data] Ballot Types: {sorted(entity_preview['ballot_types'])}")
        logger.info(f"[NLP PREVIEW][extract_table_data] Numbers: {sorted(entity_preview['numbers'])}")
        logger.info(f"[NLP PREVIEW][extract_table_data] Locations: {sorted(entity_preview['locations'])}")
        logger.info(f"[NLP PREVIEW][extract_table_data] Location column: {entity_preview['location_column']}")

        print(f"[NLP PREVIEW] Candidates: {sorted(entity_preview['candidates'])}")
        print(f"[NLP PREVIEW] Ballot Types: {sorted(entity_preview['ballot_types'])}")
        print(f"[NLP PREVIEW] Numbers: {sorted(entity_preview['numbers'])}")
        print(f"[NLP PREVIEW] Locations: {sorted(entity_preview['locations'])}")
        print(f"[NLP PREVIEW] Location column: {entity_preview['location_column']}")

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

def guess_headers_from_row(row, known_keywords=None):
    """
    Attempts to guess headers from a row's children using keywords or context.
    """
    if row is None:
        logger.warning("[TABLE BUILDER][guess_headers_from_row] Row is None, cannot guess headers.")
        return []
    if known_keywords is None:
        known_keywords = ["candidate", "votes", "party", "precinct", "choice", "option", "response", "total"]
    cells = row.locator("> *")
    headers = []
    for i in range(cells.count()):
        text = cells.nth(i).inner_text().strip().lower()
        # Use keyword if present, else generic
        header = None
        for kw in known_keywords:
            if kw in text:
                header = kw.capitalize()
                break
        if not header:
            header = f"Column {i+1}"
        headers.append(header)
    return headers

def extract_rows_and_headers_from_dom(page, extra_keywords=None, min_row_count=2, coordinator=None):
    """
    Attempts to extract tabular data from repeated DOM structures (divs, etc.).
    Returns headers, data.
    Uses advanced heuristics for ambiguous, malformed, or complex cases.
    """
    logger.info("[TABLE BUILDER][extract_rows_and_headers_from_dom] Starting DOM structure extraction.")
    repeated_rows = extract_repeated_dom_structures(page, extra_keywords=extra_keywords, min_row_count=min_row_count)
    logger.info(f"[TABLE BUILDER][extract_rows_and_headers_from_dom] Found {len(repeated_rows)} repeated rows.")
    if not repeated_rows:
        logger.warning("[TABLE BUILDER][extract_rows_and_headers_from_dom] No repeated rows found.")
        return [], []

    # --- Heuristic header detection block ---
    headers = None
    header_row_idx = None
    for idx, (heading, row) in enumerate(repeated_rows[:10]):
        if row is None:
            logger.warning(f"[TABLE BUILDER][extract_rows_and_headers_from_dom] Row locator is None at index {idx}. Skipping.")
            continue
        cells = row.locator("> *")
        cell_texts = [cells.nth(i).inner_text().strip() for i in range(cells.count())]
        logger.info(f"[TABLE BUILDER][extract_rows_and_headers_from_dom] Checking row {idx} for headers: {cell_texts}")
        # Heuristic: header row if at least 2 known fields or all non-numeric
        if is_likely_header(cell_texts) or all(not re.match(r"^\d+([,.]\d+)?$", c) for c in cell_texts):
            headers = cell_texts
            header_row_idx = idx
            logger.info(f"[TABLE BUILDER][extract_rows_and_headers_from_dom] Detected header row at index {idx}: {headers}")
            break
    if headers is not None:
        repeated_rows = repeated_rows[header_row_idx + 1 :]
    else:
        headers = guess_headers_from_row(repeated_rows[0][1])
        logger.info(f"[TABLE BUILDER][extract_rows_and_headers_from_dom] Guessed headers from first row: {headers}")

    # --- Merge split header rows (e.g., two header rows) ---
    if len(repeated_rows) > 1:
        first_row_cells = [repeated_rows[0][1].locator("> *").nth(i).inner_text().strip() for i in range(repeated_rows[0][1].locator("> *").count())]
        if all(c.isalpha() or c == "" for c in first_row_cells) and any(c for c in first_row_cells):
            logger.info(f"[TABLE BUILDER][extract_rows_and_headers_from_dom] Merging split header rows: {headers} + {first_row_cells}")
            headers = [" ".join(filter(None, [h, f])) for h, f in zip(headers, first_row_cells)]
            repeated_rows = repeated_rows[1:]

    # --- Advanced heuristics start here ---
    location_keywords = set(LOCATION_KEYWORDS)
    candidate_keywords = set(CANDIDATE_KEYWORDS)
    vote_keywords = {"votes", "total", "sum"}
    if coordinator and hasattr(coordinator, "library"):
        location_keywords.update(set(coordinator.library.get("location_patterns", [])))
        candidate_keywords.update(set(coordinator.library.get("candidate_patterns", [])))
        vote_keywords.update(set(coordinator.library.get("vote_patterns", [])))

    norm_headers = [normalize_text(h) for h in headers]
    location_idx = None
    candidate_idx = None
    vote_idx = None

    # Find likely location, candidate, and vote columns
    for idx, h in enumerate(norm_headers):
        for kw in location_keywords:
            if kw in h:
                location_idx = idx
                logger.info(f"[TABLE BUILDER][extract_rows_and_headers_from_dom] Found location column at index {idx}: {headers[idx]}")
                break
        for kw in candidate_keywords:
            if kw in h:
                candidate_idx = idx
                logger.info(f"[TABLE BUILDER][extract_rows_and_headers_from_dom] Found candidate column at index {idx}: {headers[idx]}")
                break
        for kw in vote_keywords:
            if kw in h:
                vote_idx = idx
                logger.info(f"[TABLE BUILDER][extract_rows_and_headers_from_dom] Found vote column at index {idx}: {headers[idx]}")
                break

    # --- Extra heuristics: all-numeric, all-empty, low-uniqueness columns ---
    sample_rows = []
    for heading, row in repeated_rows[:20]:
        cells = row.locator("> *")
        cell_texts = [cells.nth(i).inner_text().strip() for i in range(cells.count())]
        sample_rows.append(cell_texts)
    logger.info(f"[TABLE BUILDER][extract_rows_and_headers_from_dom] Sample rows for stats: {sample_rows[:3]}")

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
    logger.info(f"[TABLE BUILDER][extract_rows_and_headers_from_dom] Column stats: {col_stats}")

    # Prefer location column: not all-numeric, not all-empty, high uniqueness
    likely_loc = None
    for idx, stat in enumerate(col_stats):
        if stat["empty_ratio"] < 0.8 and stat["numeric_ratio"] < 0.5 and stat["unique_vals"] > 3:
            likely_loc = idx
            logger.info(f"[TABLE BUILDER][extract_rows_and_headers_from_dom] Heuristic: inferred location column at {likely_loc} based on uniqueness/numeric ratio.")
            break
    if likely_loc is not None and (location_idx is None or likely_loc != location_idx):
        location_idx = likely_loc

    # --- ADVANCED: Detect "totals" or "footer" rows and remove them ---
    if sample_rows:
        last_row = sample_rows[-1]
        if any(any(kw in normalize_text(str(cell)) for kw in TOTAL_KEYWORDS.union(MISC_FOOTER_KEYWORDS)) for cell in last_row):
            logger.info("[TABLE BUILDER][extract_rows_and_headers_from_dom] Removing likely totals/footer row at end of data.")
            repeated_rows = repeated_rows[:-1]
            sample_rows = sample_rows[:-1]

    # --- ADVANCED: Remove columns with >90% repeated value (e.g., "Reported", "Yes" everywhere) ---
    always_keep = {"location", "precinct", "district", "ward", "town", "city", "municipal", "county", "% reported", "percent reported", "fully reported"}
    always_keep = {h.lower() for h in always_keep}
    protected_headers = set(h.lower() for h in headers if any(kw in h.lower() for kw in always_keep))

    repeated_val_cols = []
    for idx, stat in enumerate(col_stats):
        h_lower = headers[idx].lower()
        # --- Only remove if all values are empty and not protected
        if stat["unique_vals"] == 1 and stat["empty_ratio"] == 1.0 and h_lower not in protected_headers:
            repeated_val_cols.append(idx)
    if repeated_val_cols:
        logger.info(f"[TABLE BUILDER][extract_rows_and_headers_from_dom] Removing columns that are all empty: {[headers[i] for i in repeated_val_cols]}")
        headers = [h for i, h in enumerate(headers) if i not in repeated_val_cols]
        col_stats = [stat for i, stat in enumerate(col_stats) if i not in repeated_val_cols]

    # --- ADVANCED: Remove all-empty columns, but keep protected headers ---
    # Only remove if all values are empty and not protected, and keep if few columns remain
    min_columns = 3
    non_empty_cols = [i for i, stat in enumerate(col_stats) if stat["empty_ratio"] < 1.0 or headers[i].lower() in protected_headers]
    if len(non_empty_cols) < len(headers) and len(headers) > min_columns:
        logger.info(f"[TABLE BUILDER][extract_rows_and_headers_from_dom] Removing all-empty columns: {[headers[i] for i in range(len(headers)) if i not in non_empty_cols]}")
        headers = [headers[i] for i in non_empty_cols]
        col_stats = [col_stats[i] for i in non_empty_cols]

    # --- ADVANCED: If only one row remains, treat as summary, not table ---
    if len(sample_rows) == 1:
        logger.info("[TABLE BUILDER][extract_rows_and_headers_from_dom] Only one row detected, treating as summary row.")
        data = [dict(zip(headers, sample_rows[0]))]
        logger.info(f"[TABLE BUILDER][extract_rows_and_headers_from_dom] Final headers: {headers}")
        logger.info(f"[TABLE BUILDER][extract_rows_and_headers_from_dom] Final data: {data}")
        return harmonize_headers_and_data(headers, data)

    # --- ADVANCED: If header names are all generic (Column 1, Column 2...), try to infer from first data row ---
    if all(re.match(r"Column \d+", h) for h in headers) and sample_rows:
        logger.info("[TABLE BUILDER][extract_rows_and_headers_from_dom] All headers are generic, inferring from first data row.")
        headers = sample_rows[0]
        sample_rows = sample_rows[1:]

    # --- ADVANCED: Remove rows that are all empty or all repeated values ---
    data = []
    for heading, row in repeated_rows:
        cells = row.locator("> *")
        cell_values = [cells.nth(i).inner_text().strip() for i in range(cells.count())]
        # If we swapped headers, swap cell values accordingly
        if location_idx is not None and location_idx != 0 and len(cell_values) > location_idx:
            cell_values = [cell_values[location_idx]] + cell_values[:location_idx] + cell_values[location_idx+1:]
        elif (candidate_idx == 0 or vote_idx == 0) and location_idx not in (None, 0) and len(cell_values) > location_idx:
            cell_values = [cell_values[location_idx]] + cell_values[:location_idx] + cell_values[location_idx+1:]
        elif 'likely_loc' in locals() and likely_loc is not None and likely_loc != 0 and len(cell_values) > likely_loc:
            cell_values = [cell_values[likely_loc]] + cell_values[:likely_loc] + cell_values[likely_loc+1:]
        # Remove all-empty columns
        if len(cell_values) > len(headers):
            cell_values = cell_values[:len(headers)]
        row_data = {headers[idx]: cell_values[idx] if idx < len(cell_values) else "" for idx in range(len(headers))}
        if row_data and any(v.strip() for v in row_data.values()):
            data.append(row_data)

    logger.info(f"[TABLE BUILDER][extract_rows_and_headers_from_dom] Extracted {len(data)} rows with headers: {headers}")

    # Centralized row filtering
    data = remove_footer_and_summary_rows(data, headers)
    data = remove_outlier_and_empty_rows(data)

    # --- ADVANCED: If still ambiguous, log a warning and save HTML for manual inspection ---
    if len(data) < 2:
        logger.warning("[TABLE BUILDER][extract_rows_and_headers_from_dom] Too few rows after advanced heuristics. Saving HTML for manual inspection.")
        try:
            html = page.content()
            fpath = get_safe_log_path("debug_dom_extract_fallback.html")
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(html)
            logger.info(f"[TABLE BUILDER][extract_rows_and_headers_from_dom] Saved fallback HTML to {fpath}")
        except Exception as e:
            logger.error(f"[TABLE BUILDER][extract_rows_and_headers_from_dom] Could not save fallback HTML: {e}")

    logger.info(f"[TABLE BUILDER][extract_rows_and_headers_from_dom] Finished: {len(data)} rows, {len(headers)} columns.")
    return harmonize_headers_and_data(headers, data)

def extract_with_patterns(page, context=None, log_path=None):
    """
    Attempts to extract tabular data using approved DOM patterns.
    Returns list of (heading, row_locator, pattern_used)
    """
    patterns = load_dom_patterns(log_path)
    approved = [p for p in patterns if p.get("approved")]
    results = []
    for pat in approved:
        selector = pat["selector"]
        # Support multiple cell selectors for robustness
        cell_selectors = pat.get("cell_selectors") or [pat.get("cell_selector", "> *")]
        containers = page.locator(selector)
        for i in range(containers.count()):
            container = containers.nth(i)
            heading = pat.get("heading") or f"Pattern: {selector} #{i+1}"
            for cell_selector in cell_selectors:
                children = container.locator(cell_selector)
                if children.count() > 0:
                    for j in range(children.count()):
                        row = children.nth(j)
                        # Optionally, filter by tag/class/text if specified
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
    return results

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
    log_dir = os.path.join(os.path.dirname(BASE_DIR), "log")
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

def extract_all_candidates_from_data(headers, data):
    candidates = set()
    for row in data:
        val = row.get("Candidate", "")
        # Split on newline or known party/candidate patterns
        for part in val.split("\n"):
            part = part.strip()
            if part and not part.lower().startswith(("democratic", "republican", "working families", "conservative", "green", "libertarian", "independent", "write-in", "other")):
                candidates.add(part)
    return candidates
# 1. ML-based table detection (e.g., using a model to find tables in arbitrary HTML)
def ml_based_table_detection(page, extraction_context=None):
    """
    Use a machine learning model to detect and extract tables from arbitrary HTML.
    Returns a list of (headers, data) tuples.
    """
    try:
        # Example: Assume you have a model or service for table detection
        # hypothetical module
        ml_tables = detect_tables_ml(page.content())
        results = []
        for table_dict in ml_tables:
            headers = table_dict.get("headers", [])
            data = table_dict.get("data", [])
            if headers and data:
                results.append((headers, data))
        return results
    except Exception as e:
        logger.error(f"[ML TABLE DETECTION] Error: {e}")
        return []

# 2. Nested table extraction (see handle_nested_tables)
def nested_table_extraction(page):
    """
    Extract tables that are nested within other tables or complex DOM structures.
    Returns a list of (headers, data) tuples.
    """
    try:
        results = []
        tables = page.locator("table table")
        for i in range(tables.count()):
            table = tables.nth(i)
            if table is not None:
                headers, data, _ = extract_table_data(table)
                if headers and data:
                    results.append((headers, data))
        return results
    except Exception as e:
        logger.error(f"[NESTED TABLE EXTRACTION] Error: {e}")
        return []

# 3. Robust HTML fallback using BeautifulSoup (see robust_html_fallback)
def robust_html_fallback_extraction(page):
    """
    Use BeautifulSoup to parse HTML and extract tables as a last-resort fallback.
    Returns a list of (headers, data) tuples.
    """
    try:
        html = page.content()
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        tables = soup.find_all("table")
        all_tables = []
        for table in tables:
            rows = table.find_all("tr")
            if not rows:
                continue
            headers = [th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])]
            data = []
            for row in rows[1:]:
                cells = row.find_all(["td", "th"])
                data.append({headers[i]: cells[i].get_text(strip=True) if i < len(cells) else "" for i in range(len(headers))})
            if headers and data:
                all_tables.append((headers, data))
        return all_tables
    except Exception as e:
        logger.error(f"[HTML FALLBACK] Error: {e}")
        return []

# 4. Custom per-county or per-state extraction strategies (plug-in architecture)
def custom_plugin_extraction(page, extraction_context=None):
    """
    Use custom extraction plugins based on county/state or other context.
    Returns a list of (headers, data) tuples.
    """
    try:
        plugins = extraction_context.get("plugins") if extraction_context else []
        results = []
        for plugin in plugins:
            try:
                plugin_result = plugin.extract(page, extraction_context)
                if plugin_result:
                    for headers, data in plugin_result:
                        if headers and data:
                            results.append((headers, data))
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
        if extraction_context and extraction_context.get("interactive"):
            print("\n[FEEDBACK] Review extracted headers and data:")
            print("Headers:", headers)
            for i, row in enumerate(data[:5]):
                print(f"Row {i+1}:", row)
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
    
# ===================================================================
# HARMONIZATION & CLEANING
# ===================================================================

def harmonize_headers_and_data(headers: list, data: list) -> tuple:
    """
    Ensures all rows have the same headers, filling missing fields with empty string.
    Deduplicates rows using a composite key of Location, Candidate, and Ballot Type columns.
    Never collapses rows from different locations or ballot types with the same candidate.
    Logs unique values in the detected location column for verification.
    --- Only deduplicate if a valid composite key exists (location and candidate).
    """
    # Deduplicate headers
    all_headers = [h for h in headers if h is not None]
    seen = set()
    deduped_headers = []
    for h in all_headers:
        if h not in deduped_headers:
            deduped_headers.append(h)

    # Detect location column
    location_keywords = {"precinct", "ward", "district", "location", "area", "city", "municipal", "town"}
    location_col = None
    for h in deduped_headers:
        if is_location_header(h):
            location_col = h
            break

    # Detect candidate column
    candidate_col = None
    for h in deduped_headers:
        if "candidate" in h.lower():
            candidate_col = h
            break

    # Detect ballot type columns
    ballot_type_cols = [h for h in deduped_headers if any(bt.lower() in h.lower() for bt in BALLOT_TYPES)]

    # Always include all columns in output, but deduplicate by composite key
    harmonized = []
    seen_keys = set()
    for row in data:
        # --- Only deduplicate if both location_col and candidate_col are present and non-empty
        if location_col and candidate_col and row.get(location_col, "") and row.get(candidate_col, ""):
            key = (
                row.get(location_col, ""),
                row.get(candidate_col, ""),
                *(row.get(bt, "") for bt in ballot_type_cols)
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
        harmonized.append({h: row.get(h, "") for h in deduped_headers})

    # Only prune columns if all values are empty/zero for that column across all rows
    keep = []
    n_rows = len(harmonized)
    for h in deduped_headers:
        if any(row.get(h, "") not in ("", "0") for row in harmonized):
            keep.append(h)
    if not keep and deduped_headers:
        keep = deduped_headers
    harmonized = [{h: row.get(h, "") for h in keep} for row in harmonized]

    # --- Location column detection and logging ---
    unique_locations = set(row.get(location_col, "") for row in harmonized if location_col in row)
    logger.info(f"[HARMONIZE] Unique values in location column '{location_col}': {sorted(unique_locations)}")
    print(f"[HARMONIZE] Unique values in location column '{location_col}': {sorted(unique_locations)}")
    if location_col and len(unique_locations) <= 1:
        logger.warning(f"[HARMONIZE] WARNING: Only one unique value found in location column '{location_col}'. Extraction may be incorrect.")

    return keep, harmonized

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
                if any(row.get(k) == mrow.get(k) and row.get(k) for k in all_headers):
                    match = mrow
                    break
            if match:
                for h in all_headers:
                    if not match.get(h) and row.get(h):
                        match[h] = row[h]
            else:
                merged_data.append(row)
    # Only harmonize once at the end
    return harmonize_headers_and_data(all_headers, merged_data)

def merge_multiline_candidate_rows(headers, data):
    """
    Merge rows where candidate name and party are split across two rows.
    Returns new headers and data.
    """
    if "Candidate" not in headers:
        return headers, data
    merged_data = []
    i = 0
    while i < len(data):
        row = data[i]
        candidate_val = row.get("Candidate", "")
        # If candidate cell has a newline, merge into one cell
        if "\n" in candidate_val:
            parts = [p.strip() for p in candidate_val.split("\n") if p.strip()]
            if len(parts) >= 2:
                merged_name = f"{parts[0]} ({' '.join(parts[1:])})"
                row["Candidate"] = merged_name
            else:
                row["Candidate"] = candidate_val.replace("\n", " ")
            merged_data.append(row)
            i += 1
        # If next row is just a party, merge it
        elif i + 1 < len(data):
            next_row = data[i + 1]
            next_candidate_val = next_row.get("Candidate", "")
            # Only merge if all other columns are empty in next row
            if next_candidate_val and all(not v for k, v in next_row.items() if k != "Candidate"):
                merged_name = f"{candidate_val} ({next_candidate_val})"
                row["Candidate"] = merged_name
                merged_data.append(row)
                i += 2
            else:
                merged_data.append(row)
                i += 1
        else:
            merged_data.append(row)
            i += 1
    return headers, merged_data

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
    Works in concert with DOM/pattern extraction.
    Returns enriched headers, data, and entity_info summary.
    """
    logger.info("[TABLE_CORE][nlp_entity_annotate_table] Starting NLP entity annotation.")
    if not coordinator:
        logger.warning("[TABLE_CORE][nlp_entity_annotate_table] No coordinator provided, skipping NLP annotation.")
        return headers, data, {}

    entity_info = {
        "people": set(),
        "locations": set(),
        "ballot_types": set(),
        "numbers": set(),
        "row_entities": []
    }
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
    logger.info("[TABLE_CORE][verify_table_structure] Verifying table structure using NLP and DOM info.")
    missing = []
    # Check for location
    has_location = bool(entity_info.get("locations")) or any(
        any(lk in h.lower() for lk in LOCATION_KEYWORDS) for h in headers
    )
    if not has_location:
        missing.append("location")
    # Check for candidate/person
    has_candidate = bool(entity_info.get("people")) or any(
        coordinator and any(label == "PERSON" for ent, label in coordinator.extract_entities(h)) for h in headers
    )
    if not has_candidate:
        missing.append("candidate")
    # Check for ballot type
    has_ballot_type = bool(entity_info.get("ballot_types")) or any(
        any(bt.lower() in h.lower() for bt in BALLOT_TYPES) for h in headers
    )
    if not has_ballot_type:
        missing.append("ballot_type")
    # Check for numbers
    has_numbers = bool(entity_info.get("numbers")) or any(
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
    ballot_type_headers = [h for h in headers if any(bt.lower() in h.lower() for bt in BALLOT_TYPES)]
    if not ballot_type_headers:
        logger.warning("[TABLE BUILDER][progressive_table_verification] No ballot type columns detected.")
    else:
        logger.info(f"[TABLE BUILDER][progressive_table_verification] Detected ballot type columns: {ballot_type_headers}")

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
        ballot_vals = [row.get(h, "") for h in ballot_type_headers]
        candidate_vals = [row.get(h, "") for h in candidate_headers]
        logger.info(f"[TABLE BUILDER][progressive_table_verification] Row {i}: location={loc_val}, ballot_types={ballot_vals}, candidates={candidate_vals}")

    # 6. Structure info summary
    structure_info = {
        "location_header": location_header,
        "ballot_type_headers": ballot_type_headers,
        "candidate_headers": candidate_headers,
        "total_header": total_header,
        "verified": all([location_header, ballot_type_headers, candidate_headers, total_header])
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
    logger.info("[TABLE_CORE][detect_table_structure] Analyzing table structure.")
    if entity_info is None:
        entity_info = {}

    # Heuristic: If first column is "Candidate" and the rest are ballot types, it's already wide
    if headers and headers[0].lower() == "candidate" and all(
        any(bt in h.lower() for bt in ["election day", "early voting", "absentee", "mail", "total"]) for h in headers[1:]
    ):
        return {"type": "already-wide", "candidate_col": 0, "ballot_type_cols": list(range(1, len(headers)))}
    # Use entity_info and header heuristics
    candidate_cols = []
    location_cols = []
    ballot_type_cols = []
    for idx, h in enumerate(headers):
        if entity_info.get("people") and any(p in h for p in entity_info["people"]):
            candidate_cols.append(idx)
        if entity_info.get("locations") and any(l in h for l in entity_info["locations"]):
            location_cols.append(idx)
        if entity_info.get("ballot_types") and any(bt in h for bt in entity_info["ballot_types"]):
            ballot_type_cols.append(idx)
        # Fallback: heuristics
        if is_location_header(h):
            location_cols.append(idx)
        if any(bt.lower() in h.lower() for bt in BALLOT_TYPES):
            ballot_type_cols.append(idx)
    # Heuristic: if first col is candidate, columns are ballot types
    if candidate_cols and set(ballot_type_cols) == set(range(1, len(headers))):
        return {"type": "candidate-major", "candidate_col": candidate_cols[0], "ballot_type_cols": ballot_type_cols}
    if location_cols and set(candidate_cols) == set(range(1, len(headers))):
        return {"type": "precinct-major", "location_col": location_cols[0], "candidate_cols": candidate_cols}
    return {"type": "ambiguous", "candidate_cols": candidate_cols, "location_cols": location_cols, "ballot_type_cols": ballot_type_cols}

def handle_candidate_major(headers, data, coordinator, context):
    """
    Handles tables where each row is a candidate, columns are ballot types.
    """
    location_header, percent_header = dynamic_detect_location_header(headers, coordinator)
    if not location_header:
        location_header = "Precinct"
    if not percent_header:
        percent_header = "Percent Reported"
    structure_info = detect_table_structure(headers, data, coordinator)
    candidate_col = structure_info.get("candidate_col", 0)
    ballot_type_cols = structure_info.get("ballot_type_cols", list(range(1, len(headers))))
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
        for idx in ballot_type_cols:
            bt = headers[idx]
            output_headers.append(f"{candidate} ({party}) - {bt}")
        output_headers.append(f"{candidate} ({party}) - Total")
    output_headers.append("Grand Total")
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
            for candidate, party in candidate_party_map.items():
                candidate_total = 0
                for idx in ballot_type_cols:
                    bt = headers[idx]
                    col = f"{candidate} ({party}) - {bt}"
                    val = ""
                    if row[headers[candidate_col]] == candidate:
                        val = row.get(headers[idx], "")
                    try:
                        ival = int(val.replace(",", "")) if val else 0
                    except Exception:
                        ival = 0
                    out_row[col] = str(ival) if val != "" else ""
                    candidate_total += ival
                total_col = f"{candidate} ({party}) - Total"
                out_row[total_col] = str(candidate_total)
                grand_total += candidate_total
        out_row["Grand Total"] = str(grand_total)
        output_data.append(out_row)
    return harmonize_headers_and_data(output_headers, output_data)

def handle_precinct_major(headers, data, coordinator, context):
    """
    Handles tables where each row is a precinct, columns are candidates.
    """
    return pivot_precinct_major_to_wide(headers, data, coordinator, context)

def handle_ambiguous(headers, data, coordinator, context):
    """
    Handles ambiguous tables by trying both handlers and picking the one with more filled data.
    """
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

    # 1. Detect location header robustly, but never use "Candidate"
    location_header = None
    for h in headers:
        if is_location_header(h) and h.lower() != "candidate":
            location_header = h
            break
    if not location_header:
        location_header = "Location"

    # 2. Gather all unique candidates and ballot types
    candidates = set(entity_info.get("people", []))
    ballot_types = set(entity_info.get("ballot_types", []))

    # Fallback: try to infer from headers if not found
    if not candidates:
        # Try to extract from data
        for row in data:
            if "Candidate" in row:
                candidates.add(row["Candidate"])
    if not ballot_types:
        for h in headers:
            if any(bt.lower() in h.lower() for bt in BALLOT_TYPES):
                ballot_types.add(h)
    if not ballot_types:
        ballot_types = set(h for h in headers if h != location_header and h != "Candidate")

    # --- If only one unique location, synthesize from contest title/context ---
    location_values = set(row.get(location_header, "") for row in data if row.get(location_header, ""))
    if len(location_values) <= 1:
        synthetic_location = None
        if context and "contest_title" in context:
            synthetic_location = context["contest_title"]
        elif context and "html_context" in context and "selected_race" in context["html_context"]:
            synthetic_location = context["html_context"]["selected_race"]
        else:
            synthetic_location = "All"
        for row in data:
            row[location_header] = synthetic_location
        logger.warning(f"[TABLE_CORE][pivot_to_wide_format] Only one unique location found. Synthesized location: {synthetic_location}")
        location_values = set([synthetic_location])

    # 3. Build wide headers: always include all candidate/ballot type combos
    wide_headers = [location_header]
    for candidate in sorted(candidates):
        for bt in sorted(ballot_types):
            wide_headers.append(f"{candidate} - {bt}")
    wide_headers.append("Grand Total")

    # 4. Build wide data, one row per unique location
    wide_data = []
    for loc in sorted(location_values):
        out_row = {h: "" for h in wide_headers}
        out_row[location_header] = loc
        grand_total = 0
        for row in data:
            if row.get(location_header, "") == loc:
                candidate = row.get("Candidate", "")
                for bt in ballot_types:
                    key = f"{candidate} - {bt}"
                    val = row.get(bt, "")
                    if val and key in out_row:
                        out_row[key] = val
                        try:
                            grand_total += int(val.replace(",", ""))
                        except Exception:
                            pass
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
    location_header, percent_header = dynamic_detect_location_header(headers, coordinator)
    if not percent_header:
        percent_header = "Percent Reported"

    # Parse headers
    candidate_party_ballot = {}  # (candidate, party) -> {ballot_type: header}
    ballot_types_set = set()
    misc_columns = []
    candidate_party_set = set()

    for h in headers:
        m = re.match(r"(.+?)\s*\((.+?)\)\s*-\s*(.+)", h)
        if m:
            candidate, party, ballot_type = m.groups()
            candidate = candidate.strip()
            party = party.strip()
            ballot_type = ballot_type.strip()
            candidate_party_set.add((candidate, party))
            ballot_types_set.add(ballot_type)
            candidate_party_ballot.setdefault((candidate, party), {})[ballot_type] = h
        else:
            # Try: Candidate - BallotType
            m = re.match(r"(.+?)\s*-\s*(.+)", h)
            if m:
                candidate, ballot_type = m.groups()
                candidate = candidate.strip()
                party = ""
                ballot_type = ballot_type.strip()
                candidate_party_set.add((candidate, party))
                ballot_types_set.add(ballot_type)
                candidate_party_ballot.setdefault((candidate, party), {})[ballot_type] = h
            else:
                # Try: BallotType only (miscellaneous totals)
                ballot_type = h.strip()
                ballot_types_set.add(ballot_type)
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
    for bt in ["Election Day", "Early Voting", "Absentee", "Mail", "Absentee Mail"]:
        if bt in ballot_types_set:
            ballot_types.append(bt)
    for bt in sorted(ballot_types_set):
        if bt not in ballot_types:
            ballot_types.append(bt)

    # Build output headers
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
     # Use patterns from the context library if available
    location_patterns = coordinator.library.get("location_patterns", [
        "precinct", "ward", "district", "city", "municipal", "location", "area",
        "ed", "district name", "division", "subdistrict", "polling place"
    ])
    percent_patterns = coordinator.library.get("percent_patterns", [
        "% precincts reporting", "% reporting", "percent reporting"
    ])

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
    if not location_header:
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
    known_fields = {"candidate", "votes", "percent", "party", "district"}
    return sum(1 for cell in row if cell.lower() in known_fields) >= 2

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
        parent_class = parent.get_attribute("class") or ""
        parent_id = parent.get_attribute("id") or ""
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
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_entry, f, indent=2)
        logger.error(f"[TABLE BUILDER] Failed container logged: {log_path}")
    except Exception as e:
        logger.error(f"[TABLE BUILDER] Could not log failed container: {e}")

def get_safe_log_path(filename="dom_pattern_log.jsonl"):
    """
    Returns a safe log path inside the PROJECT_ROOT/log directory.
    Prevents path-injection and directory traversal.
    """
    # Use the parent of BASE_DIR as the project root
    project_root = os.path.dirname(BASE_DIR)
    log_dir = os.path.join(project_root, "log")
    os.makedirs(log_dir, exist_ok=True)
    safe_filename = os.path.basename(filename)
    return os.path.join(log_dir, safe_filename)

def suggest_new_row_classes_from_logs(log_dir):
    """
    Analyze failed container logs and suggest new likely row classes/IDs.
    """
    class_counter = Counter()
    parent_counter = Counter()
    for path in glob.glob(os.path.join(log_dir, "failed_container_*.json")):
        with open(path, "r", encoding="utf-8") as f:
            entry = json.load(f)
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
    print("Suggested new row classes:", suggested_classes)
    print("Suggested new row IDs:", suggested_ids)
    return suggested_classes, suggested_ids

def load_dom_patterns(log_path=None):
    """
    Loads all DOM patterns, returns a list of dicts.
    """
    if log_path is None:
        log_path = get_safe_log_path()
    if not os.path.exists(log_path):
        return []
    with open(log_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

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
        LOG_PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "log"))
        log_path = os.path.join(LOG_PARENT_DIR, "table_structure_learning_log.jsonl")
    if not os.path.exists(log_path):
        print("No learned table structures found.")
        return

    entries = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                entries.append(entry)
            except Exception:
                continue

    for idx, entry in enumerate(entries):
        print(f"\n[{idx}] Contest: {entry.get('contest_title')}")
        print(f"    Headers: {entry.get('headers')}")
        print(f"    Context: {entry.get('context')}")
        print(f"    Result: {entry.get('result')}")
        print("-" * 40)

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
                    print("Entry deleted.")
                elif action == "e":
                    new_headers = input("Enter new headers as comma-separated values: ").strip().split(",")
                    entries[idx]["headers"] = [h.strip() for h in new_headers]
                    print("Headers updated.")
                else:
                    print("Unknown action.")
            else:
                print("Invalid entry number.")
        # Save changes
        with open(log_path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
        print("Changes saved.")

def table_signature(headers):
    return hashlib.md5(json.dumps(headers, sort_keys=True).encode()).hexdigest()

def load_table_structure_cache():
    if os.path.exists(TABLE_STRUCTURE_CACHE_PATH):
        with open(TABLE_STRUCTURE_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_table_structure_cache(cache):
    with open(TABLE_STRUCTURE_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

def cache_table_structure(domain, headers, structure):
    cache = load_table_structure_cache()
    sig = f"{domain}:{table_signature(headers)}"
    cache[sig] = structure
    save_table_structure_cache(cache)

def get_cached_table_structure(domain, headers):
    cache = load_table_structure_cache()
    sig = f"{domain}:{table_signature(headers)}"
    return cache.get(sig)

def guess_contest_title(table_headers, known_titles):
    """
    Try to match table headers to known contest titles using fuzzy matching.
    """
    import difflib
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
    import difflib
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
        from bs4 import BeautifulSoup
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
    Robustly determine if a header is a location column using fuzzy, substring, and regex matching.
    """
    header_norm = normalize_for_matching(header)
    for kw in LOCATION_KEYWORDS:
        if fuzzy_in(kw, header_norm) or contains_location_keyword(header_norm, LOCATION_KEYWORDS):
            return True
    return False

# ===================================================================
# END OF FILE
# ===================================================================