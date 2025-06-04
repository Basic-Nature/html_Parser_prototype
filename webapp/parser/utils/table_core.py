import os
import json
import re
from typing import List, Dict, Any, Tuple
from .shared_logger import logger
import unicodedata
import glob
from collections import Counter
from ..config import BASE_DIR
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..Context_Integration.context_coordinator import ContextCoordinator
context_cache = {}

BALLOT_TYPES = [
    "Election Day", "Early Voting", "Absentee", "Mail", "Provisional", "Affidavit", "Other", "Void"
]


LOCATION_KEYWORDS = {"precinct", "ward", "district", "location", "area", "city", "municipal", "town"}
TOTAL_KEYWORDS = {"total", "sum", "votes", "overall", "all"}

MISC_FOOTER_KEYWORDS = {"undervote", "overvote", "scattering", "write-in", "blank", "void", "spoiled"}


def extract_table_data(table) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Extracts headers and data from a Playwright table locator.
    Handles malformed HTML, empty tables, and logs errors.
    """
    logger.info("[TABLE BUILDER][extract_table_data] Starting table extraction.")
    headers = []
    data = []
    try:
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

        rows = table.locator("tbody tr")
        logger.info(f"[TABLE BUILDER][extract_table_data] Found {rows.count()} rows in tbody.")
        if rows.count() == 0:
            all_rows = table.locator("tr")
            logger.info(f"[TABLE BUILDER][extract_table_data] No tbody rows, using all tr: {all_rows.count()} rows.")
            if all_rows.count() > 1:
                rows = all_rows.nth(1).locator("xpath=following-sibling::tr")
            else:
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

        if not headers and data:
            max_cols = max(len(row) for row in data)
            headers = [f"Column {i+1}" for i in range(max_cols)]
            logger.warning("[TABLE BUILDER][extract_table_data] No headers but there is data. Generating generic headers.")
            new_data = []
            for row in data:
                if len(row) != len(headers):
                    logger.warning(f"[TABLE BUILDER][extract_table_data] Row length mismatch: {row}")
                new_row = {}
                for idx, h in enumerate(headers):
                    new_row[h] = list(row.values())[idx] if idx < len(row) else ""
                new_data.append(new_row)
            data = new_data

        if not headers and not data:
            logger.warning("[TABLE BUILDER][extract_table_data] Empty table encountered.")
    except Exception as e:
        logger.error(f"[TABLE BUILDER][extract_table_data] Malformed HTML or extraction error: {e}")
        return [], []
    logger.info(f"[TABLE BUILDER][extract_table_data] Finished: {len(data)} rows, {len(headers)} columns.")
    return headers, data

def guess_headers_from_row(row, known_keywords=None):
    """
    Attempts to guess headers from a row's children using keywords or context.
    """
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

def detect_table_structure(headers, data, coordinator):
    """
    Classifies the table structure: candidate-major, precinct-major, or ambiguous.
    Returns a dict with structure type and key indices.
    """
    # Use NLP to detect candidate/party/ballot/location columns
    candidate_cols = []
    party_cols = []
    ballot_type_cols = []
    location_cols = []
    for idx, h in enumerate(headers):
        ents = coordinator.extract_entities(h)
        for ent, label in ents:
            if label in {"PERSON"}:
                candidate_cols.append(idx)
            elif label in {"ORG", "NORP"}:
                party_cols.append(idx)
            elif any(bt.lower() in h.lower() for bt in BALLOT_TYPES):
                ballot_type_cols.append(idx)
            elif label in {"GPE", "LOC", "FAC"} or any(lk in h.lower() for lk in LOCATION_KEYWORDS):
                location_cols.append(idx)
    # Heuristic: if first col is candidate, and columns are ballot types, it's candidate-major
    if candidate_cols and (set(ballot_type_cols) == set(range(1, len(headers)))):
        return {"type": "candidate-major", "candidate_col": candidate_cols[0], "ballot_type_cols": ballot_type_cols}
    # If first col is location, and columns are candidates, it's precinct-major
    if location_cols and (set(candidate_cols) == set(range(1, len(headers)))):
        return {"type": "precinct-major", "location_col": location_cols[0], "candidate_cols": candidate_cols}
    # PATCH: If header set matches common candidate-major pattern, force it
    candidate_major_headers = {"Candidate", "Election Day", "Early Voting", "Absentee Mail", "Total Votes"}
    if set(headers) == candidate_major_headers:
        return {"type": "candidate-major", "candidate_col": 0, "ballot_type_cols": [1, 2, 3]}
    # Fallback: ambiguous
    return {"type": "ambiguous"}

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

def is_likely_header(row):
    known_fields = {"candidate", "votes", "percent", "party", "district"}
    return sum(1 for cell in row if cell.lower() in known_fields) >= 2

def robust_table_extraction(page, extraction_context=None, existing_headers=None, existing_data=None):
    """
    Attempts all extraction strategies in order, merging partial results.
    Uses extraction_context for all steps and logs context/anomalies.
    DOM pattern extraction is prioritized.

    - If existing_headers/data are provided and non-empty, merges new results into them.
    - Never overwrites non-empty headers/data unless extraction fails.
    - Always harmonizes headers after merging.
    """
    import types

    def safe_json(obj):
        """Recursively remove non-serializable objects (like functions, classes, custom objects) from dicts/lists."""
        import types
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

    headers_list = []
    data_list = []
    extraction_logs = []

    # If existing headers/data are provided and non-empty, add them first
    if existing_headers and existing_data and len(existing_headers) > 0 and len(existing_data) > 0:
        headers_list.append(existing_headers)
        data_list.append(existing_data)

    def log_page_html(page, context, prefix=""):
        """Save the current page HTML for debugging extraction issues."""
        try:
            html = page.content()
            contest_title = context.get("selected_race") or context.get("contest_title") or "unknown"
            safe_title = re.sub(r'[^a-zA-Z0-9_\-]', '_', contest_title)[:40]
            fname = f"debug_{prefix}{safe_title}.html"
            fpath = os.path.join(BASE_DIR, "log", fname)
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(html)
            logger.info(f"[TABLE BUILDER] Saved page HTML to {fpath}")
        except Exception as e:
            logger.error(f"[TABLE BUILDER] Could not save page HTML: {e}")

    # 1. Pattern-based extraction (approved DOM patterns, prioritized)
    try:
        pattern_rows = extract_with_patterns(page, extraction_context)
        if pattern_rows:
            headers = guess_headers_from_row(pattern_rows[0][1])
            data = []
            for heading, row, pat in pattern_rows:
                cells = row.locator("> *")
                row_data = {}
                for idx in range(cells.count()):
                    row_data[headers[idx] if idx < len(headers) else f"Column {idx+1}"] = cells.nth(idx).inner_text().strip()
                if row_data:
                    data.append(row_data)
            extraction_logs.append({
                "method": "pattern",
                "headers": headers,
                "rows": len(data),
                "columns": len(headers),
                "success": bool(headers and data),
                "context": extraction_context
            })
            if headers == ["Label", "Votes"] and data:
                possible_headers = list(data[0].values())
                if all(isinstance(h, str) and len(h) > 0 for h in possible_headers):
                    headers = possible_headers
                    data = data[1:]
    except Exception as e:
        logger.error(f"[TABLE BUILDER] Pattern extraction failed: {e}")
        extraction_logs.append({
            "method": "pattern",
            "error": str(e),
            "success": False,
            "context": extraction_context
        })

    # 2. Standard HTML table extraction
    try:
        tables = page.locator("table")
        for i in range(tables.count()):
            headers, data = extract_table_data(tables.nth(i))
            extraction_logs.append({
                "method": "table",
                "headers": headers,
                "rows": len(data),
                "columns": len(headers),
                "success": bool(headers and data),
                "context": extraction_context
            })
            if headers == ["Label", "Votes"] and data:
                possible_headers = list(data[0].values())
                if all(isinstance(h, str) and len(h) > 0 for h in possible_headers):
                    headers = possible_headers
                    data = data[1:]
    except Exception as e:
        logger.error(f"[TABLE BUILDER] Table extraction failed: {e}")
        extraction_logs.append({
            "method": "table",
            "error": str(e),
            "success": False,
            "context": extraction_context
        })

    # 3. Repeated DOM structures (divs, lists, etc.)
    try:
        headers, data = extract_rows_and_headers_from_dom(page)
        logger.info(f"[TABLE BUILDER] DOM structure headers: {headers}")
        logger.info(f"[TABLE BUILDER] First 3 rows: {data[:3]}")
        extraction_logs.append({
            "method": "repeated_dom",
            "headers": headers,
            "rows": len(data),
            "columns": len(headers),
            "success": bool(headers and data),
            "context": extraction_context
        })
        if headers == ["Label", "Votes"] and data:
            possible_headers = list(data[0].values())
            if all(isinstance(h, str) and len(h) > 0 for h in possible_headers):
                headers = possible_headers
                data = data[1:]
    except Exception as e:
        logger.error(f"[TABLE BUILDER] Repeated DOM extraction failed: {e}")
        extraction_logs.append({
            "method": "repeated_dom",
            "error": str(e),
            "success": False,
            "context": extraction_context
        })

    # 4. NLP/keyword-based fallback extraction
    try:
        headers, data = fallback_nlp_candidate_vote_scan(page)
        # PATCH: If first data row looks like headers, use it
        if headers == ["Label", "Votes"] and data:
            possible_headers = list(data[0].values())
            if all(isinstance(h, str) and len(h) > 0 for h in possible_headers):
                headers = possible_headers
                data = data[1:]
        extraction_logs.append({
            "method": "nlp_fallback",
            "headers": headers,
            "rows": len(data),
            "columns": len(headers),
            "success": bool(headers and data),
            "context": extraction_context
        })
        if headers and data:
            headers_list.append(headers)
            data_list.append(data)
            logger.warning("[TABLE BUILDER] Fallback NLP extraction used. Only candidate/vote pairs extracted.")
    except Exception as e:
        logger.error(f"[TABLE BUILDER] NLP fallback extraction failed: {e}")
        extraction_logs.append({
            "method": "nlp_fallback",
            "error": str(e),
            "success": False,
            "context": extraction_context
        })

    # --- Safe JSON logging ---
    logger.info(f"[TABLE BUILDER] Extraction summary: {json.dumps(safe_json(extraction_logs), indent=2)}")

    # Merge all results (including any existing headers/data)
    if headers_list and data_list:
        merged_headers, merged_data = merge_table_data(headers_list, data_list)
        merged_headers, merged_data = harmonize_headers_and_data(merged_headers, merged_data)
        logger.info(f"[TABLE BUILDER] Merged extraction: {len(merged_data)} rows, {len(merged_headers)} columns.")

        # PATCH: Add progressive verification and interactive feedback loop
        headers, data, structure_info = progressive_table_verification(merged_headers, merged_data, extraction_context.get("coordinator"), extraction_context)
        if not structure_info.get("verified"):
            headers, data, structure_info = interactive_feedback_loop(headers, data, structure_info)
        return headers, data

    logger.warning("[TABLE BUILDER] No extraction method succeeded.")
    return [], []

def harmonize_headers_and_data(headers: list, data: list) -> tuple:
    """
    Ensures all rows have the same headers, filling missing fields with empty string.
    Deduplicates headers and prunes empty/zero columns.
    """
    all_headers = [h for h in headers if h is not None]
    # Only expand headers from the input headers; do not expand from row keys here.
    # Let prune_empty_or_zero_columns and prioritize_real_data_columns handle column filtering and ordering.
    # Deduplicate headers
    seen = set()
    deduped_headers = []
    for h in all_headers:
        norm = normalize_header_name(h)
        if norm not in seen:
            deduped_headers.append(h)
            seen.add(norm)
    harmonized = [{h: row.get(h, "") for h in deduped_headers} for row in data]
    # Prune empty/zero columns
    pruned_headers, harmonized = prune_empty_or_zero_columns(deduped_headers, harmonized)
    # Prioritize columns with real data
    prioritized_headers, harmonized = prioritize_real_data_columns(pruned_headers, harmonized)
    return prioritized_headers, harmonized
        
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
    location_keywords = {"precinct", "ward", "district", "location", "area", "city", "municipal", "ed", "division", "subdistrict", "polling place"}
    candidate_keywords = {"candidate", "name", "nominee"}
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
    repeated_val_cols = []
    for idx, stat in enumerate(col_stats):
        if stat["unique_vals"] == 1 and stat["empty_ratio"] < 0.9:
            repeated_val_cols.append(idx)
    if repeated_val_cols:
        logger.info(f"[TABLE BUILDER][extract_rows_and_headers_from_dom] Removing columns with only repeated values: {[headers[i] for i in repeated_val_cols]}")
        headers = [h for i, h in enumerate(headers) if i not in repeated_val_cols]
        col_stats = [stat for i, stat in enumerate(col_stats) if i not in repeated_val_cols]

    # --- ADVANCED: Detect and merge multi-line cells (e.g., candidate name and party in one cell) ---
    def split_multiline_cells(row):
        new_row = []
        for cell in row:
            if "\n" in cell:
                parts = [p.strip() for p in cell.split("\n") if p.strip()]
                new_row.extend(parts)
            else:
                new_row.append(cell)
        return new_row
    sample_rows = [split_multiline_cells(row) for row in sample_rows]

    # --- ADVANCED: If all columns are numeric except one, that one is likely the label/location ---
    numeric_cols = [i for i, stat in enumerate(col_stats) if stat["numeric_ratio"] > 0.8]
    if len(numeric_cols) == len(headers) - 1:
        non_numeric_idx = [i for i in range(len(headers)) if i not in numeric_cols][0]
        if location_idx is None or location_idx != non_numeric_idx:
            logger.info(f"[TABLE BUILDER][extract_rows_and_headers_from_dom] Only one non-numeric column at {non_numeric_idx}, using as location.")
            location_idx = non_numeric_idx

    # --- ADVANCED: If first column is not a location, but another column is, swap them ---
    if location_idx is not None and location_idx != 0:
        logger.info(f"[TABLE BUILDER][extract_rows_and_headers_from_dom] Swapping column {location_idx} ('{headers[location_idx]}') to front as location column.")
        headers = [headers[location_idx]] + headers[:location_idx] + headers[location_idx+1:]
        for row in sample_rows:
            if len(row) > location_idx:
                row.insert(0, row.pop(location_idx))
        norm_headers = [normalize_text(h) for h in headers]

    # --- ADVANCED: Remove all-empty columns ---
    non_empty_cols = [i for i, stat in enumerate(col_stats) if stat["empty_ratio"] < 1.0]
    if len(non_empty_cols) < len(headers):
        logger.info(f"[TABLE BUILDER][extract_rows_and_headers_from_dom] Removing all-empty columns: {[headers[i] for i in range(len(headers)) if i not in non_empty_cols]}")
        headers = [headers[i] for i in non_empty_cols]

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
        if label:
            data.append({"Label": label, "Votes": vote_val})
    headers = ["Label", "Votes"]
    logger.info(f"[TABLE BUILDER] Robust NLP fallback: {len(data)} rows, {len(headers)} columns.")
    return headers, data

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
                        results.append((heading, row, pat))
    return results

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

def prune_empty_or_zero_columns(headers, data, required_cols=None, min_data_threshold=0.05):
    """
    Remove columns where all values are empty or zero, unless required.
    Also remove columns with (Other) if all values are empty/zero.
    Optionally, flag columns with less than min_data_threshold non-empty/non-zero values.
    """
    if required_cols is None:
        required_cols = {"Grand Total", "Precinct", "Percent Reported", "Location"}
    keep = []
    flagged = []
    n_rows = len(data)
    for h in headers:
        col_vals = [row.get(h, "") for row in data]
        non_empty = [v for v in col_vals if v not in ("", "0", 0, None)]
        # Remove if all values are empty or zero, unless required
        if h in required_cols:
            keep.append(h)
        elif "(Other)" in h and all(v in ("", "0", 0, None) for v in col_vals):
            continue
        elif n_rows > 0 and len(non_empty) / n_rows < min_data_threshold:
            flagged.append(h)
        elif any(v not in ("", "0", 0, None) for v in col_vals):
            keep.append(h)
    # Optionally, print or log flagged columns for review
    if flagged:
        logger.info(f"[TABLE BUILDER] Columns flagged for low data: {flagged}")
    new_data = [{h: row.get(h, "") for h in keep} for row in data]
    return keep, new_data

def prioritize_real_data_columns(headers, data, required_cols=None):
    """
    Move columns with real data to the front for user review.
    """
    if required_cols is None:
        required_cols = {"Grand Total", "Precinct", "Percent Reported", "Location"}
    n_rows = len(data)
    def col_score(h):
        vals = [row.get(h, "") for row in data]
        return sum(1 for v in vals if v not in ("", "0", 0, None))
    scored = sorted(headers, key=lambda h: (h not in required_cols, -col_score(h)))
    new_data = [{h: row.get(h, "") for h in scored} for row in data]
    return scored, new_data

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
                        results.append((heading, row))
            except Exception as e:
                log_failed_container(page, container, selector, i, str(e))
    return results

def normalize_text(text):
    """
    Normalize text for comparison: lowercase, strip, remove accents.
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.strip().lower()
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    return text

def log_failed_container(page, container, selector, idx, error_msg):
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
    """
    filtered = []
    for row in data:
        values = list(row.values())
        if any("total" in str(v).lower() or "summary" in str(v).lower() for v in values):
            continue
        if set(normalize_header_name(h) for h in headers) == set(normalize_header_name(str(v)) for v in values):
            continue
        filtered.append(row)
    return filtered

def remove_outlier_and_empty_rows(data, min_non_empty=2):
    """
    Remove rows with too many empty or repeated values.
    """
    filtered = []
    for row in data:
        values = list(row.values())
        non_empty = [v for v in values if v not in ("", None)]
        if len(non_empty) >= min_non_empty and len(set(values)) > 1:
            filtered.append(row)
    return filtered

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

def interactive_feedback_loop(headers, data, structure_info):
    """
    If structure_info['verified'] is False, interactively prompt the user to correct or confirm table structure.
    Returns possibly corrected headers and data.
    """
    import pprint
    print("\n[FEEDBACK] Table structure could not be fully verified.")
    print("Detected structure:")
    pprint.pprint(structure_info)
    print("\nSample headers:", headers)
    print("Sample row:", data[0] if data else "NO DATA")
    print("\nPlease review the detected columns:")
    print("1. Location column:", structure_info.get("location_header"))
    print("2. Ballot type columns:", structure_info.get("ballot_type_headers"))
    print("3. Candidate columns:", structure_info.get("candidate_headers"))
    print("4. Grand Total column:", structure_info.get("total_header"))
    print("\nIf any are incorrect, enter the correct header names (comma-separated), or press Enter to accept as-is.")

    # Location
    loc = input("Location column (current: {}): ".format(structure_info.get("location_header") or "None"))
    if loc.strip():
        structure_info["location_header"] = loc.strip()

    # Ballot types
    bt = input("Ballot type columns (current: {}): ".format(structure_info.get("ballot_type_headers") or "None"))
    if bt.strip():
        structure_info["ballot_type_headers"] = [b.strip() for b in bt.split(",")]

    # Candidates
    cand = input("Candidate columns (current: {}): ".format(structure_info.get("candidate_headers") or "None"))
    if cand.strip():
        structure_info["candidate_headers"] = [c.strip() for c in cand.split(",")]

    # Grand Total
    tot = input("Grand Total column (current: {}): ".format(structure_info.get("total_header") or "None"))
    if tot.strip():
        structure_info["total_header"] = tot.strip()

    # Optionally, you could re-harmonize or re-verify here
    print("\n[FEEDBACK] Updated structure info:")
    pprint.pprint(structure_info)
    print("Continuing with these settings...\n")
    return headers, data, structure_info