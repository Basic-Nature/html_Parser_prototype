# table_builder.py
# ===================================================================
# Election Data Cleaner - Table Extraction and Cleaning Utilities
# Context-integrated version: uses context_library.json for config
# ===================================================================

import re
import os
from typing import List, Dict, Tuple, Any, Optional
from .shared_logger import logger
from .shared_logic import normalize_text

# Load context config for precinct headers, etc.
import json

CONTEXT_LIBRARY_PATH = os.path.join(
    os.path.dirname(__file__), "..", "Context_Integration", "context_library.json"
)
if os.path.exists(CONTEXT_LIBRARY_PATH):
    with open(CONTEXT_LIBRARY_PATH, "r", encoding="utf-8") as f:
        CONTEXT_LIBRARY = json.load(f)
    COMMON_PRECINCT_HEADERS = [h.lower() for h in CONTEXT_LIBRARY.get("common_precinct_headers", [])]
else:
    logger.error("[table_builder] context_library.json not found. Precinct header detection will be limited.")
    COMMON_PRECINCT_HEADERS = []

def is_precinct_header(header: str) -> bool:
    """Check if a header is a precinct/district header."""
    return normalize_text(header) in COMMON_PRECINCT_HEADERS

def extract_table_data(table_locator) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Extracts headers and row data from a Playwright Locator for a table element.
    Returns (headers, data) where headers is a list of column names,
    and data is a list of dicts mapping headers to cell values.
    Ensures all rows have the same headers.
    """
    try:
        # Try to get headers from thead first, then from first tbody row
        headers = []
        header_locator = table_locator.locator("thead tr th")
        if header_locator.count() == 0:
            header_locator = table_locator.locator("tbody tr:first-child th")
        for i in range(header_locator.count()):
            headers.append(header_locator.nth(i).inner_text().strip())
        if not headers:
            raise RuntimeError("No headers found in the table.")

        data = []
        row_locator = table_locator.locator("tbody tr")
        for r_idx in range(row_locator.count()):
            r = row_locator.nth(r_idx)
            # Skip header rows in tbody
            if r.locator("th").count() > 0 and r.locator("td").count() == 0:
                continue
            cell_locator = r.locator("td")
            if cell_locator.count() == 0:
                continue
            cells = [cell_locator.nth(i).inner_text().strip() for i in range(cell_locator.count())]
            row_data = {headers[i]: cells[i] for i in range(min(len(headers), len(cells)))}
            data.append(row_data)
        if not data:
            logger.info(f"[TABLE BUILDER] Extracted 0 rows from the table.")
            raise RuntimeError("No rows found in the table.")
        # Harmonize all rows to have the same headers
        data = harmonize_rows(headers, data)
        return headers, data
    except Exception as e:
        logger.error(f"[ERROR] Failed to extract table from page: {e}")
        return [], []

def harmonize_rows(headers: List[str], data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ensures all rows have the same headers, filling missing fields with empty string.
    """
    return [{h: row.get(h, "") for h in headers} for row in data]

def clean_candidate_name(name: str) -> str:
    """
    Cleans and normalizes candidate names:
    - Strips whitespace and punctuation
    - Handles suffixes (Jr., Sr., II, III, etc.)
    - Capitalizes names properly
    - Removes party abbreviations if attached
    """
    name = name.strip()
    name = re.sub(r"[^\w\s\-\']", '', name)
    suffixes = ['Jr', 'Sr', 'II', 'III', 'IV', 'V']
    parts = name.split()
    if parts and parts[-1].replace('.', '') in suffixes:
        suffix = parts.pop(-1)
        name = ' '.join(parts)
        name = f"{name} {suffix}"
    else:
        name = ' '.join(parts)
    def smart_cap(word):
        if word.lower().startswith("mc") and len(word) > 2:
            return "Mc" + word[2:].capitalize()
        if word.lower().startswith("mac") and len(word) > 3:
            return "Mac" + word[3:].capitalize()
        if "'" in word:
            return "'".join([w.capitalize() for w in word.split("'")])
        if "-" in word:
            return "-".join([w.capitalize() for w in word.split("-")])
        return word.capitalize()
    name = ' '.join(smart_cap(w) for w in name.split())
    return name

def parse_candidate_col(col: str) -> Tuple[str, str, str]:
    """
    Attempts to parse a column header into (candidate, party, method).
    """
    m = re.match(r"(.+?)\s+\((.+?)\)\s*-\s*(.+)", col)
    if m:
        return m.groups()
    parts = col.rsplit(" - ", 2)
    if len(parts) == 2:
        return parts[0], "", parts[1]
    return col, "", ""

def detect_table_orientation(headers: List[str], data: List[Dict[str, Any]]) -> str:
    """
    Returns 'precincts_in_rows' if first header is a precinct/district,
    'candidates_in_rows' if first header is a candidate, else 'unknown'.
    """
    if headers and is_precinct_header(headers[0]):
        return 'precincts_in_rows'
    return 'unknown'

def normalize_header(header: str) -> str:
    return re.sub(r'\s+', ' ', header.strip().lower())

def format_table_data_for_output(
    headers: List[str],
    data: List[Dict[str, Any]],
    handler_options: Optional[dict] = None
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Returns (headers, data) in either wide or long format, based on .env OUTPUT_LONG_FORMAT or handler_options.
    Dynamically detects if the first column is a precinct/district and pivots accordingly.
    handler_options: dict for handler-specific overrides.
    """
    output_long = os.getenv("OUTPUT_LONG_FORMAT", "false").lower() == "true"
    if handler_options and "output_long" in handler_options:
        output_long = handler_options["output_long"]
    if not output_long:
        logger.info("[TABLE BUILDER] Outputting in wide format.")
        return headers, data  # Wide format

    orientation = detect_table_orientation(headers, data)
    logger.info(f"[TABLE BUILDER] Detected table orientation: {orientation}")

    if orientation == 'precincts_in_rows':
        # Candidates are in columns, precincts in rows
        first_header = headers[0]
        long_rows = []
        for row in data:
            precinct = row.get(first_header, "")
            for col in headers[1:]:
                candidate, party, method = parse_candidate_col(col)
                value = row.get(col, "")
                if value and value.strip() not in {"", "-", "0"}:
                    long_rows.append({
                        first_header: precinct,
                        "Candidate": clean_candidate_name(candidate.strip()),
                        "Party": party.strip(),
                        "Method": method.strip(),
                        "Votes": value
                    })
        logger.info(f"[TABLE BUILDER] Converted {len(data)} wide rows to {len(long_rows)} long rows.")
        return [first_header, "Candidate", "Party", "Method", "Votes"], long_rows

    else:
        logger.warning("[TABLE BUILDER] Unknown table orientation; outputting in wide format.")
        return headers, data

def review_and_fill_missing_data(headers: List[str], data: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Review process to catch and fill missing data after initial break.
    Ensures all rows have all headers and attempts to fill missing values with empty string.
    Summarizes missing data warnings at the end.
    """
    data = harmonize_rows(headers, data)
    missing_summary = []
    for idx, row in enumerate(data):
        missing = [h for h in headers if not row.get(h)]
        if missing:
            # Collect warning instead of logging immediately
            missing_summary.append((idx, missing))
    # Log up to 5 individual warnings, then a summary
    for idx, missing in missing_summary[:5]:
        logger.warning(f"[TABLE BUILDER] Row {idx} missing data for columns: {missing}")
    if len(missing_summary) > 5:
        logger.warning(f"[TABLE BUILDER] {len(missing_summary)} rows had missing data (showing first 5).")
    elif missing_summary:
        logger.warning(f"[TABLE BUILDER] {len(missing_summary)} rows had missing data.")
    return headers, data

def score_and_break_on_match(
    candidates: List[Tuple[float, Any]],
    threshold: float,
    logger=None
) -> Optional[Any]:
    """
    Given a list of (score, element) tuples, returns the first element above threshold.
    If none found, returns None. Logs diagnostics.
    """
    candidates = sorted(candidates, reverse=True, key=lambda x: x[0])
    for score, el in candidates:
        if score >= threshold:
            if logger:
                logger.info(f"[TABLE BUILDER] Match found with score {score:.2f}. Breaking early.")
            return el
    if logger:
        logger.warning(f"[TABLE BUILDER] No match found above threshold {threshold}.")
    return None

def calculate_grand_totals(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Sums all numeric columns across a list of parsed precinct rows.
    Returns a 'Grand Total' row.
    Skips fields like 'Precinct' and '% Precincts Reporting'.
    Ensures all keys present in any row are included in the grand total row.
    """
    skip_fields = {"Precinct", "% Precincts Reporting"}
    totals = {}

    all_keys = set()
    for row in rows:
        if isinstance(row, dict):
            all_keys.update(row.keys())
    for key in all_keys:
        if key in skip_fields:
            continue
        totals[key] = 0.0
    for row in rows:
        if not isinstance(row, dict):
            continue
        for k in all_keys:
            if k in skip_fields:
                continue
            v = row.get(k, "")
            if not isinstance(v, str) or not v.strip():
                continue
            try:
                val = float(v.replace(",", "").replace("-", "0"))
            except Exception:
                continue
            totals[k] = totals.get(k, 0) + val
    totals["Precinct"] = "Grand Total"
    totals["% Precincts Reporting"] = "100.00%"
    if "Total" in totals:
        totals["Total"] = str(int(totals["Total"]))
    return {k: (str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)) for k, v in totals.items()}