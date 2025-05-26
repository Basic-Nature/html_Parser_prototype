# table_builder.py
# ===================================================================
# Election Data Cleaner - Table Extraction and Cleaning Utilities
# ===================================================================

import re
from typing import List, Dict, Tuple, Any
from .shared_logger import logger

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
            logger.info(f"[HTML Handler] Extracted 0 rows from the table.")
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
    # Remove common party abbreviations at the end (if present)
    name = re.sub(r'\b(DEM|REP|IND|LIB|GRE|CON|WFP|NPP|NPA|U|D|R|G|L|I|C|S|N|P|NP|NONPARTISAN|UNAFFILIATED)$', '', name, flags=re.IGNORECASE).strip()
    # Remove extra punctuation except hyphens and apostrophes
    name = re.sub(r"[^\w\s\-\']", '', name)
    # Handle suffixes
    suffixes = ['Jr', 'Sr', 'II', 'III', 'IV', 'V']
    parts = name.split()
    if parts and parts[-1].replace('.', '') in suffixes:
        suffix = parts.pop(-1)
        name = ' '.join(parts)
        name = f"{name} {suffix}"
    else:
        name = ' '.join(parts)
    # Proper capitalization (handles Mc/Mac, O', hyphens, etc.)
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

def parse_candidate_vote_table(
    table_element, 
    current_precinct: str, 
    method_names: List[str], 
    reporting_pct: str = "0.00%"
) -> Dict[str, Any]:
    """
    Converts a DOM table element into a Smart Elections-style row for a single precinct.
    Returns a dict with standardized candidate-method vote fields and metadata.
    Handles edge cases where candidate/party columns are missing or extra columns are present.
    """
    def is_total_row(cell_text):
        totals = {
            "total", "total votes", "total ballots", "total votes cast", "total ballots cast",
            "total votes counted", "total ballots counted", "total votes remaining", "total ballots remaining",
            "total votes outstanding", "total ballots outstanding", "total votes uncounted", "total ballots uncounted",
            "total votes disputed", "total ballots disputed", "total votes invalid", "total ballots invalid",
            "total votes spoiled", "total ballots spoiled", "total votes rejected", "total ballots rejected",
            "total votes canceled", "total ballots canceled", "total votes disqualified", "total ballots disqualified",
            "total votes nullified", "total ballots nullified", "total votes voided", "total ballots voided"
        }
        return cell_text.strip().lower() in totals

    row = {"Precinct": current_precinct, "% Precincts Reporting": reporting_pct}
    try:
        header_locator = table_element.locator('thead tr th')
        if header_locator.count() == 0:
            header_locator = table_element.locator('tbody tr:first-child th')
        row_locator = table_element.locator('tbody tr')
        for r_idx in range(row_locator.count()):
            r = row_locator.nth(r_idx)
            cells = [r.locator('td').nth(i) for i in range(r.locator('td').count())]
            if len(cells) < 2:
                continue
            full_name = cells[0].inner_text().strip()
            if is_total_row(full_name):
                continue

            # Candidate/party extraction
            name_parts = full_name.split()
            candidate_name, party = "", ""
            if len(name_parts) >= 3:
                candidate_name = " ".join(name_parts[1:-1])
                party = name_parts[-1]
            elif len(name_parts) == 2:
                candidate_name, party = name_parts
            elif len(name_parts) == 1 and not name_parts[0].isdigit():
                candidate_name = name_parts[0]
            else:
                candidate_name = full_name

            candidate_name = clean_candidate_name(candidate_name)
            party = party.strip().upper()
            if party in {"DEM", "REP", "IND", "LIB", "GRE"}:
                party = party.title()
            canonical = f"{candidate_name} ({party})" if party else candidate_name

            # Avoid duplicate candidates
            if canonical in row:
                continue

            # Extract method votes and total
            method_votes = [c.inner_text().strip().replace(",", "").replace("-", "0") for c in cells[1:-1]]
            total = cells[-1].inner_text().strip().replace(",", "").replace("-", "0")

            # Validate method votes
            if len(method_votes) != len(method_names):
                logger.debug(f"[TABLE] Number of method votes ({len(method_votes)}) does not match number of method names ({len(method_names)}).")
                continue
            if not re.match(r"^\d+(\.\d+)?$", total):
                logger.debug(f"[TABLE] Total '{total}' is not a valid number.")
                continue

            # Add method votes
            for method, vote in zip(method_names, method_votes):
                if not re.match(r"^\d+(\.\d+)?$", vote):
                    logger.debug(f"[TABLE] Vote '{vote}' is not a valid number.")
                    continue
                row[f"{canonical} - {method}"] = vote
            row[f"{canonical} - Total"] = total

    except Exception as e:
        logger.error(f"[TABLE] Failed to parse candidate vote table: {e}")
    return row

def calculate_grand_totals(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Sums all numeric columns across a list of parsed precinct rows.
    Returns a 'Grand Total' row.
    Skips fields like 'Precinct' and '% Precincts Reporting'.
    Ensures all keys present in any row are included in the grand total row.
    """
    totals = {}
    skip_fields = {
        "Precinct", "% Precincts Reporting", "Total", "Precincts Reporting",
        "Total Votes", "Total Ballots", "Total Votes Cast", "Total Ballots Cast",
        "Total Votes Counted", "Total Ballots Counted", "Total Votes Remaining", "Total Ballots Remaining",
        "Total Votes Outstanding", "Total Ballots Outstanding", "Total Votes Uncounted", "Total Ballots Uncounted",
        "Total Votes Disputed", "Total Ballots Disputed", "Total Votes Invalid", "Total Ballots Invalid",
        "Total Votes Spoiled", "Total Ballots Spoiled", "Total Votes Rejected", "Total Ballots Rejected",
        "Total Votes Canceled", "Total Ballots Canceled", "Total Votes Disqualified", "Total Ballots Disqualified",
        "Total Votes Nullified", "Total Ballots Nullified", "Total Votes Voided", "Total Ballots Voided"
    }
    # Collect all possible keys
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
    # Optionally add a "Total" column if present in data
    if "Total" in totals:
        totals["Total"] = str(int(totals["Total"]))
    # Convert all floats to string for CSV output
    return {k: (str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)) for k, v in totals.items()}