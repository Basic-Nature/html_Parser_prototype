import os
import re
import json
from typing import List, Dict, Any, Tuple
from .shared_logger import logger, rprint
import unicodedata
import time
from rich.table import Table
import glob
import hashlib
from collections import Counter
from .table_builder import (
    extract_table_data,
    guess_headers_from_row,
    harmonize_headers_and_data,
    detect_table_structure,
    handle_candidate_major,
    handle_precinct_major,
    is_likely_header,
    robust_table_extraction
)
from ..config import BASE_DIR
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..Context_Integration.context_coordinator import ContextCoordinator
context_cache = {}

# --- Robust Table Type Detection Helpers ---
BALLOT_TYPES = [
    "Election Day", "Early Voting", "Absentee", "Mail", "Provisional", "Affidavit", "Other", "Void"
]
CANDIDATE_KEYWORDS = {"candidate", "candidates", "name", "nominee"}
LOCATION_KEYWORDS = {"precinct", "ward", "district", "location", "area", "city", "municipal", "town"}
TOTAL_KEYWORDS = {"total", "sum", "votes", "overall", "all"}
BALLOT_TYPE_KEYWORDS = {"election day", "early voting", "absentee", "mail", "provisional", "affidavit", "other", "void"}
MISC_FOOTER_KEYWORDS = {"undervote", "overvote", "scattering", "write-in", "blank", "void", "spoiled"}
TABLE_STRUCTURE_CACHE_PATH = os.path.join(BASE_DIR, "parser", "Context_Integration", "Context_Library", "table_structure_cache.json")

def dynamic_table_extractor(page, context, coordinator):
    # 1. Find all tabular candidates in the DOM
    candidates = find_tabular_candidates(page)
    enriched_candidates = []
    for cand in candidates:
        # 2. Run NLP analysis on headers and sample rows
        cand = analyze_candidate_nlp(cand, coordinator)
        # 3. Score/rank using DOM+NLP features
        cand['score'], cand['rationale'] = score_candidate(cand, context, coordinator)
        enriched_candidates.append(cand)
    # 4. Sort and select best
    enriched_candidates.sort(key=lambda c: c['score'], reverse=True)
    best = enriched_candidates[0] if enriched_candidates else None
    # 5. Feedback loop
    if best:
        best = feedback_and_confirm(best, context, coordinator)
    return best['headers'], best['rows'] if best else ([], [])

# Support functions (case-based, not monolithic)
def find_tabular_candidates(page): ...
def analyze_candidate_nlp(candidate, coordinator): ...

def feedback_and_confirm(candidate, context, coordinator): ...

## Core Extraction & Harmonization ##

def extract_table_data(table) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Extracts headers and data from a Playwright table locator.
    Handles malformed HTML, empty tables, and logs errors.
    """
    headers = []
    data = []
    try:
        header_cells = table.locator("thead tr th")
        if header_cells.count() == 0:
            first_row = table.locator("tr").first
            header_cells = first_row.locator("th, td")
        for i in range(header_cells.count()):
            text = header_cells.nth(i).inner_text().strip()
            headers.append(text if text else f"Column {i+1}")

        rows = table.locator("tbody tr")
        if rows.count() == 0:
            all_rows = table.locator("tr")
            if all_rows.count() > 1:
                rows = all_rows.nth(1).locator("xpath=following-sibling::tr")
            else:
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

        if not headers and data:
            max_cols = max(len(row) for row in data)
            headers = [f"Column {i+1}" for i in range(max_cols)]
            new_data = []
            for row in data:
                if len(row) != len(headers):
                    logger.warning(f"[TABLE BUILDER] No headers but there is data Row length mismatch: {row}")
                new_row = {}
                for idx, h in enumerate(headers):
                    new_row[h] = list(row.values())[idx] if idx < len(row) else ""
                new_data.append(new_row)
            data = new_data

        if not headers and not data:
            logger.warning("[TABLE BUILDER] Empty table encountered.")
    except Exception as e:
        logger.error(f"[TABLE BUILDER] Malformed HTML or extraction error: {e}")
        return [], []
    logger.info(f"[TABLE BUILDER] Extracted final table: {len(data)} rows, {len(headers)} columns (learned structure).")
    return headers, data

def extract_rows_and_headers_from_dom(page, extra_keywords=None, min_row_count=2, coordinator=None):
    """
    Attempts to extract tabular data from repeated DOM structures (divs, etc.).
    Returns headers, data.
    Uses advanced heuristics for ambiguous, malformed, or complex cases.
    """
    repeated_rows = extract_repeated_dom_structures(page, extra_keywords=extra_keywords, min_row_count=min_row_count)
    if not repeated_rows:
        return [], []

    # --- Heuristic header detection block ---
    headers = None
    header_row_idx = None
    for idx, (heading, row) in enumerate(repeated_rows[:10]):
        cells = row.locator("> *")
        cell_texts = [cells.nth(i).inner_text().strip() for i in range(cells.count())]
        # Heuristic: header row if at least 2 known fields or all non-numeric
        if is_likely_header(cell_texts) or all(not re.match(r"^\d+([,.]\d+)?$", c) for c in cell_texts):
            headers = cell_texts
            header_row_idx = idx
            break
    if headers is not None:
        repeated_rows = repeated_rows[header_row_idx + 1 :]
    else:
        headers = guess_headers_from_row(repeated_rows[0][1])

    # --- Merge split header rows (e.g., two header rows) ---
    if len(repeated_rows) > 1:
        first_row_cells = [repeated_rows[0][1].locator("> *").nth(i).inner_text().strip() for i in range(repeated_rows[0][1].locator("> *").count())]
        if all(c.isalpha() or c == "" for c in first_row_cells) and any(c for c in first_row_cells):
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
                break
        for kw in candidate_keywords:
            if kw in h:
                candidate_idx = idx
                break
        for kw in vote_keywords:
            if kw in h:
                vote_idx = idx
                break

    # --- Extra heuristics: all-numeric, all-empty, low-uniqueness columns ---
    sample_rows = []
    for heading, row in repeated_rows[:20]:
        cells = row.locator("> *")
        cell_texts = [cells.nth(i).inner_text().strip() for i in range(cells.count())]
        sample_rows.append(cell_texts)
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

    # Prefer location column: not all-numeric, not all-empty, high uniqueness
    likely_loc = None
    for idx, stat in enumerate(col_stats):
        if stat["empty_ratio"] < 0.8 and stat["numeric_ratio"] < 0.5 and stat["unique_vals"] > 3:
            likely_loc = idx
            break
    if likely_loc is not None and (location_idx is None or likely_loc != location_idx):
        logger.info(f"[TABLE BUILDER] Heuristic: inferred location column at {likely_loc} based on uniqueness/numeric ratio.")
        location_idx = likely_loc

    # --- ADVANCED: Detect "totals" or "footer" rows and remove them ---
    if sample_rows:
        last_row = sample_rows[-1]
        if any(any(kw in normalize_text(str(cell)) for kw in TOTAL_KEYWORDS.union(MISC_FOOTER_KEYWORDS)) for cell in last_row):
            logger.info("[TABLE BUILDER] Removing likely totals/footer row at end of data.")
            repeated_rows = repeated_rows[:-1]
            sample_rows = sample_rows[:-1]

    # --- ADVANCED: Remove columns with >90% repeated value (e.g., "Reported", "Yes" everywhere) ---
    repeated_val_cols = []
    for idx, stat in enumerate(col_stats):
        if stat["unique_vals"] == 1 and stat["empty_ratio"] < 0.9:
            repeated_val_cols.append(idx)
    if repeated_val_cols:
        logger.info(f"[TABLE BUILDER] Removing columns with only repeated values: {[headers[i] for i in repeated_val_cols]}")
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
            logger.info(f"[TABLE BUILDER] Only one non-numeric column at {non_numeric_idx}, using as location.")
            location_idx = non_numeric_idx

    # --- ADVANCED: If first column is not a location, but another column is, swap them ---
    if location_idx is not None and location_idx != 0:
        logger.info(f"[TABLE BUILDER] Swapping column {location_idx} ('{headers[location_idx]}') to front as location column.")
        headers = [headers[location_idx]] + headers[:location_idx] + headers[location_idx+1:]
        for row in sample_rows:
            if len(row) > location_idx:
                row.insert(0, row.pop(location_idx))
        norm_headers = [normalize_text(h) for h in headers]

    # --- ADVANCED: Remove all-empty columns ---
    non_empty_cols = [i for i, stat in enumerate(col_stats) if stat["empty_ratio"] < 1.0]
    if len(non_empty_cols) < len(headers):
        logger.info(f"[TABLE BUILDER] Removing all-empty columns: {[headers[i] for i in range(len(headers)) if i not in non_empty_cols]}")
        headers = [headers[i] for i in non_empty_cols]

    # --- ADVANCED: If only one row remains, treat as summary, not table ---
    if len(sample_rows) == 1:
        logger.info("[TABLE BUILDER] Only one row detected, treating as summary row.")
        data = [dict(zip(headers, sample_rows[0]))]
        return harmonize_headers_and_data(headers, data)

    # --- ADVANCED: If header names are all generic (Column 1, Column 2...), try to infer from first data row ---
    if all(re.match(r"Column \d+", h) for h in headers) and sample_rows:
        logger.info("[TABLE BUILDER] All headers are generic, inferring from first data row.")
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

    # Centralized row filtering
    data = remove_footer_and_summary_rows(data, headers)
    data = remove_outlier_and_empty_rows(data)

    # --- ADVANCED: If still ambiguous, log a warning and save HTML for manual inspection ---
    if len(data) < 2:
        logger.warning("[TABLE BUILDER] Too few rows after advanced heuristics. Saving HTML for manual inspection.")
        try:
            html = page.content()
            fpath = get_safe_log_path("debug_dom_extract_fallback.html")
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(html)
            logger.info(f"[TABLE BUILDER] Saved fallback HTML to {fpath}")
        except Exception as e:
            logger.error(f"[TABLE BUILDER] Could not save fallback HTML: {e}")

    logger.info(f"[TABLE BUILDER] Extracted rows and headers from DOM (location-first): {len(data)} rows, {len(headers)} columns.")
    return harmonize_headers_and_data(headers, data)

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

def harmonize_headers_and_data(headers: list, data: list) -> tuple:
    """
    Ensures all rows have the same headers, filling missing fields with empty string.
    Deduplicates headers and prunes empty/zero columns.
    """
    all_headers = list(headers)
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

# --- ROW FILTERING ---

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

# --- COLUMN FILTERING ---

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

## DOM/NLP Analysis & Scoring ## 

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

def advanced_party_candidate_detection(headers, coordinator):
    """
    Use NER and context to better distinguish between candidate, party, and location columns.
    """
    result = {"candidate": [], "party": [], "location": []}
    for idx, h in enumerate(headers):
        ents = coordinator.extract_entities(h)
        for ent, label in ents:
            if label in {"PERSON"}:
                result["candidate"].append(idx)
            elif label in {"ORG", "NORP"}:
                result["party"].append(idx)
            elif label in {"GPE", "LOC", "FAC"}:
                result["location"].append(idx)
    return result

def infer_column_types(headers, data):
    types = {}
    for h in headers:
        col_vals = [row.get(h, "") for row in data]
        if all(re.fullmatch(r"\d{1,3}(,\d{3})*", v) or v == "" for v in col_vals):
            types[h] = "int"
        elif all(re.fullmatch(r"\d+(\.\d+)?%", v) or v == "" for v in col_vals):
            types[h] = "percent"
        else:
            types[h] = "str"
    return types

# --- COLUMN TYPE INFERENCE ---

def infer_column_types_advanced(headers, data):
    """
    Use statistics to infer column types: numeric, categorical, date, etc.
    """
    import numpy as np
    import dateutil.parser
    types = {}
    for h in headers:
        col_vals = [row.get(h, "") for row in data]
        non_empty = [v for v in col_vals if v not in ("", None)]
        try:
            nums = [float(v.replace(",", "")) for v in non_empty if v.replace(",", "").replace(".", "", 1).isdigit()]
        except Exception:
            nums = []
        if len(nums) > 0 and len(nums) / len(non_empty) > 0.7:
            mean = np.mean(nums)
            std = np.std(nums)
            types[h] = "numeric"
        elif all(is_date_like(v) for v in non_empty):
            types[h] = "date"
        elif len(set(non_empty)) < 10:
            types[h] = "categorical"
        else:
            types[h] = "string"
    return types

# --- HEADER DETECTION HEURISTICS ---

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

def normalize_text(text):
    """
    Normalize text for comparison: lowercase, strip, remove accents.
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.strip().lower()
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    return text

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

def extract_candidates_and_parties(headers: List[str], coordinator: "ContextCoordinator") -> Dict[str, Dict[str, List[str]]]:
    """
    Returns a dict: {party: {candidate: [ballot_types]}}
    """
    # Use coordinator to extract all known parties and ballot types
    known_parties = [
        "Democratic", "DEM", "dem", 
        "Republican", "REP", "rep", 
        "Working Families", "WOR", "wor" 
        "Conservative", "CON", "con", 
        "Green", "GRN", "grn", 
        "Libertarian", "LIB", "lib", 
        "Independent", "IND", "ind",
        "Larouche", "Write-In", "Other"                     
    ]
    ballot_types = BALLOT_TYPES

    # Group headers by candidate/party/ballot type
    candidate_party_map = {}
    for h in headers:
        # Try to parse: Candidate (Party) - BallotType
        m = re.match(r"(.+?)\s*\((.+?)\)\s*-\s*(.+)", h)
        if m:
            candidate, party, ballot_type = m.groups()
        else:
            # Try: Candidate - BallotType
            m = re.match(r"(.+?)\s*-\s*(.+)", h)
            if m:
                candidate, ballot_type = m.groups()
                party = ""
            else:
                candidate, party, ballot_type = h, "", ""
        candidate = candidate.strip()
        party = party.strip()
        ballot_type = ballot_type.strip()
        # Fuzzy match party
        if party:
            best_party, score = max(((p, coordinator.fuzzy_score(party, p)) for p in known_parties), key=lambda x: x[1])
            if score > 80:
                party = best_party
        else:
            # Try to infer party from candidate name using NER
            entities = coordinator.extract_entities(candidate)
            for ent, label in entities:
                if label in {"ORG", "NORP"}:
                    party = ent
                    break
        if not party:
            party = "Other"
        if party not in candidate_party_map:
            candidate_party_map[party] = {}
        if candidate not in candidate_party_map[party]:
            candidate_party_map[party][candidate] = []
        if ballot_type and ballot_type not in candidate_party_map[party][candidate]:
            candidate_party_map[party][candidate].append(ballot_type)
    return candidate_party_map

def entity_linking(header, known_entities):
    """
    Link header to known candidates/parties for normalization.
    """
    import difflib
    best, score = None, 0
    for ent in known_entities:
        s = difflib.SequenceMatcher(None, normalize_header(header), normalize_header(ent)).ratio()
        if s > score:
            best, score = ent, s
    return best if score > 0.8 else header

def score_candidate(candidate, context, coordinator): ...

## Candidate Selection & Feedback ## 
# --- USER FEEDBACK LOOP ---

def prompt_user_to_confirm_table_structure(headers, data, domain, contest_title, coordinator):
    """
    Interactive and semi-automated CLI prompt for user to confirm/correct table structure.
    After any user modification, always harmonize headers and data.
    If ML confidence is low and NLP suggests better header names, auto-apply those suggestions.
    """
    import copy

    should_log = True
    columns_changed = False
    new_headers = copy.deepcopy(headers)
    denied_structures_path = os.path.join(BASE_DIR, "log", "denied_table_structures.json")
    denied_structures = {}
    # Ensure log directory exists before file operations
    denied_structures_dir = os.path.dirname(denied_structures_path)
    os.makedirs(denied_structures_dir, exist_ok=True)
    # Load denied structures count
    if os.path.exists(denied_structures_path):
        with open(denied_structures_path, "r", encoding="utf-8") as f:
            denied_structures = json.load(f)
    sig = f"{domain}:{table_signature(headers)}"
    denied_count = denied_structures.get(sig, 0)

    # Feedback loop for columns repeatedly removed
    removed_columns_log_path = os.path.join(BASE_DIR, "log", "removed_columns_log.json")
    removed_columns_log_dir = os.path.dirname(removed_columns_log_path)
    os.makedirs(removed_columns_log_dir, exist_ok=True)
    if os.path.exists(removed_columns_log_path):
        with open(removed_columns_log_path, "r", encoding="utf-8") as f:
            removed_columns_log = json.load(f)
    else:
        removed_columns_log = {}

    # ML/NLP suggestions
    ml_scores = []
    nlp_suggestions = []
    for h in new_headers:
        score = coordinator.score_header(h, {"contest_title": contest_title})
        ml_scores.append(score)
        ents = coordinator.extract_entities(h)
        if ents:
            ent, label = ents[0]
            nlp_suggestions.append((h, ent, label))
        else:
            nlp_suggestions.append((h, None, None))

    avg_score = sum(ml_scores) / len(ml_scores) if ml_scores else 0
    auto_accept_threshold = 0.93  # Accept automatically if ML is very confident

    # PATCH: If ML confidence is low and NLP suggests better header names, auto-apply those suggestions
    if avg_score < 0.7 and any(ent and ent != h for h, ent, label in nlp_suggestions):
        logger.info("[TABLE BUILDER] ML confidence low and NLP suggests better header names. Auto-applying suggestions.")
        for idx, (h, ent, label) in enumerate(nlp_suggestions):
            if ent and ent != h:
                new_headers[idx] = ent
        new_headers, data = harmonize_headers_and_data(new_headers, data)
        # Recompute ML scores after change
        ml_scores = [coordinator.score_header(h, {"contest_title": contest_title}) for h in new_headers]
        avg_score = sum(ml_scores) / len(ml_scores) if ml_scores else 0

    # Multiple structure candidates (if available)
    structure_candidates = [new_headers]
    # Optionally, try to generate alternative header orders/types using ML/NLP
    alt_headers = []
    for idx, (h, ent, label) in enumerate(nlp_suggestions):
        if ent and ent != h:
            alt = copy.deepcopy(new_headers)
            alt[idx] = ent
            alt_headers.append(alt)
    if alt_headers:
        structure_candidates += alt_headers

    candidate_idx = 0
    while True:
        candidate_headers = structure_candidates[candidate_idx]
        # Show ML/NLP confidence and suggestions
        rprint(f"\n[bold yellow][Table Builder] Candidate structure {candidate_idx+1}/{len(structure_candidates)} for '{contest_title}':[/bold yellow]")
        preview_table = Table(show_header=True, header_style="bold magenta")
        N = min(5, len(data))  # Show up to 5 values, or all if fewer rows
        rprint(f"[bold green]Column content preview (first {N} rows):[/bold green]")
        for h in candidate_headers:
            preview_table.add_column(h)
            values = [str(row.get(h, "")) for row in data[:N]]
            preview_vals = [v if len(v) < 30 else v[:27] + "..." for v in values]
            rprint(f"[cyan]{h}[/cyan]: {preview_vals}")
        for row in data[:5]:
            preview_table.add_row(*(str(row.get(h, "")) for h in candidate_headers))
        rprint(preview_table)
        rprint(f"[cyan]ML average confidence: {avg_score:.2f}[/cyan]")
        if nlp_suggestions:
            rprint("[cyan]NLP suggestions:[/cyan]")
            for h, ent, label in nlp_suggestions:
                if ent and ent != h:
                    rprint(f"  [green]{h}[/green] → [yellow]{ent}[/yellow] ({label})")
        if len(structure_candidates) > 1:
            rprint(f"[cyan]Use [N]ext/[P]revious to cycle through {len(structure_candidates)} candidates.[/cyan]")

        # Auto-accept if ML is very confident
        if avg_score >= auto_accept_threshold:
            rprint("[green]ML confidence is high. Auto-accepting this structure.[/green]")
            new_headers = candidate_headers
            break

        rprint("[bold cyan]Options:[/bold cyan]")
        rprint("  [Y] Accept as correct")
        rprint("  [N] Reject (log as denied structure)")
        rprint("  [C] Mark columns as incorrect (remove)")
        rprint("  [O] Reorder columns")
        rprint("  [R] Rename columns")
        rprint("  [A] Add missing columns")
        if len(structure_candidates) > 1:
            rprint("  [Next] Show next candidate structure")
            rprint("  [Prev] Show previous candidate structure")
        resp = input("Accept, Reject, mark Columns, reorder, Rename, Add, Next, or Prev? [Y/n/c/o/r/a/next/prev]: ").strip().lower()
        if resp in ("", "y", "yes"):
            new_headers = candidate_headers
            should_log = True
            break
        elif resp in ("n", "no"):
            denied_structures[sig] = denied_structures.get(sig, 0) + 1
            with open(denied_structures_path, "w", encoding="utf-8") as f:
                json.dump(denied_structures, f, indent=2)
            logger.info(f"[TABLE BUILDER] User declined to log table structure for '{contest_title}'. Denied {denied_structures[sig]} times.")
            if denied_structures[sig] >= 3:
                logger.warning(f"[TABLE BUILDER] Structure for '{contest_title}' denied {denied_structures[sig]} times. Will not auto-apply in future.")
            retry = input("Would you like to retry correction? [y/N]: ").strip().lower()
            if retry in ("y", "yes"):
                continue
            else:
                return headers, data
        elif resp == "c":
            rprint("Enter column numbers (comma-separated) that are incorrect (starting from 1):")
            for idx, h in enumerate(candidate_headers):
                rprint(f"  {idx+1}: {h}")
            wrong_cols = input("Columns to mark as incorrect: ").strip()
            if wrong_cols:
                wrong_idxs = [int(i)-1 for i in wrong_cols.split(",") if i.strip().isdigit()]
                for idx in wrong_idxs:
                    if 0 <= idx < len(candidate_headers):
                        rprint(f"[red]Column '{candidate_headers[idx]}' marked as incorrect.[/red]")
                        col_name = candidate_headers[idx]
                        removed_columns_log.setdefault(contest_title, {})
                        removed_columns_log[contest_title][col_name] = removed_columns_log[contest_title].get(col_name, 0) + 1
                candidate_headers = [h for i, h in enumerate(candidate_headers) if i not in wrong_idxs]
                data = [{h: row.get(h, "") for h in candidate_headers} for row in data]
                columns_changed = True
                structure_candidates[candidate_idx] = candidate_headers
            with open(removed_columns_log_path, "w", encoding="utf-8") as f:
                json.dump(removed_columns_log, f, indent=2)
        elif resp == "o":
            rprint("Enter new order of columns as space/comma-separated numbers (starting from 1):")
            for idx, h in enumerate(candidate_headers):
                rprint(f"  {idx+1}: {h}")
            order = input("New order: ").replace(",", " ").split()
            try:
                new_order = [candidate_headers[int(i)-1] for i in order if i.strip().isdigit() and 0 < int(i) <= len(candidate_headers)]
                if new_order:
                    candidate_headers = new_order
                    data = [{h: row.get(h, "") for h in candidate_headers} for row in data]
                    columns_changed = True
                    structure_candidates[candidate_idx] = candidate_headers
                    rprint(f"[green]Columns reordered.[/green]")
            except Exception as e:
                rprint(f"[red]Invalid order: {e}[/red]")
        elif resp == "r":
            rprint("Enter column numbers (comma-separated) to rename (starting from 1):")
            for idx, h in enumerate(candidate_headers):
                rprint(f"  {idx+1}: {h}")
            col_nums = input("Columns to rename: ").strip()
            if col_nums:
                rename_idxs = [int(i)-1 for i in col_nums.split(",") if i.strip().isdigit() and 0 <= int(i)-1 < len(candidate_headers)]
                for idx in rename_idxs:
                    old_name = candidate_headers[idx]
                    new_name = input(f"Rename column '{old_name}' to: ").strip()
                    if new_name:
                        rprint(f"[yellow]Renamed '{old_name}' to '{new_name}'[/yellow]")
                        candidate_headers[idx] = new_name
                data = [{h: row.get(h, "") for h in candidate_headers} for row in data]
                columns_changed = True
                structure_candidates[candidate_idx] = candidate_headers
        elif resp == "a":
            rprint("Enter names of columns to add, separated by commas:")
            add_cols = input("Columns to add: ").split(",")
            for col in add_cols:
                col = col.strip()
                if col and col not in candidate_headers:
                    candidate_headers.append(col)
                    for row in data:
                        row[col] = ""
                    rprint(f"[green]Added column '{col}'[/green]")
            columns_changed = True
            structure_candidates[candidate_idx] = candidate_headers
        elif resp in ("next", "nxt"):
            candidate_idx = (candidate_idx + 1) % len(structure_candidates)
            continue
        elif resp in ("prev", "previous"):
            candidate_idx = (candidate_idx - 1) % len(structure_candidates)
            continue
        else:
            rprint("[red]Unknown option. Please try again.[/red]")

        # Always harmonize after user modification
        candidate_headers, data = harmonize_headers_and_data(candidate_headers, data)

    # Save user-confirmed structure for future ML learning
    if should_log and hasattr(coordinator, "log_table_structure"):
        coordinator.log_table_structure(contest_title, new_headers, context={"domain": domain})
        cache_table_structure(domain, new_headers, new_headers)
        logger.info(f"[TABLE BUILDER] Logged confirmed table structure for '{contest_title}'.")
        if hasattr(coordinator, "save_table_structure_to_db"):
            coordinator.save_table_structure_to_db(
                contest_title=contest_title,
                headers=new_headers,
                context={"domain": domain},
                ml_confidence=avg_score if 'avg_score' in locals() else None,
                confirmed_by_user=True
            )
    # Always harmonize before returning
    new_headers, data = harmonize_headers_and_data(new_headers, data)
    return new_headers, data

def interactive_batch_operations(headers, data):
    """
    Allow batch renaming, reordering, or removal of columns in the CLI.
    """
    import copy
    history = []
    while True:
        rprint("\n[bold cyan]Batch Operations: [R]ename, [O]rder, [D]elete, [U]ndo, [Q]uit[/bold cyan]")
        cmd = input("Choose operation: ").strip().lower()
        if cmd == "r":
            rprint("Enter column numbers (comma-separated) to rename:")
            for idx, h in enumerate(headers):
                rprint(f"  {idx+1}: {h}")
            col_nums = input("Columns to rename: ").strip()
            if col_nums:
                rename_idxs = [int(i)-1 for i in col_nums.split(",") if i.strip().isdigit() and 0 <= int(i)-1 < len(headers)]
                history.append((copy.deepcopy(headers), copy.deepcopy(data)))
                for idx in rename_idxs:
                    old_name = headers[idx]
                    new_name = input(f"Rename column '{old_name}' to: ").strip()
                    if new_name:
                        headers[idx] = new_name
                data = [{h: row.get(h, "") for h in headers} for row in data]
        elif cmd == "o":
            rprint("Enter new order of columns as space/comma-separated numbers (starting from 1):")
            for idx, h in enumerate(headers):
                rprint(f"  {idx+1}: {h}")
            order = input("New order: ").replace(",", " ").split()
            try:
                new_order = [headers[int(i)-1] for i in order if i.strip().isdigit() and 0 < int(i) <= len(headers)]
                if new_order:
                    history.append((copy.deepcopy(headers), copy.deepcopy(data)))
                    headers = new_order
                    data = [{h: row.get(h, "") for h in headers} for row in data]
            except Exception as e:
                rprint(f"[red]Invalid order: {e}[/red]")
        elif cmd == "d":
            rprint("Enter column numbers (comma-separated) to delete:")
            for idx, h in enumerate(headers):
                rprint(f"  {idx+1}: {h}")
            del_nums = input("Columns to delete: ").strip()
            if del_nums:
                del_idxs = [int(i)-1 for i in del_nums.split(",") if i.strip().isdigit() and 0 <= int(i)-1 < len(headers)]
                history.append((copy.deepcopy(headers), copy.deepcopy(data)))
                headers = [h for i, h in enumerate(headers) if i not in del_idxs]
                data = [{h: row.get(h, "") for h in headers} for row in data]
        elif cmd == "u":
            if history:
                headers, data = history.pop()
                rprint("[green]Undo successful.[/green]")
            else:
                rprint("[yellow]Nothing to undo.[/yellow]")
        elif cmd == "q":
            break
        else:
            rprint("[red]Unknown option.[/red]")
    return headers, data

def auto_suggest_corrections(headers, data, coordinator):
    """
    Suggest likely corrections based on previous user feedback or ML confidence.
    """
    suggestions = []
    for h in headers:
        score = coordinator.score_header(h, {})
        if score < 0.7:
            suggestions.append((h, "Low ML confidence"))
    # Add more suggestions based on previous feedback logs if available
    return suggestions

# --- ML/NLP INTEGRATION ---

def dynamic_confidence_threshold(history, default=0.93):
    """
    Adjust threshold for auto-accepting structures based on past accuracy.
    """
    # Example: If last 5 were correct, raise threshold, else lower
    if not history:
        return default
    correct = sum(1 for h in history[-5:] if h["accepted"])
    if correct >= 4:
        return min(0.98, default + 0.02)
    elif correct <= 2:
        return max(0.85, default - 0.05)
    return default

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

## Pattern/Selector Discovery & Logging ## 

def find_tables_with_headings(page, dom_segments=None, heading_tags=None, include_section_context=True):
    """
    Finds all tables on the page and pairs each with its nearest heading or ARIA landmark.
    - If dom_segments is provided (from scan_html_for_context), uses that for robust matching.
    - Otherwise, falls back to Playwright DOM traversal.
    - Supports nested sections, ARIA landmarks, and fieldset legends.
    Returns a list of (heading, table_locator) tuples.
    """
    if heading_tags is None:
        heading_tags = ("h1", "h2", "h3", "h4", "h5", "h6")

    results = []

    def extract_text_from_html(html: str) -> str:
        """
        Extracts visible text from an HTML string.
        - Handles tags like <span>, <div>, <a>, <li>, <b>, <strong>, <em>, <u>, <i>, <p>, <br>, <th>, <td>, <button>, <label>, <h1>-<h6>.
        - Strips all tags and returns the concatenated text.
        - Handles nested tags and ignores script/style.
        """
        # Remove script and style blocks
        html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
        # Replace <br> and <br/> with newlines
        html = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
        # Remove all other tags, keeping their content
        text = re.sub(r"<[^>]+>", "", html)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    if dom_segments:
        tables = [seg for seg in dom_segments if seg.get("tag") == "table"]
        for i, table_seg in enumerate(tables):
            heading = None
            section_context = None
            idx = table_seg.get("_idx", None)
            # 1. Walk backwards for nearest heading
            if idx is not None:
                for j in range(idx-1, -1, -1):
                    tag = dom_segments[j].get("tag", "")
                    if tag in heading_tags:
                        heading_html = dom_segments[j].get("html", "")
                        heading = extract_text_from_html(heading_html)
                        break
            # 2. If not found, walk up for ARIA landmarks or section/fieldset
            if not heading and idx is not None:
                # Walk up the DOM tree for section/fieldset/region
                parent_idx = table_seg.get("_parent_idx", None)
                visited = set()
                while parent_idx is not None and parent_idx not in visited:
                    visited.add(parent_idx)
                    parent_seg = dom_segments[parent_idx]
                    tag = parent_seg.get("tag", "")
                    attrs = parent_seg.get("attrs", {})
                    # ARIA region/landmark
                    aria_label = attrs.get("aria-label") or attrs.get("aria-labelledby")
                    role = attrs.get("role", "")
                    if role in ("region", "complementary", "main", "navigation", "search") or aria_label:
                        section_context = aria_label or role
                        break
                    # Section/fieldset/legend
                    if tag in ("section", "fieldset"):
                        # Try to find a legend or heading inside this section
                        for k in range(parent_idx+1, len(dom_segments)):
                            if dom_segments[k].get("_parent_idx") == parent_idx:
                                child_tag = dom_segments[k].get("tag", "")
                                if child_tag == "legend":
                                    heading = extract_text_from_html(dom_segments[k].get("html", ""))
                                    break
                                if child_tag in heading_tags:
                                    heading = extract_text_from_html(dom_segments[k].get("html", ""))
                                    break
                        if heading:
                            break
                        section_context = tag
                        break
                    parent_idx = parent_seg.get("_parent_idx", None)
            # 3. Compose heading with section context if desired
            if not heading:
                heading = f"Precinct {i+1}"
            if include_section_context and section_context:
                heading = f"{section_context}: {heading}"
            # Use Playwright to get the table locator by index
            table_locator = page.locator("table").nth(i)
            results.append((heading, table_locator))
    else:
        # Fallback: Use Playwright only
        tables = page.locator("table")
        for i in range(tables.count()):
            table = tables.nth(i)
            heading = None
            section_context = None
            try:
                # Try ARIA landmarks/regions
                parent = table
                for _ in range(5):  # Walk up to 5 ancestors
                    parent = parent.locator("xpath=..")
                    attrs = parent.evaluate("el => ({'role': el.getAttribute('role'), 'aria-label': el.getAttribute('aria-label'), 'aria-labelledby': el.getAttribute('aria-labelledby'), 'tag': el.tagName.toLowerCase()})")
                    if attrs.get("role") in ("region", "complementary", "main", "navigation", "search") or attrs.get("aria-label"):
                        section_context = attrs.get("aria-label") or attrs.get("role")
                        break
                    if attrs.get("tag") in ("section", "fieldset"):
                        # Try to find a legend or heading inside this section
                        legend = parent.locator("legend")
                        if legend.count() > 0:
                            heading = legend.nth(0).inner_text().strip()
                            break
                        for tag in heading_tags:
                            h = parent.locator(tag)
                            if h.count() > 0:
                                heading = h.nth(0).inner_text().strip()
                                break
                        if heading:
                            break
                        section_context = attrs.get("tag")
                        break
                # Try previous heading sibling
                if not heading:
                    header_locator = table.locator("xpath=preceding-sibling::*[self::h1 or self::h2 or self::h3 or self::h4 or self::h5 or self::h6][1]")
                    if header_locator.count() > 0:
                        heading = header_locator.nth(0).inner_text().strip()
            except Exception:
                pass
            if not heading:
                heading = f"Precinct {i+1}"
            if include_section_context and section_context:
                heading = f"{section_context}: {heading}"
            results.append((heading, table))
    return results

def discover_container_selectors(page, extra_keywords=None, min_row_count=2):
    """
    Dynamically discovers container selectors (divs, sections, etc.) with relevant keywords or tabular structure.
    Returns a list of selectors, ranked by likelihood.
    """
    if extra_keywords is None:
        extra_keywords = ["vote", "result", "candidate", "precinct", "choice", "option", "ballot", "row", "table", "summary"]
    selectors = set()
    class_scores = {}

    all_divs = page.locator("div")
    for i in range(all_divs.count()):
        div = all_divs.nth(i)
        cls = div.get_attribute("class") or ""
        id_ = div.get_attribute("id") or ""
        text = div.inner_text().strip().lower()
        score = 0

        # Score based on keywords in class/id/text
        for kw in extra_keywords:
            if kw in cls.lower() or kw in id_.lower() or kw in text:
                score += 2
        # Score based on number of children (tabular structure)
        children = div.locator("> *")
        if children.count() >= min_row_count:
            score += 2
        # Score based on presence of numbers (votes)
        if any(char.isdigit() for char in text):
            score += 1

        # Build selector and store score
        if cls:
            sel = "div." + ".".join(cls.split())
            class_scores[sel] = class_scores.get(sel, 0) + score
        if id_:
            sel = f"div#{id_}"
            class_scores[sel] = class_scores.get(sel, 0) + score

    # Return selectors sorted by score
    sorted_selectors = [sel for sel, _ in sorted(class_scores.items(), key=lambda x: -x[1])]
    # Add some generic selectors as fallback
    sorted_selectors += ["section", "ul", "ol"]
    return sorted_selectors

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
        
def log_new_dom_pattern(example_html, selector, context=None, log_path=None):
    """
    Logs a new DOM pattern for future learning/updating of extraction logic.
    Uses a safe log path.
    """
    if log_path is None:
        log_path = get_safe_log_path()
    entry = {
        "selector": selector,
        "example_html": example_html,
        "context": context or {}
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

def review_dom_patterns(log_path=None):
    """
    CLI to review, approve, or delete learned DOM patterns.
    """
    if log_path is None:
        log_path = get_safe_log_path()
    if not os.path.exists(log_path):
        print("No learned DOM patterns found.")
        return

    with open(log_path, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    for idx, entry in enumerate(entries):
        print(f"\n[{idx}] Selector: {entry.get('selector')}")
        print(f"    Example HTML: {entry.get('example_html')[:200]}...")
        print(f"    Context: {entry.get('context')}")
        print("-" * 40)

    while True:
        cmd = input("\nEnter entry number to approve/delete, or 'q' to quit: ").strip()
        if cmd.lower() == "q":
            break
        if cmd.isdigit():
            idx = int(cmd)
            if 0 <= idx < len(entries):
                action = input("Approve (a) or Delete (d) this entry? [a/d]: ").strip().lower()
                if action == "d":
                    entries.pop(idx)
                    print("Entry deleted.")
                elif action == "a":
                    entries[idx]["approved"] = True
                    print("Entry approved.")
                else:
                    print("Unknown action.")
            else:
                print("Invalid entry number.")
        # Save changes
        with open(log_path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
        print("Changes saved.")

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

def auto_approve_dom_pattern(selector, log_path=None, min_count=2):
    """
    Auto-approves a pattern if it appears at least min_count times.
    """
    patterns = load_dom_patterns(log_path)
    count = sum(1 for p in patterns if p.get("selector") == selector)
    for p in patterns:
        if p.get("selector") == selector and count >= min_count:
            p["approved"] = True
    # Save back
    if log_path is None:
        log_path = get_safe_log_path()
    with open(log_path, "w", encoding="utf-8") as f:
        for p in patterns:
            f.write(json.dumps(p) + "\n")

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

## Structure Handling ## 
# auto-detection in detect_table_structure ---

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

def is_candidate_major_row(headers, data, coordinator, context):
    # First column is candidate, rest are vote types or totals
    if not headers or not data:
        headers, data = robust_table_extraction(context.get("page"), context)
        if not headers or not data:
            logger.error("[TABLE BUILDER] No data could be extracted from the page.")
            return [], []
    candidate_major_headers = {"Candidate", "Election Day", "Early Voting", "Absentee Mail", "Total Votes"}
    if set(headers) == candidate_major_headers:
        structure_info = {"type": "candidate-major", "candidate_col": 0, "ballot_type_cols": [1, 2, 3]}
    else:
        structure_info = detect_table_structure(headers, data, coordinator)
    logger.info(f"[TABLE BUILDER] Detected table structure: {structure_info}")        
    first_col = normalize_text(headers[0])
    return first_col in CANDIDATE_KEYWORDS and len(data) > 1

def is_candidate_major_col(headers, data, context):
    # First row is vote type, columns are candidates (not location)
    if not headers or not data:
        headers, data = robust_table_extraction(context.get("page"), context)
        return False
    return (
        all(normalize_text(h) not in LOCATION_KEYWORDS for h in headers)
        and any(normalize_text(h) in CANDIDATE_KEYWORDS for h in headers)
    )

def is_precinct_major(headers, coordinator):
    # First column is a location/precinct/district
    location_patterns = set(coordinator.library.get("location_patterns", LOCATION_KEYWORDS))
    return headers and normalize_text(headers[0]) in location_patterns

def is_flat_candidate_table(headers):
    # Only candidate and total columns (no locations)
    if not headers:
        rprint("[red][ERROR] No headers extracted from table. Skipping this table.[/red]")
        return False
    first_col = normalize_text(headers[0])
    return (
        first_col in CANDIDATE_KEYWORDS and
        all(
            any(kw in normalize_text(h) for kw in TOTAL_KEYWORDS.union(CANDIDATE_KEYWORDS))
            for h in headers
        )
    )

def is_single_row_summary(data):
    # Only one row, likely a summary
    return len(data) == 1

def is_candidate_footer(data):
    # Last row contains candidate or misc footer keywords
    if not data or not data[-1]:
        return False
    last_row = data[-1]
    return any(
        any(kw in normalize_text(str(v)) for kw in CANDIDATE_KEYWORDS.union(MISC_FOOTER_KEYWORDS))
        for v in last_row.values()
    )
    
# --- STRUCTURE DETECTION ---

def detect_wide_vs_long(headers, data):
    """
    Detect if table is wide or long format.
    """
    # Heuristic: if there are many columns and few rows, it's wide
    if len(headers) > 10 and len(data) < 10:
        return "wide"
    # If there are few columns and many rows, it's long
    if len(headers) <= 5 and len(data) > 10:
        return "long"
    return "ambiguous"

def classify_ambiguous_tables(headers, data, coordinator):
    """
    Use ML or rules to classify ambiguous structures.
    """
    # Example: Use ML model or rules
    # For now, use NER and heuristics
    col_types = advanced_party_candidate_detection(headers, coordinator)
    if col_types["candidate"] and col_types["location"]:
        return "precinct-major"
    elif col_types["candidate"]:
        return "candidate-major"
    else:
        return "ambiguous"

## Other Utilities ## 

# --- PERFORMANCE & LOGGING ---

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
    
# --- EDGE CASES ---

def handle_nested_tables(page):
    """
    Handle tables within tables or complex nested DOM structures.
    """
    tables = page.locator("table table")
    results = []
    for i in range(tables.count()):
        table = tables.nth(i)
        headers, data = extract_table_data(table)
        results.append((headers, data))
    return results

def review_learned_table_structures(log_path="log/table_structure_learning_log.jsonl"):
    """
    CLI to review/edit learned table structures.
    """
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

## Optional/Advanced ## 

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

def is_date_like(val):
    import dateutil.parser
    try:
        dateutil.parser.parse(val)
        return True
    except Exception:
        return False

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





