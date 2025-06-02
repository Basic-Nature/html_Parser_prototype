# table_builder.py
# ===================================================================
# Election Data Cleaner - Table Extraction and Cleaning Utilities
# Context-integrated version: uses ContextCoordinator for config
# ===================================================================
import difflib
import hashlib
import re
import os
import json
from typing import List, Dict, Tuple, Any, Optional
from .logger_instance import logger
from .shared_logger import rprint
from .shared_logic import normalize_text
from ..config import CONTEXT_LIBRARY_PATH
from typing import TYPE_CHECKING
from rich.table import Table

if TYPE_CHECKING:
    from ..Context_Integration.context_coordinator import ContextCoordinator
context_cache = {}
from ..config import BASE_DIR

def get_safe_log_path(filename="dom_pattern_log.jsonl"):
    """
    Returns a safe log path inside the BASE_DIR/log directory.
    Prevents path-injection and directory traversal.
    """
    log_dir = os.path.join(BASE_DIR, "log")
    os.makedirs(log_dir, exist_ok=True)
    safe_filename = os.path.basename(filename)
    return os.path.join(log_dir, safe_filename)

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

# --- Robust Table Type Detection Helpers ---
BALLOT_TYPES = [
    "Election Day", "Early Voting", "Absentee", "Mail", "Provisional", "Affidavit", "Other", "Void"
]
CANDIDATE_KEYWORDS = {"candidate", "candidates", "name", "nominee"}
LOCATION_KEYWORDS = {"precinct", "ward", "district", "location", "area", "city", "municipal"}
TOTAL_KEYWORDS = {"total", "sum", "votes", "overall", "all"}
BALLOT_TYPE_KEYWORDS = {"election day", "early voting", "absentee", "mail", "provisional", "affidavit", "other", "void"}
MISC_FOOTER_KEYWORDS = {"undervote", "overvote", "scattering", "write-in", "blank", "void", "spoiled"}
TABLE_STRUCTURE_CACHE_PATH = os.path.join(BASE_DIR, "parser", "Context_Integration", "Context_Library", "table_structure_cache.json")

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
def is_likely_header(row):
    known_fields = {"candidate", "votes", "percent", "party", "district"}
    return sum(1 for cell in row if cell.lower() in known_fields) >= 2

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
            if len(row) != len(headers):
                logger.warning(f"[TABLE BUILDER] merge_table_data Row length mismatch: {row}")
            # Try to find a matching row (by candidate or location, if possible)
            match = None
            for mrow in merged_data:
                # Simple heuristic: match on all overlapping keys
                if any(row.get(k) == mrow.get(k) and row.get(k) for k in all_headers):
                    match = mrow
                    break
            if match:
                for h in all_headers:
                    if not match.get(h) and row.get(h):
                        match[h] = row[h]
            else:
                merged_data.append({h: row.get(h, "") for h in all_headers})
    return all_headers, merged_data

EXTRACTION_ATTEMPTS = [
    # 1. Standard HTML table extraction
    lambda page, context: extract_table_data(page.locator("table").first) if page.locator("table").count() > 0 else ([], []),
    # 2. Pattern-based extraction using learned DOM patterns
    lambda page, context: (
        (lambda pr: (
            guess_headers_from_row(pr[0][1]),
            [
                {
                    guess_headers_from_row(pr[0][1])[idx] if idx < len(guess_headers_from_row(pr[0][1])) else f"Column {idx+1}": row.locator("> *").nth(idx).inner_text().strip()
                    for idx in range(row.locator("> *").count())
                }
                for _, row, _ in pr
            ]
        ) if pr else ([], []))(extract_with_patterns(page, context))
    ),
    # 3. Extraction from repeated DOM structures (divs, lists, etc.)
    lambda page, context: extract_rows_and_headers_from_dom(page),
    # 4. NLP/keyword-based fallback extraction
    lambda page, context: fallback_nlp_candidate_vote_scan(page),
]

def robust_table_extraction(page, extraction_context=None):
    """
    Attempts all extraction strategies in order, merging partial results.
    Uses extraction_context for all steps and logs context/anomalies.
    DOM pattern extraction is prioritized.
    """
    import types

    def safe_json(obj):
        """Recursively remove non-serializable objects (like functions) from dicts/lists."""
        if isinstance(obj, dict):
            return {k: safe_json(v) for k, v in obj.items() if not isinstance(v, types.FunctionType)}
        elif isinstance(obj, list):
            return [safe_json(v) for v in obj if not isinstance(v, types.FunctionType)]
        else:
            return obj

    headers_list = []
    data_list = []
    extraction_logs = []

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
            if headers and data:
                headers_list.append(headers)
                data_list.append(data)
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
            if headers and data:
                headers_list.append(headers)
                data_list.append(data)
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
        # --- DEBUG LOGGING: Show headers and first 3 rows ---
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
        if headers and data:
            headers_list.append(headers)
            data_list.append(data)
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

    # Merge all results
    if headers_list and data_list:
        merged_headers, merged_data = merge_table_data(headers_list, data_list)
        logger.info(f"[TABLE BUILDER] Merged extraction: {len(merged_data)} rows, {len(merged_headers)} columns.")
        return merged_headers, merged_data

    logger.warning("[TABLE BUILDER] No extraction method succeeded.")
    return [], []

def is_candidate_major_row(headers, data, context):
    # First column is candidate, rest are vote types or totals
    if not headers or not data:
        headers, data = robust_table_extraction(context.get("page"), context)
        if not headers or not data:
            logger.error("[TABLE BUILDER] No data could be extracted from the page.")
            return [], []
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

def build_dynamic_table(
    domain: str,
    headers: List[str],
    data: List[Dict[str, Any]],
    coordinator: "ContextCoordinator",
    context: dict = None,
    max_feedback_loops: int = 3,
    learning_mode: bool = True,
    confirm_table_structure_callback=None
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Robustly builds a uniform, context-aware table with dynamic verification and feedback.
    Handles candidate-major, precinct-major, flat, wide, ambiguous, and edge-case tables.
    All output tables will have:
      - Location column (Precinct/District/Area)
      - Percent Reported column (if available)
      - Candidate columns (Candidate (Party) - BallotType, Candidate (Party) - Total)
      - Grand Total column (if applicable)
    """
    if context is None:
        context = {}

    contest_title = context.get("selected_race") or context.get("contest_title") or context.get("title", "")

    # 0. Try to auto-apply a learned structure from the log/database
    learned_structure = None
    if hasattr(coordinator, "get_table_structure"):
        learned_structure = coordinator.get_table_structure(contest_title, context=context, learning_mode=True)
    # --- NEW: Try DB if not found ---
    if not learned_structure and hasattr(coordinator, "get_table_structure_from_db"):
        learned_structure = coordinator.get_table_structure_from_db(contest_title, context=context)
    if learned_structure:
        output_headers = learned_structure.get("headers", [])
        output_headers, output_data = harmonize_headers_and_data(output_headers, data)
        logger.info(f"[TABLE BUILDER] Auto-applied learned table structure for '{contest_title}'.")
        return output_headers, output_data

    # 1. Extraction fallback if needed
    if not headers or not data:
        headers, data = robust_table_extraction(context.get("page"), context)
        if not headers or not data:
            logger.error("[TABLE BUILDER] No data could be extracted from the page.")
            return [], []

    # 2. Harmonize headers/data
    headers, data = harmonize_headers_and_data(headers, data)
    location_header, percent_header = dynamic_detect_location_header(headers, coordinator)
    if not location_header:
        location_header = "Precinct"
    if not percent_header:
        percent_header = "Percent Reported"

    # 3. Identify candidate/party/ballot type structure
    candidate_party_map = extract_candidates_and_parties(headers, coordinator)
    ballot_types = BALLOT_TYPES.copy()
    # Find all ballot types present in headers
    present_ballot_types = set()
    for h in headers:
        for bt in ballot_types:
            if bt.lower() in h.lower():
                present_ballot_types.add(bt)
    if present_ballot_types:
        ballot_types = [bt for bt in ballot_types if bt in present_ballot_types]

    # 4. Build output headers
    output_headers = []
    if percent_header and percent_header not in output_headers:
        output_headers.append(percent_header)
    if location_header and location_header not in output_headers:
        output_headers.append(location_header)
    candidate_columns = []
    for party in sorted(candidate_party_map.keys()):
        for candidate in sorted(candidate_party_map[party].keys()):
            for bt in ballot_types:
                candidate_columns.append(f"{candidate} ({party}) - {bt}")
            candidate_columns.append(f"{candidate} ({party}) - Total")
        output_headers.extend(candidate_columns)
        candidate_columns = []
    # Add Grand Total if any candidate columns exist
    if any("Total" in h for h in output_headers):
        output_headers.append("Grand Total")

    # 5. Build output rows
    rows_by_location = {}
    for row in data:
        # Determine location value
        location_val = row.get(location_header, "") or row.get("Precinct", "") or row.get("District", "") or "All"
        percent_val = row.get(percent_header, "") if percent_header in row else ""
        if location_val not in rows_by_location:
            rows_by_location[location_val] = {h: "" for h in output_headers}
            rows_by_location[location_val][location_header] = location_val
            if percent_header:
                rows_by_location[location_val][percent_header] = percent_val
        # Fill candidate/party/ballot type values
        for party in candidate_party_map:
            for candidate in candidate_party_map[party]:
                candidate_total = 0
                for bt in ballot_types:
                    col = f"{candidate} ({party}) - {bt}"
                    # Try to find the value in the row by matching header
                    val = ""
                    for h in headers:
                        if candidate in h and party in h and bt in h:
                            val = row.get(h, "")
                            break
                        # Fallback: match candidate and bt only
                        if candidate in h and bt in h and party == "Other":
                            val = row.get(h, "")
                            break
                    try:
                        ival = int(val.replace(",", "")) if val else 0
                    except Exception:
                        ival = 0
                    rows_by_location[location_val][col] = str(ival) if val != "" else ""
                    candidate_total += ival
                # Candidate total
                total_col = f"{candidate} ({party}) - Total"
                rows_by_location[location_val][total_col] = str(candidate_total)
        # Grand total
        grand_total = 0
        for h in output_headers:
            if h.endswith(" - Total"):
                try:
                    grand_total += int(rows_by_location[location_val][h]) if rows_by_location[location_val][h] else 0
                except Exception:
                    pass
        if "Grand Total" in output_headers:
            rows_by_location[location_val]["Grand Total"] = str(grand_total)

    # 6. Add a totals row if more than one location
    if len(rows_by_location) > 1:
        totals_row = {h: "" for h in output_headers}
        totals_row[location_header] = "TOTAL"
        if percent_header:
            totals_row[percent_header] = ""
        for h in output_headers:
            if h not in (location_header, percent_header):
                try:
                    values = [r.get(h, "0").replace(",", "") for r in rows_by_location.values()]
                    if all(v == "" or v.isdigit() or (v.startswith('-') and v[1:].isdigit()) for v in values):
                        totals_row[h] = str(sum(int(v) for v in values if v != ""))
                except Exception:
                    totals_row[h] = ""
        rows_by_location["TOTAL"] = totals_row

    output_data = [rows_by_location[loc] for loc in rows_by_location]
    output_headers, output_data = harmonize_headers_and_data(output_headers, output_data)

    # Prompt user to confirm/correct table structure if in learning mode
    if learning_mode and hasattr(coordinator, "log_table_structure"):
        output_headers, output_data = prompt_user_to_confirm_table_structure(
            output_headers, output_data, domain, contest_title, coordinator
        )

    # 7. Learning mode: preview and log structure if enabled
    contest_title = context.get("selected_race") or context.get("contest_title") or context.get("title", "")
    if learning_mode and hasattr(coordinator, "log_table_structure"):
        should_log = True
        if confirm_table_structure_callback:
            should_log = confirm_table_structure_callback(output_headers)
        else:
            rprint(f"\n[bold yellow][Table Builder] Learned headers for '{contest_title}':[/bold yellow]")
            preview_table = Table(show_header=True, header_style="bold magenta")
            for h in output_headers:
                preview_table.add_column(h)
            for row in output_data[:5]:
                preview_table.add_row(*(str(row.get(h, "")) for h in output_headers))
            rprint(preview_table)
            rprint("[bold cyan]If this looks correct, log this table structure for future auto-application?[/bold cyan] [Y/n]: ", end="")
            resp = input().strip().lower()
            should_log = (resp in ("", "y", "yes"))
        if should_log:
            coordinator.log_table_structure(contest_title, output_headers, context=context)
            cache_table_structure(domain, output_headers, output_headers)
            logger.info(f"[TABLE BUILDER] Logged confirmed table structure for '{contest_title}'.")
    logger.info(f"[TABLE BUILDER] Final dynamic table: {len(output_data)} rows, {len(output_headers)} columns.")
    return output_headers, output_data

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

def dynamic_detect_location_header(headers: List[str], coordinator: "ContextCoordinator") -> Tuple[str, str]:
    """
    Dynamically detect the first and second location columns (e.g., precinct, ward, city, district, municipal).
    Uses context, regex, NER, and library.
    Returns (location_header, percent_reported_header)
    """
    # Use patterns from the context library if available
    location_patterns = coordinator.library.get("location_patterns", [
        "precinct", "ward", "district", "city", "municipal", "location", "area"
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
    known_parties = ["Democratic", "Republican", "Working Families", "Conservative", "Green", "Libertarian", "Independent", "Write-In", "Other"]
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

def harmonize_headers_and_data(headers: List[str], data: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Ensures all rows have the same headers, filling missing fields with empty string.
    """
    all_headers = list(headers)
    for row in data:
        if len(row) != len(headers):
            logger.warning(f"[TABLE BUILDER] harmonize_headers_and_data Row length mismatch: {row}")
        for k in row.keys():
            if k not in all_headers:
                all_headers.append(k)
    harmonized = [{h: row.get(h, "") for h in all_headers} for row in data]
    return all_headers, harmonized

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

from typing import List, Dict, Any, Tuple

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

def extract_repeated_dom_structures(page, container_selectors=None, min_row_count=2, extra_keywords=None):
    """
    Scans the DOM for repeated structures (divs, uls, etc.) that look like tabular data.
    Returns a list of (section_heading, row_locator) tuples.
    """
    if container_selectors is None:
        container_selectors = discover_container_selectors(page, extra_keywords=extra_keywords, min_row_count=min_row_count)
    results = []
    for selector in container_selectors:
        containers = page.locator(selector)
        for i in range(containers.count()):
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
    return results

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

def extract_table_from_headers(headers, data, context):
    """
    Converts headers and data into a uniform table format.
    Harmonizes all rows to the union of all keys, preserving order.
    """
    if not headers or not data:
        headers, data = robust_table_extraction(context.get("page"), context)
        if not headers or not data:
            logger.error("[TABLE BUILDER] No data could be extracted from the page.")
            return [], []
    # Use the order of the first row's keys if possible
    all_headers = list(headers)
    for row in data:
        if len(row) != len(headers):
            logger.warning(f"[TABLE BUILDER] extracttable_from_headers Row length mismatch: {row}")
        for k in row.keys():
            if k not in all_headers:
                all_headers.append(k)
    harmonized = [{h: row.get(h, "") for h in all_headers} for row in data]
    return all_headers, harmonized
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

def fallback_nlp_candidate_vote_scan(page):
    """
    Fallback: scan for elements with candidate-like names and vote-like numbers nearby.
    Returns headers, data.
    """
    import re
    candidate_pattern = re.compile(r"[A-Z][a-z]+ [A-Z][a-z]+")  # crude: "Firstname Lastname"
    vote_pattern = re.compile(r"\d{1,3}(,\d{3})*")
    elements = page.locator("*")
    candidates = []
    votes = []
    for i in range(elements.count()):
        text = elements.nth(i).inner_text().strip()
        if candidate_pattern.match(text):
            candidates.append((i, text))
        elif vote_pattern.fullmatch(text.replace(",", "")):
            votes.append((i, text))
    # Pair candidates and votes by proximity
    data = []
    for idx, cand in candidates:
        # Find nearest vote after candidate
        nearest_vote = next((v for j, v in votes if j > idx), None)
        if nearest_vote:
            data.append({"Candidate": cand, "Votes": nearest_vote})
    headers = ["Candidate", "Votes"]
    logger.info(f"[TABLE BUILDER] nlp candidate vote Final table: {len(data)} rows, {len(headers)} columns (learned structure).")
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
    location_keywords = {"precinct", "ward", "district", "location", "area", "city", "municipal"}
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
    # If the last row has mostly keywords like "total", "sum", "overall", etc., drop it
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
    # If a cell contains a newline, split it and try to assign to adjacent columns
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

    # --- ADVANCED: Remove duplicate headers (case-insensitive) ---
    seen = set()
    deduped_headers = []
    for h in headers:
        h_norm = normalize_text(h)
        if h_norm not in seen:
            deduped_headers.append(h)
            seen.add(h_norm)
    headers = deduped_headers

    # --- ADVANCED: Remove all-empty columns ---
    non_empty_cols = [i for i, stat in enumerate(col_stats) if stat["empty_ratio"] < 1.0]
    if len(non_empty_cols) < len(headers):
        logger.info(f"[TABLE BUILDER] Removing all-empty columns: {[headers[i] for i in range(len(headers)) if i not in non_empty_cols]}")
        headers = [headers[i] for i in non_empty_cols]

    # --- ADVANCED: If only one row remains, treat as summary, not table ---
    if len(sample_rows) == 1:
        logger.info("[TABLE BUILDER] Only one row detected, treating as summary row.")
        return headers, [dict(zip(headers, sample_rows[0]))]

    # --- ADVANCED: If header names are all generic (Column 1, Column 2...), try to infer from first data row ---
    if all(re.match(r"Column \d+", h) for h in headers) and sample_rows:
        logger.info("[TABLE BUILDER] All headers are generic, inferring from first data row.")
        headers = sample_rows[0]
        sample_rows = sample_rows[1:]

    # --- Build data ---
    data = []
    for heading, row in repeated_rows:
        cells = row.locator("> *")
        row_data = {}
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
        for idx in range(len(headers)):
            row_data[headers[idx]] = cell_values[idx] if idx < len(cell_values) else ""
        # Remove rows that are all empty or all repeated values
        if row_data and any(v.strip() for v in row_data.values()):
            data.append(row_data)

    # --- ADVANCED: Remove rows that are all empty or all repeated values ---
    filtered_data = []
    for row in data:
        values = list(row.values())
        if not all(v == "" for v in values) and len(set(values)) > 1:
            filtered_data.append(row)
    if not filtered_data and data:
        filtered_data = data  # fallback

    logger.info(f"[TABLE BUILDER] Extracted rows and headers from DOM (location-first): {len(filtered_data)} rows, {len(headers)} columns.")
    return headers, filtered_data
            
def prompt_user_to_confirm_table_structure(headers, data, domain, contest_title, coordinator):
    """
    Interactive and semi-automated CLI prompt for user to confirm/correct table structure.
    - Suggests corrections using ML/NLP (NER, fuzzy matching, header type prediction)
    - Handles multiple structure candidates (if available)
    - Remembers user choices and denied structures
    - Can auto-accept with high ML confidence
    Returns possibly modified headers and data.
    """
    import copy

    should_log = True
    columns_changed = False
    new_headers = copy.deepcopy(headers)
    denied_structures_path = os.path.join(BASE_DIR, "log", "denied_table_structures.json")
    denied_structures = {}
    # Load denied structures count
    if os.path.exists(denied_structures_path):
        with open(denied_structures_path, "r", encoding="utf-8") as f:
            denied_structures = json.load(f)
    sig = f"{domain}:{table_signature(headers)}"
    denied_count = denied_structures.get(sig, 0)

    # --- ML/NLP suggestions ---
    ml_scores = []
    nlp_suggestions = []
    for h in new_headers:
        score = coordinator.score_header(h, {"contest_title": contest_title})
        ml_scores.append(score)
        # Try NER for header suggestion
        ents = coordinator.extract_entities(h)
        if ents:
            ent, label = ents[0]
            nlp_suggestions.append((h, ent, label))
        else:
            nlp_suggestions.append((h, None, None))

    avg_score = sum(ml_scores) / len(ml_scores) if ml_scores else 0
    auto_accept_threshold = 0.93  # Accept automatically if ML is very confident

    # --- Multiple structure candidates (if available) ---
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
        for h in candidate_headers:
            preview_table.add_column(h)
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

        # --- Auto-accept if ML is very confident ---
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
            # Log as denied
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
                candidate_headers = [h for i, h in enumerate(candidate_headers) if i not in wrong_idxs]
                data = [{h: row.get(h, "") for h in candidate_headers} for row in data]
                columns_changed = True
                structure_candidates[candidate_idx] = candidate_headers
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
            rprint("Enter new names for columns, separated by commas (leave blank to keep current):")
            for idx, h in enumerate(candidate_headers):
                rprint(f"  {idx+1}: {h}")
            renames = input("New names (comma-separated): ").split(",")
            for idx, new_name in enumerate(renames):
                if idx < len(candidate_headers) and new_name.strip():
                    rprint(f"[yellow]Renamed '{candidate_headers[idx]}' to '{new_name.strip()}'[/yellow]")
                    candidate_headers[idx] = new_name.strip()
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

    # --- Save user-confirmed structure for future ML learning ---
    if should_log and hasattr(coordinator, "log_table_structure"):
        coordinator.log_table_structure(contest_title, new_headers, context={"domain": domain})
        cache_table_structure(domain, new_headers, new_headers)
        logger.info(f"[TABLE BUILDER] Logged confirmed table structure for '{contest_title}'.")
        # --- NEW: Save to DB if available ---
        if hasattr(coordinator, "save_table_structure_to_db"):
            coordinator.save_table_structure_to_db(
                contest_title=contest_title,
                headers=new_headers,
                context={"domain": domain},
                ml_confidence=avg_score if 'avg_score' in locals() else None,
                confirmed_by_user=True
            )
    return new_headers, data