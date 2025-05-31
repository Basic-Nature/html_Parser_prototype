# table_builder.py
# ===================================================================
# Election Data Cleaner - Table Extraction and Cleaning Utilities
# Context-integrated version: uses ContextCoordinator for config
# ===================================================================

import re
import os
import json
from typing import List, Dict, Tuple, Any, Optional
from .logger_instance import logger
from .shared_logger import rprint
from .shared_logic import normalize_text
from ..config import CONTEXT_LIBRARY_PATH
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..Context_Integration.context_coordinator import ContextCoordinator



# --- Robust Table Type Detection Helpers ---
BALLOT_TYPES = [
    "Election Day", "Early Voting", "Absentee", "Mail", "Provisional", "Affidavit", "Other", "Void"
]
CANDIDATE_KEYWORDS = {"candidate", "candidates", "name", "nominee"}
LOCATION_KEYWORDS = {"precinct", "ward", "district", "location", "area", "city", "municipal"}
TOTAL_KEYWORDS = {"total", "sum", "votes", "overall", "all"}
BALLOT_TYPE_KEYWORDS = {"election day", "early voting", "absentee", "mail", "provisional", "affidavit", "other", "void"}
MISC_FOOTER_KEYWORDS = {"undervote", "overvote", "scattering", "write-in", "blank", "void", "spoiled"}

def is_candidate_major_row(headers, data):
    # First column is candidate, rest are vote types or totals
    if not headers or not data:
        return False
    first_col = normalize_text(headers[0])
    return first_col in CANDIDATE_KEYWORDS and len(data) > 1

def is_candidate_major_col(headers, data):
    # First row is vote type, columns are candidates (not location)
    if not headers or not data:
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
    headers: List[str],
    data: List[Dict[str, Any]],
    coordinator: "ContextCoordinator",
    context: dict = None,
    max_feedback_loops: int = 3,
    learning_mode: bool = True,
    confirm_table_structure_callback=None # Optional callback for confirming table structure
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Robustly builds a uniform, context-aware table with dynamic verification and feedback.
    Handles candidate-major, precinct-major, flat, wide, ambiguous, and edge-case tables.
    Now supports table structure learning and auto-application.
    """
    if not headers:
        raise ValueError("No headers provided to build_dynamic_table. Cannot build table.")

    # Always ensure context is a dict
    if context is None:
        context = {}

    contest_title = context.get("selected_race") or context.get("contest_title") or context.get("title", "")

    # --- 0. Learning mode: check for confirmed table structure ---
    if learning_mode and hasattr(coordinator, "get_table_structure"):
        learned_headers = coordinator.get_table_structure(contest_title, context=context, learning_mode=True)
        if learned_headers:
            # Optionally remap your data to match learned_headers
            # (Here, just reorder columns if possible)
            headers = [h for h in learned_headers if h in headers] + [h for h in headers if h not in learned_headers]
            # Optionally, harmonize data as well
            data = [{h: row.get(h, "") for h in headers} for row in data]
            logger.info(f"[TABLE BUILDER] Applied learned table structure for '{contest_title}'.")
            return headers, data

    # Dynamically detect the location header
    location_header, _ = dynamic_detect_location_header(headers, coordinator)

    # If location header is not present in headers, add it as the first column
    if location_header and location_header not in headers:
        headers = [location_header] + headers

    # For each row, if location header is missing or empty, fill from context if available
    for row in data:
        if location_header and (location_header not in row or not row.get(location_header)):
            context_val = context.get("precinct") or context.get("location")
            if context_val:
                row[location_header] = context_val

    # --- Special case: Proposition/Response tables ---
    if headers and normalize_text(headers[0]) in {"response", "choice", "option"}:
        precinct_val = context.get("precinct") or context.get("location")
        if not precinct_val:
            precinct_val = data[0].get(location_header) if data and location_header in data[0] else "All"
        wide_row = {}
        # Only fill location if missing
        if location_header not in wide_row or not wide_row.get(location_header):
            wide_row[location_header] = precinct_val
        for row in data:
            response = row[headers[0]].strip()
            for h in headers[1:]:
                colname = f"{response} - {h}"
                # Only fill if not already present
                if colname not in wide_row:
                    wide_row[colname] = row[h]
        headers = [location_header] + [
            f"{row[headers[0]].strip()} - {h}"
            for row in data for h in headers[1:]
        ]
        data = [wide_row]
        logger.info("[TABLE BUILDER] Handled proposition/response-major table (pivoted Yes/No responses).")
    else:
        # --- 1. Candidate-major table (candidates as rows, vote types as columns) ---
        if is_candidate_major_row(headers, data):
            wide_row = {}
            for row in data:
                candidate = row[headers[0]]
                candidate_clean = candidate.replace('\n', ' ').strip()
                for h in headers[1:]:
                    colname = f"{candidate_clean} - {h}"
                    wide_row[colname] = row[h]
            wide_row[location_header] = context.get("precinct", "All")
            headers = [location_header] + [
                "{} - {}".format(row[headers[0]].replace('\n', ' ').strip(), h)
                for row in data for h in headers[1:]
            ]
            data = [wide_row]
            logger.info("[TABLE BUILDER] Handled candidate-as-row table (vertical candidate-major).")

        # --- 2. Candidate-major table (candidates as columns, vote types as rows) ---
        elif is_candidate_major_col(headers, data):
            vote_type_col = headers[0]
            candidate_headers = headers[1:]
            wide_row = {}
            for row in data:
                vote_type = row[vote_type_col]
                for candidate in candidate_headers:
                    colname = f"{candidate} - {vote_type}"
                    wide_row[colname] = row[candidate]
            wide_row[location_header] = context.get("precinct", "All")
            headers = [location_header] + [
                "{} - {}".format(candidate, row[vote_type_col])
                for row in data for candidate in candidate_headers
            ]
            data = [wide_row]
            logger.info("[TABLE BUILDER] Handled candidate-as-column table (horizontal candidate-major).")

        # --- 3. Candidate names in footer (last row) ---
        elif is_candidate_footer(data):
            candidate_row = data[-1]
            vote_type_rows = data[:-1]
            wide_row = {}
            for row in vote_type_rows:
                for k, v in row.items():
                    candidate = candidate_row.get(k, "").replace("\n", " ").strip()
                    if candidate:
                        colname = f"{candidate} - {k}"
                        wide_row[colname] = v
            wide_row[location_header] = context.get("precinct", "All")
            headers = [location_header] + [
                "{} - {}".format(candidate_row.get(k, '').replace('\n', ' ').strip(), k)
                for k in candidate_row if candidate_row.get(k, "")
            ]
            data = [wide_row]
            logger.info("[TABLE BUILDER] Handled candidate-footer table.")

        # --- 4. Precinct-major table (precinct/location as rows, candidates as columns) ---
        elif is_precinct_major(headers, coordinator):
            logger.info("[TABLE BUILDER] Detected precinct-major table, using as is.")
            headers, data = pivot_precinct_major_to_wide(headers, data, coordinator, context)
            return headers, data

        # --- 5. Flat candidate table (totals only, no locations) ---
        elif is_flat_candidate_table(headers):
            for row in data:
                if location_header not in row or not row.get(location_header):
                    row[location_header] = "All"
            if location_header not in headers:
                headers = [location_header] + headers
            logger.info("[TABLE BUILDER] Detected flat candidate table, added dummy location.")

        # --- 6. Single-row summary table ---
        elif is_single_row_summary(data):
            if location_header not in data[0] or not data[0].get(location_header):
                data[0][location_header] = "All"
            if location_header not in headers:
                headers = [location_header] + headers
            logger.warning("[TABLE BUILDER] Only one row found, treating as totals for all locations.")

        # --- 7. Ambiguous or wide/summary table ---
        else:
            known_candidates = coordinator.get_candidates(context.get("selected_race", ""))
            if headers and any(normalize_text(headers[0]) == normalize_text(c) for c in known_candidates):
                logger.info("[TABLE BUILDER] Ambiguous table: treating as candidate-major based on known candidates.")
                # Example: only fill guessed columns if missing
                for row in data:
                    for candidate in known_candidates:
                        if candidate not in row or not row.get(candidate):
                            row[candidate] = ""
            else:
                logger.warning("[TABLE BUILDER] Ambiguous table structure, treating first column as location.")

        location_header, percent_header = dynamic_detect_location_header(headers, coordinator)
        logger.info(f"[TABLE BUILDER] Detected location header: {location_header}, percent header: {percent_header}")

        candidate_party_map = extract_candidates_and_parties(headers, coordinator)

        # 3. Rescan and verify headers/data, feedback loop
        feedback_loops = 0
        passed = False
        while not passed and feedback_loops < max_feedback_loops:
            headers, data, passed = rescan_and_verify(headers, data, coordinator, context)
            logger.info(f"[TABLE BUILDER] Verification pass {feedback_loops+1}: {'PASS' if passed else 'FAIL'} (score threshold met: {passed})")
            feedback_loops += 1
            if not passed:
                logger.warning("[TABLE BUILDER] Feedback loop triggered: retrying header/data extraction.")

        # 4. Build uniform wide-format table
        output_headers = []
        if percent_header:
            if percent_header not in rows_by_location[location] or not rows_by_location[location][percent_header]:
                rows_by_location[location][percent_header] = row.get(percent_header, "Unknown")
            output_headers.append(percent_header)
        output_headers.append(location_header)

        # Build candidate columns: party -> candidate -> ballot types
        for party in sorted(candidate_party_map.keys()):
            for candidate in sorted(candidate_party_map[party].keys()):
                for ballot_type in BALLOT_TYPES:
                    col = f"{candidate} ({party}) - {ballot_type}"
                    if col not in output_headers:
                        output_headers.append(col)
                # Candidate total
                total_col = f"{candidate} ({party}) - Total"
                if total_col not in output_headers:
                    output_headers.append(total_col)
            # Party total
            party_total_col = f"{party} - Total"
            if party_total_col not in output_headers:
                output_headers.append(party_total_col)

        # Grand total
        if "Grand Total" not in output_headers:
            output_headers.append("Grand Total")

        # 5. Build rows: one per location (district/precinct/etc.)
        rows_by_location = {}
        for row in data:
            location = row.get(location_header, "")
            if location not in rows_by_location:
                rows_by_location[location] = {h: "" for h in output_headers}
                if percent_header:
                    rows_by_location[location][percent_header] = row.get(percent_header, "")
                rows_by_location[location][location_header] = location

            # Fill candidate/party/ballot type values
            for party in candidate_party_map:
                for candidate in candidate_party_map[party]:
                    candidate_total = 0
                    for ballot_type in BALLOT_TYPES:
                        col = f"{candidate} ({party}) - {ballot_type}"
                        # Only fill if not already present
                        if not rows_by_location[location][col]:
                            value = ""
                            for h in headers:
                                if candidate in h and party in h and ballot_type in h:
                                    value = row.get(h, "")
                                    break
                            rows_by_location[location][col] = value
                        try:
                            candidate_total += int(rows_by_location[location][col].replace(",", "")) if rows_by_location[location][col] else 0
                        except Exception:
                            pass
                    # Candidate total
                    total_col = f"{candidate} ({party}) - Total"
                    rows_by_location[location][total_col] = str(candidate_total)
                # Party total
                party_total_col = f"{party} - Total"
                party_total = sum(
                    int(rows_by_location[location][f"{candidate} ({party}) - Total"] or 0)
                    for candidate in candidate_party_map[party]
                )
                rows_by_location[location][party_total_col] = str(party_total)
            # Grand total
            grand_total = sum(
                int(rows_by_location[location][f"{candidate} ({party}) - Total"] or 0)
                for party in candidate_party_map
                for candidate in candidate_party_map[party]
            )
            rows_by_location[location]["Grand Total"] = str(grand_total)

        # 6. Return harmonized output
        output_data = [rows_by_location[loc] for loc in rows_by_location]
        output_headers, output_data = harmonize_headers_and_data(output_headers, output_data)
        logger.info(f"[TABLE BUILDER] Built {len(output_data)} rows with {len(output_headers)} columns.")

    if learning_mode and hasattr(coordinator, "log_table_structure"):
        coordinator.log_table_structure(contest_title, headers, context=context)
        logger.info(f"[TABLE BUILDER] Logged confirmed table structure for '{contest_title}'.")

    if learning_mode and hasattr(coordinator, "log_table_structure"):
        should_log = True
        if confirm_table_structure_callback:
            should_log = confirm_table_structure_callback(headers)
        else:
            # Simple CLI prompt fallback
            print(f"\n[Table Builder] Learned headers for '{contest_title}':\n{headers}")
            resp = input("Log this table structure for future auto-application? [Y/n]: ").strip().lower()
            should_log = (resp in ("", "y", "yes"))
        if should_log:
            coordinator.log_table_structure(contest_title, headers, context=context)
            logger.info(f"[TABLE BUILDER] Logged confirmed table structure for '{contest_title}'.")
        else:
            logger.info(f"[TABLE BUILDER] User declined to log table structure for '{contest_title}'.")

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
    return headers, data, passed

from typing import List, Dict, Any, Tuple

def extract_table_data(table) -> Tuple[List[str], List[Dict[str, Any]]]:
    headers = []
    data = []

    # Try to extract headers from <thead>
    header_cells = table.locator("thead tr th")
    if header_cells.count() == 0:
        # Try first row for <th> or <td>
        first_row = table.locator("tr").first
        header_cells = first_row.locator("th, td")
    for i in range(header_cells.count()):
        text = header_cells.nth(i).inner_text().strip()
        headers.append(text if text else f"Column {i+1}")

    # Try to extract rows from <tbody>, else all <tr> except header
    rows = table.locator("tbody tr")
    if rows.count() == 0:
        # Fallback: all <tr> except the first (header)
        all_rows = table.locator("tr")
        if all_rows.count() > 1:
            rows = all_rows.nth(1).locator("xpath=following-sibling::tr")
        else:
            rows = all_rows  # Only one row

    for i in range(rows.count()):
        row = {}
        cells = rows.nth(i).locator("td, th")
        # Skip empty rows
        if cells.count() == 0:
            continue
        for j in range(cells.count()):
            if j < len(headers):
                row[headers[j]] = cells.nth(j).inner_text().strip()
            else:
                # Extra columns: add generic header
                row[f"Extra_{j+1}"] = cells.nth(j).inner_text().strip()
        # Only add non-empty rows
        if any(v for v in row.values()):
            data.append(row)

    # Fallback: if no headers but there is data, use "Column 1", "Column 2", etc.
    if not headers and data:
        max_cols = max(len(row) for row in data)
        headers = [f"Column {i+1}" for i in range(max_cols)]
        # Re-map data to use these headers
        new_data = []
        for row in data:
            new_row = {}
            for idx, h in enumerate(headers):
                new_row[h] = list(row.values())[idx] if idx < len(row) else ""
            new_data.append(new_row)
        data = new_data

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

    def extract_text_from_html(html):
        # Extract visible text from HTML string
        import re
        m = re.search(r">([^<]+)<", html)
        return m.group(1).strip() if m else html

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