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
from .shared_logic import normalize_text

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..Context_Integration.context_coordinator import ContextCoordinator

BALLOT_TYPES = [
    "Election Day", "Early Voting", "Absentee", "Mail", "Provisional", "Affidavit", "Other", "Void"
]

def dynamic_detect_location_header(headers: List[str], coordinator: "ContextCoordinator") -> Tuple[str, str]:
    """
    Dynamically detect the first and second location columns (e.g., precinct, ward, city, district, municipal).
    Uses context, regex, NER, and library.
    Returns (location_header, percent_reported_header)
    """
    # Try context library patterns
    location_patterns = coordinator.library.get("precinct_patterns", []) + \
                        ["precinct", "ward", "district", "city", "municipal", "location", "area"]
    percent_patterns = ["% precincts reporting", "% reporting", "percent reporting"]

    # Normalize headers for matching
    norm_headers = [normalize_text(h) for h in headers]
    location_header = None
    percent_header = None

    # 1. Try regex/library patterns
    for idx, h in enumerate(norm_headers):
        for pat in location_patterns:
            if pat.lower() in h:
                location_header = headers[idx]
                break
        if location_header:
            break

    for idx, h in enumerate(norm_headers):
        for pat in percent_patterns:
            if pat.lower() in h:
                percent_header = headers[idx]
                break
        if percent_header:
            break

    # 2. Try spaCy NER if not found
    if not location_header:
        for idx, h in enumerate(headers):
            entities = coordinator.extract_entities(h)
            for ent, label in entities:
                if label in {"GPE", "LOC", "FAC"}:
                    location_header = headers[idx]
                    break
            if location_header:
                break

    # 3. Fallback to first column
    if not location_header and headers:
        location_header = headers[0]
    if not percent_header and headers:
        percent_header = next((h for h in headers if "%" in h), None)

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
    # Extract headers
    header_cells = table.locator("thead tr th")
    if header_cells.count() == 0:
        header_cells = table.locator("tr").nth(0).locator("th, td")
    for i in range(header_cells.count()):
        headers.append(header_cells.nth(i).inner_text().strip())
    # Extract rows
    rows = table.locator("tbody tr")
    for i in range(rows.count()):
        row = {}
        cells = rows.nth(i).locator("td")
        for j in range(cells.count()):
            if j < len(headers):
                row[headers[j]] = cells.nth(j).inner_text().strip()
        data.append(row)
    return headers, data

def build_dynamic_table(
    headers: List[str],
    data: List[Dict[str, Any]],
    coordinator: "ContextCoordinator",
    context: dict,
    max_feedback_loops: int = 3
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Main entry: builds a uniform, context-aware table with dynamic verification and feedback.
    """
    # 1. Detect location and percent headers
    location_header, percent_header = dynamic_detect_location_header(headers, coordinator)
    logger.info(f"[TABLE BUILDER] Detected location header: {location_header}, percent header: {percent_header}")

    # 2. Extract candidate/party/ballot type structure
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
    # Columns: [percent_header, location_header, ...candidates grouped by party and ballot type..., totals]
    output_headers = []
    if percent_header:
        output_headers.append(percent_header)
    output_headers.append(location_header)

    # Build candidate columns: party -> candidate -> ballot types
    for party in sorted(candidate_party_map.keys()):
        for candidate in sorted(candidate_party_map[party].keys()):
            for ballot_type in BALLOT_TYPES:
                col = f"{candidate} ({party}) - {ballot_type}"
                output_headers.append(col)
            # Candidate total
            output_headers.append(f"{candidate} ({party}) - Total")
        # Party total
        output_headers.append(f"{party} - Total")

    # Grand total
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
                    # Try to find matching header in original data
                    value = ""
                    for h in headers:
                        if candidate in h and party in h and ballot_type in h:
                            value = row.get(h, "")
                            break
                    rows_by_location[location][col] = value
                    try:
                        candidate_total += int(value.replace(",", "")) if value else 0
                    except Exception:
                        pass
                # Candidate total
                rows_by_location[location][f"{candidate} ({party}) - Total"] = str(candidate_total)
            # Party total
            party_total = sum(
                int(rows_by_location[location][f"{candidate} ({party}) - Total"] or 0)
                for candidate in candidate_party_map[party]
            )
            rows_by_location[location][f"{party} - Total"] = str(party_total)
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

    return output_headers, output_data

# Example usage for integration/testing
if __name__ == "__main__":
    from ..Context_Integration.context_coordinator import ContextCoordinator
    # Simulate a coordinator and a table extraction
    coordinator = ContextCoordinator()
    # Simulate headers/data from a table extraction
    headers = [
        "Precinct", "% Precincts Reporting",
        "John Doe (Democratic) - Election Day", "John Doe (Democratic) - Early Voting",
        "Jane Smith (Republican) - Election Day", "Jane Smith (Republican) - Early Voting",
        "Write-In (Other) - Election Day"
    ]
    data = [
        {
            "Precinct": "Ward 1",
            "% Precincts Reporting": "100%",
            "John Doe (Democratic) - Election Day": "120",
            "John Doe (Democratic) - Early Voting": "30",
            "Jane Smith (Republican) - Election Day": "110",
            "Jane Smith (Republican) - Early Voting": "25",
            "Write-In (Other) - Election Day": "2"
        },
        {
            "Precinct": "Ward 2",
            "% Precincts Reporting": "100%",
            "John Doe (Democratic) - Election Day": "140",
            "John Doe (Democratic) - Early Voting": "35",
            "Jane Smith (Republican) - Election Day": "100",
            "Jane Smith (Republican) - Early Voting": "20",
            "Write-In (Other) - Election Day": "1"
        }
    ]
    context = {"state": "NY", "county": "Rockland"}
    output_headers, output_data = build_dynamic_table(headers, data, coordinator, context)
    print("Output headers:", output_headers)
    print("Output data:", json.dumps(output_data, indent=2))