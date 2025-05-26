import csv
import json
import os
from datetime import datetime
from ..utils.shared_logger import logger
from ..utils.table_builder import format_table_data_for_output, review_and_fill_missing_data

def get_output_path(state, county, subfolder="parsed"):
    parts = ["output"]
    if state:
        parts.append(str(state).lower())
    if county:
        parts.append(str(county).lower())
    if subfolder:
        parts.append(str(subfolder))
    path = os.path.join(*parts)
    os.makedirs(path, exist_ok=True)
    return path

def format_timestamp(fmt="%Y%m%d_%H%M%S"):
    return datetime.now().strftime(fmt)

def finalize_election_output(headers, data, contest_title, metadata, handler_options=None):
    """
    Writes the parsed election data and metadata to CSV and JSON files.
    Returns a dict with the CSV and JSON paths.
    """
    output_path = get_output_path(metadata.get("state", "unknown"), metadata.get("county", "unknown"), "parsed")
    timestamp = format_timestamp()
    safe_title = "".join([c if c.isalnum() or c in " _-" else "_" for c in contest_title])
    filename = f"{safe_title.replace(' ', '_')}_results_{timestamp}.csv"
    filepath = os.path.join(output_path, filename)

    # --- Final Data Clean-up and Structure ---
    # 1. Format table data (wide/long, orientation, etc.)
    headers, data = format_table_data_for_output(headers, data, handler_options=handler_options)

    # 2. Review and fill missing data
    headers, data = review_and_fill_missing_data(headers, data)

    # 3. Sort headers for consistency (Precinct, Candidate, Party, Method, Votes, ...rest)
    preferred_order = ["Precinct", "% Precincts Reporting", "Candidate", "Party", "Method", "Votes"]
    sorted_headers = [h for h in preferred_order if h in headers] + [h for h in headers if h not in preferred_order]

    # 4. Remove duplicate rows (if any)
    seen = set()
    unique_data = []
    for row in data:
        row_tuple = tuple(row.get(h, "") for h in sorted_headers)
        if row_tuple not in seen:
            seen.add(row_tuple)
            unique_data.append(row)
    data = unique_data

    # 5. Remove empty columns (all values empty)
    non_empty_headers = []
    for h in sorted_headers:
        if any(str(row.get(h, "")).strip() for row in data):
            non_empty_headers.append(h)
    sorted_headers = non_empty_headers

    # 6. Final harmonization
    data = [{h: row.get(h, "") for h in sorted_headers} for row in data]

    # --- Write CSV ---
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted_headers)
        writer.writeheader()
        for row in data:
            writer.writerow(row)
        f.write(f"\n# Generated at: {timestamp}")

    # --- Write metadata JSON ---
    json_meta_path = filepath.replace(".csv", "_metadata.json")
    metadata_out = dict(metadata)
    metadata_out["timestamp"] = timestamp
    metadata_out["output_folder"] = output_path
    metadata_out["csv_file"] = filename
    metadata_out["headers"] = sorted_headers
    metadata_out["row_count"] = len(data)
    with open(json_meta_path, "w", encoding="utf-8") as jf:
        json.dump(metadata_out, jf, indent=2)

    logger.info(f"[OUTPUT] Wrote {len(data)} rows to {filepath}")
    logger.info(f"[OUTPUT] Metadata written to {json_meta_path}")

    return {"csv_path": filepath, "metadata_path": json_meta_path}