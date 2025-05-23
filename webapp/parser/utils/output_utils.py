import csv
import json
import os
from datetime import datetime
from ..utils.shared_logger import logger

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

def finalize_election_output(headers, data, contest_title, metadata):
    """
    Writes the parsed election data and metadata to CSV and JSON files.
    Returns a dict with the CSV and JSON paths.
    """
    output_path = get_output_path(metadata.get("state", "unknown"), metadata.get("county", "unknown"), "parsed")
    timestamp = format_timestamp()
    safe_title = "".join([c if c.isalnum() or c in " _-" else "_" for c in contest_title])
    filename = f"{safe_title.replace(' ', '_')}_results_{timestamp}.csv"
    filepath = os.path.join(output_path, filename)

    # Write CSV
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in data:
            writer.writerow(row)
        f.write(f"\n# Generated at: {timestamp}")

    # Write metadata JSON
    json_meta_path = filepath.replace(".csv", "_metadata.json")
    with open(json_meta_path, "w", encoding="utf-8") as jf:
        json.dump(metadata, jf, indent=2)

    return {"csv_path": filepath, "metadata_path": json_meta_path}