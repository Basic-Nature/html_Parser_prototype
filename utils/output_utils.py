# utils/output_utils.py
# ==============================================================
# Shared utility functions for formatting timestamps and resolving output paths.
# Used by handlers to store CSV, TXT, and metadata output consistently.
# ==============================================================

import os
import csv
import json
from datetime import datetime
from typing import Optional
from utils.html_table_extractor import calculate_grand_totals
from utils.shared_logger import log_debug, log_info, log_warning, log_error

def mark_url_processed(url):
    """
    Marks a URL as processed by creating a file with the URL's hash.
    This prevents reprocessing of the same URL in future runs.
    """
    import hashlib
    url_hash = hashlib.md5(url.encode()).hexdigest()
    processed_file = os.path.join("processed_urls", f"{url_hash}.txt")
    os.makedirs(os.path.dirname(processed_file), exist_ok=True)
    with open(processed_file, "w") as f:
        f.write(url)

def format_timestamp():
    """Return timestamp in standardized YYYYMMDD_HHMMSS format."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_output_path(state, county, subfolder):
    """
    Construct a structured output path like:
    output/{State}_{County}/{subfolder}/
    Creates the folder if it doesn't exist.
    """
    state_clean = state.replace(" ", "_") if state else "Unknown"
    county_clean = county.replace(" ", "_") if county else "Unknown"
    base_path = os.path.join("output", f"{state_clean}_{county_clean}", subfolder)
    os.makedirs(base_path, exist_ok=True)
    return base_path

def sanitize_filename(name: str):
    """
    Replace illegal or unsafe characters in filenames with underscores.
    Used to protect against invalid filesystem writes across OSes.
    """
    import re
    return re.sub(r"[\\/*?\"<>|]", "_", name).strip()

def should_timestamp():
    """
    Return True if output filenames should include timestamps.
    Controlled via .env variable: INCLUDE_TIMESTAMP_IN_FILENAME=true
    """
    return os.getenv("INCLUDE_TIMESTAMP_IN_FILENAME", "true").lower() == "true"

def resolve_output_file(base_title: str, state: str, county: Optional[str] = None, ext: str = "csv", subfolder: str = "parsed"):
    """
    Build a full output file path including:
    - Sanitized base title
    - Timestamp suffix (if enabled)
    - Subfolder such as 'parsed', 'raw', 'logs'
    - If county is missing, route to state-level folder only
    """
    base_title = sanitize_filename(base_title)
    timestamp = format_timestamp() if should_timestamp() else ""
    filename = f"{base_title}_{timestamp}.{ext}" if timestamp else f"{base_title}.{ext}"

    if not county:
        folder_name = state.replace(" ", "_")
    else:
        folder_name = f"{state.replace(' ', '_')}_{county.replace(' ', '_')}"

    output_path = os.path.join("output", folder_name, subfolder)
    os.makedirs(output_path, exist_ok=True)
    return os.path.join(output_path, filename)

def write_csv_output(headers, data, contest_title, metadata, include_grand_totals=True):
    """
    Writes Smart Elections-compliant CSV output using headers and row data.
    The filename is derived using state, county, race, and timestamp.
    Optionally includes contest_title as a column.
    """
    state = metadata.get("state", "Unknown")
    county = metadata.get("county", "Unknown")
    title = contest_title or metadata.get("race", "Election_Results")
    output_path = resolve_output_file(title, state, county, ext="csv", subfolder="parsed")
    if contest_title:
        for row in data:
            row["Contest Title"] = contest_title
        if "Contest Title" not in headers:
            headers.append("Contest Title")
    if not headers:
        # Build header union from all rows
        header_set = set()
        for row in data:
            header_set.update(row.keys())
        headers = sorted(header_set)
    # Optionally add grand totals
    if include_grand_totals and data:
        grand_totals = calculate_grand_totals(data)
        data = data + [grand_totals]
    # Write CSV file
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
        writer.writeheader()
        writer.writedata(data)
        f.write(f"# Generated at: {format_timestamp()}")

    return output_path

def write_json_metadata(metadata, path=None):
    """
    Optionally writes a metadata.json file next to the CSV output.

    Args:
        metadata (dict): Metadata fields from the handler
        path (str, optional): If provided, will override default path generation

    Returns:
        str: Path to the written JSON file
    """
    state = metadata.get("state", "Unknown")
    county = metadata.get("county", "Unknown")
    title = metadata.get("race", "Election_Metadata")

    if path:
        json_path = path
    else:
        json_path = resolve_output_file(title, state, county, ext="json", subfolder="parsed")

    # Optional cleanup of large or verbose fields before saving
    cleaned = clean_metadata_for_output(metadata)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2)

    return json_path

from rich import print as rprint

def _log_redaction(message):
    """
    Logs redaction/truncation messages based on LOG_REDACTION_LEVEL in .env.
    Also prints them to console using rich for visibility in interactive mode.
    """
    level = os.getenv("LOG_REDACTION_LEVEL", "WARNING").upper()
    if level == "DEBUG":
        log_debug(message)
    elif level == "INFO":
        log_info(message)
    elif level == "ERROR":
        log_error(message)
    else:
        log_warning(message)
    rprint(f"[bold yellow]{message}[/bold yellow]")


def clean_metadata_for_output(metadata):
    """
    Returns a cleaned and sanitized copy of the metadata dictionary for output use.
    Removes verbose entries, truncates large content, and redacts sensitive fields.
    Logs warnings if any fields are truncated or redacted.

    Args:
        metadata (dict): Original metadata dictionary

    Returns:
        dict: Cleaned version suitable for JSON output
    """

    keys_to_exclude = {"raw_text", "_source_map"}
    cleaned = {k: v for k, v in metadata.items() if k not in keys_to_exclude}

    # Truncate long string values
    for key, value in cleaned.items():
        if isinstance(value, str) and len(value) > 1000:
            cleaned[key] = value[:1000] + "... [TRUNCATED]"
            _log_redaction(f"[METADATA] Truncated long value in field '{key}'")
        elif isinstance(value, list) and len(value) > 100:
            cleaned[key] = value[:100] + ["... [TRUNCATED LIST]"]
            _log_redaction(f"[METADATA] Truncated long list in field '{key}'")

    # Redact known sensitive fields
    for sensitive in ["user_agent", "ip_address"]:
        if sensitive in cleaned:
            cleaned[sensitive] = "[REDACTED]"
            _log_redaction(f"[METADATA] Redacted sensitive field '{sensitive}'")

    return cleaned

def finalize_election_output(headers, data, contest_title, metadata, write_metadata=True, include_grand_totals=True):
    """
    Unified output finalizer that writes both CSV and optional JSON metadata.
    Ensures contest_title is passed through.
    """
    result = {}
    result['csv_path'] = write_csv_output(headers, data, contest_title, metadata, include_grand_totals=include_grand_totals)
    if not result['csv_path']:
        log_error("Failed to write CSV output.")
        return result
    if write_metadata:
        result['json_path'] = write_json_metadata(metadata)
    return result
