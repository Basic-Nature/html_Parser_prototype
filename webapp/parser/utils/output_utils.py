import csv
import json
import os
from datetime import datetime
from ..utils.shared_logger import logger
from ..utils.table_builder import format_table_data_for_output, review_and_fill_missing_data
from ..Context_Integration.context_organizer import organize_context, append_to_context_library, load_context_library
from ..utils.user_prompt import prompt_yes_no

CACHE_FILE = ".output_cache.jsonl"

def get_output_path(metadata, subfolder="parsed"):
    """
    Build output path using organized context metadata.
    Example: output/state/county/year/race/parsed/
    """
    parts = ["output"]
    state = metadata.get("state", "unknown")
    county = metadata.get("county", "unknown")
    year = metadata.get("year", None)
    race = metadata.get("race", None)
    election_type = metadata.get("election_type", None)
    if state:
        parts.append(str(state).lower())
    if county:
        parts.append(str(county).lower())
    if year:
        parts.append(str(year))
    if election_type:
        parts.append(str(election_type).lower())
    if race:
        safe_race = "".join([c if c.isalnum() or c in " _-" else "_" for c in str(race)])
        parts.append(safe_race.replace(" ", "_"))
    else:
        parts.append("unknown_race")  # <-- Add a fallback for missing race
    if subfolder:
        parts.append(str(subfolder))
    path = os.path.join(*parts)
    os.makedirs(path, exist_ok=True)
    return path

def format_timestamp(fmt="%Y%m%d_%H%M%S"):
    return datetime.now().strftime(fmt)

def update_output_cache(metadata, output_path, cache_file=CACHE_FILE):
    """
    Append output metadata to a cache file for fast lookup and deduplication.
    """
    cache_entry = {
        "timestamp": format_timestamp(),
        "output_path": output_path,
        "metadata": metadata,
    }
    with open(cache_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(cache_entry) + "\n")

def check_existing_output(metadata, cache_file=CACHE_FILE):
    """
    Check if output for this context already exists in the cache.
    """
    if not os.path.exists(cache_file):
        return None
    with open(cache_file, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            meta = entry.get("metadata", {})
            # Compare key fields for deduplication
            if (
                meta.get("state") == metadata.get("state") and
                meta.get("county") == metadata.get("county") and
                meta.get("year") == metadata.get("year") and
                meta.get("race") == metadata.get("race")
            ):
                return entry
    return None

def finalize_election_output(headers, data, contest_title, metadata, handler_options=None, batch_manifest=None):
    """
    Writes the parsed election data and metadata to CSV and JSON files.
    Updates the persistent context library and output cache.
    Returns a dict with the CSV and JSON paths.
    """
    # 1. Enrich metadata using context organizer
    organized = organize_context(metadata)
    enriched_meta = organized.get("metadata", metadata)
    # Defensive: ensure 'race' is present in enriched_meta
    if "race" not in enriched_meta:
        enriched_meta["race"] = metadata.get("race", "Unknown")
    # 2. Optionally append output info to context library
    append_to_context_library({"metadata": enriched_meta})

    # 3. Check for existing output (deduplication)
    existing = check_existing_output(enriched_meta)
    if existing:
        overwrite = prompt_yes_no(
            f"Output for {enriched_meta.get('state', 'Unknown')} {enriched_meta.get('county', 'Unknown')} {enriched_meta.get('year', 'Unknown')} {enriched_meta.get('race', 'Unknown')} already exists at {existing.get('output_path', 'Unknown')}. Overwrite?",
            default="n"
        )
        if not overwrite:
            logger.info("[OUTPUT] Skipping write due to existing output.")
            csv_path = existing["output_path"]
            json_meta_path = csv_path.replace(".csv", "_metadata.json")
            return {"csv_path": csv_path, "metadata_path": json_meta_path}

    # 4. Build output path using enriched metadata
    output_path = get_output_path(enriched_meta, "parsed")
    timestamp = format_timestamp()
    # 5. Build safe and descriptive filename
    safe_title = "".join([c if c.isalnum() or c in " _-" else "_" for c in (contest_title or enriched_meta.get("race", "results"))])
    safe_title = safe_title.replace(" ", "_")
    year = enriched_meta.get("year", "")
    state = enriched_meta.get("state", "")
    county = enriched_meta.get("county", "")
    election_type = enriched_meta.get("election_type", "")
    filename_parts = [
        str(year) if year else "",
        str(state).lower() if state else "",
        str(county).lower() if county else "",
        str(election_type).lower() if election_type else "",
        safe_title,
        "results",
        timestamp
    ]
    filename = "_".join([p for p in filename_parts if p]).replace("__", "_") + ".csv"
    filepath = os.path.join(output_path, filename)

    # --- Final Data Clean-up and Structure ---
    headers, data = format_table_data_for_output(headers, data, handler_options=handler_options)
    headers, data = review_and_fill_missing_data(headers, data)
    preferred_order = ["Precinct", "% Precincts Reporting", "Candidate", "Party", "Method", "Votes"]
    sorted_headers = [h for h in preferred_order if h in headers] + [h for h in headers if h not in preferred_order]
    seen = set()
    unique_data = []
    for row in data:
        row_tuple = tuple(row.get(h, "") for h in sorted_headers)
        if row_tuple not in seen:
            seen.add(row_tuple)
            unique_data.append(row)
    data = unique_data
    non_empty_headers = []
    for h in sorted_headers:
        if any(str(row.get(h, "")).strip() for row in data):
            non_empty_headers.append(h)
    sorted_headers = non_empty_headers
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
    metadata_out = dict(enriched_meta)
    metadata_out["timestamp"] = timestamp
    metadata_out["output_folder"] = output_path
    metadata_out["csv_file"] = filename
    metadata_out["headers"] = sorted_headers
    metadata_out["row_count"] = len(data)
    if batch_manifest:
        metadata_out["batch_manifest"] = batch_manifest
    with open(json_meta_path, "w", encoding="utf-8") as jf:
        json.dump(metadata_out, jf, indent=2)

    # --- Update output cache ---
    update_output_cache(metadata_out, filepath)

    logger.info(f"[OUTPUT] Wrote {len(data)} rows to {filepath}")
    logger.info(f"[OUTPUT] Metadata written to {json_meta_path}")

    return {"csv_path": filepath, "metadata_path": json_meta_path}