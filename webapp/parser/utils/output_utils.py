import csv
import json
import os
from datetime import datetime
from ..utils.shared_logger import rprint
from ..utils.logger_instance import logger
from ..utils.table_builder import build_dynamic_table, harmonize_headers_and_data
from ..config import CONTEXT_DB_PATH, BASE_DIR
from ..utils.user_prompt import prompt_yes_no

CACHE_FILE = os.path.join(os.path.dirname(CONTEXT_DB_PATH), ".processed_urls")

def get_output_path(metadata, subfolder="parsed", coordinator=None, feedback_context=None):
    """
    Build output path using organized context metadata.
    If any key info is missing, use feedback loop (ML/NER/user prompt) to resolve.
    """
    parts = ["output"]
    # Use coordinator to try to fill missing info if available
    state = metadata.get("state", "") or (coordinator.get_states()[0] if coordinator and coordinator.get_states() else "")
    county = metadata.get("county", "") or (coordinator.get_districts()[0] if coordinator and coordinator.get_districts() else "")
    year = metadata.get("year", "")
    race = metadata.get("race", "")
    election_type = metadata.get("election_type", "")

    def safe_filename(s):
        return "".join(c if c.isalnum() or c in " _-" else "_" for c in str(s)).strip() or "Unknown"

    # Feedback loop for missing/unknown info
    max_loops = 3
    for _ in range(max_loops):
        if not year or not str(year).isdigit() or len(str(year)) != 4:
            # Try to extract from context or prompt user
            if coordinator:
                years = coordinator.get_years()
                if years:
                    year = years[0]
            if not year and feedback_context:
                year = feedback_context.get("year", "")
            if not year:
                year = input("Year could not be determined. Please enter the election year (YYYY): ").strip()
        if not race or race.lower() == "unknown":
            if coordinator:
                contests = coordinator.get_contests()
                if contests:
                    race = contests[0].get("title", "")
            if not race and feedback_context:
                race = feedback_context.get("race", "")
            if not race:
                race = input("Race could not be determined. Please enter the race name: ").strip()
        if year and race:
            break

    # If still unknown, warn and use 'unknown'
    if not year or not str(year).isdigit() or len(str(year)) != 4:
        rprint("[yellow][OUTPUT] Year could not be verified. Using 'Unknown'.[/yellow]")
        year = "Unknown"
    if not race:
        rprint("[yellow][OUTPUT] Race could not be verified. Using 'unknown_race'.[/yellow]")
        race = "unknown_race"

    race_safe = safe_filename(race)
    county_safe = safe_filename(county)
    state_safe = safe_filename(state)
    if state:
        parts.append(state_safe.lower())
    if county:
        parts.append(county_safe.lower())
    if year and str(year).isdigit() and len(str(year)) == 4:
        parts.append(str(year))
    else:
        parts.append("Unknown")
    if election_type:
        parts.append(safe_filename(election_type).lower())
    if race:
        safe_race = "".join([c if c.isalnum() or c in " _-" else "_" for c in str(race)])
        parts.append(safe_race.replace(" ", "_"))
    else:
        parts.append("unknown_race")
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
    Handles both JSONL (one JSON object per line) and JSON array formats.
    """
    if not os.path.exists(cache_file):
        return None
    with open(cache_file, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            return None
        entries = []
        # Try JSON array first
        try:
            if content.startswith("["):
                arr = json.loads(content)
                if isinstance(arr, list):
                    entries = arr
            else:
                raise ValueError("Not a JSON array")
        except Exception:
            # Fallback: treat as JSONL
            entries = []
            for line in content.splitlines():
                if not line.strip():
                    continue
                try:
                    entries.append(json.loads(line))
                except Exception as e:
                    print(f"[DEBUG] Failed to parse line as JSON: {line!r}")
                    continue
        for entry in entries:
            meta = entry.get("metadata", {})
            # Compare key fields for deduplication
            if (
                meta.get("state", "Unknown") == metadata.get("state", "Unknown") and
                meta.get("county", "Unknown") == metadata.get("county", "Unknown") and
                meta.get("year", "Unknown") == metadata.get("year", "Unknown") and
                meta.get("race", "Unknown") == metadata.get("race", "Unknown")
            ):
                return entry
    return None

def finalize_election_output(headers, data, coordinator, contest_title, state, county):
    """
    Writes the parsed election data and metadata to CSV and JSON files.
    Updates the persistent context library and output cache.
    Returns a dict with the CSV and JSON paths.
    """
    from ..Context_Integration.context_organizer import append_to_context_library, organize_context

    meta = {
        "race": contest_title or "Unknown",
        "year": "Unknown",
        "state": state or "Unknown",
        "county": county or "Unknown"
    }
    # Try to extract year from contest_title if possible
    import re
    match = re.search(r"\b(19|20)\d{2}\b", contest_title or "")
    if match:
        meta["year"] = match.group(0)

    # 1. Enrich metadata using context organizer
    organized = organize_context(meta)
    enriched_meta = organized.get("metadata", meta)
    # Defensive: ensure 'race' is present in enriched_meta
# Defensive: ensure required fields
    if not enriched_meta.get("race"):
        enriched_meta["race"] = contest_title or "Unknown"
    if not enriched_meta.get("year") or not (str(enriched_meta["year"]).isdigit() and len(str(enriched_meta["year"])) == 4):
        enriched_meta["year"] = meta.get("year", "Unknown")
    if not enriched_meta.get("state"):
        enriched_meta["state"] = state or "Unknown"
    if not enriched_meta.get("county"):
        enriched_meta["county"] = county or "Unknown"
    append_to_context_library({"metadata": enriched_meta})

    # 3. Check for existing output (deduplication)
    existing = check_existing_output(enriched_meta)
    if existing:
        overwrite = prompt_yes_no(
            f"Output for [bold]{enriched_meta.get('state', 'Unknown')}[/bold] [bold]{enriched_meta.get('county', 'Unknown')}[/bold] [bold]{enriched_meta.get('year', 'Unknown')}[/bold] [bold]{enriched_meta.get('race', 'Unknown')}[/bold] already exists at [cyan]{existing.get('output_path', 'Unknown')}[/cyan]. Overwrite?",
            default="n"
        )
        if not overwrite:
            rprint(f"[bold yellow][OUTPUT][/bold yellow] Skipping write due to existing output.")
            csv_path = existing["output_path"]
            json_meta_path = csv_path.replace(".csv", "_metadata.json")
            return {"csv_path": csv_path, "metadata_path": json_meta_path}

    # 4. Build output path using enriched metadata and feedback loop if needed
    output_path = get_output_path(enriched_meta, "parsed", coordinator=coordinator, feedback_context=enriched_meta.get("feedback_context", {}))
    timestamp = format_timestamp()
    # 5. Build safe and descriptive filename
    safe_title = "".join([c if c.isalnum() or c in " _-" else "_" for c in (contest_title or enriched_meta.get("race", "results"))])
    safe_title = safe_title.replace(" ", "_")
    year = enriched_meta.get("year", "")
    state = enriched_meta.get("state", "")
    county = enriched_meta.get("county", "")
    election_type = enriched_meta.get("election_type", "")
    filename_parts = [
        str(year) if year and str(year).isdigit() and len(str(year)) == 4 else "",
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
    headers, data = harmonize_headers_and_data(headers, data)
    headers, data = build_dynamic_table(headers, data, coordinator, context=enriched_meta.get("context", {}))
    preferred_order = ["Precinct", "% Precincts Reporting", "Candidate", "Party", "Method", "Votes"]
    sorted_headers = [h for h in preferred_order if h in headers] + [h for h in headers if h not in preferred_order]
    if "Precinct" not in sorted_headers and any("Precinct" in row for row in data):
        sorted_headers = ["Precinct"] + sorted_headers
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
    if county:
        metadata_out["batch_manifest"] = county

    # --- Remove any absolute paths or sensitive info ---
    for k in list(metadata_out.keys()):
        if isinstance(metadata_out[k], str) and os.path.isabs(metadata_out[k]):
            del metadata_out[k]
    # Remove any 'cwd' or similar keys
    if "cwd" in metadata_out:
        del metadata_out["cwd"]

    with open(json_meta_path, "w", encoding="utf-8") as jf:
        json.dump(metadata_out, jf, indent=2)

    # --- Update output cache ---
    update_output_cache(metadata_out, filepath)

    rprint(f"[bold green][OUTPUT][/bold green] Wrote [bold]{len(data)}[/bold] rows to:\n  [cyan]{filepath}[/cyan]")
    rprint(f"[bold green][OUTPUT][/bold green] Metadata written to:\n  [cyan]{json_meta_path}[/cyan]")

    return {"csv_path": filepath, 
            "metadata_path": json_meta_path,
            "output_file": filepath
    }