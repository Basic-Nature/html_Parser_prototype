import csv
import json
import os
import collections.abc
from datetime import datetime
from ..utils.shared_logger import rprint
from ..utils.logger_instance import logger
from ..utils.table_builder import build_dynamic_table, harmonize_headers_and_data
from ..config import CONTEXT_DB_PATH, BASE_DIR
from ..utils.user_prompt import prompt_yes_no

CACHE_FILE = os.path.join(os.path.dirname(CONTEXT_DB_PATH), ".processed_urls")

def get_project_root():
    # Returns the parent directory of webapp (the project root)
    return os.path.dirname(BASE_DIR)

def get_output_root():
    # Output folder at the project root
    return os.path.join(get_project_root(), "output")

def get_log_root():
    # Log folder at the project root
    return os.path.join(get_project_root(), "log")

def safe_join(base, *paths):
    """
    Safely join paths and ensure the result is inside base.
    Prevents path traversal and path-injection.
    """
    base = os.path.abspath(base)
    path = os.path.abspath(os.path.join(base, *paths))
    if not path.startswith(base):
        raise ValueError("Unsafe path detected.")
    return path

def get_output_path(metadata, subfolder="parsed", coordinator=None, feedback_context=None):
    """
    Build output path using organized context metadata.
    If any key info is missing, use feedback loop (ML/NER/user prompt) to resolve.
    """
    parts = []
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
            if coordinator:
                years = coordinator.get_years()
                if years:
                    year = years[0]
            if not year and feedback_context:
                year = feedback_context.get("year", "")
        if not race or race.lower() == "unknown":
            if coordinator:
                contests = coordinator.get_contests()
                if contests:
                    race = contests[0].get("title", "")
            if not race and feedback_context:
                race = feedback_context.get("race", "")
        if year and race:
            break

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

    # Always use output folder at project root
    output_root = get_output_root()
    path = safe_join(output_root, *parts)
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
            if (
                meta.get("state", "Unknown") == metadata.get("state", "Unknown") and
                meta.get("county", "Unknown") == metadata.get("county", "Unknown") and
                meta.get("year", "Unknown") == metadata.get("year", "Unknown") and
                meta.get("race", "Unknown") == metadata.get("race", "Unknown")
            ):
                return entry
    return None

def convert_sets_to_lists(obj):
    if isinstance(obj, dict):
        return {k: convert_sets_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, list):
        return [convert_sets_to_lists(i) for i in obj]
    else:
        return obj

def deep_merge_dicts(dest, src):
    """
    Recursively merge src into dest.
    - If a key exists in both and both values are dicts, merge them recursively.
    - Otherwise, src overwrites dest.
    """
    for k, v in src.items():
        if (
            k in dest
            and isinstance(dest[k], dict)
            and isinstance(v, dict)
        ):
            deep_merge_dicts(dest[k], v)
        else:
            dest[k] = v
    return dest

def finalize_election_output(
    headers,
    data,
    coordinator,
    contest_title,
    state,
    county,
    context=None,
    enable_user_feedback=False
):
    """
    Finalize and write election output to CSV and metadata JSON.
    Output is always placed in a subfolder of the project root (parent of webapp).
    """
    from ..Context_Integration.context_organizer import append_to_context_library, organize_context
    from ..config import BASE_DIR
    import re

    if context is None:
        context = {}

    logger.info(f"[OUTPUT_UTILS] finalize_election_output called with contest_title: '{contest_title}'")

    meta = {
        "race": contest_title or "Unknown",
        "year": "Unknown",
        "state": state or "Unknown",
        "county": county or "Unknown"
    }
    match = re.search(r"\b(19|20)\d{2}\b", contest_title or "")
    if match:
        meta["year"] = match.group(0)

    organized = organize_context(meta)
    enriched_meta = organized.get("metadata", meta)

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

    # Build output path safely under output folder at project root
    def safe_filename(s):
        return "".join(c if c.isalnum() or c in " _-" else "_" for c in str(s)).strip() or "Unknown"

    year = enriched_meta.get("year", "")
    state = enriched_meta.get("state", "")
    county = enriched_meta.get("county", "")
    election_type = enriched_meta.get("election_type", "")
    race = enriched_meta.get("race", "")

    parts = [
        safe_filename(state).lower() if state else "",
        safe_filename(county).lower() if county else "",
        str(year) if year and str(year).isdigit() and len(str(year)) == 4 else "Unknown",
        safe_filename(election_type).lower() if election_type else "",
        safe_filename(race).replace(" ", "_") if race else "unknown_race",
        "parsed"
    ]
    parts = [p for p in parts if p]
    output_root = get_output_root()
    output_path = safe_join(output_root, *parts)

    # Ensure output_path is inside output_root
    os.makedirs(output_path, exist_ok=True)

    timestamp = format_timestamp()
    safe_title = safe_filename(contest_title or race or "results").replace(" ", "_")
    filename_parts = [
        str(year) if year and str(year).isdigit() and len(str(year)) == 4 else "",
        safe_filename(state).lower() if state else "",
        safe_filename(county).lower() if county else "",
        safe_filename(election_type).lower() if election_type else "",
        safe_title,
        "results",
        timestamp
    ]
    filename = "_".join([p for p in filename_parts if p]).replace("__", "_") + ".csv"
    filepath = safe_join(output_path, filename)

    # --- Write CSV ---
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
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
    metadata_out["headers"] = headers
    metadata_out["row_count"] = len(data)
    if county:
        metadata_out["batch_manifest"] = county

    # --- Deep merge in any extra context/meta ---
    if context:
        metadata_out = deep_merge_dicts(metadata_out, context)

    # Remove any absolute paths or sensitive info
    for k in list(metadata_out.keys()):
        if isinstance(metadata_out[k], str) and os.path.isabs(metadata_out[k]):
            del metadata_out[k]
    if "cwd" in metadata_out:
        del metadata_out["cwd"]
    if "environment" in metadata_out and isinstance(metadata_out["environment"], dict):
        metadata_out["environment"].pop("cwd", None)

    with open(json_meta_path, "w", encoding="utf-8") as jf:
        metadata_out = convert_sets_to_lists(metadata_out)
        json.dump(metadata_out, jf, indent=2)

    update_output_cache(metadata_out, filepath)

    rprint(f"[bold green][OUTPUT][/bold green] Wrote [bold]{len(data)}[/bold] rows to:\n  [cyan]{filepath}[/cyan]")
    rprint(f"[bold green][OUTPUT][/bold green] Metadata written to:\n  [cyan]{json_meta_path}[/cyan]")

    if enable_user_feedback or os.environ.get("ENABLE_USER_FEEDBACK", "false").lower() == "true":
        feedback_log_path = safe_join(output_path, "user_feedback_log.jsonl")
        feedback = input("\n[Feedback] Would you like to provide feedback or corrections for this output? (Leave blank to skip):\n> ").strip()
        if feedback:
            feedback_entry = {
                "timestamp": format_timestamp(),
                "file": filepath,
                "metadata": metadata_out,
                "feedback": feedback
            }
            with open(feedback_log_path, "a", encoding="utf-8") as fb:
                fb.write(json.dumps(feedback_entry) + "\n")
            rprint(f"[bold blue][FEEDBACK][/bold blue] Feedback logged to {feedback_log_path}")

    return {
        "csv_path": filepath,
        "metadata_path": json_meta_path,
        "output_file": filepath
    }