import csv
import orjson
import os
from datetime import datetime
from typing import Optional
from ..utils.logger_singleton import logger
from ..utils.shared_logic import safe_get_first, safe_items, safe_get, safe_lower
from ..config import CONTEXT_DB_PATH, BASE_DIR, LOG_DIR

CACHE_FILE = os.path.join(os.path.dirname(CONTEXT_DB_PATH), ".processed_urls")

def get_project_root() -> str:
    # Returns the parent directory of webapp (the project root)
    return os.path.dirname(BASE_DIR)

def get_output_root() -> str:
    # Output folder at the project root
    return os.path.join(get_project_root(), "output")

def safe_join(base: str, *paths: str) -> str:
    """
    Safely join paths and ensure the result is inside base.
    Prevents path traversal and path-injection.
    """
    base = os.path.abspath(base)
    path = os.path.abspath(os.path.join(base, *paths))
    if not path.startswith(base):
        raise ValueError("Unsafe path detected.")
    return path

def get_output_path(metadata, subfolder="parsed", coordinator=None, feedback_context=None) -> str:
    """
    Build output path using organized context metadata.
    If any key info is missing, use feedback loop (ML/NER/user prompt) to resolve.
    Safeguards all string operations and path parts.
    """
    from ..Context_Integration.context_coordinator import ContextCoordinator
    coordinator = ContextCoordinator()

    parts = []
    # Use coordinator to try to fill missing info if available
    state = safe_get(metadata, "state", "") or (
        safe_get_first(getattr(coordinator, "get_states", lambda: [])(), "state", None, logger, default="") if coordinator and hasattr(coordinator, "get_states") and coordinator.get_states() else ""
    )
    county = safe_get(metadata, "county", "") or (
        safe_get_first(getattr(coordinator, "get_precincts", lambda: [])(), "county", None, logger, default="") if coordinator and hasattr(coordinator, "get_precincts") and coordinator.get_precincts() else ""
    )
    year = safe_get(metadata, "year", "")
    contests = safe_get(metadata, "contests", "")
    election_types = safe_get(metadata, "election_types", "")

    def safe_filename(s: str) -> str:
        try:
            sanitized = "".join(c if c.isalnum() or c in " _-" else "_" for c in str(s)).strip() or "Unknown"
            sanitized = sanitized.replace("..", "_").replace("/", "_").replace("\\", "_")
            return sanitized
        except Exception:
            return "Unknown"

    # Feedback loop for missing/unknown info
    max_loops = 3
    for _ in range(max_loops):
        if not year or not str(year).isdigit() or len(str(year)) != 4:
            if coordinator and hasattr(coordinator, "get_years"):
                years = coordinator.get_years()
                if years and len(years) > 0:
                    year = safe_get_first(years, "year", None, logger)
            if not year and feedback_context:
                year = safe_get(feedback_context, "year", "")
        if not contests or safe_lower(contests) == "unknown":
            if coordinator and hasattr(coordinator, "get_contests"):
                contests_list = coordinator.get_contests()
                if contests_list and isinstance(contests_list, list) and len(contests_list) > 0:
                    first_contest = safe_get_first(contests_list, "contests", None, logger)
                    if isinstance(first_contest, dict):
                        contests = safe_get(first_contest, "title", "")
                    else:
                        contests = str(first_contest)
            if not contests and feedback_context:
                contests = safe_get(feedback_context, "contests", "")
        if year and contests:
            break

    if not year or not str(year).isdigit() or len(str(year)) != 4:
        logger.warning("[yellow][OUTPUT] Year could not be verified. Using 'Unknown'.[/yellow]")
        year = "Unknown"
    if not contests:
        logger.warning("[yellow][OUTPUT] contests could not be verified. Using 'unknown_contests'.[/yellow]")
        contests = "unknown_contests"

    contests_safe = safe_filename(contests)
    county_safe = safe_filename(county)
    state_safe = safe_filename(state)
    if contests:
        parts.append(safe_lower(contests_safe or ""))
    if state:
        parts.append(safe_lower(state_safe or ""))
    if county:
        parts.append(safe_lower(county_safe or ""))
    if year and str(year).isdigit() and len(str(year)) == 4:
        parts.append(str(year))
    else:
        parts.append("Unknown")
    if election_types:
        parts.append(safe_lower(safe_filename(election_types)))
    if contests:
        safe_contests = "".join([c if c.isalnum() or c in " _-" else "_" for c in str(contests)])
        parts.append(safe_contests.replace(" ", "_"))
    else:
        parts.append("unknown_contests")
    if subfolder:
        parts.append(str(subfolder))

    # Always use output folder at project root
    output_root = get_output_root()
    try:
        path = safe_join(output_root, *parts)
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        logger.error(f"[OUTPUT_UTILS] Failed to create output path: {e}")
        path = output_root
    return path

def format_timestamp(fmt="%Y%m%d_%H%M%S") -> str:
    return datetime.now().strftime(fmt)

def update_output_cache(metadata, output_path, cache_file=CACHE_FILE) -> None:
    """
    Append output metadata to a cache file for fast lookup and deduplication.
    Robustly handles orjson serialization and file writing.
    """
    cache_entry = {
        "timestamp": format_timestamp(),
        "output_path": output_path,
        "metadata": metadata,
    }
    try:
        serialized = orjson.dumps(cache_entry)
    except Exception as e:
        logger.error(f"[OUTPUT_UTILS] Failed to serialize cache entry: {e}")
        serialized = str(cache_entry).encode("utf-8")
    try:
        with open(cache_file, "ab") as f:
            f.write(serialized + b"\n")
    except Exception as e:
        logger.error(f"[OUTPUT_UTILS] Failed to write to cache file: {e}")

def check_existing_output(metadata, cache_file=CACHE_FILE) -> Optional[dict]:
    """
    Check if output for this context already exists in the cache.
    Handles both JSONL (one JSON object per line) and JSON array formats.
    """
    if not os.path.exists(cache_file):
        return None
    with open(cache_file, "rb") as f:
        content = f.read().strip()
        if not content:
            return None
        entries = []
        # Try JSON array first
        try:
            if content.startswith(b"["):
                arr = orjson.loads(content)
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
                    entries.append(orjson.loads(line))
                except Exception as e:
                    logger.debug(f"[DEBUG] Failed to parse line as JSON: {line!r}")
                    continue
        for entry in entries:
            meta = safe_get(entry, "metadata", {})
            if (
                safe_get(meta, "state", "Unknown") == safe_get(metadata, "state", "Unknown") and
                safe_get(meta, "county", "Unknown") == safe_get(metadata, "county", "Unknown") and
                safe_get(meta, "year", "Unknown") == safe_get(metadata, "year", "Unknown") and
                safe_get(meta, "contests", "Unknown") == safe_get(metadata, "contests", "Unknown")
            ):
                return entry
    return None

def convert_sets_to_lists(obj) -> dict:
    if isinstance(obj, dict):
        return {k: convert_sets_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, list):
        return [convert_sets_to_lists(i) for i in obj]
    else:
        return obj

def deep_merge_dicts(dest, src) -> dict:
    """
    Recursively merge src into dest.
    - If a key exists in both and both values are dicts, merge them recursively.
    - Otherwise, src overwrites dest.
    """
    for k, v in safe_items(src):
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
    contest,
    state,
    county,
    context=None,
    enable_user_feedback=False,
    session_id=None
) -> dict:
    """
    Finalize and write election output to CSV and metadata JSON.
    Output is always placed in a subfolder of the project root (parent of webapp).
    Handles path-injection risks and robustly includes coordinator/session_id.
    """
    from ..Context_Integration.context_organizer import ContextOrganizer
    import re

    if context is None:
        context = {}

    logger.info(f"[OUTPUT_UTILS] finalize_election_output called with contest: '{contest}'")

    meta = {
        "contests": contest or "Unknown",
        "year": "Unknown",
        "state": state or "Unknown",
        "county": county or "Unknown"
    }
    match = re.search(r"\b(19|20)\d{2}\b", contest or "")
    if match:
        meta["year"] = match.group(0)

    organized = ContextOrganizer.organize_context(meta)
    enriched_meta = safe_get(organized, "metadata", meta)

    # Defensive: ensure required fields
    if not safe_get(enriched_meta, "contests", []):
        enriched_meta["contests"] = contest or "Unknown"
    if not safe_get(enriched_meta, "year", []) or not (str(safe_get(enriched_meta, "year", "")).isdigit() and len(str(safe_get(enriched_meta, "year", ""))) == 4):
        enriched_meta["year"] = safe_get(meta, "year", "Unknown")
    if not safe_get(enriched_meta, "state", []):
        enriched_meta["state"] = state or "Unknown"
    if not safe_get(enriched_meta, "county", []):
        enriched_meta["county"] = county or "Unknown"
    if session_id is not None:
        enriched_meta["session_id"] = session_id
    if coordinator is not None:
        enriched_meta["coordinator"] = str(type(coordinator).__name__)

    organizer = ContextOrganizer()
    organizer.append_to_context_library({"metadata": enriched_meta})

    # --- Path-injection mitigation: sanitize all path parts and validate ---
    def safe_filename(s: str) -> str:
        # Only allow alphanumeric, space, underscore, dash
        sanitized = "".join(c if c.isalnum() or c in " _-" else "_" for c in str(s)).strip() or "Unknown"
        # Remove dangerous patterns
        sanitized = sanitized.replace("..", "_").replace("/", "_").replace("\\", "_")
        return sanitized

    year = safe_get(enriched_meta, "year", "")
    state = safe_get(enriched_meta, "state", "")
    county = safe_get(enriched_meta, "county", "")
    election_types = safe_get(enriched_meta, "election_types", "")
    contests = safe_get(enriched_meta, "contests", "")

    parts = [
        safe_filename(state).lower() if state else "",
        safe_filename(county).lower() if county else "",
        str(year) if year and str(year).isdigit() and len(str(year)) == 4 else "Unknown",
        safe_filename(election_types).lower() if election_types else "",
        safe_filename(contests).replace(" ", "_") if contests else "unknown_contests",
        "parsed"
    ]
    # Remove any empty or dangerous parts
    parts = [p for p in parts if p and p != "." and p != ".."]

    output_root = get_output_root()
    output_path = safe_join(output_root, *parts)

    # Ensure output_path is inside output_root (redundant with safe_join, but double-check)
    abs_output_root = os.path.abspath(output_root)
    abs_output_path = os.path.abspath(output_path)
    if not abs_output_path.startswith(abs_output_root):
        raise ValueError("Unsafe output path detected.")

    os.makedirs(output_path, exist_ok=True)

    timestamp = format_timestamp()
    safe_title = safe_filename(contest or contests or "results").replace(" ", "_")
    filename_parts = [
        str(year) if year and str(year).isdigit() and len(str(year)) == 4 else "",
        safe_filename(state).lower() if state else "",
        safe_filename(county).lower() if county else "",
        safe_filename(election_types).lower() if election_types else "",
        safe_title,
        "results",
        timestamp
    ]
    filename = "_".join([p for p in filename_parts if p and p != "." and p != ".."]).replace("__", "_") + ".csv"
    # Path-injection mitigation for filename
    filename = safe_filename(filename)
    # Final filename validation
    if ".." in filename or "/" in filename or "\\" in filename:
        raise ValueError("Unsafe filename detected.")

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
    if session_id is not None:
        metadata_out["session_id"] = session_id
    if coordinator is not None:
        metadata_out["coordinator"] = str(type(coordinator).__name__)

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

    with open(json_meta_path, "wb") as jf:
        metadata_out = convert_sets_to_lists(metadata_out)
        jf.write(orjson.dumps(metadata_out, option=orjson.OPT_INDENT_2))

    update_output_cache(metadata_out, filepath)

    logger.info(f"[bold green][OUTPUT][/bold green] Wrote [bold]{len(data)}[/bold] rows to:\n  [cyan]{filepath}[/cyan]")
    logger.info(f"[bold green][OUTPUT][/bold green] Metadata written to:\n  [cyan]{json_meta_path}[/cyan]")

    if enable_user_feedback or os.environ.get("ENABLE_USER_FEEDBACK", "false").lower() == "true":
        feedback_log_path = safe_join(LOG_DIR, "user_feedback_log.jsonl")
        os.makedirs(LOG_DIR, exist_ok=True)
        feedback = input("\n[Feedback] Would you like to provide feedback or corrections for this output? (Leave blank to skip):\n> ").strip()
        if feedback:
            feedback_entry = {
                "timestamp": format_timestamp(),
                "file": filepath,
                "metadata": metadata_out,
                "feedback": feedback
            }
            with open(feedback_log_path, "ab") as fb:
                fb.write(orjson.dumps(feedback_entry) + b"\n")
            logger.info(f"[bold blue][FEEDBACK][/bold blue] Feedback logged to {feedback_log_path}")

    return {
        "csv_path": filepath,
        "metadata_path": json_meta_path,
        "output_file": filepath
    }