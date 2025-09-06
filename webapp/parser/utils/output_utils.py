from __future__ import annotations
# webapp/parser/utils/output_utils.py
# ---------------------------------------------------------------
# Output utilities for Smart Elections Parser Webapp
# ---------------------------------------------------------------
import re
import csv
import orjson
import os
from datetime import datetime
import datetime as dt
from typing import List, Dict, Any, Optional
from .logger_singleton import logger
from .shared_logic import (
    safe_get_first, safe_items, safe_get, safe_lower,
    safe_filename
)
from ..config import (
    BASE_DIR, OUTPUT_DIR, ENABLE_USER_FEEDBACK, OUTPUT_CACHE
)

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

def update_output_cache(metadata, output_path, cache_file=OUTPUT_CACHE) -> None:
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

def check_existing_output(metadata, cache_file=OUTPUT_CACHE) -> Optional[dict]:
    """
    Check if output for this context already exists in the cache.
    Handles both JSONL (one JSON object per line) and JSON array formats.
    """
    if not os.path.exists(OUTPUT_CACHE):
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

def _slug(value: Optional[str], max_len: int = 80) -> str:
    if not isinstance(value, str):
        return "na"
    stem = value.strip()
    stem = re.sub(r"[^\w\s-]+", "_", stem, flags=re.UNICODE)
    stem = re.sub(r"[\s_-]+", " ", stem).strip()
    stem = stem.replace(" ", "_")
    stem = re.sub(r"_+", "_", stem)
    return stem[:max_len] or "na"

def _ensure_dir(p: str) -> None:
    try:
        os.makedirs(p, exist_ok=True)
    except Exception:
        pass

def _coerce_headers(headers: List[str], rows: List[Dict[str, Any]]) -> List[str]:
    base = [h for h in (headers or []) if isinstance(h, str) and h.strip()]
    seen = set(base)
    # Append any additional keys discovered in data, stable order
    for row in rows or []:
        if isinstance(row, dict):
            for k in row.keys():
                if k not in seen:
                    base.append(k)
                    seen.add(k)
    return base

def finalize_election_output(
    *,
    headers: List[str],
    data: List[Dict[str, Any]],
    coordinator=None,
    contest: Optional[str] = None,
    state: Optional[str] = None,
    county: Optional[str] = None,
    context: Optional[dict] = None,
    enable_user_feedback: bool = False,
    session_id: Optional[str] = None
) -> Dict[str, str]:
    """
    Centralized writer for CSV + metadata.
    Returns: {"csv_path": ..., "metadata_path": ...}
    """
    context = context or {}
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_DIR or os.path.join(os.getcwd(), "outputs")
    _ensure_dir(out_dir)

    # Build filenames
    state_slug = _slug(state)
    county_slug = _slug(county)
    contest_slug = _slug(contest, max_len=120)
    base_name = f"{state_slug}__{county_slug}__{contest_slug}__{ts}"
    csv_path = os.path.join(out_dir, f"{base_name}.csv")
    meta_path = os.path.join(out_dir, f"{base_name}.metadata.json")

    # Normalize headers and rows
    headers_final = _coerce_headers(headers or [], data or [])
    safe_rows: List[Dict[str, Any]] = []
    for row in (data or []):
        if not isinstance(row, dict):
            # Coerce non-dict rows to a single-column dict
            safe_rows.append({"value": str(row)})
            if "value" not in headers_final:
                headers_final = ["value"] + headers_final
            continue
        safe = {}
        for h in headers_final:
            val = row.get(h, "")
            # Keep scalars as-is, stringify complex structures
            if isinstance(val, (str, int, float)) or val is None:
                safe[h] = "" if val is None else val
            else:
                try:
                    safe[h] = orjson.dumps(val).decode("utf-8", errors="ignore")
                except Exception:
                    safe[h] = str(val)
        safe_rows.append(safe)

    # Write CSV
    try:
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers_final, extrasaction="ignore")
            writer.writeheader()
            for r in safe_rows:
                writer.writerow(r)
    except Exception as e:
        logger.error({
            "level": "ERROR",
            "type": "output",
            "message": f"[ERROR] Failed to write CSV: {e}",
            "session_id": session_id,
            "path": csv_path
        })
        # Best-effort path stub on failure
        return {"csv_path": "", "metadata_path": ""}

    # Build metadata
    meta = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "session_id": session_id,
        "handler": context.get("handler"),
        "input_file": context.get("input_file"),
        "contest": contest,
        "state": state,
        "county": county,
        "row_count": len(safe_rows),
        "headers": headers_final,
        "csv_path": csv_path,
        "context": context,
        "user_feedback_enabled": bool(enable_user_feedback),
    }
    try:
        with open(meta_path, "wb") as f:
            f.write(orjson.dumps(meta, option=orjson.OPT_INDENT_2))
    except Exception as e:
        logger.error({
            "level": "ERROR",
            "type": "output",
            "message": f"[ERROR] Failed to write metadata: {e}",
            "session_id": session_id,
            "path": meta_path
        })

    return {"csv_path": csv_path, "metadata_path": meta_path}