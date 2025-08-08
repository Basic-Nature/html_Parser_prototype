
from __future__ import annotations
# webapp/parser/Context_Integration/librarian.py
# -----------------------------------------------------------------------------------
# This file contains functions to manage the context library for the HTML parser,
# including loading, saving, and updating the context library, as well as
# It also includes utilities for logging unknown HTML tags and attributes,
# extending context library structures, and handling ML/LLM feedback.
# -----------------------------------------------------------------------------------
import os
import re
import orjson
import subprocess
import sys
import time
import shutil
import numpy as np
import time
import threading
import shutil
import tempfile
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Set, List, Any
from ..config import CONTEXT_LIBRARY_PATH, PROJECT_ROOT, LOG_DIR, BASE_DIR
from .Context_Library.constants import (
    BALLOT_TYPES, CANDIDATE_KEYWORDS, CANONICAL_SEGMENT_LABELS, CUSTOM_ATTR_PATTERNS, PANEL_TAGS,
    HEADING_TAGS, HTML_TAGS, LOCATION_KEYWORDS, STATE_ABBR, KNOWN_STATE_TO_COUNTY_MAP, _CANONICAL_STATE_ABBR
)
from ..utils.shared_logic import (
    safe_get, safe_merge_defaults, safe_setdefault, safe_startswith, safe_append, safe_filename
)

from ..utils.misc_utils import file_hash
from ..utils.logger_singleton import logger
_CONTEXT_LOCK = threading.Lock()
SCHEMA_VERSION = "1.0"
_TEMP_CONTEXT_LIB_TEMPFILES = set()


DEFAULT_STRUCTURE = {
    "schema_version": SCHEMA_VERSION,
    "contests": [],
    "panels": [],
    "tables": [],
    "buttons": [],
    "metadata": {},
}
_context_library_cache = None

def get_safe_log_path(filename: str) -> Path:
    """
    Returns a safe log path inside the LOG_DIR directory.
    Prevents path-injection and directory traversal.
    Ensures the log directory exists.
    """
    log_dir = LOG_DIR
    os.makedirs(log_dir, exist_ok=True)
    # Sanitize the filename robustly
    safe_name = safe_filename(os.path.basename(filename))
    log_path = Path(log_dir) / safe_name
    # Ensure the resolved path is inside LOG_DIR
    if not str(log_path.resolve()).startswith(str(Path(log_dir).resolve())):
        raise ValueError("Unsafe log path detected!")
    return log_path

def atomic_write_json(obj, path) -> None:
    """
    Atomically write JSON to path, keeping only the latest .bak and .tmp.
    - Writes to .tmp first, then moves to final path.
    - If path exists, creates a .bak (removing any old .bak).
    - Cleans up any stray .tmp before/after.
    """
    path = Path(path)
    backup_path = path.with_suffix(path.suffix + ".bak")
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    # Remove any old .tmp file before starting
    if tmp_path.exists():
        try:
            tmp_path.unlink()
        except Exception:
            pass

    # Remove any old .bak file before creating new backup
    if backup_path.exists():
        try:
            backup_path.unlink()
        except Exception:
            pass

    # Write to .tmp path
    with open(tmp_path, "wb") as tf:
        tf.write(orjson.dumps(obj, option=orjson.OPT_INDENT_2))

    # If the main file exists, back it up
    if path.exists():
        shutil.copy2(path, backup_path)

    # --- Fix: If the target file exists and is locked, try to close it or retry ---
    for _ in range(3):
        try:
            shutil.move(str(tmp_path), str(path))
            break
        except (OSError, PermissionError, FileExistsError) as e:
            # Try to remove the target file if possible (only if you are sure it's safe)
            try:
                os.remove(str(path))
            except Exception:
                pass
            time.sleep(0.5)
    else:
        raise RuntimeError(f"Could not move {tmp_path} to {path} after several attempts.")

    # Clean up any stray .tmp (should not exist, but just in case)
    if tmp_path.exists():
        try:
            tmp_path.unlink()
        except Exception:
            pass

# --- Extend/Modify Functions ---
def extend_panel_tags(new_tags: List[str]) -> None:
    global PANEL_TAGS
    PANEL_TAGS |= set(t.lower() for t in new_tags)

def extend_heading_tags(new_tags: List[str]) -> None:
    global HEADING_TAGS
    HEADING_TAGS |= set(t.lower() for t in new_tags)

def extend_html_tags(new_tags: List[str]) -> None:
    global HTML_TAGS
    HTML_TAGS |= set(t.lower() for t in new_tags)

def extend_custom_attr_patterns(new_patterns: List[str]) -> None:
    global CUSTOM_ATTR_PATTERNS
    for pat in new_patterns:
        if isinstance(pat, str):
            CUSTOM_ATTR_PATTERNS.append(re.compile(pat))
        else:
            CUSTOM_ATTR_PATTERNS.append(pat)

def extend_location_keywords(new_keywords: List[str]) -> None:
    global LOCATION_KEYWORDS
    LOCATION_KEYWORDS |= set(k.lower() for k in new_keywords)

def extend_candidate_keywords(new_keywords: List[str]) -> None:
    global CANDIDATE_KEYWORDS
    CANDIDATE_KEYWORDS |= set(k.lower() for k in new_keywords)

def extend_ballot_types(new_types: List[str]) -> None:
    global BALLOT_TYPES
    BALLOT_TYPES.extend([t for t in new_types if t not in BALLOT_TYPES])

def safe_join(base, *paths) -> str:
    final_path = os.path.abspath(os.path.join(base, *paths))
    if not safe_startswith(final_path, os.path.abspath(base)):
        logger.debug(f"DEBUG: Attempted to join {paths} to base {base} -> {final_path}")
        raise ValueError("Attempted Path Traversal Detected!")
    return final_path

def clean_for_json(obj) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items() if k != "_fixed_fields"}
    elif isinstance(obj, list):
        return [clean_for_json(i) for i in obj]
    elif isinstance(obj, set):
        return [clean_for_json(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        return clean_for_json(obj.tolist())
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    else:
        # Fallback: convert to string
        return str(obj)

# --- Context Library Integration ---
def robust_orjson_loads(val) -> Any:
    if isinstance(val, bytes):
        return orjson.loads(val)
    elif isinstance(val, str):
        return orjson.loads(val.encode("utf-8"))
    else:
        raise TypeError(f"Cannot decode type {type(val)} with orjson")

def load_context_library(path=CONTEXT_LIBRARY_PATH) -> Dict[str, Any]:
    """
    Loads the context library robustly:
    - If missing, creates with default structure.
    - If empty or corrupt, backs up and re-initializes.
    - If missing keys, adds them (preserving existing data).
    - Extends dynamic sets with loaded values.
    Uses safe_merge_defaults for robust merging.
    """
    safe_path = path
    os.makedirs(os.path.dirname(safe_path), exist_ok=True)

    # If file does not exist or is empty, create with defaults
    if not os.path.exists(safe_path) or os.path.getsize(safe_path) == 0:
        context_lib = {
            "panel_tags": list(PANEL_TAGS),
            "heading_tags": list(HEADING_TAGS),
            "custom_attr_patterns": [pat.pattern for pat in CUSTOM_ATTR_PATTERNS],
            "location_keywords": list(LOCATION_KEYWORDS),
            "candidate_keywords": list(CANDIDATE_KEYWORDS),
            "ballot_types": list(BALLOT_TYPES),
            **DEFAULT_STRUCTURE
        }
        with open(safe_path, "wb") as f:
            f.write(orjson.dumps(context_lib, option=orjson.OPT_INDENT_2))
        return context_lib

    # Try to load, back up and re-init if corrupt
    try:
        with open(safe_path, "rb") as f:
            data = f.read()
            if not data:
                # Empty file, treat as missing
                context_lib = {
                    "panel_tags": list(PANEL_TAGS),
                    "heading_tags": list(HEADING_TAGS),
                    "custom_attr_patterns": [pat.pattern for pat in CUSTOM_ATTR_PATTERNS],
                    "location_keywords": list(LOCATION_KEYWORDS),
                    "candidate_keywords": list(CANDIDATE_KEYWORDS),
                    "ballot_types": list(BALLOT_TYPES),
                    **DEFAULT_STRUCTURE
                }
                with open(safe_path, "wb") as fw:
                    fw.write(orjson.dumps(context_lib, option=orjson.OPT_INDENT_2))
                return context_lib
            context_lib = robust_orjson_loads(data)
    except Exception as e:
        logger.error(f"corrupt context library: {e}")
        # Backup corrupt file before overwriting
        backup_path = safe_path + ".corrupt"
        try:
            os.rename(safe_path, backup_path)
        except Exception:
            pass
        context_lib = {
            "panel_tags": list(PANEL_TAGS),
            "heading_tags": list(HEADING_TAGS),
            "custom_attr_patterns": [pat.pattern for pat in CUSTOM_ATTR_PATTERNS],
            "location_keywords": list(LOCATION_KEYWORDS),
            "candidate_keywords": list(CANDIDATE_KEYWORDS),
            "ballot_types": list(BALLOT_TYPES),
            **DEFAULT_STRUCTURE
        }
        with open(safe_path, "wb") as f:
            f.write(orjson.dumps(context_lib, option=orjson.OPT_INDENT_2))
        return context_lib

    # Merge in any missing keys from default (preserve existing data)
    if safe_merge_defaults(context_lib, DEFAULT_STRUCTURE):
        save_context_library(context_lib, safe_path)

    # Extend dynamic sets with loaded values
    if "panel_tags" in context_lib:
        extend_panel_tags(context_lib["panel_tags"])
    if "heading_tags" in context_lib:
        extend_heading_tags(context_lib["heading_tags"])
    if "custom_attr_patterns" in context_lib:
        extend_custom_attr_patterns(context_lib["custom_attr_patterns"])
    if "location_keywords" in context_lib:
        extend_location_keywords(context_lib["location_keywords"])
    if "candidate_keywords" in context_lib:
        extend_candidate_keywords(context_lib["candidate_keywords"])
    if "ballot_types" in context_lib:
        extend_ballot_types(context_lib["ballot_types"])

    return context_lib
    
def update_context_library(path, update_fn) -> None:
    """
    Safely update the context library at `path` by applying `update_fn(library)`.
    If a dict is passed instead of a function, it will update the library with that dict.
    """
    from .context_organizer import clean_for_json
    with _CONTEXT_LOCK:
        lib = load_context_library(path)
        # Accept either a function or a dict
        if isinstance(update_fn, dict):
            lib.update(update_fn)
        else:
            update_fn(lib)
        lib = clean_for_json(lib)  # <-- Ensure all sets are converted before saving
        save_context_library(lib, path)
       
def backup_context_library(path=CONTEXT_LIBRARY_PATH, max_backups=5) -> None:
    """
    Make a timestamped backup of the context library before overwriting,
    but only if the content has changed. Keep only the most recent `max_backups` backups.
    """
    if not os.path.exists(path):
        return

    dir_ = os.path.dirname(path)
    base = os.path.basename(path)
    # Only match timestamped .bak files
    backups = sorted(
        [f for f in os.listdir(dir_) if f.startswith(base + ".") and f.endswith(".bak")],
        reverse=True
    )
    current_hash = file_hash(path)
    if backups:
        last_backup_path = os.path.join(dir_, backups[0])
        try:
            if file_hash(last_backup_path) == current_hash:
                # No change since last backup
                return
        except Exception:
            pass

    # Remove any non-timestamped .bak (legacy or accidental)
    legacy_bak = os.path.join(dir_, base + ".bak")
    if os.path.exists(legacy_bak):
        try:
            os.remove(legacy_bak)
        except Exception:
            pass

    # Make new backup
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = f"{path}.{timestamp}.bak"
    shutil.copy2(path, backup_path)

    # Prune old backups (keep only the most recent max_backups)
    backups = sorted(
        [f for f in os.listdir(dir_) if f.startswith(base + ".") and f.endswith(".bak")],
        reverse=True
    )
    for old in backups[max_backups:]:
        try:
            os.remove(os.path.join(dir_, old))
        except Exception:
            pass

def save_context_library(lib, path=None) -> None:
    """
    Robustly save the context library:
    - Always makes a timestamped backup before writing.
    - Writes atomically (temp file, then replace).
    - Cleans up previous temp files to avoid disk bloat.
    - Never truncates or loses data on failure.
    """
    global _TEMP_CONTEXT_LIB_TEMPFILES
    if path is None:
        path = CONTEXT_LIBRARY_PATH
    safe_path = safe_join(BASE_DIR, os.path.relpath(path, BASE_DIR))
    backup_context_library(safe_path)
    data = orjson.dumps(lib, option=orjson.OPT_INDENT_2)
    dir_name = os.path.dirname(safe_path)

    # Remove any previous temp files for this context lib
    for temp_file in list(_TEMP_CONTEXT_LIB_TEMPFILES):
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception:
                pass
        _TEMP_CONTEXT_LIB_TEMPFILES.discard(temp_file)

    # Write to a temp file first
    with tempfile.NamedTemporaryFile("wb", dir=dir_name, delete=False) as tf:
        tf.write(data)
        temp_path = tf.name

    _TEMP_CONTEXT_LIB_TEMPFILES.add(temp_path)

    # Atomically replace the original file
    try:
        os.replace(temp_path, safe_path)
    except Exception as e:
        # Clean up temp file if replace fails
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        raise e

    # Remove temp file from tracker after successful replace
    _TEMP_CONTEXT_LIB_TEMPFILES.discard(temp_path)

def merge_and_save_context_library(partial_dict, path=CONTEXT_LIBRARY_PATH) -> None:
    """
    Safely merge a partial dict into the context library and save atomically.
    """
    lib = load_context_library(path)
    lib.update(partial_dict)
    save_context_library(lib, path)

def update_context_library_field(key, value, path=CONTEXT_LIBRARY_PATH) -> None:
    """
    Safely update a top-level key in the context library.
    """
    lib = load_context_library(path)
    old_value = lib.get(key, None)
    lib[key] = value
    save_context_library(lib, path)
    # Optionally log the change
    logger.info(f"Updated context_library field '{key}': {old_value} -> {value}")

def update_domain_selector_cache(domain, selector, label, success=True) -> None:
    lib = load_context_library()
    domain_selectors = safe_setdefault(lib, "domain_selectors", {})
    entry = {
        "selector": selector,
        "label": label,
        "success_count": 1 if success else 0,
        "last_used": datetime.now(timezone.utc).isoformat()
    }
    found = False
    # Safeguard iteration
    for e in safe_get(domain_selectors, domain, []):
        if safe_get(e, "selector", None) == selector:
            e["success_count"] += 1 if success else 0
            e["last_used"] = entry["last_used"]
            found = True
            break
    if not found:
        safe_setdefault(domain_selectors, domain, []).append(entry)
    update_context_library_field("domain_selectors", domain_selectors)

def get_domain_selectors(domain) -> List[Dict[str, Any]]:
    lib = load_context_library()
    domain_selectors = safe_get(lib, "domain_selectors", {})
    return safe_get(domain_selectors, domain, [])

def log_selector_attempt(domain, selector, label, success) -> None:
    """
    Robustly log a selector attempt for a domain, using safe_append and safe_get.
    """
    lib = load_context_library()
    attempts = safe_get(lib, "selector_attempts", [])
    safe_append(
        attempts,
        {
            "domain": domain,
            "selector": selector,
            "label": label,
            "success": success,
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
        logger
    )
    lib["selector_attempts"] = attempts
    update_context_library_field("selector_attempts", attempts)

# --- Unknown Tag/Attr Logging for ML/LLM Feedback ---
UNKNOWN_TAGS_LOG = set()
UNKNOWN_ATTRS_LOG = set()

def _get_log_path(filename: str) -> str:
    # Use the centralized LOG_DIR for all logs
    os.makedirs(LOG_DIR, exist_ok=True)
    return os.path.join(LOG_DIR, filename)

def _deduplicate_jsonl_log(log_path: str, key: str) -> Set[str]:
    """
    Deduplicate a JSONL log file by the given key ('tag' or 'attr').
    Keeps only the first occurrence of each value.
    """
    if not os.path.exists(log_path):
        return set()
    seen = set()
    deduped = []
    with open(log_path, "rb") as f:
        for line in f:
            try:
                entry = orjson.loads(line)
                val = entry.get(key)
                if val and val not in seen:
                    seen.add(val)
                    deduped.append(entry)
            except Exception:
                continue
    # Rewrite file with deduped entries
    with open(log_path, "wb") as f:
        for entry in deduped:
            f.write(orjson.dumps(entry) + b"\n")
    return seen

# Initialize sets with deduplication on first import
_UNKNOWN_TAGS_SET = None
_UNKNOWN_ATTRS_SET = None

def log_unknown_tag(tag: str, context_library) -> None:
    """
    Log unknown HTML tag to unknown_tags_log.jsonl as a valid JSON object per line.
    Deduplicates log file and prevents future duplicates.
    """
    global _UNKNOWN_TAGS_SET
    if _UNKNOWN_TAGS_SET is None:
        log_path = _get_log_path("unknown_tags_log.jsonl")
        _UNKNOWN_TAGS_SET = _deduplicate_jsonl_log(log_path, "tag")
    panel_tags = safe_get(context_library, "panel_tags", [])
    heading_tags = safe_get(context_library, "heading_tags", [])
    html_tags = safe_get(context_library, "html_tags", [])
    known_tags = set(panel_tags + heading_tags + html_tags)
    if isinstance(tag, str) and tag and tag not in known_tags and tag not in _UNKNOWN_TAGS_SET:
        _UNKNOWN_TAGS_SET.add(tag)
        try:
            log_path = _get_log_path("unknown_tags_log.jsonl")
            with open(log_path, "ab") as f:
                f.write(orjson.dumps({"tag": str(tag)}) + b"\n")
        except Exception as exc:
            logger.error(f"[LOG_UNKNOWN_TAG] Failed to log tag '{tag}': {exc}")

def log_unknown_attr(attr: str, context_library) -> None:
    """
    Log unknown HTML attribute to unknown_attrs_log.jsonl as a valid JSON object per line.
    Deduplicates log file and prevents future duplicates.
    """
    global _UNKNOWN_ATTRS_SET
    if _UNKNOWN_ATTRS_SET is None:
        log_path = _get_log_path("unknown_attrs_log.jsonl")
        _UNKNOWN_ATTRS_SET = _deduplicate_jsonl_log(log_path, "attr")
    pattern_strings = safe_get(context_library, "custom_attr_patterns", []) if context_library else []
    patterns = [re.compile(p) for p in pattern_strings] if pattern_strings else CUSTOM_ATTR_PATTERNS
    if not isinstance(attr, str) or not attr:
        return
    if attr.startswith("data-") or attr.startswith("aria-") or attr == "role":
        return
    if not any(pat.match(attr) for pat in patterns) and attr not in _UNKNOWN_ATTRS_SET:
        _UNKNOWN_ATTRS_SET.add(attr)
        try:
            log_path = _get_log_path("unknown_attrs_log.jsonl")
            with open(log_path, "ab") as f:
                f.write(orjson.dumps({"attr": str(attr)}) + b"\n")
        except Exception as exc:
            logger.error(f"[LOG_UNKNOWN_ATTR] Failed to log attr '{attr}': {exc}")

# --- ML/LLM Feedback Integration Example ---
def integrate_llm_feedback(new_panel_tags=None, new_heading_tags=None, new_attr_patterns=None, new_location_keywords=None, new_candidate_keywords=None, new_ballot_types=None) -> None:
    if new_panel_tags:
        extend_panel_tags(new_panel_tags)
    if new_heading_tags:
        extend_heading_tags(new_heading_tags)
    if new_attr_patterns:
        extend_custom_attr_patterns(new_attr_patterns)
    if new_location_keywords:
        extend_location_keywords(new_location_keywords)
    if new_candidate_keywords:
        extend_candidate_keywords(new_candidate_keywords)
    if new_ballot_types:
        extend_ballot_types(new_ballot_types)
    save_context_library()

def lookup_state(name: str) -> str | None:
    """
    Lookup a canonical state name from a name or abbreviation.
    Returns the canonical state name (e.g., 'arizona') or None if not found.
    """
    if not name:
        return None
    norm = name.strip().lower().replace('.', '').replace('_', ' ').replace('-', ' ')
    # Try direct match to canonical names
    if norm in STATE_ABBR.values():
        return norm
    # Try abbreviation lookup
    if norm in STATE_ABBR:
        return STATE_ABBR[norm]
    # Try matching after removing spaces
    norm_nospace = norm.replace(' ', '')
    for abbr, state in STATE_ABBR.items():
        if abbr.replace(' ', '') == norm_nospace:
            return state
    # Try matching canonical names with spaces removed
    for state in STATE_ABBR.values():
        if state.replace('_', ' ').replace(' ', '') == norm_nospace:
            return state
    return None

def get_state_abbr(state_name: str) -> str | None:
    """
    Given a canonical state name, return its standard two-letter abbreviation.
    """
    if not state_name:
        return None
    state_name = state_name.lower().replace(' ', '_')
    abbrs = _CANONICAL_STATE_ABBR.get(state_name)
    if abbrs:
        # Return the first (should be the standard two-letter abbr)
        return abbrs[0].upper()
    return None

def lookup_county(county_name: str, state_name: str = None) -> str | None:
    """
    Lookup a canonical county name, optionally within a given state.
    Returns the canonical county name (e.g., 'maricopa') or None if not found.
    """
    if not county_name:
        return None
    norm = county_name.strip().lower().replace('.', '').replace('-', ' ').replace('_', ' ')
    if state_name:
        state = lookup_state(state_name)
        if state and state in KNOWN_STATE_TO_COUNTY_MAP:
            for county in KNOWN_STATE_TO_COUNTY_MAP[state]:
                if county.replace('-', ' ').replace('_', ' ') == norm:
                    return county
    # Fallback: search all states
    for state, counties in KNOWN_STATE_TO_COUNTY_MAP.items():
        for county in counties:
            if county.replace('-', ' ').replace('_', ' ') == norm:
                return county
    return None

# --- Load context library at import time ---
load_context_library()

_segment_label_cache = {}

def normalize_segment_text(text: str) -> str:
    """Normalize segment text for canonical lookup (lowercase, strip, collapse spaces)."""
    if not text:
        return ""
    return " ".join(text.lower().strip().split())

def get_canonical_segment_label(text: str) -> str:
    """Return canonical label for normalized segment text, or None if not found."""
    norm = normalize_segment_text(text)
    return CANONICAL_SEGMENT_LABELS.get(norm)

def cache_segment_label(text: str, label: str) -> None:
    norm = normalize_segment_text(text)
    _segment_label_cache[norm] = label

def get_cached_segment_label(text: str) -> str:
    norm = normalize_segment_text(text)
    return _segment_label_cache.get(norm)

# --- Self-Heal Mode ---
def self_heal_context_library(max_retries=3, cooldown=2) -> None:
    """Self-heal: scan for misaligned NER, run correction bot, reload context library, repeat until clean or max_retries."""
    scan_script = os.path.join(os.path.dirname(__file__), "scan_misaligned_ner.py")
    for attempt in range(1, max_retries + 1):
        logger.warning(f"\n[LIBRARIAN SELF-HEAL] Attempt {attempt}...")
        scan_cmd = [sys.executable, scan_script, "--jsonl", "log/spacy_ner_train_data.jsonl"]
        scan_result = subprocess.run(scan_cmd, check=True, cwd=PROJECT_ROOT)
        if scan_result.returncode == 0:
            logger.info("[LIBRARIAN SELF-HEAL] Data is clean. Exiting self-heal mode.")
            return 0
        logger.warning("[LIBRARIAN SELF-HEAL] Misalignments found. Launching manual_correction_bot...")
        bot_cmd = [sys.executable, "-m", "webapp.parser.bots.manual_correction_bot", "--enhanced"]
        subprocess.run(bot_cmd, check=True, cwd=PROJECT_ROOT)
        logger.warning(f"[LIBRARIAN SELF-HEAL] Sleeping {cooldown}s before rescanning...")
        time.sleep(cooldown)
    logger.info("[LIBRARIAN SELF-HEAL] Max retries reached. Some misalignments may remain.")
    return 2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Librarian utility for context library management.")
    parser.add_argument("--self-heal", action="store_true", help="Loop: scan -> correct -> rescan until clean or max retries")
    parser.add_argument("--max-retries", type=int, default=3, help="Max self-heal attempts")
    parser.add_argument("--cooldown", type=int, default=2, help="Seconds to wait between self-heal attempts")
    args = parser.parse_args()
    if args.self_heal:
        sys.exit(self_heal_context_library(args.max_retries, args.cooldown))
# --- Export all sets for use in other modules ---
