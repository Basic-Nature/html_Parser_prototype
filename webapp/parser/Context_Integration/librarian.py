# webapp/parser/Context_Integration/librarian.py
# -----------------------------------------------------------------------------------
# This file contains functions to manage the context library for the HTML parser,
# including loading, saving, and updating the context library, as well as
# It also includes utilities for logging unknown HTML tags and attributes,
# extending context library structures, and handling ML/LLM feedback.
#
# SECURITY: All file operations are validated using safe_path() to prevent path traversal attacks.
# -----------------------------------------------------------------------------------
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import orjson

from ..config import BASE_DIR, CONTEXT_LIBRARY_PATH, LOG_DIR, PROJECT_ROOT
from ..utils.logger_singleton import logger
from ..utils.misc_utils import file_hash
from ..utils.shared_logic import (
    safe_append,
    safe_filename,
    safe_get,
    safe_merge_defaults,
    safe_setdefault,
)
from .vocab.loader import get_vocab_loader
from .Context_Library.constants import (
    BALLOT_TYPES,
    CANDIDATE_KEYWORDS,
    CANONICAL_SEGMENT_LABELS,
    CANONICAL_STATE_ABBR,
    CUSTOM_ATTR_PATTERNS,
    HEADING_TAGS,
    HTML_TAGS,
    KNOWN_STATE_TO_COUNTY_MAP,
    LOCATION_KEYWORDS,
    PANEL_TAGS,
    STATE_ABBR,
)

_CONTEXT_LOCK = threading.Lock()
SCHEMA_VERSION = "1.0"
_TEMP_CONTEXT_LIB_TEMPFILES = set()

# SECURITY: Define allowed root directories for all file operations
LOG_DIR_PATH = Path(LOG_DIR).resolve()
CONTEXT_LIBRARY_DIR = Path(CONTEXT_LIBRARY_PATH).parent.resolve()
PROJECT_ROOT_PATH = Path(PROJECT_ROOT).resolve()
BASE_DIR_PATH = Path(BASE_DIR).resolve()

ALLOWED_ROOTS = [LOG_DIR_PATH, CONTEXT_LIBRARY_DIR, PROJECT_ROOT_PATH, BASE_DIR_PATH]

DEFAULT_STRUCTURE = {
    "schema_version": SCHEMA_VERSION,
    "contests": [],
    "panels": [],
    "tables": [],
    "buttons": [],
    "metadata": {},
}
_context_library_cache = None


def get_vocab_constant(
    subdir: str,
    filename: str,
    *,
    mapping: bool = False,
    session_id: str | None = None,
):
    """Load a vocab constant from Context_Integration/vocab with caching."""
    loader = get_vocab_loader()
    if mapping:
        return loader.load_mapping(subdir, filename, session_id=session_id)
    return loader.load_canonical(subdir, filename, session_id=session_id)


def safe_path(path, allowed_roots=None):
    """
    Validate that a path is within allowed directories to prevent path traversal attacks.
    
    Args:
        path: Path to validate
        allowed_roots: List of allowed root directories (defaults to ALLOWED_ROOTS)
    
    Returns:
        Resolved Path object if valid
    
    Raises:
        ValueError: If path is outside allowed directories
    """
    if allowed_roots is None:
        allowed_roots = ALLOWED_ROOTS
    
    path = Path(path).resolve()
    for root in allowed_roots:
        root = Path(root).resolve()
        try:
            # Check if path is relative to root
            path.relative_to(root)
            return path
        except ValueError:
            continue
    
    raise ValueError(f"Path traversal detected: {path} is not within allowed directories {allowed_roots}")

def get_safe_log_path(filename: str) -> Path:
    """
    Returns a safe log path inside the LOG_DIR directory.
    SECURITY: Validates path after sanitization.
    Prevents path-injection and directory traversal.
    Ensures the log directory exists.
    """
    log_dir = LOG_DIR_PATH
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # SECURITY: Sanitize the filename robustly
    safe_name = safe_filename(os.path.basename(filename))
    log_path = log_dir / safe_name
    
    # SECURITY: Validate the resolved path is inside LOG_DIR
    try:
        log_path = safe_path(log_path, ALLOWED_ROOTS)
    except ValueError as e:
        raise ValueError(f"Unsafe log path detected: {filename} -> {log_path}") from e
    
    return log_path

def atomic_write_json(obj, path) -> None:
    """
    Atomically write JSON to path, keeping only the latest .bak and .tmp.
    SECURITY: Path is validated before any file operations.
    - Writes to .tmp first, then moves to final path.
    - If path exists, creates a .bak (removing any old .bak).
    - Cleans up any stray .tmp before/after.
    """
    # SECURITY: Validate path before any operations
    path = safe_path(path, ALLOWED_ROOTS)
    
    backup_path = path.with_suffix(path.suffix + ".bak")
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    
    # SECURITY: Validate derived paths
    backup_path = safe_path(backup_path, ALLOWED_ROOTS)
    tmp_path = safe_path(tmp_path, ALLOWED_ROOTS)

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
        except (OSError, PermissionError, FileExistsError):
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
    """
    SECURITY: Join paths and validate against base directory.
    Now uses safe_path for validation.
    """
    final_path = os.path.abspath(os.path.join(base, *paths))
    
    # SECURITY: Validate final path
    try:
        safe_path(final_path, ALLOWED_ROOTS)
    except ValueError:
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

def load_context_library(path=None) -> Dict[str, Any]:
    """
    Loads the context library robustly:
    SECURITY: Path validated before any operations.
    - If missing, creates with default structure.
    - If empty or corrupt, backs up and re-initializes.
    - If missing keys, adds them (preserving existing data).
    - Extends dynamic sets with loaded values.
    Uses safe_merge_defaults for robust merging.
    """
    if path is None:
        path = CONTEXT_LIBRARY_PATH
    
    # SECURITY: Validate path
    safe_path_obj = safe_path(path, ALLOWED_ROOTS)
    safe_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # If file does not exist or is empty, create with defaults
    if not safe_path_obj.exists() or safe_path_obj.stat().st_size == 0:
        context_lib = {
            "panel_tags": list(PANEL_TAGS),
            "heading_tags": list(HEADING_TAGS),
            "custom_attr_patterns": [pat.pattern for pat in CUSTOM_ATTR_PATTERNS],
            "location_keywords": list(LOCATION_KEYWORDS),
            "candidate_keywords": list(CANDIDATE_KEYWORDS),
            "ballot_types": list(BALLOT_TYPES),
            **DEFAULT_STRUCTURE
        }
        with open(safe_path_obj, "wb") as f:
            f.write(orjson.dumps(context_lib, option=orjson.OPT_INDENT_2))
        return context_lib

    # Try to load, back up and re-init if corrupt
    try:
        with open(safe_path_obj, "rb") as f:
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
                with open(safe_path_obj, "wb") as fw:
                    fw.write(orjson.dumps(context_lib, option=orjson.OPT_INDENT_2))
                return context_lib
            context_lib = robust_orjson_loads(data)
    except Exception as e:
        logger.error(f"corrupt context library: {e}")
        # SECURITY: Validate backup path
        backup_path = safe_path(str(safe_path_obj) + ".corrupt", ALLOWED_ROOTS)
        try:
            os.rename(safe_path_obj, backup_path)
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
        with open(safe_path_obj, "wb") as f:
            f.write(orjson.dumps(context_lib, option=orjson.OPT_INDENT_2))
        return context_lib

    # Merge in any missing keys from default (preserve existing data)
    if safe_merge_defaults(context_lib, DEFAULT_STRUCTURE):
        save_context_library(context_lib, safe_path_obj)

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
    SECURITY: Path validated before operations.
    If a dict is passed instead of a function, it will update the library with that dict.
    """
    with _CONTEXT_LOCK:
        lib = load_context_library(path)
        # Accept either a function or a dict
        if isinstance(update_fn, dict):
            lib.update(update_fn)
        else:
            update_fn(lib)
        lib = clean_for_json(lib)  # <-- Ensure all sets are converted before saving
        save_context_library(lib, path)
       
def backup_context_library(path=None, max_backups=5) -> None:
    """
    Make a timestamped backup of the context library before overwriting,
    SECURITY: All paths validated.
    but only if the content has changed. Keep only the most recent `max_backups` backups.
    """
    if path is None:
        path = CONTEXT_LIBRARY_PATH
    
    # SECURITY: Validate path
    path = safe_path(path, ALLOWED_ROOTS)
    
    if not path.exists():
        return

    dir_ = path.parent
    base = path.name
    # Only match timestamped .bak files
    backups = sorted(
        [f for f in dir_.iterdir() if f.name.startswith(base + ".") and f.name.endswith(".bak")],
        reverse=True
    )
    current_hash = file_hash(str(path))
    if backups:
        last_backup_path = backups[0]
        # SECURITY: Validate backup path
        last_backup_path = safe_path(last_backup_path, ALLOWED_ROOTS)
        try:
            if file_hash(str(last_backup_path)) == current_hash:
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
        [f for f in dir_.iterdir() if f.name.startswith(base + ".") and f.name.endswith(".bak")],
        reverse=True
    )
    for old in backups[max_backups:]:
        try:
            os.remove(old)
        except Exception:
            pass

def save_context_library(lib, path=None) -> None:
    """
    Robustly save the context library:
    SECURITY: All paths validated before operations.
    - Always makes a timestamped backup before writing.
    - Writes atomically (temp file, then replace).
    - Cleans up previous temp files to avoid disk bloat.
    - Never truncates or loses data on failure.
    """
    global _TEMP_CONTEXT_LIB_TEMPFILES
    if path is None:
        path = CONTEXT_LIBRARY_PATH
    
    # SECURITY: Validate path
    safe_path_obj = safe_path(path, ALLOWED_ROOTS)
    
    backup_context_library(safe_path_obj)
    data = orjson.dumps(lib, option=orjson.OPT_INDENT_2)
    dir_name = safe_path_obj.parent

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

    # SECURITY: Validate temp file path
    temp_path = safe_path(temp_path, ALLOWED_ROOTS)
    
    _TEMP_CONTEXT_LIB_TEMPFILES.add(str(temp_path))

    # Atomically replace the original file
    try:
        os.replace(temp_path, safe_path_obj)
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
    SECURITY: Paths validated.
    """
    lib = load_context_library(path)
    lib.update(partial_dict)
    save_context_library(lib, path)

def update_context_library_field(key, value, path=CONTEXT_LIBRARY_PATH) -> None:
    """
    Safely update a top-level key in the context library.
    SECURITY: Path validated.
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
    """
    Get log path with security validation.
    SECURITY: Filename sanitized and path validated.
    """
    LOG_DIR_PATH.mkdir(parents=True, exist_ok=True)
    
    # SECURITY: Sanitize filename
    safe_name = safe_filename(os.path.basename(filename))
    log_path = LOG_DIR_PATH / safe_name
    
    # SECURITY: Validate path
    log_path = safe_path(log_path, ALLOWED_ROOTS)
    
    return str(log_path)

def _deduplicate_jsonl_log(log_path: str, key: str) -> Set[str]:
    """
    Deduplicate a JSONL log file by the given key ('tag' or 'attr').
    SECURITY: Path validated before operations.
    Keeps only the first occurrence of each value.
    """
    try:
        log_path_obj = safe_path(log_path, ALLOWED_ROOTS)
    except ValueError:
        return set()
    
    if not log_path_obj.exists():
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
    abbrs = CANONICAL_STATE_ABBR.get(state_name)
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
    """
    Self-heal: scan for misaligned NER, run correction bot, reload context library, repeat until clean or max_retries.
    SECURITY: Script paths validated before subprocess execution.
    """
    scan_script = Path(os.path.dirname(__file__)) / "../health/scan_misaligned_ner.py"
    
    # SECURITY: Validate scan script path
    try:
        scan_script = safe_path(scan_script, ALLOWED_ROOTS)
    except ValueError as e:
        logger.error(f"[SECURITY] Invalid scan script path: {scan_script} - {e}")
        return 2
    
    for attempt in range(1, max_retries + 1):
        logger.warning(f"\n[LIBRARIAN SELF-HEAL] Attempt {attempt}...")
        scan_cmd = [sys.executable, str(scan_script), "--jsonl", "log/spacy_ner_train_data.jsonl"]
        
        # SECURITY: Validate PROJECT_ROOT before use
        project_root = safe_path(PROJECT_ROOT, ALLOWED_ROOTS)
        
        scan_result = subprocess.run(scan_cmd, check=True, cwd=str(project_root))
        if scan_result.returncode == 0:
            logger.info("[LIBRARIAN SELF-HEAL] Data is clean. Exiting self-heal mode.")
            return 0
        logger.warning("[LIBRARIAN SELF-HEAL] Misalignments found. Launching manual_correction...")
        bot_cmd = [sys.executable, "-m", "webapp.parser.health.manual_correction", "--enhanced"]
        subprocess.run(bot_cmd, check=True, cwd=str(project_root))
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
def parse_filename_for_location(filename: str) -> Dict[str, Any]:
    """
    Parse filename for location hints, returning a dict with 'state', 'county', 'location', 'contest'.
    Special handling for known patterns like 'New York' locations.
    """
    fname = (
        filename.lower()
        .replace(".pdf", "")
        .replace(".csv", "")
        .replace(".json", "")
        .replace(".html", "")
        .replace(".htm", "")
    )
    state = None
    county = None
    location = ""
    contest = ""
    year = None

    # Extract year
    m = re.search(r"(19|20)\d{2}", fname)
    if m:
        try:
            year = int(m.group(0))
        except Exception:
            year = None

    # Split and parse parts
    parts = [p for p in fname.split("_") if p]

    # Build mapping of state tokens without spaces for quick lookup
    state_token_map = {s.replace("_", ""): s for s in STATE_ABBR.values()}
    state_token_map.update({abbr.lower(): name for abbr, name in STATE_ABBR.items()})

    detected_state_token = None
    for part in parts:
        token = re.sub(r"[^a-z]", "", part)
        if len(token) == 2 and token in STATE_ABBR:
            detected_state_token = STATE_ABBR[token]
            break
        if token in state_token_map:
            detected_state_token = state_token_map[token]
            break
    if detected_state_token:
        state = detected_state_token.replace("_", " ")

    # County: pick the last non-state, non-year part
    for part in reversed(parts):
        if part.isdigit():
            continue
        token = re.sub(r"[^a-z]", "", part)
        if state and token in (state.replace("_", ""), state.replace(" ", ""), state.split("_")[-1]):
            continue
        county = token.title()
        break

    # Location/contest basic split (best-effort)
    if county:
        location = county
    if not county and parts:
        location = parts[-1].title()

    def _state_display(raw: Optional[str]) -> Optional[str]:
        if not raw:
            return None
        return raw.replace("_", " ").title().replace(" ", "")

    return {
        "state": _state_display(state) if state else None,
        "county": county,
        "location": location,
        "contest": contest.strip(),
        "year": year,
    }
