"""
manual_correction_bot.py

Deep ML/LLM-enhanced batch review and correction bot for all context fields.
- Reads all *_selection_log.jsonl logs produced by ContextCoordinator.
- Allows user to review, accept, edit, or remove corrections for any field.
- Updates the context library atomically and (optionally) the DB.
- Integrates with integrity_check for anomaly/suspicion highlighting.
- Uses spaCy, ML, and external LLMs for advanced feedback, context awareness, and self-improvement.
- Can connect to ContextCoordinator and context_organizer for deeper learning and automation.
- Supports advanced debate/decision logic, including LLM-powered suggestions and process improvement.
"""

import argparse
import os
import orjson
import shutil
from pathlib import Path
from collections import defaultdict, Counter
import shelve
from datetime import datetime, timedelta
import hashlib
import subprocess
import sys
import time
from tempfile import NamedTemporaryFile
import importlib
from fastapi import FastAPI
import uvicorn
# --- Unified logger import ---
from ..utils.shared_logger import log_info, log_warning, log_error, log_debug
from us.states import lookup as us_state_lookup
import re
from ..bots.librarian import (
    update_context_library,
    SCHEMA_VERSION,
    DEFAULT_STRUCTURE,
    load_context_library,
)
# --- Config ---
# --- Directory and file constants ---
from ..config import PROJECT_ROOT, CONTEXT_LIBRARY_PATH, LOG_DIR, CONTEXT_LIBRARY_DIR, CACHE_DIR
from webapp.parser.Context_Integration.context_coordinator import ContextCoordinator
coordinator = ContextCoordinator()
# Ensure these are Path objects
LOG_DIR = Path(LOG_DIR)
CONTEXT_LIBRARY_PATH = Path(CONTEXT_LIBRARY_PATH)
CONTEXT_LIBRARY_DIR = Path(CONTEXT_LIBRARY_DIR)

# Ensure directories exist
LOG_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = LOG_DIR / "manual_correction_cache.db"
AUDIT_LOG_PATH = LOG_DIR / "manual_correction_audit.jsonl"
BATCH_SIZE = 100

def load_cache(expire_days=None):
    cache = shelve.open(str(CACHE_PATH))
    if expire_days is not None:
        now = datetime.now()
        expired = []
        for k, v in cache.items():
            ts = v.get("timestamp")
            if ts and (now - datetime.fromisoformat(ts)) > timedelta(days=expire_days):
                expired.append(k)
        for k in expired:
            del cache[k]
    return cache

def close_cache(cache):
    cache.close()

# --- Audit log ---
def write_audit_log(action, entry, user=None, before=None, after=None):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "action": action,
        "entry_hash": str(hash(orjson.dumps(entry))),
        "user": user,
        "before": before,
        "after": after,
        "entry": entry,
    }
    with open(AUDIT_LOG_PATH, "ab") as f:
        f.write(orjson.dumps(log_entry) + b"\n")

def process_logs_with_cache(log_files, cache):
    for log_file in log_files:
        entries = load_jsonl(log_file)
        for entry in entries:
            entry_id = str(hash(orjson.dumps(entry)))
            if entry_id in cache:
                continue  # Already processed
            # ...review logic...
            # After processing:
            cache[entry_id] = {"status": "accepted"}  # or "removed"/"edited"
    cache.sync()

def process_and_sync(log_files, context_library, cache, batch_size=100, sync_db=False):
    batch = []
    for log_file in log_files:
        entries = load_jsonl(log_file)
        for entry in entries:
            entry_id = str(hash(orjson.dumps(entry)))
            if entry_id in cache:
                continue
            # ...review logic...
            batch.append(entry)
            cache[entry_id] = {"status": "accepted", "timestamp": datetime.now().isoformat()}
            if len(batch) >= batch_size:
                update_context_with_new_entries(context_library, batch, entries)
                if sync_db:
                    update_database_with_context(context_library)
                batch.clear()
    # Final flush
    if batch:
        update_context_with_new_entries(context_library, batch, entries)
        if sync_db:
            update_database_with_context(context_library)
    cache.sync()

# Log and data file paths
FIELD_LOG_SUFFIX = "_selection_log.jsonl"
SEGMENT_FEEDBACK_LOG = LOG_DIR / "segment_feedback_log.jsonl"
PATTERN_KB_FILE = LOG_DIR / "dom_pattern_kb.jsonl"
DOWNLOAD_LINKS_LOG = LOG_DIR / "download_links_log.jsonl"
ANOMALY_LOG = LOG_DIR / "anomaly_log.jsonl"
EXPORT_DIR = LOG_DIR / "correction_exports"
MAIN_FIELDS = [
    "buttons", "panels", "tables", "contests", "districts", "states", "election_types", "years", "party", "candidate"
]
AUX_FIELDS = [
    # Feedback and error logs
    "segment_feedback", "structure_feedback", "html_handler_routing_failures",
    # NER/ML logs
    "spacy_ner_misaligned", "spacy_ner_train_data",
    # Unknowns
    "unknown_attrs", "unknown_tags",
    # Cache/auxiliary logs
    "context_cache", "removed_columns", "segment_label_cache",
    # Other possible logs
    "download_links", "anomaly", "dom_pattern_kb"
]
ALL_FIELDS = MAIN_FIELDS + AUX_FIELDS
SUCCESS_RESULTS = {"pass", "fuzzy_pass", "manual_correction", "user_corrected"}

def discover_field_types_from_logs(log_files, max_lines=100):
    """Scan log files and return a set of all field_type values found."""
    field_types = set()
    for log_file in log_files:
        try:
            with open(log_file, "rb") as f:
                for i, line in enumerate(f):
                    if i >= max_lines:
                        break
                    try:
                        entry = orjson.loads(line)
                        if isinstance(entry, dict) and "field_type" in entry:
                            field_types.add(entry["field_type"])
                    except Exception:
                        continue
        except Exception:
            continue
    return sorted(field_types)

# --- Utility: Atomic JSON write with backup ---
def atomic_write_json(obj, path):
    """
    Atomically write JSON to path, keeping only the latest .bak and .tmp.
    - Writes to .tmp first, then moves to final path.
    - If path exists, creates a .bak (removing any old .bak).
    - Cleans up any stray .tmp before/after.
    """
    import os
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
    import time
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

# --- Path security utility ---
def safe_path(path, allowed_roots):
    path = Path(path).resolve()
    for root in allowed_roots:
        root = Path(root).resolve()
        if str(path).startswith(str(root)):
            return path
    raise ValueError(f"Unsafe path detected: {path}")

# --- Optional: spaCy and LLM integration ---
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None

def llm_suggest_action(entry, context=None, api_key=None, model="gpt-4-turbo", provider="openai", system_prompt=None, temperature=0.2, max_tokens=200, extra_instructions=None):
    """
    Use an external LLM (OpenAI, Anthropic, etc.) to suggest a field or correction for the entry.
    """
    prompt = (
        "You are an expert election data context classifier and corrector.\n"
        "Given the following extracted value from an election context, and the context dictionary, "
        "suggest the most appropriate field (e.g., year, state, candidate, contest, etc.), a confidence score (0-1), "
        "and, if possible, a correction or improvement. "
        "If the value is ambiguous, explain why and suggest a process improvement or flag for review.\n"
        f"Extracted value: '{entry.get('extracted_value', '')}'\n"
        f"Context: {orjson.dumps(context or {}, ensure_ascii=False)}\n"
    )
    if extra_instructions:
        prompt += f"\nAdditional instructions: {extra_instructions}\n"
    system_prompt = system_prompt or (
        "You are a highly reliable, context-aware election data assistant. "
        "Always provide clear, actionable suggestions and flag ambiguous cases."
    )
    try:
        if provider == "openai":
            import openai
            openai.api_key = api_key
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message["content"]
        elif provider == "anthropic":
            import anthropic #type: ignore
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        else:
            log_error(f"Unknown LLM provider: {provider}")
    except Exception as e:
        log_error(f"LLM suggestion failed ({provider}): {e}")
    return None

def ml_score_entry(entry, coordinator=None):
    """
    Use ML/NER or coordinator's ML model to score the entry for likely correctness.
    Returns a float score between 0 and 1.
    """
    text = entry.get("extracted_value", "")
    score = 0.0
    if coordinator and hasattr(coordinator, "score_entry"):
        try:
            score = coordinator.score_entry(entry)
        except Exception as e:
            log_warning(f"Coordinator ML scoring failed: {e}")
    if nlp and text:
        doc = nlp(str(text))
        if doc.ents:
            score += 0.5
        if any(ent.label_ in {"DATE", "GPE", "ORG"} for ent in doc.ents):
            score += 0.2
        if len(text.split()) > 2:
            score += 0.1
    return min(score, 1.0)

def ml_suggest_field(entry, coordinator=None):
    """
    Use ML/NER or coordinator to suggest a better field for the entry.
    """
    text = entry.get("extracted_value", "")
    if coordinator and hasattr(coordinator, "suggest_field"):
        try:
            return coordinator.suggest_field(entry)
        except Exception as e:
            log_warning(f"Coordinator field suggestion failed: {e}")
    if nlp and text:
        doc = nlp(str(text))
        if doc.ents:
            return doc.ents[0].label_
    return None

def find_log_files(log_dir=LOG_DIR, cache_dir=None, suffixes=(".jsonl", ".json")):
    """
    Recursively find all log files with given suffixes in log_dir and cache_dir.
    """
    log_dir = safe_path(log_dir, [LOG_DIR])
    files = []
    for suf in suffixes:
        files.extend(log_dir.rglob(f"*{suf}"))
    if cache_dir:
        cache_dir = safe_path(cache_dir, [CONTEXT_LIBRARY_DIR])
        for suf in suffixes:
            files.extend(cache_dir.rglob(f"*{suf}"))
    return files

# --- JSONL utilities ---

def load_jsonl(path):
    path = safe_path(path, [LOG_DIR, CONTEXT_LIBRARY_DIR])
    if not path.exists():
        log_warning(f"Log file not found: {path}")
        return []
    entries = []
    with open(path, "rb") as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                try:
                    entries.append(orjson.loads(line))
                except Exception as e:
                    log_warning(f"[CORRUPT] {path} line {i}: {e}")
    return entries

# --- Log file hash/timestamp and offset tracking ---
def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def check_and_fix_json_files(
    directories=None, suffixes=(".json", ".jsonl"), auto_delete=True, verbose=True, quarantine=True, try_fix=True
):
    """
    Scan directories for JSON/JSONL files, try to fix corruption, and delete/quarantine if unrecoverable.
    Returns list of corrupted files (fixed or deleted/quarantined).
    """
    import re

    # Try to import tolerant JSON parser
    try:
        import json5
        has_json5 = True
    except ImportError:
        has_json5 = False

    if directories is None:
        directories = [LOG_DIR, CONTEXT_LIBRARY_DIR, CACHE_DIR]
    corrupted = []
    for directory in directories:
        directory = Path(directory)
        if not directory.exists():
            continue
        for suf in suffixes:
            for file in directory.rglob(f"*{suf}"):
                try:
                    if suf == ".jsonl":
                        # Try to load all lines, salvage valid ones if needed
                        valid_lines = []
                        with open(file, "rb") as f:
                            for line in f:
                                if line.strip():
                                    try:
                                        orjson.loads(line)
                                        valid_lines.append(line)
                                    except Exception:
                                        if verbose:
                                            log_warning(f"[CORRUPT-LINE] {file}: {line[:80]}...")
                        with open(file, "rb") as f:
                            all_lines = f.readlines()
                        if try_fix and len(valid_lines) < len(all_lines):
                            fixed_path = file.with_suffix(file.suffix + ".fixed")
                            with open(fixed_path, "wb") as out:
                                for line in valid_lines:
                                    out.write(line)
                            shutil.move(fixed_path, file)
                            if verbose:
                                log_warning(f"[FIXED] Salvaged {len(valid_lines)}/{len(all_lines)} lines in {file}")
                            continue  # File is now fixed, skip deletion
                        elif len(valid_lines) == len(all_lines):
                            continue  # All lines valid
                        else:
                            raise Exception("Unrecoverable .jsonl corruption")
                    else:
                        # Try to load as JSON, if fails, try tolerant parsing or salvage array elements
                        with open(file, "rb") as f:
                            content = f.read()
                        try:
                            orjson.loads(content)
                            continue  # Valid
                        except Exception:
                            pass
                        # Try tolerant parser (json5)
                        if try_fix and has_json5:
                            try:
                                with open(file, "r", encoding="utf-8") as f:
                                    text = f.read()
                                obj = json5.loads(text)
                                # If json5 can parse, rewrite as strict JSON
                                fixed_path = file.with_suffix(file.suffix + ".fixed")
                                with open(fixed_path, "wb") as out:
                                    out.write(orjson.dumps(obj, option=orjson.OPT_INDENT_2))
                                shutil.move(fixed_path, file)
                                if verbose:
                                    log_info(f"[FIXED] Tolerant parse (json5) succeeded for {file}")
                                continue
                            except Exception:
                                pass
                        # Try to salvage up to the last valid closing bracket
                        if try_fix:
                            try:
                                text = content.decode("utf-8", errors="ignore")
                                # Find last closing bracket for array or object
                                last_brace = max(text.rfind("}"), text.rfind("]"))
                                if last_brace != -1:
                                    truncated = text[:last_brace+1]
                                    try:
                                        obj = orjson.loads(truncated.encode("utf-8"))
                                        fixed_path = file.with_suffix(file.suffix + ".fixed")
                                        with open(fixed_path, "wb") as out:
                                            out.write(orjson.dumps(obj, option=orjson.OPT_INDENT_2))
                                        shutil.move(fixed_path, file)
                                        if verbose:
                                            log_info(f"[FIXED] Truncated and recovered {file}")
                                        continue
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                        # Try to salvage array elements (if file is a JSON array)
                        if try_fix:
                            try:
                                text = content.decode("utf-8", errors="ignore")
                                items = re.findall(r"\{.*?\}", text, re.DOTALL)
                                valid_objs = []
                                for item in items:
                                    try:
                                        valid_objs.append(orjson.loads(item.encode("utf-8")))
                                    except Exception:
                                        if verbose:
                                            log_warning(f"[CORRUPT-OBJ] {file}: {item[:80]}...")
                                if valid_objs:
                                    fixed_path = file.with_suffix(file.suffix + ".fixed")
                                    with open(fixed_path, "wb") as out:
                                        out.write(orjson.dumps(valid_objs, option=orjson.OPT_INDENT_2))
                                    shutil.move(fixed_path, file)
                                    if verbose:
                                        log_info(f"[FIXED] Salvaged {len(valid_objs)} objects in {file}")
                                    continue
                            except Exception:
                                pass
                        raise Exception("Unrecoverable .json corruption")
                except Exception as e:
                    corrupted.append(str(file))
                    if verbose:
                        log_warning(f"[CORRUPT] {file}: {e}")
                    if auto_delete:
                        try:
                            if quarantine:
                                quarantine_dir = file.parent / "corrupt"
                                quarantine_dir.mkdir(exist_ok=True)
                                file.rename(quarantine_dir / file.name)
                                if verbose:
                                    log_warning(f"[QUARANTINED] {file} -> {quarantine_dir / file.name}")
                            else:
                                file.unlink()
                                if verbose:
                                    log_warning(f"[DELETED] {file}")
                        except Exception as del_e:
                            log_error(f"[ERROR] Could not remove {file}: {del_e}")
    if verbose:
        log_info(f"[SUMMARY] Corrupted files found: {corrupted}")
    return corrupted

def find_log_files(log_dir=LOG_DIR, cache_dir=CACHE_DIR, suffixes=(".jsonl", ".json")):
    """
    Recursively find all log files with given suffixes in log_dir and cache_dir.
    """
    log_dir = safe_path(log_dir, [LOG_DIR])
    files = []
    for suf in suffixes:
        files.extend(log_dir.rglob(f"*{suf}"))
    if cache_dir:
        cache_dir = safe_path(cache_dir, [CACHE_DIR])
        for suf in suffixes:
            files.extend(cache_dir.rglob(f"*{suf}"))
    return files

def load_jsonl_incremental(path, cache):
    """Read only new lines since last offset for this file."""
    path = safe_path(path, [LOG_DIR, CONTEXT_LIBRARY_DIR])
    file_id = str(path)
    last_offset = cache.get(f"{file_id}_offset", 0)
    entries = []
    with open(path, "rb") as f:
        f.seek(last_offset)
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    entries.append(orjson.loads(line))
                except Exception as e:
                    log_warning(f"[CORRUPT] {path} line {line_num}: {e}")
        cache[f"{file_id}_offset"] = f.tell()
    cache[f"{file_id}_hash"] = file_hash(path)
    return entries

def save_jsonl(path, entries):
    path = safe_path(path, [LOG_DIR, CONTEXT_LIBRARY_DIR])
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        for entry in entries:
            f.write(orjson.dumps(entry) + b"\n")
    shutil.move(tmp_path, path)

# --- Deduplication utilities ---
def deduplicate_entries(entries, key_fields=("extracted_value", "field_type", "context_key")):
    """
    Deduplicate a list of dict entries by key fields (tuple of field names).
    Returns a list of unique entries and a count of duplicates skipped.
    """
    seen = set()
    unique = []
    for entry in entries:
        key = tuple(entry.get(f, None) for f in key_fields)
        if key not in seen:
            seen.add(key)
            unique.append(entry)
    return unique, len(entries) - len(unique)

def entry_key(entry):
    """
    Returns a tuple key for an entry for deduplication and lookup.
    """
    return (
        entry.get("extracted_value"),
        entry.get("field_type"),
        entry.get("context_key", "default")
    )

# --- Enhanced aggregate with deduplication and context check ---
def aggregate_successful_field_entries(log_file: Path, context_library=None, field_type=None, success_results=None, fast_mode=False):
    if success_results is None:
        success_results = SUCCESS_RESULTS
    field_entries = defaultdict(list)
    entries = load_jsonl(log_file)
    # Deduplicate log entries
    unique_entries, dup_count = deduplicate_entries(entries)
    # If context_library and field_type provided, skip already-existing entries
    skipped_existing = 0
    if context_library and field_type in context_library:
        existing_set = set()
        for e in context_library[field_type]:
            key = (
                e.get("extracted_value"),
                e.get("field_type"),
                e.get("context_key", "default")
            )
            existing_set.add(key)
        filtered = []
        for entry in unique_entries:
            key = (
                entry.get("extracted_value"),
                entry.get("field_type"),
                entry.get("context_key", "default")
            )
            if key not in existing_set:
                filtered.append(entry)
            elif fast_mode:
                # In fast mode, auto-accept exact duplicates
                pass
            else:
                skipped_existing += 1
        unique_entries = filtered
    # Group by context_key
    for entry in unique_entries:
        if entry.get("result") in success_results:
            context_key = entry.get("context_key", "default")
            field_entries[context_key].append(entry)
    return field_entries, dup_count, skipped_existing, len(unique_entries)

# --- Feedback loop (interactive and LLM/ML-powered) ---
def feedback_loop(new_entries, field_type, context_library_path, enhanced=True, coordinator=None, context_organizer=None, llm_api_key=None, llm_provider="openai", llm_model="gpt-4-turbo", llm_system_prompt=None, llm_extra_instructions=None, fast_mode=False):
    context_library_path = safe_path(context_library_path, [CONTEXT_LIBRARY_DIR])
    if not new_entries:
        log_info(f"No new entries to review for {field_type}.")
        return 0, 0, 0
    log_info(f"\n[FEEDBACK] Review new context library entries for {field_type}:")
    context_library = load_context_library(context_library_path)
    log_debug("DEBUG: Loaded context library:", type(context_library))
    if not isinstance(context_library, dict):
        log_error("ERROR: Context library is not a dictionary. Check your context library loading logic.")
        raise ValueError("Context library must be a dictionary. Check your context library loading logic.")
    changed = False
    accepted, edited, removed = 0, 0, 0
    # Summary preview
    total_new = sum(len(v) for v in new_entries.values())
    preview = Counter(entry.get("extracted_value") for vals in new_entries.values() for entry in vals)
    log_info(f"[SUMMARY] {total_new} new entries to review. Top values:")
    for val, count in preview.most_common(5):
        log_info(f"  {val!r}: {count} times")
    for context_key, values in new_entries.items():
        log_info(f"\nContext: {context_key}")
        for idx, val in enumerate(values):
            # Fast mode: auto-accept if exact duplicate in context library
            is_duplicate = False
            if fast_mode and field_type in context_library:
                for existing in context_library[field_type]:
                    if entry_key(existing) == entry_key(val):
                        is_duplicate = True
                        break
            if is_duplicate:
                accepted += 1
                continue
            log_info(f"  [{idx}] {val}")
            if enhanced:
                ml_score = ml_score_entry(val, coordinator)
                ml_field = ml_suggest_field(val, coordinator)
                log_info(f"    [ML] Score: {ml_score:.2f} | ML Field: {ml_field}")
                if llm_api_key:
                    llm_suggestion = llm_suggest_action(
                        val, context=context_library, api_key=llm_api_key, model=llm_model, provider=llm_provider,
                        system_prompt=llm_system_prompt, extra_instructions=llm_extra_instructions
                    )
                    log_info(f"    [LLM] Suggestion: {llm_suggestion}")
            action = "a" if fast_mode else (input("Accept (a), Edit (e), Remove (r), Skip (s)? [a]: ").strip().lower() or "a")
            if action == "a":
                accepted += 1
            elif action == "e":
                new_val = input("Edit entry (as JSON): ")
                try:
                    values[idx] = orjson.loads(new_val)
                    edited += 1
                except Exception as e:
                    log_warning(f"Invalid JSON, skipping edit: {e}")
            elif action == "r":
                values[idx] = None
                removed += 1
            else:
                continue
        # Remove deleted
        values = [v for v in values if v]
        new_entries[context_key] = values
    # Save accepted/edited entries
    update_context_with_new_entries(context_library_path, field_type, new_entries)
    log_info(f"[SUMMARY] Accepted: {accepted}, Edited: {edited}, Removed: {removed}")
    return accepted, edited, removed

# --- Log file cleanup ---
def trim_log_file(path: Path):
    """Remove duplicate entries from a log file, keeping only the first occurrence."""
    entries = load_jsonl(path)
    deduped, _ = deduplicate_entries(entries)
    save_jsonl(path, deduped)

# --- Context library update logic (atomic, validated, backup) ---
def update_context_with_new_entries(context_path, field_type, field_entries):
    context_path = safe_path(context_path, [CONTEXT_LIBRARY_DIR])
    def updater(library):
        if field_type not in library or not isinstance(library[field_type], dict):
            library[field_type] = {}
        for context_key, entries in field_entries.items():
            if context_key not in library[field_type]:
                library[field_type][context_key] = []
            for entry in entries:
                if entry not in library[field_type][context_key]:
                    library[field_type][context_key].append(entry)
    library = load_context_library(context_path)
    updater(library)
    # TODO: Add JSON schema validation here if desired
    atomic_write_json(library, context_path)

# --- Integrity check integration ---
def extract_year(text):
    # Use spaCy NER for DATE entities, fallback to regex for 4-digit years
    if nlp and text:
        doc = nlp(str(text))
        for ent in doc.ents:
            if ent.label_ == "DATE":
                # Try to extract a year from the DATE entity
                year_match = re.search(r"(19|20)\d{2}", ent.text)
                if year_match:
                    return int(year_match.group())
        # Fallback: regex anywhere in text
    match = re.search(r"(19|20)\d{2}", text)
    return int(match.group()) if match else None

def extract_state(text):
    # Use spaCy NER for GPE/LOC, fallback to regex for state abbreviations
    if nlp and text:
        doc = nlp(str(text))
        for ent in doc.ents:
            if ent.label_ in {"GPE", "LOC"}:
                # Try to match US state abbreviations or full names
                abbrev_match = re.match(r"^[A-Z]{2}$", ent.text.strip())
                if abbrev_match:
                    return ent.text.strip()
                # Try to map full state names to abbreviations
                try:
                    state_obj = us_state_lookup(ent.text.strip())
                    if state_obj:
                        return state_obj.abbr
                except Exception:
                    pass
    # Fallback: regex for state abbreviation
    match = re.search(r"\b([A-Z]{2})\b", text)
    return match.group(1) if match else None

def extract_county(text):
    # Use spaCy NER for GPE/LOC, fallback to regex for "X County"
    if nlp and text:
        doc = nlp(str(text))
        for ent in doc.ents:
            if ent.label_ in {"GPE", "LOC"} and "county" in ent.text.lower():
                # Extract just the county name
                county_match = re.match(r"([A-Za-z ]+) County", ent.text, re.IGNORECASE)
                if county_match:
                    return county_match.group(1).strip()
    match = re.search(r"([A-Za-z ]+) County", text)
    return match.group(1).strip() if match else None

def extract_type(text):
    # Use spaCy NER for EVENT or ORG, fallback to keyword search
    if nlp and text:
        doc = nlp(str(text))
        for ent in doc.ents:
            if ent.label_ == "EVENT":
                # Look for election types in the event
                for t in ["General", "Primary", "Special"]:
                    if t.lower() in ent.text.lower():
                        return t
            if ent.label_ == "ORG":
                for t in ["General", "Primary", "Special"]:
                    if t.lower() in ent.text.lower():
                        return t
    # Fallback: keyword search
    for t in ["General", "Primary", "Special"]:
        if t.lower() in text.lower():
            return t
    return None

def autofix_contest_fields(contest):
    changed = False
    title = contest.get("title", "")
    raw = contest.get("raw", {})
    # Try to fill year
    if not contest.get("year"):
        year = (
            extract_year(title)
            or extract_year(raw.get("title", ""))
            or raw.get("year")
        )
        if year:
            contest["year"] = year
            changed = True
    # Try to fill state
    if not contest.get("state"):
        state = (
            extract_state(title)
            or extract_state(raw.get("title", ""))
            or raw.get("state")
        )
        if state:
            contest["state"] = state
            changed = True
    # Try to fill county
    if not contest.get("county"):
        county = (
            extract_county(title)
            or extract_county(raw.get("title", ""))
            or raw.get("county")
        )
        if county:
            contest["county"] = county
            changed = True
    # Try to fill type
    if not contest.get("type_"):
        ctype = (
            extract_type(title)
            or extract_type(raw.get("title", ""))
            or raw.get("type_")
        )
        if ctype:
            contest["type_"] = ctype
            changed = True
    return changed

def highlight_anomalies(context_library, field_type, context_path=None, autofix=True):
    try:
        from ..Context_Integration.Integrity_check import analyze_contest_titles, summarize_context_entities
    except ImportError:
        log_warning("Could not import integrity_check for anomaly highlighting.")
        return
    if field_type == "contests" and "contests" in context_library:
        contests = context_library["contests"]
        results = analyze_contest_titles(contests)
        fixed_count = 0
        if results.get("integrity_issues"):
            log_info("[INTEGRITY] Issues detected:", results["integrity_issues"])
            if autofix:
                for issue in results["integrity_issues"]:
                    # Each issue should have a 'context' with the contest dict
                    contest = issue.get("context")
                    if contest and autofix_contest_fields(contest):
                        fixed_count += 1
        if results.get("flagged_suspicious"):
            log_info("[INTEGRITY] Suspicious entries:", results["flagged_suspicious"])
        entity_summary = summarize_context_entities(contests)
        log_info("\n[ENTITY SUMMARY]:")
        for label, count in entity_summary.items():
            log_info(f"  {label}: {count}")
        # Save fixes if any
        if autofix and fixed_count and context_path:
            update_context_library(context_path, context_library)
            log_info(f"[INTEGRITY] Auto-fixed {fixed_count} contests with missing fields and updated context library.")

# --- DB update logic (batch, periodic, error handling) ---
def update_database_with_context(library, db_path=None, coordinator=None, enhanced=True):
    if not db_path:
        db_path = CONTEXT_LIBRARY_DIR / "context_library.json"
    db_path = safe_path(db_path, [CONTEXT_LIBRARY_DIR])
    try:
        if enhanced and coordinator and hasattr(coordinator, "update_db_with_context"):
            coordinator.update_db_with_context(library, db_path)
        else:
            atomic_write_json(library, db_path)
        log_info(f"Database updated at {db_path}")
    except Exception as e:
        log_error(f"Failed to update DB: {e}")

# --- CLI/REST API hooks (REST stub) ---
def run_rest_api():
    try:
        app = FastAPI()
        @app.get("/status")
        def status():
            return {"status": "ok"}
        # Add more endpoints as needed
        uvicorn.run(app, host="127.0.0.1", port=8000)
    except ImportError:
        log_warning("FastAPI/uvicorn not installed.")

# --- Export/Import correction sessions ---
def export_correction_session(log_paths, export_dir=EXPORT_DIR):
    export_dir = safe_path(export_dir, [LOG_DIR])
    export_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_files = []
    for path in log_paths:
        path = safe_path(path, [LOG_DIR])
        dest = export_dir / f"{Path(path).stem}_{timestamp}.jsonl"
        shutil.copy2(path, dest)
        export_files.append(str(dest))
    log_info(f"[INFO] Exported correction session logs to: {export_files}")

def import_correction_session(import_file, dest_path):
    import_file = safe_path(import_file, [LOG_DIR, CONTEXT_LIBRARY_DIR])
    dest_path = safe_path(dest_path, [LOG_DIR, CONTEXT_LIBRARY_DIR])
    shutil.copy2(import_file, dest_path)
    log_info(f"[INFO] Imported correction session from {import_file} to {dest_path}")

def field_matches_log(field, log_name):
    """
    Robustly match a field name to a log file name.
    Handles singular/plural, underscores, and partial matches.
    """
    # Normalize: lowercase, singular/plural, underscores
    field_base = field.rstrip('s').lower()
    patterns = [
        rf"\b{re.escape(field_base)}s?\b",  # match singular or plural as a word
        rf"{re.escape(field_base)}(_|\b)",  # match as prefix with underscore or word boundary
    ]
    for pat in patterns:
        if re.search(pat, log_name.lower()):
            return True
    return False

# --- Example: Context Library Initialization and Version Check ---
def ensure_context_library(path):
    """
    Ensure the context library exists and is at the correct schema version.
    If missing, create with DEFAULT_STRUCTURE. Warn if schema version mismatches.
    """
    path = safe_path(path, [CONTEXT_LIBRARY_DIR])
    if not path.exists():
        log_info(f"Context library not found at {path}, initializing with default structure.")
        struct = DEFAULT_STRUCTURE.copy()
        struct["schema_version"] = SCHEMA_VERSION
        update_context_library(path, struct)
        return struct
    context_lib = load_context_library(path)
    log_debug("DEBUG: Loaded context library:", type(context_lib))
    if not isinstance(context_lib, dict):
        log_error("ERROR: Context library is not a dictionary. Check your context library loading logic.")
        raise ValueError("Context library must be a dictionary. Check your context library loading logic.")
    # Always set schema_version if missing
    if "schema_version" not in context_lib:
        context_lib["schema_version"] = SCHEMA_VERSION
        update_context_library(path, context_lib)
    if context_lib.get("schema_version") != SCHEMA_VERSION:
        log_warning(f"Schema version mismatch: found {context_lib.get('schema_version')}, expected {SCHEMA_VERSION}. Consider migrating.")
    return context_lib

def validate_training_data(train_data, nlp):
    """
    Validate and skip misaligned spaCy NER training examples to avoid [W030] warnings.
    """
    from spacy.training import offsets_to_biluo_tags
    valid_data = []
    for text, annots in train_data:
        try:
            tags = offsets_to_biluo_tags(nlp.make_doc(text), annots["entities"])
            if "-" in tags:
                log_warning(f"Skipping misaligned entity in: {text}")
                continue
            valid_data.append((text, annots))
        except Exception as e:
            log_warning(f"Error validating entity alignment: {e}")
    return valid_data

def summarize_misaligned_entities(log_path=None, top_n=10):
    from pathlib import Path
    from ..config import LOG_DIR
    if log_path is None:
        log_path = Path(LOG_DIR) / "spacy_ner_misaligned.jsonl"
    if not log_path.exists():
        log_warning("[MISALIGNED] No misaligned NER examples found.")
        return
    counter = Counter()
    with open(log_path, "rb") as f:
        for line in f:
            try:
                obj = orjson.loads(line)
                text = obj.get("text", "")
                counter[text] += 1
            except Exception:
                continue
    if not counter:
        log_warning("[MISALIGNED] No misaligned NER examples found.")
        return
    log_warning(f"\n[MISALIGNED] Top {top_n} most frequent misaligned NER texts:")
    for text, count in counter.most_common(top_n):
        log_warning(f"  {repr(text)}: {count} times")
    log_warning("[MISALIGNED] Consider cleaning or pattern-excluding these from your training data.")

# --- Main CLI logic ---
def main():
    parser = argparse.ArgumentParser(
        description="Deep ML/LLM-enhanced batch review and correction bot for all context fields.\n"
                    "Log files are matched to fields by checking if the field name is a substring of the log file name. "
                    "If no files match, you may need to adjust your log file naming or field list."
    )
    parser.add_argument("--context", type=str, default=str(CONTEXT_LIBRARY_PATH), help="Path to context_library.json")
    parser.add_argument("--log-dir", type=str, default=str(LOG_DIR), help="Directory containing *_selection_log.jsonl files")
    parser.add_argument("--fields", type=str, nargs="*", default=ALL_FIELDS, help="Fields to process (default: all)")
    parser.add_argument("--auto", action="store_true", help="Automatically accept all new entries (no prompt)")
    parser.add_argument("--flush-cache", action="store_true", help="Flush the cache of processed entries")
    parser.add_argument("--cache-expire-days", type=int, default=None, help="Expire cache entries older than N days")
    parser.add_argument("--sync-db", action="store_true", help="Sync context library to DB now")
    parser.add_argument("--export-audit-log", type=str, help="Export audit log to given path")
    parser.add_argument("--rest-api", action="store_true", help="Run REST API server")
    parser.add_argument("--enhanced", action="store_true", help="Enable enhanced learning and automation (spaCy, coordinator, context_organizer, LLM)")
    parser.add_argument("--llm-api-key", type=str, default=None, help="API key for external LLM (e.g., OpenAI/Anthropic)")
    parser.add_argument("--llm-provider", type=str, default="openai", help="LLM provider: openai or anthropic")
    parser.add_argument("--llm-model", type=str, default="gpt-4-turbo", help="LLM model name")
    parser.add_argument("--llm-system-prompt", type=str, default=None, help="Custom system prompt for LLM")
    parser.add_argument("--llm-extra-instructions", type=str, default=None, help="Extra instructions for LLM prompt")
    parser.add_argument("--integrity", action="store_true", help="Highlight anomalies using integrity_check")
    parser.add_argument("--update-db", action="store_true", help="Update the DB with the new context library after processing")
    parser.add_argument("--db-path", type=str, default=None, help="Path to DB file (if --update-db is set)")
    parser.add_argument("--feedback", action="store_true", help="Enable feedback mode (no-op, for compatibility)")
    parser.add_argument("--fast", action="store_true", help="Fast mode: auto-accept exact duplicates, skip review for them.")
    parser.add_argument("--batch", action="store_true", help="Batch review: allow accepting/removing all entries in a group at once.")
    parser.add_argument("--self-heal", action="store_true", help="Loop: scan -> correct -> rescan until clean or max retries")
    parser.add_argument("--max-retries", type=int, default=3, help="Max self-heal attempts")
    parser.add_argument("--cooldown", type=int, default=2, help="Seconds to wait between self-heal attempts")
    parser.add_argument("--dry-run", action="store_true", help="Preview what would be processed/accepted/removed, but make no changes.")
    parser.add_argument("--fix-corrupt-json", action="store_true", help="Scan and fix (delete) corrupted JSON/JSONL files in log/cache/library dirs")
    args = parser.parse_args()

    if args.rest_api:
        run_rest_api()
        return

    if args.flush_cache:
        cache = load_cache()
        cache.clear()
        close_cache(cache)
        log_info("Cache flushed.")
        return

    cache = load_cache(expire_days=args.cache_expire_days)

    if args.export_audit_log:
        shutil.copy2(AUDIT_LOG_PATH, args.export_audit_log)
        log_info(f"Audit log exported to {args.export_audit_log}")
        return

    if args.self_heal:
        scan_script = os.path.join(os.path.dirname(__file__), "scan_misaligned_ner.py")
        for attempt in range(1, args.max_retries + 1):
            log_info(f"\n[SELF-HEAL] Attempt {attempt}...")
            scan_cmd = [sys.executable, scan_script, "--jsonl", "log/spacy_ner_train_data.jsonl"]
            scan_result = subprocess.run(scan_cmd, check=True, cwd=PROJECT_ROOT)
            if scan_result.returncode == 0:
                log_info("[SELF-HEAL] Data is clean. Exiting self-heal mode.")
                break
            log_info("[SELF-HEAL] Misalignments found. Running manual correction...")
            # Run the normal correction logic (call main() recursively, but without --self-heal)
            args.self_heal = False
            main()
            log_info(f"[SELF-HEAL] Sleeping {args.cooldown}s before rescanning...")
            time.sleep(args.cooldown)
        else:
            log_info("[SELF-HEAL] Max retries reached. Some misalignments may remain.")
        return

    context_path = safe_path(args.context, [CONTEXT_LIBRARY_DIR])
    # --- Ensure context library exists and is valid ---
    context_library = ensure_context_library(context_path)
    # Ensure 'metadata' key exists and is a dict
    if "metadata" not in context_library or not isinstance(context_library["metadata"], dict):
        context_library["metadata"] = {}
    context_library["metadata"]["last_accessed"] = datetime.now().isoformat()
    # Only write at end if changed
    context_library_changed = False
    log_dir = safe_path(LOG_DIR, [LOG_DIR]) if LOG_DIR else None
    cache_dir = safe_path(CACHE_DIR, [CACHE_DIR]) if CACHE_DIR else None
    log_files = find_log_files(log_dir, cache_dir)
    log_info(f"Discovered {len(log_files)} log files in {log_dir}")
    log_debug(f"[DEBUG] Discovered log files: {[str(f) for f in log_files]}")
    discovered_fields = discover_field_types_from_logs(log_files)
    log_debug(f"[DEBUG] Discovered field types in logs: {discovered_fields}")
    # Use discovered fields if --fields is not set or empty
    fields = args.fields if args.fields else discovered_fields or ALL_FIELDS
    log_debug(f"[DEBUG] Fields to process: {fields}")
    if args.fix_corrupt_json:
        check_and_fix_json_files()
        return
    # --- Improved log file matching ---
    file_field_map = []
    for log_file in log_files:
        entries = load_jsonl(log_file)
        found_any = False
        for field in fields:
            # Check if any entry in the file matches the field_type
            if any(isinstance(entry, dict) and entry.get("field_type") == field for entry in entries):
                file_field_map.append((log_file, field))
                found_any = True
        if not found_any:
            # Still add the file for all fields, fallback to process and filter inside
            for field in fields:
                file_field_map.append((log_file, field))

    # log_debug which files will be attempted for which fields
    log_debug("[DEBUG] File/field processing plan:")
    for log_file, field in file_field_map:
        log_info(f"  Will process {log_file.name} for field '{field}'")

    if not file_field_map:
        log_warning("No log files matched any of the specified fields by content. Will attempt to process all log files and filter entries by field_type.")
        # Fallback: process all log files for all fields
        for log_file in log_files:
            for field in fields:
                file_field_map.append((log_file, field))

    batch_entries = []
    for log_file in log_files:
        for field in fields:
            if field in log_file.name:
                log_info(f"Processing {log_file} for field {field}")
                entries = load_jsonl_incremental(log_file, cache)
                unique_entries, _ = deduplicate_entries(entries)
                # ...review logic (auto/batch/interactive/feedback loop)...
                # For demo, auto-accept all:
                field_entries = defaultdict(list)
                for entry in unique_entries:
                    field_entries[entry.get("context_key", "default")].append(entry)
                    cache[str(hash(orjson.dumps(entry)))] = {
                        "status": "accepted",
                        "timestamp": datetime.now().isoformat(),
                        "action": "auto-accept",
                        "user": os.environ.get("USER", "system"),
                    }
                    write_audit_log("accept", entry, user=os.environ.get("USER", "system"))
                update_context_with_new_entries(context_path, field, field_entries)
                batch_entries.extend(unique_entries)
                # Periodic DB sync
                if args.sync_db and len(batch_entries) >= BATCH_SIZE:
                    context_library = load_context_library(context_path)
                    log_debug("DEBUG: Loaded context library:", type(context_library))
                    if not isinstance(context_library, dict):
                        log_error("ERROR: Context library is not a dictionary. Check your context library loading logic.")
                        raise ValueError("Context library must be a dictionary. Check your context library loading logic.")
                    update_database_with_context(context_library)
                    batch_entries.clear()
                break

    # Optionally: connect to coordinator/context_organizer if enhanced
    coordinator = None
    context_organizer = None
    if args.enhanced:
        try:
            coordinator_mod = importlib.import_module("webapp.parser.Context_Integration.context_coordinator")
            coordinator = getattr(coordinator_mod, "ContextCoordinator", None)
        except Exception as e:
            log_warning(f"Could not import context_coordinator: {e}")
            coordinator = None

        try:
            organizer_mod = importlib.import_module("webapp.parser.Context_Integration.context_organizer")
            context_organizer = getattr(organizer_mod, "ContextOrganizer", None)
        except Exception as e:
            log_warning(f"Could not import context_organizer: {e}")
            context_organizer = None

    # --- Refactored single processing loop ---
    total_accepted, total_edited, total_removed = 0, 0, 0
    total_duplicates, total_existing_skipped, total_new = 0, 0, 0
    processed_logs = 0
    for log_file, field in file_field_map:
        log_info(f"Processing {log_file} for field {field}")
        try:
            # Deduplicate and skip existing
            field_entries, dup_count, skipped_existing, n_new = aggregate_successful_field_entries(
                log_file, context_library, field, fast_mode=args.fast
            )
            total_duplicates += dup_count
            total_existing_skipped += skipped_existing
            total_new += n_new
            processed_logs += 1
            # log_info summary before review
            log_info(f"\n[SUMMARY] {log_file.name} | Field: {field}")
            log_info(f"  Unique new entries: {n_new}")
            log_info(f"  Duplicates skipped: {dup_count}")
            log_info(f"  Already in context library: {skipped_existing}")
            # Preview top 3 entries
            preview = []
            for v in field_entries.values():
                preview.extend(v)
            log_info(f"  Preview: {preview[:3]}")
            if args.dry_run:
                log_info(f"[DRY-RUN] Would process {n_new} new entries for field {field} from {log_file}")
                continue
            if args.auto or args.fast:
                update_context_with_new_entries(context_path, field, field_entries)
                log_info(f"Auto-accepted new entries for {field}.")
                total_accepted += sum(len(v) for v in field_entries.values())
                context_library_changed = True
            else:
                # Feedback loop returns accepted, edited, removed counts
                if args.batch:
                    for context_key, values in field_entries.items():
                        log_info(f"\nBatch review for context: {context_key}")
                        log_info(f"  Entries: {values}")
                        action = input("Accept all (a), Remove all (r), Skip (s)? [a]: ").strip().lower() or "a"
                        if action == "a":
                            update_context_with_new_entries(context_path, field, {context_key: values})
                            total_accepted += len(values)
                            context_library_changed = True
                        elif action == "r":
                            total_removed += len(values)
                        else:
                            continue
                else:
                    accepted, edited, removed = feedback_loop(
                        field_entries, field, context_path,
                        enhanced=args.enhanced,
                        coordinator=coordinator,
                        context_organizer=context_organizer,
                        llm_api_key=args.llm_api_key,
                        llm_provider=args.llm_provider,
                        llm_model=args.llm_model,
                        llm_system_prompt=args.llm_system_prompt,
                        llm_extra_instructions=args.llm_extra_instructions
                    )
                    total_accepted += accepted
                    total_edited += edited
                    total_removed += removed
                    context_library_changed = True
            # Optionally run integrity check
            if args.integrity:
                context_library = load_context_library(context_path)
                log_debug("DEBUG: Loaded context library:", type(context_library))
                if not isinstance(context_library, dict):
                    log_error("ERROR: Context library is not a dictionary. Check your context library loading logic.")
                    raise ValueError("Context library must be a dictionary. Check your context library loading logic.")
                highlight_anomalies(context_library, field, context_path, autofix=True)
            # Optionally update DB
            if args.update_db:
                context_library = load_context_library(context_path)
                log_debug("DEBUG: Loaded context library:", type(context_library))
                if not isinstance(context_library, dict):
                    log_error("ERROR: Context library is not a dictionary. Check your context library loading logic.")
                    raise ValueError("Context library must be a dictionary. Check your context library loading logic.")
                update_database_with_context(context_library, db_path=args.db_path, enhanced=args.enhanced, coordinator=coordinator)
            # Clean up log file after processing
            try:
                os.remove(log_file)
                log_info(f"Deleted processed log file: {log_file}")
            except Exception as e:
                log_warning(f"Could not delete log file {log_file}: {e}")
        except Exception as e:
            log_error(f"Failed to process {log_file} for field {field}: {e}")

    # Write context library only if changed
    if context_library_changed and not args.dry_run:
        context_library = load_context_library(context_path)
        log_debug("DEBUG: Loaded context library:", type(context_library))
        if not isinstance(context_library, dict):
            log_error("ERROR: Context library is not a dictionary. Check your context library loading logic.")
            raise ValueError("Context library must be a dictionary. Check your context library loading logic.")
        update_context_library(context_path, context_library)
        log_info(f"Context library updated at {context_path}")

    log_info("\n[SUMMARY] Manual Correction Bot Run Complete.")
    log_info(f"Log files processed: {processed_logs}")
    log_info(f"Total unique new entries: {total_new}")
    log_info(f"Total duplicates skipped: {total_duplicates}")
    log_info(f"Total already in context library: {total_existing_skipped}")
    log_info(f"Total accepted: {total_accepted}, Total edited: {total_edited}, Total removed: {total_removed}")
    if processed_logs == 0 or total_new == 0:
        log_warning("[WARNING] No entries were processed. Check your log file naming, field configuration, or use --dry-run for debugging.")

if __name__ == "__main__":
    main()
    summarize_misaligned_entities()