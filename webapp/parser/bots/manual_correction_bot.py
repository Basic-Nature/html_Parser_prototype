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
from ..utils.shared_logger import SharedLogger
from us.states import lookup as us_state_lookup
import re
from ..utils.misc_utils import file_hash
from ..Context_Integration.librarian import (
    update_context_library,
    SCHEMA_VERSION,
    DEFAULT_STRUCTURE,
    load_context_library,
)
# --- Config ---
# --- Directory and file constants ---
from ..config import PROJECT_ROOT, CONTEXT_LIBRARY_PATH, LOG_DIR, CONTEXT_LIBRARY_DIR, CACHE_DIR
from webapp.parser.Context_Integration.context_coordinator import ContextCoordinator
from ..utils.model_registry import ModelRegistry
logger = SharedLogger()
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
# To be moved over to centralized logic later
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

def discover_field_types_from_logs(log_files, all_fields=None, max_lines=100):
    """
    Scan log files and return a set of all field names found, based on ALL_FIELDS naming convention.
    Matches any key in the log entry that matches a known field name.
    """
    if all_fields is None:
        # Import or define ALL_FIELDS at the top of your file if not already
        all_fields = MAIN_FIELDS + AUX_FIELDS
    field_types = set()
    all_fields_set = set(all_fields)
    for log_file in log_files:
        try:
            with open(log_file, "rb") as f:
                for i, line in enumerate(f):
                    if max_lines is not None and i >= max_lines:
                        break
                    try:
                        entry = orjson.loads(line)
                        if isinstance(entry, dict):
                            # Check for explicit field_type key
                            if "field_type" in entry and entry["field_type"] in all_fields_set:
                                field_types.add(entry["field_type"])
                            # Otherwise, check for any key matching a known field
                            for key in entry.keys():
                                if key in all_fields_set:
                                    field_types.add(key)
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
            logger.error(f"Unknown LLM provider: {provider}")
    except Exception as e:
        logger.error(f"LLM suggestion failed ({provider}): {e}")
    return None

def ml_score_entry(entry, coordinator=None):
    """
    Use ML/NER or coordinator's ML model to score the entry for likely correctness.
    Returns a float score between 0 and 1.
    """
    if coordinator is None:
        coordinator = ContextCoordinator()
    text = entry.get("extracted_value", "")
    score = 0.0
    if coordinator and hasattr(coordinator, "score_entry"):
        try:
            score = coordinator.score_entry(entry)
        except Exception as e:
            logger.warning(f"Coordinator ML scoring failed: {e}")
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
    if coordinator is None:
        coordinator = ContextCoordinator()
    text = entry.get("extracted_value", "")
    if coordinator and hasattr(coordinator, "suggest_field"):
        try:
            return coordinator.suggest_field(entry)
        except Exception as e:
            logger.warning(f"Coordinator field suggestion failed: {e}")
    if nlp and text:
        doc = nlp(str(text))
        if doc.ents:
            return doc.ents[0].label_
    return None

# --- JSONL utilities ---

def load_jsonl(path):
    path = safe_path(path, [LOG_DIR, CONTEXT_LIBRARY_DIR])
    if not path.exists():
        logger.warning(f"Log file not found: {path}")
        return []
    entries = []
    with open(path, "rb") as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                try:
                    entries.append(orjson.loads(line))
                except Exception as e:
                    logger.warning(f"[CORRUPT] {path} line {i}: {e}")
    return entries

def check_and_fix_json_files(
    directories=None,
    suffixes=(".json", ".jsonl"),
    auto_delete=True,
    verbose=True,
    quarantine=True,
    try_fix=True,
    max_file_size_mb=50,
    schema_validator=None,
):
    """
    Robust, fast scan and correction for JSON/JSONL files.
    - Salvages valid lines/objects, quarantines unrecoverable, recreates minimal valid files if needed.
    - Uses regex, json5, and partial line merging for .jsonl.
    - Optionally validates schema.
    - Always logs summary and never aborts on error.
    """
    import json5
    import re

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
                    if not file.exists():
                        if verbose:
                            logger.warning(f"[SKIP] File not found: {file}")
                        continue
                    if file.stat().st_size > max_file_size_mb * 1024 * 1024:
                        if verbose:
                            logger.warning(f"[SKIP] File too large: {file}")
                        continue
                    backup_path = file.with_suffix(file.suffix + ".bak")
                    if try_fix and not backup_path.exists():
                        shutil.copy2(file, backup_path)
                    valid_objs = []
                    corrupt_items = []
                    # --- .jsonl logic ---
                    if suf == ".jsonl":
                        with open(file, "r", encoding="utf-8-sig", errors="replace") as f:
                            partial_line = ""
                            for i, line in enumerate(f):
                                line = line.strip()
                                if not line:
                                    continue
                                if partial_line:
                                    line = partial_line + line
                                    partial_line = ""
                                try:
                                    obj = json5.loads(line)
                                    if schema_validator and not schema_validator(obj):
                                        raise ValueError("Schema validation failed")
                                    valid_objs.append(obj)
                                except Exception:
                                    fixed_line = line
                                    fixed_line = re.sub(r",\s*$", "", fixed_line)
                                    fixed_line = re.sub(r"'", '"', fixed_line)
                                    fixed_line = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', fixed_line)
                                    fixed_line = fixed_line.replace('\ufeff', '').replace('\x00', '')
                                    if fixed_line.count("{") > fixed_line.count("}"):
                                        partial_line = fixed_line
                                        continue
                                    try:
                                        obj = json5.loads(fixed_line)
                                        if schema_validator and not schema_validator(obj):
                                            raise ValueError("Schema validation failed")
                                        valid_objs.append(obj)
                                        if verbose:
                                            logger.warning(f"[FIXED-LINE] {file} line {i+1}: {line[:80]}... -> {fixed_line[:80]}...")
                                    except Exception as e2:
                                        corrupt_items.append((i, line, str(e2)))
                                        if verbose:
                                            logger.warning(f"[CORRUPT-LINE] {file} line {i+1}: {line[:80]}... ({e2})")
                        # Write valid lines back
                        if try_fix:
                            with open(file, "w", encoding="utf-8") as out:
                                for obj in valid_objs:
                                    out.write(json5.dumps(obj, indent=2) + "\n")
                        if corrupt_items:
                            corrupt_path = file.with_suffix(file.suffix + ".corrupt")
                            with open(corrupt_path, "w", encoding="utf-8") as out:
                                for i, line, err in corrupt_items:
                                    out.write(f"Line {i+1}: {line}\nError: {err}\n\n")
                            if verbose:
                                logger.warning(f"[CORRUPT] {len(corrupt_items)} lines saved to {corrupt_path}")
                        if not valid_objs and try_fix:
                            with open(file, "w", encoding="utf-8") as out:
                                pass
                            if verbose:
                                logger.warning(f"[FIXED] All lines invalid, recreated empty .jsonl file: {file}")
                        continue
                    # --- .json logic ---
                    else:
                        try:
                            with open(file, "r", encoding="utf-8-sig", errors="replace") as f:
                                text = f.read()
                            obj = json5.loads(text)
                            if schema_validator and not schema_validator(obj):
                                raise ValueError("Schema validation failed")
                            valid_objs.append(obj)
                        except Exception:
                            fixed_text = text
                            fixed_text = re.sub(r",\s*([\]}])", r"\1", fixed_text)
                            fixed_text = re.sub(r"'", '"', fixed_text)
                            fixed_text = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', fixed_text)
                            fixed_text = fixed_text.replace('\ufeff', '').replace('\x00', '')
                            try:
                                obj = json5.loads(fixed_text)
                                if schema_validator and not schema_validator(obj):
                                    raise ValueError("Schema validation failed")
                                valid_objs.append(obj)
                                if verbose:
                                    logger.warning(f"[FIXED] {file}: applied regex fixes.")
                            except Exception as e2:
                                corrupt_items.append((0, text, str(e2)))
                                if verbose:
                                    logger.warning(f"[CORRUPT] {file}: {e2}")
                        if try_fix and valid_objs:
                            with open(file, "w", encoding="utf-8") as out:
                                out.write(json5.dumps(valid_objs[0], indent=2))
                            if verbose:
                                logger.info(f"[FIXED] Salvaged valid JSON in {file}")
                        if corrupt_items:
                            corrupt_path = file.with_suffix(file.suffix + ".corrupt")
                            with open(corrupt_path, "w", encoding="utf-8") as out:
                                for i, text, err in corrupt_items:
                                    out.write(f"Error: {err}\n\n{text}\n\n")
                            if verbose:
                                logger.warning(f"[CORRUPT] Corrupt JSON saved to {corrupt_path}")
                        if not valid_objs and try_fix:
                            minimal = "[]" if "array" in file.name or file.name.endswith("s.json") else "{}"
                            with open(file, "w", encoding="utf-8") as out:
                                out.write(minimal)
                            if verbose:
                                logger.warning(f"[FIXED] All content invalid, recreated minimal valid JSON in {file}")
                        continue
                except Exception as e:
                    corrupted.append(str(file))
                    if verbose:
                        logger.warning(f"[CORRUPT] {file}: {e}")
                    if auto_delete:
                        try:
                            if file.exists():
                                if quarantine:
                                    quarantine_dir = file.parent / "corrupt"
                                    quarantine_dir.mkdir(exist_ok=True)
                                    file.rename(quarantine_dir / file.name)
                                    if verbose:
                                        logger.warning(f"[QUARANTINED] {file} -> {quarantine_dir / file.name}")
                                else:
                                    file.unlink()
                                    if verbose:
                                        logger.warning(f"[DELETED] {file}")
                            else:
                                if verbose:
                                    logger.warning(f"[SKIP-DELETE] File already missing: {file}")
                        except Exception as del_e:
                            logger.error(f"[ERROR] Could not remove {file}: {del_e}")
    if verbose:
        logger.info(f"[SUMMARY] Corrupted files found: {corrupted}")
    return corrupted

def find_log_files(
    dirs=None,
    suffixes=(".jsonl", ".json"),
    field_filter=None,
    regex_filter=None,
    allowed_roots=None,
    dedupe=True
) -> list[Path]:
    """
    Recursively find all log files with given suffixes in dirs.
    Optionally filter by field name or regex.
    Returns a list of Path objects.
    """
    logger.debug(f"[DEBUG] Searching in dirs: {dirs} with suffixes: {suffixes}")
    if isinstance(dirs, (str, Path)):
        dirs = [dirs]
    if dirs is None:
        dirs = [LOG_DIR, CONTEXT_LIBRARY_DIR, CACHE_DIR]
    if allowed_roots is None:
        allowed_roots = [LOG_DIR, CONTEXT_LIBRARY_DIR, CACHE_DIR]
    found = []
    for d in dirs:
        try:
            d = safe_path(d, allowed_roots)
            d = Path(d)
            if not d.exists() or not d.is_dir():
                continue
            for suf in suffixes:
                if not isinstance(suf, str):
                    suf = str(suf)
                logger.debug(f"[DEBUG] Searching after isinstance in dirs: {dirs} with suffixes: {suffixes}")
                for f in d.rglob(f"*{suf}"):
                    if field_filter and field_filter not in f.name:
                        continue
                    if regex_filter and not re.search(regex_filter, str(f)):
                        continue
                    found.append(f)
        except Exception as e:
            logger.warning(f"[FIND-LOGS] Skipped {d}: {e}")
    if dedupe:
        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for f in found:
            if str(f) not in seen:
                seen.add(str(f))
                unique.append(f)
        found = unique
    return found

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
                    logger.warning(f"[CORRUPT] {path} line {line_num}: {e}")
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
    if coordinator is None:
        coordinator = ContextCoordinator()
    context_library_path = safe_path(context_library_path, [CONTEXT_LIBRARY_DIR])
    if not new_entries:
        logger.info(f"No new entries to review for {field_type}.")
        return 0, 0, 0
    logger.info(f"\n[FEEDBACK] Review new context library entries for {field_type}:")
    context_library = load_context_library(context_library_path)
    logger.debug("DEBUG: Loaded context library:", type(context_library))
    if not isinstance(context_library, dict):
        logger.error("ERROR: Context library is not a dictionary. Check your context library loading logic.")
        raise ValueError("Context library must be a dictionary. Check your context library loading logic.")
    changed = False
    accepted, edited, removed = 0, 0, 0
    # Summary preview
    new_entries_values = new_entries.values() if isinstance(new_entries, dict) else new_entries
    total_new = sum(len(v) for v in new_entries_values)
    preview = Counter(entry.get("extracted_value") for vals in new_entries_values for entry in vals)
    logger.info(f"[SUMMARY] {total_new} new entries to review. Top values:")
    for val, count in preview.most_common(5):
        logger.info(f"  {val!r}: {count} times")
    for context_key, values in new_entries.items():
        logger.info(f"\nContext: {context_key}")
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
            logger.info(f"  [{idx}] {val}")
            if enhanced:
                ml_score = ml_score_entry(val, coordinator)
                ml_field = ml_suggest_field(val, coordinator)
                logger.info(f"    [ML] Score: {ml_score:.2f} | ML Field: {ml_field}")
                if llm_api_key:
                    llm_suggestion = llm_suggest_action(
                        val, context=context_library, api_key=llm_api_key, model=llm_model, provider=llm_provider,
                        system_prompt=llm_system_prompt, extra_instructions=llm_extra_instructions
                    )
                    logger.info(f"    [LLM] Suggestion: {llm_suggestion}")
            action = "a" if fast_mode else (input("Accept (a), Edit (e), Remove (r), Skip (s)? [a]: ").strip().lower() or "a")
            if action == "a":
                accepted += 1
            elif action == "e":
                new_val = input("Edit entry (as JSON): ")
                try:
                    values[idx] = orjson.loads(new_val)
                    edited += 1
                except Exception as e:
                    logger.warning(f"Invalid JSON, skipping edit: {e}")
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
    logger.info(f"[SUMMARY] Accepted: {accepted}, Edited: {edited}, Removed: {removed}")
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

def suggest_fields_with_models(contest, nlp=None):
    """
    Suggest missing fields using spaCy NER and other ML models.
    Returns a dict of {field: suggestion or None}.
    """
    title = contest.get("title", "")
    raw = contest.get("raw", {})
    suggestions = {}

    # Load models if not provided
    if not nlp:
        try:
            nlp = ModelRegistry.get_spacy_model()
        except Exception:
            nlp = None
    if not torch_model:
        try:
            torch_model = ModelRegistry.get_torch_contest_model()
        except Exception:
            torch_model = None

    # Torch-based suggestion (if available)
    torch_suggestion = {}
    if torch_model:
        try:
            torch_suggestion = torch_model.predict(title)
        except Exception:
            torch_suggestion = {}

    # Helper for extracting with spaCy
    def extract_with_spacy(label, text):
        if nlp and text:
            doc = nlp(str(text))
            for ent in doc.ents:
                if ent.label_ == label:
                    return ent.text
        return None

    # Year
    if not contest.get("year"):
        year = torch_suggestion.get("year") \
            or extract_with_spacy("DATE", title) \
            or extract_with_spacy("DATE", raw.get("title", ""))
        if not year:
            m = re.search(r"(19|20)\d{2}", title)
            year = m.group(0) if m else None
        suggestions["year"] = year

    # State
    if not contest.get("state"):
        state = torch_suggestion.get("state") \
            or extract_with_spacy("GPE", title) \
            or extract_with_spacy("GPE", raw.get("title", ""))
        suggestions["state"] = state

    # County
    if not contest.get("county"):
        county = torch_suggestion.get("county") \
            or extract_with_spacy("LOC", title) \
            or extract_with_spacy("LOC", raw.get("title", ""))
        if not county:
            m = re.search(r"([A-Za-z ]+) County", title)
            county = m.group(1).strip() if m else None
        suggestions["county"] = county

    # Type
    if not contest.get("type_"):
        ctype = torch_suggestion.get("type_") \
            or extract_with_spacy("EVENT", title) \
            or extract_with_spacy("ORG", title)
        if not ctype:
            for t in ["General", "Primary", "Special"]:
                if t.lower() in title.lower():
                    ctype = t
                    break
        suggestions["type_"] = ctype

    return suggestions

def prompt_for_missing_fields(contest, suggestions):
    """
    Prompt user for all missing fields, showing model suggestions.
    Updates contest in-place.
    """
    print(f"\n[INTEGRITY] Contest missing fields: {contest.get('title', '')}")
    for field, suggestion in suggestions.items():
        if contest.get(field):
            continue
        prompt = f"Enter {field} (suggested: {suggestion!r}, leave blank to skip): "
        value = input(prompt).strip()
        if not value and suggestion:
            value = suggestion
        if value:
            # For year, ensure int
            if field == "year":
                try:
                    value = int(re.search(r"(19|20)\d{2}", str(value)).group(0))
                except Exception:
                    print(f"Could not parse year from input: {value}")
                    continue
            contest[field] = value

def highlight_anomalies(context_library, field_type, context_path=None, autofix=True):
    try:
        from ..Context_Integration.Integrity_check import analyze_contests, summarize_context_entities
    except ImportError:
        logger.warning("Could not import integrity_check for anomaly highlighting.")
        return
    if field_type == "contests" and "contests" in context_library:
        contests = context_library["contests"]
        results = analyze_contests(contests)
        fixed_count = 0
        nlp = None
        try:
            nlp = ModelRegistry.get_spacy_model()
        except Exception:
            pass
        if results.get("integrity_issues"):
            logger.info("[INTEGRITY] Issues detected:", results["integrity_issues"])
            for issue in results["integrity_issues"]:
                contest = issue.get("context")
                if contest:
                    # Suggest all missing fields using ML models
                    suggestions = suggest_fields_with_models(contest, nlp=nlp)
                    missing = [f for f, v in suggestions.items() if v or not contest.get(f)]
                    if missing:
                        prompt_for_missing_fields(contest, suggestions)
                        fixed_count += 1
        if results.get("flagged_suspicious"):
            logger.info("[INTEGRITY] Suspicious entries:", results["flagged_suspicious"])
        entity_summary = summarize_context_entities(contests)
        logger.info("\n[ENTITY SUMMARY]:")
        for label, count in entity_summary.items():
            logger.info(f"  {label}: {count}")
        # Save fixes if any
        if autofix and fixed_count and context_path:
            update_context_library(context_path, context_library)
            logger.info(f"[INTEGRITY] Auto-fixed {fixed_count} contests with missing fields and updated context library.")

# --- DB update logic (batch, periodic, error handling) ---
def update_database_with_context(library, db_path=None, coordinator=None, enhanced=True):
    if coordinator is None:
        coordinator = ContextCoordinator()
    if not db_path:
        db_path = CONTEXT_LIBRARY_DIR / "context_library.json"
    db_path = safe_path(db_path, [CONTEXT_LIBRARY_DIR])
    try:
        if enhanced and coordinator and hasattr(coordinator, "update_db_with_context"):
            coordinator.update_db_with_context(library, db_path)
        else:
            atomic_write_json(library, db_path)
        logger.info(f"Database updated at {db_path}")
    except Exception as e:
        logger.error(f"Failed to update DB: {e}")

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
        logger.warning("FastAPI/uvicorn not installed.")

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
    logger.info(f"[INFO] Exported correction session logs to: {export_files}")

def import_correction_session(import_file, dest_path):
    import_file = safe_path(import_file, [LOG_DIR, CONTEXT_LIBRARY_DIR])
    dest_path = safe_path(dest_path, [LOG_DIR, CONTEXT_LIBRARY_DIR])
    shutil.copy2(import_file, dest_path)
    logger.info(f"[INFO] Imported correction session from {import_file} to {dest_path}")

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
        logger.info(f"Context library not found at {path}, initializing with default structure.")
        struct = DEFAULT_STRUCTURE.copy()
        struct["schema_version"] = SCHEMA_VERSION
        update_context_library(path, struct)
        return struct
    context_lib = load_context_library(path)
    logger.debug("DEBUG: Loaded context library:", type(context_lib))
    if not isinstance(context_lib, dict):
        logger.error("ERROR: Context library is not a dictionary. Check your context library loading logic.")
        raise ValueError("Context library must be a dictionary. Check your context library loading logic.")
    # Always set schema_version if missing
    if "schema_version" not in context_lib:
        context_lib["schema_version"] = SCHEMA_VERSION
        update_context_library(path, context_lib)
    if context_lib.get("schema_version") != SCHEMA_VERSION:
        logger.warning(f"Schema version mismatch: found {context_lib.get('schema_version')}, expected {SCHEMA_VERSION}. Consider migrating.")
    return context_lib

def process_auto_mode(file_field_map, context_path, cache, batch_size=BATCH_SIZE):
    total_processed = 0
    total_skipped = 0
    total_errors = 0
    batch_field_entries = defaultdict(lambda: defaultdict(list))  # field -> context_key -> entries

    for log_file, field in file_field_map:
        try:
            # Use aggregate_successful_field_entries for dedup/group
            field_entries, dup_count, skipped_existing, n_new = aggregate_successful_field_entries(
                log_file, None, field, fast_mode=True
            )
            for context_key, entries in field_entries.items():
                for entry in entries:
                    entry_id = str(hash(orjson.dumps(entry)))
                    if entry_id in cache:
                        total_skipped += 1
                        continue
                    batch_field_entries[field][context_key].append(entry)
                    cache[entry_id] = {
                        "status": "accepted",
                        "timestamp": datetime.now().isoformat(),
                        "action": "auto-accept",
                        "user": os.environ.get("USER", "system"),
                    }
                    write_audit_log("accept", entry, user=os.environ.get("USER", "system"))
                    total_processed += 1

            # Remove processed log file if it exists
            if Path(log_file).exists():
                try:
                    os.remove(log_file)
                    logger.info(f"[AUTO] Deleted processed log file: {log_file}")
                except Exception as e:
                    logger.warning(f"[AUTO] Could not delete log file {log_file}: {e}")
        except Exception as e:
            logger.error(f"[AUTO] Error processing {log_file} for field {field}: {e}")
            total_errors += 1

        # Periodic progress log
        if total_processed % 100 == 0 and total_processed > 0:
            logger.info(f"[AUTO] Processed {total_processed} entries so far...")

        # Periodic batch update
        if total_processed % batch_size == 0 and total_processed > 0:
            for field, context_entries in batch_field_entries.items():
                update_context_with_new_entries(context_path, field, context_entries)
            batch_field_entries.clear()

    # Final flush
    if batch_field_entries:
        for field, context_entries in batch_field_entries.items():
            update_context_with_new_entries(context_path, field, context_entries)

    cache.sync()
    logger.info(f"[AUTO] Finished. Total processed: {total_processed}, skipped: {total_skipped}, errors: {total_errors}")

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
        logger.info("Cache flushed.")
        return

    cache = load_cache(expire_days=args.cache_expire_days)

    if args.export_audit_log:
        shutil.copy2(AUDIT_LOG_PATH, args.export_audit_log)
        logger.info(f"Audit log exported to {args.export_audit_log}")
        return

    if args.self_heal:
        scan_script = os.path.join(os.path.dirname(__file__), "scan_misaligned_ner.py")
        for attempt in range(1, args.max_retries + 1):
            logger.info(f"\n[SELF-HEAL] Attempt {attempt}...")
            scan_cmd = [sys.executable, scan_script, "--jsonl", "log/spacy_ner_train_data.jsonl"]
            scan_result = subprocess.run(scan_cmd, check=True, cwd=PROJECT_ROOT)
            if scan_result.returncode == 0:
                logger.info("[SELF-HEAL] Data is clean. Exiting self-heal mode.")
                break
            logger.info("[SELF-HEAL] Misalignments found. Running manual correction...")
            args.self_heal = False
            main()
            logger.info(f"[SELF-HEAL] Sleeping {args.cooldown}s before rescanning...")
            time.sleep(args.cooldown)
        else:
            logger.info("[SELF-HEAL] Max retries reached. Some misalignments may remain.")
        return

    context_path = safe_path(args.context, [CONTEXT_LIBRARY_DIR])
    context_library = ensure_context_library(context_path)
    if "metadata" not in context_library or not isinstance(context_library["metadata"], dict):
        context_library["metadata"] = {}
    context_library["metadata"]["last_accessed"] = datetime.now().isoformat()
    context_library_changed = False

    log_files = find_log_files(
        dirs=[LOG_DIR, CONTEXT_LIBRARY_DIR, CACHE_DIR],
        suffixes=(".jsonl", ".json"),
    )
    logger.info(f"Discovered {len(log_files)} log files in {[str(d) for d in [LOG_DIR, CONTEXT_LIBRARY_DIR, CACHE_DIR]]}")

    discovered_fields = discover_field_types_from_logs(log_files)
    logger.debug(f"[DEBUG] Discovered field types in logs: {discovered_fields}")
    fields = args.fields if args.fields else discovered_fields or ALL_FIELDS
    logger.debug(f"[DEBUG] Fields to process: {fields}")

    if args.fix_corrupt_json:
        check_and_fix_json_files()
        return

    file_field_map = []
    for log_file in log_files:
        try:
            entries = load_jsonl(log_file)
        except Exception as e:
            logger.warning(f"[SKIP] Could not load {log_file}: {e}")
            continue
        found_any = False
        for field in fields:
            if any(isinstance(entry, dict) and entry.get("field_type") == field for entry in entries) or field_matches_log(field, log_file.name):
                file_field_map.append((log_file, field))
                found_any = True
        if not found_any:
            for field in fields:
                file_field_map.append((log_file, field))

    logger.debug("[DEBUG] File/field processing plan:")
    for log_file, field in file_field_map:
        logger.info(f"  Will process {log_file.name} for field '{field}'")

    if not file_field_map:
        logger.warning("No log files matched any of the specified fields. Will attempt to process all log files for all fields.")
        for log_file in log_files:
            for field in fields:
                file_field_map.append((log_file, field))

    total_accepted, total_edited, total_removed = 0, 0, 0
    total_duplicates, total_existing_skipped, total_new = 0, 0, 0
    processed_logs = 0

    for log_file, field in file_field_map:
        logger.info(f"Processing {log_file} for field {field}")
        try:
            field_entries, dup_count, skipped_existing, n_new = aggregate_successful_field_entries(
                log_file, context_library, field, fast_mode=args.fast
            )
            total_duplicates += dup_count
            total_existing_skipped += skipped_existing
            total_new += n_new
            processed_logs += 1
            logger.info(f"\n[SUMMARY] {log_file.name} | Field: {field}")
            logger.info(f"  Unique new entries: {n_new}")
            logger.info(f"  Duplicates skipped: {dup_count}")
            logger.info(f"  Already in context library: {skipped_existing}")
            preview = []
            field_entries_values = field_entries.values() if isinstance(field_entries, dict) else field_entries
            for v in field_entries_values:
                preview.extend(v)
            logger.info(f"  Preview: {preview[:3]}")
            if args.dry_run:
                logger.info(f"[DRY-RUN] Would process {n_new} new entries for field {field} from {log_file}")
                continue
            if args.auto:
                logger.info(f"[AUTO] Automatically accepting all new entries for field {field} from {log_file}")
                process_auto_mode(file_field_map, context_path, cache, batch_size=BATCH_SIZE)
                context_library_changed = True
            else:
                if args.batch:
                    for context_key, values in field_entries.items():
                        logger.info(f"\nBatch review for context: {context_key}")
                        logger.info(f"  Entries: {values}")
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
                        coordinator=None,
                        context_organizer=None,
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
            if args.integrity:
                context_library = load_context_library(context_path)
                logger.debug("DEBUG: Loaded context library:", type(context_library))
                if not isinstance(context_library, dict):
                    logger.error("ERROR: Context library is not a dictionary. Check your context library loading logic.")
                    raise ValueError("Context library must be a dictionary. Check your context library loading logic.")
                highlight_anomalies(context_library, field, context_path, autofix=True)
            if args.update_db:
                context_library = load_context_library(context_path)
                logger.debug("DEBUG: Loaded context library:", type(context_library))
                if not isinstance(context_library, dict):
                    logger.error("ERROR: Context library is not a dictionary. Check your context library loading logic.")
                    raise ValueError("Context library must be a dictionary. Check your context library loading logic.")
                update_database_with_context(context_library, db_path=args.db_path, enhanced=args.enhanced, coordinator=None)
            try:
                os.remove(log_file)
                logger.info(f"Deleted processed log file: {log_file}")
            except Exception as e:
                logger.warning(f"Could not delete log file {log_file}: {e}")
        except Exception as e:
            logger.error(f"Failed to process {log_file} for field {field}: {e}")

    if context_library_changed and not args.dry_run:
        context_library = load_context_library(context_path)
        logger.debug("DEBUG: Loaded context library:", type(context_library))
        if not isinstance(context_library, dict):
            logger.error("ERROR: Context library is not a dictionary. Check your context library loading logic.")
            raise ValueError("Context library must be a dictionary. Check your context library loading logic.")
        update_context_library(context_path, context_library)
        logger.info(f"Context library updated at {context_path}")

    logger.info("\n[SUMMARY] Manual Correction Bot Run Complete.")
    logger.info(f"Log files processed: {processed_logs}")
    logger.info(f"Total unique new entries: {total_new}")
    logger.info(f"Total duplicates skipped: {total_duplicates}")
    logger.info(f"Total already in context library: {total_existing_skipped}")
    logger.info(f"Total accepted: {total_accepted}, Total edited: {total_edited}, Total removed: {total_removed}")
    if processed_logs == 0 or total_new == 0:
        logger.warning("[WARNING] No entries were processed. Check your log file naming, field configuration, or use --dry-run for debugging.")

if __name__ == "__main__":
    main()