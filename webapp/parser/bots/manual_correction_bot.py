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
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import threading
import subprocess
import sys
import time

# --- Unified logger import ---
from ..utils.shared_logger import logger
from ..utils.shared_logic import (
    save_context_library,
    update_context_library,
    SCHEMA_VERSION,
    DEFAULT_STRUCTURE,
    load_context_library,
)
# --- Config ---
# --- Directory and file constants ---
from ..config import PROJECT_ROOT, BASE_DIR, CONTEXT_LIBRARY_PATH

LOG_DIR = Path(PROJECT_ROOT) / "log"
CONTEXT_LIBRARY_DIR = Path(BASE_DIR) / "parser" / "Context_Integration" / "Context_Library"

# Log and data file paths
FIELD_LOG_SUFFIX = "_selection_log.jsonl"
SEGMENT_FEEDBACK_LOG = LOG_DIR / "segment_feedback_log.jsonl"
PATTERN_KB_FILE = LOG_DIR / "dom_pattern_kb.jsonl"
DOWNLOAD_LINKS_LOG = LOG_DIR / "download_links_log.jsonl"
ANOMALY_LOG = LOG_DIR / "anomaly_log.jsonl"
EXPORT_DIR = LOG_DIR / "correction_exports"
ALL_FIELDS = [
    "buttons", "panels", "tables", "contests", "districts", "states", "election_types", "years", "party", "candidate"
]
SUCCESS_RESULTS = {"pass", "fuzzy_pass", "manual_correction", "user_corrected"}

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

# --- Log file discovery ---
def find_log_files(log_dir=LOG_DIR):
    log_dir = safe_path(log_dir, [LOG_DIR])
    if not log_dir.exists():
        logger.warning(f"Log directory not found: {log_dir}")
        return []
    return list(log_dir.glob(f"*{FIELD_LOG_SUFFIX}"))

# --- JSONL utilities ---
def load_jsonl(path):
    path = safe_path(path, [LOG_DIR, CONTEXT_LIBRARY_DIR])
    if not path.exists():
        logger.warning(f"Log file not found: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [orjson.loads(line) for line in f if line.strip()]

def save_jsonl(path, entries):
    path = safe_path(path, [LOG_DIR, CONTEXT_LIBRARY_DIR])
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(orjson.dumps(entry, ensure_ascii=False) + "\n")

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
def feedback_loop(new_entries, field_type, context_library_path, enhanced=True, coordinator=None, llm_api_key=None, llm_provider="openai", llm_model="gpt-4-turbo", llm_system_prompt=None, llm_extra_instructions=None, fast_mode=False):
    context_library_path = safe_path(context_library_path, [CONTEXT_LIBRARY_DIR])
    if not new_entries:
        logger.info(f"No new entries to review for {field_type}.")
        return 0, 0, 0
    print(f"\n[FEEDBACK] Review new context library entries for {field_type}:")
    context_library = load_context_library(context_library_path)
    changed = False
    accepted, edited, removed = 0, 0, 0
    # Summary preview
    total_new = sum(len(v) for v in new_entries.values())
    preview = Counter(entry.get("extracted_value") for vals in new_entries.values() for entry in vals)
    print(f"[SUMMARY] {total_new} new entries to review. Top values:")
    for val, count in preview.most_common(5):
        print(f"  {val!r}: {count} times")
    for context_key, values in new_entries.items():
        print(f"\nContext: {context_key}")
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
            print(f"  [{idx}] {val}")
            if enhanced:
                ml_score = ml_score_entry(val, coordinator)
                ml_field = ml_suggest_field(val, coordinator)
                print(f"    [ML] Score: {ml_score:.2f} | ML Field: {ml_field}")
                if llm_api_key:
                    llm_suggestion = llm_suggest_action(
                        val, context=context_library, api_key=llm_api_key, model=llm_model, provider=llm_provider,
                        system_prompt=llm_system_prompt, extra_instructions=llm_extra_instructions
                    )
                    print(f"    [LLM] Suggestion: {llm_suggestion}")
            action = "a" if fast_mode else (input("Accept (a), Edit (e), Remove (r), Skip (s)? [a]: ").strip().lower() or "a")
            if action == "a":
                accepted += 1
            elif action == "e":
                new_val = input("Edit entry (as JSON): ")
                try:
                    values[idx] = orjson.loads(new_val)
                    edited += 1
                except Exception as e:
                    print(f"Invalid JSON, skipping edit: {e}")
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
    print(f"[SUMMARY] Accepted: {accepted}, Edited: {edited}, Removed: {removed}")
    return accepted, edited, removed

# --- Log file cleanup ---
def trim_log_file(path: Path):
    """Remove duplicate entries from a log file, keeping only the first occurrence."""
    entries = load_jsonl(path)
    deduped, _ = deduplicate_entries(entries)
    save_jsonl(path, deduped)

# --- Context library update logic ---
def update_context_with_new_entries(context_path, field_type, field_entries):
    context_path = safe_path(context_path, [CONTEXT_LIBRARY_DIR])
    def updater(library):
        # Ensure the field_type exists and is a dict (grouped by context_key)
        if field_type not in library or not isinstance(library[field_type], dict):
            library[field_type] = {}
        for context_key, entries in field_entries.items():
            if context_key not in library[field_type]:
                library[field_type][context_key] = []
            for entry in entries:
                if entry not in library[field_type][context_key]:
                    library[field_type][context_key].append(entry)
    update_context_library(context_path, updater)

# --- Integrity check integration ---
def highlight_anomalies(context_library, field_type):
    try:
        from ..Context_Integration.Integrity_check import analyze_contest_titles, summarize_context_entities
    except ImportError:
        logger.warning("Could not import integrity_check for anomaly highlighting.")
        return
    if field_type == "contests" and "contests" in context_library:
        contests = context_library["contests"]
        results = analyze_contest_titles(contests)
        if results.get("integrity_issues"):
            print("[INTEGRITY] Issues detected:", results["integrity_issues"])
        if results.get("flagged_suspicious"):
            print("[INTEGRITY] Suspicious entries:", results["flagged_suspicious"])
        entity_summary = summarize_context_entities(contests)
        print("\n[ENTITY SUMMARY]:")
        for label, count in entity_summary.items():
            print(f"  {label}: {count}")

# --- DB update logic (optional) ---
def update_database_with_context(library, db_path=None, enhanced=True, coordinator=None):
    if not db_path:
        db_path = CONTEXT_LIBRARY_DIR / "context_library.json"
    db_path = safe_path(db_path, [CONTEXT_LIBRARY_DIR])
    try:
        if enhanced and coordinator and hasattr(coordinator, "update_db_with_context"):
            coordinator.update_db_with_context(library, db_path)
        else:
            with open(db_path, "w", encoding="utf-8") as f:
                orjson.dumps(library, f, indent=2, ensure_ascii=False)
        logger.info(f"Database updated at {db_path}")
    except Exception as e:
        logger.error(f"Failed to update DB: {e}")

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
    print(f"[INFO] Exported correction session logs to: {export_files}")

def import_correction_session(import_file, dest_path):
    import_file = safe_path(import_file, [LOG_DIR, CONTEXT_LIBRARY_DIR])
    dest_path = safe_path(dest_path, [LOG_DIR, CONTEXT_LIBRARY_DIR])
    shutil.copy2(import_file, dest_path)
    print(f"[INFO] Imported correction session from {import_file} to {dest_path}")

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
        save_context_library(struct, path)
        return struct
    context_lib = load_context_library(path)
    # Always set schema_version if missing
    if "schema_version" not in context_lib:
        context_lib["schema_version"] = SCHEMA_VERSION
        save_context_library(context_lib, path)
    if context_lib.get("schema_version") != SCHEMA_VERSION:
        logger.warning(f"Schema version mismatch: found {context_lib.get('schema_version')}, expected {SCHEMA_VERSION}. Consider migrating.")
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
                logger.warning(f"Skipping misaligned entity in: {text}")
                continue
            valid_data.append((text, annots))
        except Exception as e:
            logger.warning(f"Error validating entity alignment: {e}")
    return valid_data

def summarize_misaligned_entities(log_path=None, top_n=10):
    from pathlib import Path
    from ..config import LOG_DIR
    if log_path is None:
        log_path = Path(LOG_DIR) / "spacy_ner_misaligned.jsonl"
    if not log_path.exists():
        print("[MISALIGNED] No misaligned NER examples found.")
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
        print("[MISALIGNED] No misaligned NER examples found.")
        return
    print(f"\n[MISALIGNED] Top {top_n} most frequent misaligned NER texts:")
    for text, count in counter.most_common(top_n):
        print(f"  {repr(text)}: {count} times")
    print("[MISALIGNED] Consider cleaning or pattern-excluding these from your training data.")

# --- Main CLI logic ---
def main():
    parser = argparse.ArgumentParser(description="Deep ML/LLM-enhanced batch review and correction bot for all context fields.")
    parser.add_argument("--context", type=str, default=str(CONTEXT_LIBRARY_PATH), help="Path to context_library.json")
    parser.add_argument("--log-dir", type=str, default=str(LOG_DIR), help="Directory containing *_selection_log.jsonl files")
    parser.add_argument("--fields", type=str, nargs="*", default=ALL_FIELDS, help="Fields to process (default: all)")
    parser.add_argument("--auto", action="store_true", help="Automatically accept all new entries (no prompt)")
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
    args = parser.parse_args()

    if args.self_heal:
        scan_script = os.path.join(os.path.dirname(__file__), "scan_misaligned_ner.py")
        for attempt in range(1, args.max_retries + 1):
            print(f"\n[SELF-HEAL] Attempt {attempt}...")
            scan_cmd = [sys.executable, scan_script, "--jsonl", "log/spacy_ner_train_data.jsonl"]
            scan_result = subprocess.run(scan_cmd, check=True, cwd=PROJECT_ROOT)
            if scan_result.returncode == 0:
                print("[SELF-HEAL] Data is clean. Exiting self-heal mode.")
                break
            print("[SELF-HEAL] Misalignments found. Running manual correction...")
            # Run the normal correction logic (call main() recursively, but without --self-heal)
            args.self_heal = False
            main()
            print(f"[SELF-HEAL] Sleeping {args.cooldown}s before rescanning...")
            time.sleep(args.cooldown)
        else:
            print("[SELF-HEAL] Max retries reached. Some misalignments may remain.")
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
    log_dir = safe_path(args.log_dir, [LOG_DIR])
    fields = args.fields
    log_files = find_log_files(log_dir)
    logger.info(f"Discovered {len(log_files)} log files in {log_dir}")

    # Optionally: connect to coordinator/context_organizer if enhanced
    coordinator = None
    context_organizer = None
    if args.enhanced:
        try:
            import importlib
            coordinator_mod = importlib.import_module("webapp.parser.Context_Integration.context_coordinator")
            coordinator = getattr(coordinator_mod, "ContextCoordinator", None)
            organizer_mod = importlib.import_module("webapp.parser.Context_Integration.context_organizer")
            context_organizer = getattr(organizer_mod, "context_organizer", None)
        except Exception as e:
            logger.warning(f"Could not import coordinator/context_organizer: {e}")

    total_accepted, total_edited, total_removed = 0, 0, 0
    total_duplicates, total_existing_skipped, total_new = 0, 0, 0
    processed_logs = 0
    for log_file in log_files:
        # Infer field type from filename
        for field in fields:
            if field in log_file.name:
                logger.info(f"Processing {log_file} for field {field}")
                # Deduplicate and skip existing
                field_entries, dup_count, skipped_existing, n_new = aggregate_successful_field_entries(
                    log_file, context_library, field, fast_mode=args.fast
                )
                total_duplicates += dup_count
                total_existing_skipped += skipped_existing
                total_new += n_new
                processed_logs += 1
                # Print summary before review
                print(f"\n[SUMMARY] {log_file.name} | Field: {field}")
                print(f"  Unique new entries: {n_new}")
                print(f"  Duplicates skipped: {dup_count}")
                print(f"  Already in context library: {skipped_existing}")
                # Preview top 3 entries
                preview = []
                for v in field_entries.values():
                    preview.extend(v)
                print(f"  Preview: {preview[:3]}")
                if args.auto or args.fast:
                    # Auto-accept all new entries
                    update_context_with_new_entries(context_path, field, field_entries)
                    logger.info(f"Auto-accepted new entries for {field}.")
                    total_accepted += sum(len(v) for v in field_entries.values())
                    context_library_changed = True
                else:
                    # Feedback loop returns accepted, edited, removed counts
                    # Batch review support
                    if args.batch:
                        for context_key, values in field_entries.items():
                            print(f"\nBatch review for context: {context_key}")
                            print(f"  Entries: {values}")
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
                        feedback_loop(
                            field_entries, field, context_path,
                            enhanced=args.enhanced,
                            coordinator=coordinator,
                            llm_api_key=args.llm_api_key,
                            llm_provider=args.llm_provider,
                            llm_model=args.llm_model,
                            llm_system_prompt=args.llm_system_prompt,
                            llm_extra_instructions=args.llm_extra_instructions
                        )
                        # Assume all accepted for summary (could be improved to track edits/removes)
                        total_accepted += sum(len(v) for v in field_entries.values())
                        context_library_changed = True
                # Optionally run integrity check
                if args.integrity:
                    context_library = load_context_library(context_path)
                    highlight_anomalies(context_library, field)
                # Optionally update DB
                if args.update_db:
                    context_library = load_context_library(context_path)
                    update_database_with_context(context_library, db_path=args.db_path, enhanced=args.enhanced, coordinator=coordinator)
                # Clean up log file after processing
                try:
                    os.remove(log_file)
                    logger.info(f"Deleted processed log file: {log_file}")
                except Exception as e:
                    logger.warning(f"Could not delete log file {log_file}: {e}")
                break

    # Write context library only if changed
    if context_library_changed:
        context_library = load_context_library(context_path)
        save_context_library(context_library, context_path)
        logger.info(f"Context library updated at {context_path}")

    print("\n[SUMMARY] Manual Correction Bot Run Complete.")
    print(f"Log files processed: {processed_logs}")
    print(f"Total unique new entries: {total_new}")
    print(f"Total duplicates skipped: {total_duplicates}")
    print(f"Total already in context library: {total_existing_skipped}")
    print(f"Total accepted: {total_accepted}, Total edited: {total_edited}, Total removed: {total_removed}")
    print("If you see repeated model save failures, close any file explorers or editors viewing the model directory.")
    print("If you see spaCy lexeme normalization warnings, you can ignore them for English. To suppress, install spacy-lookups-data and load the table if needed.")
    print("If you see spaCy entity alignment warnings, consider cleaning your training data or using the provided validation function.")

if __name__ == "__main__":
    main()
    summarize_misaligned_entities()