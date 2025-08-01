"""
log_cache_cleaner_bot.py

Automated log/cache cleaner for Smart Elections pipeline.
- Scans the log and Context_Library directories for all .json/.jsonl/.html files.
- Deduplicates and compacts each file.
- Tracks file sizes and cleans files that exceed a configurable threshold (default: 200MB).
- Handles malformed files gracefully and reports errors.
- Flags files that remain too large after cleaning.
- Optionally, handles [MISALIGNED] warnings for NER data.
- Optionally, performs PostgreSQL VACUUM/ANALYZE maintenance using SQLAlchemy.
- Can be called from other scripts or run as a scheduled daemon.

Usage:
    python -m webapp.parser.bots.log_cache_cleaner_bot [--log-dir log] [--context-lib-dir .../Context_Library] [--max-size-mb 200] [--daemon] [--interval-min 60] [--db-maintenance]
Manual one-off clean:
python -m webapp.parser.bots.log_cache_cleaner_bot
Daemon mode (every 30 minutes):
python -m webapp.parser.bots.log_cache_cleaner_bot --daemon --interval-min 30
From another script:
from webapp.parser.bots.log_cache_cleaner_bot import run_log_cache_cleaner
run_log_cache_cleaner()
"""
import os
import sys
import orjson
import argparse
import time
import threading
from pathlib import Path
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from ..utils.logger_singleton import logger
# --- SQLAlchemy imports for DB maintenance ---
from ..utils.db_utils import get_engine
from .context_migration import migrate_all
from ..config import LOG_DIR, CONTEXT_LIBRARY_DIR, CACHE_DIR

DEFAULT_MAX_SIZE_MB = 1024 # Default max size for files before cleaning 250MB, 500MB, 1024MB, 2048MB
MISALIGNED_KEYWORDS = ["misaligned", "pattern-excluding"]
ALLOWED_EXTS = (".json", ".jsonl", ".html")
EMPTY_WATCH_FILE = os.path.join(CONTEXT_LIBRARY_DIR, "empty_entries_watch.jsonl")

def is_jsonl_file(fname) -> bool:
    return fname.endswith(".jsonl")

def is_json_file(fname) -> bool:
    return fname.endswith(".json")

def is_html_file(fname) -> bool:
    return fname.endswith(".html")

def safe_path(path, allowed_roots) -> str:
    path = os.path.abspath(path)
    for root in allowed_roots:
        root = os.path.abspath(root)
        if path.startswith(root):
            return path
    raise ValueError(f"Unsafe path detected: {path}")

def log_empty_entry(file_path, entry_type, key_or_index, entry) -> None:
    """Append info about an empty entry to the watch file for traceability."""
    record = {
        "file_path": file_path,
        "entry_type": entry_type,
        "key_or_index": key_or_index,
        "entry": entry,
    }
    with open(EMPTY_WATCH_FILE, "ab") as f:
        f.write(orjson.dumps(record) + b"\n")

def clean_jsonl(path, required_fields=None, backup=True) -> dict:
    """
    Enhanced cleaner for .jsonl files:
    - Deduplicates entries (by full serialization)
    - Skips malformed, null, empty, and non-dict lines (counts each)
    - Logs up to 5 examples of each problem type for diagnostics
    - Flags entries with misaligned keywords
    - Optionally checks for required fields and logs/removes entries missing them
    - Optionally backs up the original file before cleaning (only one .bak kept)
    - Handles empty files gracefully
    - Returns detailed stats and errors
    """
    import shutil
    malformed_count = 0
    null_count = 0
    empty_count = 0
    nondict_count = 0
    missing_required_count = 0
    malformed_examples = []
    null_examples = []
    empty_examples = []
    nondict_examples = []
    missing_required_examples = []
    try:
        if os.path.getsize(path) == 0:
            with open(path, "wb") as f:
                pass
            return 0, 0, 0, None
        if backup:
            bak_path = path + ".bak"
            if os.path.exists(bak_path):
                try:
                    os.remove(bak_path)
                except Exception:
                    pass
            shutil.copy2(path, bak_path)
        with open(path, "rb") as f:
            lines = [line for line in f if line.strip()]
        entries = []
        seen = set()
        misaligned = []
        for idx, line in enumerate(lines, 1):
            try:
                entry = orjson.loads(line)
            except Exception as e:
                malformed_count += 1
                if len(malformed_examples) < 5:
                    malformed_examples.append(line[:100])
                continue
            if entry is None:
                null_count += 1
                if len(null_examples) < 5:
                    null_examples.append(line[:100])
                continue
            if isinstance(entry, (dict, list)) and not entry:
                empty_count += 1
                if len(empty_examples) < 5:
                    empty_examples.append(line[:100])
                continue
            if not isinstance(entry, dict):
                nondict_count += 1
                if len(nondict_examples) < 5:
                    nondict_examples.append(str(entry)[:100])
                continue
            # Check for required fields
            if required_fields and not all(field in entry for field in required_fields):
                missing_required_count += 1
                if len(missing_required_examples) < 5:
                    missing_required_examples.append(str(entry)[:100])
                continue
            key = orjson.dumps(entry)
            if key not in seen:
                seen.add(key)
                entries.append(entry)
            if any(kw in str(entry).lower() for kw in MISALIGNED_KEYWORDS):
                misaligned.append(entry)
        with open(path, "wb") as f:
            for entry in entries:
                if not isinstance(entry, dict) or not entry:
                    logger.warning(f"Skipping non-dict entry in spacy_ner_train_data.jsonl: {entry}")
                    continue
                f.write(orjson.dumps(entry) + b"\n")
        error_parts = []
        if malformed_count:
            error_parts.append(f"Malformed: {malformed_count} (examples: {malformed_examples})")
        if null_count:
            error_parts.append(f"Null: {null_count} (examples: {null_examples})")
        if empty_count:
            error_parts.append(f"Empty: {empty_count} (examples: {empty_examples})")
        if nondict_count:
            error_parts.append(f"Non-dict: {nondict_count} (examples: {nondict_examples})")
        if missing_required_count:
            error_parts.append(f"Missing required fields: {missing_required_count} (examples: {missing_required_examples})")
        error_str = "; ".join(error_parts) if error_parts else None
        return (
            len(lines),
            len(entries),
            len(misaligned),
            error_str
        )
    except Exception as e:
        return None, None, None, str(e)

def clean_json(path, required_fields=None, backup=True) -> tuple:
    """
    Enhanced cleaner for .json files:
    - Handles empty files (overwrites with {})
    - Deduplicates dict keys or list entries
    - Skips malformed entries in lists
    - Removes null/empty dict/empty list entries
    - Handles malformed JSON gracefully
    - Optionally checks for required fields and logs/removes entries missing them
    - Optionally backs up the original file before cleaning (only one .bak kept)
    - Optionally sorts dict keys
    - Handles files that are a mix of lists and dicts
    """
    import shutil
    malformed_count = 0
    null_count = 0
    empty_count = 0
    missing_required_count = 0
    empty_keys = []
    try:
        if os.path.getsize(path) == 0:
            with open(path, "wb") as f:
                f.write(orjson.dumps({}))
            return 0, 0, 0, None
        if backup:
            bak_path = path + ".bak"
            if os.path.exists(bak_path):
                try:
                    os.remove(bak_path)
                except Exception:
                    pass
            shutil.copy2(path, bak_path)
        with open(path, "rb") as f:
            try:
                data = orjson.loads(f.read())
            except Exception as e:
                with open(path, "wb") as wf:
                    wf.write(orjson.dumps({}))
                return 0, 0, 0, f"Malformed JSON, reset to empty: {e}"
        # Handle dict
        if isinstance(data, dict):
            before = len(data)
            seen = set()
            deduped = {}
            for k, v in data.items():
                if v is None:
                    null_count += 1
                    continue
                if isinstance(v, (dict, list)) and not v:
                    empty_count += 1
                    empty_keys.append(k)
                    log_empty_entry(path, "dict", k, v)  # <-- log empty dict/list value
                    continue
                if required_fields and not all(field in v for field in required_fields if isinstance(v, dict)):
                    missing_required_count += 1
                    continue
                if k not in seen:
                    seen.add(k)
                    deduped[k] = v
            after = len(deduped)
            with open(path, "wb") as f:
                f.write(orjson.dumps(deduped, option=orjson.OPT_INDENT_2))
            if empty_keys:
                logger.info(f"[CLEAN][INFO] Removed empty entries for keys: {empty_keys} in {path}")
            if missing_required_count > 0 or malformed_count > 0:
                return before, after, 0, f"Malformed: {malformed_count}, Missing required: {missing_required_count}"
            else:
                # Just log info if only null/empty were removed
                if null_count > 0 or empty_count > 0:
                    logger.info(f"[CLEAN][INFO] Removed {null_count} null and {empty_count} empty entries from {path}")
                return before, after, 0, None
        # Handle list
        elif isinstance(data, list):
            before = len(data)
            seen = set()
            deduped = []
            empty_indices = []
            for idx, entry in enumerate(data, 1):
                try:
                    if entry is None:
                        null_count += 1
                        continue
                    if isinstance(entry, (dict, list)) and not entry:
                        empty_count += 1
                        empty_indices.append(idx-1)
                        log_empty_entry(path, "list", idx-1, entry)  # <-- log empty list/dict entry
                        continue
                    if required_fields and not all(field in entry for field in required_fields if isinstance(entry, dict)):
                        missing_required_count += 1
                        continue
                    key = orjson.dumps(entry)
                    if key not in seen:
                        seen.add(key)
                        deduped.append(entry)
                except Exception:
                    malformed_count += 1
                    continue
            after = len(deduped)
            with open(path, "wb") as f:
                f.write(orjson.dumps(deduped, option=orjson.OPT_INDENT_2))
            if empty_indices:
                logger.info(f"[CLEAN][INFO] Removed empty entries at indices: {empty_indices} in {path}")
            if missing_required_count > 0 or malformed_count > 0:
                return before, after, 0, f"Malformed: {malformed_count}, Missing required: {missing_required_count}"
            else:
                # Just log info if only null/empty were removed
                if null_count > 0 or empty_count > 0:
                    logger.info(f"[CLEAN][INFO] Removed {null_count} null and {empty_count} empty entries from {path}")
                return before, after, 0, None
        else:
            with open(path, "wb") as f:
                f.write(orjson.dumps({}))
            return 0, 0, 0, "Unknown JSON structure, reset to empty"
    except Exception as e:
        if "zero-length" in str(e) or "empty document" in str(e):
            with open(path, "wb") as f:
                f.write(orjson.dumps({}))
            return 0, 0, 0, None
        return None, None, None, str(e)
    
def clean_html(path, backup=True) -> tuple:
    """
    Robust cleaner for .html files:
    - Removes duplicate lines
    - Minifies whitespace
    - Handles empty and malformed files gracefully
    - Removes lines that are only whitespace or HTML comments
    - Optionally strips HTML tags (commented out, can be enabled)
    - Handles encoding errors
    - Optionally backs up the original file before cleaning (only one .bak kept)
    - Logs up to 5 examples of each problem type for diagnostics
    - Returns detailed stats and errors
    """
    import shutil
    malformed_count = 0
    empty_count = 0
    comment_count = 0
    malformed_examples = []
    empty_examples = []
    comment_examples = []
    try:
        if os.path.getsize(path) == 0:
            with open(path, "w", encoding="utf-8") as f:
                f.write("")
            return 0, 0, 0, None
        if backup:
            bak_path = path + ".bak"
            if os.path.exists(bak_path):
                try:
                    os.remove(bak_path)
                except Exception:
                    pass
            shutil.copy2(path, bak_path)
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        except Exception as e:
            malformed_count += 1
            if len(malformed_examples) < 5:
                malformed_examples.append(str(e))
            with open(path, "w", encoding="utf-8") as f:
                f.write("")
            return 0, 0, 0, f"Malformed HTML, reset to empty: {e}"

        seen = set()
        deduped = []
        for line in lines:
            orig_line = line
            line = line.strip()
            if not line:
                empty_count += 1
                if len(empty_examples) < 5:
                    empty_examples.append(orig_line[:100])
                continue
            # Remove HTML comments
            if line.startswith("<!--") and line.endswith("-->"):
                comment_count += 1
                if len(comment_examples) < 5:
                    comment_examples.append(line[:100])
                continue
            # Optionally, strip HTML tags (uncomment if needed)
            # import re
            # line = re.sub(r'<[^>]+>', '', line)
            if line not in seen:
                seen.add(line)
                deduped.append(line)

        # Optionally, reformat as minimal HTML if file is now empty
        if not deduped:
            with open(path, "w", encoding="utf-8") as f:
                f.write("")
            return len(lines), 0, comment_count, None

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(deduped))

        error_parts = []
        if malformed_count:
            error_parts.append(f"Malformed: {malformed_count} (examples: {malformed_examples})")
        if empty_count:
            error_parts.append(f"Empty: {empty_count} (examples: {empty_examples})")
        if comment_count:
            error_parts.append(f"Comments: {comment_count} (examples: {comment_examples})")
        error_str = "; ".join(error_parts) if error_parts else None

        return len(lines), len(deduped), comment_count, error_str
    except Exception as e:
        if "zero-length" in str(e) or "empty document" in str(e):
            with open(path, "w", encoding="utf-8") as f:
                f.write("")
            return 0, 0, 0, None
        return None, None, None, str(e)

def human_size(num_bytes) -> str:
    for unit in ['B','KB','MB','GB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f}TB"

def clean_dir(target_dir, allowed_roots, max_size_bytes, full_sweep=False) -> tuple:
    cleaned_files = 0
    total_before = 0
    total_after = 0
    errors = []
    flagged_large = []
    misaligned_summary = []
    # Recursively find all files with allowed extensions
    for path in Path(target_dir).rglob("*"):
        if not path.is_file() or not path.suffix in ALLOWED_EXTS:
            continue
        fname = str(path)
        try:
            safe_path(fname, allowed_roots)
        except Exception as e:
            errors.append((fname, f"Unsafe path: {e}"))
            continue
        size = os.path.getsize(fname)
        needs_clean = size > max_size_bytes
        # Only clean if too big, or if full_sweep is True
        if needs_clean or full_sweep:
            if is_jsonl_file(fname):
                before, after, misaligned, err = clean_jsonl(fname)
            elif is_json_file(fname):
                before, after, misaligned, err = clean_json(fname)
            elif is_html_file(fname):
                before, after, misaligned, err = clean_html(fname)
            else:
                continue
            if err:
                logger.error(f"[CLEAN][ERROR] Failed to clean {fname}: {err}")
                errors.append((fname, err))
                continue
            logger.info(f"[CLEAN] Cleaned {fname}. Original: {before}, After: {after}{' | MISALIGNED: '+str(misaligned) if misaligned else ''}")
            cleaned_files += 1
            total_before += before or 0
            total_after += after or 0
            if misaligned:
                misaligned_summary.append((fname, misaligned))
            # Check if still too large
            new_size = os.path.getsize(fname)
            if new_size > max_size_bytes:
                flagged_large.append((fname, human_size(new_size)))
            pass
    return cleaned_files, total_before, total_after, flagged_large, misaligned_summary, errors

def run_db_maintenance(engine=None, session=None) -> dict:
    """
    Perform PostgreSQL VACUUM and ANALYZE on all tables using SQLAlchemy.
    - Handles connection errors, permission errors, and logs all actions.
    - Skips system tables and warns if no tables found.
    - Optionally supports ANALYZE only if VACUUM is not allowed.
    - Returns a summary of actions and errors.
    """
    logger.info("[DB] Starting PostgreSQL VACUUM/ANALYZE maintenance...")
    summary = {"vacuumed": [], "skipped": [], "errors": []}
    try:
        if engine is None:
            engine = get_engine()
        with engine.connect() as conn:
            # Get all user tables (skip system tables)
            tables = conn.execute(
                text("SELECT tablename FROM pg_tables WHERE schemaname = 'public';")
            ).fetchall()
            if not tables:
                logger.warning("[DB][WARNING] No user tables found in schema 'public'.")
                return summary
            for (table,) in tables:
                if table.startswith("pg_") or table.startswith("sql_"):
                    summary["skipped"].append(table)
                    continue
                logger.info(f"[DB] VACUUM (ANALYZE) {table} ...")
                try:
                    conn.execute(text(f"VACUUM (ANALYZE) {table};"))
                    summary["vacuumed"].append(table)
                except SQLAlchemyError as e:
                    logger.error(f"[DB][ERROR] Could not vacuum {table}: {e}")
                    # Try ANALYZE only if VACUUM fails
                    try:
                        conn.execute(text(f"ANALYZE {table};"))
                        logger.info(f"[DB][INFO] ANALYZE succeeded for {table} after VACUUM failed.")
                        summary["vacuumed"].append(f"{table} (ANALYZE only)")
                    except Exception as e2:
                        logger.error(f"[DB][ERROR] Could not analyze {table}: {e2}")
                        summary["errors"].append((table, str(e2)))
            logger.info(f"[DB] VACUUM/ANALYZE complete. Tables vacuumed: {len(summary['vacuumed'])}, skipped: {len(summary['skipped'])}, errors: {len(summary['errors'])}")
    except Exception as e:
        logger.error(f"[DB][ERROR] Maintenance failed: {e}")
        summary["errors"].append(("__connection__", str(e)))
    return summary

def run_log_cache_cleaner(log_dir=LOG_DIR, context_lib_dir=CONTEXT_LIBRARY_DIR, cache_dir=CACHE_DIR, max_size_mb=DEFAULT_MAX_SIZE_MB, db_maintenance=False, full_sweep=False) -> list:
    max_size_bytes = int(max_size_mb * 1024 * 1024)
    allowed_roots = [log_dir, context_lib_dir, cache_dir]
    logger.info(f"[CLEAN] Cleaning log dir: {log_dir}")
    cleaned1, before1, after1, flagged1, misaligned1, errors1 = clean_dir(log_dir, allowed_roots, max_size_bytes, full_sweep=full_sweep)
    logger.info(f"[CLEAN] Cleaning context library dir: {context_lib_dir}")
    cleaned2, before2, after2, flagged2, misaligned2, errors2 = clean_dir(context_lib_dir, allowed_roots, max_size_bytes, full_sweep=full_sweep)
    logger.info(f"[CLEAN] Cleaning cache dir: {cache_dir}")
    cleaned3, before3, after3, flagged3, misaligned3, errors3 = clean_dir(cache_dir, allowed_roots, max_size_bytes, full_sweep=full_sweep)
    cleaned_files = cleaned1 + cleaned2 + cleaned3
    total_before = before1 + before2 + before3
    total_after = after1 + after2 + after3
    flagged_large = flagged1 + flagged2 + flagged3
    misaligned_summary = misaligned1 + misaligned2 + misaligned3
    errors = errors1 + errors2 + errors3
    logger.info(f"[CLEAN] Finished cleaning {cleaned_files} files. Total entries: {total_before} -> {total_after}")
    if flagged_large:
        logger.warning("[CLEAN][WARNING] The following files are still too large after cleaning:")
        for fname, sz in flagged_large:
            logger.info(f"  {fname}: {sz}")
    if misaligned_summary:
        logger.warning("[MISALIGNED] Consider cleaning or pattern-excluding these from your training data:")
        for fname, count in misaligned_summary:
            logger.info(f"  {fname}: {count} entries flagged as misaligned")
    if errors:
        logger.error("[CLEAN][ERROR] Some files could not be cleaned:")
        for fname, err in errors:
            logger.info(f"  {fname}: {err}")
    if db_maintenance:
        run_db_maintenance()
    migrate_all()
    logger.info("[CLEAN] Context/log migration to PostgreSQL complete.")
    return errors

def schedule_log_cache_cleaner(interval_min=60, db_maintenance=False, **kwargs) -> threading.Thread:
    def loop():
        while True:
            run_log_cache_cleaner(db_maintenance=db_maintenance, **kwargs)
            time.sleep(interval_min * 60)
    t = threading.Thread(target=loop, daemon=True)
    t.start()
    logger.info(f"[CLEAN] Log cleaner scheduled every {interval_min} minutes.")
    return t

def main() -> None:
    parser = argparse.ArgumentParser(description="Automated log/cache cleaner for Smart Elections pipeline.")
    parser.add_argument("--log-dir", type=str, default=LOG_DIR, help="Directory containing log/cache files")
    parser.add_argument("--context-lib-dir", type=str, default=CONTEXT_LIBRARY_DIR, help="Context_Library directory")
    parser.add_argument("--cache-dir", type=str, default=CACHE_DIR, help="Cache directory")
    parser.add_argument("--max-size-mb", type=float, default=DEFAULT_MAX_SIZE_MB, help="Max file size in MB before cleaning is triggered")
    parser.add_argument("--daemon", action="store_true", help="Run as a background daemon (periodic cleaning)")
    parser.add_argument("--interval-min", type=int, default=60, help="Interval in minutes for daemon mode")
    parser.add_argument("--db-maintenance", action="store_true", help="Perform PostgreSQL VACUUM/ANALYZE maintenance after cleaning")
    parser.add_argument("--full-sweep", action="store_true", help="Clean all files, not just those exceeding size threshold")
    args = parser.parse_args()
    if args.daemon:
        schedule_log_cache_cleaner(
            interval_min=args.interval_min,
            log_dir=args.log_dir,
            context_lib_dir=args.context_lib_dir,
            cache_dir=args.cache_dir,
            max_size_mb=args.max_size_mb,
            db_maintenance=args.db_maintenance,
            full_sweep=args.full_sweep
        )
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            logger.info("[CLEAN] Daemon stopped.")
    else:
        run_log_cache_cleaner(
            log_dir=args.log_dir,
            context_lib_dir=args.context_lib_dir,
            cache_dir=args.cache_dir,
            max_size_mb=args.max_size_mb,
            db_maintenance=args.db_maintenance,
            full_sweep=args.full_sweep
        )

if __name__ == "__main__":
    main()
