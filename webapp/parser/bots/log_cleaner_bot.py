"""
log_cleaner_bot.py

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
    python -m webapp.parser.bots.log_cleaner_bot [--log-dir log] [--context-lib-dir .../Context_Library] [--max-size-mb 200] [--daemon] [--interval-min 60] [--db-maintenance]
Manual one-off clean:
python -m webapp.parser.bots.log_cleaner_bot
Daemon mode (every 30 minutes):
python -m webapp.parser.bots.log_cleaner_bot --daemon --interval-min 30
From another script:
from webapp.parser.bots.log_cleaner_bot import run_log_cleaner
run_log_cleaner()
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

# --- SQLAlchemy imports for DB maintenance ---
from ..utils.db_utils import get_engine, get_session
from ..utils.context_migration import migrate_all

DEFAULT_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "log")
DEFAULT_CONTEXT_LIB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "webapp", "parser", "Context_Integration", "Context_Library")
DEFAULT_MAX_SIZE_MB = 10
MISALIGNED_KEYWORDS = ["misaligned", "pattern-excluding"]
ALLOWED_EXTS = (".json", ".jsonl", ".html")


def is_jsonl_file(fname):
    return fname.endswith(".jsonl")

def is_json_file(fname):
    return fname.endswith(".json")

def is_html_file(fname):
    return fname.endswith(".html")

def safe_path(path, allowed_roots):
    path = os.path.abspath(path)
    for root in allowed_roots:
        root = os.path.abspath(root)
        if path.startswith(root):
            return path
    raise ValueError(f"Unsafe path detected: {path}")

def clean_jsonl(path):
    try:
        with open(path, "rb") as f:
            lines = [line for line in f if line.strip()]
        entries = []
        seen = set()
        misaligned = []
        for line in lines:
            try:
                entry = orjson.loads(line)
                key = orjson.dumps(entry)
                if key not in seen:
                    seen.add(key)
                    entries.append(entry)
                # Flag misaligned entries
                if any(kw in str(entry).lower() for kw in MISALIGNED_KEYWORDS):
                    misaligned.append(entry)
            except Exception:
                continue  # skip malformed lines
        with open(path, "wb") as f:
            for entry in entries:
                f.write(orjson.dumps(entry) + b"\n")
        return len(lines), len(entries), len(misaligned), None
    except Exception as e:
        return None, None, None, str(e)

def clean_json(path):
    try:
        with open(path, "rb") as f:
            data = orjson.loads(f.read())
        if isinstance(data, dict):
            before = len(data)
            seen = set()
            deduped = {}
            for k, v in data.items():
                if k not in seen:
                    seen.add(k)
                    deduped[k] = v
            after = len(deduped)
            with open(path, "wb") as f:
                f.write(orjson.dumps(deduped, option=orjson.OPT_INDENT_2))
            return before, after, 0, None
        elif isinstance(data, list):
            before = len(data)
            seen = set()
            deduped = []
            for entry in data:
                key = orjson.dumps(entry)
                if key not in seen:
                    seen.add(key)
                    deduped.append(entry)
            after = len(deduped)
            with open(path, "wb") as f:
                f.write(orjson.dumps(deduped, option=orjson.OPT_INDENT_2))
            return before, after, 0, None
        else:
            return 0, 0, 0, "Unknown JSON structure"
    except Exception as e:
        return None, None, None, str(e)

def clean_html(path):
    try:
        # Remove duplicate lines, minify whitespace, keep only unique lines
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [line.strip() for line in f if line.strip()]
        seen = set()
        deduped = []
        for line in lines:
            if line not in seen:
                seen.add(line)
                deduped.append(line)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(deduped))
        return len(lines), len(deduped), 0, None
    except Exception as e:
        return None, None, None, str(e)

def human_size(num_bytes):
    for unit in ['B','KB','MB','GB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f}TB"

def clean_dir(target_dir, allowed_roots, max_size_bytes):
    cleaned_files = 0
    total_before = 0
    total_after = 0
    errors = []
    flagged_large = []
    misaligned_summary = []
    for root, dirs, files in os.walk(target_dir):
        for fname in files:
            if not fname.endswith(ALLOWED_EXTS):
                continue
            path = os.path.join(root, fname)
            try:
                safe_path(path, allowed_roots)
            except Exception as e:
                errors.append((fname, f"Unsafe path: {e}"))
                continue
            size = os.path.getsize(path)
            needs_clean = size > max_size_bytes
            # Always clean if too big, else only clean if user wants full sweep
            if needs_clean or True:
                if is_jsonl_file(fname):
                    before, after, misaligned, err = clean_jsonl(path)
                elif is_json_file(fname):
                    before, after, misaligned, err = clean_json(path)
                elif is_html_file(fname):
                    before, after, misaligned, err = clean_html(path)
                else:
                    continue
                if err:
                    print(f"[CLEAN][ERROR] Failed to clean {fname}: {err}")
                    errors.append((fname, err))
                    continue
                print(f"[CLEAN] Cleaned {fname}. Original: {before}, After: {after}{' | MISALIGNED: '+str(misaligned) if misaligned else ''}")
                cleaned_files += 1
                total_before += before or 0
                total_after += after or 0
                if misaligned:
                    misaligned_summary.append((fname, misaligned))
                # Check if still too large
                new_size = os.path.getsize(path)
                if new_size > max_size_bytes:
                    flagged_large.append((fname, human_size(new_size)))
    return cleaned_files, total_before, total_after, flagged_large, misaligned_summary, errors

def run_db_maintenance(engine=None, session=None):
    """
    Perform PostgreSQL VACUUM and ANALYZE on all tables using SQLAlchemy.
    """
    print("[DB] Starting PostgreSQL VACUUM/ANALYZE maintenance...")
    try:
        if engine is None:
            engine = get_engine()
        with engine.connect() as conn:
            # Get all table names
            tables = conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname = 'public';")).fetchall()
            for (table,) in tables:
                print(f"[DB] VACUUM (ANALYZE) {table} ...")
                try:
                    conn.execute(text(f"VACUUM (ANALYZE) {table};"))
                except SQLAlchemyError as e:
                    print(f"[DB][ERROR] Could not vacuum {table}: {e}")
        print("[DB] VACUUM/ANALYZE complete.")
    except Exception as e:
        print(f"[DB][ERROR] Maintenance failed: {e}")

def run_log_cleaner(log_dir=DEFAULT_LOG_DIR, context_lib_dir=DEFAULT_CONTEXT_LIB_DIR, max_size_mb=DEFAULT_MAX_SIZE_MB, db_maintenance=False):
    max_size_bytes = int(max_size_mb * 1024 * 1024)
    allowed_roots = [log_dir, context_lib_dir]
    print(f"[CLEAN] Cleaning log dir: {log_dir}")
    cleaned1, before1, after1, flagged1, misaligned1, errors1 = clean_dir(log_dir, allowed_roots, max_size_bytes)
    print(f"[CLEAN] Cleaning context library dir: {context_lib_dir}")
    cleaned2, before2, after2, flagged2, misaligned2, errors2 = clean_dir(context_lib_dir, allowed_roots, max_size_bytes)
    cleaned_files = cleaned1 + cleaned2
    total_before = before1 + before2
    total_after = after1 + after2
    flagged_large = flagged1 + flagged2
    misaligned_summary = misaligned1 + misaligned2
    errors = errors1 + errors2
    print(f"[CLEAN] Finished cleaning {cleaned_files} files. Total entries: {total_before} -> {total_after}")
    if flagged_large:
        print("[CLEAN][WARNING] The following files are still too large after cleaning:")
        for fname, sz in flagged_large:
            print(f"  {fname}: {sz}")
    if misaligned_summary:
        print("[MISALIGNED] Consider cleaning or pattern-excluding these from your training data:")
        for fname, count in misaligned_summary:
            print(f"  {fname}: {count} entries flagged as misaligned")
    if errors:
        print("[CLEAN][ERROR] Some files could not be cleaned:")
        for fname, err in errors:
            print(f"  {fname}: {err}")
    if db_maintenance:
        run_db_maintenance()
    migrate_all()
    print("[CLEAN] Context/log migration to PostgreSQL complete.")

def schedule_log_cleaner(interval_min=60, db_maintenance=False, **kwargs):
    def loop():
        while True:
            run_log_cleaner(db_maintenance=db_maintenance, **kwargs)
            time.sleep(interval_min * 60)
    t = threading.Thread(target=loop, daemon=True)
    t.start()
    print(f"[CLEAN] Log cleaner scheduled every {interval_min} minutes.")
    return t

def main():
    parser = argparse.ArgumentParser(description="Automated log/cache cleaner for Smart Elections pipeline.")
    parser.add_argument("--log-dir", type=str, default=DEFAULT_LOG_DIR, help="Directory containing log/cache files")
    parser.add_argument("--context-lib-dir", type=str, default=DEFAULT_CONTEXT_LIB_DIR, help="Context_Library directory")
    parser.add_argument("--max-size-mb", type=float, default=DEFAULT_MAX_SIZE_MB, help="Max file size in MB before cleaning is triggered")
    parser.add_argument("--daemon", action="store_true", help="Run as a background daemon (periodic cleaning)")
    parser.add_argument("--interval-min", type=int, default=60, help="Interval in minutes for daemon mode")
    parser.add_argument("--db-maintenance", action="store_true", help="Perform PostgreSQL VACUUM/ANALYZE maintenance after cleaning")
    args = parser.parse_args()
    if args.daemon:
        schedule_log_cleaner(interval_min=args.interval_min, log_dir=args.log_dir, context_lib_dir=args.context_lib_dir, max_size_mb=args.max_size_mb, db_maintenance=args.db_maintenance)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            print("[CLEAN] Daemon stopped.")
    else:
        run_log_cleaner(log_dir=args.log_dir, context_lib_dir=args.context_lib_dir, max_size_mb=args.max_size_mb, db_maintenance=args.db_maintenance)

if __name__ == "__main__":
    main()
