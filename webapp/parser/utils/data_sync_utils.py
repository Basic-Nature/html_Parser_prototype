"""
Data Sync Utility for Smart Elections Parser

- Safely import/export .jsonl/.json and .db (SQLite) files between PostgreSQL and local folders.
- Handles both log/ and Context_Integration/Context_Library/ directories.
- Uses config.py and .env for all paths and DB connections.
- Prevents path injection and only processes files in allowed directories.

USAGE EXAMPLES:
    python data_sync_utils.py --import-json      # Import all .jsonl/.json to PostgreSQL
    python data_sync_utils.py --export-json      # Export all PostgreSQL tables to .jsonl
    python data_sync_utils.py --import-sqlite    # Import all .db files to PostgreSQL
    python data_sync_utils.py --export-sqlite    # Export all .db tables to .jsonl

REQUIRES:
    - pandas
    - sqlalchemy
    - psycopg2-binary
    - python-dotenv

CONFIGURATION:
    - All paths and DB URLs are loaded from config.py and .env
    - Only files in log/ and Context_Integration/Context_Library/ are processed
"""
import os
import glob
import pandas as pd
from sqlalchemy import create_engine, inspect
from dotenv import load_dotenv
from pathlib import Path
import sys
from webapp.parser.utils.shared_logger import log_info, log_warning, log_error
# --- Import config robustly ---
try:
    from . import config
except ImportError:
    # Allow running as a script from project root
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    import webapp.parser.config as config

# Load environment variables
load_dotenv()

# --- Safe path handling ---
def is_safe_path(path, allowed_roots):
    path = os.path.abspath(path)
    for root in allowed_roots:
        root = os.path.abspath(root)
        if path.startswith(root):
            return True
    return False

# --- Config paths ---
LOG_DIR = os.path.abspath(os.path.join(config.PROJECT_ROOT, "log"))
CONTEXT_LIB_DIR = os.path.abspath(os.path.join(config.BASE_DIR, "parser", "Context_Integration", "Context_Library"))
ALLOWED_ROOTS = [LOG_DIR, CONTEXT_LIB_DIR]

PG_URL = os.getenv("POSTGRES_URL", config.POSTGRES_URL)
engine = create_engine(PG_URL)

def import_json_to_postgres():
    """Import all .jsonl/.json files from allowed dirs to PostgreSQL."""
    for folder in ALLOWED_ROOTS:
        for ext in ("*.jsonl", "*.json"):
            for path in glob.glob(os.path.join(folder, ext)):
                if not is_safe_path(path, ALLOWED_ROOTS):
                    log_warning(f"[SKIP] Unsafe path: {path}")
                    continue
                table_name = os.path.splitext(os.path.basename(path))[0]
                log_info(f"Importing {path} to table {table_name}...")
                try:
                    if path.endswith(".jsonl"):
                        df = pd.read_json(path, lines=True)
                    else:
                        df = pd.read_json(path)
                    df.to_sql(table_name, engine, if_exists="replace", index=False)
                except Exception as e:
                    log_error(f"[ERROR] {path}: {e}")

def export_postgres_to_json():
    """Export all PostgreSQL tables to .jsonl in log/ directory."""
    insp = inspect(engine)
    for table_name in insp.get_table_names():
        out_path = os.path.join(LOG_DIR, f"{table_name}.jsonl")
        if not is_safe_path(out_path, ALLOWED_ROOTS):
            log_warning(f"[SKIP] Unsafe output path: {out_path}")
            continue
        log_info(f"Exporting table {table_name} to {out_path}...")
        try:
            df = pd.read_sql_table(table_name, engine)
            df.to_json(out_path, orient="records", lines=True)
        except Exception as e:
            log_error(f"[ERROR] {table_name}: {e}")

def import_sqlite_to_postgres():
    """Import all .db files from allowed dirs to PostgreSQL."""
    from sqlalchemy import create_engine as create_sqlite_engine
    for folder in ALLOWED_ROOTS:
        for db_path in glob.glob(os.path.join(folder, "*.db")):
            if not is_safe_path(db_path, ALLOWED_ROOTS):
                log_warning(f"[SKIP] Unsafe path: {db_path}")
                continue
            sqlite_engine = create_sqlite_engine(f"sqlite:///{db_path}")
            insp = inspect(sqlite_engine)
            for table_name in insp.get_table_names():
                log_info(f"Importing SQLite table {table_name} from {db_path} to PostgreSQL...")
                try:
                    df = pd.read_sql_table(table_name, sqlite_engine)
                    df.to_sql(table_name, engine, if_exists="replace", index=False)
                except Exception as e:
                    log_error(f"[ERROR] {db_path}:{table_name}: {e}")

def export_sqlite_to_json():
    """Export all tables from .db files in allowed dirs to .jsonl in log/."""
    from sqlalchemy import create_engine as create_sqlite_engine
    for folder in ALLOWED_ROOTS:
        for db_path in glob.glob(os.path.join(folder, "*.db")):
            if not is_safe_path(db_path, ALLOWED_ROOTS):
                log_warning(f"[SKIP] Unsafe path: {db_path}")
                continue
            sqlite_engine = create_sqlite_engine(f"sqlite:///{db_path}")
            insp = inspect(sqlite_engine)
            for table_name in insp.get_table_names():
                out_path = os.path.join(LOG_DIR, f"{table_name}_from_sqlite.jsonl")
                if not is_safe_path(out_path, ALLOWED_ROOTS):
                    log_warning(f"[SKIP] Unsafe output path: {out_path}")
                    continue
                log_info(f"Exporting SQLite table {table_name} from {db_path} to {out_path}...")
                try:
                    df = pd.read_sql_table(table_name, sqlite_engine)
                    df.to_json(out_path, orient="records", lines=True)
                except Exception as e:
                    log_error(f"[ERROR] {db_path}:{table_name}: {e}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Sync data between JSON/SQLite and PostgreSQL.")
    parser.add_argument("--import-json", action="store_true", help="Import all .jsonl/.json to PostgreSQL")
    parser.add_argument("--export-json", action="store_true", help="Export all PostgreSQL tables to .jsonl")
    parser.add_argument("--import-sqlite", action="store_true", help="Import all .db files to PostgreSQL")
    parser.add_argument("--export-sqlite", action="store_true", help="Export all .db tables to .jsonl")
    args = parser.parse_args()

    if args.import_json:
        import_json_to_postgres()
    if args.export_json:
        export_postgres_to_json()
    if args.import_sqlite:
        import_sqlite_to_postgres()
    if args.export_sqlite:
        export_sqlite_to_json()
    if not any(vars(args).values()):
        parser.print_help()

if __name__ == "__main__":
    main()
