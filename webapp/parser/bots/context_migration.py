import orjson
from pathlib import Path
from ..utils.db_utils import create_table_structure, get_session
from ..utils.models import TableStructure
from ..config import CACHE_DIR, LOG_DIR, CONTEXT_LIBRARY_DIR
import json

def table_structure_exists(session, contest_title, headers, context):
    return session.query(TableStructure).filter_by(
        contest_title=contest_title,
        headers=headers,
        context=context
    ).first() is not None
    
def migrate_table_structures_from_jsonl(jsonl_path):
    print(f"[MIGRATE] Migrating from {jsonl_path} ...")
    count = 0
    with get_session() as session:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, 1):
                try:
                    entry = orjson.loads(line)
                except Exception as e:
                    print(f"[MIGRATE][WARN] {jsonl_path} line {idx}: Skipping malformed line: {e}")
                    continue
                if not isinstance(entry, dict):
                    print(f"[MIGRATE][WARN] {jsonl_path} line {idx}: Skipping non-dict entry: {entry}")
                    continue
                if entry.get("result") == "learning_confirmed" or entry.get("confirmed_by_user", False):
                    contest_title = entry.get("contest_title", "")
                    headers = orjson.dumps(entry.get("headers", [])).decode()
                    context = orjson.dumps(entry.get("context", {})).decode()
                    if not table_structure_exists(session, contest_title, headers, context):
                        create_table_structure(
                            contest_title=contest_title,
                            headers=headers,
                            context=context,
                            confirmed_by_user=True
                        )
                        count += 1
    print(f"[MIGRATE] Inserted {count} new table structures from {jsonl_path}")

def migrate_table_structures_from_json(json_path):
    print(f"[MIGRATE] Migrating from {json_path} ...")
    count = 0
    with get_session() as session:
        with open(json_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"[MIGRATE][WARN] {json_path}: Skipping malformed file: {e}")
                return
            if isinstance(data, list):
                entries = data
            elif isinstance(data, dict):
                entries = data.get("table_structures", [])
            else:
                entries = []
            for idx, entry in enumerate(entries, 1):
                if not isinstance(entry, dict):
                    print(f"[MIGRATE][WARN] {json_path} entry {idx}: Skipping non-dict entry: {entry}")
                    continue
                contest_title = entry.get("contest_title", "")
                headers = orjson.dumps(entry.get("headers", [])).decode()
                context = orjson.dumps(entry.get("context", {})).decode()
                if not table_structure_exists(session, contest_title, headers, context):
                    create_table_structure(
                        contest_title=contest_title,
                        headers=headers,
                        context=context,
                        confirmed_by_user=entry.get("confirmed_by_user", True)
                    )
                    count += 1
    print(f"[MIGRATE] Inserted {count} new table structures from {json_path}")

def migrate_all():
    # Only .jsonl in LOG_DIR, only .json in CACHE_DIR, both in CONTEXT_LIBRARY_DIR
    files_to_migrate = []
    # LOG_DIR: only .jsonl
    files_to_migrate += list(Path(LOG_DIR).glob("*.jsonl"))
    # CACHE_DIR: only .json
    files_to_migrate += list(Path(CACHE_DIR).glob("*.json"))
    # CONTEXT_LIBRARY_DIR: both .jsonl and .json
    files_to_migrate += list(Path(CONTEXT_LIBRARY_DIR).glob("*.jsonl"))
    files_to_migrate += list(Path(CONTEXT_LIBRARY_DIR).glob("*.json"))

    for file_path in files_to_migrate:
        if file_path.suffix == ".jsonl":
            migrate_table_structures_from_jsonl(file_path)
        elif file_path.suffix == ".json":
            migrate_table_structures_from_json(file_path)
        else:
            print(f"[MIGRATE] Skipped {file_path} (unsupported extension)")

if __name__ == "__main__":
    migrate_all()
    print("[MIGRATE] Migration complete.")