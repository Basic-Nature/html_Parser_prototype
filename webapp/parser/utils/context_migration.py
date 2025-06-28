import orjson
from pathlib import Path
from ..utils.db_utils import create_table_structure, get_session
from ..utils.models import TableStructure
from ..config import PROJECT_ROOT
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
            for line in f:
                entry = orjson.loads(line)
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
            data = json.load(f)
            if isinstance(data, list):
                entries = data
            elif isinstance(data, dict):
                entries = data.get("table_structures", [])
            else:
                entries = []
            for entry in entries:
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
    log_dir = Path(PROJECT_ROOT) / "log"
    context_dir = Path(PROJECT_ROOT) / "webapp" / "parser" / "Context_Integration" / "Context_Library"
    files_to_migrate = list(log_dir.glob("*.jsonl")) + list(log_dir.glob("*.json")) + \
                       list(context_dir.glob("*.jsonl")) + list(context_dir.glob("*.json"))
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