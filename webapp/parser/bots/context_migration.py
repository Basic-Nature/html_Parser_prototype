import orjson
from pathlib import Path
from typing import Any, Dict, List
from ..utils.db_utils import get_session
from ..utils.models import Contest, Panel, TableStructure, CandidatePanel, LocationPanel, Heading, BallotType, ResultsTimestamp, PartyLabel, VoteMethod
from ..config import CACHE_DIR, LOG_DIR, CONTEXT_LIBRARY_DIR
from ..utils.shared_logger import RichConsoleProxy, SharedLogger
from ..utils.html_scanner import export_context_cache_for_db
from ..bots.manual_correction_bot import MAIN_FIELDS, AUX_FIELDS

ALL_FIELDS = MAIN_FIELDS + AUX_FIELDS

console = RichConsoleProxy()
logger = SharedLogger()

MIGRATION_STATE_FILE = Path(CONTEXT_LIBRARY_DIR) / "migration_state.json"

def table_structure_exists(session, contest_title: str, headers: str, context: str) -> bool:
    return session.query(TableStructure).filter_by(
        contest_title=contest_title,
        headers=headers,
        context=context
    ).first() is not None

def create_table_structure(session, contest_title: str, headers: str, context: str, confirmed_by_user: bool = True):
    """
    Insert a new TableStructure row if not exists.
    """
    ts = TableStructure(
        contest_title=contest_title,
        headers=headers,
        context=context,
        confirmed_by_user=confirmed_by_user
    )
    session.add(ts)
    console.table(f"[MIGRATE] Added TableStructure: {contest_title}")

def migrate_table_structures_from_jsonl(jsonl_path: Path):
    console.panel(f"[MIGRATE] Migrating from {jsonl_path} ...")
    count = 0
    with get_session() as session:
        with open(jsonl_path, "rb") as f:
            for idx, line in enumerate(f, 1):
                try:
                    entry = orjson.loads(line)
                except Exception as e:
                    console.log(f"{jsonl_path} line {idx}: Skipping malformed line: {e}")
                    continue
                if not isinstance(entry, dict):
                    console.log(f"{jsonl_path} line {idx}: Skipping non-dict entry: {entry}")
                    continue
                if entry.get("result") == "learning_confirmed" or entry.get("confirmed_by_user", False):
                    contest_title = entry.get("contest_title", "")
                    headers = orjson.dumps(entry.get("headers", [])).decode()
                    context = orjson.dumps(entry.get("context", {})).decode()
                    if not table_structure_exists(session, contest_title, headers, context):
                        create_table_structure(session, contest_title, headers, context, True)
                        count += 1
        session.commit()
    console.log(f"[MIGRATE] Inserted {count} new table structures from {jsonl_path}")

def migrate_table_structures_from_json(json_path: Path):
    console.log(f"[MIGRATE] Migrating from {json_path} ...")
    count = 0
    with get_session() as session:
        with open(json_path, "rb") as f:
            try:
                data = orjson.loads(f.read())
            except Exception as e:
                console.log(f"{json_path}: Skipping malformed file: {e}")
                return
            if isinstance(data, list):
                entries = data
            elif isinstance(data, dict):
                entries = data.get("table_structures", [])
            else:
                entries = []
            for idx, entry in enumerate(entries, 1):
                if not isinstance(entry, dict):
                    console.log(f"{json_path} entry {idx}: Skipping non-dict entry: {entry}")
                    continue
                contest_title = entry.get("contest_title", "")
                headers = orjson.dumps(entry.get("headers", [])).decode()
                context = orjson.dumps(entry.get("context", {})).decode()
                if not table_structure_exists(session, contest_title, headers, context):
                    create_table_structure(
                        session,
                        contest_title,
                        headers,
                        context,
                        confirmed_by_user=entry.get("confirmed_by_user", True)
                    )
                    count += 1
        session.commit()
    console.panel(f"[MIGRATE] Inserted {count} new table structures from {json_path}")

def load_migration_state() -> Dict[str, Any]:
    if MIGRATION_STATE_FILE.exists():
        with open(MIGRATION_STATE_FILE, "rb") as f:
            return orjson.loads(f.read())
    return {}

def save_migration_state(state: Dict[str, Any]):
    MIGRATION_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    with open(MIGRATION_STATE_FILE, "wb") as f:
        f.write(orjson.dumps(state))

def migrate_all():
    """
    Migrate all context/log/aux files from LOG_DIR, CACHE_DIR, CONTEXT_LIBRARY_DIR.
    Only migrates changed files (by mtime).
    """
    state = load_migration_state()
    files_to_migrate: List[Path] = []
    patterns = []
    for field in ALL_FIELDS:
        patterns.append(f"*{field}*.jsonl")
        patterns.append(f"*{field}*.json")
    migrate_context_cache_to_db()
    for pattern in patterns:
        files_to_migrate += list(Path(LOG_DIR).glob(pattern))
        files_to_migrate += list(Path(CONTEXT_LIBRARY_DIR).glob(pattern))
        files_to_migrate += list(Path(CACHE_DIR).glob(pattern))

    for file_path in files_to_migrate:
        mtime = file_path.stat().st_mtime
        if str(file_path) in state and state[str(file_path)] == mtime:
            continue  # Skip unchanged
        try:
            # Route to the correct migration function based on file type or field
            if "table_structure" in file_path.name:
                if file_path.suffix == ".jsonl":
                    migrate_table_structures_from_jsonl(file_path)
                elif file_path.suffix == ".json":
                    migrate_table_structures_from_json(file_path)
            # Add more elifs for other field types as needed
            state[str(file_path)] = mtime
        except Exception as e:
            console.log(f"[MIGRATE] Failed to migrate {file_path}: {e}")
    save_migration_state(state)
    
def migrate_context_cache_to_db():
    """
    Migrate all context cache entries (from export_context_cache_for_db) into normalized DB tables.
    """
    entries = export_context_cache_for_db()
    with get_session() as session:
        for entry in entries:
            # --- Contests ---
            for contest in entry.get("contests", []):
                obj = Contest(
                    title=contest.get("title"),
                    year=contest.get("year"),
                    type_=contest.get("type_"),
                    # Add state/county/district/office if you have mapping logic
                )
                session.merge(obj)  # merge = upsert

            # --- Panels ---
            for panel in entry.get("panels", []):
                obj = Panel(
                    panel_text=panel.get("panel_text"),
                    panel_html=panel.get("panel_html"),
                    segment_hash=panel.get("segment_hash"),
                )
                session.merge(obj)

            # --- Candidate Panels ---
            for cp in entry.get("candidate_panels", []):
                obj = CandidatePanel(
                    candidate_panel_text=cp.get("candidate_panel_text"),
                    candidate_panel_html=cp.get("candidate_panel_html"),
                    year=cp.get("year"),
                    type_=cp.get("type_"),
                    segment_hash=cp.get("segment_hash"),
                )
                session.merge(obj)

            # --- Location Panels ---
            for lp in entry.get("location_panels", []):
                obj = LocationPanel(
                    location_panel_text=lp.get("location_panel_text"),
                    location_panel_html=lp.get("location_panel_html"),
                    year=lp.get("year"),
                    type_=lp.get("type_"),
                    segment_hash=lp.get("segment_hash"),
                )
                session.merge(obj)

            # --- Headings ---
            for heading in entry.get("headings", []):
                obj = Heading(
                    heading_text=heading.get("heading_text"),
                    heading_html=heading.get("heading_html"),
                    heading_type=heading.get("heading_type"),
                    segment_hash=heading.get("segment_hash"),
                )
                session.merge(obj)

            # --- Ballot Types ---
            for bt in entry.get("ballot_types", []):
                obj = BallotType(
                    ballot_types_text=bt.get("ballot_types_text"),
                    ballot_types_html=bt.get("ballot_types_html"),
                    year=bt.get("year"),
                    type_=bt.get("type_"),
                    segment_hash=bt.get("segment_hash"),
                )
                session.merge(obj)

            # --- Results Timestamps ---
            for ts in entry.get("results_timestamps", []):
                obj = ResultsTimestamp(
                    timestamp_text=ts.get("timestamp_text"),
                    timestamp_html=ts.get("timestamp_html"),
                    segment_hash=ts.get("segment_hash"),
                )
                session.merge(obj)

            # --- Party Labels ---
            for pl in entry.get("party_labels", []):
                obj = PartyLabel(
                    party_label_text=pl.get("party_label_text"),
                    party_label_html=pl.get("party_label_html"),
                    segment_hash=pl.get("segment_hash"),
                )
                session.merge(obj)

            # --- Vote Methods ---
            for vm in entry.get("vote_methods", []):
                obj = VoteMethod(
                    vote_method_text=vm.get("vote_method_text"),
                    vote_method_html=vm.get("vote_method_html"),
                    segment_hash=vm.get("segment_hash"),
                )
                session.merge(obj)

        session.commit()
    console.log(f"[MIGRATE] Migrated {len(entries)} context cache entries to DB.")
    
if __name__ == "__main__":
    migrate_all()
    console.rule("[MIGRATE] Migration complete.")