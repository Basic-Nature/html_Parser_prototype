from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import orjson

from ..config import CACHE_DIR, CONTEXT_LIBRARY_DIR, LOG_DIR, OUTPUT_DIR
from ..Context_Integration.librarian import clean_for_json
from ..utils.db_utils import get_or_create_county, get_or_create_state, get_session
from ..utils.html_scanner import export_context_cache_for_db
from ..utils.logger_singleton import console
from ..utils.models import (
    BallotType,
    CandidatePanel,
    Contest,
    Heading,
    LocationPanel,
    Panel,
    PartyLabel,
    ResultsTimestamp,
    TableStructure,
    VoteMethod,
)
from .manual_correction_bot import AUX_FIELDS, MAIN_FIELDS

ALL_FIELDS = MAIN_FIELDS + AUX_FIELDS

MIGRATION_STATE_FILE = Path(CONTEXT_LIBRARY_DIR) / "migration_state.json"

def table_structure_exists(session, contest: str, headers: str, context: str) -> bool:
    return session.query(TableStructure).filter_by(
        contest=contest,
        headers=headers,
        context=context
    ).first() is not None

def create_table_structure(session, contest: str, headers: str, context: str, confirmed_by_user: bool = True):
    """
    Insert a new TableStructure row if not exists.
    """
    ts = TableStructure(
        contest=contest,
        headers=headers,
        context=context,
        confirmed_by_user=confirmed_by_user
    )
    session.add(ts)
    console.table(f"[MIGRATE] Added TableStructure: {contest}")

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
                    contest = entry.get("contest", "")
                    headers = orjson.dumps(entry.get("headers", [])).decode()
                    context = orjson.dumps(entry.get("context", {})).decode()
                    if not table_structure_exists(session, contest, headers, context):
                        create_table_structure(session, contest, headers, context, True)
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
                contest = entry.get("contest", "")
                headers = orjson.dumps(entry.get("headers", [])).decode()
                context = orjson.dumps(entry.get("context", {})).decode()
                if not table_structure_exists(session, contest, headers, context):
                    create_table_structure(
                        session,
                        contest,
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

def _normalize_geo(value: str | None) -> str | None:
    if not value:
        return None
    value = str(value).strip()
    if not value or value.lower() == "unknown":
        return None
    return value

def _coerce_year(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        raw = value.strip()
        if len(raw) == 4 and raw.isdigit():
            return int(raw)
    return None

def _ensure_contest_for_snapshot(
    session,
    *,
    title: str,
    year: int | None,
    contest_type: str | None,
    election_types: str | None,
    state_name: str | None,
    county_name: str | None,
):
    state_obj = get_or_create_state(session, state_name) if state_name else None
    county_obj = get_or_create_county(session, county_name, state_obj) if county_name and state_obj else None

    query = session.query(Contest).filter(Contest.title == title)
    if year is not None:
        query = query.filter(Contest.year == year)
    else:
        query = query.filter(Contest.year.is_(None))
    if contest_type:
        query = query.filter(Contest.type_ == contest_type)
    else:
        query = query.filter(Contest.type_.is_(None))
    if state_obj:
        query = query.filter(Contest.state_id == state_obj.id)
    else:
        query = query.filter(Contest.state_id.is_(None))
    if county_obj:
        query = query.filter(Contest.county_id == county_obj.id)
    else:
        query = query.filter(Contest.county_id.is_(None))

    contest = query.first()
    if contest is None:
        contest = Contest(
            title=title,
            year=year,
            type_=contest_type,
            election_types=election_types,
            state=state_obj,
            county=county_obj,
            metastats={},
        )
        session.add(contest)
    else:
        if state_obj and contest.state is None:
            contest.state = state_obj
        if county_obj and contest.county is None:
            contest.county = county_obj
        if election_types and not contest.election_types:
            contest.election_types = election_types
        if contest_type and not contest.type_:
            contest.type_ = contest_type
    return contest

def migrate_context_snapshot_from_metadata(metadata_path: Path) -> None:
    try:
        payload = orjson.loads(metadata_path.read_bytes())
    except Exception as exc:
        console.log(f"[MIGRATE] Skipping malformed metadata {metadata_path}: {exc}")
        return

    meta_context = payload.get("context") or {}
    snapshot = payload.get("context_snapshot") or meta_context.get("context_snapshot")
    if not snapshot:
        console.log(f"[MIGRATE] No context_snapshot found in metadata {metadata_path}. Skipping.")
        return

    contest_title = (
        payload.get("contest")
        or payload.get("race")
        or meta_context.get("contest")
        or meta_context.get("race")
    )
    if not contest_title:
        console.log(f"[MIGRATE] Missing contest name in metadata {metadata_path}")
        return

    state_name = _normalize_geo(payload.get("state") or meta_context.get("state"))
    county_name = _normalize_geo(payload.get("county") or meta_context.get("county"))
    handler = meta_context.get("handler") or payload.get("handler") or "json_handler"
    contest_type = meta_context.get("contest_type") or payload.get("contest_type")
    election_types = meta_context.get("election_types") or payload.get("election_types")
    year_value = payload.get("year") if payload.get("year") is not None else meta_context.get("year")
    year = _coerce_year(year_value)
    coverage = payload.get("coverage") or meta_context.get("coverage")
    row_count = payload.get("row_count")
    headers = payload.get("headers")
    csv_path = payload.get("csv_path")

    snapshot_entry = {
        "context_snapshot": snapshot,
        "coverage": coverage,
        "row_count": row_count,
        "headers": headers,
        "metadata_path": str(metadata_path),
        "source_handler": handler,
        "csv_path": csv_path,
        "imported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "context_summary": {
            key: meta_context.get(key)
            for key in (
                "state",
                "county",
                "contest",
                "session_id",
                "contest_slug",
                "source_slug",
                "year",
            )
            if key in meta_context
        },
    }

    with get_session() as session:
        contest = _ensure_contest_for_snapshot(
            session,
            title=contest_title,
            year=year,
            contest_type=contest_type,
            election_types=election_types,
            state_name=state_name,
            county_name=county_name,
        )

        existing_meta = contest.metastats if isinstance(contest.metastats, dict) else {}
        json_exports = existing_meta.get("json_exports")
        if not isinstance(json_exports, list):
            json_exports = []

        metadata_str = snapshot_entry["metadata_path"]
        json_exports = [entry for entry in json_exports if isinstance(entry, dict) and entry.get("metadata_path") != metadata_str]
        json_exports.append(snapshot_entry)
        json_exports.sort(key=lambda item: item.get("imported_at", ""), reverse=True)

        existing_meta["json_exports"] = json_exports
        existing_meta["latest_context_snapshot"] = snapshot
        existing_meta["latest_json_export_snapshot_path"] = metadata_str

        contest.metastats = clean_for_json(existing_meta)

        console.log(
            f"[MIGRATE] Snapshot captured for contest='{contest_title}' state='{state_name or 'Unknown'}'"
        )

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
    files_to_migrate += list(Path(OUTPUT_DIR).glob("**/*.metadata.json"))

    for file_path in files_to_migrate:
        mtime = file_path.stat().st_mtime
        if str(file_path) in state and state[str(file_path)] == mtime:
            continue  # Skip unchanged
        try:
            if file_path.name.endswith(".metadata.json"):
                migrate_context_snapshot_from_metadata(file_path)
                state[str(file_path)] = mtime
                continue
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