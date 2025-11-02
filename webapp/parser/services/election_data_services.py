"""
ElectionDataService: Service layer for all election DB operations.
Encapsulates CRUD, queries, and integrity helpers for contests, tables, batches, and related entities.
This allows orchestrator classes (ContextOrganizer, ContextCoordinator, etc.) to focus on business logic,
not DB details.

All methods are annotated and include docstrings for clarity.
"""
from typing import Any, Dict, Iterator, List, Optional, Protocol, Type, Union

from sqlalchemy import inspect
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeMeta, Session
from sqlalchemy.sql.schema import Column, Table

from ..Context_Integration.librarian import clean_for_json
from ..utils.db_utils import (
    SessionLocal,
    check_missing_tables,
    create_batch_metadata,
    create_staging_election_result,
    create_warehouse_election_result,
    fetch_contest_full,
    fetch_table_structures,
    get_batch_metadata,
    get_engine,
    get_or_create_county,
    get_or_create_party,
    get_or_create_state,
    get_session,
    get_staging_results_by_batch,
    get_table_structure_from_db,
    get_warehouse_results_by_batch,
    save_table_structure_to_db,
    search_table_structures,
    select_table_structures_by_title,
    update_batch_metadata,
    update_table_structure_fields,
    upsert_contest,
)
from ..utils.logger_singleton import logger
from ..utils.models import (
    BallotType,
    Base,
    Button,
    Candidate,
    CandidatePanel,
    Contest,
    County,
    District,
    Heading,
    LocationPanel,
    Office,
    Panel,
    Party,
    PartyLabel,
    Result,
    ResultsTimestamp,
    State,
    TableStructure,
    VoteMethod,
)
from ..utils.shared_logic import safe_items, safe_values


class DictConvertible(Protocol):
    def as_dict(self) -> Dict[str, Any]:
        """
        Return a dict of column names and their values for this ORM instance.
        Uses utility functions for robust access.
        """
        columns = get_table_columns(self)  # Safely get columns
        if columns:
            names = columns_to_names(columns)  # Safely get column names
            return {name: getattr(self, name, None) for name in names}
        # Fallback: try __dict__ if no columns found
        if hasattr(self, "__dict__"):
            # Exclude private attributes and SQLAlchemy internals
            return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        # Last resort: return empty dict
        return {}

def get_decl_class_registry(base: DeclarativeMeta) -> Iterator[Type[Any]]:
    """
    Safely yield ORM classes from SQLAlchemy Base._decl_class_registry.
    """
    registry = getattr(base, "_decl_class_registry", None)
    for cls in safe_values(registry):
        if isinstance(cls, type):
            yield cls

def iter_orm_classes(base: DeclarativeMeta) -> Iterator[Type[Any]]:
    """
    Iterate over ORM classes registered in the SQLAlchemy Base.
    """
    for cls in get_decl_class_registry(base):
        if isinstance(cls, type) and hasattr(cls, "__tablename__"):
            yield cls

def get_orm_class_by_tablename(base: DeclarativeMeta, table_name: str) -> Optional[Type[Any]]:
    """
    Return the ORM class for a given table name.
    """
    for cls in iter_orm_classes(base):
        if getattr(cls, "__tablename__", None) == table_name:
            return cls
    return None

def get_table_columns(obj: Any) -> List[Column]:
    """
    Robustly return a list of SQLAlchemy Column objects from an ORM row or class.
    Handles missing __table__ or columns attributes gracefully.
    """
    table: Table = getattr(obj, "__table__", None)
    if table is not None and hasattr(table, "columns"):
        return list(table.columns)
    return []

def get_row_table(row: Any) -> Optional[Table]:
    """
    Get the SQLAlchemy Table object from an ORM row.
    """
    return getattr(row, "__table__", None)

def iter_row_columns(row: Any) -> Iterator[Column]:
    """
    Iterate over columns of an ORM row's table.
    """
    table = get_row_table(row)
    if table is not None and hasattr(table, "columns"):
        return iter(table.columns)
    return iter([])

def row_to_dict(row: DictConvertible) -> Dict[str, Any]:
    """
    Convert an ORM row object to a dict, using __table__.columns for robust access.
    """
    table: Table = getattr(row, "__table__", None)
    if table is not None and hasattr(table, "columns"):
        return {col.name: getattr(row, col.name) for col in table.columns}
    # Fallback: try as_dict if available
    if hasattr(row, "as_dict"):
        return row.as_dict()
    return dict(row)

def _get_contest_id(session: Session, contest: Union[dict, int]) -> Optional[int]:
    """Helper to get contest_id from a contest dict or id."""
    if isinstance(contest, dict):
        if contest.get("id"):
            return contest["id"]
        filters = {k: contest.get(k) for k in ("title", "year", "type_")}
        q = session.query(Contest)
        for k, v in filters.items():
            if v:
                q = q.filter(getattr(Contest, k) == v)
        obj = q.first()
        return obj.id if obj else None
    elif isinstance(contest, int):
        return contest
    # Defensive: handle string or other types gracefully
    logger.error(f"_get_contest_id: Unexpected contest type: {type(contest)} value={contest}", exc_info=True)
    return None

def columns_to_names(columns: List[Column]) -> List[str]:
    """
    Get column names from a list of SQLAlchemy Column objects.
    """
    return [col.name for col in columns]

def get_metadata_tables() -> Dict[str, Any]:
    """
    Safely get the tables dict from SQLAlchemy Base metadata.
    Returns an empty dict if not available.
    """
    metadata = getattr(Base, "metadata", None)
    if not metadata or not hasattr(metadata, "tables"):
        return {}
    tables = getattr(metadata, "tables", {})
    if not isinstance(tables, dict):
        return {}
    return tables

class ElectionDataService(object):
    """
    Service layer for all election-related DB operations.
    """
    def __init__(self) -> None:
        """
        Initialize the service.
        This can be extended to accept configuration or dependencies if needed.
        """
        pass

    def get_full_contest(self, contest_id: int) -> Optional[dict]:
        """
        Fetch a contest with all related data (state, county, office, candidates, results).
        """
        try:
            with get_session() as session:
                return fetch_contest_full(session, {"id": contest_id})
        except Exception as e:
            logger.error(f"[get_full_contest] Error fetching contest_id={contest_id}: {e}", exc_info=True)
            return None

    def get_contests_by_advanced_filter(
        self,
        filters: Optional[dict] = None,
        columns: Optional[list] = None,
        limit: int = 100
    ) -> List[dict]:
        """
        Fetch contests with advanced filters and optional column selection.
        Args:
            filters: dict of field:value pairs to filter on (AND logic).
            columns: list of column names to return (if None, returns all fields).
            limit: max number of results.
        Returns:
            List of dicts, each representing a contest.
        """
        try:
            with get_session() as session:
                query = session.query(Contest)
                # Defensive: Only apply filters if they are valid and not None/empty
                for k, v in safe_items(filters):
                    if hasattr(Contest, k) and v is not None and v != "":
                        query = query.filter(getattr(Contest, k) == v)
                query = query.limit(limit)
                # Select columns if specified
                if columns:
                    query = query.with_entities(*[getattr(Contest, col) for col in columns])
                    # Return as dicts with column names
                    return [dict(zip(columns, row)) for row in query.all()]
                else:
                    # Return as dicts (all fields)
                    contest_columns = get_table_columns(Contest)  # List[Column]
                    column_names = columns_to_names(contest_columns)   # List[str]
                    results = []
                    for row in query.all():
                        if hasattr(row, "as_dict"):
                            results.append(row_to_dict(row))
                        else:
                            # Build dict from column names and row values
                            results.append({name: getattr(row, name, None) for name in column_names})
                    return results
        except Exception as e:
            logger.error(f"[get_contests_by_advanced_filter] Error: {e}", exc_info=True)
            return []

    def get_all_full_contests(self, filters: Optional[dict] = None, limit: int = 100) -> List[dict]:
        """
        Fetch all contests with related data, optionally filtered.
        """
        contests = self.get_contests_by_advanced_filter(filters, limit=limit)
        with get_session() as session:
            return [fetch_contest_full(session, c) for c in contests]

    def get_sample_rows(self, table_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Robustly fetch sample rows from a table using SQLAlchemy ORM.
        Handles missing tables, unknown columns, and returns clean dicts.
        """
        engine: Engine = get_engine()
        inspector = inspect(engine)
        table_names: List[str] = inspector.get_table_names()
        if table_name not in table_names:
            return []

        orm_class: Optional[Type[Any]] = get_orm_class_by_tablename(Base, table_name)
        if orm_class is None:
            return []

        session = SessionLocal()
        try:
            rows: List[Any] = session.query(orm_class).limit(limit).all()
            result: List[Dict[str, Any]] = []
            for row in rows:
                row_dict: Dict[str, Any] = row_to_dict(row)
                result.append(clean_for_json(row_dict))
            return result
        except Exception as e:
            logger.error(f"[get_sample_rows] Error fetching rows for table '{table_name}': {e}")
            return []
        finally:
            session.close()

    def get_all_panels(self, limit=100) -> list:
        """Fetch all panels from the DB as list of dicts."""
        try:
            with get_session() as session:
                # If you have a Panel table:
                if hasattr(session, "query") and hasattr(Panel, "panel_text"):
                    panels = session.query(Panel).order_by(Panel.id.desc()).limit(limit).all()
                    return [
                        {
                            "panel_text": p.panel_text,
                            "panel_html": getattr(p, "panel_html", None),
                            "segment_hash": getattr(p, "segment_hash", None),
                        }
                        for p in panels
                    ]
                # Fallback: Try TableStructure or Contest if Panel table doesn't exist
                return []
        except Exception as e:
            logger.error(f"[get_all_panels] Error fetching panels: {e}", exc_info=True)
            return []

    def get_all_tables(self, limit=100) -> list:
        """Fetch all tables from the DB as list of dicts."""
        try:
            with get_session() as session:
                tables = session.query(TableStructure).order_by(TableStructure.id.desc()).limit(limit).all()
                return [
                    {
                        "table_text": getattr(t, "table_text", None),
                    "table_html": getattr(t, "table_html", None),
                    "year": getattr(t, "year", None),
                    "type_": getattr(t, "type_", None),
                    "segment_hash": getattr(t, "segment_hash", None),
                }
                for t in tables
            ]
            return []
        except Exception as e:
            logger.error(f"[get_all_tables] Error fetching tables: {e}", exc_info=True)
            return []

    def get_all_candidate_panels(self, limit=100) -> list:
        """Fetch all candidate panels from the DB as list of dicts."""
        try:
            with get_session() as session:
                if hasattr(session, "query") and hasattr(CandidatePanel, "candidate_panel_text"):
                    panels = session.query(CandidatePanel).order_by(CandidatePanel.id.desc()).limit(limit).all()
                    return [
                        {
                            "candidate_panel_text": p.candidate_panel_text,
                            "candidate_panel_html": getattr(p, "candidate_panel_html", None),
                            "year": getattr(p, "year", None),
                            "type_": getattr(p, "type_", None),
                            "segment_hash": getattr(p, "segment_hash", None),
                        }
                        for p in panels
                    ]
                return []
        except Exception as e:
            logger.error(f"[get_all_candidate_panels] Error fetching candidate panels: {e}", exc_info=True)
            return []

    def get_all_location_panels(self, limit=100) -> list:
        """Fetch all location panels from the DB as list of dicts."""
        try:
            with get_session() as session:
                if hasattr(session, "query") and hasattr(LocationPanel, "location_panel_text"):
                    panels = session.query(LocationPanel).order_by(LocationPanel.id.desc()).limit(limit).all()
                    return [
                        {
                            "location_panel_text": p.location_panel_text,
                            "location_panel_html": getattr(p, "location_panel_html", None),
                            "year": getattr(p, "year", None),
                            "type_": getattr(p, "type_", None),
                            "segment_hash": getattr(p, "segment_hash", None),
                        }
                        for p in panels
                    ]
                return []
        except Exception as e:
            logger.error(f"[get_all_location_panels] Error fetching location panels: {e}", exc_info=True)
            return []

    def get_all_headings(self, limit=100) -> list:
        """Fetch all headings from the DB as list of dicts."""
        try:
            with get_session() as session:
                if hasattr(session, "query") and hasattr(Heading, "heading_text"):
                    headings = session.query(Heading).order_by(Heading.id.desc()).limit(limit).all()
                    return [
                        {
                            "heading_text": h.heading_text,
                            "heading_html": getattr(h, "heading_html", None),
                            "segment_hash": getattr(h, "segment_hash", None),
                            "heading_type": getattr(h, "heading_type", None),
                        }
                        for h in headings
                    ]
                return []
        except Exception as e:
            logger.error(f"[get_all_headings] Error fetching headings: {e}", exc_info=True)
            return []

    def get_all_ballot_types(self, limit=100) -> list:
        """Fetch all ballot types from the DB as list of dicts."""
        try:
            with get_session() as session:
                if hasattr(session, "query") and hasattr(BallotType, "ballot_types_text"):
                    ballot_types = session.query(BallotType).order_by(BallotType.id.desc()).limit(limit).all()
                    return [
                        {
                            "ballot_types_text": b.ballot_types_text,
                            "ballot_types_html": getattr(b, "ballot_types_html", None),
                            "year": getattr(b, "year", None),
                            "type_": getattr(b, "type_", None),
                            "segment_hash": getattr(b, "segment_hash", None),
                        }
                        for b in ballot_types
                    ]
                return []
        except Exception as e:
            logger.error(f"[get_all_ballot_types] Error fetching ballot types: {e}", exc_info=True)
            return []

    def get_all_results_timestamps(self, limit=100) -> list:
        """Fetch all results timestamps from the DB as list of dicts."""
        try:
            with get_session() as session:
                if hasattr(session, "query") and hasattr(ResultsTimestamp, "timestamp_text"):
                    timestamps = session.query(ResultsTimestamp).order_by(ResultsTimestamp.id.desc()).limit(limit).all()
                    return [
                        {
                            "timestamp_text": t.timestamp_text,
                            "timestamp_html": getattr(t, "timestamp_html", None),
                            "segment_hash": getattr(t, "segment_hash", None),
                        }
                        for t in timestamps
                    ]
                return []
        except Exception as e:
            logger.error(f"[get_all_results_timestamps] Error fetching results timestamps: {e}", exc_info=True)
            return []

    def get_all_party_labels(self, limit=100) -> list:
        """Fetch all party labels from the DB as list of dicts."""
        try:
            with get_session() as session:
                if hasattr(session, "query") and hasattr(PartyLabel, "party_label_text"):
                    party_labels = session.query(PartyLabel).order_by(PartyLabel.id.desc()).limit(limit).all()
                    return [
                        {
                            "party_label_text": p.party_label_text,
                            "party_label_html": getattr(p, "party_label_html", None),
                            "segment_hash": getattr(p, "segment_hash", None),
                        }
                        for p in party_labels
                    ]
                return []
        except Exception as e:
            logger.error(f"[get_all_party_labels] Error fetching party labels: {e}", exc_info=True)
            return []

    def get_all_vote_methods(self, limit=100) -> list:
        """Fetch all vote methods from the DB as list of dicts."""
        try:
            with get_session() as session:
                if hasattr(session, "query") and hasattr(VoteMethod, "vote_method_text"):
                    vote_methods = session.query(VoteMethod).order_by(VoteMethod.id.desc()).limit(limit).all()
                    return [
                        {
                            "vote_method_text": v.vote_method_text,
                            "vote_method_html": getattr(v, "vote_method_html", None),
                            "segment_hash": getattr(v, "segment_hash", None),
                        }
                        for v in vote_methods
                    ]
                return []
        except Exception as e:
            logger.error(f"[get_all_vote_methods] Error fetching vote methods: {e}", exc_info=True)
            return []

    # --- Contest Operations ---

    def upsert_contest(self, contest_dict: dict, auto_create_related: bool = True) -> None:
        """
        Insert or update a contest, auto-creating related State/County if needed.
        """
        with get_session() as session:
            upsert_contest(session, contest_dict, auto_create_related=auto_create_related)
            session.commit()

    def upsert_panel(self, contest, panel):
        """Insert or update a panel for a contest."""
        with get_session() as session:
            contest_id = _get_contest_id(session, contest)
            panel_dict = panel if isinstance(panel, dict) else {}
            obj = session.query(Panel).filter_by(
                panel_text=panel_dict.get("panel_text"),
                contest_id=contest_id
            ).first()
            if obj:
                obj.panel_html = panel_dict.get("panel_html")
                obj.segment_hash = panel_dict.get("segment_hash")
                obj.metastats = clean_for_json(panel)
            else:
                obj = Panel(
                    panel_text=panel_dict.get("panel_text"),
                    panel_html=panel_dict.get("panel_html"),
                    segment_hash=panel_dict.get("segment_hash"),
                    contest_id=contest_id,
                    metastats=clean_for_json(panel)
                )
                session.add(obj)
            session.commit()

    def upsert_button(self, contest, button):
        """Insert or update a button for a contest."""
        with get_session() as session:
            contest_id = _get_contest_id(session, contest)
            button_dict = button if isinstance(button, dict) else {}
            obj = session.query(Button).filter_by(
                label=button_dict.get("label"),
                selector=button_dict.get("selector"),
                contest_id=contest_id
            ).first()
            if obj:
                obj.is_visible = button_dict.get("is_visible", True)
                obj.is_clickable = button_dict.get("is_clickable", True)
                obj.source = button_dict.get("source")
                obj.metastats = clean_for_json(button)
            else:
                obj = Button(
                    label=button_dict.get("label"),
                    selector=button_dict.get("selector"),
                    contest_id=contest_id,
                    is_visible=button_dict.get("is_visible", True),
                    is_clickable=button_dict.get("is_clickable", True),
                    source=button_dict.get("source"),
                    metastats=clean_for_json(button)
                )
                session.add(obj)
            session.commit()

    def upsert_candidate(self, candidate):
        """Insert or update a candidate."""
        with get_session() as session:
            candidate_dict = candidate if isinstance(candidate, dict) else {}
            obj = session.query(Candidate).filter_by(
                name=candidate_dict.get("name"),
                office_id=candidate_dict.get("office_id"),
                district_id=candidate_dict.get("district_id"),
            ).first()
            if obj:
                obj.party_id = candidate_dict.get("party_id")
                obj.metastats = clean_for_json(candidate)
            else:
                obj = Candidate(
                    name=candidate_dict.get("name"),
                    party_id=candidate_dict.get("party_id"),
                    office_id=candidate_dict.get("office_id"),
                    district_id=candidate_dict.get("district_id"),
                    metastats=clean_for_json(candidate)
                )
                session.add(obj)
            session.commit()

    def upsert_party(self, party):
        """Insert or update a party."""
        with get_session() as session:
            party_dict = party if isinstance(party, dict) else {}
            obj = session.query(Party).filter_by(name=party_dict.get("name")).first()
            if obj:
                obj.abbreviation = party_dict.get("abbreviation")
            else:
                obj = Party(
                    name=party_dict.get("name"),
                    abbreviation=party_dict.get("abbreviation")
                )
                session.add(obj)
            session.commit()

    def upsert_office(self, office):
        """Insert or update an office."""
        from ..utils.models import OfficeLevelEnum
        with get_session() as session:
            office_dict = office if isinstance(office, dict) else {}
            obj = session.query(Office).filter_by(name=office_dict.get("name")).first()
            if obj:
                obj.level = OfficeLevelEnum(office_dict.get("level")) if office_dict.get("level") else obj.level
            else:
                obj = Office(
                    name=office_dict.get("name"),
                    level=OfficeLevelEnum(office_dict.get("level")) if office_dict.get("level") else None
                )
                session.add(obj)
            session.commit()

    def upsert_district(self, district):
        """Insert or update a district."""
        with get_session() as session:
            district_dict = district if isinstance(district, dict) else {}
            obj = session.query(District).filter_by(
                name=district_dict.get("name"),
                state_id=district_dict.get("state_id"),
                county_id=district_dict.get("county_id")
            ).first()
            if obj:
                obj.type_ = district_dict.get("type_")
            else:
                obj = District(
                    name=district_dict.get("name"),
                    type_=district_dict.get("type_"),
                    state_id=district_dict.get("state_id"),
                    county_id=district_dict.get("county_id")
                )
                session.add(obj)
            session.commit()

    def upsert_result(self, result):
        """Insert or update an election result."""
        with get_session() as session:
            result_dict = result if isinstance(result, dict) else {}
            obj = session.query(Result).filter_by(
                candidate_id=result_dict.get("candidate_id"),
                contest_id=result_dict.get("contest_id")
            ).first()
            if obj:
                obj.votes = result_dict.get("votes")
                obj.percent = result_dict.get("percent")
                obj.is_winner = result_dict.get("is_winner")
                obj.is_incumbent = result_dict.get("is_incumbent")
                obj.vote_method = result_dict.get("vote_method")
                obj.metastats = clean_for_json(result)
            else:
                obj = Result(
                    candidate_id=result_dict.get("candidate_id"),
                    contest_id=result_dict.get("contest_id"),
                    votes=result_dict.get("votes"),
                    percent=result_dict.get("percent"),
                    is_winner=result_dict.get("is_winner"),
                    is_incumbent=result_dict.get("is_incumbent"),
                    vote_method=result_dict.get("vote_method"),
                    metastats=clean_for_json(result)
                )
                session.add(obj)
            session.commit()

    def upsert_entity(self, entity):
        """Insert or update a generic entity."""
        from ..utils.models import Entity
        with get_session() as session:
            entity_dict = entity if isinstance(entity, dict) else {}
            obj = session.query(Entity).filter_by(
                entity_type=entity_dict.get("entity_type"),
                value=entity_dict.get("value")
            ).first()
            if obj:
                obj.metastats = clean_for_json(entity)
            else:
                obj = Entity(
                    entity_type=entity_dict.get("entity_type"),
                    value=entity_dict.get("value"),
                    metastats=clean_for_json(entity)
                )
                session.add(obj)
            session.commit()

    def upsert_table_structure(self, table_structure):
        """Insert or update a table structure."""
        from ..utils.models import TableStructure
        with get_session() as session:
            ts_dict = table_structure if isinstance(table_structure, dict) else {}
            obj = session.query(TableStructure).filter_by(
                contest=ts_dict.get("contest")
            ).first()
            if obj:
                obj.headers = ts_dict.get("headers")
                obj.context = ts_dict.get("context")
                obj.ml_confidence = ts_dict.get("ml_confidence")
                obj.confirmed_by_user = ts_dict.get("confirmed_by_user", False)
            else:
                obj = TableStructure(
                    contest=ts_dict.get("contest"),
                    headers=ts_dict.get("headers"),
                    context=ts_dict.get("context"),
                    ml_confidence=ts_dict.get("ml_confidence"),
                    confirmed_by_user=ts_dict.get("confirmed_by_user", False)
                )
                session.add(obj)
            session.commit()

    def upsert_batch_metadata(self, batch_metadata):
        """Insert or update batch metadata."""
        from ..utils.models import BatchMetadata
        with get_session() as session:
            bm_dict = batch_metadata if isinstance(batch_metadata, dict) else {}
            obj = session.query(BatchMetadata).filter_by(
                batch_id=bm_dict.get("batch_id")
            ).first()
            if obj:
                obj.source = bm_dict.get("source")
                obj.status = bm_dict.get("status")
                obj.metastats = clean_for_json(batch_metadata)
            else:
                obj = BatchMetadata(
                    batch_id=bm_dict.get("batch_id"),
                    source=bm_dict.get("source"),
                    status=bm_dict.get("status"),
                    metastats=clean_for_json(batch_metadata)
                )
                session.add(obj)
            session.commit()

    def upsert_alert(self, alert):
        """Insert or update an alert."""
        from ..utils.models import Alert
        with get_session() as session:
            alert_dict = alert if isinstance(alert, dict) else {}
            obj = session.query(Alert).filter_by(
                level=alert_dict.get("level"),
                message=alert_dict.get("message")
            ).first()
            if obj:
                obj.context = alert_dict.get("context")
            else:
                obj = Alert(
                    level=alert_dict.get("level"),
                    message=alert_dict.get("message"),
                    context=alert_dict.get("context")
                )
                session.add(obj)
            session.commit()

    def upsert_embedding(self, embedding):
        """Insert or update an embedding."""
        from ..utils.models import EmbeddingCache
        with get_session() as session:
            emb_dict = embedding if isinstance(embedding, dict) else {}
            obj = session.query(EmbeddingCache).filter_by(
                segment_hash=emb_dict.get("segment_hash")
            ).first()
            if obj:
                obj.embedding = emb_dict.get("embedding")
            else:
                obj = EmbeddingCache(
                    segment_hash=emb_dict.get("segment_hash"),
                    embedding=emb_dict.get("embedding")
                )
                session.add(obj)
            session.commit()

    def update_contest_in_db(self, contest_update: dict) -> None:
        """
        Update an existing contest in the database by ID.
        contest_update must include the 'id' field.
        """
        contest_id = contest_update.get("id")
        if not contest_id:
            raise ValueError("contest_update must include 'id'")
        with get_session() as session:
            contest = session.query(Contest).filter_by(id=contest_id).first()
            if not contest:
                raise ValueError(f"Contest with id={contest_id} not found")
            for k, v in contest_update.items():
                if k != "id" and hasattr(contest, k):
                    setattr(contest, k, v)
            session.commit()

    # --- Table Structure Operations ---

    def save_table_structure(self, contest: str, headers: Any, context: Any, ml_confidence: Optional[float] = None, confirmed_by_user: bool = False) -> None:
        """
        Upsert a table structure for a contest.
        """
        save_table_structure_to_db(contest, headers, context, ml_confidence, confirmed_by_user)

    def get_table_structure(self, contest: str, context: Any = None) -> Optional[dict]:
        """
        Retrieve the best-matching table structure for a contest.
        """
        return get_table_structure_from_db(contest, context)

    def fetch_table_structures(self, filters: Optional[dict] = None, limit: int = 100, order_by=None, confirmed_only: bool = False) -> List[TableStructure]:
        """
        Fetch table structures with optional filters, ordering, and confirmation status.
        """
        return fetch_table_structures(filters, limit, order_by, confirmed_only)

    def search_table_structures(self, search_terms: dict, limit: int = 100) -> List[TableStructure]:
        """
        Search TableStructure using dynamic AND/OR conditions.
        """
        return search_table_structures(search_terms, limit)

    def update_table_structure_fields(self, table_id: int, fields: dict) -> int:
        """
        Update fields on a TableStructure by ID.
        """
        return update_table_structure_fields(table_id, fields)

    def select_table_structures_by_title(self, title: str, limit: int = 10) -> List[TableStructure]:
        """
        Fetch TableStructures by contest.
        """
        return select_table_structures_by_title(title, limit)

    # --- Batch Metadata Operations ---

    def create_batch_metadata(self, source: str, status: str = 'pending'):
        """
        Create a new batch metadata record.
        """
        return create_batch_metadata(source, status)

    def update_batch_metadata(self, batch_id, **kwargs):
        """
        Update fields on a batch metadata record.
        """
        return update_batch_metadata(batch_id, **kwargs)

    def get_batch_metadata(self, batch_id):
        """
        Fetch a batch metadata record by ID.
        """
        return get_batch_metadata(batch_id)

    # --- Staging/Warehouse Election Result Operations ---

    def create_staging_election_result(self, **kwargs):
        """
        Create a new staging election result.
        """
        return create_staging_election_result(**kwargs)

    def get_staging_results_by_batch(self, batch_id):
        """
        Fetch all staging results for a batch.
        """
        return get_staging_results_by_batch(batch_id)

    def create_warehouse_election_result(self, **kwargs):
        """
        Create a new warehouse election result.
        """
        return create_warehouse_election_result(**kwargs)

    def get_warehouse_results_by_batch(self, batch_id):
        """
        Fetch all warehouse results for a batch.
        """
        return get_warehouse_results_by_batch(batch_id)

    # --- State/County/Party CRUD ---

    def get_or_create_state(self, state_name: str) -> Optional[State]:
        """
        Get or create a State by name.
        """
        with get_session() as session:
            return get_or_create_state(session, state_name)

    def get_or_create_county(self, county_name: str, state: State) -> Optional[County]:
        """
        Get or create a County by name and State.
        """
        with get_session() as session:
            return get_or_create_county(session, county_name, state)

    def get_or_create_party(self, party_name: str) -> Optional[Party]:
        """
        Get or create a Party by name.
        """
        with get_session() as session:
            return get_or_create_party(session, party_name)

    # --- DB Schema/Diagnostics ---

    def list_tables(self) -> List[str]:
        """
        List all table names in the current DB schema, with robust annotation safeguards.
        """
        tables = get_metadata_tables()
        return list(tables.keys())

    def describe_table(self, table_name: str) -> Optional[dict]:
        """
        Return columns and relationships for a given table, with robust annotation safeguards.
        """
        tables = get_metadata_tables()
        table = tables.get(table_name)
        if table is None or not hasattr(table, "columns"):
            return None
        columns = [getattr(col, "name", None) for col in getattr(table, "columns", []) if hasattr(col, "name")]
        table_args = str(getattr(table, "table_args", ""))
        return {"columns": columns, "table_args": table_args}

    def get_table_metadata(self, table_name: str) -> Optional[dict]:
        """
        Return column names and types for a given table, with robust guards.
        """
        tables = get_metadata_tables()
        table = tables.get(table_name)
        if table is None or not hasattr(table, "columns"):
            return None
        columns = list(getattr(table, "columns", []))
        column_info = {}
        for col in columns:
            col_name = getattr(col, "name", None)
            col_type = getattr(col, "type", None)
            if col_name is not None and col_type is not None:
                column_info[col_name] = str(col_type)
        table_args = str(getattr(table, "table_args", ""))
        return {"columns": column_info, "table_args": table_args}

    def check_missing_tables(self) -> List[str]:
        """
        Return a list of expected tables that are missing in the DB.
        """
        return check_missing_tables(self)

    # --- Utility ---

    def clean_for_json(self, obj) -> dict:
        """
        Clean an object for JSON serialization (handles sets, numpy, etc.).
        """
        return clean_for_json(obj)