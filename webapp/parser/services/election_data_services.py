"""
ElectionDataService: Service layer for all election DB operations.
Encapsulates CRUD, queries, and integrity helpers for contests, tables, batches, and related entities.
This allows orchestrator classes (ContextOrganizer, ContextCoordinator, etc.) to focus on business logic,
not DB details.

All methods are annotated and include docstrings for clarity.
"""

from typing import Optional, List, Any
from ..utils.db_utils import (
    get_session, upsert_contest, fetch_contest_full,
    get_table_structure_from_db, save_table_structure_to_db,
    create_batch_metadata, update_batch_metadata, get_batch_metadata,
    create_staging_election_result, get_staging_results_by_batch,
    create_warehouse_election_result, get_warehouse_results_by_batch,
    fetch_table_structures, search_table_structures, update_table_structure_fields,
    select_table_structures_by_title, clean_for_json, get_or_create_state,
    get_or_create_county, get_or_create_party, check_missing_tables
)
from ..utils.models import (
    Base, Contest, State, County, Party, 
    TableStructure, Panel, CandidatePanel, LocationPanel, 
    Heading, BallotType, ResultsTimestamp, PartyLabel, VoteMethod,
    Candidate, Office, District, Result, Button
)

def _get_contest_id(session, contest):
    """Helper to get contest_id from a contest dict or id."""
    if isinstance(contest, dict):
        if contest.get("id"):
            return contest["id"]
        # Try to find by title/year/type/state/county if id missing
        filters = {k: contest.get(k) for k in ("title", "year", "type_")}
        q = session.query(Contest)
        for k, v in filters.items():
            if v:
                q = q.filter(getattr(Contest, k) == v)
        obj = q.first()
        return obj.id if obj else None
    elif isinstance(contest, int):
        return contest
    return None

class ElectionDataService(object):
    """
    Service layer for all election-related DB operations.
    """

    def get_all_panels(self, limit=100) -> list:
        """Fetch all panels from the DB as list of dicts."""
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

    def get_all_tables(self, limit=100) -> list:
        """Fetch all tables from the DB as list of dicts."""
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

    def get_all_candidate_panels(self, limit=100) -> list:
        """Fetch all candidate panels from the DB as list of dicts."""
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

    def get_all_location_panels(self, limit=100) -> list:
        """Fetch all location panels from the DB as list of dicts."""
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

    def get_all_headings(self, limit=100) -> list:
        """Fetch all headings from the DB as list of dicts."""
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

    def get_all_ballot_types(self, limit=100) -> list:
        """Fetch all ballot types from the DB as list of dicts."""
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

    def get_all_results_timestamps(self, limit=100) -> list:
        """Fetch all results timestamps from the DB as list of dicts."""
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

    def get_all_party_labels(self, limit=100) -> list:
        """Fetch all party labels from the DB as list of dicts."""
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

    def get_all_vote_methods(self, limit=100) -> list:
        """Fetch all vote methods from the DB as list of dicts."""
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
            obj = session.query(Panel).filter_by(
                panel_text=panel.get("panel_text"),
                contest_id=contest_id
            ).first()
            if obj:
                obj.panel_html = panel.get("panel_html")
                obj.segment_hash = panel.get("segment_hash")
                obj.metastats = clean_for_json(panel)
            else:
                obj = Panel(
                    panel_text=panel.get("panel_text"),
                    panel_html=panel.get("panel_html"),
                    segment_hash=panel.get("segment_hash"),
                    contest_id=contest_id,
                    metastats=clean_for_json(panel)
                )
                session.add(obj)
            session.commit()

    def upsert_button(self, contest, button):
        """Insert or update a button for a contest."""
        with get_session() as session:
            contest_id = _get_contest_id(session, contest)
            obj = session.query(Button).filter_by(
                label=button.get("label"),
                selector=button.get("selector"),
                contest_id=contest_id
            ).first()
            if obj:
                obj.is_visible = button.get("is_visible", True)
                obj.is_clickable = button.get("is_clickable", True)
                obj.source = button.get("source")
                obj.metastats = clean_for_json(button)
            else:
                obj = Button(
                    label=button.get("label"),
                    selector=button.get("selector"),
                    contest_id=contest_id,
                    is_visible=button.get("is_visible", True),
                    is_clickable=button.get("is_clickable", True),
                    source=button.get("source"),
                    metastats=clean_for_json(button)
                )
                session.add(obj)
            session.commit()

    def upsert_candidate(self, candidate):
        """Insert or update a candidate."""
        with get_session() as session:
            obj = session.query(Candidate).filter_by(
                name=candidate.get("name"),
                office_id=candidate.get("office_id"),
                district_id=candidate.get("district_id"),
            ).first()
            if obj:
                obj.party_id = candidate.get("party_id")
                obj.metastats = clean_for_json(candidate)
            else:
                obj = Candidate(
                    name=candidate.get("name"),
                    party_id=candidate.get("party_id"),
                    office_id=candidate.get("office_id"),
                    district_id=candidate.get("district_id"),
                    metastats=clean_for_json(candidate)
                )
                session.add(obj)
            session.commit()

    def upsert_party(self, party):
        """Insert or update a party."""
        with get_session() as session:
            obj = session.query(Party).filter_by(name=party.get("name")).first()
            if obj:
                obj.abbreviation = party.get("abbreviation")
            else:
                obj = Party(
                    name=party.get("name"),
                    abbreviation=party.get("abbreviation")
                )
                session.add(obj)
            session.commit()

    def upsert_office(self, office):
        """Insert or update an office."""
        from ..utils.models import OfficeLevelEnum
        with get_session() as session:
            obj = session.query(Office).filter_by(name=office.get("name")).first()
            if obj:
                obj.level = OfficeLevelEnum(office.get("level")) if office.get("level") else obj.level
            else:
                obj = Office(
                    name=office.get("name"),
                    level=OfficeLevelEnum(office.get("level")) if office.get("level") else None
                )
                session.add(obj)
            session.commit()

    def upsert_district(self, district):
        """Insert or update a district."""
        with get_session() as session:
            obj = session.query(District).filter_by(
                name=district.get("name"),
                state_id=district.get("state_id"),
                county_id=district.get("county_id")
            ).first()
            if obj:
                obj.type_ = district.get("type_")
            else:
                obj = District(
                    name=district.get("name"),
                    type_=district.get("type_"),
                    state_id=district.get("state_id"),
                    county_id=district.get("county_id")
                )
                session.add(obj)
            session.commit()

    def upsert_result(self, result):
        """Insert or update an election result."""
        with get_session() as session:
            obj = session.query(Result).filter_by(
                candidate_id=result.get("candidate_id"),
                contest_id=result.get("contest_id")
            ).first()
            if obj:
                obj.votes = result.get("votes")
                obj.percent = result.get("percent")
                obj.is_winner = result.get("is_winner")
                obj.is_incumbent = result.get("is_incumbent")
                obj.vote_method = result.get("vote_method")
                obj.metastats = clean_for_json(result)
            else:
                obj = Result(
                    candidate_id=result.get("candidate_id"),
                    contest_id=result.get("contest_id"),
                    votes=result.get("votes"),
                    percent=result.get("percent"),
                    is_winner=result.get("is_winner"),
                    is_incumbent=result.get("is_incumbent"),
                    vote_method=result.get("vote_method"),
                    metastats=clean_for_json(result)
                )
                session.add(obj)
            session.commit()

    def upsert_entity(self, entity):
        """Insert or update a generic entity."""
        from ..utils.models import Entity
        with get_session() as session:
            obj = session.query(Entity).filter_by(
                entity_type=entity.get("entity_type"),
                value=entity.get("value")
            ).first()
            if obj:
                obj.metastats = clean_for_json(entity)
            else:
                obj = Entity(
                    entity_type=entity.get("entity_type"),
                    value=entity.get("value"),
                    metastats=clean_for_json(entity)
                )
                session.add(obj)
            session.commit()

    def upsert_table_structure(self, table_structure):
        """Insert or update a table structure."""
        from ..utils.models import TableStructure
        with get_session() as session:
            obj = session.query(TableStructure).filter_by(
                contest=table_structure.get("contest")
            ).first()
            if obj:
                obj.headers = table_structure.get("headers")
                obj.context = table_structure.get("context")
                obj.ml_confidence = table_structure.get("ml_confidence")
                obj.confirmed_by_user = table_structure.get("confirmed_by_user", False)
            else:
                obj = TableStructure(
                    contest=table_structure.get("contest"),
                    headers=table_structure.get("headers"),
                    context=table_structure.get("context"),
                    ml_confidence=table_structure.get("ml_confidence"),
                    confirmed_by_user=table_structure.get("confirmed_by_user", False)
                )
                session.add(obj)
            session.commit()

    def upsert_batch_metadata(self, batch_metadata):
        """Insert or update batch metadata."""
        from ..utils.models import BatchMetadata
        with get_session() as session:
            obj = session.query(BatchMetadata).filter_by(
                batch_id=batch_metadata.get("batch_id")
            ).first()
            if obj:
                obj.source = batch_metadata.get("source")
                obj.status = batch_metadata.get("status")
                obj.metastats = clean_for_json(batch_metadata)
            else:
                obj = BatchMetadata(
                    batch_id=batch_metadata.get("batch_id"),
                    source=batch_metadata.get("source"),
                    status=batch_metadata.get("status"),
                    metastats=clean_for_json(batch_metadata)
                )
                session.add(obj)
            session.commit()

    def upsert_alert(self, alert):
        """Insert or update an alert."""
        from ..utils.models import Alert
        with get_session() as session:
            obj = session.query(Alert).filter_by(
                level=alert.get("level"),
                message=alert.get("message")
            ).first()
            if obj:
                obj.context = alert.get("context")
            else:
                obj = Alert(
                    level=alert.get("level"),
                    message=alert.get("message"),
                    context=alert.get("context")
                )
                session.add(obj)
            session.commit()

    def upsert_embedding(self, embedding):
        """Insert or update an embedding."""
        from ..utils.models import EmbeddingCache
        with get_session() as session:
            obj = session.query(EmbeddingCache).filter_by(
                segment_hash=embedding.get("segment_hash")
            ).first()
            if obj:
                obj.embedding = embedding.get("embedding")
            else:
                obj = EmbeddingCache(
                    segment_hash=embedding.get("segment_hash"),
                    embedding=embedding.get("embedding")
                )
                session.add(obj)
            session.commit()

    def get_full_contest(self, contest_id: int) -> Optional[dict]:
        """
        Fetch a contest with all related data (state, county, office, candidates, results).
        """
        with get_session() as session:
            return fetch_contest_full(session, {"id": contest_id})

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
        with get_session() as session:
            query = session.query(Contest)
            # Apply filters
            for k, v in (filters or {}).items():
                if hasattr(Contest, k):
                    query = query.filter(getattr(Contest, k) == v)
            query = query.limit(limit)
            # Select columns if specified
            if columns:
                query = query.with_entities(*[getattr(Contest, col) for col in columns])
                # Return as dicts with column names
                return [dict(zip(columns, row)) for row in query.all()]
            else:
                # Return as dicts (all fields)
                return [row.as_dict() if hasattr(row, "as_dict") else {c.name: getattr(row, c.name) for c in Contest.__table__.columns} for row in query.all()]

    def get_all_full_contests(self, filters: Optional[dict] = None, limit: int = 100) -> List[dict]:
        """
        Fetch all contests with related data, optionally filtered.
        """
        contests = self.get_contests_by_advanced_filter(filters, limit=limit)
        with get_session() as session:
            return [fetch_contest_full(session, c) for c in contests]

    def update_contest_in_db(self, contest_update: dict) -> None:
        """
        Update an existing contest in the database by ID.
        contest_update must include the 'id' field.
        """
        from ..utils.db_utils import get_session
        from ..utils.models import Contest
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
        List all table names in the current DB schema.
        """
        return list(Base.metadata.tables.keys())

    def describe_table(self, table_name: str) -> Optional[dict]:
        """
        Return columns and relationships for a given table.
        """
        table = Base.metadata.tables.get(table_name)
        if not table:
            return None
        columns = [col.name for col in table.columns]
        return {
            "columns": columns,
            "table_args": str(table.table_args),
        }

    def get_table_metadata(self, table_name: str) -> Optional[dict]:
        """
        Return column names and types for a given table.
        """
        table = Base.metadata.tables.get(table_name)
        if not table:
            return None
        return {col.name: str(col.type) for col in table.columns}

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