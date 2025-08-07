from __future__ import annotations
# webapp/parser/utils/db_utils.py
# ---------------------------------------------------------------
# Database utility functions for Smart Elections Parser Webapp
# ---------------------------------------------------------------
import orjson
from typing import Optional, List, Generator
from sqlalchemy import create_engine, update, select, and_, or_, desc
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import inspect
from contextlib import contextmanager
from .models import (
    Contest, TableStructure, BatchMetadata, 
    StagingElectionResult, WarehouseElectionResult, Base,
    State, County, Party,
)
from ..Context_Integration.librarian import (
    clean_for_json
)
from .logger_singleton import logger
from ..config import POSTGRES_URL

# Set up SQLAlchemy engine and session
engine = create_engine(POSTGRES_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
_engine = None  # for lazy initialization if needed

def robust_orjson_loads(val) -> dict:
    """Load JSON robustly from either bytes or str."""
    if isinstance(val, bytes):
        return orjson.loads(val)
    elif isinstance(val, str):
        return orjson.loads(val.encode("utf-8"))
    else:
        raise TypeError(f"Cannot decode type {type(val)} with orjson")

# --- Robust session context manager ---
@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Yield a SQLAlchemy session, ensuring proper cleanup."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

def get_engine() -> create_engine:
    global _engine
    if _engine is None:
        _engine = create_engine(POSTGRES_URL, echo=False, future=True)
    return _engine

# --- Contest Operations ---
def update_contest_in_db(contest: dict, session: Optional[Session] = None) -> None:
    """
    Update a contest in the database using SQLAlchemy.
    """
    close_session = False
    if session is None:
        session = SessionLocal()
        close_session = True
    try:
        db_obj = session.get(Contest, contest.get("id"))
        if db_obj:
            db_obj.title = contest.get("title")
            db_obj.year = contest.get("year")
            db_obj.type_ = contest.get("type_")
            db_obj.state = contest.get("state")
            db_obj.county = contest.get("county")
            db_obj.metadata = contest
            session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        raise e
    finally:
        if close_session:
            session.close()

def fetch_contests_by_filter(filters: Optional[dict] = None, limit: int = 100, session: Optional[Session] = None) -> List[dict]:
    """
    Fetch contests from the database with optional filters and limit.
    """
    close_session = False
    if session is None:
        session = SessionLocal()
        close_session = True
    try:
        query = session.query(Contest)
        if filters:
            for k, v in filters.items():
                query = query.filter(getattr(Contest, k) == v)
        query = query.order_by(desc(Contest.id)).limit(limit)
        contests = []
        for row in query:
            contest = {
                "id": row.id,
                "title": row.title,
                "year": row.year,
                "type_": row.type_,
                "state": row.state,
                "county": row.county,
                **(row.metadata if isinstance(row.metadata, dict) else {})
            }
            contests.append(contest)
        return contests
    finally:
        if close_session:
            session.close()



# --- Utility: Create all tables (run once at startup or migration) ---
def create_all_tables() -> None:
    Base.metadata.create_all(engine)

# --- BatchMetadata Operations ---
def create_batch_metadata(source: str, status: str = 'pending') -> BatchMetadata:
    with get_session() as session:
        batch = BatchMetadata(source=source, status=status)
        session.add(batch)
        session.flush()  # get batch_id
        return batch

def update_batch_metadata(batch_id, **kwargs) -> Optional[BatchMetadata]:
    with get_session() as session:
        batch = session.get(BatchMetadata, batch_id)
        if batch:
            for k, v in kwargs.items():
                setattr(batch, k, v)
            session.commit()
        return batch

def get_batch_metadata(batch_id) -> Optional[BatchMetadata]:
    with get_session() as session:
        return session.get(BatchMetadata, batch_id)

# --- StagingElectionResult Operations ---
def create_staging_election_result(**kwargs) -> StagingElectionResult:
    with get_session() as session:
        result = StagingElectionResult(**kwargs)
        session.add(result)
        session.flush()
        return result

def get_staging_results_by_batch(batch_id) -> List[StagingElectionResult]:
    with get_session() as session:
        return session.query(StagingElectionResult).filter_by(batch_id=batch_id).all()

# --- WarehouseElectionResult Operations ---
def create_warehouse_election_result(**kwargs) -> WarehouseElectionResult:
    with get_session() as session:
        result = WarehouseElectionResult(**kwargs)
        session.add(result)
        session.flush()
        return result

def get_warehouse_results_by_batch(batch_id) -> List[WarehouseElectionResult]:
    with get_session() as session:
        return session.query(WarehouseElectionResult).filter_by(batch_id=batch_id).all()

def create_table_structure(contest, headers, context, ml_confidence=None, confirmed_by_user=False):
    with get_session() as session:
        obj = TableStructure(
            contest=contest,
            headers=headers,
            context=context,
            ml_confidence=ml_confidence,
            confirmed_by_user=confirmed_by_user
        )
        session.add(obj)
        session.flush()
        return obj

def update_table_structure(table_id, **kwargs) -> Optional[TableStructure]:
    with get_session() as session:
        obj = session.get(TableStructure, table_id)
        if obj:
            for k, v in kwargs.items():
                setattr(obj, k, v)
            session.commit()
        return obj

def get_table_structure_by_id(table_id) -> Optional[TableStructure]:
    with get_session() as session:
        return session.get(TableStructure, table_id)

def fetch_table_structures(filters: Optional[dict] = None, limit: int = 100, order_by=None, confirmed_only=False) -> list:
    with get_session() as session:
        query = session.query(TableStructure)
        if filters:
            for k, v in filters.items():
                query = query.filter(getattr(TableStructure, k) == v)
        if confirmed_only:
            query = query.filter(TableStructure.confirmed_by_user == True)
        if order_by:
            query = query.order_by(order_by)
        else:
            query = query.order_by(desc(TableStructure.id))
        return query.limit(limit).all()

def search_table_structures(search_terms: dict, limit: int = 100) -> list:
    """
    Search TableStructure using dynamic AND/OR conditions.
    search_terms: dict with keys as column names and values as search values (can be list for OR)
    """
    with get_session() as session:
        conditions = []
        for k, v in search_terms.items():
            col = getattr(TableStructure, k)
            if isinstance(v, list):
                conditions.append(or_(*[col == val for val in v]))
            else:
                conditions.append(col == v)
        query = session.query(TableStructure).filter(and_(*conditions)).order_by(desc(TableStructure.id)).limit(limit)
        return query.all()

def update_table_structure_fields(table_id, fields: dict) -> int:
    """
    Use SQLAlchemy's update construct for dynamic field updates on TableStructure.
    """
    with get_session() as session:
        stmt = (
            update(TableStructure)
            .where(TableStructure.id == table_id)
            .values(**fields)
            .execution_options(synchronize_session="fetch")
        )
        result = session.execute(stmt)
        session.commit()
        return result.rowcount

def select_table_structures_by_title(title: str, limit: int = 10) -> List[TableStructure]:
    """
    Use SQLAlchemy's select construct to fetch TableStructures by contest.
    """
    with get_session() as session:
        stmt = select(TableStructure).where(TableStructure.contest == title).limit(limit)
        return session.execute(stmt).scalars().all()
    
def save_table_structure_to_db(contest, headers, context, ml_confidence=None, confirmed_by_user=False) -> None:
    """
    Upsert a table structure using SQLAlchemy ORM. Updates if contest exists, else inserts.
    """
    try:
        with get_session() as session:
            obj = session.execute(
                select(TableStructure).where(TableStructure.contest == contest)
            ).scalar_one_or_none()
            if obj:
                obj.headers = clean_for_json(headers)
                obj.context = clean_for_json(context)
                obj.ml_confidence = ml_confidence
                obj.confirmed_by_user = confirmed_by_user
            else:
                obj = TableStructure(
                    contest=contest,
                    headers=clean_for_json(headers),
                    context=clean_for_json(context),
                    ml_confidence=ml_confidence,
                    confirmed_by_user=confirmed_by_user
                )
                session.add(obj)
            session.commit()
    except SQLAlchemyError as e:
        logger.error(f"[DB][TableStructure] Error saving: {e}")
        raise

def get_table_structure_from_db(contest, context=None) -> dict:
    """
    Retrieve the best-matching table structure for a contest using SQLAlchemy ORM.
    """
    try:
        with get_session() as session:
            row = session.execute(
                select(TableStructure).where(TableStructure.contest == contest)
                .order_by(TableStructure.confirmed_by_user.desc(), TableStructure.ml_confidence.desc())
                .limit(1)
            ).scalar_one_or_none()
        if row:
            headers = robust_orjson_loads(row.headers)
            context = robust_orjson_loads(row.context)
            ml_confidence = row.ml_confidence
            return {"headers": headers, "context": context, "ml_confidence": ml_confidence}
        return None
    except SQLAlchemyError as e:
        logger.error(f"[DB][TableStructure] Error loading: {e}")
        return None

def upsert_contest(session, contest_dict, auto_create_related=True) -> None:
    """
    Upsert a contest using SQLAlchemy ORM. Updates if exists, else inserts.
    Handles state/county as relationships robustly.
    """
    
    # --- Resolve state and county relationships ---
    state_name = contest_dict.get("state")
    county_name = contest_dict.get("county")

    # Use get_or_create helpers for robust linking
    state_obj = get_or_create_state(session, state_name) if auto_create_related else session.query(State).filter_by(name=state_name).first()
    county_obj = get_or_create_county(session, county_name, state_obj) if auto_create_related else session.query(County).filter_by(name=county_name, state=state_obj).first()

    # Find the state and county objects by name
    state_obj = session.query(State).filter(State.name == state_name).first() if state_name else None
    county_obj = session.query(County).filter(
        County.name == county_name,
        County.state == state_obj  # Ensure county is in the correct state
    ).first() if county_name and state_obj else None

    # Optionally, create if not found (uncomment if you want auto-create)
    # if not state_obj and state_name:
    #     state_obj = State(name=state_name)
    #     session.add(state_obj)
    #     session.flush()
    # if not county_obj and county_name:
    #     county_obj = County(name=county_name, state=state_obj)
    #     session.add(county_obj)
    #     session.flush()

    # Build filters for upsert
    filters = [
        Contest.title == contest_dict.get("title"),
        Contest.year == contest_dict.get("year"),
        Contest.type_ == contest_dict.get("type_"),
        Contest.state_id == (state_obj.id if state_obj else None),
        Contest.county_id == (county_obj.id if county_obj else None),
    ]

    obj = session.execute(
        select(Contest).where(and_(*filters))
    ).scalar_one_or_none()

    if obj:
        obj = session.merge(obj)
        obj.election_types = contest_dict.get("election_types")
        obj.metastats = clean_for_json(contest_dict)
    else:
        obj = Contest(
            title=contest_dict.get("title"),
            year=contest_dict.get("year"),
            type_=contest_dict.get("type_"),
            election_types=contest_dict.get("election_types"),
            state=state_obj,
            county=county_obj,
            metastats=clean_for_json(contest_dict)
        )
        session.add(obj)
        
def get_or_create_state(session, state_name) -> Optional[State]:
    state = session.query(State).filter_by(name=state_name).first()
    if not state and state_name:
        state = State(name=state_name)
        session.add(state)
        session.flush()
    return state

def get_or_create_county(session, county_name, state) -> Optional[County]:
    county = session.query(County).filter_by(name=county_name, state=state).first()
    if not county and county_name and state:
        county = County(name=county_name, state=state)
        session.add(county)
        session.flush()
    return county

def get_or_create_party(session, party_name) -> Optional[Party]:
    party = session.query(Party).filter_by(name=party_name).first()
    if not party and party_name:
        party = Party(name=party_name)
        session.add(party)
        session.flush()
    return party

def fetch_contest_full(session, contest) -> Optional[dict]:
    # contest: ORM object or dict with id
    if isinstance(contest, dict):
        obj = session.query(Contest).filter_by(id=contest.get("id")).first()
    else:
        obj = contest
    if not obj:
        return None
    return {
        "id": obj.id,
        "title": obj.title,
        "year": obj.year,
        "type_": obj.type_,
        "state": obj.state.name if obj.state else None,
        "county": obj.county.name if obj.county else None,
        "office": obj.office.name if obj.office else None,
        "candidates": [c.name for c in getattr(obj, "candidates", [])],
        "results": [
            {
                "candidate": r.candidate.name if r.candidate else None,
                "votes": r.votes,
                "percent": r.percent,
                "is_winner": r.is_winner,
            }
            for r in getattr(obj, "results", [])
        ],
        "metastats": obj.metastats,
    }
def check_missing_tables(self):
    """Return a list of expected tables that are missing in the DB."""
    engine = get_engine()
    inspector = inspect(engine)
    db_tables = set(inspector.get_table_names())
    expected_tables = set(Base.metadata.tables.keys())
    return list(expected_tables - db_tables)