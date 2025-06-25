import os
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, List
from sqlalchemy import create_engine, update, select, and_, or_, desc
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from contextlib import contextmanager
from .models import Contest, TableStructure, BatchMetadata, StagingElectionResult, WarehouseElectionResult, Base
from ..config import POSTGRES_URL, CONTEXT_LIBRARY_PATH

# Set up SQLAlchemy engine and session
engine = create_engine(POSTGRES_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
_engine = None  # for lazy initialization if needed

# --- Robust session context manager ---
@contextmanager
def get_session():
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

def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(POSTGRES_URL, echo=False, future=True)
    return _engine

# --- DB Path Safety (for legacy compatibility, not used for SQLAlchemy) ---
def _safe_db_path(path):
    return str(Path(path or CONTEXT_LIBRARY_PATH).resolve())

# --- Contest Operations ---
def update_contest_in_db(contest: dict, session: Optional[Session] = None):
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
            db_obj.type = contest.get("type")
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
                "type": row.type,
                "state": row.state,
                "county": row.county,
                **(row.metadata or {})
            }
            contests.append(contest)
        return contests
    finally:
        if close_session:
            session.close()

def append_to_context_library(data, path=None):
    from ..utils.shared_logic import load_context_library
    if path is None:
        path = CONTEXT_LIBRARY_PATH
    safe_path = _safe_db_path(path)
    library = load_context_library(safe_path)
    with open(safe_path, "w", encoding="utf-8") as f:
        json.dump(library, f, indent=2, ensure_ascii=False)

def normalize_label(label):
    if not label:
        return ""
    return re.sub(r"\W+", "", str(label).strip().lower())

# --- Utility: Processed URL cache (unchanged, not DB) ---
def load_processed_urls() -> Dict[str, Any]:
    from ..utils.output_utils import CACHE_FILE
    cache_path = Path(CACHE_FILE).resolve()
    if not cache_path.exists() or os.path.getsize(cache_path) == 0:
        return {}
    with cache_path.open('r', encoding="utf-8") as f:
        try:
            entries = json.load(f)
            if not isinstance(entries, list):
                entries = []
        except Exception:
            entries = []
    processed = {}
    for entry in entries:
        url = entry.get("url")
        if url:
            processed[url] = entry
    return processed

def load_output_cache(path=None):
    if path is None:
        from ..Context_Integration.context_organizer import OUTPUT_CACHE
        path = OUTPUT_CACHE
    safe_path = Path(_safe_db_path(path)).resolve()
    if not safe_path.exists():
        return []
    with open(safe_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

# --- Utility: Create all tables (run once at startup or migration) ---
def create_all_tables():
    Base.metadata.create_all(engine)

# --- BatchMetadata Operations ---
def create_batch_metadata(source: str, status: str = 'pending') -> BatchMetadata:
    with get_session() as session:
        batch = BatchMetadata(source=source, status=status)
        session.add(batch)
        session.flush()  # get batch_id
        return batch

def update_batch_metadata(batch_id, **kwargs):
    with get_session() as session:
        batch = session.get(BatchMetadata, batch_id)
        if batch:
            for k, v in kwargs.items():
                setattr(batch, k, v)
            session.commit()
        return batch

def get_batch_metadata(batch_id):
    with get_session() as session:
        return session.get(BatchMetadata, batch_id)

# --- StagingElectionResult Operations ---
def create_staging_election_result(**kwargs) -> StagingElectionResult:
    with get_session() as session:
        result = StagingElectionResult(**kwargs)
        session.add(result)
        session.flush()
        return result

def get_staging_results_by_batch(batch_id):
    with get_session() as session:
        return session.query(StagingElectionResult).filter_by(batch_id=batch_id).all()

# --- WarehouseElectionResult Operations ---
def create_warehouse_election_result(**kwargs) -> WarehouseElectionResult:
    with get_session() as session:
        result = WarehouseElectionResult(**kwargs)
        session.add(result)
        session.flush()
        return result

def get_warehouse_results_by_batch(batch_id):
    with get_session() as session:
        return session.query(WarehouseElectionResult).filter_by(batch_id=batch_id).all()

def create_table_structure(contest_title, headers, context, ml_confidence=None, confirmed_by_user=False):
    with get_session() as session:
        obj = TableStructure(
            contest_title=contest_title,
            headers=headers,
            context=context,
            ml_confidence=ml_confidence,
            confirmed_by_user=confirmed_by_user
        )
        session.add(obj)
        session.flush()
        return obj

def update_table_structure(table_id, **kwargs):
    with get_session() as session:
        obj = session.get(TableStructure, table_id)
        if obj:
            for k, v in kwargs.items():
                setattr(obj, k, v)
            session.commit()
        return obj

def get_table_structure_by_id(table_id):
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

def update_table_structure_fields(table_id, fields: dict):
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

def select_table_structures_by_title(title: str, limit: int = 10):
    """
    Use SQLAlchemy's select construct to fetch TableStructures by contest_title.
    """
    with get_session() as session:
        stmt = select(TableStructure).where(TableStructure.contest_title == title).limit(limit)
        return session.execute(stmt).scalars().all()