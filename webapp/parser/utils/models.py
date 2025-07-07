from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    Text,
    JSON,
    ForeignKey,
    Boolean,
    Float,
    LargeBinary,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, relationship
import uuid
from datetime import datetime, timezone

# Use the engine/session from db_utils for all DB operations
try:
    from .db_utils import get_engine, SessionLocal
except ImportError:
    get_engine = None
    SessionLocal = None

Base = declarative_base()

# --- Core Election Entity Relationship Schema ---

class Entity(Base):
    __tablename__ = "entities"
    id = Column(Integer, primary_key=True)
    entity_type = Column(String, nullable=False)
    value = Column(String, nullable=False)

class Party(Base):
    __tablename__ = "parties"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    abbreviation = Column(String)
    candidates = relationship("Candidate", back_populates="party")

class State(Base):
    __tablename__ = "states"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    abbreviation = Column(String)
    counties = relationship("County", back_populates="state")
    districts = relationship("District", back_populates="state")

class County(Base):
    __tablename__ = "counties"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    state_id = Column(Integer, ForeignKey("states.id"))
    state = relationship("State", back_populates="counties")
    contests = relationship("Contest", back_populates="county")

class District(Base):
    __tablename__ = "districts"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    type_ = Column(String)
    state_id = Column(Integer, ForeignKey("states.id"))
    state = relationship("State", back_populates="districts")
    candidates = relationship("Candidate", back_populates="district")
    contests = relationship("Contest", back_populates="district")

class Office(Base):
    __tablename__ = "offices"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    level = Column(String)
    candidates = relationship("Candidate", back_populates="office")
    contests = relationship("Contest", back_populates="office")

class Candidate(Base):
    __tablename__ = "candidates"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    party_id = Column(Integer, ForeignKey("parties.id"))
    party = relationship("Party", back_populates="candidates")
    district_id = Column(Integer, ForeignKey("districts.id"))
    district = relationship("District", back_populates="candidates")
    office_id = Column(Integer, ForeignKey("offices.id"))
    office = relationship("Office", back_populates="candidates")
    results = relationship("Result", back_populates="candidate")

class Contest(Base):
    __tablename__ = "contests"
    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    election_types = Column(String)
    year = Column(Integer)
    type_ = Column("type_", String)  # Avoid using 'type' as it's a reserved keyword
    state_id = Column(Integer, ForeignKey("states.id"))
    state = relationship("State")
    county_id = Column(Integer, ForeignKey("counties.id"))
    county = relationship("County", back_populates="contests")
    district_id = Column(Integer, ForeignKey("districts.id"))
    district = relationship("District", back_populates="contests")
    office_id = Column(Integer, ForeignKey("offices.id"))
    office = relationship("Office", back_populates="contests")
    results = relationship("Result", back_populates="contest")

class Result(Base):
    __tablename__ = "results"
    id = Column(Integer, primary_key=True)
    candidate_id = Column(Integer, ForeignKey("candidates.id"))
    candidate = relationship("Candidate", back_populates="results")
    contest_id = Column(Integer, ForeignKey("contests.id"))
    contest = relationship("Contest", back_populates="results")
    votes = Column(Integer)
    percent = Column(Float)
    is_winner = Column(Boolean)
    is_incumbent = Column(Boolean)
    vote_method = Column(String)

# --- Optional: MiscEntity for generic/legacy entities ---
class MiscEntity(Base):
    __tablename__ = "misc_entities"
    id = Column(Integer, primary_key=True)
    value = Column(String, nullable=False)
    type_ = Column(String, nullable=False)

# --- ML, Logging, and Legacy Support Models ---

class TableStructure(Base):
    __tablename__ = 'table_structures'
    id = Column(Integer, primary_key=True)
    contest_title = Column(String, nullable=False, index=True)
    headers = Column(Text, nullable=False)
    context = Column(Text, nullable=False)
    confirmed_by_user = Column(Boolean, default=False)
    ml_confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.now(timezone.utc))

    def __repr__(self):
        return f"<TableStructure(id={self.id}, contest_title={self.contest_title})>"

class BatchMetadata(Base):
    __tablename__ = 'batch_metadata'
    batch_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source = Column(String)
    started_at = Column(DateTime, default=datetime.now(timezone.utc))
    completed_at = Column(DateTime)
    status = Column(String)

    def __repr__(self):
        return f"<BatchMetadata(batch_id={self.batch_id}, source={self.source}, status={self.status})>"

class StagingElectionResult(Base):
    __tablename__ = 'staging_election_results'
    id = Column(Integer, primary_key=True)
    batch_id = Column(UUID(as_uuid=True), ForeignKey('batch_metadata.batch_id'), nullable=False)
    state = Column(String)
    county = Column(String)
    source_url = Column(String)
    raw_html = Column(Text)
    parsed_at = Column(DateTime, default=datetime.now(timezone.utc))
    status = Column(String, default='pending')

    def __repr__(self):
        return f"<StagingElectionResult(id={self.id}, batch_id={self.batch_id}, state={self.state})>"

class WarehouseElectionResult(Base):
    __tablename__ = 'warehouse_election_results'
    id = Column(Integer, primary_key=True)
    batch_id = Column(UUID(as_uuid=True), ForeignKey('batch_metadata.batch_id'), nullable=False)
    state = Column(String)
    county = Column(String)
    contest_title = Column(String)
    candidate = Column(String)
    party = Column(String)
    votes = Column(Integer)
    precinct = Column(String)
    election_date = Column(DateTime)
    processed_at = Column(DateTime, default=datetime.now(timezone.utc))

    def __repr__(self):
        return f"<WarehouseElectionResult(id={self.id}, contest_title={self.contest_title}, candidate={self.candidate})>"

class EmbeddingCache(Base):
    __tablename__ = 'embeddings'
    segment_hash = Column(String, primary_key=True)
    embedding = Column(LargeBinary)

    def __repr__(self):
        return f"<EmbeddingCache(segment_hash={self.segment_hash})>"

class Alert(Base):
    __tablename__ = 'alerts'
    id = Column(Integer, primary_key=True)
    level = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    context = Column(JSON)
    created_at = Column(DateTime, default=datetime.now(timezone.utc))

    def __repr__(self):
        return f"<Alert(id={self.id}, level={self.level})>"

# --- Session/context manager utility ---

def get_session():
    """
    Context manager for SQLAlchemy session.
    Usage:
        with get_session() as session:
            ...
    """
    if SessionLocal is None:
        raise RuntimeError("SessionLocal is not available. Check db_utils import.")
    from contextlib import contextmanager
    @contextmanager
    def _session_scope():
        session = SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    return _session_scope()

# --- Utility function for table creation ---

def main():
    """
    Create all tables in the configured database.
    """
    if get_engine is None:
        print("[MODELS][ERROR] get_engine not available. Cannot create tables.")
        return
    from sqlalchemy import inspect
    try:
        engine = get_engine()
        print("[MODELS] Creating all tables in the configured database...")
        Base.metadata.create_all(engine)
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        print(f"[MODELS] Tables present after creation: {tables}")
        print("[MODELS] All tables created successfully.")
    except Exception as e:
        print(f"[MODELS][ERROR] Failed to create tables: {e}")

if __name__ == "__main__":
    main()