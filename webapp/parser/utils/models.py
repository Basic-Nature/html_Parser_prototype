from sqlalchemy import (
    Column, Integer, String, DateTime, Text, JSON, ForeignKey, Boolean, Float, LargeBinary,
    UniqueConstraint, Index, Enum
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, relationship, backref
import uuid
from datetime import datetime, timezone
import enum

Base = declarative_base()

# --- ENUMS ---

class ElectionTypeEnum(enum.Enum):
    GENERAL = "general"
    PRIMARY = "primary"
    SPECIAL = "special"
    RUNOFF = "runoff"

class OfficeLevelEnum(enum.Enum):
    FEDERAL = "federal"
    STATE = "state"
    COUNTY = "county"
    LOCAL = "local"

class StatusEnum(enum.Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    ERROR = "error"

# --- CORE MODELS ---

class State(Base):
    """
    US State or territory.
    """
    __tablename__ = "states"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False, index=True)
    abbreviation = Column(String, unique=True)
    counties = relationship("County", back_populates="state", cascade="all, delete-orphan")
    districts = relationship("District", back_populates="state", cascade="all, delete-orphan")
    contests = relationship("Contest", back_populates="state", cascade="all, delete-orphan")

class County(Base):
    """
    County or equivalent jurisdiction.
    """
    __tablename__ = "counties"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, index=True)
    state_id = Column(Integer, ForeignKey("states.id"), nullable=False, index=True)
    state = relationship("State", back_populates="counties")
    contests = relationship("Contest", back_populates="county", cascade="all, delete-orphan")
    districts = relationship("District", back_populates="county", cascade="all, delete-orphan")
    __table_args__ = (UniqueConstraint('name', 'state_id', name='_county_state_uc'),)

class District(Base):
    """
    Congressional, legislative, or local district.
    """
    __tablename__ = "districts"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    type_ = Column(String)
    state_id = Column(Integer, ForeignKey("states.id"), nullable=False)
    state = relationship("State", back_populates="districts")
    county_id = Column(Integer, ForeignKey("counties.id"), nullable=True)
    county = relationship("County", back_populates="districts")
    candidates = relationship("Candidate", back_populates="district")
    contests = relationship("Contest", back_populates="district")

class Office(Base):
    """
    Elected office (e.g., President, Governor, Mayor).
    """
    __tablename__ = "offices"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, index=True)
    level = Column(Enum(OfficeLevelEnum), index=True)
    candidates = relationship("Candidate", back_populates="office")
    contests = relationship("Contest", back_populates="office")

class Party(Base):
    """
    Political party.
    """
    __tablename__ = "parties"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    abbreviation = Column(String)
    candidates = relationship("Candidate", back_populates="party")

class Candidate(Base):
    """
    Candidate for office.
    """
    __tablename__ = "candidates"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, index=True)
    party_id = Column(Integer, ForeignKey("parties.id"))
    party = relationship("Party", back_populates="candidates")
    district_id = Column(Integer, ForeignKey("districts.id"))
    district = relationship("District", back_populates="candidates")
    office_id = Column(Integer, ForeignKey("offices.id"))
    office = relationship("Office", back_populates="candidates")
    results = relationship("Result", back_populates="candidate")
    metadata = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))

class Contest(Base):
    """
    Election contest (race).
    """
    __tablename__ = "contests"
    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False, index=True)
    election_types = Column(String)  # Or use Enum(ElectionTypeEnum)
    year = Column(Integer, index=True)
    type_ = Column("type_", String)
    state_id = Column(Integer, ForeignKey("states.id"), index=True)
    state = relationship("State", back_populates="contests")
    county_id = Column(Integer, ForeignKey("counties.id"), index=True)
    county = relationship("County", back_populates="contests")
    district_id = Column(Integer, ForeignKey("districts.id"))
    district = relationship("District", back_populates="contests")
    office_id = Column(Integer, ForeignKey("offices.id"))
    office = relationship("Office", back_populates="contests")
    results = relationship("Result", back_populates="contest")
    metadata = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))
    __table_args__ = (
        UniqueConstraint('title', 'year', 'type_', 'state_id', 'county_id', name='_contest_uc'),
        Index('ix_contest_title_year', 'title', 'year'),
    )

class Result(Base):
    """
    Result for a candidate in a contest.
    """
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
    metadata = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.now(timezone.utc))

# --- OPTIONAL/GENERIC MODELS ---

class Entity(Base):
    """
    Generic entity for extensibility.
    """
    __tablename__ = "entities"
    id = Column(Integer, primary_key=True)
    entity_type = Column(String, nullable=False)
    value = Column(String, nullable=False)
    metadata = Column(JSON, default=dict)

class MiscEntity(Base):
    """
    Miscellaneous or legacy entity.
    """
    __tablename__ = "misc_entities"
    id = Column(Integer, primary_key=True)
    value = Column(String, nullable=False)
    type_ = Column(String, nullable=False)
    metadata = Column(JSON, default=dict)

# --- ML, LOGGING, AND SUPPORT MODELS ---

class TableStructure(Base):
    """
    Stores ML-inferred or user-confirmed table structures.
    """
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
    """
    Metadata for batch processing.
    """
    __tablename__ = 'batch_metadata'
    batch_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source = Column(String)
    started_at = Column(DateTime, default=datetime.now(timezone.utc))
    completed_at = Column(DateTime)
    status = Column(Enum(StatusEnum), default=StatusEnum.PENDING)
    metadata = Column(JSON, default=dict)

    def __repr__(self):
        return f"<BatchMetadata(batch_id={self.batch_id}, source={self.source}, status={self.status})>"

class StagingElectionResult(Base):
    """
    Raw parsed election results before normalization.
    """
    __tablename__ = 'staging_election_results'
    id = Column(Integer, primary_key=True)
    batch_id = Column(UUID(as_uuid=True), ForeignKey('batch_metadata.batch_id'), nullable=False)
    state = Column(String)
    county = Column(String)
    source_url = Column(String)
    raw_html = Column(Text)
    parsed_at = Column(DateTime, default=datetime.now(timezone.utc))
    status = Column(Enum(StatusEnum), default=StatusEnum.PENDING)
    metadata = Column(JSON, default=dict)

    def __repr__(self):
        return f"<StagingElectionResult(id={self.id}, batch_id={self.batch_id}, state={self.state})>"

class WarehouseElectionResult(Base):
    """
    Normalized, warehouse-ready election results.
    """
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
    metadata = Column(JSON, default=dict)

    def __repr__(self):
        return f"<WarehouseElectionResult(id={self.id}, contest_title={self.contest_title}, candidate={self.candidate})>"

class EmbeddingCache(Base):
    """
    Stores ML embeddings for text segments.
    """
    __tablename__ = 'embeddings'
    segment_hash = Column(String, primary_key=True)
    embedding = Column(LargeBinary)
    created_at = Column(DateTime, default=datetime.now(timezone.utc))

    def __repr__(self):
        return f"<EmbeddingCache(segment_hash={self.segment_hash})>"

class Alert(Base):
    """
    System or user alerts.
    """
    __tablename__ = 'alerts'
    id = Column(Integer, primary_key=True)
    level = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    context = Column(JSON)
    created_at = Column(DateTime, default=datetime.now(timezone.utc))

    def __repr__(self):
        return f"<Alert(id={self.id}, level={self.level})>"

# --- SESSION/UTILITY ---

def get_session():
    """
    Context manager for SQLAlchemy session.
    Usage:
        with get_session() as session:
            ...
    """
    try:
        from .db_utils import SessionLocal
    except ImportError:
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

def main():
    """
    Create all tables in the configured database.
    """
    try:
        from .db_utils import get_engine
    except ImportError:
        raise RuntimeError("get_engine not available. Cannot create tables.")
    from .shared_logger import log_info, log_error
    from sqlalchemy import inspect
    try:
        engine = get_engine()
        log_info("[MODELS] Creating all tables in the configured database...")
        Base.metadata.create_all(engine)
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        log_info(f"[MODELS] Tables present after creation: {tables}")
        log_info("[MODELS] All tables created successfully.")
    except Exception as e:
        log_error(f"[MODELS][ERROR] Failed to create tables: {e}")

if __name__ == "__main__":
    main()