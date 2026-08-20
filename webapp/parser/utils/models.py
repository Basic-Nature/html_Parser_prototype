from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone
from typing import Any, Protocol

# webapp/parser/utils/models.py
# ---------------------------------------------------------------
# Core database models for Smart Elections Parser Webapp
# ---------------------------------------------------------------
from sqlalchemy import (
    JSON,  # portable JSON for all dialects
    Boolean,
    CheckConstraint,
    Column,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.engine.base import Engine
from sqlalchemy.orm import backref, declarative_base, relationship


Base = declarative_base()


# Flexible parser / ML-NLP evidence may evolve without being canonical truth.
#
# PostgreSQL uses JSONB so evidence is queryable and has native equality
# semantics. SQLite and other portability/test dialects continue to use JSON.
#
# IMPORTANT: this type is intentionally limited to the recovered legacy
# observation/context fields. Canonical publication JSON fields have their own
# storage contract and must not be silently swept into this one.
EVIDENCE_JSON = JSON().with_variant(JSONB(), "postgresql")

class MetaDataProtocol(Protocol):
    tables: Any
    def create_all(self, bind: Engine) -> None: ...

class DeclarativeBaseProtocol(Protocol):
    metadata: MetaDataProtocol

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
    metastats = Column(EVIDENCE_JSON, default=dict)
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
    buttons = relationship("Button", back_populates="contest", cascade="all, delete-orphan")
    metastats = Column(EVIDENCE_JSON, default=dict)
    created_at = Column(DateTime, default=datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))
    __table_args__ = (
        UniqueConstraint('title', 'year', 'type_', 'state_id', 'county_id', name='_contest_uc'),
        Index('ix_contest_year', 'title', 'year'),
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
    metastats = Column(EVIDENCE_JSON, default=dict)
    created_at = Column(DateTime, default=datetime.now(timezone.utc))

class Panel(Base):
    """
    Stores extracted or user-confirmed panels from election pages.
    """
    __tablename__ = "panels"
    id = Column(Integer, primary_key=True)
    panel_text = Column(Text, nullable=False, index=True)
    panel_html = Column(Text)
    segment_hash = Column(String, index=True)
    contest_id = Column(Integer, ForeignKey("contests.id"), nullable=True)
    contest = relationship("Contest", backref=backref("panels", cascade="all, delete-orphan"))
    created_at = Column(DateTime, default=datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))
    metastats = Column(EVIDENCE_JSON, default=dict)

class Button(Base):
    __tablename__ = "buttons"
    id = Column(Integer, primary_key=True)
    label = Column(String, nullable=False)
    selector = Column(String, nullable=True)
    contest_id = Column(Integer, ForeignKey("contests.id"), nullable=True)
    is_visible = Column(Boolean, default=True)
    is_clickable = Column(Boolean, default=True)
    source = Column(String, nullable=True)
    metastats = Column(EVIDENCE_JSON, nullable=True)

    contest = relationship("Contest", back_populates="buttons")

class CandidatePanel(Base):
    """
    Stores candidate panels (grouped candidate info) from election pages.
    """
    __tablename__ = "candidate_panels"
    id = Column(Integer, primary_key=True)
    candidate_panel_text = Column(Text, nullable=False, index=True)
    candidate_panel_html = Column(Text)
    year = Column(Integer, index=True)
    type_ = Column(String)
    segment_hash = Column(String, index=True)
    contest_id = Column(Integer, ForeignKey("contests.id"), nullable=True)
    contest = relationship("Contest", backref=backref("candidate_panels", cascade="all, delete-orphan"))
    created_at = Column(DateTime, default=datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))
    metastats = Column(EVIDENCE_JSON, default=dict)

class LocationPanel(Base):
    """
    Stores location panels (jurisdiction info) from election pages.
    """
    __tablename__ = "location_panels"
    id = Column(Integer, primary_key=True)
    location_panel_text = Column(Text, nullable=False, index=True)
    location_panel_html = Column(Text)
    year = Column(Integer, index=True)
    type_ = Column(String)
    segment_hash = Column(String, index=True)
    contest_id = Column(Integer, ForeignKey("contests.id"), nullable=True)
    contest = relationship("Contest", backref=backref("location_panels", cascade="all, delete-orphan"))
    created_at = Column(DateTime, default=datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))
    metastats = Column(EVIDENCE_JSON, default=dict)

class Heading(Base):
    """
    Stores headings (section titles, update info) from election pages.
    """
    __tablename__ = "headings"
    id = Column(Integer, primary_key=True)
    heading_text = Column(Text, nullable=False, index=True)
    heading_html = Column(Text)
    heading_type = Column(String)  # e.g., "last_webpage_update", "content"
    segment_hash = Column(String, index=True)
    contest_id = Column(Integer, ForeignKey("contests.id"), nullable=True)
    contest = relationship("Contest", backref=backref("headings", cascade="all, delete-orphan"))
    created_at = Column(DateTime, default=datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))
    metastats = Column(EVIDENCE_JSON, default=dict)

class BallotType(Base):
    """
    Stores ballot type info (e.g., absentee, provisional) from election pages.
    """
    __tablename__ = "ballot_types"
    id = Column(Integer, primary_key=True)
    ballot_types_text = Column(Text, nullable=False, index=True)
    ballot_types_html = Column(Text)
    year = Column(Integer, index=True)
    type_ = Column(String)
    segment_hash = Column(String, index=True)
    contest_id = Column(Integer, ForeignKey("contests.id"), nullable=True)
    contest = relationship("Contest", backref=backref("ballot_types", cascade="all, delete-orphan"))
    created_at = Column(DateTime, default=datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))
    metastats = Column(EVIDENCE_JSON, default=dict)

class ResultsTimestamp(Base):
    """
    Stores results timestamp info (last updated, etc.) from election pages.
    """
    __tablename__ = "results_timestamps"
    id = Column(Integer, primary_key=True)
    timestamp_text = Column(Text, nullable=False, index=True)
    timestamp_html = Column(Text)
    segment_hash = Column(String, index=True)
    contest_id = Column(Integer, ForeignKey("contests.id"), nullable=True)
    contest = relationship("Contest", backref=backref("results_timestamps", cascade="all, delete-orphan"))
    created_at = Column(DateTime, default=datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))
    metastats = Column(EVIDENCE_JSON, default=dict)

class PartyLabel(Base):
    """
    Stores party label info from election pages.
    """
    __tablename__ = "party_labels"
    id = Column(Integer, primary_key=True)
    party_label_text = Column(Text, nullable=False, index=True)
    party_label_html = Column(Text)
    segment_hash = Column(String, index=True)
    contest_id = Column(Integer, ForeignKey("contests.id"), nullable=True)
    contest = relationship("Contest", backref=backref("party_labels", cascade="all, delete-orphan"))
    created_at = Column(DateTime, default=datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))
    metastats = Column(EVIDENCE_JSON, default=dict)

class VoteMethod(Base):
    """
    Stores vote method info (e.g., in-person, mail-in) from election pages.
    """
    __tablename__ = "vote_methods"
    id = Column(Integer, primary_key=True)
    vote_method_text = Column(Text, nullable=False, index=True)
    vote_method_html = Column(Text)
    segment_hash = Column(String, index=True)
    contest_id = Column(Integer, ForeignKey("contests.id"), nullable=True)
    contest = relationship("Contest", backref=backref("vote_methods", cascade="all, delete-orphan"))
    created_at = Column(DateTime, default=datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))
    metastats = Column(EVIDENCE_JSON, default=dict)

# --- OPTIONAL/GENERIC MODELS ---

class Entity(Base):
    """
    Generic entity for extensibility.
    """
    __tablename__ = "entities"
    id = Column(Integer, primary_key=True)
    entity_type = Column(String, nullable=False)
    value = Column(String, nullable=False)
    metastats = Column(EVIDENCE_JSON, default=dict)

class MiscEntity(Base):
    """
    Miscellaneous or legacy entity.
    """
    __tablename__ = "misc_entities"
    id = Column(Integer, primary_key=True)
    value = Column(String, nullable=False)
    type_ = Column(String, nullable=False)
    metastats = Column(EVIDENCE_JSON, default=dict)

# --- ML, LOGGING, AND SUPPORT MODELS ---

class TableStructure(Base):
    """
    Stores ML-inferred or user-confirmed table structures.
    """
    __tablename__ = 'table_structures'
    id = Column(Integer, primary_key=True)
    contest = Column(String, nullable=False, index=True)
    headers = Column(Text, nullable=False)
    context = Column(Text, nullable=False)
    confirmed_by_user = Column(Boolean, default=False)
    ml_confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.now(timezone.utc))

    def __repr__(self):
        return f"<TableStructure(id={self.id}, contest={self.contest})>"

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
    metastats = Column(EVIDENCE_JSON, default=dict)

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
    metastats = Column(EVIDENCE_JSON, default=dict)

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
    contest = Column(String)
    candidate = Column(String)
    party = Column(String)
    votes = Column(Integer)
    precinct = Column(String)
    election_date = Column(DateTime)
    processed_at = Column(DateTime, default=datetime.now(timezone.utc))
    verification_status = Column(String(16), default='unverified')  # unverified, pending, verified, rejected
    source_url = Column(String(2048), nullable=True)  # Track which URL produced this data
    source_principal = Column(String(256), nullable=True)  # Who/what added this
    verification_notes = Column(Text, nullable=True)
    verified_at = Column(DateTime, nullable=True)
    verified_by = Column(String(256), nullable=True)
    metastats = Column(EVIDENCE_JSON, default=dict)

    def __repr__(self):
        return f"<WarehouseElectionResult(id={self.id}, contest={self.contest}, candidate={self.candidate})>"


# --- CANONICAL VERIFIED ELECTION DATA ---
#
# The warehouse table above is an import/evidence layer. The canonical
# tables below are the publication boundary consumed by Data Framework.
#
# DL1 and DL2 are independent comparison lanes. They are NOT additive
# result stores. For each production race, QA selected exactly one lane,
# recorded as selected_dl_source, and Database-Lite contains the finalized
# production payload for that race.


class CanonicalSourceArtifact(Base):
    """Immutable source-artifact identity used by canonical seed provenance."""

    __tablename__ = "canonical_source_artifacts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    artifact_role = Column(String(64), nullable=False)
    filename = Column(String(512), nullable=False)
    sha256 = Column(String(64), nullable=False)
    row_count = Column(Integer, nullable=True)
    race_count = Column(Integer, nullable=True)
    imported_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    provenance = Column(JSON, default=dict, nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "sha256",
            name="uq_canonical_source_artifact_sha256",
        ),
        Index(
            "ix_canonical_source_artifacts_role",
            "artifact_role",
        ),
    )


class CanonicalElectionRace(Base):
    """One QA-approved production race."""

    __tablename__ = "canonical_election_races"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_race_id = Column(String(64), nullable=False)
    election_year = Column(Integer, nullable=False)
    election_date = Column(Date, nullable=True)
    date_precision = Column(String(16), default="year", nullable=False)
    state = Column(String(64), nullable=False)
    contest = Column(String(128), nullable=False)
    office_basic = Column(String(64), nullable=True)
    production_status = Column(
        String(32),
        default="prod_loaded",
        nullable=False,
    )
    selected_dl_source = Column(String(3), nullable=False)
    source_url = Column(String(2048), nullable=True)
    verification_status = Column(
        String(32),
        default="pending",
        nullable=False,
    )
    verified_at = Column(DateTime(timezone=True), nullable=True)
    payload_artifact_id = Column(
        UUID(as_uuid=True),
        ForeignKey("canonical_source_artifacts.id"),
        nullable=False,
    )
    approval_artifact_id = Column(
        UUID(as_uuid=True),
        ForeignKey("canonical_source_artifacts.id"),
        nullable=False,
    )
    qa_metadata = Column(JSON, default=dict, nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "source_race_id",
            name="uq_canonical_race_source_race_id",
        ),
        CheckConstraint(
            "selected_dl_source IN ('DL1', 'DL2')",
            name="ck_canonical_race_selected_dl",
        ),
        CheckConstraint(
            "date_precision IN ('year', 'date')",
            name="ck_canonical_race_date_precision",
        ),
        Index(
            "ix_canonical_races_year_state",
            "election_year",
            "state",
        ),
        Index(
            "ix_canonical_races_selected_dl",
            "selected_dl_source",
        ),
        Index(
            "ix_canonical_races_verification",
            "verification_status",
        ),
    )


class CanonicalElectionResult(Base):
    """One canonical candidate/choice result within a production race."""

    __tablename__ = "canonical_election_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    race_id = Column(
        UUID(as_uuid=True),
        ForeignKey(
            "canonical_election_races.id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )
    source_row_index = Column(Integer, nullable=False)
    source_row_hash = Column(String(64), nullable=False)
    source_jurisdiction_label = Column(String(256), nullable=False)
    jurisdiction_key = Column(String(384), nullable=False)
    jurisdiction_name = Column(String(256), nullable=False)
    jurisdiction_type = Column(String(32), nullable=True)
    aggregation_scope = Column(
        String(32),
        default="jurisdiction",
        nullable=False,
    )
    precinct = Column(String(256), nullable=True)
    ballot_candidate_name = Column(String(512), nullable=True)
    candidate = Column(String(512), nullable=False)
    ballot_party = Column(String(128), nullable=True)
    party = Column(String(64), nullable=True)
    fec_id = Column(String(64), nullable=True)
    is_write_in = Column(Boolean, default=False, nullable=False)
    total_votes = Column(Integer, nullable=False)
    source_url = Column(String(2048), nullable=True)
    provenance = Column(JSON, default=dict, nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "race_id",
            "source_row_index",
            name="uq_canonical_result_race_source_row_index",
        ),
        UniqueConstraint(
            "race_id",
            "source_row_hash",
            name="uq_canonical_result_race_source_row_hash",
        ),
        CheckConstraint(
            "aggregation_scope IN ('jurisdiction', 'precinct')",
            name="ck_canonical_result_aggregation_scope",
        ),
        Index(
            "ix_canonical_results_race_jurisdiction",
            "race_id",
            "jurisdiction_key",
        ),
        Index(
            "ix_canonical_results_candidate",
            "candidate",
        ),
        Index(
            "ix_canonical_results_fec_id",
            "fec_id",
        ),
    )


class CanonicalVoteComponent(Base):
    """Normalized vote-method component for one canonical result.

    Signed values are allowed because historical source evidence contains
    at least one internally consistent negative adjustment.
    """

    __tablename__ = "canonical_vote_components"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    result_id = Column(
        UUID(as_uuid=True),
        ForeignKey(
            "canonical_election_results.id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )
    vote_method = Column(String(32), nullable=False)
    votes = Column(Integer, nullable=False)
    source_column = Column(String(128), nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "result_id",
            "vote_method",
            name="uq_canonical_vote_component_method",
        ),
        Index(
            "ix_canonical_vote_components_result",
            "result_id",
        ),
    )


class CanonicalVerificationEvent(Base):
    """Race-level QA / production provenance event."""

    __tablename__ = "canonical_verification_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    race_id = Column(
        UUID(as_uuid=True),
        ForeignKey(
            "canonical_election_races.id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )
    stage = Column(String(64), nullable=False)
    status = Column(String(64), nullable=False)
    selected_dl_source = Column(String(3), nullable=True)
    actor = Column(String(256), nullable=True)
    occurred_at = Column(DateTime(timezone=True), nullable=True)
    notes = Column(Text, nullable=True)
    event_metadata = Column(JSON, default=dict, nullable=False)

    __table_args__ = (
        CheckConstraint(
            (
                "selected_dl_source IS NULL OR "
                "selected_dl_source IN ('DL1', 'DL2')"
            ),
            name="ck_canonical_verification_selected_dl",
        ),
        Index(
            "ix_canonical_verification_race_stage",
            "race_id",
            "stage",
        ),
    )


class DataFrameworkPreviewCache(Base):
    """
    Temporary preview cache for Data Framework UI sampling.
    """
    __tablename__ = "data_framework_preview_cache"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(String(128), index=True)
    mode = Column(String(24), default="idle", index=True)
    state = Column(String)
    county = Column(String)
    contest = Column(String)
    year = Column(Integer)
    source = Column(String(64), default="warehouse")
    payload = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.now(timezone.utc))
    last_accessed = Column(DateTime, default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))
    expires_at = Column(DateTime, index=True)

    def __repr__(self):
        return f"<DataFrameworkPreviewCache(id={self.id}, mode={self.mode}, state={self.state}, county={self.county})>"

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
    context = Column(EVIDENCE_JSON)
    created_at = Column(DateTime, default=datetime.now(timezone.utc))

    def __repr__(self):
        return f"<Alert(id={self.id}, level={self.level})>"
