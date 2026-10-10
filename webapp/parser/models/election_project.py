"""J1 private investigation context; not a Source Registry/Workflow authority."""
from __future__ import annotations
import uuid
from datetime import datetime, timezone
from sqlalchemy import CheckConstraint, Column, Date, DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint, Uuid
from webapp.parser.utils.models import Base

def _utcnow():
    return datetime.now(timezone.utc)

class ElectionProject(Base):
    __tablename__ = "election_projects"
    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner_key = Column(String(64), nullable=False)  # opaque keyed digest, never raw credential
    creation_key = Column(Uuid(as_uuid=True), nullable=False)
    title = Column(String(160), nullable=False)
    description = Column(Text, nullable=False, default="")
    lifecycle = Column(String(16), nullable=False, default="active")
    row_version = Column(Integer, nullable=False, default=1)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    __table_args__ = (
        UniqueConstraint("owner_key", "creation_key", name="uq_election_project_create"),
        CheckConstraint("lifecycle IN ('active','archived')", name="ck_election_project_lifecycle"),
        CheckConstraint("row_version >= 1", name="ck_election_project_version"),
        Index("ix_election_projects_owner_update", "owner_key", "updated_at"),
    )

class ElectionProjectScope(Base):
    __tablename__ = "election_project_scopes"
    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(Uuid(as_uuid=True), ForeignKey("election_projects.id", ondelete="CASCADE"), nullable=False)
    election_year = Column(Integer, nullable=True)
    election_date = Column(Date, nullable=True)
    state_code = Column(String(2), nullable=True)
    jurisdiction = Column(String(160), nullable=True)
    contest = Column(String(200), nullable=True)
    granularity = Column(String(24), nullable=True)
    __table_args__ = (
        CheckConstraint("election_year IS NULL OR (election_year BETWEEN 1788 AND 2100)", name="ck_election_project_scope_year"),
        CheckConstraint("granularity IS NULL OR granularity IN ('statewide','county','municipality','precinct','unknown')", name="ck_election_project_scope_granularity"),
        Index("ix_election_project_scopes_project", "project_id"),
    )

class ElectionProjectSourceRef(Base):
    __tablename__ = "election_project_source_refs"
    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(Uuid(as_uuid=True), ForeignKey("election_projects.id", ondelete="CASCADE"), nullable=False)
    # Source Registry is registered under SourceRegistryBase.metadata, not Base.metadata.
    # The Alembic migration, not the ORM metadata, owns the cross-metadata FKs.
    registry_binding_id = Column(Uuid(as_uuid=True), nullable=False)
    registry_revision_id = Column(Uuid(as_uuid=True), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    __table_args__ = (
        UniqueConstraint("project_id", "registry_binding_id", "registry_revision_id", name="uq_election_project_ref_revision"),
        Index("ix_election_project_refs_project", "project_id"),
    )
