from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from webapp.parser.services.workflow_staging_binding import (
    WORKFLOW_STAGING_BINDING_CONTRACT,
    WorkflowStagingBindingConflict,
    WorkflowStagingBindingError,
    WorkflowStagingBindingSourceRejected,
    begin_workflow_staging_binding,
    finalize_workflow_staging_binding,
    validate_completed_workflow_staging_binding,
)
from webapp.parser.utils.models import (
    Base,
    BatchMetadata,
    StagingElectionResult,
    StatusEnum,
    WorkflowArtifactLink,
    WorkflowEvent,
    WorkflowItem,
    WorkflowPass,
)


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )
    session = Session()
    try:
        yield session
    finally:
        session.rollback()
        session.close()
        engine.dispose()


@pytest.fixture()
def registry(tmp_path: Path) -> Path:
    path = tmp_path / "urls.txt"
    path.write_text(
        "\n".join([
            "# === Curated | test ===",
            (
                "2024\tPresident\tIowa\tstatewide\tPDF\tCertified\t"
                "https://sos.example.gov/results.pdf"
            ),
            "# === Backlog ===",
            (
                "2020\tPresident\tIowa\tstatewide\tPDF\tArchive\t"
                "https://sos.example.gov/archive.pdf"
            ),
        ])
        + "\n",
        encoding="utf-8",
    )
    return path


def _seed_claimed_dl1(
    session,
    *,
    principal: str = "principal:dl1",
    source_url: str = "https://sos.example.gov/results.pdf",
):
    item = WorkflowItem(
        id=uuid4(),
        lifecycle_state="active",
        current_stage="independent_acquisition",
        stage_condition="in_progress",
        priority=0,
        election_year=2024,
        election_date=None,
        state="Iowa",
        jurisdiction_name=None,
        jurisdiction_type=None,
        contest="President",
        office_basic="President",
        election_type=None,
        source_race_id="2024PRESIA",
        source_url=source_url,
        canonical_race_id=None,
        blocked_reason_code=None,
        blocker_detail=None,
        created_by_principal=None,
        workflow_metadata={},
        row_version=2,
    )
    workflow_pass = WorkflowPass(
        id=uuid4(),
        workflow_item_id=item.id,
        pass_number=1,
        pass_label="DL1",
        revision_number=1,
        is_current=True,
        status="in_progress",
        assigned_principal=principal,
        source_evidence_ref=None,
        staging_batch_id=None,
        candidate_check_status=None,
        candidate_check_result=None,
        semantic_validation_status=None,
        semantic_validation_result=None,
    )
    session.add_all([item, workflow_pass])
    session.flush()
    return item, workflow_pass


def _begin(session, registry, *, principal="principal:dl1"):
    item, workflow_pass = _seed_claimed_dl1(
        session,
        principal=principal,
    )
    result = begin_workflow_staging_binding(
        session,
        item.id,
        workflow_pass.id,
        principal=principal,
        registry_path=registry,
        now=datetime(2026, 9, 8, 20, 0, tzinfo=timezone.utc),
    )
    return item, workflow_pass, result


def _add_staging_row(session, batch_id, *, source_url=None):
    row = StagingElectionResult(
        batch_id=batch_id,
        state="Iowa",
        county=None,
        source_url=(
            source_url
            if source_url is not None
            else "https://sos.example.gov/results.pdf"
        ),
        raw_html="<table>fixture</table>",
    )
    session.add(row)
    session.flush()
    return row


def test_begin_binds_pending_batch_to_current_dl1_without_commit(
    db_session,
    registry,
):
    item, workflow_pass, result = _begin(db_session, registry)
    assert result["contract"] == WORKFLOW_STAGING_BINDING_CONTRACT
    assert result["binding_state"] == "pending"
    assert result["committed"] is False

    batch = db_session.get(BatchMetadata, workflow_pass.staging_batch_id)
    assert batch is not None
    assert batch.status == StatusEnum.PENDING
    assert batch.metastats["workflow_item_id"] == str(item.id)
    assert batch.metastats["workflow_pass_id"] == str(workflow_pass.id)
    assert batch.metastats["assigned_principal"] == "principal:dl1"
    assert batch.metastats["exact_source_url"] == item.source_url
    assert batch.metastats["artifact_sha256"] is None

    events = (
        db_session.query(WorkflowEvent)
        .filter(WorkflowEvent.workflow_item_id == item.id)
        .all()
    )
    assert [event.event_type for event in events] == [
        "staging_binding_started"
    ]


def test_begin_rejects_wrong_principal_and_duplicate_binding(
    db_session,
    registry,
):
    item, workflow_pass = _seed_claimed_dl1(db_session)

    with pytest.raises(WorkflowStagingBindingConflict):
        begin_workflow_staging_binding(
            db_session,
            item.id,
            workflow_pass.id,
            principal="principal:other",
            registry_path=registry,
        )

    begin_workflow_staging_binding(
        db_session,
        item.id,
        workflow_pass.id,
        principal="principal:dl1",
        registry_path=registry,
    )
    with pytest.raises(WorkflowStagingBindingConflict):
        begin_workflow_staging_binding(
            db_session,
            item.id,
            workflow_pass.id,
            principal="principal:dl1",
            registry_path=registry,
        )


def test_begin_requires_exact_curated_registry_source(
    db_session,
    registry,
):
    item, workflow_pass = _seed_claimed_dl1(
        db_session,
        source_url="https://sos.example.gov/archive.pdf",
    )
    with pytest.raises(WorkflowStagingBindingSourceRejected):
        begin_workflow_staging_binding(
            db_session,
            item.id,
            workflow_pass.id,
            principal="principal:dl1",
            registry_path=registry,
        )


def test_finalize_requires_rows_exact_source_and_valid_hash(
    db_session,
    registry,
):
    item, workflow_pass, _ = _begin(db_session, registry)
    batch_id = workflow_pass.staging_batch_id

    with pytest.raises(WorkflowStagingBindingConflict):
        finalize_workflow_staging_binding(
            db_session,
            item.id,
            workflow_pass.id,
            batch_id,
            principal="principal:dl1",
            source_evidence_ref="evidence://dl1/source.pdf",
            artifact_ref="staging://dl1/payload.json",
            artifact_sha256="a" * 64,
        )

    _add_staging_row(
        db_session,
        batch_id,
        source_url="https://evil.example/other.pdf",
    )
    with pytest.raises(WorkflowStagingBindingConflict):
        finalize_workflow_staging_binding(
            db_session,
            item.id,
            workflow_pass.id,
            batch_id,
            principal="principal:dl1",
            source_evidence_ref="evidence://dl1/source.pdf",
            artifact_ref="staging://dl1/payload.json",
            artifact_sha256="a" * 64,
        )

    db_session.query(StagingElectionResult).delete()
    _add_staging_row(db_session, batch_id)
    with pytest.raises(WorkflowStagingBindingError):
        finalize_workflow_staging_binding(
            db_session,
            item.id,
            workflow_pass.id,
            batch_id,
            principal="principal:dl1",
            source_evidence_ref="evidence://dl1/source.pdf",
            artifact_ref="staging://dl1/payload.json",
            artifact_sha256="A" * 64,
        )


def test_finalize_freezes_completed_binding_artifact_and_audit_event(
    db_session,
    registry,
):
    item, workflow_pass, _ = _begin(db_session, registry)
    batch_id = workflow_pass.staging_batch_id
    _add_staging_row(db_session, batch_id)

    result = finalize_workflow_staging_binding(
        db_session,
        item.id,
        workflow_pass.id,
        batch_id,
        principal="principal:dl1",
        source_evidence_ref="evidence://dl1/source.pdf",
        artifact_ref="staging://dl1/payload.json",
        artifact_sha256="b" * 64,
        now=datetime(2026, 9, 8, 20, 5, tzinfo=timezone.utc),
    )
    assert result["binding_state"] == "complete"
    assert result["row_count"] == 1
    assert result["committed"] is False

    batch = db_session.get(BatchMetadata, batch_id)
    assert batch.status == StatusEnum.COMPLETED
    assert batch.metastats["artifact_ref"] == "staging://dl1/payload.json"
    assert batch.metastats["artifact_sha256"] == "b" * 64
    assert workflow_pass.source_evidence_ref == "evidence://dl1/source.pdf"
    assert workflow_pass.candidate_check_status is None
    assert workflow_pass.semantic_validation_status is None

    links = (
        db_session.query(WorkflowArtifactLink)
        .filter(WorkflowArtifactLink.pass_id == workflow_pass.id)
        .all()
    )
    assert len(links) == 1
    assert links[0].artifact_sha256 == "b" * 64
    assert links[0].staging_batch_id == batch_id

    events = (
        db_session.query(WorkflowEvent)
        .filter(WorkflowEvent.workflow_item_id == item.id)
        .order_by(WorkflowEvent.occurred_at)
        .all()
    )
    assert [event.event_type for event in events] == [
        "staging_binding_started",
        "staging_binding_completed",
    ]


def test_finalize_rejects_replay_and_does_not_duplicate_artifact(
    db_session,
    registry,
):
    item, workflow_pass, _ = _begin(db_session, registry)
    batch_id = workflow_pass.staging_batch_id
    _add_staging_row(db_session, batch_id)
    finalize_workflow_staging_binding(
        db_session,
        item.id,
        workflow_pass.id,
        batch_id,
        principal="principal:dl1",
        source_evidence_ref="evidence://dl1/source.pdf",
        artifact_ref="staging://dl1/payload.json",
        artifact_sha256="c" * 64,
    )

    with pytest.raises(WorkflowStagingBindingConflict):
        finalize_workflow_staging_binding(
            db_session,
            item.id,
            workflow_pass.id,
            batch_id,
            principal="principal:dl1",
            source_evidence_ref="evidence://dl1/source.pdf",
            artifact_ref="staging://dl1/payload.json",
            artifact_sha256="c" * 64,
        )

    assert (
        db_session.query(WorkflowArtifactLink)
        .filter(WorkflowArtifactLink.pass_id == workflow_pass.id)
        .count()
        == 1
    )


def test_completed_binding_validator_reconciles_all_authorities(
    db_session,
    registry,
):
    item, workflow_pass, _ = _begin(db_session, registry)
    batch_id = workflow_pass.staging_batch_id
    _add_staging_row(db_session, batch_id)
    finalize_workflow_staging_binding(
        db_session,
        item.id,
        workflow_pass.id,
        batch_id,
        principal="principal:dl1",
        source_evidence_ref="evidence://dl1/source.pdf",
        artifact_ref="staging://dl1/payload.json",
        artifact_sha256="d" * 64,
    )

    validated = validate_completed_workflow_staging_binding(
        db_session,
        item.id,
        workflow_pass.id,
        batch_id,
        principal="principal:dl1",
    )
    assert validated["binding_state"] == "complete"
    assert validated["artifact_sha256"] == "d" * 64
    assert validated["row_count"] == 1


def test_completed_binding_validator_rejects_tampered_artifact_identity(
    db_session,
    registry,
):
    item, workflow_pass, _ = _begin(db_session, registry)
    batch_id = workflow_pass.staging_batch_id
    _add_staging_row(db_session, batch_id)
    finalize_workflow_staging_binding(
        db_session,
        item.id,
        workflow_pass.id,
        batch_id,
        principal="principal:dl1",
        source_evidence_ref="evidence://dl1/source.pdf",
        artifact_ref="staging://dl1/payload.json",
        artifact_sha256="e" * 64,
    )

    link = (
        db_session.query(WorkflowArtifactLink)
        .filter(WorkflowArtifactLink.pass_id == workflow_pass.id)
        .one()
    )
    link.artifact_sha256 = "f" * 64
    db_session.flush()

    with pytest.raises(WorkflowStagingBindingConflict):
        validate_completed_workflow_staging_binding(
            db_session,
            item.id,
            workflow_pass.id,
            batch_id,
            principal="principal:dl1",
        )
