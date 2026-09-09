from __future__ import annotations

from datetime import date
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from webapp.parser.contracts.workflow_comparison import semantic_sha256
from webapp.parser.services.workflow_actions import (
    WorkflowSubmitConflict,
    submit_first_workflow_pass,
)
from webapp.parser.services.workflow_pass_corrections import (
    create_dl1_correction_revision,
)
from webapp.parser.services.workflow_pre_qc_validation import (
    validate_first_workflow_pass_pre_qc,
)
from webapp.parser.services.workflow_staging_binding import (
    begin_workflow_staging_binding,
    finalize_workflow_staging_binding,
)
from webapp.parser.utils.models import (
    Base,
    StagingElectionResult,
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
        ])
        + "\n",
        encoding="utf-8",
    )
    return path


def _seed_dl1(session):
    item = WorkflowItem(
        id=uuid4(),
        lifecycle_state="active",
        current_stage="independent_acquisition",
        stage_condition="in_progress",
        priority=0,
        election_year=2024,
        election_date=date(2024, 11, 5),
        state="Iowa",
        jurisdiction_name=None,
        jurisdiction_type=None,
        contest="President",
        office_basic="President",
        election_type=None,
        source_race_id="2024PRESIA",
        source_url="https://sos.example.gov/results.pdf",
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
        assigned_principal="principal:dl1",
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


def _semantic():
    return {
        "scope": {
            "election_year": 2024,
            "election_date": "2024-11-05",
            "state": "Iowa",
            "jurisdiction_name": None,
            "jurisdiction_type": None,
            "contest": "President",
        },
        "records": [{
            "reporting_unit": {"name": "Precinct 1", "type": "precinct"},
            "percent_reporting": {"state": "value", "value": "100"},
            "vote_methods": [
                "Election Day",
                "Early Voting",
                "Absentee Mail",
                "Provisional",
            ],
            "method_totals": [
                {"method": "Election Day", "state": "value", "votes": 11},
                {"method": "Early Voting", "state": "value", "votes": 7},
                {"method": "Absentee Mail", "state": "value", "votes": 2},
                {"method": "Provisional", "state": "value", "votes": 0},
            ],
            "candidates": [
                {
                    "name": "Jane Doe",
                    "party": "DEM",
                    "method_votes": [
                        {"method": "Election Day", "state": "value", "votes": 6},
                        {"method": "Early Voting", "state": "value", "votes": 4},
                        {"method": "Absentee Mail", "state": "value", "votes": 1},
                        {"method": "Provisional", "state": "value", "votes": 0},
                    ],
                    "total_votes": {"state": "value", "votes": 11},
                },
                {
                    "name": "John Smith",
                    "party": "REP",
                    "method_votes": [
                        {"method": "Election Day", "state": "value", "votes": 5},
                        {"method": "Early Voting", "state": "value", "votes": 3},
                        {"method": "Absentee Mail", "state": "value", "votes": 1},
                        {"method": "Provisional", "state": "value", "votes": 0},
                    ],
                    "total_votes": {"state": "value", "votes": 9},
                },
            ],
            "grand_total": {"state": "value", "votes": 20},
        }],
    }


def _bind_and_validate(session, registry, item, workflow_pass, *, hash_char="a"):
    begin_workflow_staging_binding(
        session,
        item.id,
        workflow_pass.id,
        principal="principal:dl1",
        registry_path=registry,
    )
    session.add(
        StagingElectionResult(
            batch_id=workflow_pass.staging_batch_id,
            state="Iowa",
            county=None,
            source_url=item.source_url,
            raw_html="<table>fixture</table>",
        )
    )
    session.flush()
    artifact_ref = (
        f"staging://normalized/dl1-r{workflow_pass.revision_number}.json"
    )
    evidence_ref = (
        f"evidence://dl1/rev{workflow_pass.revision_number}.pdf"
    )
    artifact_sha = hash_char * 64
    finalize_workflow_staging_binding(
        session,
        item.id,
        workflow_pass.id,
        workflow_pass.staging_batch_id,
        principal="principal:dl1",
        source_evidence_ref=evidence_ref,
        artifact_ref=artifact_ref,
        artifact_sha256=artifact_sha,
    )
    semantic = _semantic()
    payload = {
        "schema": "workflow_normalized_semantic_comparison_payload_v1",
        "schema_version": 1,
        "comparison_version": 1,
        "binding": {
            "workflow_item_id": str(item.id),
            "workflow_pass_id": str(workflow_pass.id),
            "pass_number": 1,
            "revision_number": workflow_pass.revision_number,
            "source_evidence_ref": evidence_ref,
            "staging_batch_id": str(workflow_pass.staging_batch_id),
            "normalized_artifact_ref": artifact_ref,
            "normalized_artifact_sha256": artifact_sha,
        },
        "semantic": semantic,
        "semantic_sha256": semantic_sha256(semantic),
    }
    validate_first_workflow_pass_pre_qc(
        session,
        item.id,
        workflow_pass.id,
        workflow_pass.staging_batch_id,
        principal="principal:dl1",
        normalized_payload=payload,
    )
    return {
        "pass_id": str(workflow_pass.id),
        "staging_batch_id": str(workflow_pass.staging_batch_id),
        "source_evidence_ref": evidence_ref,
        "artifact_ref": artifact_ref,
        "artifact_sha256": artifact_sha,
    }


def _submit(session, item, assertions):
    return submit_first_workflow_pass(
        session,
        item.id,
        pass_id=assertions["pass_id"],
        principal="principal:dl1",
        expected_row_version=item.row_version,
        staging_batch_id=assertions["staging_batch_id"],
        source_evidence_ref=assertions["source_evidence_ref"],
        artifact_ref=assertions["artifact_ref"],
        artifact_sha256=assertions["artifact_sha256"],
    )


def test_submit_success_is_atomic_ready_and_audited(db_session, registry):
    item, workflow_pass = _seed_dl1(db_session)
    assertions = _bind_and_validate(db_session, registry, item, workflow_pass)
    original_batch = workflow_pass.staging_batch_id
    original_evidence = workflow_pass.source_evidence_ref
    result = _submit(db_session, item, assertions)
    assert result["status"] == "submitted"
    assert result["committed"] is False
    assert result["row_version"] == 3
    assert workflow_pass.status == "submitted"
    assert workflow_pass.submitted_at is not None
    assert workflow_pass.staging_batch_id == original_batch
    assert workflow_pass.source_evidence_ref == original_evidence
    assert workflow_pass.candidate_check_status == "complete"
    assert workflow_pass.semantic_validation_status == "complete"
    assert item.lifecycle_state == "active"
    assert item.current_stage == "independent_acquisition"
    assert item.stage_condition == "ready"
    events = (
        db_session.query(WorkflowEvent)
        .filter(WorkflowEvent.event_type == "pass_submitted")
        .all()
    )
    assert len(events) == 1
    assert events[0].related_pass_id == workflow_pass.id


def test_submit_requires_completed_server_pre_qc(db_session, registry):
    item, workflow_pass = _seed_dl1(db_session)
    begin_workflow_staging_binding(
        db_session,
        item.id,
        workflow_pass.id,
        principal="principal:dl1",
        registry_path=registry,
    )
    db_session.add(
        StagingElectionResult(
            batch_id=workflow_pass.staging_batch_id,
            state="Iowa",
            county=None,
            source_url=item.source_url,
            raw_html="<table>fixture</table>",
        )
    )
    db_session.flush()
    finalize_workflow_staging_binding(
        db_session,
        item.id,
        workflow_pass.id,
        workflow_pass.staging_batch_id,
        principal="principal:dl1",
        source_evidence_ref="evidence://dl1/rev1.pdf",
        artifact_ref="staging://normalized/dl1-r1.json",
        artifact_sha256="a" * 64,
    )
    with pytest.raises(WorkflowSubmitConflict):
        submit_first_workflow_pass(
            db_session,
            item.id,
            pass_id=workflow_pass.id,
            principal="principal:dl1",
            expected_row_version=2,
            staging_batch_id=workflow_pass.staging_batch_id,
            source_evidence_ref=workflow_pass.source_evidence_ref,
            artifact_ref="staging://normalized/dl1-r1.json",
            artifact_sha256="a" * 64,
        )
    assert workflow_pass.status == "in_progress"
    assert item.stage_condition == "in_progress"


def test_submit_stale_row_version_fails_without_mutation(db_session, registry):
    item, workflow_pass = _seed_dl1(db_session)
    assertions = _bind_and_validate(db_session, registry, item, workflow_pass)
    with pytest.raises(WorkflowSubmitConflict):
        submit_first_workflow_pass(
            db_session,
            item.id,
            pass_id=assertions["pass_id"],
            principal="principal:dl1",
            expected_row_version=99,
            staging_batch_id=assertions["staging_batch_id"],
            source_evidence_ref=assertions["source_evidence_ref"],
            artifact_ref=assertions["artifact_ref"],
            artifact_sha256=assertions["artifact_sha256"],
        )
    assert workflow_pass.status == "in_progress"
    assert item.row_version == 2


def test_submit_client_assertion_mismatch_fails_closed(db_session, registry):
    item, workflow_pass = _seed_dl1(db_session)
    assertions = _bind_and_validate(db_session, registry, item, workflow_pass)
    with pytest.raises(WorkflowSubmitConflict):
        submit_first_workflow_pass(
            db_session,
            item.id,
            pass_id=assertions["pass_id"],
            principal="principal:dl1",
            expected_row_version=2,
            staging_batch_id=assertions["staging_batch_id"],
            source_evidence_ref=assertions["source_evidence_ref"],
            artifact_ref=assertions["artifact_ref"],
            artifact_sha256="f" * 64,
        )
    assert workflow_pass.status == "in_progress"


def test_submit_wrong_principal_and_noncurrent_pass_fail_closed(db_session, registry):
    item, workflow_pass = _seed_dl1(db_session)
    assertions = _bind_and_validate(db_session, registry, item, workflow_pass)
    with pytest.raises(WorkflowSubmitConflict):
        submit_first_workflow_pass(
            db_session,
            item.id,
            pass_id=assertions["pass_id"],
            principal="principal:other",
            expected_row_version=2,
            staging_batch_id=assertions["staging_batch_id"],
            source_evidence_ref=assertions["source_evidence_ref"],
            artifact_ref=assertions["artifact_ref"],
            artifact_sha256=assertions["artifact_sha256"],
        )
    workflow_pass.is_current = False
    db_session.flush()
    with pytest.raises(WorkflowSubmitConflict):
        submit_first_workflow_pass(
            db_session,
            item.id,
            pass_id=assertions["pass_id"],
            principal="principal:dl1",
            expected_row_version=2,
            staging_batch_id=assertions["staging_batch_id"],
            source_evidence_ref=assertions["source_evidence_ref"],
            artifact_ref=assertions["artifact_ref"],
            artifact_sha256=assertions["artifact_sha256"],
        )


def test_duplicate_submit_is_conflict_and_event_not_duplicated(db_session, registry):
    item, workflow_pass = _seed_dl1(db_session)
    assertions = _bind_and_validate(db_session, registry, item, workflow_pass)
    first = _submit(db_session, item, assertions)
    assert first["row_version"] == 3
    with pytest.raises(WorkflowSubmitConflict):
        submit_first_workflow_pass(
            db_session,
            item.id,
            pass_id=assertions["pass_id"],
            principal="principal:dl1",
            expected_row_version=3,
            staging_batch_id=assertions["staging_batch_id"],
            source_evidence_ref=assertions["source_evidence_ref"],
            artifact_ref=assertions["artifact_ref"],
            artifact_sha256=assertions["artifact_sha256"],
        )
    assert (
        db_session.query(WorkflowEvent)
        .filter(WorkflowEvent.event_type == "pass_submitted")
        .count()
        == 1
    )


def test_corrected_revision_requires_new_binding_then_can_submit(db_session, registry):
    item, rev1 = _seed_dl1(db_session)
    rev1_assertions = _bind_and_validate(
        db_session, registry, item, rev1, hash_char="a"
    )
    correction = create_dl1_correction_revision(
        db_session,
        item.id,
        rev1.id,
        principal="principal:dl1",
        expected_row_version=2,
        reason_code="operator_correction",
    )
    rev2 = db_session.get(
        WorkflowPass,
        UUID(correction["replacement_pass_id"]),
    )
    assert rev2.staging_batch_id is None
    rev2_assertions = _bind_and_validate(
        db_session, registry, item, rev2, hash_char="b"
    )
    result = _submit(db_session, item, rev2_assertions)
    assert rev1.status == "superseded"
    assert rev1.staging_batch_id == UUID(rev1_assertions["staging_batch_id"])
    assert rev2.status == "submitted"
    assert rev2.revision_number == 2
    assert result["row_version"] == 4


def test_submit_does_not_create_dl2_or_advance_to_comparison(db_session, registry):
    item, workflow_pass = _seed_dl1(db_session)
    assertions = _bind_and_validate(db_session, registry, item, workflow_pass)
    _submit(db_session, item, assertions)
    dl2 = (
        db_session.query(WorkflowPass)
        .filter(
            WorkflowPass.workflow_item_id == item.id,
            WorkflowPass.pass_number == 2,
        )
        .all()
    )
    assert dl2 == []
    assert item.current_stage == "independent_acquisition"
    assert item.stage_condition == "ready"
