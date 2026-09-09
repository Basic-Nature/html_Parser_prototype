from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from webapp.parser.contracts.workflow_comparison import semantic_sha256
from webapp.parser.services.workflow_actions import (
    WORKFLOW_DL2_SUBMIT_CONTRACT,
    WorkflowDL2SubmitConflict,
    submit_second_workflow_pass,
)
from webapp.parser.services.workflow_pass_corrections import (
    create_dl2_correction_revision,
)
from webapp.parser.services.workflow_pre_qc_validation import (
    validate_second_workflow_pass_pre_qc,
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


def _seed_dl2(
    session,
    *,
    dl1_status="submitted",
    dl1_principal="principal:dl1",
    dl2_principal="principal:dl2",
):
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
        row_version=5,
    )
    dl1 = WorkflowPass(
        id=uuid4(),
        workflow_item_id=item.id,
        pass_number=1,
        pass_label="DL1",
        revision_number=1,
        is_current=True,
        status=dl1_status,
        assigned_principal=dl1_principal,
        source_evidence_ref="evidence://dl1/submitted.pdf",
        staging_batch_id=None,
        candidate_check_status="complete",
        candidate_check_result={"preserved": "dl1"},
        semantic_validation_status="complete",
        semantic_validation_result={"preserved": "dl1"},
        submitted_at=(
            datetime(2026, 9, 9, 2, 30, tzinfo=timezone.utc)
            if dl1_status == "submitted"
            else None
        ),
    )
    dl2 = WorkflowPass(
        id=uuid4(),
        workflow_item_id=item.id,
        pass_number=2,
        pass_label="DL2",
        revision_number=1,
        is_current=True,
        status="in_progress",
        assigned_principal=dl2_principal,
        source_evidence_ref=None,
        staging_batch_id=None,
        candidate_check_status=None,
        candidate_check_result=None,
        semantic_validation_status=None,
        semantic_validation_result=None,
        submitted_at=None,
    )
    session.add_all([item, dl1, dl2])
    session.flush()
    return item, dl1, dl2


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
            "reporting_unit": {
                "name": "Precinct 1",
                "type": "precinct",
            },
            "percent_reporting": {
                "state": "value",
                "value": "100",
            },
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


def _bind_and_validate(session, registry, item, dl2, *, hash_char="b"):
    begin_workflow_staging_binding(
        session,
        item.id,
        dl2.id,
        principal=str(dl2.assigned_principal),
        registry_path=registry,
    )
    session.add(
        StagingElectionResult(
            batch_id=dl2.staging_batch_id,
            state="Iowa",
            county=None,
            source_url=item.source_url,
            raw_html="<table>fixture</table>",
        )
    )
    session.flush()

    artifact_ref = (
        f"staging://normalized/dl2-r{dl2.revision_number}.json"
    )
    evidence_ref = (
        f"evidence://dl2/rev{dl2.revision_number}.pdf"
    )
    artifact_sha = hash_char * 64
    finalize_workflow_staging_binding(
        session,
        item.id,
        dl2.id,
        dl2.staging_batch_id,
        principal=str(dl2.assigned_principal),
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
            "workflow_pass_id": str(dl2.id),
            "pass_number": 2,
            "revision_number": dl2.revision_number,
            "source_evidence_ref": evidence_ref,
            "staging_batch_id": str(dl2.staging_batch_id),
            "normalized_artifact_ref": artifact_ref,
            "normalized_artifact_sha256": artifact_sha,
        },
        "semantic": semantic,
        "semantic_sha256": semantic_sha256(semantic),
    }
    validate_second_workflow_pass_pre_qc(
        session,
        item.id,
        dl2.id,
        dl2.staging_batch_id,
        principal=str(dl2.assigned_principal),
        normalized_payload=payload,
    )
    return {
        "pass_id": str(dl2.id),
        "staging_batch_id": str(dl2.staging_batch_id),
        "source_evidence_ref": evidence_ref,
        "artifact_ref": artifact_ref,
        "artifact_sha256": artifact_sha,
    }


def _submit(session, item, dl2, assertions):
    return submit_second_workflow_pass(
        session,
        item.id,
        pass_id=assertions["pass_id"],
        principal=str(dl2.assigned_principal),
        expected_row_version=item.row_version,
        staging_batch_id=assertions["staging_batch_id"],
        source_evidence_ref=assertions["source_evidence_ref"],
        artifact_ref=assertions["artifact_ref"],
        artifact_sha256=assertions["artifact_sha256"],
    )


def test_dl2_submit_success_preserves_dl1_and_stops_before_comparison(
    db_session,
    registry,
):
    item, dl1, dl2 = _seed_dl2(db_session)
    assertions = _bind_and_validate(db_session, registry, item, dl2)
    dl1_snapshot = (
        dl1.status,
        dl1.is_current,
        dl1.source_evidence_ref,
        dl1.candidate_check_result,
        dl1.semantic_validation_result,
        dl1.submitted_at,
    )
    original_dl2_evidence = dl2.source_evidence_ref
    original_dl2_batch = dl2.staging_batch_id

    result = _submit(db_session, item, dl2, assertions)

    assert result["contract"] == WORKFLOW_DL2_SUBMIT_CONTRACT
    assert result["pass_number"] == 2
    assert result["pass_label"] == "DL2"
    assert result["status"] == "submitted"
    assert result["row_version"] == 6
    assert result["comparison_created"] is False
    assert result["strict_comparison_stage_advanced"] is False
    assert result["committed"] is False

    assert dl2.status == "submitted"
    assert dl2.submitted_at is not None
    assert dl2.source_evidence_ref == original_dl2_evidence
    assert dl2.staging_batch_id == original_dl2_batch
    assert dl2.candidate_check_status == "complete"
    assert dl2.semantic_validation_status == "complete"

    assert (
        dl1.status,
        dl1.is_current,
        dl1.source_evidence_ref,
        dl1.candidate_check_result,
        dl1.semantic_validation_result,
        dl1.submitted_at,
    ) == dl1_snapshot
    assert item.lifecycle_state == "active"
    assert item.current_stage == "independent_acquisition"
    assert item.stage_condition == "ready"


def test_dl2_submit_requires_current_submitted_dl1(db_session, registry):
    item, dl1, dl2 = _seed_dl2(
        db_session,
        dl1_status="in_progress",
    )
    assertions = _bind_and_validate(db_session, registry, item, dl2)

    with pytest.raises(WorkflowDL2SubmitConflict):
        _submit(db_session, item, dl2, assertions)

    assert dl1.status == "in_progress"
    assert dl2.status == "in_progress"
    assert item.row_version == 5


def test_dl2_submit_requires_independent_principal(db_session, registry):
    item, dl1, dl2 = _seed_dl2(
        db_session,
        dl1_principal="principal:same",
        dl2_principal="principal:same",
    )
    assertions = _bind_and_validate(db_session, registry, item, dl2)

    with pytest.raises(WorkflowDL2SubmitConflict):
        _submit(db_session, item, dl2, assertions)

    assert dl1.status == "submitted"
    assert dl2.status == "in_progress"
    assert item.row_version == 5


def test_dl2_submit_wrong_principal_and_noncurrent_fail_closed(
    db_session,
    registry,
):
    item, _dl1, dl2 = _seed_dl2(db_session)
    assertions = _bind_and_validate(db_session, registry, item, dl2)

    with pytest.raises(WorkflowDL2SubmitConflict):
        submit_second_workflow_pass(
            db_session,
            item.id,
            pass_id=assertions["pass_id"],
            principal="principal:other",
            expected_row_version=5,
            staging_batch_id=assertions["staging_batch_id"],
            source_evidence_ref=assertions["source_evidence_ref"],
            artifact_ref=assertions["artifact_ref"],
            artifact_sha256=assertions["artifact_sha256"],
        )

    dl2.is_current = False
    db_session.flush()
    with pytest.raises(WorkflowDL2SubmitConflict):
        _submit(db_session, item, dl2, assertions)


def test_dl2_submit_requires_completed_server_pre_qc(db_session, registry):
    item, _dl1, dl2 = _seed_dl2(db_session)

    begin_workflow_staging_binding(
        db_session,
        item.id,
        dl2.id,
        principal="principal:dl2",
        registry_path=registry,
    )
    db_session.add(
        StagingElectionResult(
            batch_id=dl2.staging_batch_id,
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
        dl2.id,
        dl2.staging_batch_id,
        principal="principal:dl2",
        source_evidence_ref="evidence://dl2/rev1.pdf",
        artifact_ref="staging://normalized/dl2-r1.json",
        artifact_sha256="b" * 64,
    )

    with pytest.raises(WorkflowDL2SubmitConflict):
        submit_second_workflow_pass(
            db_session,
            item.id,
            pass_id=dl2.id,
            principal="principal:dl2",
            expected_row_version=5,
            staging_batch_id=dl2.staging_batch_id,
            source_evidence_ref=dl2.source_evidence_ref,
            artifact_ref="staging://normalized/dl2-r1.json",
            artifact_sha256="b" * 64,
        )

    assert dl2.status == "in_progress"
    assert item.stage_condition == "in_progress"


def test_dl2_submit_stale_row_version_fails_without_mutation(
    db_session,
    registry,
):
    item, _dl1, dl2 = _seed_dl2(db_session)
    assertions = _bind_and_validate(db_session, registry, item, dl2)

    with pytest.raises(WorkflowDL2SubmitConflict):
        submit_second_workflow_pass(
            db_session,
            item.id,
            pass_id=assertions["pass_id"],
            principal="principal:dl2",
            expected_row_version=99,
            staging_batch_id=assertions["staging_batch_id"],
            source_evidence_ref=assertions["source_evidence_ref"],
            artifact_ref=assertions["artifact_ref"],
            artifact_sha256=assertions["artifact_sha256"],
        )

    assert dl2.status == "in_progress"
    assert item.row_version == 5


def test_dl2_submit_client_assertion_mismatch_fails_closed(
    db_session,
    registry,
):
    item, _dl1, dl2 = _seed_dl2(db_session)
    assertions = _bind_and_validate(db_session, registry, item, dl2)

    with pytest.raises(WorkflowDL2SubmitConflict):
        submit_second_workflow_pass(
            db_session,
            item.id,
            pass_id=assertions["pass_id"],
            principal="principal:dl2",
            expected_row_version=5,
            staging_batch_id=assertions["staging_batch_id"],
            source_evidence_ref=assertions["source_evidence_ref"],
            artifact_ref=assertions["artifact_ref"],
            artifact_sha256="f" * 64,
        )

    assert dl2.status == "in_progress"
    assert item.row_version == 5


def test_duplicate_dl2_submit_is_conflict_and_event_not_duplicated(
    db_session,
    registry,
):
    item, _dl1, dl2 = _seed_dl2(db_session)
    assertions = _bind_and_validate(db_session, registry, item, dl2)
    first = _submit(db_session, item, dl2, assertions)
    assert first["row_version"] == 6

    item.stage_condition = "in_progress"
    db_session.flush()
    with pytest.raises(WorkflowDL2SubmitConflict):
        submit_second_workflow_pass(
            db_session,
            item.id,
            pass_id=assertions["pass_id"],
            principal="principal:dl2",
            expected_row_version=6,
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


def test_corrected_dl2_revision_requires_new_binding_then_can_submit(
    db_session,
    registry,
):
    item, dl1, rev1 = _seed_dl2(db_session)
    _bind_and_validate(db_session, registry, item, rev1, hash_char="b")

    correction = create_dl2_correction_revision(
        db_session,
        item.id,
        rev1.id,
        principal="principal:dl2",
        expected_row_version=5,
        reason_code="operator_correction",
    )
    rev2 = db_session.get(
        WorkflowPass,
        UUID(correction["replacement_pass_id"]),
    )
    assert rev2.staging_batch_id is None

    assertions = _bind_and_validate(
        db_session,
        registry,
        item,
        rev2,
        hash_char="c",
    )
    result = _submit(db_session, item, rev2, assertions)

    assert rev1.status == "superseded"
    assert rev2.status == "submitted"
    assert rev2.revision_number == 2
    assert result["row_version"] == 7
    assert dl1.status == "submitted"


def test_dl2_submit_writes_one_pass_event_with_no_comparison_identity(
    db_session,
    registry,
):
    item, dl1, dl2 = _seed_dl2(db_session)
    assertions = _bind_and_validate(db_session, registry, item, dl2)
    result = _submit(db_session, item, dl2, assertions)

    events = (
        db_session.query(WorkflowEvent)
        .filter(
            WorkflowEvent.workflow_item_id == item.id,
            WorkflowEvent.event_type == "pass_submitted",
        )
        .all()
    )
    assert len(events) == 1
    event = events[0]
    assert event.related_pass_id == dl2.id
    assert event.related_comparison_id is None
    assert event.event_metadata["contract"] == WORKFLOW_DL2_SUBMIT_CONTRACT
    assert event.event_metadata["pass_number"] == 2
    assert event.event_metadata["pass_label"] == "DL2"
    assert event.event_metadata["current_dl1_pass_id"] == str(dl1.id)
    assert event.event_metadata["comparison_created"] is False
    assert event.event_metadata["strict_comparison_stage_advanced"] is False
    assert result["current_dl1_pass_id"] == str(dl1.id)
