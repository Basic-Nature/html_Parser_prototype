from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from webapp.parser.contracts.workflow_comparison import semantic_sha256
from webapp.parser.services.workflow_pre_qc_validation import (
    WorkflowPreQCValidationConflict,
    validate_first_workflow_pass_pre_qc,
    validate_second_workflow_pass_pre_qc,
    validate_workflow_pass_pre_qc,
)
from webapp.parser.services.workflow_staging_binding import (
    WorkflowStagingBindingConflict,
    begin_workflow_staging_binding,
    finalize_workflow_staging_binding,
    validate_completed_workflow_staging_binding,
)
from webapp.parser.utils.models import (
    Base,
    BatchMetadata,
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
        "records": [
            {
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
            }
        ],
    }


def _seed_bound_pass(
    session,
    registry,
    *,
    pass_number: int,
    pass_label: str,
    principal: str,
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
        row_version=4,
    )
    workflow_pass = WorkflowPass(
        id=uuid4(),
        workflow_item_id=item.id,
        pass_number=pass_number,
        pass_label=pass_label,
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

    begin_workflow_staging_binding(
        session,
        item.id,
        workflow_pass.id,
        principal=principal,
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

    slug = pass_label.lower()
    finalize_workflow_staging_binding(
        session,
        item.id,
        workflow_pass.id,
        workflow_pass.staging_batch_id,
        principal=principal,
        source_evidence_ref=f"evidence://{slug}/source.pdf",
        artifact_ref=f"staging://normalized/{slug}.json",
        artifact_sha256=("a" if pass_number == 1 else "b") * 64,
    )
    return item, workflow_pass


def _payload(item, workflow_pass):
    semantic = _semantic()
    slug = workflow_pass.pass_label.lower()
    artifact_hash = ("a" if workflow_pass.pass_number == 1 else "b") * 64
    return {
        "schema": "workflow_normalized_semantic_comparison_payload_v1",
        "schema_version": 1,
        "comparison_version": 1,
        "binding": {
            "workflow_item_id": str(item.id),
            "workflow_pass_id": str(workflow_pass.id),
            "pass_number": workflow_pass.pass_number,
            "revision_number": workflow_pass.revision_number,
            "source_evidence_ref": f"evidence://{slug}/source.pdf",
            "staging_batch_id": str(workflow_pass.staging_batch_id),
            "normalized_artifact_ref": f"staging://normalized/{slug}.json",
            "normalized_artifact_sha256": artifact_hash,
        },
        "semantic": semantic,
        "semantic_sha256": semantic_sha256(semantic),
    }


def test_dl2_staging_binding_uses_shared_contract_and_dynamic_audit(
    db_session,
    registry,
):
    item, workflow_pass = _seed_bound_pass(
        db_session,
        registry,
        pass_number=2,
        pass_label="DL2",
        principal="principal:dl2",
    )

    validated = validate_completed_workflow_staging_binding(
        db_session,
        item.id,
        workflow_pass.id,
        workflow_pass.staging_batch_id,
        principal="principal:dl2",
    )
    assert validated["binding_state"] == "complete"

    batch = db_session.get(BatchMetadata, workflow_pass.staging_batch_id)
    assert batch.metastats["pass_number"] == 2
    assert batch.metastats["pass_label"] == "DL2"

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
    assert all(event.event_metadata["pass_number"] == 2 for event in events)
    assert all(event.event_metadata["pass_label"] == "DL2" for event in events)
    assert all("DL2" in event.summary for event in events)


def test_staging_rejects_unsupported_governed_pass_pair(db_session, registry):
    item = WorkflowItem(
        id=uuid4(),
        lifecycle_state="active",
        current_stage="independent_acquisition",
        stage_condition="in_progress",
        priority=0,
        election_year=2024,
        state="Iowa",
        contest="President",
        office_basic="President",
        source_race_id="2024PRESIA",
        source_url="https://sos.example.gov/results.pdf",
        workflow_metadata={},
        row_version=4,
    )
    workflow_pass = WorkflowPass(
        id=uuid4(),
        workflow_item_id=item.id,
        pass_number=3,
        pass_label="DL3",
        revision_number=1,
        is_current=True,
        status="in_progress",
        assigned_principal="principal:dl3",
    )
    db_session.add_all([item, workflow_pass])
    db_session.flush()

    with pytest.raises(WorkflowStagingBindingConflict):
        begin_workflow_staging_binding(
            db_session,
            item.id,
            workflow_pass.id,
            principal="principal:dl3",
            registry_path=registry,
        )


def test_dl2_pre_qc_wrapper_persists_only_dl2_validation(
    db_session,
    registry,
):
    item, workflow_pass = _seed_bound_pass(
        db_session,
        registry,
        pass_number=2,
        pass_label="DL2",
        principal="principal:dl2",
    )
    row_version = item.row_version

    result = validate_second_workflow_pass_pre_qc(
        db_session,
        item.id,
        workflow_pass.id,
        workflow_pass.staging_batch_id,
        principal="principal:dl2",
        normalized_payload=_payload(item, workflow_pass),
        now=datetime(2026, 9, 9, 3, 0, tzinfo=timezone.utc),
    )

    assert result["already_validated"] is False
    assert result["candidate_check_status"] == "complete"
    assert result["semantic_validation_status"] == "complete"
    assert workflow_pass.candidate_check_status == "complete"
    assert workflow_pass.semantic_validation_status == "complete"
    assert item.row_version == row_version

    event = (
        db_session.query(WorkflowEvent)
        .filter(WorkflowEvent.event_type == "pre_qc_pass_validated")
        .one()
    )
    assert event.event_metadata["pass_number"] == 2
    assert event.event_metadata["pass_label"] == "DL2"
    assert "DL2" in event.summary


def test_generic_pre_qc_derives_dl2_identity_from_server_loaded_pass(
    db_session,
    registry,
):
    item, workflow_pass = _seed_bound_pass(
        db_session,
        registry,
        pass_number=2,
        pass_label="DL2",
        principal="principal:dl2",
    )

    result = validate_workflow_pass_pre_qc(
        db_session,
        item.id,
        workflow_pass.id,
        workflow_pass.staging_batch_id,
        principal="principal:dl2",
        normalized_payload=_payload(item, workflow_pass),
    )
    assert result["semantic_validation_status"] == "complete"


def test_dl1_wrapper_rejects_dl2_without_validation_mutation(
    db_session,
    registry,
):
    item, workflow_pass = _seed_bound_pass(
        db_session,
        registry,
        pass_number=2,
        pass_label="DL2",
        principal="principal:dl2",
    )
    with pytest.raises(WorkflowPreQCValidationConflict):
        validate_first_workflow_pass_pre_qc(
            db_session,
            item.id,
            workflow_pass.id,
            workflow_pass.staging_batch_id,
            principal="principal:dl2",
            normalized_payload=_payload(item, workflow_pass),
        )
    assert workflow_pass.candidate_check_status is None
    assert workflow_pass.semantic_validation_status is None


def test_dl2_wrapper_rejects_dl1_without_validation_mutation(
    db_session,
    registry,
):
    item, workflow_pass = _seed_bound_pass(
        db_session,
        registry,
        pass_number=1,
        pass_label="DL1",
        principal="principal:dl1",
    )
    with pytest.raises(WorkflowPreQCValidationConflict):
        validate_second_workflow_pass_pre_qc(
            db_session,
            item.id,
            workflow_pass.id,
            workflow_pass.staging_batch_id,
            principal="principal:dl1",
            normalized_payload=_payload(item, workflow_pass),
        )
    assert workflow_pass.candidate_check_status is None
    assert workflow_pass.semantic_validation_status is None


def test_dl2_payload_pass_identity_mismatch_fails_without_mutation(
    db_session,
    registry,
):
    item, workflow_pass = _seed_bound_pass(
        db_session,
        registry,
        pass_number=2,
        pass_label="DL2",
        principal="principal:dl2",
    )
    payload = _payload(item, workflow_pass)
    payload["binding"]["pass_number"] = 1

    with pytest.raises(WorkflowPreQCValidationConflict):
        validate_second_workflow_pass_pre_qc(
            db_session,
            item.id,
            workflow_pass.id,
            workflow_pass.staging_batch_id,
            principal="principal:dl2",
            normalized_payload=payload,
        )
    assert workflow_pass.candidate_check_status is None


def test_dl2_pre_qc_exact_replay_is_idempotent(
    db_session,
    registry,
):
    item, workflow_pass = _seed_bound_pass(
        db_session,
        registry,
        pass_number=2,
        pass_label="DL2",
        principal="principal:dl2",
    )
    payload = _payload(item, workflow_pass)

    validate_second_workflow_pass_pre_qc(
        db_session,
        item.id,
        workflow_pass.id,
        workflow_pass.staging_batch_id,
        principal="principal:dl2",
        normalized_payload=payload,
    )
    second = validate_second_workflow_pass_pre_qc(
        db_session,
        item.id,
        workflow_pass.id,
        workflow_pass.staging_batch_id,
        principal="principal:dl2",
        normalized_payload=payload,
    )
    assert second["already_validated"] is True
    assert (
        db_session.query(WorkflowEvent)
        .filter(WorkflowEvent.event_type == "pre_qc_pass_validated")
        .count()
        == 1
    )
