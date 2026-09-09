from __future__ import annotations

import copy
from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from webapp.parser.contracts.workflow_comparison import semantic_sha256
from webapp.parser.services.workflow_pre_qc_validation import (
    WORKFLOW_PRE_QC_VALIDATION_CONTRACT,
    WorkflowPreQCValidationConflict,
    WorkflowPreQCValidationError,
    validate_first_workflow_pass_pre_qc,
)
from webapp.parser.services.workflow_staging_binding import (
    begin_workflow_staging_binding,
    finalize_workflow_staging_binding,
)
from webapp.parser.utils.models import (
    Base,
    StagingElectionResult,
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
        ])
        + "\n",
        encoding="utf-8",
    )
    return path


def _seed_bound_pass(session, registry):
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

    begin_workflow_staging_binding(
        session,
        item.id,
        workflow_pass.id,
        principal="principal:dl1",
        registry_path=registry,
    )
    batch_id = workflow_pass.staging_batch_id
    session.add(
        StagingElectionResult(
            batch_id=batch_id,
            state="Iowa",
            county=None,
            source_url=item.source_url,
            raw_html="<table>fixture</table>",
        )
    )
    session.flush()
    finalize_workflow_staging_binding(
        session,
        item.id,
        workflow_pass.id,
        batch_id,
        principal="principal:dl1",
        source_evidence_ref="evidence://dl1/source.pdf",
        artifact_ref="staging://normalized/dl1.json",
        artifact_sha256="a" * 64,
    )
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


def _payload(item, workflow_pass):
    semantic = _semantic()
    return {
        "schema": "workflow_normalized_semantic_comparison_payload_v1",
        "schema_version": 1,
        "comparison_version": 1,
        "binding": {
            "workflow_item_id": str(item.id),
            "workflow_pass_id": str(workflow_pass.id),
            "pass_number": 1,
            "revision_number": 1,
            "source_evidence_ref": "evidence://dl1/source.pdf",
            "staging_batch_id": str(workflow_pass.staging_batch_id),
            "normalized_artifact_ref": "staging://normalized/dl1.json",
            "normalized_artifact_sha256": "a" * 64,
        },
        "semantic": semantic,
        "semantic_sha256": semantic_sha256(semantic),
    }


def test_success_persists_server_owned_pre_qc_evidence_without_commit(
    db_session,
    registry,
):
    item, workflow_pass = _seed_bound_pass(db_session, registry)
    row_version = item.row_version

    result = validate_first_workflow_pass_pre_qc(
        db_session,
        item.id,
        workflow_pass.id,
        workflow_pass.staging_batch_id,
        principal="principal:dl1",
        normalized_payload=_payload(item, workflow_pass),
        now=datetime(2026, 9, 8, 21, 0, tzinfo=timezone.utc),
    )

    assert result["contract"] == WORKFLOW_PRE_QC_VALIDATION_CONTRACT
    assert result["committed"] is False
    assert result["already_validated"] is False
    assert workflow_pass.candidate_check_status == "complete"
    assert workflow_pass.semantic_validation_status == "complete"
    assert item.row_version == row_version
    assert (
        workflow_pass.semantic_validation_result["null_zero_missing_policy"]
        == "preserved_distinct"
    )
    assert (
        workflow_pass.candidate_check_result["roster_completeness_claim"]
        is False
    )

    events = (
        db_session.query(WorkflowEvent)
        .filter(WorkflowEvent.event_type == "pre_qc_pass_validated")
        .all()
    )
    assert len(events) == 1


def test_exact_replay_is_idempotent_and_does_not_duplicate_event(
    db_session,
    registry,
):
    item, workflow_pass = _seed_bound_pass(db_session, registry)
    payload = _payload(item, workflow_pass)

    validate_first_workflow_pass_pre_qc(
        db_session,
        item.id,
        workflow_pass.id,
        workflow_pass.staging_batch_id,
        principal="principal:dl1",
        normalized_payload=payload,
    )
    second = validate_first_workflow_pass_pre_qc(
        db_session,
        item.id,
        workflow_pass.id,
        workflow_pass.staging_batch_id,
        principal="principal:dl1",
        normalized_payload=payload,
    )

    assert second["already_validated"] is True
    assert (
        db_session.query(WorkflowEvent)
        .filter(WorkflowEvent.event_type == "pre_qc_pass_validated")
        .count()
        == 1
    )


def test_binding_mismatch_fails_without_pre_qc_mutation(
    db_session,
    registry,
):
    item, workflow_pass = _seed_bound_pass(db_session, registry)
    payload = _payload(item, workflow_pass)
    payload["binding"]["source_evidence_ref"] = "evidence://wrong"

    with pytest.raises(WorkflowPreQCValidationConflict):
        validate_first_workflow_pass_pre_qc(
            db_session,
            item.id,
            workflow_pass.id,
            workflow_pass.staging_batch_id,
            principal="principal:dl1",
            normalized_payload=payload,
        )

    assert workflow_pass.candidate_check_status is None
    assert workflow_pass.semantic_validation_status is None


def test_scope_mismatch_fails_without_pre_qc_mutation(
    db_session,
    registry,
):
    item, workflow_pass = _seed_bound_pass(db_session, registry)
    payload = _payload(item, workflow_pass)
    payload["semantic"]["scope"]["jurisdiction_name"] = "Invented County"
    payload["semantic_sha256"] = semantic_sha256(payload["semantic"])

    with pytest.raises(WorkflowPreQCValidationConflict):
        validate_first_workflow_pass_pre_qc(
            db_session,
            item.id,
            workflow_pass.id,
            workflow_pass.staging_batch_id,
            principal="principal:dl1",
            normalized_payload=payload,
        )
    assert workflow_pass.candidate_check_status is None


def test_empty_candidate_record_fails_pre_qc(
    db_session,
    registry,
):
    item, workflow_pass = _seed_bound_pass(db_session, registry)
    payload = _payload(item, workflow_pass)
    payload["semantic"]["records"][0]["candidates"] = []
    payload["semantic_sha256"] = semantic_sha256(payload["semantic"])

    with pytest.raises(WorkflowPreQCValidationError):
        validate_first_workflow_pass_pre_qc(
            db_session,
            item.id,
            workflow_pass.id,
            workflow_pass.staging_batch_id,
            principal="principal:dl1",
            normalized_payload=payload,
        )
    assert workflow_pass.candidate_check_status is None


def test_candidate_total_exact_mismatch_fails(
    db_session,
    registry,
):
    item, workflow_pass = _seed_bound_pass(db_session, registry)
    payload = _payload(item, workflow_pass)
    payload["semantic"]["records"][0]["candidates"][0]["total_votes"]["votes"] = 12
    payload["semantic_sha256"] = semantic_sha256(payload["semantic"])

    with pytest.raises(WorkflowPreQCValidationError):
        validate_first_workflow_pass_pre_qc(
            db_session,
            item.id,
            workflow_pass.id,
            workflow_pass.staging_batch_id,
            principal="principal:dl1",
            normalized_payload=payload,
        )


def test_null_missing_are_not_coerced_to_zero_and_known_subtotal_is_allowed(
    db_session,
    registry,
):
    item, workflow_pass = _seed_bound_pass(db_session, registry)
    payload = _payload(item, workflow_pass)
    record = payload["semantic"]["records"][0]

    record["method_totals"][2] = {
        "method": "Absentee Mail",
        "state": "null",
        "votes": None,
    }
    record["candidates"][0]["method_votes"][2] = {
        "method": "Absentee Mail",
        "state": "null",
        "votes": None,
    }
    record["candidates"][1]["method_votes"][2] = {
        "method": "Absentee Mail",
        "state": "missing",
        "votes": None,
    }
    record["candidates"][0]["total_votes"] = {
        "state": "value",
        "votes": 11,
    }
    record["candidates"][1]["total_votes"] = {
        "state": "value",
        "votes": 9,
    }
    payload["semantic_sha256"] = semantic_sha256(payload["semantic"])

    result = validate_first_workflow_pass_pre_qc(
        db_session,
        item.id,
        workflow_pass.id,
        workflow_pass.staging_batch_id,
        principal="principal:dl1",
        normalized_payload=payload,
    )
    assert result["semantic_validation_status"] == "complete"


def test_method_total_below_known_candidate_subtotal_fails(
    db_session,
    registry,
):
    item, workflow_pass = _seed_bound_pass(db_session, registry)
    payload = _payload(item, workflow_pass)
    record = payload["semantic"]["records"][0]
    record["method_totals"][0]["votes"] = 10
    payload["semantic_sha256"] = semantic_sha256(payload["semantic"])

    with pytest.raises(WorkflowPreQCValidationError):
        validate_first_workflow_pass_pre_qc(
            db_session,
            item.id,
            workflow_pass.id,
            workflow_pass.staging_batch_id,
            principal="principal:dl1",
            normalized_payload=payload,
        )


def test_w4_semantic_hash_failure_and_tampered_artifact_fail_closed(
    db_session,
    registry,
):
    item, workflow_pass = _seed_bound_pass(db_session, registry)
    payload = _payload(item, workflow_pass)
    payload["semantic_sha256"] = "0" * 64

    with pytest.raises(WorkflowPreQCValidationError):
        validate_first_workflow_pass_pre_qc(
            db_session,
            item.id,
            workflow_pass.id,
            workflow_pass.staging_batch_id,
            principal="principal:dl1",
            normalized_payload=payload,
        )

    link = (
        db_session.query(WorkflowArtifactLink)
        .filter(WorkflowArtifactLink.pass_id == workflow_pass.id)
        .one()
    )
    link.artifact_sha256 = "b" * 64
    db_session.flush()

    with pytest.raises(WorkflowPreQCValidationConflict):
        validate_first_workflow_pass_pre_qc(
            db_session,
            item.id,
            workflow_pass.id,
            workflow_pass.staging_batch_id,
            principal="principal:dl1",
            normalized_payload=_payload(item, workflow_pass),
        )
