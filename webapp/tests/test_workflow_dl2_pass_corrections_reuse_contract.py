from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from webapp.parser.contracts.workflow_comparison import semantic_sha256
from webapp.parser.services.workflow_pass_corrections import (
    WORKFLOW_DL2_CORRECTION_CONTRACT,
    WorkflowDL1CorrectionConflict,
    WorkflowDL2CorrectionConflict,
    WorkflowDL2CorrectionError,
    create_dl1_correction_revision,
    create_dl2_correction_revision,
    create_workflow_pass_correction_revision,
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


def _new_item(*, row_version: int = 5) -> WorkflowItem:
    return WorkflowItem(
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
        row_version=row_version,
    )


def _seed_dl2_with_submitted_dl1(session):
    item = _new_item()
    submitted_at = datetime(2026, 9, 9, 2, 30, tzinfo=timezone.utc)
    dl1 = WorkflowPass(
        id=uuid4(),
        workflow_item_id=item.id,
        pass_number=1,
        pass_label="DL1",
        revision_number=1,
        is_current=True,
        status="submitted",
        assigned_principal="principal:dl1",
        source_evidence_ref="evidence://dl1/submitted.pdf",
        staging_batch_id=None,
        candidate_check_status="complete",
        candidate_check_result={"preserved": "dl1"},
        semantic_validation_status="complete",
        semantic_validation_result={"preserved": "dl1"},
        submitted_at=submitted_at,
    )
    dl2 = WorkflowPass(
        id=uuid4(),
        workflow_item_id=item.id,
        pass_number=2,
        pass_label="DL2",
        revision_number=1,
        is_current=True,
        status="in_progress",
        assigned_principal="principal:dl2",
        source_evidence_ref=None,
        staging_batch_id=None,
        candidate_check_status=None,
        candidate_check_result=None,
        semantic_validation_status=None,
        semantic_validation_result=None,
    )
    session.add_all([item, dl1, dl2])
    session.flush()
    return item, dl1, dl2


def _bind_current(
    session,
    registry,
    item,
    workflow_pass,
    *,
    hash_char: str,
):
    principal = str(workflow_pass.assigned_principal)
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
    slug = workflow_pass.pass_label.lower()
    finalize_workflow_staging_binding(
        session,
        item.id,
        workflow_pass.id,
        workflow_pass.staging_batch_id,
        principal=principal,
        source_evidence_ref=(
            f"evidence://{slug}/rev{workflow_pass.revision_number}.pdf"
        ),
        artifact_ref=(
            f"staging://normalized/{slug}-r"
            f"{workflow_pass.revision_number}.json"
        ),
        artifact_sha256=hash_char * 64,
    )


def _seed_single_bound_pass(
    session,
    registry,
    *,
    pass_number: int,
    pass_label: str,
    principal: str,
):
    item = _new_item()
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
    _bind_current(
        session,
        registry,
        item,
        workflow_pass,
        hash_char=("a" if pass_number == 1 else "b"),
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


def _payload(item, workflow_pass, *, hash_char: str):
    semantic = _semantic()
    slug = workflow_pass.pass_label.lower()
    return {
        "schema": "workflow_normalized_semantic_comparison_payload_v1",
        "schema_version": 1,
        "comparison_version": 1,
        "binding": {
            "workflow_item_id": str(item.id),
            "workflow_pass_id": str(workflow_pass.id),
            "pass_number": workflow_pass.pass_number,
            "revision_number": workflow_pass.revision_number,
            "source_evidence_ref": workflow_pass.source_evidence_ref,
            "staging_batch_id": str(workflow_pass.staging_batch_id),
            "normalized_artifact_ref": (
                f"staging://normalized/{slug}-r"
                f"{workflow_pass.revision_number}.json"
            ),
            "normalized_artifact_sha256": hash_char * 64,
        },
        "semantic": semantic,
        "semantic_sha256": semantic_sha256(semantic),
    }


def _bind_dl2(session, registry, item, dl2, *, hash_char="b"):
    _bind_current(session, registry, item, dl2, hash_char=hash_char)


def test_dl2_wrapper_supersedes_only_dl2_and_preserves_submitted_dl1(
    db_session,
    registry,
):
    item, dl1, dl2 = _seed_dl2_with_submitted_dl1(db_session)
    _bind_dl2(db_session, registry, item, dl2)
    dl1_snapshot = (
        dl1.status,
        dl1.is_current,
        dl1.assigned_principal,
        dl1.source_evidence_ref,
        dl1.candidate_check_result,
        dl1.semantic_validation_result,
        dl1.submitted_at,
    )
    dl2_batch = dl2.staging_batch_id
    dl2_source = dl2.source_evidence_ref
    result = create_dl2_correction_revision(
        db_session,
        item.id,
        dl2.id,
        principal="principal:dl2",
        expected_row_version=5,
        reason_code="operator_correction",
    )
    replacement = db_session.get(
        WorkflowPass,
        UUID(result["replacement_pass_id"]),
    )
    assert result["contract"] == WORKFLOW_DL2_CORRECTION_CONTRACT
    assert result["pass_number"] == 2
    assert result["pass_label"] == "DL2"
    assert result["replacement_revision_number"] == 2
    assert result["row_version"] == 6
    assert result["committed"] is False
    assert dl2.status == "superseded"
    assert dl2.is_current is False
    assert dl2.staging_batch_id == dl2_batch
    assert dl2.source_evidence_ref == dl2_source
    assert replacement.pass_number == 2
    assert replacement.pass_label == "DL2"
    assert replacement.revision_number == 2
    assert replacement.status == "in_progress"
    assert replacement.is_current is True
    assert replacement.assigned_principal == "principal:dl2"
    assert replacement.staging_batch_id is None
    assert replacement.source_evidence_ref is None
    assert replacement.candidate_check_status is None
    assert replacement.semantic_validation_status is None
    assert (
        dl1.status,
        dl1.is_current,
        dl1.assigned_principal,
        dl1.source_evidence_ref,
        dl1.candidate_check_result,
        dl1.semantic_validation_result,
        dl1.submitted_at,
    ) == dl1_snapshot


def test_generic_engine_derives_dl2_identity_from_server_loaded_pass(
    db_session,
    registry,
):
    item, _dl1, dl2 = _seed_dl2_with_submitted_dl1(db_session)
    _bind_dl2(db_session, registry, item, dl2)
    result = create_workflow_pass_correction_revision(
        db_session,
        item.id,
        dl2.id,
        principal="principal:dl2",
        expected_row_version=5,
        reason_code="source_evidence_correction",
    )
    assert result["contract"] == WORKFLOW_DL2_CORRECTION_CONTRACT
    assert result["pass_number"] == 2
    assert result["pass_label"] == "DL2"


def test_dl2_completed_pre_qc_is_preserved_on_superseded_revision(
    db_session,
    registry,
):
    item, _dl1, dl2 = _seed_dl2_with_submitted_dl1(db_session)
    _bind_dl2(db_session, registry, item, dl2)
    validate_second_workflow_pass_pre_qc(
        db_session,
        item.id,
        dl2.id,
        dl2.staging_batch_id,
        principal="principal:dl2",
        normalized_payload=_payload(item, dl2, hash_char="b"),
    )
    old_candidate = dl2.candidate_check_result
    old_semantic = dl2.semantic_validation_result
    old_links = (
        db_session.query(WorkflowArtifactLink)
        .filter(WorkflowArtifactLink.pass_id == dl2.id)
        .count()
    )
    result = create_dl2_correction_revision(
        db_session,
        item.id,
        dl2.id,
        principal="principal:dl2",
        expected_row_version=5,
        reason_code="pre_qc_validation_failed",
    )
    replacement = db_session.get(
        WorkflowPass,
        UUID(result["replacement_pass_id"]),
    )
    assert dl2.candidate_check_status == "complete"
    assert dl2.semantic_validation_status == "complete"
    assert dl2.candidate_check_result == old_candidate
    assert dl2.semantic_validation_result == old_semantic
    assert (
        db_session.query(WorkflowArtifactLink)
        .filter(WorkflowArtifactLink.pass_id == dl2.id)
        .count()
        == old_links
    )
    assert replacement.candidate_check_status is None
    assert replacement.candidate_check_result is None
    assert replacement.semantic_validation_status is None
    assert replacement.semantic_validation_result is None


def test_dl1_wrapper_rejects_dl2_without_supersession(
    db_session,
    registry,
):
    item, _dl1, dl2 = _seed_dl2_with_submitted_dl1(db_session)
    _bind_dl2(db_session, registry, item, dl2)
    with pytest.raises(WorkflowDL1CorrectionConflict):
        create_dl1_correction_revision(
            db_session,
            item.id,
            dl2.id,
            principal="principal:dl2",
            expected_row_version=5,
            reason_code="operator_correction",
        )
    assert dl2.is_current is True
    assert dl2.status == "in_progress"
    assert item.row_version == 5


def test_dl2_wrapper_rejects_dl1_without_supersession(
    db_session,
    registry,
):
    item, dl1 = _seed_single_bound_pass(
        db_session,
        registry,
        pass_number=1,
        pass_label="DL1",
        principal="principal:dl1",
    )
    with pytest.raises(WorkflowDL2CorrectionConflict):
        create_dl2_correction_revision(
            db_session,
            item.id,
            dl1.id,
            principal="principal:dl1",
            expected_row_version=5,
            reason_code="operator_correction",
        )
    assert dl1.is_current is True
    assert dl1.status == "in_progress"
    assert item.row_version == 5


def test_dl2_stale_row_version_fails_without_supersession(
    db_session,
    registry,
):
    item, _dl1, dl2 = _seed_dl2_with_submitted_dl1(db_session)
    _bind_dl2(db_session, registry, item, dl2)
    with pytest.raises(WorkflowDL2CorrectionConflict):
        create_dl2_correction_revision(
            db_session,
            item.id,
            dl2.id,
            principal="principal:dl2",
            expected_row_version=4,
            reason_code="operator_correction",
        )
    assert dl2.is_current is True
    assert dl2.status == "in_progress"
    assert item.row_version == 5


def test_dl2_wrong_principal_and_unknown_reason_fail_closed(
    db_session,
    registry,
):
    item, _dl1, dl2 = _seed_dl2_with_submitted_dl1(db_session)
    _bind_dl2(db_session, registry, item, dl2)
    with pytest.raises(WorkflowDL2CorrectionConflict):
        create_dl2_correction_revision(
            db_session,
            item.id,
            dl2.id,
            principal="principal:other",
            expected_row_version=5,
            reason_code="operator_correction",
        )
    with pytest.raises(WorkflowDL2CorrectionError):
        create_dl2_correction_revision(
            db_session,
            item.id,
            dl2.id,
            principal="principal:dl2",
            expected_row_version=5,
            reason_code="free_form_reason",
        )
    assert dl2.is_current is True
    assert dl2.status == "in_progress"
    assert item.row_version == 5


def test_dl2_revision_chain_preserves_each_prior_artifact(
    db_session,
    registry,
):
    item, _dl1, rev1 = _seed_dl2_with_submitted_dl1(db_session)
    _bind_dl2(db_session, registry, item, rev1, hash_char="b")
    first = create_dl2_correction_revision(
        db_session,
        item.id,
        rev1.id,
        principal="principal:dl2",
        expected_row_version=5,
        reason_code="normalized_artifact_correction",
    )
    rev2 = db_session.get(
        WorkflowPass,
        UUID(first["replacement_pass_id"]),
    )
    _bind_dl2(db_session, registry, item, rev2, hash_char="c")
    second = create_dl2_correction_revision(
        db_session,
        item.id,
        rev2.id,
        principal="principal:dl2",
        expected_row_version=6,
        reason_code="source_evidence_correction",
    )
    rev3 = db_session.get(
        WorkflowPass,
        UUID(second["replacement_pass_id"]),
    )
    assert rev1.status == "superseded"
    assert rev2.status == "superseded"
    assert rev3.revision_number == 3
    assert rev3.is_current is True
    assert rev3.staging_batch_id is None
    assert item.row_version == 7
    links1 = (
        db_session.query(WorkflowArtifactLink)
        .filter(WorkflowArtifactLink.pass_id == rev1.id)
        .one()
    )
    links2 = (
        db_session.query(WorkflowArtifactLink)
        .filter(WorkflowArtifactLink.pass_id == rev2.id)
        .one()
    )
    assert links1.artifact_sha256 == "b" * 64
    assert links2.artifact_sha256 == "c" * 64


def test_submitted_dl2_cannot_use_pre_submit_correction(
    db_session,
    registry,
):
    item, _dl1, dl2 = _seed_dl2_with_submitted_dl1(db_session)
    _bind_dl2(db_session, registry, item, dl2)
    dl2.status = "submitted"
    dl2.submitted_at = datetime.now(timezone.utc)
    db_session.flush()
    with pytest.raises(WorkflowDL2CorrectionConflict):
        create_dl2_correction_revision(
            db_session,
            item.id,
            dl2.id,
            principal="principal:dl2",
            expected_row_version=5,
            reason_code="operator_correction",
        )


def test_dl2_correction_writes_one_dynamic_event_and_one_current_dl2(
    db_session,
    registry,
):
    item, dl1, dl2 = _seed_dl2_with_submitted_dl1(db_session)
    _bind_dl2(db_session, registry, item, dl2)
    result = create_dl2_correction_revision(
        db_session,
        item.id,
        dl2.id,
        principal="principal:dl2",
        expected_row_version=5,
        reason_code="operator_correction",
    )
    current_dl2 = (
        db_session.query(WorkflowPass)
        .filter(
            WorkflowPass.workflow_item_id == item.id,
            WorkflowPass.pass_number == 2,
            WorkflowPass.is_current.is_(True),
        )
        .all()
    )
    events = (
        db_session.query(WorkflowEvent)
        .filter(
            WorkflowEvent.workflow_item_id == item.id,
            WorkflowEvent.event_type
                == "pass_correction_revision_created",
        )
        .all()
    )
    assert len(current_dl2) == 1
    assert str(current_dl2[0].id) == result["replacement_pass_id"]
    assert len(events) == 1
    event = events[0]
    assert event.reason_code == "operator_correction"
    assert event.event_metadata["contract"] == WORKFLOW_DL2_CORRECTION_CONTRACT
    assert event.event_metadata["pass_number"] == 2
    assert event.event_metadata["pass_label"] == "DL2"
    assert event.event_metadata["superseded_pass_id"] == str(dl2.id)
    assert (
        event.event_metadata["replacement_pass_id"]
        == result["replacement_pass_id"]
    )
    assert event.prior_state["current_pass_number"] == 2
    assert event.prior_state["current_pass_label"] == "DL2"
    assert event.new_state["current_dl2_revision"] == 2
    assert dl1.status == "submitted"
    assert dl1.is_current is True
    assert dl1.source_evidence_ref == "evidence://dl1/submitted.pdf"
