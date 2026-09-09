from __future__ import annotations

from datetime import date
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from webapp.parser.contracts.workflow_comparison import semantic_sha256
from webapp.parser.services.workflow_pass_corrections import (
    WORKFLOW_DL1_CORRECTION_CONTRACT,
    WorkflowDL1CorrectionConflict,
    WorkflowDL1CorrectionError,
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


def _bind_current(session, registry, item, workflow_pass, *, hash_char="a"):
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
    finalize_workflow_staging_binding(
        session,
        item.id,
        workflow_pass.id,
        workflow_pass.staging_batch_id,
        principal="principal:dl1",
        source_evidence_ref=(
            f"evidence://dl1/rev{workflow_pass.revision_number}.pdf"
        ),
        artifact_ref=(
            f"staging://normalized/dl1-r"
            f"{workflow_pass.revision_number}.json"
        ),
        artifact_sha256=hash_char * 64,
    )


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


def _payload(item, workflow_pass, hash_char="a"):
    semantic = _semantic()
    return {
        "schema": "workflow_normalized_semantic_comparison_payload_v1",
        "schema_version": 1,
        "comparison_version": 1,
        "binding": {
            "workflow_item_id": str(item.id),
            "workflow_pass_id": str(workflow_pass.id),
            "pass_number": 1,
            "revision_number": workflow_pass.revision_number,
            "source_evidence_ref": workflow_pass.source_evidence_ref,
            "staging_batch_id": str(workflow_pass.staging_batch_id),
            "normalized_artifact_ref": (
                f"staging://normalized/dl1-r"
                f"{workflow_pass.revision_number}.json"
            ),
            "normalized_artifact_sha256": hash_char * 64,
        },
        "semantic": semantic,
        "semantic_sha256": semantic_sha256(semantic),
    }


def test_success_supersedes_n_and_creates_clean_n_plus_1(
    db_session,
    registry,
):
    item, current = _seed_dl1(db_session)
    _bind_current(db_session, registry, item, current)
    old_batch = current.staging_batch_id
    old_source = current.source_evidence_ref

    result = create_dl1_correction_revision(
        db_session,
        item.id,
        current.id,
        principal="principal:dl1",
        expected_row_version=2,
        reason_code="pre_qc_validation_failed",
    )

    replacement = db_session.get(
        WorkflowPass,
        UUID(result["replacement_pass_id"]),
    )
    assert result["contract"] == WORKFLOW_DL1_CORRECTION_CONTRACT
    assert result["replacement_revision_number"] == 2
    assert result["row_version"] == 3
    assert result["committed"] is False

    assert current.status == "superseded"
    assert current.is_current is False
    assert current.staging_batch_id == old_batch
    assert current.source_evidence_ref == old_source
    assert current.superseded_at is not None

    assert replacement.revision_number == 2
    assert replacement.status == "in_progress"
    assert replacement.is_current is True
    assert replacement.assigned_principal == "principal:dl1"
    assert replacement.staging_batch_id is None
    assert replacement.source_evidence_ref is None
    assert replacement.candidate_check_status is None
    assert replacement.semantic_validation_status is None
    assert item.stage_condition == "in_progress"


def test_success_preserves_completed_pre_qc_on_superseded_revision(
    db_session,
    registry,
):
    item, current = _seed_dl1(db_session)
    _bind_current(db_session, registry, item, current)

    validate_first_workflow_pass_pre_qc(
        db_session,
        item.id,
        current.id,
        current.staging_batch_id,
        principal="principal:dl1",
        normalized_payload=_payload(item, current),
    )
    old_candidate = current.candidate_check_result
    old_semantic = current.semantic_validation_result
    old_links = (
        db_session.query(WorkflowArtifactLink)
        .filter(WorkflowArtifactLink.pass_id == current.id)
        .count()
    )

    result = create_dl1_correction_revision(
        db_session,
        item.id,
        current.id,
        principal="principal:dl1",
        expected_row_version=2,
        reason_code="operator_correction",
    )
    replacement = db_session.get(
        WorkflowPass,
        UUID(result["replacement_pass_id"]),
    )

    assert current.candidate_check_status == "complete"
    assert current.semantic_validation_status == "complete"
    assert current.candidate_check_result == old_candidate
    assert current.semantic_validation_result == old_semantic
    assert (
        db_session.query(WorkflowArtifactLink)
        .filter(WorkflowArtifactLink.pass_id == current.id)
        .count()
        == old_links
    )
    assert replacement.candidate_check_result is None
    assert replacement.semantic_validation_result is None


def test_stale_row_version_fails_without_supersession(
    db_session,
    registry,
):
    item, current = _seed_dl1(db_session)
    _bind_current(db_session, registry, item, current)

    with pytest.raises(WorkflowDL1CorrectionConflict):
        create_dl1_correction_revision(
            db_session,
            item.id,
            current.id,
            principal="principal:dl1",
            expected_row_version=1,
            reason_code="pre_qc_validation_failed",
        )

    assert current.is_current is True
    assert current.status == "in_progress"
    assert item.row_version == 2


def test_wrong_principal_and_unknown_reason_fail_closed(
    db_session,
    registry,
):
    item, current = _seed_dl1(db_session)
    _bind_current(db_session, registry, item, current)

    with pytest.raises(WorkflowDL1CorrectionConflict):
        create_dl1_correction_revision(
            db_session,
            item.id,
            current.id,
            principal="principal:other",
            expected_row_version=2,
            reason_code="operator_correction",
        )

    with pytest.raises(WorkflowDL1CorrectionError):
        create_dl1_correction_revision(
            db_session,
            item.id,
            current.id,
            principal="principal:dl1",
            expected_row_version=2,
            reason_code="free_form_reason",
        )


def test_submitted_revision_cannot_use_pre_submit_correction(
    db_session,
    registry,
):
    item, current = _seed_dl1(db_session)
    _bind_current(db_session, registry, item, current)
    current.status = "submitted"
    current.submitted_at = __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc
    )
    db_session.flush()

    with pytest.raises(WorkflowDL1CorrectionConflict):
        create_dl1_correction_revision(
            db_session,
            item.id,
            current.id,
            principal="principal:dl1",
            expected_row_version=2,
            reason_code="operator_correction",
        )


def test_correction_requires_real_completed_binding(
    db_session,
):
    item, current = _seed_dl1(db_session)

    with pytest.raises(WorkflowDL1CorrectionConflict):
        create_dl1_correction_revision(
            db_session,
            item.id,
            current.id,
            principal="principal:dl1",
            expected_row_version=2,
            reason_code="operator_correction",
        )


def test_revision_chain_uses_next_revision_and_new_artifact_each_time(
    db_session,
    registry,
):
    item, rev1 = _seed_dl1(db_session)
    _bind_current(db_session, registry, item, rev1, hash_char="a")

    first = create_dl1_correction_revision(
        db_session,
        item.id,
        rev1.id,
        principal="principal:dl1",
        expected_row_version=2,
        reason_code="normalized_artifact_correction",
    )
    rev2 = db_session.get(
        WorkflowPass,
        UUID(first["replacement_pass_id"]),
    )
    _bind_current(db_session, registry, item, rev2, hash_char="b")

    second = create_dl1_correction_revision(
        db_session,
        item.id,
        rev2.id,
        principal="principal:dl1",
        expected_row_version=3,
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
    assert item.row_version == 4

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
    assert links1.artifact_sha256 == "a" * 64
    assert links2.artifact_sha256 == "b" * 64


def test_exactly_one_correction_event_and_one_current_dl1(
    db_session,
    registry,
):
    item, current = _seed_dl1(db_session)
    _bind_current(db_session, registry, item, current)

    result = create_dl1_correction_revision(
        db_session,
        item.id,
        current.id,
        principal="principal:dl1",
        expected_row_version=2,
        reason_code="operator_correction",
    )

    current_rows = (
        db_session.query(WorkflowPass)
        .filter(
            WorkflowPass.workflow_item_id == item.id,
            WorkflowPass.pass_number == 1,
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
    assert len(current_rows) == 1
    assert str(current_rows[0].id) == result["replacement_pass_id"]
    assert len(events) == 1
    assert events[0].reason_code == "operator_correction"
    assert (
        events[0].event_metadata["superseded_pass_id"]
        == str(current.id)
    )
