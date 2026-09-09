from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from webapp.parser.services.workflow_discrepancy_resolution import (
    WorkflowDiscrepancyResolutionConflict,
    WorkflowDiscrepancyResolutionError,
    resolve_workflow_comparison_discrepancies,
)
from webapp.parser.utils.models import (
    Base,
    WorkflowComparison,
    WorkflowDiscrepancy,
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


def _seed_resolution(db_session):
    checked = datetime(2026, 9, 9, 1, 2, 3, tzinfo=timezone.utc)
    item = WorkflowItem(
        id=uuid4(),
        lifecycle_state="active",
        current_stage="discrepancy_resolution",
        stage_condition="pending",
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
        source_url="https://sos.example.gov/results.pdf",
        canonical_race_id=None,
        blocked_reason_code=None,
        blocker_detail=None,
        created_by_principal=None,
        workflow_metadata={},
        row_version=8,
    )
    dl1 = WorkflowPass(
        id=uuid4(),
        workflow_item_id=item.id,
        pass_number=1,
        pass_label="DL1",
        revision_number=1,
        is_current=True,
        status="submitted",
        assigned_principal="principal:dl1",
        source_evidence_ref="evidence:dl1",
        staging_batch_id=None,
        candidate_check_status="complete",
        candidate_check_result={},
        semantic_validation_status="complete",
        semantic_validation_result={},
        started_at=checked,
        submitted_at=checked,
        superseded_at=None,
        notes=None,
    )
    dl2 = WorkflowPass(
        id=uuid4(),
        workflow_item_id=item.id,
        pass_number=2,
        pass_label="DL2",
        revision_number=1,
        is_current=True,
        status="submitted",
        assigned_principal="principal:dl2",
        source_evidence_ref="evidence:dl2",
        staging_batch_id=None,
        candidate_check_status="complete",
        candidate_check_result={},
        semantic_validation_status="complete",
        semantic_validation_result={},
        started_at=checked,
        submitted_at=checked,
        superseded_at=None,
        notes=None,
    )
    comparison = WorkflowComparison(
        id=uuid4(),
        workflow_item_id=item.id,
        left_pass_id=dl1.id,
        right_pass_id=dl2.id,
        comparison_version=1,
        status="complete",
        strict_equality_passed=False,
        difference_count=2,
        difference_summary={
            "difference_count": 2,
            "left_semantic_sha256": "a" * 64,
            "right_semantic_sha256": "b" * 64,
            "category_counts": {"value_mismatch": 2},
        },
        checked_at=checked,
        checked_by_service_version="strict-comparison-test",
        reviewed_by_principal=None,
        reviewed_at=None,
        created_at=checked,
    )
    discrepancy1 = WorkflowDiscrepancy(
        id=uuid4(),
        comparison_id=comparison.id,
        workflow_item_id=item.id,
        category="value_mismatch",
        semantic_key=["records", "Precinct 1", "Jane Doe", "Election Day"],
        left_value=10,
        right_value=11,
        left_value_state="value",
        right_value_state="value",
        severity=None,
        resolution_status="open",
        resolution_code=None,
        resolution_notes=None,
        resolved_by_principal=None,
        resolved_at=None,
        created_at=checked,
    )
    discrepancy2 = WorkflowDiscrepancy(
        id=uuid4(),
        comparison_id=comparison.id,
        workflow_item_id=item.id,
        category="null_vs_zero",
        semantic_key=["records", "Precinct 1", "Jane Doe", "Provisional"],
        left_value=None,
        right_value=0,
        left_value_state="null",
        right_value_state="value",
        severity=None,
        resolution_status="open",
        resolution_code=None,
        resolution_notes=None,
        resolved_by_principal=None,
        resolved_at=None,
        created_at=checked,
    )
    db_session.add_all([
        item,
        dl1,
        dl2,
        comparison,
        discrepancy1,
        discrepancy2,
    ])
    db_session.flush()
    return item, dl1, dl2, comparison, [discrepancy1, discrepancy2]


def test_resolution_selects_dl1_atomically_and_advances_qc1(db_session):
    item, dl1, dl2, comparison, discrepancies = _seed_resolution(db_session)
    timestamp = datetime(2026, 9, 9, 2, 3, 4, tzinfo=timezone.utc)

    payload = resolve_workflow_comparison_discrepancies(
        db_session,
        item.id,
        comparison.id,
        principal="principal:reviewer",
        expected_row_version=8,
        resolution_code="select_dl1",
        resolution_notes="Official evidence supports DL1.",
        now=timestamp,
    )

    assert payload["committed"] is False
    assert payload["already_resolved"] is False
    assert payload["selected_pass_id"] == str(dl1.id)
    assert payload["selected_pass_label"] == "DL1"
    assert payload["resolved_discrepancy_count"] == 2
    assert item.current_stage == "qc1_review"
    assert item.stage_condition == "pending"
    assert item.row_version == 9
    assert comparison.reviewed_by_principal == "principal:reviewer"
    assert comparison.reviewed_at == timestamp

    for row in discrepancies:
        assert row.resolution_status == "resolved"
        assert row.resolution_code == "select_dl1"
        assert row.resolution_notes == "Official evidence supports DL1."
        assert row.resolved_by_principal == "principal:reviewer"
        assert row.resolved_at == timestamp

    assert dl1.status == "submitted"
    assert dl2.status == "submitted"
    assert dl1.is_current is True
    assert dl2.is_current is True

    events = db_session.query(WorkflowEvent).filter(
        WorkflowEvent.workflow_item_id == item.id,
        WorkflowEvent.event_type == "discrepancy_resolution_completed",
    ).all()
    assert len(events) == 1
    event = events[0]
    assert event.related_pass_id == dl1.id
    assert event.related_comparison_id == comparison.id
    assert event.event_metadata["resolution_code"] == "select_dl1"
    assert event.event_metadata["mixed_side_value_merge"] is False
    assert event.event_metadata["direct_value_edit"] is False


def test_resolution_selects_dl2_server_side(db_session):
    item, _, dl2, comparison, _ = _seed_resolution(db_session)
    payload = resolve_workflow_comparison_discrepancies(
        db_session,
        item.id,
        comparison.id,
        principal="principal:reviewer",
        expected_row_version=8,
        resolution_code="select_dl2",
        resolution_notes="DL2 is supported by the source evidence.",
    )
    assert payload["selected_pass_id"] == str(dl2.id)
    assert payload["selected_pass_number"] == 2
    assert payload["selected_pass_label"] == "DL2"


def test_resolution_rejects_dl1_or_dl2_principal_as_resolver(db_session):
    item, _, _, comparison, _ = _seed_resolution(db_session)
    with pytest.raises(WorkflowDiscrepancyResolutionConflict):
        resolve_workflow_comparison_discrepancies(
            db_session,
            item.id,
            comparison.id,
            principal="principal:dl1",
            expected_row_version=8,
            resolution_code="select_dl1",
            resolution_notes="Not allowed.",
        )
    db_session.rollback()

    item, _, _, comparison, _ = _seed_resolution(db_session)
    with pytest.raises(WorkflowDiscrepancyResolutionConflict):
        resolve_workflow_comparison_discrepancies(
            db_session,
            item.id,
            comparison.id,
            principal="principal:dl2",
            expected_row_version=8,
            resolution_code="select_dl2",
            resolution_notes="Not allowed.",
        )


def test_resolution_rejects_second_completed_comparison_for_current_pair(
    db_session,
):
    item, dl1, dl2, comparison, _ = _seed_resolution(db_session)
    duplicate = WorkflowComparison(
        id=uuid4(),
        workflow_item_id=item.id,
        left_pass_id=dl1.id,
        right_pass_id=dl2.id,
        comparison_version=2,
        status="complete",
        strict_equality_passed=False,
        difference_count=2,
        difference_summary={"difference_count": 2},
        checked_at=comparison.checked_at,
        checked_by_service_version="strict-comparison-duplicate",
        reviewed_by_principal=None,
        reviewed_at=None,
        created_at=comparison.created_at,
    )
    db_session.add(duplicate)
    db_session.flush()

    with pytest.raises(WorkflowDiscrepancyResolutionConflict):
        resolve_workflow_comparison_discrepancies(
            db_session,
            item.id,
            comparison.id,
            principal="principal:reviewer",
            expected_row_version=8,
            resolution_code="select_dl1",
            resolution_notes="DL1 supported.",
        )


def test_resolution_rejects_invalid_resolution_code(db_session):
    item, _, _, comparison, _ = _seed_resolution(db_session)
    with pytest.raises(WorkflowDiscrepancyResolutionError):
        resolve_workflow_comparison_discrepancies(
            db_session,
            item.id,
            comparison.id,
            principal="principal:reviewer",
            expected_row_version=8,
            resolution_code="merge_values",
            resolution_notes="No mixed merge is allowed.",
        )


def test_resolution_requires_bounded_nonempty_notes(db_session):
    item, _, _, comparison, _ = _seed_resolution(db_session)
    with pytest.raises(WorkflowDiscrepancyResolutionError):
        resolve_workflow_comparison_discrepancies(
            db_session,
            item.id,
            comparison.id,
            principal="principal:reviewer",
            expected_row_version=8,
            resolution_code="select_dl1",
            resolution_notes="   ",
        )
    with pytest.raises(WorkflowDiscrepancyResolutionError):
        resolve_workflow_comparison_discrepancies(
            db_session,
            item.id,
            comparison.id,
            principal="principal:reviewer",
            expected_row_version=8,
            resolution_code="select_dl1",
            resolution_notes="x" * 4001,
        )


def test_resolution_rejects_partial_or_nonpristine_discrepancy_set(db_session):
    item, _, _, comparison, discrepancies = _seed_resolution(db_session)
    discrepancies[0].resolution_status = "resolved"
    discrepancies[0].resolution_code = "select_dl1"
    discrepancies[0].resolution_notes = "partial"
    discrepancies[0].resolved_by_principal = "principal:reviewer"
    discrepancies[0].resolved_at = datetime.now(timezone.utc)

    with pytest.raises(WorkflowDiscrepancyResolutionConflict):
        resolve_workflow_comparison_discrepancies(
            db_session,
            item.id,
            comparison.id,
            principal="principal:reviewer",
            expected_row_version=8,
            resolution_code="select_dl1",
            resolution_notes="partial",
        )


def test_resolution_exact_completed_replay_is_idempotent(db_session):
    item, _, _, comparison, _ = _seed_resolution(db_session)
    first = resolve_workflow_comparison_discrepancies(
        db_session,
        item.id,
        comparison.id,
        principal="principal:reviewer",
        expected_row_version=8,
        resolution_code="select_dl1",
        resolution_notes="DL1 supported.",
    )
    assert first["row_version"] == 9

    second = resolve_workflow_comparison_discrepancies(
        db_session,
        item.id,
        comparison.id,
        principal="principal:reviewer",
        expected_row_version=9,
        resolution_code="select_dl1",
        resolution_notes="DL1 supported.",
    )
    assert second["already_resolved"] is True
    assert second["row_version"] == 9

    events = db_session.query(WorkflowEvent).filter(
        WorkflowEvent.workflow_item_id == item.id,
        WorkflowEvent.event_type == "discrepancy_resolution_completed",
    ).all()
    assert len(events) == 1


def test_resolution_replay_rejects_nonexact_reviewer_or_notes(db_session):
    item, _, _, comparison, _ = _seed_resolution(db_session)
    resolve_workflow_comparison_discrepancies(
        db_session,
        item.id,
        comparison.id,
        principal="principal:reviewer",
        expected_row_version=8,
        resolution_code="select_dl1",
        resolution_notes="DL1 supported.",
    )

    with pytest.raises(WorkflowDiscrepancyResolutionConflict):
        resolve_workflow_comparison_discrepancies(
            db_session,
            item.id,
            comparison.id,
            principal="principal:other-reviewer",
            expected_row_version=9,
            resolution_code="select_dl1",
            resolution_notes="DL1 supported.",
        )

    with pytest.raises(WorkflowDiscrepancyResolutionConflict):
        resolve_workflow_comparison_discrepancies(
            db_session,
            item.id,
            comparison.id,
            principal="principal:reviewer",
            expected_row_version=9,
            resolution_code="select_dl1",
            resolution_notes="different notes",
        )
