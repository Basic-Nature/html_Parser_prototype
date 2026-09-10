from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from webapp.parser.services.workflow_reviews import (
    WorkflowQCReviewConflict,
    WorkflowQCReviewError,
    record_workflow_qc_review,
)
from webapp.parser.utils.models import (
    Base,
    WorkflowComparison,
    WorkflowDiscrepancy,
    WorkflowEvent,
    WorkflowItem,
    WorkflowPass,
    WorkflowReview,
)


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Session = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )
    Base.metadata.create_all(engine)
    session = Session()
    try:
        yield session
    finally:
        session.rollback()
        session.close()
        engine.dispose()


def _seed_qc2(
    session,
    *,
    strict_equal=True,
    resolved_code="select_dl2",
    row_version=6,
    qc1_principal="principal:qc1",
):
    checked = datetime(2026, 9, 9, 10, 0, tzinfo=timezone.utc)
    item = WorkflowItem(
        id=uuid4(),
        lifecycle_state="active",
        current_stage="qc1_review",
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
        row_version=row_version,
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
        staging_batch_id=uuid4(),
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
        staging_batch_id=uuid4(),
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
        strict_equality_passed=bool(strict_equal),
        difference_count=0 if strict_equal else 2,
        difference_summary={
            "difference_count": 0 if strict_equal else 2,
        },
        checked_at=checked,
        checked_by_service_version="strict-comparison-test",
        reviewed_by_principal=(
            None if strict_equal else "principal:resolver"
        ),
        reviewed_at=None if strict_equal else checked,
        created_at=checked,
    )
    rows = [item, dl1, dl2, comparison]
    discrepancies = []
    if not strict_equal:
        for index in range(2):
            discrepancy = WorkflowDiscrepancy(
                id=uuid4(),
                comparison_id=comparison.id,
                workflow_item_id=item.id,
                category="value_mismatch",
                semantic_key=["records", f"Precinct {index + 1}"],
                left_value=10,
                right_value=11,
                left_value_state="value",
                right_value_state="value",
                severity=None,
                resolution_status="resolved",
                resolution_code=resolved_code,
                resolution_notes="Resolved from official evidence.",
                resolved_by_principal="principal:resolver",
                resolved_at=checked,
                created_at=checked,
            )
            discrepancies.append(discrepancy)
            rows.append(discrepancy)
    session.add_all(rows)
    session.flush()

    qc1_payload = record_workflow_qc_review(
        session,
        item.id,
        review_stage="qc1",
        principal=qc1_principal,
        expected_row_version=row_version,
        decision="approved",
        checklist_version="qc1-v1",
        checklist_result={
            "source_matches_scope": True,
            "totals_reconcile": True,
        },
        reason_codes=[],
        notes="",
        now=checked,
    )
    qc1_review = (
        session.query(WorkflowReview)
        .filter(WorkflowReview.review_stage == "qc1")
        .one()
    )
    qc1_event = (
        session.query(WorkflowEvent)
        .filter(WorkflowEvent.event_type == "qc1_review_approved")
        .one()
    )
    assert qc1_payload["review_id"] == str(qc1_review.id)
    assert item.current_stage == "qc2_review"
    assert item.stage_condition == "pending"
    return (
        item,
        dl1,
        dl2,
        comparison,
        discrepancies,
        qc1_review,
        qc1_event,
    )


def _review_qc2(session, item, **overrides):
    kwargs = {
        "review_stage": "qc2",
        "principal": "principal:qc2",
        "expected_row_version": item.row_version,
        "decision": "approved",
        "checklist_version": "qc2-v1",
        "checklist_result": {
            "selected_pass_reconciles": True,
            "publication_ready": True,
        },
        "reason_codes": [],
        "notes": "",
    }
    kwargs.update(overrides)
    return record_workflow_qc_review(session, item.id, **kwargs)


def test_equal_approved_inherits_qc1_dl1_and_advances_publication_handoff(
    db_session,
):
    item, dl1, dl2, comparison, _, qc1, _ = _seed_qc2(db_session)
    payload = _review_qc2(db_session, item)
    assert payload["committed"] is False
    assert payload["already_reviewed"] is False
    assert payload["qc1_review_id"] == str(qc1.id)
    assert payload["selected_pass_id"] == str(dl1.id)
    assert payload["selected_staging_batch_id"] == str(
        dl1.staging_batch_id
    )
    assert (
        item.lifecycle_state,
        item.current_stage,
        item.stage_condition,
        item.row_version,
    ) == (
        "ready_for_publication",
        "publication_handoff",
        "ready",
        8,
    )
    reviews = (
        db_session.query(WorkflowReview)
        .filter(WorkflowReview.review_stage == "qc2")
        .all()
    )
    assert len(reviews) == 1
    review = reviews[0]
    assert review.selected_pass_id == dl1.id
    assert review.selected_staging_batch_id == dl1.staging_batch_id
    events = (
        db_session.query(WorkflowEvent)
        .filter(WorkflowEvent.event_type == "qc2_review_approved")
        .all()
    )
    assert len(events) == 1
    event = events[0]
    assert event.related_comparison_id == comparison.id
    assert event.related_review_id == review.id
    assert event.event_metadata["qc1_review_id"] == str(qc1.id)
    assert (
        event.event_metadata["selection_reason"]
        == "approved_qc1_review_inherited"
    )
    assert event.event_metadata["canonical_writer_invoked"] is False
    assert dl1.status == "submitted"
    assert dl2.status == "submitted"


def test_mismatch_qc2_inherits_qc1_selected_dl2(db_session):
    (
        item,
        _dl1,
        dl2,
        _comparison,
        _discrepancies,
        qc1,
        _event,
    ) = _seed_qc2(
        db_session,
        strict_equal=False,
        resolved_code="select_dl2",
        row_version=9,
    )
    payload = _review_qc2(
        db_session,
        item,
        expected_row_version=10,
    )
    assert qc1.selected_pass_id == dl2.id
    assert payload["selected_pass_id"] == str(dl2.id)
    assert payload["selected_pass_label"] == "DL2"
    assert item.current_stage == "publication_handoff"
    assert item.row_version == 11


def test_qc2_reviewer_must_differ_from_dl1_dl2_and_qc1(db_session):
    for principal in (
        "principal:dl1",
        "principal:dl2",
        "principal:qc1",
    ):
        item, *_ = _seed_qc2(db_session)
        with pytest.raises(WorkflowQCReviewConflict):
            _review_qc2(db_session, item, principal=principal)
        db_session.rollback()


def test_qc2_may_be_existing_discrepancy_resolver(db_session):
    item, _dl1, dl2, *_ = _seed_qc2(
        db_session,
        strict_equal=False,
        resolved_code="select_dl2",
    )
    payload = _review_qc2(
        db_session,
        item,
        principal="principal:resolver",
    )
    assert payload["reviewer_principal"] == "principal:resolver"
    assert payload["selected_pass_id"] == str(dl2.id)


def test_returned_stays_qc2_awaiting_dependency(db_session):
    item, dl1, *_ = _seed_qc2(db_session)
    payload = _review_qc2(
        db_session,
        item,
        decision="returned",
        checklist_result={
            "selected_pass_reconciles": True,
            "publication_ready": False,
        },
        reason_codes=["publication_not_ready"],
        notes="Return for governed prerequisite correction.",
    )
    assert payload["selected_pass_id"] == str(dl1.id)
    assert (
        item.lifecycle_state,
        item.current_stage,
        item.stage_condition,
        item.row_version,
    ) == (
        "active",
        "qc2_review",
        "awaiting_dependency",
        8,
    )
    assert item.blocked_reason_code is None
    assert item.blocker_detail is None


def test_rejected_blocks_qc2(db_session):
    item, *_ = _seed_qc2(db_session)
    _review_qc2(
        db_session,
        item,
        decision="rejected",
        checklist_result={
            "selected_pass_reconciles": False,
            "publication_ready": False,
        },
        reason_codes=["publication_rejected"],
        notes="QC2 found evidence unsuitable for publication.",
    )
    assert (
        item.lifecycle_state,
        item.current_stage,
        item.stage_condition,
        item.row_version,
    ) == (
        "blocked",
        "qc2_review",
        "failed",
        8,
    )
    assert item.blocked_reason_code == "qc2_rejected"
    assert (
        item.blocker_detail
        == "QC2 found evidence unsuitable for publication."
    )


def test_qc2_approved_requires_all_checks_true_and_no_reasons(db_session):
    item, *_ = _seed_qc2(db_session)
    with pytest.raises(WorkflowQCReviewError):
        _review_qc2(
            db_session,
            item,
            checklist_result={
                "selected_pass_reconciles": True,
                "publication_ready": False,
            },
        )
    with pytest.raises(WorkflowQCReviewError):
        _review_qc2(
            db_session,
            item,
            reason_codes=["manual_override"],
        )


def test_qc2_returned_and_rejected_require_reasons_and_notes(db_session):
    item, *_ = _seed_qc2(db_session)
    with pytest.raises(WorkflowQCReviewError):
        _review_qc2(
            db_session,
            item,
            decision="returned",
            reason_codes=[],
            notes="Needs work.",
        )
    with pytest.raises(WorkflowQCReviewError):
        _review_qc2(
            db_session,
            item,
            decision="rejected",
            reason_codes=["publication_rejected"],
            notes="   ",
        )


def test_qc2_exact_replay_is_idempotent(db_session):
    item, *_ = _seed_qc2(db_session)
    first = _review_qc2(db_session, item)
    second = _review_qc2(
        db_session,
        item,
        expected_row_version=8,
    )
    assert second["already_reviewed"] is True
    assert second["review_id"] == first["review_id"]
    assert second["row_version"] == 8
    assert (
        db_session.query(WorkflowReview)
        .filter(WorkflowReview.review_stage == "qc2")
        .count()
        == 1
    )
    assert (
        db_session.query(WorkflowEvent)
        .filter(WorkflowEvent.event_type == "qc2_review_approved")
        .count()
        == 1
    )


def test_qc2_nonexact_replay_conflicts(db_session):
    item, *_ = _seed_qc2(db_session)
    _review_qc2(db_session, item)
    with pytest.raises(WorkflowQCReviewConflict):
        _review_qc2(
            db_session,
            item,
            expected_row_version=8,
            principal="principal:other-qc2",
        )


def test_qc2_rejects_qc1_selection_authority_drift(db_session):
    item, dl1, dl2, _comparison, _ds, qc1, _event = _seed_qc2(
        db_session
    )
    assert qc1.selected_pass_id == dl1.id
    qc1.selected_pass_id = dl2.id
    qc1.selected_staging_batch_id = dl2.staging_batch_id
    db_session.flush()
    with pytest.raises(WorkflowQCReviewConflict):
        _review_qc2(db_session, item)
