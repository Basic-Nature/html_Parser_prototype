from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from webapp.parser.contracts.workflow_lifecycle import (
    DISCREPANCY_RESOLUTION_SELECTION_CODES,
    REVIEW_DECISIONS,
    REVIEW_STAGES,
    assert_forward_stage_transition,
)
from webapp.parser.utils.models import (
    WorkflowComparison,
    WorkflowDiscrepancy,
    WorkflowEvent,
    WorkflowItem,
    WorkflowPass,
    WorkflowReview,
)

WORKFLOW_QC_REVIEW_CONTRACT = "workflow_qc_review_v1"
WORKFLOW_QC_REVIEW_SERVICE = "workflow_reviews"
_MAX_NOTES = 4000
_MAX_TOKEN = 64
_TOKEN_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")


class WorkflowQCReviewError(RuntimeError):
    status_code = 400
    code = "workflow_qc_review_error"


class WorkflowQCReviewNotFound(WorkflowQCReviewError):
    status_code = 404
    code = "workflow_qc_review_not_found"


class WorkflowQCReviewConflict(WorkflowQCReviewError):
    status_code = 409
    code = "workflow_qc_review_conflict"


def _uuid(value: UUID | str, *, name: str) -> UUID:
    try:
        return value if isinstance(value, UUID) else UUID(str(value))
    except (TypeError, ValueError) as exc:
        raise WorkflowQCReviewError(f"{name} must be a UUID.") from exc


def _expected_version(value: object) -> int:
    if isinstance(value, bool):
        raise WorkflowQCReviewError("expected_row_version must be an integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise WorkflowQCReviewError(
            "expected_row_version must be an integer."
        ) from exc
    if parsed < 0:
        raise WorkflowQCReviewError("expected_row_version must be >= 0.")
    return parsed


def _actor(principal: object) -> str:
    actor = str(principal or "").strip()
    if not actor:
        raise WorkflowQCReviewError(
            "Authenticated reviewer principal is required."
        )
    return actor


def _review_stage(value: object) -> str:
    stage = str(value or "").strip()
    if stage not in REVIEW_STAGES:
        raise WorkflowQCReviewError(
            "review_stage must be server-owned qc1 or qc2."
        )
    return stage


def _decision(value: object) -> str:
    decision = str(value or "").strip()
    if decision not in REVIEW_DECISIONS:
        raise WorkflowQCReviewError(
            "decision must be approved, returned, or rejected."
        )
    return decision


def _checklist_version(value: object) -> str:
    version = str(value or "").strip()
    if not version:
        raise WorkflowQCReviewError("checklist_version must be non-empty.")
    if len(version) > _MAX_TOKEN:
        raise WorkflowQCReviewError(
            "checklist_version must be <= 64 characters."
        )
    return version


def _checklist_result(value: object) -> dict[str, bool]:
    if not isinstance(value, dict) or not value:
        raise WorkflowQCReviewError(
            "checklist_result must be a non-empty object."
        )
    normalized: dict[str, bool] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key or "").strip()
        if _TOKEN_RE.fullmatch(key) is None:
            raise WorkflowQCReviewError(
                "checklist_result keys must be stable lowercase check IDs."
            )
        if not isinstance(raw_value, bool):
            raise WorkflowQCReviewError(
                "checklist_result values must be bool."
            )
        normalized[key] = raw_value
    return normalized


def _reason_codes(value: object) -> list[str]:
    if not isinstance(value, list):
        raise WorkflowQCReviewError("reason_codes must be an array.")
    normalized: list[str] = []
    for raw in value:
        code = str(raw or "").strip()
        if _TOKEN_RE.fullmatch(code) is None:
            raise WorkflowQCReviewError(
                "reason_codes must contain stable lowercase codes."
            )
        if code in normalized:
            raise WorkflowQCReviewError(
                "reason_codes must not contain duplicates."
            )
        normalized.append(code)
    return normalized


def _notes(value: object) -> str:
    notes = str(value or "").strip()
    if len(notes) > _MAX_NOTES:
        raise WorkflowQCReviewError("notes must be <= 4000 characters.")
    return notes


def _validate_review_payload(
    *,
    review_stage: str,
    decision: object,
    checklist_version: object,
    checklist_result: object,
    reason_codes: object,
    notes: object,
) -> tuple[str, str, dict[str, bool], list[str], str]:
    d = _decision(decision)
    v = _checklist_version(checklist_version)
    c = _checklist_result(checklist_result)
    r = _reason_codes(reason_codes)
    n = _notes(notes)
    label = review_stage.upper()
    if d == "approved":
        if not all(c.values()):
            raise WorkflowQCReviewError(
                f"approved {label} review requires every checklist result true."
            )
        if r:
            raise WorkflowQCReviewError(
                f"approved {label} review requires empty reason_codes."
            )
    else:
        if not r:
            raise WorkflowQCReviewError(
                f"returned or rejected {label} review requires reason_codes."
            )
        if not n:
            raise WorkflowQCReviewError(
                f"returned or rejected {label} review requires notes."
            )
    return d, v, c, r, n


def _utc(now: datetime | None) -> datetime:
    value = now or datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _load_current_submitted_pair(
    session: Session,
    item: WorkflowItem,
) -> tuple[WorkflowPass, WorkflowPass]:
    rows = session.execute(
        select(WorkflowPass)
        .where(
            WorkflowPass.workflow_item_id == item.id,
            WorkflowPass.pass_number.in_((1, 2)),
            WorkflowPass.is_current.is_(True),
        )
        .with_for_update()
    ).scalars().all()
    if len(rows) != 2:
        raise WorkflowQCReviewConflict(
            "QC review requires exactly one current DL1 and DL2."
        )
    by_pair = {
        (int(row.pass_number), str(row.pass_label)): row
        for row in rows
    }
    if set(by_pair) != {(1, "DL1"), (2, "DL2")}:
        raise WorkflowQCReviewConflict(
            "Current acquisition pass identities are invalid."
        )
    dl1 = by_pair[(1, "DL1")]
    dl2 = by_pair[(2, "DL2")]
    p1 = str(dl1.assigned_principal or "").strip()
    p2 = str(dl2.assigned_principal or "").strip()
    if (
        dl1.status != "submitted"
        or dl2.status != "submitted"
        or dl1.submitted_at is None
        or dl2.submitted_at is None
        or dl1.staging_batch_id is None
        or dl2.staging_batch_id is None
        or not p1
        or not p2
        or p1 == p2
    ):
        raise WorkflowQCReviewConflict(
            "QC review requires immutable submitted independent DL1/DL2 "
            "with staging bindings."
        )
    return dl1, dl2


def _load_exact_comparison(
    session: Session,
    item: WorkflowItem,
    dl1: WorkflowPass,
    dl2: WorkflowPass,
) -> WorkflowComparison:
    rows = session.execute(
        select(WorkflowComparison)
        .where(
            WorkflowComparison.workflow_item_id == item.id,
            WorkflowComparison.left_pass_id == dl1.id,
            WorkflowComparison.right_pass_id == dl2.id,
            WorkflowComparison.status == "complete",
        )
        .with_for_update()
    ).scalars().all()
    if len(rows) != 1:
        raise WorkflowQCReviewConflict(
            "QC review requires exactly one completed comparison for the "
            "current submitted DL1/DL2 pair."
        )
    comparison = rows[0]
    if (
        comparison.checked_at is None
        or not str(comparison.checked_by_service_version or "").strip()
        or not isinstance(comparison.difference_count, int)
        or comparison.difference_count < 0
    ):
        raise WorkflowQCReviewConflict(
            "Completed comparison authority is incomplete."
        )
    return comparison


def _derive_selected_pass(
    session: Session,
    item: WorkflowItem,
    dl1: WorkflowPass,
    dl2: WorkflowPass,
    comparison: WorkflowComparison,
) -> tuple[WorkflowPass, str]:
    discrepancies = session.execute(
        select(WorkflowDiscrepancy)
        .where(
            WorkflowDiscrepancy.workflow_item_id == item.id,
            WorkflowDiscrepancy.comparison_id == comparison.id,
        )
        .order_by(WorkflowDiscrepancy.id)
        .with_for_update()
    ).scalars().all()
    if comparison.strict_equality_passed is True:
        if (
            comparison.difference_count != 0
            or discrepancies
            or comparison.reviewed_by_principal is not None
            or comparison.reviewed_at is not None
        ):
            raise WorkflowQCReviewConflict(
                "Strict-equal comparison does not reconcile with zero "
                "discrepancies and no resolution fields."
            )
        return dl1, "strict_equal_dl1"
    if comparison.strict_equality_passed is not False:
        raise WorkflowQCReviewConflict(
            "Comparison strict_equality_passed must be bool."
        )
    if (
        comparison.difference_count <= 0
        or len(discrepancies) != comparison.difference_count
        or not str(comparison.reviewed_by_principal or "").strip()
        or comparison.reviewed_at is None
    ):
        raise WorkflowQCReviewConflict(
            "Resolved mismatch comparison authority is incomplete."
        )
    codes = {row.resolution_code for row in discrepancies}
    resolvers = {
        str(row.resolved_by_principal or "").strip()
        for row in discrepancies
    }
    statuses = {row.resolution_status for row in discrepancies}
    notes = {str(row.resolution_notes or "").strip() for row in discrepancies}
    if (
        statuses != {"resolved"}
        or len(codes) != 1
        or next(iter(codes)) not in DISCREPANCY_RESOLUTION_SELECTION_CODES
        or resolvers
        != {str(comparison.reviewed_by_principal).strip()}
        or "" in resolvers
        or len(notes) != 1
        or "" in notes
        or any(row.resolved_at is None for row in discrepancies)
    ):
        raise WorkflowQCReviewConflict(
            "Mismatch discrepancies must be one complete uniformly resolved "
            "W9 selection."
        )
    code = next(iter(codes))
    return (
        dl1 if code == "select_dl1" else dl2,
        str(code),
    )


def _load_approved_qc1_authority(
    session: Session,
    item: WorkflowItem,
    *,
    dl1: WorkflowPass,
    dl2: WorkflowPass,
    comparison: WorkflowComparison,
    derived_selected: WorkflowPass,
) -> WorkflowReview:
    reviews = session.execute(
        select(WorkflowReview)
        .where(
            WorkflowReview.workflow_item_id == item.id,
            WorkflowReview.review_stage == "qc1",
        )
        .with_for_update()
    ).scalars().all()
    if len(reviews) != 1:
        raise WorkflowQCReviewConflict(
            "QC2 requires exactly one QC1 review authority."
        )
    review = reviews[0]
    reviewer = str(review.reviewer_principal or "").strip()
    dl_principals = {
        str(dl1.assigned_principal).strip(),
        str(dl2.assigned_principal).strip(),
    }
    if (
        review.decision != "approved"
        or not reviewer
        or reviewer in dl_principals
        or review.selected_pass_id != derived_selected.id
        or review.selected_staging_batch_id
        != derived_selected.staging_batch_id
        or not str(review.checklist_version or "").strip()
        or not isinstance(review.checklist_result, dict)
        or not review.checklist_result
        or not all(value is True for value in review.checklist_result.values())
        or (review.reason_codes or []) != []
        or review.reviewed_at is None
    ):
        raise WorkflowQCReviewConflict(
            "QC2 requires one approved QC1 review that exactly reconciles "
            "with the current server-selected pass and staging authority."
        )

    events = session.execute(
        select(WorkflowEvent).where(
            WorkflowEvent.workflow_item_id == item.id,
            WorkflowEvent.related_review_id == review.id,
            WorkflowEvent.event_type == "qc1_review_approved",
        )
    ).scalars().all()
    if len(events) != 1:
        raise WorkflowQCReviewConflict(
            "QC2 requires one exact QC1 approval audit event."
        )
    event = events[0]
    metadata = (
        event.event_metadata if isinstance(event.event_metadata, dict) else {}
    )
    if (
        event.actor_type != "principal"
        or event.actor_principal != reviewer
        or event.actor_service is not None
        or event.stage != "qc1_review"
        or event.related_pass_id != derived_selected.id
        or event.related_staging_batch_id
        != derived_selected.staging_batch_id
        or event.related_comparison_id != comparison.id
        or metadata.get("contract") != WORKFLOW_QC_REVIEW_CONTRACT
        or metadata.get("review_stage") != "qc1"
        or metadata.get("decision") != "approved"
        or metadata.get("selected_pass_id") != str(derived_selected.id)
        or metadata.get("selected_staging_batch_id")
        != str(derived_selected.staging_batch_id)
        or metadata.get("canonical_writer_invoked") is not False
        or metadata.get("client_selected_pass_authority") is not False
        or metadata.get("mixed_side_value_merge") is not False
        or metadata.get("direct_value_edit") is not False
    ):
        raise WorkflowQCReviewConflict(
            "QC1 approval audit provenance does not reconcile for QC2."
        )
    return review


def _expected_effect(
    review_stage: str,
    decision: str,
) -> dict[str, object]:
    if review_stage == "qc1":
        if decision == "approved":
            return {
                "lifecycle_state": "active",
                "current_stage": "qc2_review",
                "stage_condition": "pending",
                "blocked_reason_code": None,
                "event_type": "qc1_review_approved",
            }
        if decision == "returned":
            return {
                "lifecycle_state": "active",
                "current_stage": "qc1_review",
                "stage_condition": "awaiting_dependency",
                "blocked_reason_code": None,
                "event_type": "qc1_review_returned",
            }
        return {
            "lifecycle_state": "blocked",
            "current_stage": "qc1_review",
            "stage_condition": "failed",
            "blocked_reason_code": "qc1_rejected",
            "event_type": "qc1_review_rejected",
        }

    if decision == "approved":
        return {
            "lifecycle_state": "ready_for_publication",
            "current_stage": "publication_handoff",
            "stage_condition": "ready",
            "blocked_reason_code": None,
            "event_type": "qc2_review_approved",
        }
    if decision == "returned":
        return {
            "lifecycle_state": "active",
            "current_stage": "qc2_review",
            "stage_condition": "awaiting_dependency",
            "blocked_reason_code": None,
            "event_type": "qc2_review_returned",
        }
    return {
        "lifecycle_state": "blocked",
        "current_stage": "qc2_review",
        "stage_condition": "failed",
        "blocked_reason_code": "qc2_rejected",
        "event_type": "qc2_review_rejected",
    }


def _exact_replay(
    session: Session,
    item: WorkflowItem,
    comparison: WorkflowComparison,
    selected: WorkflowPass,
    *,
    review_stage: str,
    actor: str,
    decision: str,
    checklist_version: str,
    checklist_result: dict[str, bool],
    reason_codes: list[str],
    notes: str,
    qc1_authority: WorkflowReview | None,
) -> dict[str, Any] | None:
    reviews = session.execute(
        select(WorkflowReview)
        .where(
            WorkflowReview.workflow_item_id == item.id,
            WorkflowReview.review_stage == review_stage,
        )
        .with_for_update()
    ).scalars().all()
    if not reviews:
        return None
    if len(reviews) != 1:
        raise WorkflowQCReviewConflict(
            f"{review_stage.upper()} review set is not uniquely replayable."
        )
    review = reviews[0]
    effect = _expected_effect(review_stage, decision)
    expected_blocker_detail = notes if decision == "rejected" else None
    if (
        review.reviewer_principal != actor
        or review.decision != decision
        or review.selected_pass_id != selected.id
        or review.selected_staging_batch_id != selected.staging_batch_id
        or review.checklist_version != checklist_version
        or review.checklist_result != checklist_result
        or review.reason_codes != reason_codes
        or (review.notes or "") != notes
        or review.reviewed_at is None
        or item.lifecycle_state != effect["lifecycle_state"]
        or item.current_stage != effect["current_stage"]
        or item.stage_condition != effect["stage_condition"]
        or item.blocked_reason_code != effect["blocked_reason_code"]
        or item.blocker_detail != expected_blocker_detail
    ):
        raise WorkflowQCReviewConflict(
            f"Existing {review_stage.upper()} review is not an exact replay."
        )
    events = session.execute(
        select(WorkflowEvent).where(
            WorkflowEvent.workflow_item_id == item.id,
            WorkflowEvent.related_review_id == review.id,
            WorkflowEvent.event_type == effect["event_type"],
        )
    ).scalars().all()
    if len(events) != 1:
        raise WorkflowQCReviewConflict(
            f"Existing {review_stage.upper()} review audit event set is not exact."
        )
    event = events[0]
    metadata = (
        event.event_metadata if isinstance(event.event_metadata, dict) else {}
    )
    if (
        event.actor_type != "principal"
        or event.actor_principal != actor
        or event.actor_service is not None
        or event.stage != f"{review_stage}_review"
        or event.related_pass_id != selected.id
        or event.related_staging_batch_id != selected.staging_batch_id
        or event.related_comparison_id != comparison.id
        or metadata.get("contract") != WORKFLOW_QC_REVIEW_CONTRACT
        or metadata.get("review_stage") != review_stage
        or metadata.get("decision") != decision
        or metadata.get("selected_pass_id") != str(selected.id)
        or metadata.get("selected_staging_batch_id")
        != str(selected.staging_batch_id)
        or metadata.get("canonical_writer_invoked") is not False
        or metadata.get("client_selected_pass_authority") is not False
        or metadata.get("mixed_side_value_merge") is not False
        or metadata.get("direct_value_edit") is not False
    ):
        raise WorkflowQCReviewConflict(
            f"Existing {review_stage.upper()} review audit event does not reconcile."
        )
    if review_stage == "qc2":
        if (
            qc1_authority is None
            or metadata.get("qc1_review_id") != str(qc1_authority.id)
            or metadata.get("selection_reason")
            != "approved_qc1_review_inherited"
        ):
            raise WorkflowQCReviewConflict(
                "Existing QC2 review does not preserve QC1 selection authority."
            )
    return {
        "success": True,
        "contract": WORKFLOW_QC_REVIEW_CONTRACT,
        "task_id": str(item.id),
        "review_id": str(review.id),
        "review_stage": review_stage,
        "decision": decision,
        "reviewer_principal": actor,
        "selected_pass_id": str(selected.id),
        "selected_pass_number": int(selected.pass_number),
        "selected_pass_label": str(selected.pass_label),
        "selected_staging_batch_id": str(selected.staging_batch_id),
        "comparison_id": str(comparison.id),
        "qc1_review_id": (
            str(qc1_authority.id)
            if qc1_authority is not None
            else None
        ),
        "current_stage": item.current_stage,
        "stage_condition": item.stage_condition,
        "lifecycle_state": item.lifecycle_state,
        "row_version": int(item.row_version),
        "event_id": str(event.id),
        "already_reviewed": True,
        "committed": False,
    }


def record_workflow_qc_review(
    session: Session,
    item_id: UUID | str,
    *,
    review_stage: str,
    principal: str,
    expected_row_version: int,
    decision: str,
    checklist_version: str,
    checklist_result: dict[str, bool],
    reason_codes: list[str],
    notes: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    normalized_item = _uuid(item_id, name="item_id")
    stage = _review_stage(review_stage)
    actor = _actor(principal)
    expected = _expected_version(expected_row_version)
    d, cv, cr, rc, nt = _validate_review_payload(
        review_stage=stage,
        decision=decision,
        checklist_version=checklist_version,
        checklist_result=checklist_result,
        reason_codes=reason_codes,
        notes=notes,
    )
    timestamp = _utc(now)

    item = session.execute(
        select(WorkflowItem)
        .where(WorkflowItem.id == normalized_item)
        .with_for_update()
    ).scalar_one_or_none()
    if item is None:
        raise WorkflowQCReviewNotFound("Workflow item was not found.")
    if int(item.row_version) != expected:
        raise WorkflowQCReviewConflict(
            f"Workflow row_version changed before {stage.upper()} review."
        )

    dl1, dl2 = _load_current_submitted_pair(session, item)
    comparison = _load_exact_comparison(session, item, dl1, dl2)
    selected, selection_reason = _derive_selected_pass(
        session,
        item,
        dl1,
        dl2,
        comparison,
    )

    dl_principals = {
        str(dl1.assigned_principal).strip(),
        str(dl2.assigned_principal).strip(),
    }
    qc1_authority: WorkflowReview | None = None
    if stage == "qc1":
        if actor in dl_principals:
            raise WorkflowQCReviewConflict(
                "QC1 reviewer must differ from DL1 and DL2 principals."
            )
    else:
        qc1_authority = _load_approved_qc1_authority(
            session,
            item,
            dl1=dl1,
            dl2=dl2,
            comparison=comparison,
            derived_selected=selected,
        )
        qc1_principal = str(qc1_authority.reviewer_principal).strip()
        if actor in dl_principals or actor == qc1_principal:
            raise WorkflowQCReviewConflict(
                "QC2 reviewer must differ from DL1, DL2, and QC1 principals."
            )
        selection_reason = "approved_qc1_review_inherited"

    replay = _exact_replay(
        session,
        item,
        comparison,
        selected,
        review_stage=stage,
        actor=actor,
        decision=d,
        checklist_version=cv,
        checklist_result=cr,
        reason_codes=rc,
        notes=nt,
        qc1_authority=qc1_authority,
    )
    if replay is not None:
        return replay

    expected_stage = f"{stage}_review"
    if (
        item.lifecycle_state,
        item.current_stage,
        item.stage_condition,
    ) != ("active", expected_stage, "pending"):
        raise WorkflowQCReviewConflict(
            f"Workflow item is not pending {stage.upper()} review."
        )

    if d == "approved":
        next_stage = (
            "qc2_review"
            if stage == "qc1"
            else "publication_handoff"
        )
        assert_forward_stage_transition(expected_stage, next_stage)

    prior_state = {
        "lifecycle_state": item.lifecycle_state,
        "current_stage": item.current_stage,
        "stage_condition": item.stage_condition,
        "blocked_reason_code": item.blocked_reason_code,
        "blocker_detail": item.blocker_detail,
        "row_version": item.row_version,
    }

    review = WorkflowReview(
        workflow_item_id=item.id,
        review_stage=stage,
        reviewer_principal=actor,
        decision=d,
        selected_pass_id=selected.id,
        selected_staging_batch_id=selected.staging_batch_id,
        checklist_version=cv,
        checklist_result=cr,
        reason_codes=rc,
        notes=nt,
        reviewed_at=timestamp,
    )
    session.add(review)
    session.flush()

    effect = _expected_effect(stage, d)
    item.lifecycle_state = str(effect["lifecycle_state"])
    item.current_stage = str(effect["current_stage"])
    item.stage_condition = str(effect["stage_condition"])
    item.blocked_reason_code = effect["blocked_reason_code"]
    item.blocker_detail = nt if d == "rejected" else None
    item.row_version = expected + 1
    item.updated_at = timestamp

    event_metadata = {
        "contract": WORKFLOW_QC_REVIEW_CONTRACT,
        "review_stage": stage,
        "decision": d,
        "checklist_version": cv,
        "reason_codes": list(rc),
        "selected_pass_id": str(selected.id),
        "selected_pass_number": int(selected.pass_number),
        "selected_pass_label": str(selected.pass_label),
        "selected_staging_batch_id": str(selected.staging_batch_id),
        "selection_reason": selection_reason,
        "client_selected_pass_authority": False,
        "mixed_side_value_merge": False,
        "direct_value_edit": False,
        "canonical_writer_invoked": False,
    }
    if qc1_authority is not None:
        event_metadata["qc1_review_id"] = str(qc1_authority.id)

    event = WorkflowEvent(
        workflow_item_id=item.id,
        actor_type="principal",
        actor_principal=actor,
        actor_service=None,
        event_type=str(effect["event_type"]),
        stage=expected_stage,
        prior_state=prior_state,
        new_state={
            "lifecycle_state": item.lifecycle_state,
            "current_stage": item.current_stage,
            "stage_condition": item.stage_condition,
            "blocked_reason_code": item.blocked_reason_code,
            "blocker_detail": item.blocker_detail,
            "row_version": item.row_version,
        },
        related_pass_id=selected.id,
        related_comparison_id=comparison.id,
        related_review_id=review.id,
        related_staging_batch_id=selected.staging_batch_id,
        related_canonical_race_id=item.canonical_race_id,
        reason_code=None if d == "approved" else rc[0],
        summary=f"{stage.upper()} reviewer recorded {d} decision.",
        event_metadata=event_metadata,
    )
    session.add(event)
    session.flush()

    return {
        "success": True,
        "contract": WORKFLOW_QC_REVIEW_CONTRACT,
        "task_id": str(item.id),
        "review_id": str(review.id),
        "review_stage": stage,
        "decision": d,
        "reviewer_principal": actor,
        "selected_pass_id": str(selected.id),
        "selected_pass_number": int(selected.pass_number),
        "selected_pass_label": str(selected.pass_label),
        "selected_staging_batch_id": str(selected.staging_batch_id),
        "comparison_id": str(comparison.id),
        "qc1_review_id": (
            str(qc1_authority.id)
            if qc1_authority is not None
            else None
        ),
        "current_stage": item.current_stage,
        "stage_condition": item.stage_condition,
        "lifecycle_state": item.lifecycle_state,
        "row_version": int(item.row_version),
        "event_id": str(event.id),
        "already_reviewed": False,
        "committed": False,
    }


def load_workflow_publication_approval_authority(
    session: Session,
    item_id: UUID | str,
    *,
    require_ready_state: bool = True,
) -> dict[str, Any]:
    """Reconstruct the exact QC-approved publication authority server-side.

    This is a read/lock helper for the publication orchestrator. It does not
    mutate Workflow or canonical state and never calls the canonical writer.
    """
    normalized_item = _uuid(item_id, name="item_id")
    item = session.execute(
        select(WorkflowItem)
        .where(WorkflowItem.id == normalized_item)
        .with_for_update()
    ).scalar_one_or_none()
    if item is None:
        raise WorkflowQCReviewConflict("Workflow item was not found.")

    dl1, dl2 = _load_current_submitted_pair(session, item)
    comparison = _load_exact_comparison(session, item, dl1, dl2)
    selected, selection_reason = _derive_selected_pass(
        session,
        item,
        dl1,
        dl2,
        comparison,
    )
    qc1 = _load_approved_qc1_authority(
        session,
        item,
        dl1=dl1,
        dl2=dl2,
        comparison=comparison,
        derived_selected=selected,
    )

    qc2_rows = session.execute(
        select(WorkflowReview)
        .where(
            WorkflowReview.workflow_item_id == item.id,
            WorkflowReview.review_stage == "qc2",
        )
        .with_for_update()
    ).scalars().all()
    if len(qc2_rows) != 1:
        raise WorkflowQCReviewConflict(
            "Publication requires exactly one QC2 review authority."
        )
    qc2 = qc2_rows[0]
    qc1_principal = str(qc1.reviewer_principal or "").strip()
    qc2_principal = str(qc2.reviewer_principal or "").strip()
    dl1_principal = str(dl1.assigned_principal or "").strip()
    dl2_principal = str(dl2.assigned_principal or "").strip()
    if (
        qc2.decision != "approved"
        or not qc2_principal
        or qc2_principal in {dl1_principal, dl2_principal, qc1_principal}
        or qc2.selected_pass_id != selected.id
        or qc2.selected_staging_batch_id != selected.staging_batch_id
        or not str(qc2.checklist_version or "").strip()
        or not isinstance(qc2.checklist_result, dict)
        or not qc2.checklist_result
        or not all(value is True for value in qc2.checklist_result.values())
        or (qc2.reason_codes or []) != []
        or qc2.reviewed_at is None
    ):
        raise WorkflowQCReviewConflict(
            "Publication requires one approved QC2 review that exactly "
            "inherits the approved QC1 selected pass and staging authority."
        )

    qc2_events = session.execute(
        select(WorkflowEvent).where(
            WorkflowEvent.workflow_item_id == item.id,
            WorkflowEvent.related_review_id == qc2.id,
            WorkflowEvent.event_type == "qc2_review_approved",
        )
    ).scalars().all()
    if len(qc2_events) != 1:
        raise WorkflowQCReviewConflict(
            "Publication requires one exact QC2 approval audit event."
        )
    qc2_event = qc2_events[0]
    metadata = (
        qc2_event.event_metadata
        if isinstance(qc2_event.event_metadata, dict)
        else {}
    )
    if (
        qc2_event.actor_type != "principal"
        or qc2_event.actor_principal != qc2_principal
        or qc2_event.actor_service is not None
        or qc2_event.stage != "qc2_review"
        or qc2_event.related_pass_id != selected.id
        or qc2_event.related_staging_batch_id != selected.staging_batch_id
        or qc2_event.related_comparison_id != comparison.id
        or metadata.get("contract") != WORKFLOW_QC_REVIEW_CONTRACT
        or metadata.get("review_stage") != "qc2"
        or metadata.get("decision") != "approved"
        or metadata.get("selected_pass_id") != str(selected.id)
        or metadata.get("selected_staging_batch_id")
        != str(selected.staging_batch_id)
        or metadata.get("qc1_review_id") != str(qc1.id)
        or metadata.get("selection_reason")
        != "approved_qc1_review_inherited"
        or metadata.get("canonical_writer_invoked") is not False
        or metadata.get("client_selected_pass_authority") is not False
        or metadata.get("mixed_side_value_merge") is not False
        or metadata.get("direct_value_edit") is not False
    ):
        raise WorkflowQCReviewConflict(
            "QC2 approval audit provenance does not reconcile for publication."
        )

    open_discrepancies = session.execute(
        select(WorkflowDiscrepancy.id).where(
            WorkflowDiscrepancy.workflow_item_id == item.id,
            WorkflowDiscrepancy.comparison_id == comparison.id,
            WorkflowDiscrepancy.resolution_status == "open",
        )
    ).all()
    if open_discrepancies:
        raise WorkflowQCReviewConflict(
            "Publication requires zero open discrepancies."
        )

    ready_state = (
        "ready_for_publication",
        "publication_handoff",
        "ready",
    )
    published_state = (
        "published",
        "publication_handoff",
        "complete",
    )
    current_state = (
        item.lifecycle_state,
        item.current_stage,
        item.stage_condition,
    )
    if require_ready_state:
        if current_state != ready_state or item.canonical_race_id is not None:
            raise WorkflowQCReviewConflict(
                "Workflow item is not exactly ready for publication handoff."
            )
    elif current_state not in {ready_state, published_state}:
        raise WorkflowQCReviewConflict(
            "Workflow item is neither publication-ready nor published."
        )
    elif current_state == published_state and item.canonical_race_id is None:
        raise WorkflowQCReviewConflict(
            "Published Workflow item is missing canonical linkage."
        )

    return {
        "item": item,
        "dl1": dl1,
        "dl2": dl2,
        "comparison": comparison,
        "selected_pass": selected,
        "selection_reason": selection_reason,
        "qc1_review": qc1,
        "qc2_review": qc2,
        "qc2_event": qc2_event,
        "dl1_principal": dl1_principal,
        "dl2_principal": dl2_principal,
        "qc1_principal": qc1_principal,
        "qc2_principal": qc2_principal,
        "open_discrepancy_count": 0,
    }
