from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from webapp.parser.contracts.workflow_lifecycle import (
    DISCREPANCY_RESOLUTION_SELECTION_CODES,
    assert_discrepancy_resolver_separation,
    assert_forward_stage_transition,
)
from webapp.parser.utils.models import (
    WorkflowComparison,
    WorkflowDiscrepancy,
    WorkflowEvent,
    WorkflowItem,
    WorkflowPass,
)


WORKFLOW_DISCREPANCY_RESOLUTION_CONTRACT = "workflow_discrepancy_resolution_v1"
WORKFLOW_DISCREPANCY_RESOLUTION_SERVICE = "workflow_discrepancy_resolution"
_MAX_RESOLUTION_NOTES = 4000


class WorkflowDiscrepancyResolutionError(RuntimeError):
    status_code = 400
    code = "workflow_discrepancy_resolution_error"


class WorkflowDiscrepancyResolutionNotFound(WorkflowDiscrepancyResolutionError):
    status_code = 404
    code = "workflow_discrepancy_resolution_not_found"


class WorkflowDiscrepancyResolutionConflict(WorkflowDiscrepancyResolutionError):
    status_code = 409
    code = "workflow_discrepancy_resolution_conflict"


def _uuid(value: UUID | str, *, name: str) -> UUID:
    try:
        return value if isinstance(value, UUID) else UUID(str(value))
    except (TypeError, ValueError) as exc:
        raise WorkflowDiscrepancyResolutionError(
            f"{name} must be a UUID."
        ) from exc


def _expected_version(value: object) -> int:
    if isinstance(value, bool):
        raise WorkflowDiscrepancyResolutionError(
            "expected_row_version must be an integer."
        )
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise WorkflowDiscrepancyResolutionError(
            "expected_row_version must be an integer."
        ) from exc
    if parsed < 0:
        raise WorkflowDiscrepancyResolutionError(
            "expected_row_version must be >= 0."
        )
    return parsed


def _actor(principal: object) -> str:
    actor = str(principal or "").strip()
    if not actor:
        raise WorkflowDiscrepancyResolutionError(
            "Authenticated reviewer principal is required."
        )
    return actor


def _resolution_code(value: object) -> str:
    code = str(value or "").strip()
    if code not in DISCREPANCY_RESOLUTION_SELECTION_CODES:
        raise WorkflowDiscrepancyResolutionError(
            "resolution_code must be select_dl1 or select_dl2."
        )
    return code


def _resolution_notes(value: object) -> str:
    notes = str(value or "").strip()
    if not notes:
        raise WorkflowDiscrepancyResolutionError(
            "resolution_notes must be non-empty."
        )
    if len(notes) > _MAX_RESOLUTION_NOTES:
        raise WorkflowDiscrepancyResolutionError(
            f"resolution_notes must be <= {_MAX_RESOLUTION_NOTES} characters."
        )
    return notes


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
        raise WorkflowDiscrepancyResolutionConflict(
            "Resolution requires exactly one current DL1 and DL2."
        )

    by_pair = {
        (int(row.pass_number), str(row.pass_label)): row
        for row in rows
    }
    if set(by_pair) != {(1, "DL1"), (2, "DL2")}:
        raise WorkflowDiscrepancyResolutionConflict(
            "Current acquisition pass identities are invalid."
        )

    dl1 = by_pair[(1, "DL1")]
    dl2 = by_pair[(2, "DL2")]
    dl1_principal = str(dl1.assigned_principal or "").strip()
    dl2_principal = str(dl2.assigned_principal or "").strip()

    if (
        dl1.status != "submitted"
        or dl2.status != "submitted"
        or dl1.submitted_at is None
        or dl2.submitted_at is None
        or not dl1_principal
        or not dl2_principal
        or dl1_principal == dl2_principal
    ):
        raise WorkflowDiscrepancyResolutionConflict(
            "Resolution requires immutable submitted independent DL1/DL2."
        )

    return dl1, dl2


def _selected_pass(
    code: str,
    dl1: WorkflowPass,
    dl2: WorkflowPass,
) -> WorkflowPass:
    return dl1 if code == "select_dl1" else dl2


def _load_discrepancies(
    session: Session,
    item: WorkflowItem,
    comparison: WorkflowComparison,
) -> list[WorkflowDiscrepancy]:
    rows = session.execute(
        select(WorkflowDiscrepancy)
        .where(
            WorkflowDiscrepancy.workflow_item_id == item.id,
            WorkflowDiscrepancy.comparison_id == comparison.id,
        )
        .order_by(WorkflowDiscrepancy.id)
        .with_for_update()
    ).scalars().all()

    if (
        not isinstance(comparison.difference_count, int)
        or comparison.difference_count <= 0
        or len(rows) != comparison.difference_count
    ):
        raise WorkflowDiscrepancyResolutionConflict(
            "Discrepancy set does not reconcile with comparison difference_count."
        )
    return rows


def _validate_comparison(
    item: WorkflowItem,
    comparison: WorkflowComparison,
    dl1: WorkflowPass,
    dl2: WorkflowPass,
) -> None:
    if (
        comparison.workflow_item_id != item.id
        or comparison.left_pass_id != dl1.id
        or comparison.right_pass_id != dl2.id
        or comparison.status != "complete"
        or comparison.strict_equality_passed is not False
        or comparison.checked_at is None
        or not str(comparison.checked_by_service_version or "").strip()
    ):
        raise WorkflowDiscrepancyResolutionConflict(
            "Resolution requires the exact completed mismatching DL1/DL2 comparison."
        )


def _assert_unique_completed_current_pair_comparison(
    session: Session,
    item: WorkflowItem,
    comparison: WorkflowComparison,
    dl1: WorkflowPass,
    dl2: WorkflowPass,
) -> None:
    ids = session.execute(
        select(WorkflowComparison.id)
        .where(
            WorkflowComparison.workflow_item_id == item.id,
            WorkflowComparison.left_pass_id == dl1.id,
            WorkflowComparison.right_pass_id == dl2.id,
            WorkflowComparison.status == "complete",
        )
        .with_for_update()
    ).scalars().all()
    if len(ids) != 1 or ids[0] != comparison.id:
        raise WorkflowDiscrepancyResolutionConflict(
            "Resolution requires exactly one completed comparison for the "
            "current submitted DL1/DL2 pair."
        )


def _exact_replay(
    session: Session,
    item: WorkflowItem,
    comparison: WorkflowComparison,
    discrepancies: list[WorkflowDiscrepancy],
    *,
    actor: str,
    code: str,
    notes: str,
    selected: WorkflowPass,
) -> dict[str, Any] | None:
    statuses = {row.resolution_status for row in discrepancies}
    if statuses == {"open"}:
        return None
    if statuses != {"resolved"}:
        raise WorkflowDiscrepancyResolutionConflict(
            "Partial or non-exact discrepancy resolution state is not replayable."
        )

    if (
        item.lifecycle_state != "active"
        or item.current_stage != "qc1_review"
        or item.stage_condition != "pending"
        or comparison.reviewed_by_principal != actor
        or comparison.reviewed_at is None
    ):
        raise WorkflowDiscrepancyResolutionConflict(
            "Completed resolution downstream state does not reconcile."
        )

    for row in discrepancies:
        if (
            row.resolution_code != code
            or row.resolution_notes != notes
            or row.resolved_by_principal != actor
            or row.resolved_at is None
        ):
            raise WorkflowDiscrepancyResolutionConflict(
                "Completed discrepancy resolution is not an exact replay."
            )

    events = session.execute(
        select(WorkflowEvent).where(
            WorkflowEvent.workflow_item_id == item.id,
            WorkflowEvent.related_comparison_id == comparison.id,
            WorkflowEvent.event_type == "discrepancy_resolution_completed",
        )
    ).scalars().all()
    if len(events) != 1:
        raise WorkflowDiscrepancyResolutionConflict(
            "Completed resolution audit event set is not exact."
        )

    event = events[0]
    metadata = event.event_metadata if isinstance(event.event_metadata, dict) else {}
    if (
        event.actor_type != "principal"
        or event.actor_principal != actor
        or event.actor_service is not None
        or event.stage != "discrepancy_resolution"
        or metadata.get("contract") != WORKFLOW_DISCREPANCY_RESOLUTION_CONTRACT
        or metadata.get("resolution_code") != code
        or metadata.get("selected_pass_id") != str(selected.id)
        or metadata.get("resolved_discrepancy_count") != len(discrepancies)
    ):
        raise WorkflowDiscrepancyResolutionConflict(
            "Completed resolution audit event does not reconcile."
        )

    return {
        "success": True,
        "contract": WORKFLOW_DISCREPANCY_RESOLUTION_CONTRACT,
        "task_id": str(item.id),
        "comparison_id": str(comparison.id),
        "resolution_code": code,
        "resolution_notes": notes,
        "selected_pass_id": str(selected.id),
        "selected_pass_number": int(selected.pass_number),
        "selected_pass_label": str(selected.pass_label),
        "resolved_discrepancy_count": len(discrepancies),
        "current_stage": item.current_stage,
        "stage_condition": item.stage_condition,
        "row_version": item.row_version,
        "event_id": str(event.id),
        "already_resolved": True,
        "committed": False,
    }


def resolve_workflow_comparison_discrepancies(
    session: Session,
    item_id: UUID | str,
    comparison_id: UUID | str,
    *,
    principal: str,
    expected_row_version: int,
    resolution_code: str,
    resolution_notes: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    # Resolve one complete discrepancy set by selecting frozen DL1 or DL2.
    # The selected pass is derived server-side from resolution_code. This
    # service never edits election values, merges sides, mutates pass/artifact
    # evidence, creates WorkflowReview rows, writes canonical data, commits,
    # or rolls back.

    normalized_item = _uuid(item_id, name="item_id")
    normalized_comparison = _uuid(comparison_id, name="comparison_id")
    actor = _actor(principal)
    expected = _expected_version(expected_row_version)
    code = _resolution_code(resolution_code)
    notes = _resolution_notes(resolution_notes)
    timestamp = _utc(now)

    item = session.execute(
        select(WorkflowItem)
        .where(WorkflowItem.id == normalized_item)
        .with_for_update()
    ).scalar_one_or_none()
    if item is None:
        raise WorkflowDiscrepancyResolutionNotFound(
            "Workflow item was not found."
        )
    if int(item.row_version) != expected:
        raise WorkflowDiscrepancyResolutionConflict(
            "Workflow row_version changed before discrepancy resolution."
        )

    comparison = session.execute(
        select(WorkflowComparison)
        .where(WorkflowComparison.id == normalized_comparison)
        .with_for_update()
    ).scalar_one_or_none()
    if comparison is None:
        raise WorkflowDiscrepancyResolutionNotFound(
            "Workflow comparison was not found."
        )

    dl1, dl2 = _load_current_submitted_pair(session, item)
    _validate_comparison(item, comparison, dl1, dl2)
    _assert_unique_completed_current_pair_comparison(
        session,
        item,
        comparison,
        dl1,
        dl2,
    )
    try:
        assert_discrepancy_resolver_separation(
            dl1_principal=str(dl1.assigned_principal),
            dl2_principal=str(dl2.assigned_principal),
            resolver_principal=actor,
        )
    except ValueError as exc:
        raise WorkflowDiscrepancyResolutionConflict(str(exc)) from exc

    selected = _selected_pass(code, dl1, dl2)
    discrepancies = _load_discrepancies(session, item, comparison)

    replay = _exact_replay(
        session,
        item,
        comparison,
        discrepancies,
        actor=actor,
        code=code,
        notes=notes,
        selected=selected,
    )
    if replay is not None:
        return replay

    if (
        item.lifecycle_state,
        item.current_stage,
        item.stage_condition,
    ) != ("active", "discrepancy_resolution", "pending"):
        raise WorkflowDiscrepancyResolutionConflict(
            "Workflow item is not pending discrepancy resolution."
        )

    if (
        comparison.reviewed_by_principal is not None
        or comparison.reviewed_at is not None
    ):
        raise WorkflowDiscrepancyResolutionConflict(
            "Unresolved comparison already contains reviewer resolution fields."
        )

    for row in discrepancies:
        if (
            row.resolution_status != "open"
            or row.resolution_code is not None
            or row.resolution_notes is not None
            or row.resolved_by_principal is not None
            or row.resolved_at is not None
        ):
            raise WorkflowDiscrepancyResolutionConflict(
                "All discrepancies must be pristine open rows before resolution."
            )

    assert_forward_stage_transition(
        "discrepancy_resolution",
        "qc1_review",
    )

    prior_state = {
        "lifecycle_state": item.lifecycle_state,
        "current_stage": item.current_stage,
        "stage_condition": item.stage_condition,
        "row_version": item.row_version,
    }

    for row in discrepancies:
        row.resolution_status = "resolved"
        row.resolution_code = code
        row.resolution_notes = notes
        row.resolved_by_principal = actor
        row.resolved_at = timestamp

    comparison.reviewed_by_principal = actor
    comparison.reviewed_at = timestamp

    item.current_stage = "qc1_review"
    item.stage_condition = "pending"
    item.row_version = expected + 1
    item.updated_at = timestamp

    event = WorkflowEvent(
        workflow_item_id=item.id,
        actor_type="principal",
        actor_principal=actor,
        actor_service=None,
        event_type="discrepancy_resolution_completed",
        stage="discrepancy_resolution",
        prior_state=prior_state,
        new_state={
            "lifecycle_state": item.lifecycle_state,
            "current_stage": item.current_stage,
            "stage_condition": item.stage_condition,
            "row_version": item.row_version,
        },
        related_pass_id=selected.id,
        related_comparison_id=comparison.id,
        related_review_id=None,
        related_staging_batch_id=selected.staging_batch_id,
        related_canonical_race_id=item.canonical_race_id,
        reason_code=code,
        summary=(
            f"Reviewer resolved complete discrepancy set by selecting "
            f"{selected.pass_label}."
        ),
        event_metadata={
            "contract": WORKFLOW_DISCREPANCY_RESOLUTION_CONTRACT,
            "comparison_id": str(comparison.id),
            "resolution_code": code,
            "selected_pass_id": str(selected.id),
            "selected_pass_number": int(selected.pass_number),
            "selected_pass_label": str(selected.pass_label),
            "resolved_discrepancy_count": len(discrepancies),
            "mixed_side_value_merge": False,
            "direct_value_edit": False,
            "selected_pass_is_noncanonical": True,
        },
        occurred_at=timestamp,
    )
    session.add(event)
    session.flush()

    return {
        "success": True,
        "contract": WORKFLOW_DISCREPANCY_RESOLUTION_CONTRACT,
        "task_id": str(item.id),
        "comparison_id": str(comparison.id),
        "resolution_code": code,
        "resolution_notes": notes,
        "selected_pass_id": str(selected.id),
        "selected_pass_number": int(selected.pass_number),
        "selected_pass_label": str(selected.pass_label),
        "resolved_discrepancy_count": len(discrepancies),
        "current_stage": item.current_stage,
        "stage_condition": item.stage_condition,
        "row_version": item.row_version,
        "event_id": str(event.id),
        "already_resolved": False,
        "committed": False,
    }
