"""Governed contributor actions for the noncanonical workflow plane.

This service never commits. Callers own commit/rollback and must enforce
contributor authority before entering a mutation function.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from webapp.parser.utils.models import (
    WorkflowEvent,
    WorkflowItem,
    WorkflowPass,
)
from webapp.parser.utils.url_registry import lookup_exact_registry_entry


WORKFLOW_CLAIM_CONTRACT = "w3_pass_claim_v1"
APPROVED_SOURCE_CONTRACT = "w3_approved_source_projection_v1"


class WorkflowActionError(RuntimeError):
    status_code = 400
    code = "workflow_action_error"


class WorkflowActionNotFound(WorkflowActionError):
    status_code = 404
    code = "workflow_item_not_found"


class WorkflowActionConflict(WorkflowActionError):
    status_code = 409
    code = "workflow_claim_conflict"


class WorkflowSourceNotApproved(WorkflowActionError):
    status_code = 409
    code = "workflow_source_not_approved"


class WorkflowSourceAccessDenied(WorkflowActionError):
    status_code = 403
    code = "workflow_source_access_denied"


def _normalize_item_id(item_id: UUID | str) -> UUID:
    try:
        return item_id if isinstance(item_id, UUID) else UUID(str(item_id))
    except (TypeError, ValueError) as exc:
        raise WorkflowActionError("workflow item id must be a UUID.") from exc


def read_approved_workflow_source(
    session: Session,
    item_id: UUID | str,
    *,
    principal: str,
    registry_path: Path,
) -> dict[str, Any]:
    normalized = _normalize_item_id(item_id)
    item = session.get(WorkflowItem, normalized)
    if item is None:
        raise WorkflowActionNotFound("Workflow item was not found.")

    actor = str(principal or "").strip()
    if not actor:
        raise WorkflowSourceAccessDenied(
            "Authenticated internal principal is required for source disclosure."
        )

    claimable = (
        item.lifecycle_state,
        item.current_stage,
        item.stage_condition,
    ) == ("queued", "source_intake", "pending")

    assigned = session.execute(
        select(WorkflowPass.id).where(
            WorkflowPass.workflow_item_id == normalized,
            WorkflowPass.is_current.is_(True),
            WorkflowPass.assigned_principal == actor,
        )
    ).first() is not None

    if not claimable and not assigned:
        raise WorkflowSourceAccessDenied(
            "Contributor source disclosure requires a claimable task "
            "or a current pass assigned to the requesting principal."
        )

    entry = lookup_exact_registry_entry(
        str(item.source_url or ""),
        path=registry_path,
    )
    if entry is None:
        raise WorkflowSourceNotApproved(
            "Workflow source is not an exact maintained-registry entry."
        )
    if entry.registry_category != "curated":
        raise WorkflowSourceNotApproved(
            "Workflow source is not in the curated registry category."
        )

    return {
        "success": True,
        "contract": APPROVED_SOURCE_CONTRACT,
        "task_id": str(item.id),
        "source_race_id": item.source_race_id,
        "source_url": entry.url,
        "registry_category": entry.registry_category,
        "registry_format": entry.registry_format,
        "registry_scope": entry.registry_scope,
        "source_url_editable": False,
        "arbitrary_url_submission": False,
        "arbitrary_url_execution": False,
        "exact_registry_entry_required": True,
    }


def assert_independent_second_pass(
    first_principal: str | None,
    second_principal: str | None,
) -> None:
    first = str(first_principal or "").strip()
    second = str(second_principal or "").strip()
    if not first or not second:
        raise WorkflowActionConflict(
            "Both independent-pass principals must be present."
        )
    if first == second:
        raise WorkflowActionConflict(
            "DL2 must be claimed by a different principal from DL1."
        )


def _item_state(item: WorkflowItem) -> dict[str, Any]:
    return {
        "lifecycle_state": item.lifecycle_state,
        "current_stage": item.current_stage,
        "stage_condition": item.stage_condition,
        "row_version": item.row_version,
    }


def claim_first_workflow_pass(
    session: Session,
    item_id: UUID | str,
    *,
    principal: str,
    expected_row_version: int,
    now: datetime | None = None,
) -> dict[str, Any]:
    actor = str(principal or "").strip()
    if not actor:
        raise WorkflowActionError(
            "Authenticated internal principal is required."
        )

    normalized = _normalize_item_id(item_id)
    try:
        expected_version = int(expected_row_version)
    except (TypeError, ValueError) as exc:
        raise WorkflowActionError(
            "expected_row_version must be an integer."
        ) from exc

    timestamp = now or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    timestamp = timestamp.astimezone(timezone.utc)

    item = session.execute(
        select(WorkflowItem)
        .where(WorkflowItem.id == normalized)
        .with_for_update()
    ).scalar_one_or_none()
    if item is None:
        raise WorkflowActionNotFound("Workflow item was not found.")

    if int(item.row_version) != expected_version:
        raise WorkflowActionConflict(
            "Workflow row_version changed before claim."
        )

    actual_state = (
        item.lifecycle_state,
        item.current_stage,
        item.stage_condition,
    )
    if actual_state != ("queued", "source_intake", "pending"):
        raise WorkflowActionConflict(
            "Workflow item is not available for initial DL1 claim."
        )

    existing = session.execute(
        select(WorkflowPass.id).where(
            WorkflowPass.workflow_item_id == normalized,
            WorkflowPass.pass_number == 1,
            WorkflowPass.is_current.is_(True),
        )
    ).first()
    if existing is not None:
        raise WorkflowActionConflict(
            "Current DL1 pass already exists for workflow item."
        )

    prior_state = _item_state(item)

    workflow_pass = WorkflowPass(
        workflow_item_id=item.id,
        pass_number=1,
        pass_label="DL1",
        revision_number=1,
        is_current=True,
        status="in_progress",
        assigned_principal=actor,
        source_evidence_ref=None,
        staging_batch_id=None,
        candidate_check_status=None,
        candidate_check_result=None,
        semantic_validation_status=None,
        semantic_validation_result=None,
        started_at=timestamp,
        submitted_at=None,
        superseded_at=None,
        notes=None,
        created_at=timestamp,
        updated_at=timestamp,
    )
    session.add(workflow_pass)
    session.flush()

    item.lifecycle_state = "active"
    item.current_stage = "independent_acquisition"
    item.stage_condition = "in_progress"
    item.row_version = expected_version + 1
    item.updated_at = timestamp

    new_state = _item_state(item)

    event = WorkflowEvent(
        workflow_item_id=item.id,
        actor_type="principal",
        actor_principal=actor,
        actor_service=None,
        event_type="pass_claimed",
        stage="independent_acquisition",
        prior_state=prior_state,
        new_state=new_state,
        related_pass_id=workflow_pass.id,
        related_comparison_id=None,
        related_review_id=None,
        related_staging_batch_id=None,
        related_canonical_race_id=item.canonical_race_id,
        reason_code=None,
        summary="DL1 claimed for independent acquisition.",
        event_metadata={
            "contract": WORKFLOW_CLAIM_CONTRACT,
            "pass_number": 1,
            "pass_label": "DL1",
        },
        occurred_at=timestamp,
    )
    session.add(event)
    session.flush()

    return {
        "success": True,
        "contract": WORKFLOW_CLAIM_CONTRACT,
        "task_id": str(item.id),
        "pass_id": str(workflow_pass.id),
        "event_id": str(event.id),
        "pass_number": 1,
        "pass_label": "DL1",
        "status": "in_progress",
        "lifecycle_state": item.lifecycle_state,
        "current_stage": item.current_stage,
        "stage_condition": item.stage_condition,
        "row_version": item.row_version,
        "committed": False,
    }
