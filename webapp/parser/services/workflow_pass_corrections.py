"""Immutable pre-submit correction revisions for governed DL1 passes.

A correction never overwrites revision N evidence. The current DL1 revision
must already have a valid completed workflow_staging_binding_v1 artifact.
This service then:
- locks the WorkflowItem and current DL1 pass,
- enforces expected row_version,
- preserves all revision-N evidence/artifact/validation fields,
- marks revision N superseded/non-current,
- creates revision N+1 for the same DL1 principal with all evidence and
  validation fields reset,
- increments WorkflowItem.row_version exactly once,
- appends one pass_correction_revision_created audit event.

This service never commits or rolls back. Callers own the transaction and must
enforce authorization before entry.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from webapp.parser.contracts.workflow_lifecycle import next_pass_revision
from webapp.parser.services.workflow_staging_binding import (
    WorkflowStagingBindingError,
    validate_completed_workflow_staging_binding,
)
from webapp.parser.utils.models import (
    WorkflowEvent,
    WorkflowItem,
    WorkflowPass,
)


WORKFLOW_DL1_CORRECTION_CONTRACT = "workflow_dl1_correction_revision_v1"
WORKFLOW_DL1_CORRECTION_REASON_CODES = frozenset({
    "pre_qc_validation_failed",
    "source_evidence_correction",
    "normalized_artifact_correction",
    "operator_correction",
})


class WorkflowDL1CorrectionError(RuntimeError):
    status_code = 400
    code = "workflow_dl1_correction_error"


class WorkflowDL1CorrectionNotFound(WorkflowDL1CorrectionError):
    status_code = 404
    code = "workflow_dl1_correction_not_found"


class WorkflowDL1CorrectionConflict(WorkflowDL1CorrectionError):
    status_code = 409
    code = "workflow_dl1_correction_conflict"


def _uuid(value: UUID | str, *, name: str) -> UUID:
    try:
        return value if isinstance(value, UUID) else UUID(str(value))
    except (TypeError, ValueError) as exc:
        raise WorkflowDL1CorrectionError(
            f"{name} must be a UUID."
        ) from exc


def _actor(principal: str) -> str:
    actor = str(principal or "").strip()
    if not actor:
        raise WorkflowDL1CorrectionError(
            "Authenticated internal principal is required."
        )
    return actor


def _expected_version(value: object) -> int:
    if isinstance(value, bool):
        raise WorkflowDL1CorrectionError(
            "expected_row_version must be an integer."
        )
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise WorkflowDL1CorrectionError(
            "expected_row_version must be an integer."
        ) from exc
    if parsed < 0:
        raise WorkflowDL1CorrectionError(
            "expected_row_version must be >= 0."
        )
    return parsed


def _utc(now: datetime | None) -> datetime:
    value = now or datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def create_dl1_correction_revision(
    session: Session,
    item_id: UUID | str,
    pass_id: UUID | str,
    *,
    principal: str,
    expected_row_version: int,
    reason_code: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Supersede current DL1 revision N and create clean revision N+1."""

    actor = _actor(principal)
    normalized_item = _uuid(item_id, name="item_id")
    normalized_pass = _uuid(pass_id, name="pass_id")
    expected_version = _expected_version(expected_row_version)
    reason = str(reason_code or "").strip()
    if reason not in WORKFLOW_DL1_CORRECTION_REASON_CODES:
        raise WorkflowDL1CorrectionError(
            "reason_code is not an accepted DL1 correction reason."
        )
    timestamp = _utc(now)

    item = session.execute(
        select(WorkflowItem)
        .where(WorkflowItem.id == normalized_item)
        .with_for_update()
    ).scalar_one_or_none()
    if item is None:
        raise WorkflowDL1CorrectionNotFound(
            "Workflow item was not found."
        )

    if int(item.row_version) != expected_version:
        raise WorkflowDL1CorrectionConflict(
            "Workflow row_version changed before correction."
        )

    if (
        item.lifecycle_state,
        item.current_stage,
        item.stage_condition,
    ) != ("active", "independent_acquisition", "in_progress"):
        raise WorkflowDL1CorrectionConflict(
            "Workflow item is not in active independent acquisition."
        )

    current = session.execute(
        select(WorkflowPass)
        .where(WorkflowPass.id == normalized_pass)
        .with_for_update()
    ).scalar_one_or_none()
    if current is None:
        raise WorkflowDL1CorrectionNotFound(
            "Workflow pass was not found."
        )

    if (
        current.workflow_item_id != item.id
        or current.pass_number != 1
        or current.pass_label != "DL1"
        or current.is_current is not True
        or current.status != "in_progress"
        or str(current.assigned_principal or "").strip() != actor
        or current.submitted_at is not None
    ):
        raise WorkflowDL1CorrectionConflict(
            "Correction requires the current in-progress pre-submit DL1 "
            "revision assigned to the requesting principal."
        )

    other_current = session.execute(
        select(WorkflowPass.id).where(
            WorkflowPass.workflow_item_id == item.id,
            WorkflowPass.pass_number == 1,
            WorkflowPass.is_current.is_(True),
            WorkflowPass.id != current.id,
        )
    ).first()
    if other_current is not None:
        raise WorkflowDL1CorrectionConflict(
            "Multiple current DL1 revisions exist; refusing correction."
        )

    if current.staging_batch_id is None:
        raise WorkflowDL1CorrectionConflict(
            "Correction requires a completed immutable staging binding."
        )

    try:
        prior_binding = validate_completed_workflow_staging_binding(
            session,
            item.id,
            current.id,
            current.staging_batch_id,
            principal=actor,
        )
    except WorkflowStagingBindingError as exc:
        raise WorkflowDL1CorrectionConflict(
            "Current DL1 revision does not have a valid completed staging "
            f"binding: {exc}"
        ) from exc

    old_evidence_snapshot = {
        "pass_id": str(current.id),
        "revision_number": current.revision_number,
        "status": current.status,
        "is_current": current.is_current,
        "assigned_principal": current.assigned_principal,
        "source_evidence_ref": current.source_evidence_ref,
        "staging_batch_id": str(current.staging_batch_id),
        "candidate_check_status": current.candidate_check_status,
        "candidate_check_result": current.candidate_check_result,
        "semantic_validation_status": current.semantic_validation_status,
        "semantic_validation_result": current.semantic_validation_result,
        "submitted_at": (
            current.submitted_at.isoformat()
            if current.submitted_at is not None
            else None
        ),
        "artifact_ref": prior_binding["artifact_ref"],
        "artifact_sha256": prior_binding["artifact_sha256"],
    }

    new_revision_number = next_pass_revision(current.revision_number)

    current.is_current = False
    current.status = "superseded"
    current.superseded_at = timestamp
    current.updated_at = timestamp

    replacement = WorkflowPass(
        workflow_item_id=item.id,
        pass_number=1,
        pass_label="DL1",
        revision_number=new_revision_number,
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
    session.add(replacement)
    session.flush()

    item.row_version = expected_version + 1
    item.updated_at = timestamp

    event = WorkflowEvent(
        workflow_item_id=item.id,
        actor_type="principal",
        actor_principal=actor,
        actor_service=None,
        event_type="pass_correction_revision_created",
        stage="independent_acquisition",
        prior_state={
            "lifecycle_state": item.lifecycle_state,
            "current_stage": item.current_stage,
            "stage_condition": item.stage_condition,
            "row_version": expected_version,
            "current_dl1_pass_id": str(current.id),
            "current_dl1_revision": current.revision_number,
        },
        new_state={
            "lifecycle_state": item.lifecycle_state,
            "current_stage": item.current_stage,
            "stage_condition": item.stage_condition,
            "row_version": item.row_version,
            "current_dl1_pass_id": str(replacement.id),
            "current_dl1_revision": replacement.revision_number,
        },
        related_pass_id=replacement.id,
        related_comparison_id=None,
        related_review_id=None,
        related_staging_batch_id=None,
        related_canonical_race_id=item.canonical_race_id,
        reason_code=reason,
        summary=(
            "Immutable DL1 correction revision created; prior revision "
            "preserved as superseded."
        ),
        event_metadata={
            "contract": WORKFLOW_DL1_CORRECTION_CONTRACT,
            "superseded_pass_id": str(current.id),
            "superseded_revision_number": current.revision_number,
            "superseded_staging_batch_id": str(current.staging_batch_id),
            "superseded_artifact_ref": prior_binding["artifact_ref"],
            "superseded_artifact_sha256": prior_binding["artifact_sha256"],
            "replacement_pass_id": str(replacement.id),
            "replacement_revision_number": replacement.revision_number,
            "replacement_evidence_reset": True,
        },
        occurred_at=timestamp,
    )
    session.add(event)
    session.flush()

    # Fail closed if controlled supersession accidentally altered the immutable
    # evidence identity of revision N.
    if (
        current.source_evidence_ref
            != old_evidence_snapshot["source_evidence_ref"]
        or str(current.staging_batch_id)
            != old_evidence_snapshot["staging_batch_id"]
        or current.candidate_check_status
            != old_evidence_snapshot["candidate_check_status"]
        or current.candidate_check_result
            != old_evidence_snapshot["candidate_check_result"]
        or current.semantic_validation_status
            != old_evidence_snapshot["semantic_validation_status"]
        or current.semantic_validation_result
            != old_evidence_snapshot["semantic_validation_result"]
        or current.submitted_at is not None
    ):
        raise WorkflowDL1CorrectionConflict(
            "Supersession changed immutable revision-N evidence."
        )

    return {
        "success": True,
        "contract": WORKFLOW_DL1_CORRECTION_CONTRACT,
        "task_id": str(item.id),
        "superseded_pass_id": str(current.id),
        "superseded_revision_number": current.revision_number,
        "replacement_pass_id": str(replacement.id),
        "replacement_revision_number": replacement.revision_number,
        "reason_code": reason,
        "row_version": item.row_version,
        "event_id": str(event.id),
        "replacement_evidence_reset": True,
        "committed": False,
    }
