"""Immutable pre-submit correction revisions for governed Workflow passes.

A correction never overwrites revision N evidence. The current governed pass
must already have a valid completed workflow_staging_binding_v1 artifact.
This service then:
- locks the WorkflowItem and current server-loaded pass,
- enforces expected row_version,
- preserves all revision-N evidence/artifact/validation fields,
- marks revision N superseded/non-current,
- creates revision N+1 for the same pass identity and principal with all
  evidence and validation fields reset,
- increments WorkflowItem.row_version exactly once,
- appends one pass_correction_revision_created audit event.

Supported governed pass identities are 1|DL1 and 2|DL2. Client-supplied pass
identity is never authoritative here; the engine derives identity from the
locked WorkflowPass row. Compatibility wrappers may additionally require one
specific governed pair.

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
WORKFLOW_DL2_CORRECTION_CONTRACT = "workflow_dl2_correction_revision_v1"

WORKFLOW_PASS_CORRECTION_REASON_CODES = frozenset({
    "pre_qc_validation_failed",
    "source_evidence_correction",
    "normalized_artifact_correction",
    "operator_correction",
})
WORKFLOW_DL1_CORRECTION_REASON_CODES = WORKFLOW_PASS_CORRECTION_REASON_CODES
WORKFLOW_DL2_CORRECTION_REASON_CODES = WORKFLOW_PASS_CORRECTION_REASON_CODES

_SUPPORTED_GOVERNED_PASS_PAIRS = frozenset({
    (1, "DL1"),
    (2, "DL2"),
})
_CORRECTION_CONTRACT_BY_PAIR = {
    (1, "DL1"): WORKFLOW_DL1_CORRECTION_CONTRACT,
    (2, "DL2"): WORKFLOW_DL2_CORRECTION_CONTRACT,
}


class WorkflowPassCorrectionError(RuntimeError):
    status_code = 400
    code = "workflow_pass_correction_error"


class WorkflowPassCorrectionNotFound(WorkflowPassCorrectionError):
    status_code = 404
    code = "workflow_pass_correction_not_found"


class WorkflowPassCorrectionConflict(WorkflowPassCorrectionError):
    status_code = 409
    code = "workflow_pass_correction_conflict"


class WorkflowDL1CorrectionError(RuntimeError):
    status_code = 400
    code = "workflow_dl1_correction_error"


class WorkflowDL1CorrectionNotFound(WorkflowDL1CorrectionError):
    status_code = 404
    code = "workflow_dl1_correction_not_found"


class WorkflowDL1CorrectionConflict(WorkflowDL1CorrectionError):
    status_code = 409
    code = "workflow_dl1_correction_conflict"


class WorkflowDL2CorrectionError(RuntimeError):
    status_code = 400
    code = "workflow_dl2_correction_error"


class WorkflowDL2CorrectionNotFound(WorkflowDL2CorrectionError):
    status_code = 404
    code = "workflow_dl2_correction_not_found"


class WorkflowDL2CorrectionConflict(WorkflowDL2CorrectionError):
    status_code = 409
    code = "workflow_dl2_correction_conflict"


def _uuid(value: UUID | str, *, name: str) -> UUID:
    try:
        return value if isinstance(value, UUID) else UUID(str(value))
    except (TypeError, ValueError) as exc:
        raise WorkflowPassCorrectionError(
            f"{name} must be a UUID."
        ) from exc


def _actor(principal: str) -> str:
    actor = str(principal or "").strip()
    if not actor:
        raise WorkflowPassCorrectionError(
            "Authenticated internal principal is required."
        )
    return actor


def _expected_version(value: object) -> int:
    if isinstance(value, bool):
        raise WorkflowPassCorrectionError(
            "expected_row_version must be an integer."
        )
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise WorkflowPassCorrectionError(
            "expected_row_version must be an integer."
        ) from exc
    if parsed < 0:
        raise WorkflowPassCorrectionError(
            "expected_row_version must be >= 0."
        )
    return parsed


def _utc(now: datetime | None) -> datetime:
    value = now or datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _normalize_required_pair(
    value: tuple[int, str] | None,
) -> tuple[int, str] | None:
    if value is None:
        return None
    pair = (int(value[0]), str(value[1]))
    if pair not in _SUPPORTED_GOVERNED_PASS_PAIRS:
        raise WorkflowPassCorrectionError(
            "required governed pass identity is not supported."
        )
    return pair


def create_workflow_pass_correction_revision(
    session: Session,
    item_id: UUID | str,
    pass_id: UUID | str,
    *,
    principal: str,
    expected_row_version: int,
    reason_code: str,
    now: datetime | None = None,
    _required_pair: tuple[int, str] | None = None,
) -> dict[str, Any]:
    """Supersede server-loaded revision N and create clean revision N+1."""

    actor = _actor(principal)
    normalized_item = _uuid(item_id, name="item_id")
    normalized_pass = _uuid(pass_id, name="pass_id")
    expected_version = _expected_version(expected_row_version)
    required_pair = _normalize_required_pair(_required_pair)

    reason = str(reason_code or "").strip()
    if reason not in WORKFLOW_PASS_CORRECTION_REASON_CODES:
        label = required_pair[1] if required_pair is not None else "Workflow pass"
        raise WorkflowPassCorrectionError(
            f"reason_code is not an accepted {label} correction reason."
        )
    timestamp = _utc(now)

    item = session.execute(
        select(WorkflowItem)
        .where(WorkflowItem.id == normalized_item)
        .with_for_update()
    ).scalar_one_or_none()
    if item is None:
        raise WorkflowPassCorrectionNotFound(
            "Workflow item was not found."
        )

    if int(item.row_version) != expected_version:
        raise WorkflowPassCorrectionConflict(
            "Workflow row_version changed before correction."
        )

    if (
        item.lifecycle_state,
        item.current_stage,
        item.stage_condition,
    ) != ("active", "independent_acquisition", "in_progress"):
        raise WorkflowPassCorrectionConflict(
            "Workflow item is not in active independent acquisition."
        )

    current = session.execute(
        select(WorkflowPass)
        .where(WorkflowPass.id == normalized_pass)
        .with_for_update()
    ).scalar_one_or_none()
    if current is None:
        raise WorkflowPassCorrectionNotFound(
            "Workflow pass was not found."
        )

    pair = (int(current.pass_number), str(current.pass_label))
    if pair not in _SUPPORTED_GOVERNED_PASS_PAIRS:
        raise WorkflowPassCorrectionConflict(
            "Correction requires a supported governed acquisition pass."
        )
    if required_pair is not None and pair != required_pair:
        raise WorkflowPassCorrectionConflict(
            f"Correction requires the current {required_pair[1]} pass."
        )

    pass_number, pass_label = pair
    contract = _CORRECTION_CONTRACT_BY_PAIR[pair]

    if (
        current.workflow_item_id != item.id
        or current.is_current is not True
        or current.status != "in_progress"
        or str(current.assigned_principal or "").strip() != actor
        or current.submitted_at is not None
    ):
        raise WorkflowPassCorrectionConflict(
            f"Correction requires the current in-progress pre-submit "
            f"{pass_label} revision assigned to the requesting principal."
        )

    other_current = session.execute(
        select(WorkflowPass.id).where(
            WorkflowPass.workflow_item_id == item.id,
            WorkflowPass.pass_number == pass_number,
            WorkflowPass.is_current.is_(True),
            WorkflowPass.id != current.id,
        )
    ).first()
    if other_current is not None:
        raise WorkflowPassCorrectionConflict(
            f"Multiple current {pass_label} revisions exist; refusing correction."
        )

    if current.staging_batch_id is None:
        raise WorkflowPassCorrectionConflict(
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
        raise WorkflowPassCorrectionConflict(
            f"Current {pass_label} revision does not have a valid completed "
            f"staging binding: {exc}"
        ) from exc

    old_evidence_snapshot = {
        "pass_id": str(current.id),
        "pass_number": pass_number,
        "pass_label": pass_label,
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
        pass_number=pass_number,
        pass_label=pass_label,
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

    state_prefix = f"current_{pass_label.lower()}"
    prior_state = {
        "lifecycle_state": item.lifecycle_state,
        "current_stage": item.current_stage,
        "stage_condition": item.stage_condition,
        "row_version": expected_version,
        "current_pass_number": pass_number,
        "current_pass_label": pass_label,
        "current_pass_id": str(current.id),
        "current_pass_revision": current.revision_number,
        f"{state_prefix}_pass_id": str(current.id),
        f"{state_prefix}_revision": current.revision_number,
    }
    new_state = {
        "lifecycle_state": item.lifecycle_state,
        "current_stage": item.current_stage,
        "stage_condition": item.stage_condition,
        "row_version": item.row_version,
        "current_pass_number": pass_number,
        "current_pass_label": pass_label,
        "current_pass_id": str(replacement.id),
        "current_pass_revision": replacement.revision_number,
        f"{state_prefix}_pass_id": str(replacement.id),
        f"{state_prefix}_revision": replacement.revision_number,
    }

    event = WorkflowEvent(
        workflow_item_id=item.id,
        actor_type="principal",
        actor_principal=actor,
        actor_service=None,
        event_type="pass_correction_revision_created",
        stage="independent_acquisition",
        prior_state=prior_state,
        new_state=new_state,
        related_pass_id=replacement.id,
        related_comparison_id=None,
        related_review_id=None,
        related_staging_batch_id=None,
        related_canonical_race_id=item.canonical_race_id,
        reason_code=reason,
        summary=(
            f"Immutable {pass_label} correction revision created; prior "
            "revision preserved as superseded."
        ),
        event_metadata={
            "contract": contract,
            "pass_number": pass_number,
            "pass_label": pass_label,
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
        raise WorkflowPassCorrectionConflict(
            "Supersession changed immutable revision-N evidence."
        )

    return {
        "success": True,
        "contract": contract,
        "task_id": str(item.id),
        "pass_number": pass_number,
        "pass_label": pass_label,
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


def _translate_correction_error(
    exc: WorkflowPassCorrectionError,
    *,
    error_cls,
    not_found_cls,
    conflict_cls,
):
    if isinstance(exc, WorkflowPassCorrectionNotFound):
        return not_found_cls(str(exc))
    if isinstance(exc, WorkflowPassCorrectionConflict):
        return conflict_cls(str(exc))
    return error_cls(str(exc))


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
    """Compatibility wrapper for immutable DL1 pre-submit correction."""

    try:
        return create_workflow_pass_correction_revision(
            session,
            item_id,
            pass_id,
            principal=principal,
            expected_row_version=expected_row_version,
            reason_code=reason_code,
            now=now,
            _required_pair=(1, "DL1"),
        )
    except WorkflowPassCorrectionError as exc:
        raise _translate_correction_error(
            exc,
            error_cls=WorkflowDL1CorrectionError,
            not_found_cls=WorkflowDL1CorrectionNotFound,
            conflict_cls=WorkflowDL1CorrectionConflict,
        ) from exc


def create_dl2_correction_revision(
    session: Session,
    item_id: UUID | str,
    pass_id: UUID | str,
    *,
    principal: str,
    expected_row_version: int,
    reason_code: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Thin wrapper for immutable DL2 pre-submit correction."""

    try:
        return create_workflow_pass_correction_revision(
            session,
            item_id,
            pass_id,
            principal=principal,
            expected_row_version=expected_row_version,
            reason_code=reason_code,
            now=now,
            _required_pair=(2, "DL2"),
        )
    except WorkflowPassCorrectionError as exc:
        raise _translate_correction_error(
            exc,
            error_cls=WorkflowDL2CorrectionError,
            not_found_cls=WorkflowDL2CorrectionNotFound,
            conflict_cls=WorkflowDL2CorrectionConflict,
        ) from exc
