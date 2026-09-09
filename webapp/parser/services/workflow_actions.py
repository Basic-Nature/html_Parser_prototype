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
from webapp.parser.services.workflow_pre_qc_validation import (
    WORKFLOW_PRE_QC_VALIDATION_CONTRACT,
)
from webapp.parser.services.workflow_staging_binding import (
    WorkflowStagingBindingError,
    validate_completed_workflow_staging_binding,
)
from webapp.parser.utils.url_registry import lookup_exact_registry_entry


WORKFLOW_CLAIM_CONTRACT = "w3_pass_claim_v1"
APPROVED_SOURCE_CONTRACT = "w3_approved_source_projection_v1"
WORKFLOW_DL1_SUBMIT_CONTRACT = "workflow_dl1_submit_operation_v1"


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


class WorkflowSubmitConflict(WorkflowActionError):
    status_code = 409
    code = "workflow_dl1_submit_conflict"


def _normalize_pass_id(pass_id: UUID | str) -> UUID:
    try:
        return pass_id if isinstance(pass_id, UUID) else UUID(str(pass_id))
    except (TypeError, ValueError) as exc:
        raise WorkflowActionError(
            "workflow pass id must be a UUID."
        ) from exc


def _submit_expected_version(value: object) -> int:
    if isinstance(value, bool):
        raise WorkflowActionError(
            "expected_row_version must be an integer."
        )
    try:
        version = int(value)
    except (TypeError, ValueError) as exc:
        raise WorkflowActionError(
            "expected_row_version must be an integer."
        ) from exc
    if version < 0:
        raise WorkflowActionError(
            "expected_row_version must be >= 0."
        )
    return version


def _require_submit_pre_qc_binding(
    workflow_pass: WorkflowPass,
    binding: dict[str, Any],
) -> str:
    candidate = workflow_pass.candidate_check_result
    semantic = workflow_pass.semantic_validation_result

    if (
        workflow_pass.candidate_check_status != "complete"
        or workflow_pass.semantic_validation_status != "complete"
        or not isinstance(candidate, dict)
        or not isinstance(semantic, dict)
    ):
        raise WorkflowSubmitConflict(
            "DL1 submit requires completed server-owned Pre-QC validation."
        )

    semantic_hash = str(semantic.get("semantic_sha256") or "").strip()
    checks = (
        candidate.get("contract") == WORKFLOW_PRE_QC_VALIDATION_CONTRACT,
        semantic.get("contract") == WORKFLOW_PRE_QC_VALIDATION_CONTRACT,
        candidate.get("status") == "complete",
        semantic.get("status") == "complete",
        candidate.get("semantic_sha256") == semantic_hash,
        bool(semantic_hash),
        candidate.get("staging_batch_id") == binding["staging_batch_id"],
        semantic.get("staging_batch_id") == binding["staging_batch_id"],
        candidate.get("normalized_artifact_ref") == binding["artifact_ref"],
        semantic.get("normalized_artifact_ref") == binding["artifact_ref"],
        candidate.get("normalized_artifact_sha256")
            == binding["artifact_sha256"],
        semantic.get("normalized_artifact_sha256")
            == binding["artifact_sha256"],
        candidate.get("roster_completeness_claim") is False,
        semantic.get("null_zero_missing_policy") == "preserved_distinct",
        semantic.get("scope_exact_match") is True,
        semantic.get("binding_exact_match") is True,
    )
    if not all(checks):
        raise WorkflowSubmitConflict(
            "Stored Pre-QC evidence does not reconcile to the completed "
            "staging provenance binding."
        )
    return semantic_hash


def submit_first_workflow_pass(
    session: Session,
    item_id: UUID | str,
    *,
    pass_id: UUID | str,
    principal: str,
    expected_row_version: int,
    staging_batch_id: UUID | str,
    source_evidence_ref: str,
    artifact_ref: str,
    artifact_sha256: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Submit the current validated DL1 revision.

    Request values identify/assert the already-server-owned evidence. They are
    never used to create, rewrite, or backfill provenance or validation state.
    """

    actor = str(principal or "").strip()
    if not actor:
        raise WorkflowActionError(
            "Authenticated internal principal is required."
        )

    normalized_item = _normalize_item_id(item_id)
    normalized_pass = _normalize_pass_id(pass_id)
    try:
        normalized_batch = (
            staging_batch_id
            if isinstance(staging_batch_id, UUID)
            else UUID(str(staging_batch_id))
        )
    except (TypeError, ValueError) as exc:
        raise WorkflowActionError(
            "staging_batch_id must be a UUID."
        ) from exc

    expected_version = _submit_expected_version(expected_row_version)
    evidence_ref = str(source_evidence_ref or "").strip()
    artifact = str(artifact_ref or "").strip()
    artifact_hash = str(artifact_sha256 or "").strip()
    if not evidence_ref or not artifact or not artifact_hash:
        raise WorkflowActionError(
            "source_evidence_ref, artifact_ref, and artifact_sha256 "
            "must be non-empty."
        )

    timestamp = now or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    timestamp = timestamp.astimezone(timezone.utc)

    item = session.execute(
        select(WorkflowItem)
        .where(WorkflowItem.id == normalized_item)
        .with_for_update()
    ).scalar_one_or_none()
    if item is None:
        raise WorkflowActionNotFound("Workflow item was not found.")

    if int(item.row_version) != expected_version:
        raise WorkflowSubmitConflict(
            "Workflow row_version changed before DL1 submit."
        )

    if (
        item.lifecycle_state,
        item.current_stage,
        item.stage_condition,
    ) != ("active", "independent_acquisition", "in_progress"):
        raise WorkflowSubmitConflict(
            "Workflow item is not in submit-eligible DL1 acquisition state."
        )

    workflow_pass = session.execute(
        select(WorkflowPass)
        .where(WorkflowPass.id == normalized_pass)
        .with_for_update()
    ).scalar_one_or_none()
    if workflow_pass is None:
        raise WorkflowActionNotFound("Workflow pass was not found.")

    if (
        workflow_pass.workflow_item_id != item.id
        or workflow_pass.pass_number != 1
        or workflow_pass.pass_label != "DL1"
        or workflow_pass.is_current is not True
        or workflow_pass.status != "in_progress"
        or workflow_pass.submitted_at is not None
        or str(workflow_pass.assigned_principal or "").strip() != actor
    ):
        raise WorkflowSubmitConflict(
            "DL1 submit requires the current in-progress pre-submit "
            "revision assigned to the requesting principal."
        )

    if workflow_pass.staging_batch_id != normalized_batch:
        raise WorkflowSubmitConflict(
            "Requested staging_batch_id is not the current server-bound DL1 batch."
        )

    try:
        binding = validate_completed_workflow_staging_binding(
            session,
            item.id,
            workflow_pass.id,
            normalized_batch,
            principal=actor,
        )
    except WorkflowStagingBindingError as exc:
        raise WorkflowSubmitConflict(
            f"Completed staging provenance validation failed: {exc}"
        ) from exc

    request_assertions = {
        "source_evidence_ref": evidence_ref,
        "artifact_ref": artifact,
        "artifact_sha256": artifact_hash,
    }
    binding_assertions = {
        "source_evidence_ref": binding["source_evidence_ref"],
        "artifact_ref": binding["artifact_ref"],
        "artifact_sha256": binding["artifact_sha256"],
    }
    if request_assertions != binding_assertions:
        raise WorkflowSubmitConflict(
            "DL1 submit evidence assertions do not match server-owned provenance."
        )

    semantic_hash = _require_submit_pre_qc_binding(
        workflow_pass,
        binding,
    )

    prior_state = {
        "lifecycle_state": item.lifecycle_state,
        "current_stage": item.current_stage,
        "stage_condition": item.stage_condition,
        "row_version": item.row_version,
        "pass_id": str(workflow_pass.id),
        "pass_status": workflow_pass.status,
    }

    workflow_pass.status = "submitted"
    workflow_pass.submitted_at = timestamp
    workflow_pass.updated_at = timestamp

    item.stage_condition = "ready"
    item.row_version = expected_version + 1
    item.updated_at = timestamp

    new_state = {
        "lifecycle_state": item.lifecycle_state,
        "current_stage": item.current_stage,
        "stage_condition": item.stage_condition,
        "row_version": item.row_version,
        "pass_id": str(workflow_pass.id),
        "pass_status": workflow_pass.status,
    }

    event = WorkflowEvent(
        workflow_item_id=item.id,
        actor_type="principal",
        actor_principal=actor,
        actor_service=None,
        event_type="pass_submitted",
        stage="independent_acquisition",
        prior_state=prior_state,
        new_state=new_state,
        related_pass_id=workflow_pass.id,
        related_comparison_id=None,
        related_review_id=None,
        related_staging_batch_id=workflow_pass.staging_batch_id,
        related_canonical_race_id=item.canonical_race_id,
        reason_code=None,
        summary="Validated DL1 revision submitted for DL2 eligibility.",
        event_metadata={
            "contract": WORKFLOW_DL1_SUBMIT_CONTRACT,
            "pass_number": 1,
            "pass_label": "DL1",
            "revision_number": workflow_pass.revision_number,
            "source_evidence_ref": binding["source_evidence_ref"],
            "artifact_ref": binding["artifact_ref"],
            "artifact_sha256": binding["artifact_sha256"],
            "semantic_sha256": semantic_hash,
            "dl2_auto_claimed": False,
        },
        occurred_at=timestamp,
    )
    session.add(event)
    session.flush()

    return {
        "success": True,
        "contract": WORKFLOW_DL1_SUBMIT_CONTRACT,
        "task_id": str(item.id),
        "pass_id": str(workflow_pass.id),
        "pass_number": 1,
        "pass_label": "DL1",
        "revision_number": workflow_pass.revision_number,
        "status": workflow_pass.status,
        "lifecycle_state": item.lifecycle_state,
        "current_stage": item.current_stage,
        "stage_condition": item.stage_condition,
        "row_version": item.row_version,
        "staging_batch_id": str(workflow_pass.staging_batch_id),
        "source_evidence_ref": binding["source_evidence_ref"],
        "artifact_ref": binding["artifact_ref"],
        "artifact_sha256": binding["artifact_sha256"],
        "semantic_sha256": semantic_hash,
        "event_id": str(event.id),
        "dl2_auto_claimed": False,
        "committed": False,
    }
