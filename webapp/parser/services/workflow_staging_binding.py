"""Server-owned provenance binding for governed Workflow staging batches.

This module is not a route and does not authorize callers. Callers enforce
Workflow authority before entering it. All functions use the caller-provided
SQLAlchemy Session and never commit or roll back.

Binding is two-phase:
1. begin_workflow_staging_binding() creates a PENDING BatchMetadata row and
   binds its batch_id to the current governed acquisition WorkflowPass.
2. finalize_workflow_staging_binding() validates staged rows, freezes evidence
   identity, marks the batch COMPLETED, creates a WorkflowArtifactLink, and
   appends an audit event.

Candidate and semantic validation are separate Pre-QC responsibilities.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from webapp.parser.utils.models import (
    BatchMetadata,
    StagingElectionResult,
    StatusEnum,
    WorkflowArtifactLink,
    WorkflowEvent,
    WorkflowItem,
    WorkflowPass,
)
from webapp.parser.utils.url_registry import lookup_exact_registry_entry


WORKFLOW_STAGING_BINDING_CONTRACT = "workflow_staging_binding_v1"
WORKFLOW_STAGING_BINDING_RELATION = "workflow_staging_provenance"
WORKFLOW_STAGING_BINDING_ARTIFACT_TYPE = "workflow_staging_artifact"
WORKFLOW_STAGING_BINDING_SERVICE = "workflow_staging_binding"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class WorkflowStagingBindingError(RuntimeError):
    status_code = 400
    code = "workflow_staging_binding_error"


class WorkflowStagingBindingNotFound(WorkflowStagingBindingError):
    status_code = 404
    code = "workflow_staging_binding_not_found"


class WorkflowStagingBindingConflict(WorkflowStagingBindingError):
    status_code = 409
    code = "workflow_staging_binding_conflict"


class WorkflowStagingBindingSourceRejected(WorkflowStagingBindingError):
    status_code = 409
    code = "workflow_staging_binding_source_rejected"


def _uuid(value: UUID | str, *, name: str) -> UUID:
    try:
        return value if isinstance(value, UUID) else UUID(str(value))
    except (TypeError, ValueError) as exc:
        raise WorkflowStagingBindingError(f"{name} must be a UUID.") from exc


def _actor(principal: str) -> str:
    actor = str(principal or "").strip()
    if not actor:
        raise WorkflowStagingBindingError(
            "Authenticated internal principal is required."
        )
    return actor


def _utc(now: datetime | None) -> datetime:
    value = now or datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _status_is(value: Any, expected: StatusEnum) -> bool:
    return (
        value is expected
        or value == expected
        or getattr(value, "name", None) == expected.name
        or str(value) == expected.name
        or str(value) == expected.value
        or str(value) == f"StatusEnum.{expected.name}"
    )


_SUPPORTED_GOVERNED_PASS_PAIRS = frozenset({
    (1, "DL1"),
    (2, "DL2"),
})


def _pass_matches_governed_acquisition(
    workflow_pass: WorkflowPass,
    *,
    item_id: UUID,
    actor: str,
) -> bool:
    return (
        workflow_pass.workflow_item_id == item_id
        and (
            workflow_pass.pass_number,
            workflow_pass.pass_label,
        ) in _SUPPORTED_GOVERNED_PASS_PAIRS
        and workflow_pass.is_current is True
        and workflow_pass.status == "in_progress"
        and str(workflow_pass.assigned_principal or "").strip() == actor
    )


def _snapshot(
    *,
    item: WorkflowItem,
    workflow_pass: WorkflowPass,
    actor: str,
    source_url: str,
    binding_state: str,
    artifact_ref: str | None,
    artifact_sha256: str | None,
    source_evidence_ref: str | None,
    row_count: int | None,
) -> dict[str, Any]:
    return {
        "contract": WORKFLOW_STAGING_BINDING_CONTRACT,
        "binding_state": binding_state,
        "workflow_item_id": str(item.id),
        "workflow_pass_id": str(workflow_pass.id),
        "pass_number": workflow_pass.pass_number,
        "pass_label": workflow_pass.pass_label,
        "revision_number": workflow_pass.revision_number,
        "assigned_principal": actor,
        "exact_source_url": source_url,
        "source_race_id": item.source_race_id,
        "artifact_ref": artifact_ref,
        "artifact_sha256": artifact_sha256,
        "source_evidence_ref": source_evidence_ref,
        "row_count": row_count,
    }


def _load_item_pass_for_update(
    session: Session,
    *,
    item_id: UUID,
    pass_id: UUID,
    actor: str,
) -> tuple[WorkflowItem, WorkflowPass]:
    item = session.execute(
        select(WorkflowItem)
        .where(WorkflowItem.id == item_id)
        .with_for_update()
    ).scalar_one_or_none()
    if item is None:
        raise WorkflowStagingBindingNotFound("Workflow item was not found.")

    workflow_pass = session.execute(
        select(WorkflowPass)
        .where(WorkflowPass.id == pass_id)
        .with_for_update()
    ).scalar_one_or_none()
    if workflow_pass is None:
        raise WorkflowStagingBindingNotFound("Workflow pass was not found.")

    if (
        item.lifecycle_state,
        item.current_stage,
        item.stage_condition,
    ) != ("active", "independent_acquisition", "in_progress"):
        raise WorkflowStagingBindingConflict(
            "Workflow item is not in active independent acquisition state."
        )
    if not _pass_matches_governed_acquisition(
        workflow_pass,
        item_id=item.id,
        actor=actor,
    ):
        raise WorkflowStagingBindingConflict(
            f"Current {workflow_pass.pass_label} pass does not belong to the requesting principal."
        )
    return item, workflow_pass


def begin_workflow_staging_binding(
    session: Session,
    item_id: UUID | str,
    pass_id: UUID | str,
    *,
    principal: str,
    registry_path: Path,
    now: datetime | None = None,
) -> dict[str, Any]:
    actor = _actor(principal)
    normalized_item = _uuid(item_id, name="item_id")
    normalized_pass = _uuid(pass_id, name="pass_id")
    timestamp = _utc(now)

    item, workflow_pass = _load_item_pass_for_update(
        session,
        item_id=normalized_item,
        pass_id=normalized_pass,
        actor=actor,
    )
    if workflow_pass.staging_batch_id is not None:
        raise WorkflowStagingBindingConflict(
            f"Current {workflow_pass.pass_label} pass already has a staging batch binding."
        )

    entry = lookup_exact_registry_entry(
        str(item.source_url or ""),
        path=registry_path,
    )
    if entry is None or entry.registry_category != "curated":
        raise WorkflowStagingBindingSourceRejected(
            "Workflow staging requires the exact curated registry source."
        )

    batch = BatchMetadata(
        source=(
            f"workflow:{item.id}:"
            f"{workflow_pass.pass_label}:r{workflow_pass.revision_number}"
        ),
        started_at=timestamp,
        completed_at=None,
        status=StatusEnum.PENDING,
        metastats=_snapshot(
            item=item,
            workflow_pass=workflow_pass,
            actor=actor,
            source_url=entry.url,
            binding_state="pending",
            artifact_ref=None,
            artifact_sha256=None,
            source_evidence_ref=None,
            row_count=None,
        ),
    )
    session.add(batch)
    session.flush()

    workflow_pass.staging_batch_id = batch.batch_id
    workflow_pass.updated_at = timestamp

    event = WorkflowEvent(
        workflow_item_id=item.id,
        actor_type="service",
        actor_principal=actor,
        actor_service=WORKFLOW_STAGING_BINDING_SERVICE,
        event_type="staging_binding_started",
        stage="independent_acquisition",
        prior_state={
            "pass_id": str(workflow_pass.id),
            "staging_batch_id": None,
        },
        new_state={
            "pass_id": str(workflow_pass.id),
            "staging_batch_id": str(batch.batch_id),
            "binding_state": "pending",
        },
        related_pass_id=workflow_pass.id,
        related_comparison_id=None,
        related_review_id=None,
        related_staging_batch_id=batch.batch_id,
        related_canonical_race_id=item.canonical_race_id,
        reason_code=None,
        summary=(f"Server-owned {workflow_pass.pass_label} staging "
                 "provenance binding started."),
        event_metadata={
            "contract": WORKFLOW_STAGING_BINDING_CONTRACT,
            "pass_number": workflow_pass.pass_number,
            "pass_label": workflow_pass.pass_label,
        },
        occurred_at=timestamp,
    )
    session.add(event)
    session.flush()

    return {
        "success": True,
        "contract": WORKFLOW_STAGING_BINDING_CONTRACT,
        "task_id": str(item.id),
        "pass_id": str(workflow_pass.id),
        "staging_batch_id": str(batch.batch_id),
        "event_id": str(event.id),
        "binding_state": "pending",
        "source_url": entry.url,
        "committed": False,
    }


def finalize_workflow_staging_binding(
    session: Session,
    item_id: UUID | str,
    pass_id: UUID | str,
    staging_batch_id: UUID | str,
    *,
    principal: str,
    source_evidence_ref: str,
    artifact_ref: str,
    artifact_sha256: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    actor = _actor(principal)
    normalized_item = _uuid(item_id, name="item_id")
    normalized_pass = _uuid(pass_id, name="pass_id")
    normalized_batch = _uuid(staging_batch_id, name="staging_batch_id")
    timestamp = _utc(now)

    evidence_ref = str(source_evidence_ref or "").strip()
    artifact = str(artifact_ref or "").strip()
    artifact_hash = str(artifact_sha256 or "").strip()
    if not evidence_ref:
        raise WorkflowStagingBindingError(
            "source_evidence_ref must be non-empty."
        )
    if not artifact:
        raise WorkflowStagingBindingError("artifact_ref must be non-empty.")
    if _SHA256_RE.fullmatch(artifact_hash) is None:
        raise WorkflowStagingBindingError(
            "artifact_sha256 must be lowercase 64-character hexadecimal."
        )

    item, workflow_pass = _load_item_pass_for_update(
        session,
        item_id=normalized_item,
        pass_id=normalized_pass,
        actor=actor,
    )
    if workflow_pass.staging_batch_id != normalized_batch:
        raise WorkflowStagingBindingConflict(
            f"Staging batch is not the server-bound batch for current {workflow_pass.pass_label}."
        )

    batch = session.execute(
        select(BatchMetadata)
        .where(BatchMetadata.batch_id == normalized_batch)
        .with_for_update()
    ).scalar_one_or_none()
    if batch is None:
        raise WorkflowStagingBindingNotFound(
            "Workflow staging batch was not found."
        )
    if not _status_is(batch.status, StatusEnum.PENDING):
        raise WorkflowStagingBindingConflict(
            "Workflow staging batch is not pending."
        )

    metadata = batch.metastats if isinstance(batch.metastats, dict) else {}
    expected = _snapshot(
        item=item,
        workflow_pass=workflow_pass,
        actor=actor,
        source_url=str(item.source_url or ""),
        binding_state="pending",
        artifact_ref=None,
        artifact_sha256=None,
        source_evidence_ref=None,
        row_count=None,
    )
    if metadata != expected:
        raise WorkflowStagingBindingConflict(
            "Pending staging binding identity no longer matches Workflow authority."
        )

    rows = session.execute(
        select(StagingElectionResult)
        .where(StagingElectionResult.batch_id == normalized_batch)
    ).scalars().all()
    if not rows:
        raise WorkflowStagingBindingConflict(
            "Workflow staging batch contains no staged election rows."
        )
    if any(str(row.source_url or "") != expected["exact_source_url"] for row in rows):
        raise WorkflowStagingBindingConflict(
            "Staged row source URL does not exactly match bound Workflow source."
        )

    existing_link = session.execute(
        select(WorkflowArtifactLink.id).where(
            WorkflowArtifactLink.workflow_item_id == item.id,
            WorkflowArtifactLink.pass_id == workflow_pass.id,
            WorkflowArtifactLink.relation_type
                == WORKFLOW_STAGING_BINDING_RELATION,
        )
    ).first()
    if existing_link is not None:
        raise WorkflowStagingBindingConflict(
            "Completed staging provenance artifact link already exists."
        )

    row_count = len(rows)
    batch.status = StatusEnum.COMPLETED
    batch.completed_at = timestamp
    batch.metastats = _snapshot(
        item=item,
        workflow_pass=workflow_pass,
        actor=actor,
        source_url=expected["exact_source_url"],
        binding_state="complete",
        artifact_ref=artifact,
        artifact_sha256=artifact_hash,
        source_evidence_ref=evidence_ref,
        row_count=row_count,
    )
    workflow_pass.source_evidence_ref = evidence_ref
    workflow_pass.updated_at = timestamp

    link = WorkflowArtifactLink(
        workflow_item_id=item.id,
        pass_id=workflow_pass.id,
        relation_type=WORKFLOW_STAGING_BINDING_RELATION,
        artifact_type=WORKFLOW_STAGING_BINDING_ARTIFACT_TYPE,
        artifact_ref=artifact,
        artifact_sha256=artifact_hash,
        canonical_source_artifact_id=None,
        staging_batch_id=batch.batch_id,
        artifact_metadata={
            "contract": WORKFLOW_STAGING_BINDING_CONTRACT,
            "binding_state": "complete",
            "source_evidence_ref": evidence_ref,
            "row_count": row_count,
            "exact_source_url": expected["exact_source_url"],
        },
        created_at=timestamp,
    )
    session.add(link)
    session.flush()

    event = WorkflowEvent(
        workflow_item_id=item.id,
        actor_type="service",
        actor_principal=actor,
        actor_service=WORKFLOW_STAGING_BINDING_SERVICE,
        event_type="staging_binding_completed",
        stage="independent_acquisition",
        prior_state={
            "pass_id": str(workflow_pass.id),
            "staging_batch_id": str(batch.batch_id),
            "binding_state": "pending",
        },
        new_state={
            "pass_id": str(workflow_pass.id),
            "staging_batch_id": str(batch.batch_id),
            "binding_state": "complete",
            "artifact_sha256": artifact_hash,
            "row_count": row_count,
        },
        related_pass_id=workflow_pass.id,
        related_comparison_id=None,
        related_review_id=None,
        related_staging_batch_id=batch.batch_id,
        related_canonical_race_id=item.canonical_race_id,
        reason_code=None,
        summary=(f"Server-owned {workflow_pass.pass_label} staging "
                 "provenance binding completed."),
        event_metadata={
            "contract": WORKFLOW_STAGING_BINDING_CONTRACT,
            "pass_number": workflow_pass.pass_number,
            "pass_label": workflow_pass.pass_label,
            "artifact_ref": artifact,
            "artifact_sha256": artifact_hash,
        },
        occurred_at=timestamp,
    )
    session.add(event)
    session.flush()

    return {
        "success": True,
        "contract": WORKFLOW_STAGING_BINDING_CONTRACT,
        "task_id": str(item.id),
        "pass_id": str(workflow_pass.id),
        "staging_batch_id": str(batch.batch_id),
        "artifact_link_id": str(link.id),
        "event_id": str(event.id),
        "binding_state": "complete",
        "source_evidence_ref": evidence_ref,
        "artifact_ref": artifact,
        "artifact_sha256": artifact_hash,
        "row_count": row_count,
        "committed": False,
    }


def validate_completed_workflow_staging_binding(
    session: Session,
    item_id: UUID | str,
    pass_id: UUID | str,
    staging_batch_id: UUID | str,
    *,
    principal: str,
) -> dict[str, Any]:
    actor = _actor(principal)
    normalized_item = _uuid(item_id, name="item_id")
    normalized_pass = _uuid(pass_id, name="pass_id")
    normalized_batch = _uuid(staging_batch_id, name="staging_batch_id")

    item = session.get(WorkflowItem, normalized_item)
    workflow_pass = session.get(WorkflowPass, normalized_pass)
    batch = session.get(BatchMetadata, normalized_batch)
    if item is None or workflow_pass is None or batch is None:
        raise WorkflowStagingBindingNotFound(
            "Completed Workflow staging binding was not found."
        )
    if not _pass_matches_governed_acquisition(
        workflow_pass,
        item_id=item.id,
        actor=actor,
    ):
        raise WorkflowStagingBindingConflict(
            f"Completed binding does not belong to current {workflow_pass.pass_label} principal."
        )
    if workflow_pass.staging_batch_id != batch.batch_id:
        raise WorkflowStagingBindingConflict(
            f"Current {workflow_pass.pass_label} pass does not reference the supplied staging batch."
        )
    if not _status_is(batch.status, StatusEnum.COMPLETED):
        raise WorkflowStagingBindingConflict(
            "Workflow staging batch is not completed."
        )

    metadata = batch.metastats if isinstance(batch.metastats, dict) else {}
    if (
        metadata.get("contract") != WORKFLOW_STAGING_BINDING_CONTRACT
        or metadata.get("binding_state") != "complete"
        or metadata.get("workflow_item_id") != str(item.id)
        or metadata.get("workflow_pass_id") != str(workflow_pass.id)
        or metadata.get("pass_number") != workflow_pass.pass_number
        or metadata.get("pass_label") != workflow_pass.pass_label
        or metadata.get("revision_number") != workflow_pass.revision_number
        or metadata.get("assigned_principal") != actor
        or metadata.get("exact_source_url") != str(item.source_url or "")
        or metadata.get("source_race_id") != item.source_race_id
        or not metadata.get("source_evidence_ref")
        or not metadata.get("artifact_ref")
        or _SHA256_RE.fullmatch(str(metadata.get("artifact_sha256") or "")) is None
        or not isinstance(metadata.get("row_count"), int)
        or metadata.get("row_count") < 1
    ):
        raise WorkflowStagingBindingConflict(
            "Completed staging metadata failed provenance validation."
        )

    link = session.execute(
        select(WorkflowArtifactLink).where(
            WorkflowArtifactLink.workflow_item_id == item.id,
            WorkflowArtifactLink.pass_id == workflow_pass.id,
            WorkflowArtifactLink.relation_type
                == WORKFLOW_STAGING_BINDING_RELATION,
            WorkflowArtifactLink.staging_batch_id == batch.batch_id,
        )
    ).scalar_one_or_none()
    if link is None:
        raise WorkflowStagingBindingConflict(
            "Completed staging provenance artifact link is missing."
        )
    if (
        link.artifact_type != WORKFLOW_STAGING_BINDING_ARTIFACT_TYPE
        or link.artifact_ref != metadata["artifact_ref"]
        or link.artifact_sha256 != metadata["artifact_sha256"]
        or workflow_pass.source_evidence_ref != metadata["source_evidence_ref"]
    ):
        raise WorkflowStagingBindingConflict(
            "Completed staging provenance artifact identity does not reconcile."
        )

    return {
        "success": True,
        "contract": WORKFLOW_STAGING_BINDING_CONTRACT,
        "task_id": str(item.id),
        "pass_id": str(workflow_pass.id),
        "staging_batch_id": str(batch.batch_id),
        "binding_state": "complete",
        "source_evidence_ref": metadata["source_evidence_ref"],
        "artifact_ref": metadata["artifact_ref"],
        "artifact_sha256": metadata["artifact_sha256"],
        "row_count": metadata["row_count"],
        "committed": False,
    }


def validate_frozen_workflow_staging_binding(
    session: Session,
    item_id: UUID | str,
    pass_id: UUID | str,
    staging_batch_id: UUID | str,
    *,
    required_pass_number: int,
    required_pass_label: str,
) -> dict[str, Any]:
    # Read-only submitted-pass provenance validation for comparison service.
    normalized_item = _uuid(item_id, name="item_id")
    normalized_pass = _uuid(pass_id, name="pass_id")
    normalized_batch = _uuid(staging_batch_id, name="staging_batch_id")

    item = session.get(WorkflowItem, normalized_item)
    workflow_pass = session.get(WorkflowPass, normalized_pass)
    batch = session.get(BatchMetadata, normalized_batch)
    if item is None or workflow_pass is None or batch is None:
        raise WorkflowStagingBindingNotFound(
            "Frozen submitted Workflow staging binding was not found."
        )

    pair = (required_pass_number, required_pass_label)
    if pair not in _SUPPORTED_GOVERNED_PASS_PAIRS:
        raise WorkflowStagingBindingConflict(
            "Frozen binding requires a supported governed pass identity."
        )
    actor = str(workflow_pass.assigned_principal or "").strip()
    if (
        workflow_pass.workflow_item_id != item.id
        or (workflow_pass.pass_number, workflow_pass.pass_label) != pair
        or workflow_pass.is_current is not True
        or workflow_pass.status != "submitted"
        or workflow_pass.submitted_at is None
        or not actor
        or workflow_pass.staging_batch_id != batch.batch_id
    ):
        raise WorkflowStagingBindingConflict(
            "Frozen binding requires the exact current submitted governed pass."
        )
    if not _status_is(batch.status, StatusEnum.COMPLETED):
        raise WorkflowStagingBindingConflict(
            "Frozen Workflow staging batch is not completed."
        )

    metadata = batch.metastats if isinstance(batch.metastats, dict) else {}
    if (
        metadata.get("contract") != WORKFLOW_STAGING_BINDING_CONTRACT
        or metadata.get("binding_state") != "complete"
        or metadata.get("workflow_item_id") != str(item.id)
        or metadata.get("workflow_pass_id") != str(workflow_pass.id)
        or metadata.get("pass_number") != workflow_pass.pass_number
        or metadata.get("pass_label") != workflow_pass.pass_label
        or metadata.get("revision_number") != workflow_pass.revision_number
        or metadata.get("assigned_principal") != actor
        or metadata.get("exact_source_url") != str(item.source_url or "")
        or metadata.get("source_race_id") != item.source_race_id
        or not metadata.get("source_evidence_ref")
        or not metadata.get("artifact_ref")
        or _SHA256_RE.fullmatch(str(metadata.get("artifact_sha256") or ""))
            is None
        or not isinstance(metadata.get("row_count"), int)
        or metadata.get("row_count") < 1
    ):
        raise WorkflowStagingBindingConflict(
            "Frozen completed staging metadata failed provenance validation."
        )

    links = session.execute(
        select(WorkflowArtifactLink).where(
            WorkflowArtifactLink.workflow_item_id == item.id,
            WorkflowArtifactLink.pass_id == workflow_pass.id,
            WorkflowArtifactLink.relation_type
                == WORKFLOW_STAGING_BINDING_RELATION,
            WorkflowArtifactLink.staging_batch_id == batch.batch_id,
        )
    ).scalars().all()
    if len(links) != 1:
        raise WorkflowStagingBindingConflict(
            "Frozen staging provenance requires exactly one artifact link."
        )
    link = links[0]
    link_metadata = (
        link.artifact_metadata
        if isinstance(link.artifact_metadata, dict)
        else {}
    )
    if (
        link.artifact_type != WORKFLOW_STAGING_BINDING_ARTIFACT_TYPE
        or link.artifact_ref != metadata["artifact_ref"]
        or link.artifact_sha256 != metadata["artifact_sha256"]
        or workflow_pass.source_evidence_ref != metadata["source_evidence_ref"]
        or link_metadata.get("contract") != WORKFLOW_STAGING_BINDING_CONTRACT
        or link_metadata.get("binding_state") != "complete"
        or link_metadata.get("source_evidence_ref")
            != metadata["source_evidence_ref"]
        or link_metadata.get("row_count") != metadata["row_count"]
        or link_metadata.get("exact_source_url")
            != metadata["exact_source_url"]
    ):
        raise WorkflowStagingBindingConflict(
            "Frozen staging provenance artifact identity does not reconcile."
        )

    rows = session.execute(
        select(StagingElectionResult).where(
            StagingElectionResult.batch_id == batch.batch_id
        )
    ).scalars().all()
    if (
        len(rows) != metadata["row_count"]
        or any(
            str(row.source_url or "") != metadata["exact_source_url"]
            for row in rows
        )
    ):
        raise WorkflowStagingBindingConflict(
            "Frozen staging rows do not reconcile to completed provenance."
        )

    comparison_binding = {
        "workflow_item_id": str(item.id),
        "workflow_pass_id": str(workflow_pass.id),
        "pass_number": workflow_pass.pass_number,
        "revision_number": workflow_pass.revision_number,
        "source_evidence_ref": metadata["source_evidence_ref"],
        "staging_batch_id": str(batch.batch_id),
        "normalized_artifact_ref": metadata["artifact_ref"],
        "normalized_artifact_sha256": metadata["artifact_sha256"],
    }
    return {
        "success": True,
        "contract": WORKFLOW_STAGING_BINDING_CONTRACT,
        "task_id": str(item.id),
        "pass_id": str(workflow_pass.id),
        "pass_number": workflow_pass.pass_number,
        "pass_label": workflow_pass.pass_label,
        "assigned_principal": actor,
        "staging_batch_id": str(batch.batch_id),
        "source_evidence_ref": metadata["source_evidence_ref"],
        "artifact_ref": metadata["artifact_ref"],
        "artifact_sha256": metadata["artifact_sha256"],
        "row_count": metadata["row_count"],
        "comparison_binding": comparison_binding,
        "committed": False,
    }

