from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
from typing import Callable, Mapping
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from webapp.parser.contracts.workflow_authorization import (
    CAP_COMPARISON_EXECUTE,
    ROLE_COMPARISON_SERVICE,
)
from webapp.parser.contracts.workflow_comparison import (
    DISCREPANCY_CATEGORIES,
    WORKFLOW_COMPARISON_VERSION,
    build_difference_summary,
    enumerate_semantic_differences,
    strict_semantic_equality,
)
from webapp.parser.contracts.workflow_lifecycle import (
    assert_forward_stage_transition,
    next_stage_after_comparison,
)
from webapp.parser.services.workflow_pre_qc_validation import (
    WorkflowPreQCValidationError,
    validate_frozen_workflow_pre_qc_payload,
)
from webapp.parser.services.workflow_staging_binding import (
    WorkflowStagingBindingError,
    validate_frozen_workflow_staging_binding,
)
from webapp.parser.utils.models import (
    WorkflowComparison,
    WorkflowDiscrepancy,
    WorkflowEvent,
    WorkflowItem,
    WorkflowPass,
)


WORKFLOW_STRICT_COMPARISON_CONTRACT = "workflow_strict_comparison_execution_v1"
WORKFLOW_STRICT_COMPARISON_SERVICE = "workflow_strict_comparison"
WORKFLOW_STRICT_COMPARISON_ROLE = ROLE_COMPARISON_SERVICE
WORKFLOW_STRICT_COMPARISON_CAPABILITY = CAP_COMPARISON_EXECUTE

ArtifactLoader = Callable[[str], bytes]
ComparisonPayloadAdapter = Callable[[str, bytes, Mapping[str, object]], object]


class WorkflowStrictComparisonError(RuntimeError):
    status_code = 422
    code = "workflow_strict_comparison_error"


class WorkflowStrictComparisonConflict(WorkflowStrictComparisonError):
    status_code = 409
    code = "workflow_strict_comparison_conflict"


def _uuid(value: UUID | str, *, name: str) -> UUID:
    try:
        return value if isinstance(value, UUID) else UUID(str(value))
    except (TypeError, ValueError) as exc:
        raise WorkflowStrictComparisonError(
            f"{name} must be a UUID."
        ) from exc


def _expected_version(value: object) -> int:
    if isinstance(value, bool):
        raise WorkflowStrictComparisonError(
            "expected_row_version must be an integer."
        )
    try:
        version = int(value)
    except (TypeError, ValueError) as exc:
        raise WorkflowStrictComparisonError(
            "expected_row_version must be an integer."
        ) from exc
    if version < 0:
        raise WorkflowStrictComparisonError(
            "expected_row_version must be >= 0."
        )
    return version


def _service_version(value: str) -> str:
    normalized = str(value or "").strip()
    if not normalized or len(normalized) > 128:
        raise WorkflowStrictComparisonError(
            "service_version must be a non-empty string up to 128 characters."
        )
    return normalized


def _utc(now: datetime | None) -> datetime:
    value = now or datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _load_current_submitted_pair(
    session: Session,
    item: WorkflowItem,
) -> tuple[WorkflowPass, WorkflowPass]:
    passes = session.execute(
        select(WorkflowPass)
        .where(
            WorkflowPass.workflow_item_id == item.id,
            WorkflowPass.is_current.is_(True),
            WorkflowPass.pass_number.in_((1, 2)),
        )
        .with_for_update()
    ).scalars().all()
    if len(passes) != 2:
        raise WorkflowStrictComparisonConflict(
            "Strict comparison requires exactly one current DL1 and DL2."
        )
    by_pair = {
        (workflow_pass.pass_number, workflow_pass.pass_label): workflow_pass
        for workflow_pass in passes
    }
    if set(by_pair) != {(1, "DL1"), (2, "DL2")}:
        raise WorkflowStrictComparisonConflict(
            "Strict comparison current pass identities are invalid."
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
        or not p1
        or not p2
        or p1 == p2
    ):
        raise WorkflowStrictComparisonConflict(
            "Strict comparison requires immutable submitted independent DL1/DL2."
        )
    return dl1, dl2


def _load_verified_payload(
    session: Session,
    item: WorkflowItem,
    workflow_pass: WorkflowPass,
    *,
    artifact_loader: ArtifactLoader,
    comparison_payload_adapter: ComparisonPayloadAdapter,
) -> tuple[dict[str, object], dict[str, object]]:
    frozen = validate_frozen_workflow_staging_binding(
        session,
        item.id,
        workflow_pass.id,
        workflow_pass.staging_batch_id,
        required_pass_number=workflow_pass.pass_number,
        required_pass_label=workflow_pass.pass_label,
    )
    artifact_ref = str(frozen["artifact_ref"])
    expected_hash = str(frozen["artifact_sha256"])
    try:
        raw = artifact_loader(artifact_ref)
    except Exception as exc:
        raise WorkflowStrictComparisonConflict(
            "Server-owned normalized artifact loader failed."
        ) from exc
    if not isinstance(raw, bytes):
        raise WorkflowStrictComparisonConflict(
            "Normalized artifact loader must return bytes."
        )
    observed_hash = hashlib.sha256(raw).hexdigest()
    if observed_hash != expected_hash:
        raise WorkflowStrictComparisonConflict(
            "Normalized artifact bytes do not match frozen SHA-256."
        )

    binding = frozen["comparison_binding"]
    assert isinstance(binding, Mapping)
    try:
        adapted = comparison_payload_adapter(
            artifact_ref,
            raw,
            dict(binding),
        )
    except Exception as exc:
        raise WorkflowStrictComparisonConflict(
            "Server-owned comparison payload adapter failed."
        ) from exc

    try:
        payload = validate_frozen_workflow_pre_qc_payload(
            workflow_pass,
            frozen,
            adapted,
        )
    except WorkflowPreQCValidationError as exc:
        raise WorkflowStrictComparisonConflict(str(exc)) from exc

    return frozen, payload


def _category_counts(
    differences: list[dict[str, object]],
) -> dict[str, int]:
    counts = Counter(str(diff["category"]) for diff in differences)
    return {
        category: int(counts.get(category, 0))
        for category in DISCREPANCY_CATEGORIES
    }


def _validate_existing_replay(
    session: Session,
    item: WorkflowItem,
    dl1: WorkflowPass,
    dl2: WorkflowPass,
    comparison: WorkflowComparison,
) -> dict[str, object]:
    if (
        comparison.left_pass_id != dl1.id
        or comparison.right_pass_id != dl2.id
        or comparison.comparison_version != WORKFLOW_COMPARISON_VERSION
        or comparison.status != "complete"
        or not isinstance(comparison.strict_equality_passed, bool)
        or not isinstance(comparison.difference_count, int)
        or comparison.difference_count < 0
        or not isinstance(comparison.difference_summary, Mapping)
        or comparison.checked_at is None
        or not str(comparison.checked_by_service_version or "").strip()
    ):
        raise WorkflowStrictComparisonConflict(
            "Existing comparison is not an exact completed replay authority."
        )

    expected_stage = next_stage_after_comparison(
        comparison.strict_equality_passed
    )
    if (
        item.lifecycle_state != "active"
        or item.current_stage != expected_stage
        or item.stage_condition != "pending"
    ):
        raise WorkflowStrictComparisonConflict(
            "Existing comparison downstream Workflow state is inconsistent."
        )

    discrepancies = session.execute(
        select(WorkflowDiscrepancy).where(
            WorkflowDiscrepancy.comparison_id == comparison.id
        )
    ).scalars().all()
    if len(discrepancies) != comparison.difference_count:
        raise WorkflowStrictComparisonConflict(
            "Existing comparison discrepancy count is inconsistent."
        )
    if comparison.strict_equality_passed and discrepancies:
        raise WorkflowStrictComparisonConflict(
            "Exact comparison replay cannot contain discrepancies."
        )
    if any(
        discrepancy.resolution_status != "open"
        or discrepancy.severity is not None
        for discrepancy in discrepancies
    ):
        raise WorkflowStrictComparisonConflict(
            "Existing comparison discrepancy state is not pristine."
        )

    counts = Counter(discrepancy.category for discrepancy in discrepancies)
    summary = comparison.difference_summary
    rebuilt = build_difference_summary(
        left_semantic_sha256=str(summary.get("left_semantic_sha256") or ""),
        right_semantic_sha256=str(summary.get("right_semantic_sha256") or ""),
        category_counts={
            category: int(counts.get(category, 0))
            for category in DISCREPANCY_CATEGORIES
        },
    )
    if dict(summary) != rebuilt:
        raise WorkflowStrictComparisonConflict(
            "Existing comparison summary does not reconcile."
        )

    events = session.execute(
        select(WorkflowEvent).where(
            WorkflowEvent.related_comparison_id == comparison.id,
            WorkflowEvent.event_type.in_((
                "strict_comparison_started",
                "strict_comparison_completed",
            )),
        )
    ).scalars().all()
    if sorted(event.event_type for event in events) != [
        "strict_comparison_completed",
        "strict_comparison_started",
    ]:
        raise WorkflowStrictComparisonConflict(
            "Existing comparison audit-event pair is incomplete."
        )

    return {
        "success": True,
        "contract": WORKFLOW_STRICT_COMPARISON_CONTRACT,
        "task_id": str(item.id),
        "comparison_id": str(comparison.id),
        "left_pass_id": str(dl1.id),
        "right_pass_id": str(dl2.id),
        "comparison_version": comparison.comparison_version,
        "status": comparison.status,
        "strict_equality_passed": comparison.strict_equality_passed,
        "difference_count": comparison.difference_count,
        "difference_summary": dict(comparison.difference_summary),
        "current_stage": item.current_stage,
        "stage_condition": item.stage_condition,
        "row_version": item.row_version,
        "already_compared": True,
        "committed": False,
    }


def execute_strict_workflow_comparison(
    session: Session,
    item_id: UUID | str,
    *,
    expected_row_version: int,
    service_version: str,
    normalized_artifact_loader: ArtifactLoader,
    comparison_payload_adapter: ComparisonPayloadAdapter,
    now: datetime | None = None,
) -> dict[str, object]:
    # Service-only comparison executor. Caller owns commit/rollback.
    normalized_item = _uuid(item_id, name="item_id")
    expected = _expected_version(expected_row_version)
    checker_version = _service_version(service_version)
    timestamp = _utc(now)

    item = session.execute(
        select(WorkflowItem)
        .where(WorkflowItem.id == normalized_item)
        .with_for_update()
    ).scalar_one_or_none()
    if item is None:
        raise WorkflowStrictComparisonConflict(
            "Workflow item was not found."
        )
    if int(item.row_version) != expected:
        raise WorkflowStrictComparisonConflict(
            "Workflow row_version changed before strict comparison."
        )

    dl1, dl2 = _load_current_submitted_pair(session, item)

    existing = session.execute(
        select(WorkflowComparison)
        .where(WorkflowComparison.workflow_item_id == item.id)
        .with_for_update()
    ).scalars().all()
    if existing:
        if len(existing) != 1:
            raise WorkflowStrictComparisonConflict(
                "Existing comparison set is not exact."
            )
        return _validate_existing_replay(
            session,
            item,
            dl1,
            dl2,
            existing[0],
        )

    if (
        item.lifecycle_state,
        item.current_stage,
        item.stage_condition,
    ) != ("active", "independent_acquisition", "ready"):
        raise WorkflowStrictComparisonConflict(
            "Workflow item is not ready for strict comparison."
        )
    assert_forward_stage_transition(
        "independent_acquisition",
        "strict_comparison",
    )

    try:
        left_frozen, left_payload = _load_verified_payload(
            session,
            item,
            dl1,
            artifact_loader=normalized_artifact_loader,
            comparison_payload_adapter=comparison_payload_adapter,
        )
        right_frozen, right_payload = _load_verified_payload(
            session,
            item,
            dl2,
            artifact_loader=normalized_artifact_loader,
            comparison_payload_adapter=comparison_payload_adapter,
        )
    except WorkflowStagingBindingError as exc:
        raise WorkflowStrictComparisonConflict(str(exc)) from exc

    comparison = WorkflowComparison(
        workflow_item_id=item.id,
        left_pass_id=dl1.id,
        right_pass_id=dl2.id,
        comparison_version=WORKFLOW_COMPARISON_VERSION,
        status="pending",
        strict_equality_passed=None,
        difference_count=None,
        difference_summary=None,
        checked_at=None,
        checked_by_service_version=None,
        reviewed_by_principal=None,
        reviewed_at=None,
        created_at=timestamp,
    )
    session.add(comparison)
    session.flush()

    start_prior = {
        "lifecycle_state": item.lifecycle_state,
        "current_stage": item.current_stage,
        "stage_condition": item.stage_condition,
        "row_version": item.row_version,
    }
    item.current_stage = "strict_comparison"
    item.stage_condition = "in_progress"
    item.row_version = expected + 1
    item.updated_at = timestamp

    start_event = WorkflowEvent(
        workflow_item_id=item.id,
        actor_type="service",
        actor_principal=None,
        actor_service=WORKFLOW_STRICT_COMPARISON_SERVICE,
        event_type="strict_comparison_started",
        stage="strict_comparison",
        prior_state=start_prior,
        new_state={
            "lifecycle_state": item.lifecycle_state,
            "current_stage": item.current_stage,
            "stage_condition": item.stage_condition,
            "row_version": item.row_version,
        },
        related_pass_id=None,
        related_comparison_id=comparison.id,
        related_review_id=None,
        related_staging_batch_id=None,
        related_canonical_race_id=item.canonical_race_id,
        reason_code=None,
        summary="Strict comparison service started immutable DL1/DL2 comparison.",
        event_metadata={
            "contract": WORKFLOW_STRICT_COMPARISON_CONTRACT,
            "left_pass_id": str(dl1.id),
            "right_pass_id": str(dl2.id),
            "comparison_version": WORKFLOW_COMPARISON_VERSION,
            "left_artifact_sha256": left_frozen["artifact_sha256"],
            "right_artifact_sha256": right_frozen["artifact_sha256"],
        },
        occurred_at=timestamp,
    )
    session.add(start_event)
    session.flush()

    differences = enumerate_semantic_differences(
        left_payload,
        right_payload,
    )
    strict = strict_semantic_equality(left_payload, right_payload)
    if strict != (len(differences) == 0):
        raise WorkflowStrictComparisonError(
            "Difference enumerator disagrees with strict semantic equality."
        )

    counts = _category_counts(differences)
    summary = build_difference_summary(
        left_semantic_sha256=str(left_payload["semantic_sha256"]),
        right_semantic_sha256=str(right_payload["semantic_sha256"]),
        category_counts=counts,
    )
    if summary["difference_count"] != len(differences):
        raise WorkflowStrictComparisonError(
            "Difference summary count does not match enumerated differences."
        )

    for difference in differences:
        session.add(
            WorkflowDiscrepancy(
                comparison_id=comparison.id,
                workflow_item_id=item.id,
                category=difference["category"],
                semantic_key=difference["semantic_key"],
                left_value=difference["left_value"],
                right_value=difference["right_value"],
                left_value_state=difference["left_value_state"],
                right_value_state=difference["right_value_state"],
                severity=None,
                resolution_status="open",
                resolution_code=None,
                resolution_notes=None,
                resolved_by_principal=None,
                resolved_at=None,
                created_at=timestamp,
            )
        )

    comparison.status = "complete"
    comparison.strict_equality_passed = strict
    comparison.difference_count = len(differences)
    comparison.difference_summary = summary
    comparison.checked_at = timestamp
    comparison.checked_by_service_version = checker_version

    next_stage = next_stage_after_comparison(strict)
    assert_forward_stage_transition("strict_comparison", next_stage)
    complete_prior = {
        "lifecycle_state": item.lifecycle_state,
        "current_stage": item.current_stage,
        "stage_condition": item.stage_condition,
        "row_version": item.row_version,
    }
    item.current_stage = next_stage
    item.stage_condition = "pending"
    item.updated_at = timestamp

    complete_event = WorkflowEvent(
        workflow_item_id=item.id,
        actor_type="service",
        actor_principal=None,
        actor_service=WORKFLOW_STRICT_COMPARISON_SERVICE,
        event_type="strict_comparison_completed",
        stage="strict_comparison",
        prior_state=complete_prior,
        new_state={
            "lifecycle_state": item.lifecycle_state,
            "current_stage": item.current_stage,
            "stage_condition": item.stage_condition,
            "row_version": item.row_version,
        },
        related_pass_id=None,
        related_comparison_id=comparison.id,
        related_review_id=None,
        related_staging_batch_id=None,
        related_canonical_race_id=item.canonical_race_id,
        reason_code=None,
        summary=(
            "Strict comparison matched; item advanced to QC1."
            if strict
            else "Strict comparison found discrepancies; item routed for resolution."
        ),
        event_metadata={
            "contract": WORKFLOW_STRICT_COMPARISON_CONTRACT,
            "left_pass_id": str(dl1.id),
            "right_pass_id": str(dl2.id),
            "left_semantic_sha256": left_payload["semantic_sha256"],
            "right_semantic_sha256": right_payload["semantic_sha256"],
            "strict_equality_passed": strict,
            "difference_count": len(differences),
            "category_counts": counts,
            "next_stage": next_stage,
        },
        occurred_at=timestamp,
    )
    session.add(complete_event)
    session.flush()

    return {
        "success": True,
        "contract": WORKFLOW_STRICT_COMPARISON_CONTRACT,
        "task_id": str(item.id),
        "comparison_id": str(comparison.id),
        "left_pass_id": str(dl1.id),
        "right_pass_id": str(dl2.id),
        "comparison_version": comparison.comparison_version,
        "status": comparison.status,
        "strict_equality_passed": strict,
        "difference_count": len(differences),
        "difference_summary": summary,
        "current_stage": item.current_stage,
        "stage_condition": item.stage_condition,
        "row_version": item.row_version,
        "already_compared": False,
        "committed": False,
    }
