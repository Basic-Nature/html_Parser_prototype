"""Server-owned Pre-QC validation for one governed Workflow pass.

This service validates an already-bound, immutable normalized artifact. It does
not accept client validation booleans and does not mutate failed validations.

The W4 normalized semantic contract remains authoritative for:
- deterministic names/methods,
- all candidate/method entries,
- unique semantic keys,
- canonical ordering,
- semantic SHA-256,
- explicit value/null/missing states.

This module adds pass binding/scope reconciliation and arithmetic integrity.
Unknown/null/missing values are never coerced to zero.

All writes use the caller-provided SQLAlchemy Session. This service never
commits or rolls back.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from webapp.parser.contracts.workflow_comparison import (
    WORKFLOW_COMPARISON_PAYLOAD_CONTRACT,
    WORKFLOW_COMPARISON_VERSION,
    WorkflowComparisonContractError,
    normalize_text,
    validate_comparison_payload,
)
from webapp.parser.services.workflow_staging_binding import (
    WORKFLOW_STAGING_BINDING_CONTRACT,
    WorkflowStagingBindingError,
    validate_completed_workflow_staging_binding,
)
from webapp.parser.utils.models import (
    WorkflowEvent,
    WorkflowItem,
    WorkflowPass,
)


WORKFLOW_PRE_QC_VALIDATION_CONTRACT = "workflow_pre_qc_pass_validation_v1"
WORKFLOW_PRE_QC_VALIDATION_SERVICE = "workflow_pre_qc_validation"
_SUPPORTED_GOVERNED_PASS_PAIRS = frozenset({
    (1, "DL1"),
    (2, "DL2"),
})


class WorkflowPreQCValidationError(RuntimeError):
    status_code = 422
    code = "workflow_pre_qc_validation_error"


class WorkflowPreQCValidationConflict(WorkflowPreQCValidationError):
    status_code = 409
    code = "workflow_pre_qc_validation_conflict"


def _uuid(value: UUID | str, *, name: str) -> UUID:
    try:
        return value if isinstance(value, UUID) else UUID(str(value))
    except (TypeError, ValueError) as exc:
        raise WorkflowPreQCValidationError(
            f"{name} must be a UUID."
        ) from exc


def _actor(principal: str) -> str:
    actor = str(principal or "").strip()
    if not actor:
        raise WorkflowPreQCValidationError(
            "Authenticated internal principal is required."
        )
    return actor


def _utc(now: datetime | None) -> datetime:
    value = now or datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _normalized_nullable(value: object) -> str | None:
    if value is None:
        return None
    return normalize_text(value)


def _expected_scope(item: WorkflowItem) -> dict[str, object]:
    election_date = item.election_date
    return {
        "election_year": item.election_year,
        "election_date": (
            election_date.isoformat()
            if election_date is not None
            else None
        ),
        "state": _normalized_nullable(item.state),
        "jurisdiction_name": _normalized_nullable(item.jurisdiction_name),
        "jurisdiction_type": _normalized_nullable(item.jurisdiction_type),
        "contest": _normalized_nullable(item.contest),
    }


def _value_votes(entry: Mapping[str, object]) -> int | None:
    if entry.get("state") != "value":
        return None
    votes = entry.get("votes")
    if isinstance(votes, bool) or not isinstance(votes, int) or votes < 0:
        raise WorkflowPreQCValidationError(
            "Validated W4 vote state contains invalid numeric votes."
        )
    return votes


def _reconcile_total(
    *,
    label: str,
    components: list[Mapping[str, object]],
    total: Mapping[str, object],
) -> dict[str, object]:
    known_values = [
        votes
        for component in components
        if (votes := _value_votes(component)) is not None
    ]
    known_sum = sum(known_values)
    all_components_value = len(known_values) == len(components)
    total_votes = _value_votes(total)

    if total_votes is not None:
        if all_components_value and total_votes != known_sum:
            raise WorkflowPreQCValidationError(
                f"{label} total must equal the sum of all known components."
            )
        if not all_components_value and total_votes < known_sum:
            raise WorkflowPreQCValidationError(
                f"{label} total cannot be less than the known component subtotal."
            )

    return {
        "known_component_sum": known_sum,
        "all_components_value": all_components_value,
        "total_state": total.get("state"),
        "total_votes": total_votes,
    }


def _candidate_accounting(
    semantic: Mapping[str, object],
) -> dict[str, object]:
    records = semantic.get("records")
    if not isinstance(records, list) or not records:
        raise WorkflowPreQCValidationError(
            "Pre-QC requires at least one normalized reporting-unit record."
        )

    candidate_count = 0
    method_count = 0
    record_results: list[dict[str, object]] = []

    for record_index, record_raw in enumerate(records):
        if not isinstance(record_raw, Mapping):
            raise WorkflowPreQCValidationError(
                "Validated W4 record is not an object."
            )
        candidates = record_raw.get("candidates")
        vote_methods = record_raw.get("vote_methods")
        method_totals = record_raw.get("method_totals")
        grand_total = record_raw.get("grand_total")

        if not isinstance(candidates, list) or not candidates:
            raise WorkflowPreQCValidationError(
                f"record[{record_index}] must contain at least one candidate."
            )
        if not isinstance(vote_methods, list) or not vote_methods:
            raise WorkflowPreQCValidationError(
                f"record[{record_index}] must contain vote methods."
            )
        if not isinstance(method_totals, list):
            raise WorkflowPreQCValidationError(
                f"record[{record_index}] method_totals must be a list."
            )
        if not isinstance(grand_total, Mapping):
            raise WorkflowPreQCValidationError(
                f"record[{record_index}] grand_total must be an object."
            )

        candidate_count += len(candidates)
        method_count += len(vote_methods)

        candidate_total_states: list[Mapping[str, object]] = []
        candidate_checks: list[dict[str, object]] = []

        for candidate_index, candidate_raw in enumerate(candidates):
            if not isinstance(candidate_raw, Mapping):
                raise WorkflowPreQCValidationError(
                    "Validated W4 candidate is not an object."
                )
            method_votes = candidate_raw.get("method_votes")
            total_votes = candidate_raw.get("total_votes")
            if not isinstance(method_votes, list) or not isinstance(
                total_votes,
                Mapping,
            ):
                raise WorkflowPreQCValidationError(
                    "Validated W4 candidate vote shape is invalid."
                )

            candidate_checks.append(
                _reconcile_total(
                    label=(
                        f"record[{record_index}]."
                        f"candidate[{candidate_index}]"
                    ),
                    components=method_votes,
                    total=total_votes,
                )
            )
            candidate_total_states.append(total_votes)

        # Each reported method total reconciles against the candidate values for
        # that exact method. W4 has already guaranteed identical method order.
        method_checks: list[dict[str, object]] = []
        for method_index, method_total_raw in enumerate(method_totals):
            if not isinstance(method_total_raw, Mapping):
                raise WorkflowPreQCValidationError(
                    "Validated W4 method total is not an object."
                )
            candidate_method_values = []
            for candidate_raw in candidates:
                assert isinstance(candidate_raw, Mapping)
                method_votes = candidate_raw["method_votes"]
                assert isinstance(method_votes, list)
                candidate_method = method_votes[method_index]
                if not isinstance(candidate_method, Mapping):
                    raise WorkflowPreQCValidationError(
                        "Validated W4 candidate method value is invalid."
                    )
                candidate_method_values.append(candidate_method)

            method_checks.append(
                _reconcile_total(
                    label=(
                        f"record[{record_index}]."
                        f"method[{method_index}]"
                    ),
                    components=candidate_method_values,
                    total=method_total_raw,
                )
            )

        candidate_grand_check = _reconcile_total(
            label=f"record[{record_index}].grand_total_by_candidates",
            components=candidate_total_states,
            total=grand_total,
        )
        method_grand_check = _reconcile_total(
            label=f"record[{record_index}].grand_total_by_methods",
            components=[
                value
                for value in method_totals
                if isinstance(value, Mapping)
            ],
            total=grand_total,
        )

        record_results.append({
            "candidate_count": len(candidates),
            "vote_method_count": len(vote_methods),
            "candidate_checks": candidate_checks,
            "method_checks": method_checks,
            "grand_total_by_candidates": candidate_grand_check,
            "grand_total_by_methods": method_grand_check,
        })

    return {
        "record_count": len(records),
        "candidate_count": candidate_count,
        "vote_method_occurrences": method_count,
        "records": record_results,
    }


def _validate_workflow_pass_pre_qc(
    session: Session,
    item_id: UUID | str,
    pass_id: UUID | str,
    staging_batch_id: UUID | str,
    *,
    required_pair: tuple[int, str] | None,
    principal: str,
    normalized_payload: object,
    now: datetime | None = None,
) -> dict[str, object]:
    """Validate one server-loaded governed acquisition pass atomically."""

    actor = _actor(principal)
    normalized_item = _uuid(item_id, name="item_id")
    normalized_pass = _uuid(pass_id, name="pass_id")
    normalized_batch = _uuid(staging_batch_id, name="staging_batch_id")
    timestamp = _utc(now)

    item = session.execute(
        select(WorkflowItem)
        .where(WorkflowItem.id == normalized_item)
        .with_for_update()
    ).scalar_one_or_none()
    workflow_pass = session.execute(
        select(WorkflowPass)
        .where(WorkflowPass.id == normalized_pass)
        .with_for_update()
    ).scalar_one_or_none()
    if item is None or workflow_pass is None:
        raise WorkflowPreQCValidationConflict(
            "Workflow item/pass was not found."
        )
    if (
        item.lifecycle_state,
        item.current_stage,
        item.stage_condition,
    ) != ("active", "independent_acquisition", "in_progress"):
        raise WorkflowPreQCValidationConflict(
            "Workflow item is not in active independent acquisition."
        )
    pass_pair = (
        workflow_pass.pass_number,
        workflow_pass.pass_label,
    )
    if (
        workflow_pass.workflow_item_id != item.id
        or pass_pair not in _SUPPORTED_GOVERNED_PASS_PAIRS
        or (
            required_pair is not None
            and pass_pair != required_pair
        )
        or workflow_pass.is_current is not True
        or workflow_pass.status != "in_progress"
        or str(workflow_pass.assigned_principal or "").strip() != actor
    ):
        raise WorkflowPreQCValidationConflict(
            "Pre-QC requires the current in-progress governed pass assignee."
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
        raise WorkflowPreQCValidationConflict(str(exc)) from exc

    try:
        payload = validate_comparison_payload(normalized_payload)
    except WorkflowComparisonContractError as exc:
        raise WorkflowPreQCValidationError(
            f"Normalized semantic contract failed: {exc}"
        ) from exc

    payload_binding = payload["binding"]
    assert isinstance(payload_binding, Mapping)

    expected_binding = {
        "workflow_item_id": str(item.id),
        "workflow_pass_id": str(workflow_pass.id),
        "pass_number": workflow_pass.pass_number,
        "revision_number": workflow_pass.revision_number,
        "source_evidence_ref": binding["source_evidence_ref"],
        "staging_batch_id": str(normalized_batch),
        "normalized_artifact_ref": binding["artifact_ref"],
        "normalized_artifact_sha256": binding["artifact_sha256"],
    }
    if dict(payload_binding) != expected_binding:
        raise WorkflowPreQCValidationConflict(
            "Normalized artifact binding does not exactly match Workflow provenance."
        )

    semantic = payload["semantic"]
    assert isinstance(semantic, Mapping)
    scope = semantic["scope"]
    assert isinstance(scope, Mapping)
    expected_scope = _expected_scope(item)
    if dict(scope) != expected_scope:
        raise WorkflowPreQCValidationConflict(
            "Normalized semantic scope does not exactly match Workflow item scope."
        )

    accounting = _candidate_accounting(semantic)
    semantic_hash = payload["semantic_sha256"]
    assert isinstance(semantic_hash, str)

    candidate_result = {
        "contract": WORKFLOW_PRE_QC_VALIDATION_CONTRACT,
        "status": "complete",
        "semantic_sha256": semantic_hash,
        "normalized_artifact_ref": binding["artifact_ref"],
        "normalized_artifact_sha256": binding["artifact_sha256"],
        "staging_batch_id": str(normalized_batch),
        "accounting": accounting,
        "roster_completeness_claim": False,
        "roster_completeness_reason": (
            "No independent candidate roster authority is available in this "
            "contract; candidate validation covers normalized structural and "
            "arithmetic integrity only."
        ),
    }
    semantic_result = {
        "contract": WORKFLOW_PRE_QC_VALIDATION_CONTRACT,
        "status": "complete",
        "comparison_payload_contract":
            WORKFLOW_COMPARISON_PAYLOAD_CONTRACT,
        "comparison_version": WORKFLOW_COMPARISON_VERSION,
        "semantic_sha256": semantic_hash,
        "normalized_artifact_ref": binding["artifact_ref"],
        "normalized_artifact_sha256": binding["artifact_sha256"],
        "staging_batch_id": str(normalized_batch),
        "null_zero_missing_policy": "preserved_distinct",
        "scope_exact_match": True,
        "binding_exact_match": True,
    }

    existing_complete = (
        workflow_pass.candidate_check_status == "complete"
        and workflow_pass.semantic_validation_status == "complete"
    )
    if existing_complete:
        if (
            workflow_pass.candidate_check_result == candidate_result
            and workflow_pass.semantic_validation_result == semantic_result
        ):
            return {
                "success": True,
                "contract": WORKFLOW_PRE_QC_VALIDATION_CONTRACT,
                "task_id": str(item.id),
                "pass_id": str(workflow_pass.id),
                "staging_batch_id": str(normalized_batch),
                "semantic_sha256": semantic_hash,
                "already_validated": True,
                "committed": False,
            }
        raise WorkflowPreQCValidationConflict(
            "Pass already contains different completed Pre-QC evidence."
        )

    if (
        workflow_pass.candidate_check_status is not None
        or workflow_pass.candidate_check_result is not None
        or workflow_pass.semantic_validation_status is not None
        or workflow_pass.semantic_validation_result is not None
    ):
        raise WorkflowPreQCValidationConflict(
            "Pass contains noncanonical partial Pre-QC state."
        )

    workflow_pass.candidate_check_status = "complete"
    workflow_pass.candidate_check_result = candidate_result
    workflow_pass.semantic_validation_status = "complete"
    workflow_pass.semantic_validation_result = semantic_result
    workflow_pass.updated_at = timestamp

    event = WorkflowEvent(
        workflow_item_id=item.id,
        actor_type="service",
        actor_principal=actor,
        actor_service=WORKFLOW_PRE_QC_VALIDATION_SERVICE,
        event_type="pre_qc_pass_validated",
        stage="independent_acquisition",
        prior_state={
            "pass_id": str(workflow_pass.id),
            "candidate_check_status": None,
            "semantic_validation_status": None,
        },
        new_state={
            "pass_id": str(workflow_pass.id),
            "candidate_check_status": "complete",
            "semantic_validation_status": "complete",
            "semantic_sha256": semantic_hash,
        },
        related_pass_id=workflow_pass.id,
        related_comparison_id=None,
        related_review_id=None,
        related_staging_batch_id=normalized_batch,
        related_canonical_race_id=item.canonical_race_id,
        reason_code=None,
        summary=(f"Server-owned {workflow_pass.pass_label} "
                 "Pre-QC validation completed."),
        event_metadata={
            "contract": WORKFLOW_PRE_QC_VALIDATION_CONTRACT,
            "pass_number": workflow_pass.pass_number,
            "pass_label": workflow_pass.pass_label,
            "staging_binding_contract":
                WORKFLOW_STAGING_BINDING_CONTRACT,
            "comparison_payload_contract":
                WORKFLOW_COMPARISON_PAYLOAD_CONTRACT,
            "semantic_sha256": semantic_hash,
        },
        occurred_at=timestamp,
    )
    session.add(event)
    session.flush()

    return {
        "success": True,
        "contract": WORKFLOW_PRE_QC_VALIDATION_CONTRACT,
        "task_id": str(item.id),
        "pass_id": str(workflow_pass.id),
        "staging_batch_id": str(normalized_batch),
        "semantic_sha256": semantic_hash,
        "already_validated": False,
        "event_id": str(event.id),
        "candidate_check_status": "complete",
        "semantic_validation_status": "complete",
        "committed": False,
    }


def validate_workflow_pass_pre_qc(
    session: Session,
    item_id: UUID | str,
    pass_id: UUID | str,
    staging_batch_id: UUID | str,
    *,
    principal: str,
    normalized_payload: object,
    now: datetime | None = None,
) -> dict[str, object]:
    """Validate a server-loaded governed acquisition pass."""
    return _validate_workflow_pass_pre_qc(
        session,
        item_id,
        pass_id,
        staging_batch_id,
        required_pair=None,
        principal=principal,
        normalized_payload=normalized_payload,
        now=now,
    )


def validate_first_workflow_pass_pre_qc(
    session: Session,
    item_id: UUID | str,
    pass_id: UUID | str,
    staging_batch_id: UUID | str,
    *,
    principal: str,
    normalized_payload: object,
    now: datetime | None = None,
) -> dict[str, object]:
    """Backward-compatible DL1-only Pre-QC entry point."""
    return _validate_workflow_pass_pre_qc(
        session,
        item_id,
        pass_id,
        staging_batch_id,
        required_pair=(1, "DL1"),
        principal=principal,
        normalized_payload=normalized_payload,
        now=now,
    )


def validate_second_workflow_pass_pre_qc(
    session: Session,
    item_id: UUID | str,
    pass_id: UUID | str,
    staging_batch_id: UUID | str,
    *,
    principal: str,
    normalized_payload: object,
    now: datetime | None = None,
) -> dict[str, object]:
    """DL2-only Pre-QC entry point using the shared pass-aware engine."""
    return _validate_workflow_pass_pre_qc(
        session,
        item_id,
        pass_id,
        staging_batch_id,
        required_pair=(2, "DL2"),
        principal=principal,
        normalized_payload=normalized_payload,
        now=now,
    )


def validate_frozen_workflow_pre_qc_payload(
    workflow_pass: WorkflowPass,
    frozen_binding: Mapping[str, object],
    normalized_payload: object,
) -> dict[str, object]:
    # Read-only submitted-pass Pre-QC reconciliation for comparison service.
    if (
        workflow_pass.is_current is not True
        or workflow_pass.status != "submitted"
        or workflow_pass.submitted_at is None
        or workflow_pass.candidate_check_status != "complete"
        or workflow_pass.semantic_validation_status != "complete"
        or not isinstance(workflow_pass.candidate_check_result, Mapping)
        or not isinstance(workflow_pass.semantic_validation_result, Mapping)
    ):
        raise WorkflowPreQCValidationConflict(
            "Frozen comparison requires a current submitted pass with completed Pre-QC."
        )

    comparison_binding = frozen_binding.get("comparison_binding")
    if not isinstance(comparison_binding, Mapping):
        raise WorkflowPreQCValidationConflict(
            "Frozen comparison binding is missing its W4 binding."
        )

    try:
        payload = validate_comparison_payload(normalized_payload)
    except WorkflowComparisonContractError as exc:
        raise WorkflowPreQCValidationConflict(
            f"Frozen comparison payload failed W4 validation: {exc}"
        ) from exc

    if dict(payload["binding"]) != dict(comparison_binding):
        raise WorkflowPreQCValidationConflict(
            "Frozen W4 payload binding does not match server provenance."
        )

    candidate = workflow_pass.candidate_check_result
    semantic = workflow_pass.semantic_validation_result
    semantic_hash = payload["semantic_sha256"]
    checks = (
        candidate.get("contract") == WORKFLOW_PRE_QC_VALIDATION_CONTRACT,
        semantic.get("contract") == WORKFLOW_PRE_QC_VALIDATION_CONTRACT,
        candidate.get("status") == "complete",
        semantic.get("status") == "complete",
        candidate.get("semantic_sha256") == semantic_hash,
        semantic.get("semantic_sha256") == semantic_hash,
        candidate.get("staging_batch_id")
            == comparison_binding["staging_batch_id"],
        semantic.get("staging_batch_id")
            == comparison_binding["staging_batch_id"],
        candidate.get("normalized_artifact_ref")
            == comparison_binding["normalized_artifact_ref"],
        semantic.get("normalized_artifact_ref")
            == comparison_binding["normalized_artifact_ref"],
        candidate.get("normalized_artifact_sha256")
            == comparison_binding["normalized_artifact_sha256"],
        semantic.get("normalized_artifact_sha256")
            == comparison_binding["normalized_artifact_sha256"],
        candidate.get("roster_completeness_claim") is False,
        semantic.get("comparison_payload_contract")
            == WORKFLOW_COMPARISON_PAYLOAD_CONTRACT,
        semantic.get("comparison_version") == WORKFLOW_COMPARISON_VERSION,
        semantic.get("null_zero_missing_policy") == "preserved_distinct",
        semantic.get("scope_exact_match") is True,
        semantic.get("binding_exact_match") is True,
    )
    if not all(checks):
        raise WorkflowPreQCValidationConflict(
            "Frozen completed Pre-QC evidence does not reconcile to payload."
        )
    return payload

