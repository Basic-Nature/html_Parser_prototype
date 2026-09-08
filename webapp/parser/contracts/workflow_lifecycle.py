"""Pure governed Workflow lifecycle vocabulary and transition contract.

This module is intentionally dependency-free. It freezes accepted vocabulary
and transition invariants before new contributor mutation routes are added.

It does not authorize principals, access the database, define the normalized
comparison payload schema, or call the canonical writer.
"""

from __future__ import annotations

from collections.abc import Collection

WORKFLOW_LIFECYCLE_CONTRACT = "workflow_lifecycle_contract_v1"

ITEM_LIFECYCLE_STATES = frozenset({
    "queued", "active", "blocked", "ready_for_publication", "published",
})
ITEM_STAGES = frozenset({
    "source_intake",
    "independent_acquisition",
    "strict_comparison",
    "discrepancy_resolution",
    "qc1_review",
    "qc2_review",
    "publication_handoff",
})
STAGE_CONDITIONS = frozenset({
    "pending", "in_progress", "awaiting_dependency", "ready", "complete", "failed",
})
PASS_STATUSES = frozenset({"pending", "in_progress", "submitted", "superseded"})
COMPARISON_STATUSES = frozenset({"pending", "complete", "superseded"})
DISCREPANCY_RESOLUTION_STATUSES = frozenset({"open", "resolved", "superseded"})
REVIEW_STAGES = frozenset({"qc1", "qc2"})
REVIEW_DECISIONS = frozenset({"approved", "returned", "rejected"})

PRE_QC_POLICY = "PASS_SUBMISSION_VALIDATION_GATE_NOT_ITEM_STAGE"
MISSING_SCOPE_POLICY = "PRESERVE_NULL_NO_JURISDICTION_INFERENCE"
CANONICAL_PUBLICATION_POLICY = (
    "WORKFLOW_READY_FOR_PUBLICATION_IS_NONCANONICAL_"
    "CANONICAL_WRITER_REQUIRED_BEFORE_PUBLISHED_LINKAGE"
)

DEFERRED_DECISIONS = (
    "exact normalized semantic comparison payload schema and version",
    "exact canonical writer callback/result contract used by publication_handoff",
)

FORWARD_STAGE_TRANSITIONS = {
    "source_intake": frozenset({"independent_acquisition"}),
    "independent_acquisition": frozenset({"strict_comparison"}),
    "strict_comparison": frozenset({"discrepancy_resolution", "qc1_review"}),
    "discrepancy_resolution": frozenset({"qc1_review"}),
    "qc1_review": frozenset({"qc2_review"}),
    "qc2_review": frozenset({"publication_handoff"}),
    "publication_handoff": frozenset(),
}


def _require_nonempty(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def require_choice(name: str, value: str, allowed: Collection[str]) -> str:
    _require_nonempty(name, value)
    if value not in allowed:
        raise ValueError(f"{name} is not an accepted workflow value: {value!r}")
    return value


def assert_forward_stage_transition(prior_stage: str, new_stage: str) -> None:
    require_choice("prior_stage", prior_stage, ITEM_STAGES)
    require_choice("new_stage", new_stage, ITEM_STAGES)
    if new_stage not in FORWARD_STAGE_TRANSITIONS[prior_stage]:
        raise ValueError(
            f"forward stage transition not allowed: {prior_stage!r} -> {new_stage!r}"
        )


def assert_dl2_claimable(
    *,
    dl1_status: str,
    candidate_check_complete: bool,
    semantic_validation_complete: bool,
    dl1_principal: str,
    dl2_principal: str,
) -> None:
    require_choice("dl1_status", dl1_status, PASS_STATUSES)
    _require_nonempty("dl1_principal", dl1_principal)
    _require_nonempty("dl2_principal", dl2_principal)
    if dl1_status != "submitted":
        raise ValueError("DL2 requires the current DL1 revision to be submitted")
    if candidate_check_complete is not True:
        raise ValueError("DL2 requires completed DL1 candidate validation")
    if semantic_validation_complete is not True:
        raise ValueError("DL2 requires completed DL1 semantic validation")
    if dl1_principal == dl2_principal:
        raise ValueError("DL2 principal must differ from DL1 principal")


def next_pass_revision(current_revision: int) -> int:
    if isinstance(current_revision, bool) or not isinstance(current_revision, int):
        raise ValueError("current_revision must be an integer")
    if current_revision < 1:
        raise ValueError("current_revision must be >= 1")
    return current_revision + 1


def next_stage_after_comparison(strict_equality_passed: bool) -> str:
    if not isinstance(strict_equality_passed, bool):
        raise ValueError("strict_equality_passed must be bool")
    return "qc1_review" if strict_equality_passed else "discrepancy_resolution"


def assert_qc_reviewer_separation(qc1_principal: str, qc2_principal: str) -> None:
    _require_nonempty("qc1_principal", qc1_principal)
    _require_nonempty("qc2_principal", qc2_principal)
    if qc1_principal == qc2_principal:
        raise ValueError("QC2 reviewer must differ from QC1 reviewer")
