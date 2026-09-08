from __future__ import annotations

import pytest

from webapp.parser.contracts.workflow_lifecycle import (
    CANONICAL_PUBLICATION_POLICY,
    COMPARISON_STATUSES,
    DEFERRED_DECISIONS,
    DISCREPANCY_RESOLUTION_STATUSES,
    FORWARD_STAGE_TRANSITIONS,
    ITEM_LIFECYCLE_STATES,
    ITEM_STAGES,
    MISSING_SCOPE_POLICY,
    PASS_STATUSES,
    PRE_QC_POLICY,
    REVIEW_DECISIONS,
    REVIEW_STAGES,
    STAGE_CONDITIONS,
    WORKFLOW_LIFECYCLE_CONTRACT,
    assert_dl2_claimable,
    assert_forward_stage_transition,
    assert_qc_reviewer_separation,
    next_pass_revision,
    next_stage_after_comparison,
)


def test_exact_vocabulary():
    assert WORKFLOW_LIFECYCLE_CONTRACT == "workflow_lifecycle_contract_v1"
    assert ITEM_LIFECYCLE_STATES == {
        "queued", "active", "blocked", "ready_for_publication", "published",
    }
    assert ITEM_STAGES == {
        "source_intake",
        "independent_acquisition",
        "strict_comparison",
        "discrepancy_resolution",
        "qc1_review",
        "qc2_review",
        "publication_handoff",
    }
    assert STAGE_CONDITIONS == {
        "pending", "in_progress", "awaiting_dependency", "ready", "complete", "failed",
    }
    assert PASS_STATUSES == {"pending", "in_progress", "submitted", "superseded"}
    assert COMPARISON_STATUSES == {"pending", "complete", "superseded"}
    assert DISCREPANCY_RESOLUTION_STATUSES == {"open", "resolved", "superseded"}
    assert REVIEW_STAGES == {"qc1", "qc2"}
    assert REVIEW_DECISIONS == {"approved", "returned", "rejected"}


def test_preqc_scope_and_canonical_boundaries():
    assert PRE_QC_POLICY == "PASS_SUBMISSION_VALIDATION_GATE_NOT_ITEM_STAGE"
    assert "pre_qc" not in ITEM_STAGES
    assert "preqc" not in ITEM_STAGES
    assert MISSING_SCOPE_POLICY == "PRESERVE_NULL_NO_JURISDICTION_INFERENCE"
    assert CANONICAL_PUBLICATION_POLICY == (
        "WORKFLOW_READY_FOR_PUBLICATION_IS_NONCANONICAL_"
        "CANONICAL_WRITER_REQUIRED_BEFORE_PUBLISHED_LINKAGE"
    )


def test_forward_stage_graph():
    assert FORWARD_STAGE_TRANSITIONS["strict_comparison"] == {
        "discrepancy_resolution", "qc1_review",
    }
    assert_forward_stage_transition("source_intake", "independent_acquisition")
    assert_forward_stage_transition("strict_comparison", "qc1_review")
    assert_forward_stage_transition("strict_comparison", "discrepancy_resolution")
    with pytest.raises(ValueError):
        assert_forward_stage_transition("source_intake", "qc1_review")


def test_dl2_claim_gate():
    assert_dl2_claimable(
        dl1_status="submitted",
        candidate_check_complete=True,
        semantic_validation_complete=True,
        dl1_principal="principal:dl1",
        dl2_principal="principal:dl2",
    )
    variants = [
        {"dl1_status": "in_progress"},
        {"candidate_check_complete": False},
        {"semantic_validation_complete": False},
        {"dl2_principal": "principal:dl1"},
    ]
    for variant in variants:
        params = {
            "dl1_status": "submitted",
            "candidate_check_complete": True,
            "semantic_validation_complete": True,
            "dl1_principal": "principal:dl1",
            "dl2_principal": "principal:dl2",
        }
        params.update(variant)
        with pytest.raises(ValueError):
            assert_dl2_claimable(**params)


def test_immutable_revision_and_comparison_branch():
    assert next_pass_revision(1) == 2
    assert next_pass_revision(9) == 10
    with pytest.raises(ValueError):
        next_pass_revision(0)
    assert next_stage_after_comparison(True) == "qc1_review"
    assert next_stage_after_comparison(False) == "discrepancy_resolution"


def test_qc2_reviewer_separation():
    assert_qc_reviewer_separation("principal:qc1", "principal:qc2")
    with pytest.raises(ValueError):
        assert_qc_reviewer_separation("principal:qc", "principal:qc")


def test_one_decision_remains_deferred_after_w4a():
    assert DEFERRED_DECISIONS == (
        "exact canonical writer callback/result contract used by publication_handoff",
    )
