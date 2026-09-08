from __future__ import annotations

from pathlib import Path

import pytest

from webapp.parser.contracts.workflow_authorization import (
    CAP_AUDIT_READ,
    CAP_COMPARISON_EXECUTE,
    CAP_DL1_CLAIM,
    CAP_DL1_SUBMIT,
    CAP_DL2_CLAIM,
    CAP_DL2_SUBMIT,
    CAP_DISCREPANCY_RESOLVE,
    CAP_PUBLICATION_HANDOFF,
    CAP_QC1_REVIEW,
    CAP_QC2_REVIEW,
    CAP_SOURCE_READ,
    CAPABILITIES,
    EXTERNAL_CAPABILITY_CLAIMS_ACCEPTED,
    HUMAN_ROLES,
    KEYCLOAK_ENABLED_BY_CONTRACT,
    KEYCLOAK_ROLE_TO_INTERNAL_ROLE,
    LEGACY_PRIVILEGE_TIER_IMPLIES_WORKFLOW_CAPABILITY,
    PUBLIC_WORKFLOW_REQUIRES_KEYCLOAK,
    PUBLICATION_OPERATOR_IS_CANONICAL_WRITER,
    PUBLICATION_OPERATOR_SEPARATION_FIELDS,
    PUBLICATION_OPERATOR_SEPARATION_POLICY,
    REMAINING_DEFERRED_DECISIONS,
    RESOLVED_W2A_DECISIONS,
    RESOLVED_W3A_DECISIONS,
    RESOLVED_W4A_DECISIONS,
    ROLE_AUDITOR,
    ROLE_CAPABILITIES,
    ROLE_COMPARISON_SERVICE,
    ROLE_CONTRIBUTOR,
    ROLE_PUBLICATION_OPERATOR,
    ROLE_REVIEWER,
    ROLES,
    SERVICE_ROLES,
    STRICT_COMPARISON_SERVICE_ONLY,
    WORKFLOW_AUTHORIZATION_CONTRACT,
    WORKFLOW_PUBLICATION_HANDOFF_IS_CANONICAL_WRITE,
    WorkflowAuthorizationError,
    assert_capability,
    assert_four_principal_separation,
    assert_publication_operator_separation,
    capabilities_for_roles,
    map_external_roles,
)
from webapp.parser.contracts.workflow_lifecycle import DEFERRED_DECISIONS


def test_exact_w2a_authorization_vocabulary():
    assert WORKFLOW_AUTHORIZATION_CONTRACT == "workflow_authorization_contract_v1"
    assert CAPABILITIES == {
        CAP_SOURCE_READ,
        CAP_DL1_CLAIM,
        CAP_DL1_SUBMIT,
        CAP_DL2_CLAIM,
        CAP_DL2_SUBMIT,
        CAP_DISCREPANCY_RESOLVE,
        CAP_QC1_REVIEW,
        CAP_QC2_REVIEW,
        CAP_PUBLICATION_HANDOFF,
        CAP_AUDIT_READ,
        CAP_COMPARISON_EXECUTE,
    }
    assert HUMAN_ROLES == {
        ROLE_CONTRIBUTOR,
        ROLE_REVIEWER,
        ROLE_PUBLICATION_OPERATOR,
        ROLE_AUDITOR,
    }
    assert SERVICE_ROLES == {ROLE_COMPARISON_SERVICE}
    assert ROLES == HUMAN_ROLES | SERVICE_ROLES


def test_exact_role_bundles_and_service_only_comparison():
    assert ROLE_CAPABILITIES[ROLE_CONTRIBUTOR] == {
        CAP_SOURCE_READ,
        CAP_DL1_CLAIM,
        CAP_DL1_SUBMIT,
        CAP_DL2_CLAIM,
        CAP_DL2_SUBMIT,
    }
    assert ROLE_CAPABILITIES[ROLE_REVIEWER] == {
        CAP_SOURCE_READ,
        CAP_DISCREPANCY_RESOLVE,
        CAP_QC1_REVIEW,
        CAP_QC2_REVIEW,
        CAP_AUDIT_READ,
    }
    assert ROLE_CAPABILITIES[ROLE_PUBLICATION_OPERATOR] == {
        CAP_PUBLICATION_HANDOFF,
        CAP_AUDIT_READ,
    }
    assert ROLE_CAPABILITIES[ROLE_AUDITOR] == {CAP_AUDIT_READ}
    assert ROLE_CAPABILITIES[ROLE_COMPARISON_SERVICE] == {
        CAP_COMPARISON_EXECUTE,
    }

    human_caps = set()
    for role in HUMAN_ROLES:
        human_caps.update(ROLE_CAPABILITIES[role])
    assert CAP_COMPARISON_EXECUTE not in human_caps
    assert STRICT_COMPARISON_SERVICE_ONLY is True


def test_external_mapping_is_allowlisted_and_never_maps_service_role():
    assert KEYCLOAK_ROLE_TO_INTERNAL_ROLE == {
        "electionpulse-workflow-contributor": ROLE_CONTRIBUTOR,
        "electionpulse-workflow-reviewer": ROLE_REVIEWER,
        "electionpulse-workflow-publication-operator":
            ROLE_PUBLICATION_OPERATOR,
        "electionpulse-workflow-auditor": ROLE_AUDITOR,
    }
    assert map_external_roles([
        "electionpulse-workflow-contributor",
        "realm-admin",
        "unknown-role",
    ]) == {ROLE_CONTRIBUTOR}
    assert ROLE_COMPARISON_SERVICE not in set(
        KEYCLOAK_ROLE_TO_INTERNAL_ROLE.values()
    )
    assert EXTERNAL_CAPABILITY_CLAIMS_ACCEPTED is False


def test_capabilities_are_server_derived_and_unknown_internal_roles_fail_closed():
    assert capabilities_for_roles([ROLE_CONTRIBUTOR]) == {
        CAP_SOURCE_READ,
        CAP_DL1_CLAIM,
        CAP_DL1_SUBMIT,
        CAP_DL2_CLAIM,
        CAP_DL2_SUBMIT,
    }
    assert_capability([ROLE_CONTRIBUTOR], CAP_DL1_CLAIM)
    with pytest.raises(WorkflowAuthorizationError):
        assert_capability([ROLE_AUDITOR], CAP_DL1_CLAIM)
    with pytest.raises(WorkflowAuthorizationError):
        capabilities_for_roles(["root_admin"])
    with pytest.raises(WorkflowAuthorizationError):
        capabilities_for_roles(["unknown-workflow-role"])


def test_four_principal_separation_is_strict_across_dl_and_qc():
    assert_four_principal_separation(
        dl1_principal="usr_dl1",
        dl2_principal="usr_dl2",
        qc1_principal="usr_qc1",
        qc2_principal="usr_qc2",
    )

    fields = {
        "dl1_principal": "usr_dl1",
        "dl2_principal": "usr_dl2",
        "qc1_principal": "usr_qc1",
        "qc2_principal": "usr_qc2",
    }
    for duplicate_field in (
        "dl2_principal",
        "qc1_principal",
        "qc2_principal",
    ):
        invalid = dict(fields)
        invalid[duplicate_field] = "usr_dl1"
        with pytest.raises(WorkflowAuthorizationError):
            assert_four_principal_separation(**invalid)

    with pytest.raises(WorkflowAuthorizationError):
        assert_four_principal_separation(
            dl1_principal="usr_dl1",
            dl2_principal="usr_dl2",
            qc1_principal="usr_same",
            qc2_principal="usr_same",
        )


def test_keycloak_and_legacy_tiers_remain_orthogonal_and_disabled():
    assert KEYCLOAK_ENABLED_BY_CONTRACT is False
    assert PUBLIC_WORKFLOW_REQUIRES_KEYCLOAK is False
    assert LEGACY_PRIVILEGE_TIER_IMPLIES_WORKFLOW_CAPABILITY is False

    source = Path(
        "webapp/parser/contracts/workflow_authorization.py"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "import flask",
        "from flask",
        "import sqlalchemy",
        "from sqlalchemy",
        "import jwt",
        "from jwt",
        "PrivilegeTier",
        "get_principal_tier",
    ):
        assert forbidden not in source


def test_publication_handoff_remains_noncanonical():
    assert WORKFLOW_PUBLICATION_HANDOFF_IS_CANONICAL_WRITE is False
    assert CAP_PUBLICATION_HANDOFF in ROLE_CAPABILITIES[
        ROLE_PUBLICATION_OPERATOR
    ]
    assert CAP_PUBLICATION_HANDOFF not in ROLE_CAPABILITIES[ROLE_CONTRIBUTOR]
    assert CAP_PUBLICATION_HANDOFF not in ROLE_CAPABILITIES[ROLE_REVIEWER]


def test_publication_operator_separation_contract_is_exact_and_fail_closed():
    assert PUBLICATION_OPERATOR_SEPARATION_POLICY == (
        "PUBLICATION_OPERATOR_DISTINCT_FROM_DL1_DL2_QC1_QC2_PER_ITEM"
    )
    assert PUBLICATION_OPERATOR_SEPARATION_FIELDS == (
        "dl1_principal",
        "dl2_principal",
        "qc1_principal",
        "qc2_principal",
    )
    assert PUBLICATION_OPERATOR_IS_CANONICAL_WRITER is False

    valid = {
        "dl1_principal": "usr_dl1",
        "dl2_principal": "usr_dl2",
        "qc1_principal": "usr_qc1",
        "qc2_principal": "usr_qc2",
        "publication_operator_principal": "usr_publication",
    }
    assert_publication_operator_separation(**valid)

    for prior_field in PUBLICATION_OPERATOR_SEPARATION_FIELDS:
        invalid = dict(valid)
        invalid["publication_operator_principal"] = valid[prior_field]
        with pytest.raises(WorkflowAuthorizationError):
            assert_publication_operator_separation(**invalid)

    invalid = dict(valid)
    invalid["publication_operator_principal"] = ""
    with pytest.raises(WorkflowAuthorizationError):
        assert_publication_operator_separation(**invalid)


def test_resolved_and_remaining_decisions_align_lifecycle_contract_after_w4a():
    assert RESOLVED_W2A_DECISIONS == (
        "exact protected contributor role/capability names and Keycloak mapping",
        "whether QC1/QC2 reviewers must also differ from DL1/DL2 principals",
    )
    assert RESOLVED_W3A_DECISIONS == (
        "whether publication operator must differ from all DL/QC principals",
    )
    assert RESOLVED_W4A_DECISIONS == (
        "exact normalized semantic comparison payload schema and version",
    )
    assert REMAINING_DEFERRED_DECISIONS == (
        "exact canonical writer callback/result contract used by publication_handoff",
    )
    assert DEFERRED_DECISIONS == REMAINING_DEFERRED_DECISIONS
