from __future__ import annotations

from pathlib import Path

import pytest

from webapp.parser.auth.workflow_runtime_authorization import (
    ROLE_PRINCIPAL_ENV,
    WORKFLOW_RUNTIME_AUTHORIZATION_SEAM,
    WorkflowRuntimeAuthorizationDenied,
    assert_workflow_runtime_capability,
    resolve_workflow_roles_for_principal,
)
from webapp.parser.contracts.workflow_authorization import (
    CAP_AUDIT_READ,
    CAP_DL1_CLAIM,
    CAP_PUBLICATION_HANDOFF,
    CAP_QC1_REVIEW,
    CAP_SOURCE_READ,
    HUMAN_ROLES,
    ROLE_AUDITOR,
    ROLE_CONTRIBUTOR,
    ROLE_PUBLICATION_OPERATOR,
    ROLE_REVIEWER,
)

MODULE_PATH = Path("webapp/parser/auth/workflow_runtime_authorization.py")


def test_exact_runtime_seam_and_server_owned_role_binding_names():
    assert WORKFLOW_RUNTIME_AUTHORIZATION_SEAM == "workflow_runtime_authorization_seam_v1"
    assert ROLE_PRINCIPAL_ENV == {
        ROLE_CONTRIBUTOR: "WORKFLOW_CONTRIBUTOR_PRINCIPALS",
        ROLE_REVIEWER: "WORKFLOW_REVIEWER_PRINCIPALS",
        ROLE_PUBLICATION_OPERATOR: "WORKFLOW_PUBLICATION_OPERATOR_PRINCIPALS",
        ROLE_AUDITOR: "WORKFLOW_AUDITOR_PRINCIPALS",
    }
    assert set(ROLE_PRINCIPAL_ENV) == set(HUMAN_ROLES)


def test_empty_and_unlisted_principals_resolve_no_roles():
    env = {"WORKFLOW_CONTRIBUTOR_PRINCIPALS": "cert:alpha"}
    assert resolve_workflow_roles_for_principal(None, environ=env) == set()
    assert resolve_workflow_roles_for_principal("", environ=env) == set()
    assert resolve_workflow_roles_for_principal("cert:unlisted", environ=env) == set()


def test_principal_matching_is_exact_not_substring_or_case_folded():
    env = {"WORKFLOW_CONTRIBUTOR_PRINCIPALS": "cert:abcdef, sso:OID-123"}
    assert resolve_workflow_roles_for_principal("cert:abcdef", environ=env) == {ROLE_CONTRIBUTOR}
    assert resolve_workflow_roles_for_principal("cert:abc", environ=env) == set()
    assert resolve_workflow_roles_for_principal("CERT:ABCDEF", environ=env) == set()


def test_multiple_human_roles_may_be_bound_to_same_principal():
    env = {
        "WORKFLOW_CONTRIBUTOR_PRINCIPALS": "sso:one",
        "WORKFLOW_REVIEWER_PRINCIPALS": "sso:one",
    }
    assert resolve_workflow_roles_for_principal("sso:one", environ=env) == {
        ROLE_CONTRIBUTOR,
        ROLE_REVIEWER,
    }


def test_contributor_capabilities_are_derived_from_w2_contract():
    env = {"WORKFLOW_CONTRIBUTOR_PRINCIPALS": "cert:contributor"}
    assert assert_workflow_runtime_capability(
        "cert:contributor", CAP_SOURCE_READ, environ=env
    ) == {ROLE_CONTRIBUTOR}
    assert assert_workflow_runtime_capability(
        "cert:contributor", CAP_DL1_CLAIM, environ=env
    ) == {ROLE_CONTRIBUTOR}


def test_role_capability_boundaries_fail_closed():
    env = {
        "WORKFLOW_REVIEWER_PRINCIPALS": "cert:reviewer",
        "WORKFLOW_PUBLICATION_OPERATOR_PRINCIPALS": "cert:publisher",
        "WORKFLOW_AUDITOR_PRINCIPALS": "cert:auditor",
    }
    assert_workflow_runtime_capability("cert:reviewer", CAP_QC1_REVIEW, environ=env)
    assert_workflow_runtime_capability(
        "cert:publisher", CAP_PUBLICATION_HANDOFF, environ=env
    )
    assert_workflow_runtime_capability("cert:auditor", CAP_AUDIT_READ, environ=env)
    with pytest.raises(
        WorkflowRuntimeAuthorizationDenied,
        match=r"^Workflow capability denied\.$",
    ):
        assert_workflow_runtime_capability(
            "cert:reviewer", CAP_DL1_CLAIM, environ=env
        )


def test_unlisted_principal_and_unknown_capability_use_generic_denial():
    with pytest.raises(
        WorkflowRuntimeAuthorizationDenied,
        match=r"^Workflow capability denied\.$",
    ):
        assert_workflow_runtime_capability(
            "cert:unlisted", CAP_SOURCE_READ, environ={}
        )
    with pytest.raises(
        WorkflowRuntimeAuthorizationDenied,
        match=r"^Workflow capability denied\.$",
    ):
        assert_workflow_runtime_capability(
            "cert:unlisted", "workflow.unknown.capability", environ={}
        )


def test_service_role_cannot_be_obtained_from_human_principal_env():
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "WORKFLOW_COMPARISON_SERVICE_PRINCIPALS" not in source
    assert "ROLE_COMPARISON_SERVICE" not in source


def test_runtime_seam_has_no_request_tier_keycloak_or_db_authority():
    source = MODULE_PATH.read_text(encoding="utf-8")
    for forbidden in (
        "PrivilegeTier",
        "get_principal_tier",
        "request.",
        "flask",
        "sqlalchemy",
        "SessionLocal",
        "KeycloakOpenID",
        "map_external_roles",
        "KEYCLOAK_ROLE_TO_INTERNAL_ROLE",
    ):
        assert forbidden not in source
    assert "assert_capability" in source
    assert "os.environ" in source
