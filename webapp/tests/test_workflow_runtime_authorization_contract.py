from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import webapp.parser.auth.workflow_runtime_authorization as runtime_auth
from webapp.parser.auth.workflow_runtime_authorization import (
    AUTHORITY_MODE_DURABLE_TRUSTED_IDENTITY,
    AUTHORITY_MODE_LEGACY_ENV,
    ROLE_PRINCIPAL_ENV,
    VALID_AUTHORITY_MODES,
    WORKFLOW_RUNTIME_AUTHORITY_MODE_ENV,
    WORKFLOW_RUNTIME_AUTHORIZATION_SEAM,
    WorkflowRuntimeAuthorizationDenied,
    _durable_roles_for_principal,
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


def test_exact_runtime_seam_and_modes():
    assert WORKFLOW_RUNTIME_AUTHORIZATION_SEAM == "workflow_runtime_authorization_seam_v1"
    assert WORKFLOW_RUNTIME_AUTHORITY_MODE_ENV == "WORKFLOW_RUNTIME_AUTHORITY_MODE"
    assert AUTHORITY_MODE_LEGACY_ENV == "legacy_env"
    assert AUTHORITY_MODE_DURABLE_TRUSTED_IDENTITY == "durable_trusted_identity"
    assert VALID_AUTHORITY_MODES == {
        AUTHORITY_MODE_LEGACY_ENV,
        AUTHORITY_MODE_DURABLE_TRUSTED_IDENTITY,
    }
    assert set(ROLE_PRINCIPAL_ENV) == set(HUMAN_ROLES)


def test_legacy_mode_remains_default_and_exact():
    env = {"WORKFLOW_CONTRIBUTOR_PRINCIPALS": "cert:abcdef, sso:OID-123"}
    assert resolve_workflow_roles_for_principal("cert:abcdef", environ=env) == {
        ROLE_CONTRIBUTOR
    }
    assert resolve_workflow_roles_for_principal("cert:abc", environ=env) == set()
    assert resolve_workflow_roles_for_principal("CERT:ABCDEF", environ=env) == set()


def test_multiple_human_roles_may_be_bound_to_same_legacy_principal():
    env = {
        "WORKFLOW_CONTRIBUTOR_PRINCIPALS": "sso:one",
        "WORKFLOW_REVIEWER_PRINCIPALS": "sso:one",
    }
    assert resolve_workflow_roles_for_principal("sso:one", environ=env) == {
        ROLE_CONTRIBUTOR,
        ROLE_REVIEWER,
    }


def test_capability_boundaries_remain_frozen():
    env = {
        "WORKFLOW_REVIEWER_PRINCIPALS": "cert:reviewer",
        "WORKFLOW_PUBLICATION_OPERATOR_PRINCIPALS": "cert:publisher",
        "WORKFLOW_AUDITOR_PRINCIPALS": "cert:auditor",
        "WORKFLOW_CONTRIBUTOR_PRINCIPALS": "cert:contributor",
    }
    assert_workflow_runtime_capability("cert:contributor", CAP_SOURCE_READ, environ=env)
    assert_workflow_runtime_capability("cert:reviewer", CAP_QC1_REVIEW, environ=env)
    assert_workflow_runtime_capability("cert:publisher", CAP_PUBLICATION_HANDOFF, environ=env)
    assert_workflow_runtime_capability("cert:auditor", CAP_AUDIT_READ, environ=env)
    with pytest.raises(WorkflowRuntimeAuthorizationDenied):
        assert_workflow_runtime_capability("cert:reviewer", CAP_DL1_CLAIM, environ=env)


def test_durable_helper_reads_existing_roles_and_closes_session():
    class FakeSession:
        closed = False
        def close(self):
            self.closed = True

    fake = FakeSession()
    seen = {}

    def repository_factory(session):
        seen["session"] = session
        return object()

    def resolver(repository, fingerprint):
        seen["repository"] = repository
        seen["fingerprint"] = fingerprint
        return SimpleNamespace(
            resolved=True,
            protected_operation_eligible=True,
            role_names=(ROLE_CONTRIBUTOR, ROLE_REVIEWER),
        )

    roles = _durable_roles_for_principal(
        "cert:" + ("a" * 64),
        session_factory=lambda: fake,
        repository_factory=repository_factory,
        resolver=resolver,
    )
    assert roles == {ROLE_CONTRIBUTOR, ROLE_REVIEWER}
    assert seen["session"] is fake
    assert seen["fingerprint"] == "a" * 64
    assert fake.closed is True


def test_durable_helper_fails_closed_and_closes_session():
    class FakeSession:
        closed = False
        def close(self):
            self.closed = True

    fake = FakeSession()

    def resolver(_repository, _fingerprint):
        raise RuntimeError("simulated outage")

    assert _durable_roles_for_principal(
        "cert:" + ("b" * 64),
        session_factory=lambda: fake,
        repository_factory=lambda _session: object(),
        resolver=resolver,
    ) == set()
    assert fake.closed is True


def test_durable_mode_has_no_legacy_fallback(monkeypatch):
    monkeypatch.setattr(
        runtime_auth,
        "_durable_roles_for_principal",
        lambda principal: (
            frozenset({ROLE_CONTRIBUTOR})
            if principal == "cert:enrolled"
            else frozenset()
        ),
    )
    env = {
        WORKFLOW_RUNTIME_AUTHORITY_MODE_ENV: AUTHORITY_MODE_DURABLE_TRUSTED_IDENTITY,
        "WORKFLOW_REVIEWER_PRINCIPALS": "cert:legacy-only",
    }
    assert resolve_workflow_roles_for_principal("cert:enrolled", environ=env) == {
        ROLE_CONTRIBUTOR
    }
    assert resolve_workflow_roles_for_principal("cert:legacy-only", environ=env) == set()


def test_unknown_mode_fails_closed():
    env = {
        WORKFLOW_RUNTIME_AUTHORITY_MODE_ENV: "unexpected",
        "WORKFLOW_CONTRIBUTOR_PRINCIPALS": "cert:legacy",
    }
    assert resolve_workflow_roles_for_principal("cert:legacy", environ=env) == set()


def test_service_role_cannot_be_obtained_from_human_principal_env():
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "WORKFLOW_COMPARISON_SERVICE_PRINCIPALS" not in source
    assert "ROLE_COMPARISON_SERVICE" not in source


def test_durable_seam_is_read_only_and_provider_bounded():
    source = MODULE_PATH.read_text(encoding="utf-8")
    for forbidden in (
        "PrivilegeTier",
        "get_principal_tier",
        "request.",
        "flask",
        "KeycloakOpenID",
        "map_external_roles",
        "KEYCLOAK_ROLE_TO_INTERNAL_ROLE",
        "db_session.add(",
        "db_session.commit(",
        "db_session.flush(",
        "db_session.delete(",
        "repository.attach_certificate_credential(",
        "repository.add_role_binding(",
        "repository.migrate_fingerprint_candidate(",
        ".attach_certificate_credential(",
        ".add_role_binding(",
        ".migrate_fingerprint_candidate(",
    ):
        assert forbidden not in source
    assert "roles.add(internal_role)" in source
    assert "SessionLocal" in source
    assert "TrustedIdentityRepository" in source
    assert "resolve_enrolled_mtls_principal" in source
    assert "db_session.close()" in source
