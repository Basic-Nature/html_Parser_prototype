"""Provider-neutral runtime bridge for governed Workflow authorization.

This module does not authenticate principals. The composition root must first
establish trusted authenticated authority. This bridge then resolves only
server-owned exact principal allowlists into ElectionPulse-internal Workflow
human roles and applies the frozen W2 role/capability contract.

It does not infer Workflow roles from legacy privilege tiers, parse request
JSON, trust external capability claims, activate Keycloak, access the database,
or mutate Workflow state.
"""

from __future__ import annotations

from collections.abc import Mapping
import os

from webapp.parser.contracts.workflow_authorization import (
    HUMAN_ROLES,
    ROLE_AUDITOR,
    ROLE_CONTRIBUTOR,
    ROLE_PUBLICATION_OPERATOR,
    ROLE_REVIEWER,
    WorkflowAuthorizationError,
    assert_capability,
)


WORKFLOW_RUNTIME_AUTHORIZATION_SEAM = "workflow_runtime_authorization_seam_v1"

ROLE_PRINCIPAL_ENV = {
    ROLE_CONTRIBUTOR: "WORKFLOW_CONTRIBUTOR_PRINCIPALS",
    ROLE_REVIEWER: "WORKFLOW_REVIEWER_PRINCIPALS",
    ROLE_PUBLICATION_OPERATOR: "WORKFLOW_PUBLICATION_OPERATOR_PRINCIPALS",
    ROLE_AUDITOR: "WORKFLOW_AUDITOR_PRINCIPALS",
}

if set(ROLE_PRINCIPAL_ENV) != set(HUMAN_ROLES):
    raise RuntimeError(
        "Workflow runtime role bindings must cover exactly the human Workflow roles"
    )


class WorkflowRuntimeAuthorizationDenied(PermissionError):
    """Generic fail-closed runtime Workflow capability denial."""


def _configured_principals(raw: object) -> frozenset[str]:
    if raw is None:
        return frozenset()
    return frozenset(
        token.strip()
        for token in str(raw).split(",")
        if token.strip()
    )


def resolve_workflow_roles_for_principal(
    principal: object,
    *,
    environ: Mapping[str, str] | None = None,
) -> frozenset[str]:
    """Resolve exact server-owned principal allowlists to human roles."""
    normalized_principal = str(principal or "").strip()
    if not normalized_principal:
        return frozenset()

    source = os.environ if environ is None else environ
    roles: set[str] = set()
    for internal_role, env_name in ROLE_PRINCIPAL_ENV.items():
        if normalized_principal in _configured_principals(source.get(env_name)):
            roles.add(internal_role)
    return frozenset(roles)


def assert_workflow_runtime_capability(
    principal: object,
    required_capability: object,
    *,
    environ: Mapping[str, str] | None = None,
) -> frozenset[str]:
    """Require a W2 capability derived only from server-owned role bindings."""
    roles = resolve_workflow_roles_for_principal(
        principal,
        environ=environ,
    )
    try:
        assert_capability(
            roles,
            str(required_capability or "").strip(),
        )
    except WorkflowAuthorizationError as exc:
        raise WorkflowRuntimeAuthorizationDenied(
            "Workflow capability denied."
        ) from exc
    return roles
