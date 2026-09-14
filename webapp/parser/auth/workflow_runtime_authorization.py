"""Provider-neutral runtime bridge for governed Workflow authorization.

Legacy compatibility mode resolves server-owned exact principal allowlists.
Durable trusted-identity mode resolves the existing cert:<fingerprint>
compatibility principal through canonical trusted-identity state.

Durable mode is read-only and fail-closed: it does not auto-create identity
state and never falls back to legacy allowlists when durable authority is
selected.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
import os
from typing import Any

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
WORKFLOW_RUNTIME_AUTHORITY_MODE_ENV = "WORKFLOW_RUNTIME_AUTHORITY_MODE"
AUTHORITY_MODE_LEGACY_ENV = "legacy_env"
AUTHORITY_MODE_DURABLE_TRUSTED_IDENTITY = "durable_trusted_identity"
VALID_AUTHORITY_MODES = frozenset({
    AUTHORITY_MODE_LEGACY_ENV,
    AUTHORITY_MODE_DURABLE_TRUSTED_IDENTITY,
})

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


def _runtime_authority_mode(environ: Mapping[str, str] | None) -> str | None:
    source = os.environ if environ is None else environ
    raw = str(source.get(WORKFLOW_RUNTIME_AUTHORITY_MODE_ENV, "") or "").strip().lower()
    if not raw:
        return AUTHORITY_MODE_LEGACY_ENV
    if raw in VALID_AUTHORITY_MODES:
        return raw
    return None


def _durable_roles_for_principal(
    principal: object,
    *,
    session_factory: Callable[[], Any] | None = None,
    repository_factory: Callable[[Any], Any] | None = None,
    resolver: Callable[..., Any] | None = None,
) -> frozenset[str]:
    """Resolve canonical roles for an existing cert principal, read-only."""
    normalized_principal = str(principal or "").strip()
    if not normalized_principal.startswith("cert:"):
        return frozenset()
    fingerprint = normalized_principal.split(":", 1)[1].strip()
    if not fingerprint:
        return frozenset()

    if session_factory is None or repository_factory is None or resolver is None:
        try:
            from webapp.parser.auth.trusted_identity_repository import (
                TrustedIdentityRepository,
            )
            from webapp.parser.auth.trusted_principal_authority import (
                resolve_enrolled_mtls_principal,
            )
            from webapp.parser.utils.db_utils import SessionLocal
        except Exception:
            return frozenset()
        if session_factory is None:
            session_factory = SessionLocal
        if repository_factory is None:
            repository_factory = TrustedIdentityRepository
        if resolver is None:
            resolver = resolve_enrolled_mtls_principal

    db_session = None
    try:
        db_session = session_factory()
        repository = repository_factory(db_session)
        decision = resolver(repository, fingerprint)
        if not bool(getattr(decision, "resolved", False)):
            return frozenset()
        if not bool(getattr(decision, "protected_operation_eligible", False)):
            return frozenset()
        return frozenset(getattr(decision, "role_names", ()) or ())
    except Exception:
        return frozenset()
    finally:
        if db_session is not None:
            try:
                db_session.close()
            except Exception:
                pass


def resolve_workflow_roles_for_principal(
    principal: object,
    *,
    environ: Mapping[str, str] | None = None,
) -> frozenset[str]:
    """Resolve Workflow roles from exactly one selected server-owned authority."""
    normalized_principal = str(principal or "").strip()
    if not normalized_principal:
        return frozenset()

    mode = _runtime_authority_mode(environ)
    if mode is None:
        return frozenset()

    if mode == AUTHORITY_MODE_DURABLE_TRUSTED_IDENTITY:
        return _durable_roles_for_principal(normalized_principal)

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
    """Require a W2 capability derived only from selected server-owned authority."""
    roles = resolve_workflow_roles_for_principal(principal, environ=environ)
    try:
        assert_capability(roles, str(required_capability or "").strip())
    except WorkflowAuthorizationError as exc:
        raise WorkflowRuntimeAuthorizationDenied(
            "Workflow capability denied."
        ) from exc
    return roles
