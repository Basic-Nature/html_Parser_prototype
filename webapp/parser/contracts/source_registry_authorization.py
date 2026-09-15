"""Pure Source Registry authorization and separation vocabulary.

This module reuses ElectionPulse durable Workflow human-role names. It does not
authenticate a principal, access a database, register routes, or mutate source
authority.
"""
from __future__ import annotations

from collections.abc import Iterable

from webapp.parser.contracts.workflow_authorization import (
    ROLE_AUDITOR,
    ROLE_CONTRIBUTOR,
    ROLE_PUBLICATION_OPERATOR,
    ROLE_REVIEWER,
)

SOURCE_REGISTRY_AUTHORIZATION_CONTRACT = "source_registry_authorization_contract_v1"

CAP_SOURCE_REGISTRY_READ = "source_registry.read"
CAP_SOURCE_REGISTRY_PROPOSE = "source_registry.propose"
CAP_SOURCE_REGISTRY_REVIEW = "source_registry.review"
CAP_SOURCE_REGISTRY_PUBLISH = "source_registry.publish"
CAP_SOURCE_REGISTRY_AUDIT_READ = "source_registry.audit.read"
CAP_SOURCE_REGISTRY_QUARANTINE = "source_registry.quarantine"

SOURCE_REGISTRY_CAPABILITIES = frozenset({
    CAP_SOURCE_REGISTRY_READ,
    CAP_SOURCE_REGISTRY_PROPOSE,
    CAP_SOURCE_REGISTRY_REVIEW,
    CAP_SOURCE_REGISTRY_PUBLISH,
    CAP_SOURCE_REGISTRY_AUDIT_READ,
    CAP_SOURCE_REGISTRY_QUARANTINE,
})

SOURCE_REGISTRY_ROLE_CAPABILITIES = {
    ROLE_CONTRIBUTOR: frozenset({
        CAP_SOURCE_REGISTRY_READ,
        CAP_SOURCE_REGISTRY_PROPOSE,
    }),
    ROLE_REVIEWER: frozenset({
        CAP_SOURCE_REGISTRY_READ,
        CAP_SOURCE_REGISTRY_REVIEW,
        CAP_SOURCE_REGISTRY_AUDIT_READ,
        CAP_SOURCE_REGISTRY_QUARANTINE,
    }),
    ROLE_PUBLICATION_OPERATOR: frozenset({
        CAP_SOURCE_REGISTRY_READ,
        CAP_SOURCE_REGISTRY_PUBLISH,
        CAP_SOURCE_REGISTRY_AUDIT_READ,
        CAP_SOURCE_REGISTRY_QUARANTINE,
    }),
    ROLE_AUDITOR: frozenset({
        CAP_SOURCE_REGISTRY_READ,
        CAP_SOURCE_REGISTRY_AUDIT_READ,
    }),
}


class SourceRegistryAuthorizationError(ValueError):
    pass


def _nonempty(name: str, value: object) -> str:
    text = str(value or "").strip()
    if not text:
        raise SourceRegistryAuthorizationError(f"{name} must be non-empty")
    return text


def capabilities_for_roles(roles: Iterable[str]) -> frozenset[str]:
    if isinstance(roles, (str, bytes)):
        raise SourceRegistryAuthorizationError("roles must be an iterable")
    result: set[str] = set()
    for raw in roles:
        role = _nonempty("role", raw)
        result.update(SOURCE_REGISTRY_ROLE_CAPABILITIES.get(role, ()))
    return frozenset(result)


def assert_capability(roles: Iterable[str], capability: str) -> None:
    wanted = _nonempty("capability", capability)
    if wanted not in SOURCE_REGISTRY_CAPABILITIES:
        raise SourceRegistryAuthorizationError(
            f"unknown Source Registry capability: {wanted!r}"
        )
    if wanted not in capabilities_for_roles(roles):
        raise SourceRegistryAuthorizationError(
            f"required Source Registry capability not granted: {wanted}"
        )


def assert_normal_publish_separation(
    *,
    proposer_principal: str,
    reviewer_principal: str,
    publisher_principal: str,
) -> None:
    proposer = _nonempty("proposer_principal", proposer_principal)
    reviewer = _nonempty("reviewer_principal", reviewer_principal)
    publisher = _nonempty("publisher_principal", publisher_principal)
    if len({proposer, reviewer, publisher}) != 3:
        raise SourceRegistryAuthorizationError(
            "Source Registry publish requires distinct proposer, reviewer, and publisher"
        )
