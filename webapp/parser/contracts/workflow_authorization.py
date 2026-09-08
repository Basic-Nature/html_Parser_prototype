"""Pure governed Workflow authorization vocabulary and separation contract.

This module is intentionally dependency-free and provider-neutral.

It defines ElectionPulse-internal Workflow roles and capabilities, future
allowlisted identity-provider role mapping, and separation-of-duty invariants.

It does NOT authenticate a principal, parse OIDC/JWT tokens, enable Keycloak,
register routes, access the database, mutate Workflow state, or grant canonical
write authority.
"""

from __future__ import annotations

from collections.abc import Iterable


WORKFLOW_AUTHORIZATION_CONTRACT = "workflow_authorization_contract_v1"

CAP_SOURCE_READ = "workflow.source.read"
CAP_DL1_CLAIM = "workflow.dl1.claim"
CAP_DL1_SUBMIT = "workflow.dl1.submit"
CAP_DL2_CLAIM = "workflow.dl2.claim"
CAP_DL2_SUBMIT = "workflow.dl2.submit"
CAP_DISCREPANCY_RESOLVE = "workflow.discrepancy.resolve"
CAP_QC1_REVIEW = "workflow.qc1.review"
CAP_QC2_REVIEW = "workflow.qc2.review"
CAP_PUBLICATION_HANDOFF = "workflow.publication.handoff"
CAP_AUDIT_READ = "workflow.audit.read"
CAP_COMPARISON_EXECUTE = "workflow.comparison.execute"

CAPABILITIES = frozenset({
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
})

ROLE_CONTRIBUTOR = "workflow_contributor"
ROLE_REVIEWER = "workflow_reviewer"
ROLE_PUBLICATION_OPERATOR = "workflow_publication_operator"
ROLE_AUDITOR = "workflow_auditor"
ROLE_COMPARISON_SERVICE = "workflow_comparison_service"

HUMAN_ROLES = frozenset({
    ROLE_CONTRIBUTOR,
    ROLE_REVIEWER,
    ROLE_PUBLICATION_OPERATOR,
    ROLE_AUDITOR,
})
SERVICE_ROLES = frozenset({ROLE_COMPARISON_SERVICE})
ROLES = HUMAN_ROLES | SERVICE_ROLES

ROLE_CAPABILITIES = {
    ROLE_CONTRIBUTOR: frozenset({
        CAP_SOURCE_READ,
        CAP_DL1_CLAIM,
        CAP_DL1_SUBMIT,
        CAP_DL2_CLAIM,
        CAP_DL2_SUBMIT,
    }),
    ROLE_REVIEWER: frozenset({
        CAP_SOURCE_READ,
        CAP_DISCREPANCY_RESOLVE,
        CAP_QC1_REVIEW,
        CAP_QC2_REVIEW,
        CAP_AUDIT_READ,
    }),
    ROLE_PUBLICATION_OPERATOR: frozenset({
        CAP_PUBLICATION_HANDOFF,
        CAP_AUDIT_READ,
    }),
    ROLE_AUDITOR: frozenset({
        CAP_AUDIT_READ,
    }),
    ROLE_COMPARISON_SERVICE: frozenset({
        CAP_COMPARISON_EXECUTE,
    }),
}

KEYCLOAK_ROLE_TO_INTERNAL_ROLE = {
    "electionpulse-workflow-contributor": ROLE_CONTRIBUTOR,
    "electionpulse-workflow-reviewer": ROLE_REVIEWER,
    "electionpulse-workflow-publication-operator": ROLE_PUBLICATION_OPERATOR,
    "electionpulse-workflow-auditor": ROLE_AUDITOR,
}

PUBLIC_WORKFLOW_REQUIRES_KEYCLOAK = False
KEYCLOAK_ENABLED_BY_CONTRACT = False
LEGACY_PRIVILEGE_TIER_IMPLIES_WORKFLOW_CAPABILITY = False
EXTERNAL_CAPABILITY_CLAIMS_ACCEPTED = False
STRICT_COMPARISON_SERVICE_ONLY = True
WORKFLOW_PUBLICATION_HANDOFF_IS_CANONICAL_WRITE = False

RESOLVED_W2A_DECISIONS = (
    "exact protected contributor role/capability names and Keycloak mapping",
    "whether QC1/QC2 reviewers must also differ from DL1/DL2 principals",
)

REMAINING_DEFERRED_DECISIONS = (
    "exact normalized semantic comparison payload schema and version",
    "exact canonical writer callback/result contract used by publication_handoff",
    "whether publication operator must differ from all DL/QC principals",
)


class WorkflowAuthorizationError(ValueError):
    """Fail-closed Workflow authorization contract violation."""


def _normalize_values(name: str, values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise WorkflowAuthorizationError(
            f"{name} must be an iterable of role strings, not a single string"
        )

    normalized: list[str] = []
    for raw in values:
        value = str(raw or "").strip()
        if not value:
            raise WorkflowAuthorizationError(
                f"{name} must not contain empty role values"
            )
        normalized.append(value)
    return tuple(normalized)


def map_external_roles(external_roles: Iterable[str]) -> frozenset[str]:
    """Map only allowlisted external human roles to internal Workflow roles.

    Unknown external roles are ignored (fail closed). Service roles are never
    obtainable from external human identity-provider claims.
    """
    normalized = _normalize_values("external_roles", external_roles)
    return frozenset(
        KEYCLOAK_ROLE_TO_INTERNAL_ROLE[role]
        for role in normalized
        if role in KEYCLOAK_ROLE_TO_INTERNAL_ROLE
    )


def capabilities_for_roles(internal_roles: Iterable[str]) -> frozenset[str]:
    """Derive capabilities server-side from already-mapped internal roles."""
    normalized = _normalize_values("internal_roles", internal_roles)
    unknown = sorted(set(normalized) - ROLES)
    if unknown:
        raise WorkflowAuthorizationError(
            f"unknown internal Workflow role(s): {unknown!r}"
        )

    capabilities: set[str] = set()
    for role in normalized:
        capabilities.update(ROLE_CAPABILITIES[role])
    return frozenset(capabilities)


def assert_capability(
    internal_roles: Iterable[str],
    required_capability: str,
) -> None:
    capability = str(required_capability or "").strip()
    if capability not in CAPABILITIES:
        raise WorkflowAuthorizationError(
            f"unknown Workflow capability: {capability!r}"
        )
    if capability not in capabilities_for_roles(internal_roles):
        raise WorkflowAuthorizationError(
            f"required Workflow capability not granted: {capability}"
        )


def assert_four_principal_separation(
    *,
    dl1_principal: str,
    dl2_principal: str,
    qc1_principal: str,
    qc2_principal: str,
) -> None:
    """Require four distinct principals for DL1, DL2, QC1, and QC2."""
    principals = {
        "dl1_principal": dl1_principal,
        "dl2_principal": dl2_principal,
        "qc1_principal": qc1_principal,
        "qc2_principal": qc2_principal,
    }

    normalized: dict[str, str] = {}
    for name, raw in principals.items():
        value = str(raw or "").strip()
        if not value:
            raise WorkflowAuthorizationError(
                f"{name} must be a non-empty principal"
            )
        normalized[name] = value

    if len(set(normalized.values())) != 4:
        raise WorkflowAuthorizationError(
            "DL1, DL2, QC1, and QC2 require four distinct principals"
        )
