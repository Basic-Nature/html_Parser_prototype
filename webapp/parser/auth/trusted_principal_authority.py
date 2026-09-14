"""Durable trusted-principal resolution foundation.

This module is intentionally NOT wired into request authentication yet.

It resolves an already-enrolled active X.509 credential into the canonical
ElectionPulse trusted principal authority stored in PostgreSQL. Authentication
requests are read-only consumers of this authority: they MUST NOT auto-create
principals, credentials, bindings, roles, sessions, devices, or grants.

The current ``cert:<fingerprint>`` application principal remains a compatibility
identifier until a later explicitly authorized runtime-activation tranche.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol
from uuid import UUID

from webapp.parser.auth.cert_trust import normalize_sha256_fingerprint
from webapp.parser.auth.trusted_identity_repository import (
    TrustedIdentityRepositoryError,
)


TRUSTED_PRINCIPAL_AUTHORITY_CONTRACT = "trusted_principal_authority_v1"
PROVIDER_ELECTIONPULSE_MTLS = "electionpulse_mtls"
CANONICAL_PRINCIPAL_PREFIX = "principal:"

PRINCIPAL_STATE_TRUSTED = "trusted"
PRINCIPAL_STATE_RESTRICTED = "restricted"
RESOLVABLE_PRINCIPAL_STATES = frozenset({
    PRINCIPAL_STATE_TRUSTED,
    PRINCIPAL_STATE_RESTRICTED,
})


class TrustedPrincipalRepositoryProtocol(Protocol):
    def resolve_active_certificate(self, fingerprint_sha256: str): ...
    def require_principal(self, principal_id: UUID): ...
    def current_role_names(
        self,
        principal_id: UUID,
        *,
        at: datetime | None = None,
    ) -> frozenset[str]: ...
    def capabilities_for_principal(self, principal_id: UUID) -> frozenset[str]: ...


@dataclass(frozen=True)
class TrustedPrincipalAuthorityDecision:
    contract_version: str
    provider: str
    resolved: bool
    protected_operation_eligible: bool
    reason_code: str
    canonical_principal: str | None
    compatibility_principal: str | None
    principal_id: str | None
    principal_type: str | None
    principal_state: str | None
    credential_id: str | None
    credential_state: str | None
    role_names: tuple[str, ...]
    capabilities: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "provider": self.provider,
            "resolved": self.resolved,
            "protected_operation_eligible": self.protected_operation_eligible,
            "reason_code": self.reason_code,
            "canonical_principal": self.canonical_principal,
            "compatibility_principal": self.compatibility_principal,
            "principal_id": self.principal_id,
            "principal_type": self.principal_type,
            "principal_state": self.principal_state,
            "credential_id": self.credential_id,
            "credential_state": self.credential_state,
            "role_names": list(self.role_names),
            "capabilities": list(self.capabilities),
        }


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _decision(
    *,
    resolved: bool,
    protected_operation_eligible: bool,
    reason_code: str,
    fingerprint: str | None,
    principal=None,
    credential=None,
    role_names: frozenset[str] = frozenset(),
    capabilities: frozenset[str] = frozenset(),
) -> TrustedPrincipalAuthorityDecision:
    principal_id = getattr(principal, "id", None)
    credential_id = getattr(credential, "id", None)
    return TrustedPrincipalAuthorityDecision(
        contract_version=TRUSTED_PRINCIPAL_AUTHORITY_CONTRACT,
        provider=PROVIDER_ELECTIONPULSE_MTLS,
        resolved=resolved,
        protected_operation_eligible=protected_operation_eligible,
        reason_code=reason_code,
        canonical_principal=(
            f"{CANONICAL_PRINCIPAL_PREFIX}{principal_id}"
            if resolved and principal_id is not None
            else None
        ),
        compatibility_principal=(
            f"cert:{fingerprint}"
            if fingerprint is not None
            else None
        ),
        principal_id=str(principal_id) if principal_id is not None else None,
        principal_type=getattr(principal, "principal_type", None),
        principal_state=getattr(principal, "state", None),
        credential_id=str(credential_id) if credential_id is not None else None,
        credential_state=getattr(credential, "state", None),
        role_names=tuple(sorted(role_names)),
        capabilities=tuple(sorted(capabilities)),
    )


def resolve_enrolled_mtls_principal(
    repository: TrustedPrincipalRepositoryProtocol,
    fingerprint_sha256: object,
    *,
    at: datetime | None = None,
) -> TrustedPrincipalAuthorityDecision:
    """Resolve an already-enrolled active certificate to durable authority.

    This function never creates or mutates trusted-identity state.
    """
    fingerprint = normalize_sha256_fingerprint(fingerprint_sha256)
    if fingerprint is None:
        return _decision(
            resolved=False,
            protected_operation_eligible=False,
            reason_code="invalid_certificate_fingerprint",
            fingerprint=None,
        )

    try:
        credential = repository.resolve_active_certificate(fingerprint)
    except TrustedIdentityRepositoryError:
        return _decision(
            resolved=False,
            protected_operation_eligible=False,
            reason_code="repository_lookup_failed",
            fingerprint=fingerprint,
        )

    if credential is None:
        return _decision(
            resolved=False,
            protected_operation_eligible=False,
            reason_code="credential_not_enrolled",
            fingerprint=fingerprint,
        )

    instant = _as_utc(at) or _utcnow()
    not_before = _as_utc(getattr(credential, "not_before", None))
    not_after = _as_utc(getattr(credential, "not_after", None))

    if not_before is not None and instant < not_before:
        return _decision(
            resolved=False,
            protected_operation_eligible=False,
            reason_code="credential_not_yet_valid",
            fingerprint=fingerprint,
            credential=credential,
        )

    if not_after is not None and instant >= not_after:
        return _decision(
            resolved=False,
            protected_operation_eligible=False,
            reason_code="credential_expired",
            fingerprint=fingerprint,
            credential=credential,
        )

    try:
        principal = repository.require_principal(credential.principal_id)
    except TrustedIdentityRepositoryError:
        return _decision(
            resolved=False,
            protected_operation_eligible=False,
            reason_code="principal_lookup_failed",
            fingerprint=fingerprint,
            credential=credential,
        )

    principal_state = str(getattr(principal, "state", "") or "")
    if principal_state not in RESOLVABLE_PRINCIPAL_STATES:
        return _decision(
            resolved=False,
            protected_operation_eligible=False,
            reason_code=f"principal_state_{principal_state or 'invalid'}",
            fingerprint=fingerprint,
            principal=principal,
            credential=credential,
        )

    # Restricted principals remain canonically resolvable so audit identity is
    # stable, but they cannot become protected-operation authority here.
    if principal_state == PRINCIPAL_STATE_RESTRICTED:
        return _decision(
            resolved=True,
            protected_operation_eligible=False,
            reason_code="principal_restricted",
            fingerprint=fingerprint,
            principal=principal,
            credential=credential,
        )

    try:
        role_names = repository.current_role_names(principal.id, at=instant)
        capabilities = repository.capabilities_for_principal(principal.id)
    except TrustedIdentityRepositoryError:
        return _decision(
            resolved=False,
            protected_operation_eligible=False,
            reason_code="authorization_lookup_failed",
            fingerprint=fingerprint,
            principal=principal,
            credential=credential,
        )

    return _decision(
        resolved=True,
        protected_operation_eligible=True,
        reason_code="trusted_principal_resolved",
        fingerprint=fingerprint,
        principal=principal,
        credential=credential,
        role_names=role_names,
        capabilities=capabilities,
    )
