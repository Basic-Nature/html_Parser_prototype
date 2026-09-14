"""Preview-only repository for canonical trusted identity state."""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from webapp.parser.auth.trusted_identity_models import TrustedCredential, TrustedPrincipal, TrustedRoleBinding

class TrustedIdentityRepositoryError(RuntimeError):
    pass

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

class TrustedIdentityRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def require_principal(self, principal_id: UUID) -> TrustedPrincipal:
        principal = self._session.get(TrustedPrincipal, principal_id)
        if principal is None:
            raise TrustedIdentityRepositoryError("canonical principal not found")
        return principal

    def resolve_active_certificate(self, fingerprint_sha256: str) -> TrustedCredential | None:
        fingerprint = str(fingerprint_sha256 or "").strip().lower()
        if len(fingerprint) != 64:
            return None
        stmt = select(TrustedCredential).where(
            TrustedCredential.credential_type == "x509",
            TrustedCredential.fingerprint_sha256 == fingerprint,
            TrustedCredential.state == "active",
        ).limit(1)
        return self._session.execute(stmt).scalar_one_or_none()

    def current_role_names(self, principal_id: UUID, *, at: datetime | None = None) -> frozenset[str]:
        instant = at or _utcnow()
        stmt = select(TrustedRoleBinding.role_name).where(
            TrustedRoleBinding.principal_id == principal_id,
            TrustedRoleBinding.state == "active",
            TrustedRoleBinding.valid_from <= instant,
            (TrustedRoleBinding.valid_until.is_(None)) | (TrustedRoleBinding.valid_until > instant),
        )
        return frozenset(self._session.execute(stmt).scalars().all())

    def attach_certificate_credential(
        self,
        *,
        principal_id: UUID,
        fingerprint_sha256: str,
        not_before: datetime | None,
        not_after: datetime | None,
        issuer_metadata: dict,
        serial_metadata: dict,
        chain_validation_state: str,
        revocation_state: str,
    ) -> TrustedCredential:
        principal = self.require_principal(principal_id)
        if principal.state == "revoked":
            raise TrustedIdentityRepositoryError("cannot attach credential to revoked principal")
        fingerprint = str(fingerprint_sha256 or "").strip().lower()
        if len(fingerprint) != 64:
            raise TrustedIdentityRepositoryError("certificate fingerprint must be 64 hex characters")
        try:
            int(fingerprint, 16)
        except ValueError as exc:
            raise TrustedIdentityRepositoryError("certificate fingerprint must be hexadecimal") from exc
        if self.resolve_active_certificate(fingerprint) is not None:
            raise TrustedIdentityRepositoryError("active certificate credential already exists")
        credential = TrustedCredential(
            principal_id=principal_id,
            credential_type="x509",
            fingerprint_sha256=fingerprint,
            state="pending",
            not_before=not_before,
            not_after=not_after,
            issuer_metadata=dict(issuer_metadata or {}),
            serial_metadata=dict(serial_metadata or {}),
            chain_validation_state=chain_validation_state,
            revocation_state=revocation_state,
        )
        self._session.add(credential)
        return credential

    def add_role_binding(
        self,
        *,
        principal_id: UUID,
        role_name: str,
        reason_code: str,
        granted_by_principal_id: UUID | None,
        valid_from: datetime | None = None,
        valid_until: datetime | None = None,
    ) -> TrustedRoleBinding:
        self.require_principal(principal_id)
        role = str(role_name or "").strip()
        reason = str(reason_code or "").strip()
        if not role or not reason:
            raise TrustedIdentityRepositoryError("explicit role_name and reason_code are required")
        binding = TrustedRoleBinding(
            principal_id=principal_id,
            role_name=role,
            state="active",
            granted_by_principal_id=granted_by_principal_id,
            reason_code=reason,
            valid_from=valid_from or _utcnow(),
            valid_until=valid_until,
        )
        self._session.add(binding)
        return binding

    def migrate_fingerprint_candidate(self, *_args, **_kwargs) -> None:
        raise TrustedIdentityRepositoryError(
            "fingerprint migration requires an explicit operator-reviewed principal mapping manifest; "
            "automatic migration is forbidden"
        )

    def capabilities_for_principal(self, principal_id: UUID) -> frozenset[str]:
        from webapp.parser.contracts.workflow_authorization import capabilities_for_roles
        return capabilities_for_roles(self.current_role_names(principal_id))
