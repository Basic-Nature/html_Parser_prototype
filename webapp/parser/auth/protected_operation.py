"""Preview-only fail-closed protected-operation evaluator."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy.orm import Session
from webapp.parser.auth.trusted_identity_models import TrustedCredential, TrustedElevationGrant, TrustedPrincipal
from webapp.parser.auth.trusted_identity_repository import TrustedIdentityRepository

@dataclass(frozen=True)
class ProtectedOperationDecision:
    allowed: bool
    code: str

def authorize_protected_operation(
    session: Session,
    *,
    principal_id: UUID,
    grant_id: UUID,
    operation_class: str,
    required_capability: str,
    now: datetime | None = None,
) -> ProtectedOperationDecision:
    instant = now or datetime.now(timezone.utc)
    repo = TrustedIdentityRepository(session)

    principal = session.get(TrustedPrincipal, principal_id)
    if principal is None:
        return ProtectedOperationDecision(False, "deny_identity")
    if principal.state != "trusted":
        return ProtectedOperationDecision(False, "deny_principal_trust")
    if required_capability not in repo.capabilities_for_principal(principal_id):
        return ProtectedOperationDecision(False, "deny_capability")

    grant = session.get(TrustedElevationGrant, grant_id)
    if grant is None or grant.principal_id != principal_id:
        return ProtectedOperationDecision(False, "deny_elevation")
    if grant.state != "active" or grant.expires_at <= instant:
        return ProtectedOperationDecision(False, "deny_elevation")
    if grant.operation_class != operation_class or grant.required_capability != required_capability:
        return ProtectedOperationDecision(False, "deny_elevation")

    credential = session.get(TrustedCredential, grant.credential_id)
    if credential is None or credential.state != "active":
        return ProtectedOperationDecision(False, "deny_credential")
    if credential.revocation_state not in {"good", "not_applicable"}:
        return ProtectedOperationDecision(False, "deny_credential")
    if credential.not_before is not None and instant < credential.not_before:
        return ProtectedOperationDecision(False, "deny_credential")
    if credential.not_after is not None and instant >= credential.not_after:
        return ProtectedOperationDecision(False, "deny_credential")

    return ProtectedOperationDecision(True, "allow")
