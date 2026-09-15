"""Preview-only resource-bound protected-operation evaluator.

No external runtime route calls this module yet. A future sensitive Workflow
route can validate and consume one resource-bound grant inside the same
transaction as the Workflow mutation. The caller owns commit/rollback.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from webapp.parser.auth.trust_audit import append_trust_event
from webapp.parser.auth.trusted_identity_models import (
    TrustedCredential,
    TrustedElevationChallenge,
    TrustedElevationGrant,
    TrustedPrincipal,
    TrustedSession,
)
from webapp.parser.auth.trusted_identity_repository import TrustedIdentityRepository

POLICY_VERSION = "operator_protected_action_policy_v1"

@dataclass(frozen=True)
class ProtectedOperationDecision:
    allowed: bool
    code: str
    grant_id: UUID | None = None


def _metadata(*, operation_class: str, required_capability: str, resource_type: str, resource_id: str, resource_version: int) -> dict[str, object]:
    return {
        "operation_class": operation_class,
        "required_capability": required_capability,
        "resource_type": resource_type,
        "resource_id": resource_id,
        "resource_version": int(resource_version),
    }


def _decision_event(session: Session, *, allowed: bool, principal_id: UUID, credential_id: UUID | None, trusted_session_id: UUID | None, challenge_id: UUID | None, grant_id: UUID | None, operation_class: str, required_capability: str, resource_type: str, resource_id: str, resource_version: int, reason_code: str) -> None:
    append_trust_event(
        session,
        principal_id=principal_id,
        actor_principal_id=principal_id,
        credential_id=credential_id,
        device_id=None,
        session_id=trusted_session_id,
        challenge_id=challenge_id,
        grant_id=grant_id,
        event_type="protected_operation_allowed" if allowed else "protected_operation_denied",
        reason_code=reason_code,
        policy_version=POLICY_VERSION,
        event_metadata=_metadata(
            operation_class=operation_class,
            required_capability=required_capability,
            resource_type=resource_type,
            resource_id=resource_id,
            resource_version=resource_version,
        ),
    )


def _deny(session: Session, *, code: str, principal_id: UUID, credential_id: UUID | None, trusted_session_id: UUID | None, challenge_id: UUID | None, grant_id: UUID | None, operation_class: str, required_capability: str, resource_type: str, resource_id: str, resource_version: int) -> ProtectedOperationDecision:
    _decision_event(
        session,
        allowed=False,
        principal_id=principal_id,
        credential_id=credential_id,
        trusted_session_id=trusted_session_id,
        challenge_id=challenge_id,
        grant_id=grant_id,
        operation_class=operation_class,
        required_capability=required_capability,
        resource_type=resource_type,
        resource_id=resource_id,
        resource_version=resource_version,
        reason_code=code,
    )
    return ProtectedOperationDecision(False, code, grant_id)


def authorize_and_consume_protected_operation(
    session: Session,
    *,
    principal_id: UUID,
    current_credential_id: UUID,
    trusted_session_id: UUID,
    browser_session_binding_hash: str,
    grant_id: UUID,
    operation_class: str,
    required_capability: str,
    resource_type: str,
    resource_id: str,
    resource_version: int,
    now: datetime | None = None,
) -> ProtectedOperationDecision:
    instant = now or datetime.now(timezone.utc)
    resource_type = str(resource_type or "").strip()
    resource_id = str(resource_id or "").strip()
    browser_session_binding_hash = str(browser_session_binding_hash or "").strip()
    if not resource_type or not resource_id or int(resource_version) < 1:
        return _deny(session, code="deny_resource", principal_id=principal_id, credential_id=current_credential_id, trusted_session_id=trusted_session_id, challenge_id=None, grant_id=grant_id, operation_class=operation_class, required_capability=required_capability, resource_type=resource_type, resource_id=resource_id, resource_version=resource_version)

    repo = TrustedIdentityRepository(session)
    principal = session.get(TrustedPrincipal, principal_id)
    if principal is None:
        return _deny(session, code="deny_identity", principal_id=principal_id, credential_id=current_credential_id, trusted_session_id=trusted_session_id, challenge_id=None, grant_id=grant_id, operation_class=operation_class, required_capability=required_capability, resource_type=resource_type, resource_id=resource_id, resource_version=resource_version)
    if principal.state != "trusted":
        return _deny(session, code="deny_principal_trust", principal_id=principal_id, credential_id=current_credential_id, trusted_session_id=trusted_session_id, challenge_id=None, grant_id=grant_id, operation_class=operation_class, required_capability=required_capability, resource_type=resource_type, resource_id=resource_id, resource_version=resource_version)
    if required_capability not in repo.capabilities_for_principal(principal_id):
        return _deny(session, code="deny_capability", principal_id=principal_id, credential_id=current_credential_id, trusted_session_id=trusted_session_id, challenge_id=None, grant_id=grant_id, operation_class=operation_class, required_capability=required_capability, resource_type=resource_type, resource_id=resource_id, resource_version=resource_version)

    trusted_session = session.execute(
        select(TrustedSession).where(TrustedSession.id == trusted_session_id).with_for_update()
    ).scalar_one_or_none()
    if trusted_session is None or trusted_session.principal_id != principal_id or trusted_session.state != "elevated" or trusted_session.expires_at <= instant or trusted_session.browser_session_binding_hash != browser_session_binding_hash:
        return _deny(session, code="deny_session", principal_id=principal_id, credential_id=current_credential_id, trusted_session_id=trusted_session_id, challenge_id=None, grant_id=grant_id, operation_class=operation_class, required_capability=required_capability, resource_type=resource_type, resource_id=resource_id, resource_version=resource_version)

    grant = session.execute(
        select(TrustedElevationGrant).where(TrustedElevationGrant.id == grant_id).with_for_update()
    ).scalar_one_or_none()
    if grant is None or grant.principal_id != principal_id or grant.trusted_session_id != trusted_session_id or grant.credential_id != current_credential_id or grant.state != "active" or grant.expires_at <= instant or grant.operation_class != operation_class or grant.required_capability != required_capability or grant.resource_type != resource_type or grant.resource_id != resource_id or int(grant.resource_version) != int(resource_version) or grant.policy_version != POLICY_VERSION:
        return _deny(session, code="deny_elevation", principal_id=principal_id, credential_id=current_credential_id, trusted_session_id=trusted_session_id, challenge_id=(grant.challenge_id if grant is not None else None), grant_id=grant_id, operation_class=operation_class, required_capability=required_capability, resource_type=resource_type, resource_id=resource_id, resource_version=resource_version)

    credential = session.get(TrustedCredential, current_credential_id)
    if credential is None or credential.principal_id != principal_id or credential.state != "active" or credential.revocation_state not in {"good", "not_applicable"} or (credential.not_before is not None and instant < credential.not_before) or (credential.not_after is not None and instant >= credential.not_after):
        return _deny(session, code="deny_credential", principal_id=principal_id, credential_id=current_credential_id, trusted_session_id=trusted_session_id, challenge_id=grant.challenge_id, grant_id=grant_id, operation_class=operation_class, required_capability=required_capability, resource_type=resource_type, resource_id=resource_id, resource_version=resource_version)

    challenge = session.execute(
        select(TrustedElevationChallenge).where(TrustedElevationChallenge.id == grant.challenge_id).with_for_update()
    ).scalar_one_or_none()
    if challenge is None or challenge.browser_session_binding_hash != browser_session_binding_hash or challenge.state != "consumed" or challenge.consumed_at is None or challenge.requested_operation_class != operation_class or challenge.required_capability != required_capability or challenge.resource_type != resource_type or challenge.resource_id != resource_id or int(challenge.resource_version) != int(resource_version) or challenge.consumed_at > grant.issued_at or challenge.expires_at <= challenge.consumed_at:
        return _deny(session, code="deny_challenge", principal_id=principal_id, credential_id=current_credential_id, trusted_session_id=trusted_session_id, challenge_id=grant.challenge_id, grant_id=grant_id, operation_class=operation_class, required_capability=required_capability, resource_type=resource_type, resource_id=resource_id, resource_version=resource_version)

    _decision_event(session, allowed=True, principal_id=principal_id, credential_id=current_credential_id, trusted_session_id=trusted_session_id, challenge_id=grant.challenge_id, grant_id=grant_id, operation_class=operation_class, required_capability=required_capability, resource_type=resource_type, resource_id=resource_id, resource_version=resource_version, reason_code="allow")
    grant.state = "consumed"
    grant.consumed_at = instant
    append_trust_event(
        session,
        principal_id=principal_id,
        actor_principal_id=principal_id,
        credential_id=current_credential_id,
        device_id=None,
        session_id=trusted_session_id,
        challenge_id=grant.challenge_id,
        grant_id=grant_id,
        event_type="protected_operation_grant_consumed",
        reason_code="single_use_consumed",
        policy_version=POLICY_VERSION,
        event_metadata=_metadata(operation_class=operation_class, required_capability=required_capability, resource_type=resource_type, resource_id=resource_id, resource_version=resource_version),
    )
    return ProtectedOperationDecision(True, "allow", grant_id)
