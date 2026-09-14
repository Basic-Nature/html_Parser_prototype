"""Preview-only append-only trusted authority audit helpers."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session
from webapp.parser.auth.trusted_identity_models import TrustedTrustEvent

class TrustAuditError(RuntimeError):
    pass

def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)

def append_trust_event(
    session: Session,
    *,
    principal_id: UUID | None,
    actor_principal_id: UUID | None,
    credential_id: UUID | None,
    device_id: UUID | None,
    session_id: UUID | None,
    challenge_id: UUID | None,
    grant_id: UUID | None,
    event_type: str,
    reason_code: str | None,
    policy_version: str | None,
    event_metadata: dict[str, Any] | None,
) -> TrustedTrustEvent:
    last_hash = session.execute(
        select(TrustedTrustEvent.event_hash)
        .order_by(TrustedTrustEvent.occurred_at.desc(), TrustedTrustEvent.id.desc())
        .limit(1)
    ).scalar_one_or_none()
    occurred_at = datetime.now(timezone.utc)
    metadata = dict(event_metadata or {})
    payload = {
        "principal_id": str(principal_id) if principal_id else None,
        "actor_principal_id": str(actor_principal_id) if actor_principal_id else None,
        "credential_id": str(credential_id) if credential_id else None,
        "device_id": str(device_id) if device_id else None,
        "session_id": str(session_id) if session_id else None,
        "challenge_id": str(challenge_id) if challenge_id else None,
        "grant_id": str(grant_id) if grant_id else None,
        "event_type": str(event_type or "").strip(),
        "reason_code": str(reason_code).strip() if reason_code else None,
        "policy_version": str(policy_version).strip() if policy_version else None,
        "event_metadata": metadata,
        "previous_event_hash": last_hash,
        "occurred_at": occurred_at.isoformat(),
    }
    if not payload["event_type"]:
        raise TrustAuditError("event_type is required")
    event_hash = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    event = TrustedTrustEvent(
        principal_id=principal_id,
        actor_principal_id=actor_principal_id,
        credential_id=credential_id,
        device_id=device_id,
        session_id=session_id,
        challenge_id=challenge_id,
        grant_id=grant_id,
        event_type=payload["event_type"],
        reason_code=payload["reason_code"],
        policy_version=payload["policy_version"],
        event_metadata=metadata,
        previous_event_hash=last_hash,
        event_hash=event_hash,
        occurred_at=occurred_at,
    )
    session.add(event)
    return event

def update_trust_event(*_args, **_kwargs) -> None:
    raise TrustAuditError("trusted trust events are append-only")

def delete_trust_event(*_args, **_kwargs) -> None:
    raise TrustAuditError("trusted trust events are append-only")
