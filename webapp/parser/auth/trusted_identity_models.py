"""Preview-only trusted identity persistence models.

Durable trusted identity authority is PostgreSQL-backed and is separate from
parser-runtime SessionManager state and signed browser cookies.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import CheckConstraint, Column, DateTime, ForeignKey, Index, Integer, JSON, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from webapp.parser.utils.models import Base

TRUST_JSON = JSON().with_variant(JSONB(), "postgresql")

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

class TrustedPrincipal(Base):
    __tablename__ = "trusted_principals"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    principal_type = Column(String(16), nullable=False)
    state = Column(String(16), nullable=False)
    state_reason_code = Column(String(64), nullable=False)
    principal_metadata = Column(TRUST_JSON, nullable=False, default=dict)
    row_version = Column(Integer, nullable=False, default=1)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow)
    __table_args__ = (
        CheckConstraint("principal_type IN ('human','service')", name="ck_trusted_principal_type"),
        CheckConstraint("state IN ('pending','trusted','restricted','suspended','revoked')", name="ck_trusted_principal_state"),
        CheckConstraint("row_version >= 1", name="ck_trusted_principal_row_version"),
        Index("ix_trusted_principals_type_state", "principal_type", "state"),
    )

class TrustedIdentityBinding(Base):
    __tablename__ = "trusted_identity_bindings"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    principal_id = Column(UUID(as_uuid=True), ForeignKey("trusted_principals.id", ondelete="RESTRICT"), nullable=False)
    provider = Column(String(48), nullable=False)
    provider_subject_hash = Column(String(64), nullable=False)
    subject_hash_version = Column(String(32), nullable=False)
    state = Column(String(16), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    disabled_at = Column(DateTime(timezone=True), nullable=True)
    __table_args__ = (
        UniqueConstraint("provider", "provider_subject_hash", "subject_hash_version", name="uq_trusted_identity_provider_subject_hash"),
        CheckConstraint("state IN ('active','disabled')", name="ck_trusted_identity_binding_state"),
        Index("ix_trusted_identity_bindings_principal", "principal_id"),
    )

class TrustedCredential(Base):
    __tablename__ = "trusted_credentials"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    principal_id = Column(UUID(as_uuid=True), ForeignKey("trusted_principals.id", ondelete="RESTRICT"), nullable=False)
    credential_type = Column(String(24), nullable=False)
    fingerprint_sha256 = Column(String(64), nullable=False)
    state = Column(String(16), nullable=False)
    not_before = Column(DateTime(timezone=True), nullable=True)
    not_after = Column(DateTime(timezone=True), nullable=True)
    issuer_metadata = Column(TRUST_JSON, nullable=False, default=dict)
    serial_metadata = Column(TRUST_JSON, nullable=False, default=dict)
    chain_validation_state = Column(String(32), nullable=False)
    revocation_state = Column(String(32), nullable=False)
    superseded_by_credential_id = Column(UUID(as_uuid=True), ForeignKey("trusted_credentials.id", ondelete="RESTRICT"), nullable=True)
    row_version = Column(Integer, nullable=False, default=1)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow)
    __table_args__ = (
        UniqueConstraint("credential_type", "fingerprint_sha256", name="uq_trusted_credential_type_fingerprint"),
        CheckConstraint("state IN ('pending','active','superseded','expired','revoked','untrusted')", name="ck_trusted_credential_state"),
        CheckConstraint("row_version >= 1", name="ck_trusted_credential_row_version"),
        Index("ix_trusted_credentials_principal_state", "principal_id", "state"),
    )

class TrustedRoleBinding(Base):
    __tablename__ = "trusted_role_bindings"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    principal_id = Column(UUID(as_uuid=True), ForeignKey("trusted_principals.id", ondelete="RESTRICT"), nullable=False)
    role_name = Column(String(96), nullable=False)
    state = Column(String(16), nullable=False)
    granted_by_principal_id = Column(UUID(as_uuid=True), ForeignKey("trusted_principals.id", ondelete="RESTRICT"), nullable=True)
    reason_code = Column(String(64), nullable=False)
    valid_from = Column(DateTime(timezone=True), nullable=False)
    valid_until = Column(DateTime(timezone=True), nullable=True)
    revoked_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    __table_args__ = (
        CheckConstraint("state IN ('active','revoked','expired')", name="ck_trusted_role_binding_state"),
        Index("ix_trusted_role_bindings_principal_state", "principal_id", "state"),
    )

class TrustedDevice(Base):
    __tablename__ = "trusted_devices"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    principal_id = Column(UUID(as_uuid=True), ForeignKey("trusted_principals.id", ondelete="RESTRICT"), nullable=False)
    credential_id = Column(UUID(as_uuid=True), ForeignKey("trusted_credentials.id", ondelete="RESTRICT"), nullable=True)
    device_binding_hash = Column(String(64), nullable=True)
    state = Column(String(16), nullable=False)
    state_reason_code = Column(String(64), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow)
    __table_args__ = (
        CheckConstraint("state IN ('unknown','trusted','restricted','revoked')", name="ck_trusted_device_state"),
        Index("ix_trusted_devices_principal_state", "principal_id", "state"),
    )

class TrustedSession(Base):
    __tablename__ = "trusted_sessions"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    principal_id = Column(UUID(as_uuid=True), ForeignKey("trusted_principals.id", ondelete="RESTRICT"), nullable=False)
    browser_session_binding_hash = Column(String(64), nullable=False)
    state = Column(String(16), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    last_seen_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    revoked_at = Column(DateTime(timezone=True), nullable=True)
    __table_args__ = (
        CheckConstraint("state IN ('ordinary','elevated','expired','revoked')", name="ck_trusted_session_state"),
        Index("ix_trusted_sessions_principal_state", "principal_id", "state"),
    )

class TrustedElevationChallenge(Base):
    __tablename__ = "trusted_elevation_challenges"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    browser_session_binding_hash = Column(String(64), nullable=False)
    requested_operation_class = Column(String(96), nullable=False)
    required_capability = Column(String(128), nullable=False)
    resource_type = Column(String(64), nullable=False)
    resource_id = Column(String(128), nullable=False)
    resource_version = Column(Integer, nullable=False)
    return_target = Column(String(1024), nullable=False)
    state = Column(String(24), nullable=False)
    issued_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    consumed_at = Column(DateTime(timezone=True), nullable=True)
    __table_args__ = (
        CheckConstraint("state IN ('pending','redirected','verified','denied','expired','consumed')", name="ck_trusted_challenge_state"),
        CheckConstraint("resource_version >= 1", name="ck_trusted_challenge_resource_version"),
        Index("ix_trusted_challenges_state_expiry", "state", "expires_at"),
        Index("ix_trusted_challenges_resource_state", "resource_type", "resource_id", "state"),
    )

class TrustedAccessHandoff(Base):
    __tablename__ = "trusted_access_handoffs"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    challenge_id = Column(UUID(as_uuid=True), ForeignKey("trusted_elevation_challenges.id", ondelete="RESTRICT"), nullable=False)
    credential_id = Column(UUID(as_uuid=True), ForeignKey("trusted_credentials.id", ondelete="RESTRICT"), nullable=False)
    code_hash = Column(String(64), nullable=False, unique=True)
    audience = Column(String(128), nullable=False)
    browser_session_binding_hash = Column(String(64), nullable=False)
    state = Column(String(24), nullable=False)
    issued_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    consumed_at = Column(DateTime(timezone=True), nullable=True)
    __table_args__ = (
        CheckConstraint("state IN ('issued','consumed','expired','revoked')", name="ck_trusted_handoff_state"),
        Index("ix_trusted_handoffs_state_expiry", "state", "expires_at"),
    )

class TrustedElevationGrant(Base):
    __tablename__ = "trusted_elevation_grants"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trusted_session_id = Column(UUID(as_uuid=True), ForeignKey("trusted_sessions.id", ondelete="RESTRICT"), nullable=False)
    challenge_id = Column(UUID(as_uuid=True), ForeignKey("trusted_elevation_challenges.id", ondelete="RESTRICT"), nullable=False)
    principal_id = Column(UUID(as_uuid=True), ForeignKey("trusted_principals.id", ondelete="RESTRICT"), nullable=False)
    credential_id = Column(UUID(as_uuid=True), ForeignKey("trusted_credentials.id", ondelete="RESTRICT"), nullable=False)
    operation_class = Column(String(96), nullable=False)
    required_capability = Column(String(128), nullable=False)
    resource_type = Column(String(64), nullable=False)
    resource_id = Column(String(128), nullable=False)
    resource_version = Column(Integer, nullable=False)
    policy_version = Column(String(64), nullable=False)
    state = Column(String(24), nullable=False)
    issued_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    revoked_at = Column(DateTime(timezone=True), nullable=True)
    consumed_at = Column(DateTime(timezone=True), nullable=True)
    reason_code = Column(String(64), nullable=True)
    __table_args__ = (
        CheckConstraint("state IN ('pending','active','expired','revoked','consumed')", name="ck_trusted_grant_state"),
        CheckConstraint("resource_version >= 1", name="ck_trusted_grant_resource_version"),
        Index("ix_trusted_grants_principal_state", "principal_id", "state"),
        Index("ix_trusted_grants_resource_state", "resource_type", "resource_id", "state"),
    )

class TrustedTrustEvent(Base):
    __tablename__ = "trusted_trust_events"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    principal_id = Column(UUID(as_uuid=True), ForeignKey("trusted_principals.id", ondelete="RESTRICT"), nullable=True)
    actor_principal_id = Column(UUID(as_uuid=True), ForeignKey("trusted_principals.id", ondelete="RESTRICT"), nullable=True)
    credential_id = Column(UUID(as_uuid=True), ForeignKey("trusted_credentials.id", ondelete="RESTRICT"), nullable=True)
    device_id = Column(UUID(as_uuid=True), ForeignKey("trusted_devices.id", ondelete="RESTRICT"), nullable=True)
    session_id = Column(UUID(as_uuid=True), nullable=True)
    challenge_id = Column(UUID(as_uuid=True), nullable=True)
    grant_id = Column(UUID(as_uuid=True), nullable=True)
    event_type = Column(String(64), nullable=False)
    reason_code = Column(String(64), nullable=True)
    policy_version = Column(String(64), nullable=True)
    event_metadata = Column(TRUST_JSON, nullable=False, default=dict)
    previous_event_hash = Column(String(64), nullable=True)
    event_hash = Column(String(64), nullable=False, unique=True)
    occurred_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    __table_args__ = (
        Index("ix_trusted_events_principal_time", "principal_id", "occurred_at"),
        Index("ix_trusted_events_type_time", "event_type", "occurred_at"),
    )
