"""Add trusted identity authority persistence foundation.

Revision ID: f18a7c3d4e92
Revises: e7b2c4d91f60
Create Date: 2026-09-13

The trusted_* tables hold ElectionPulse trust authority and bounded
authentication/elevation state. They do not write canonical election truth.
Existing certificate fingerprints are not migrated automatically and do not
implicitly create principals, roles, or capabilities.
"""

from __future__ import annotations

from typing import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "f18a7c3d4e92"
down_revision: str | None = "e7b2c4d91f60"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _trust_json_type() -> sa.types.TypeEngine:
    """Portable JSON in migration SQL; PostgreSQL materializes native JSONB."""
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        from sqlalchemy.dialects.postgresql import JSONB

        return JSONB()
    return sa.JSON()


def upgrade() -> None:
    trust_json = _trust_json_type()

    op.create_table(
        "trusted_principals",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("principal_type", sa.String(length=16), nullable=False),
        sa.Column("state", sa.String(length=16), nullable=False),
        sa.Column("state_reason_code", sa.String(length=64), nullable=False),
        sa.Column("principal_metadata", trust_json, nullable=False),
        sa.Column("row_version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "principal_type IN ('human','service')",
            name="ck_trusted_principal_type",
        ),
        sa.CheckConstraint(
            "state IN ('pending','trusted','restricted','suspended','revoked')",
            name="ck_trusted_principal_state",
        ),
        sa.CheckConstraint(
            "row_version >= 1",
            name="ck_trusted_principal_row_version",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_trusted_principals"),
    )
    op.create_index(
        "ix_trusted_principals_type_state",
        "trusted_principals",
        ["principal_type", "state"],
        unique=False,
    )

    op.create_table(
        "trusted_identity_bindings",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("principal_id", sa.UUID(), nullable=False),
        sa.Column("provider", sa.String(length=48), nullable=False),
        sa.Column("provider_subject_hash", sa.String(length=64), nullable=False),
        sa.Column("subject_hash_version", sa.String(length=32), nullable=False),
        sa.Column("state", sa.String(length=16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("disabled_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "state IN ('active','disabled')",
            name="ck_trusted_identity_binding_state",
        ),
        sa.ForeignKeyConstraint(
            ["principal_id"],
            ["trusted_principals.id"],
            name="fk_trusted_identity_bindings_principal_id",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_trusted_identity_bindings"),
        sa.UniqueConstraint(
            "provider",
            "provider_subject_hash",
            "subject_hash_version",
            name="uq_trusted_identity_provider_subject_hash",
        ),
    )
    op.create_index(
        "ix_trusted_identity_bindings_principal",
        "trusted_identity_bindings",
        ["principal_id"],
        unique=False,
    )

    op.create_table(
        "trusted_credentials",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("principal_id", sa.UUID(), nullable=False),
        sa.Column("credential_type", sa.String(length=24), nullable=False),
        sa.Column("fingerprint_sha256", sa.String(length=64), nullable=False),
        sa.Column("state", sa.String(length=16), nullable=False),
        sa.Column("not_before", sa.DateTime(timezone=True), nullable=True),
        sa.Column("not_after", sa.DateTime(timezone=True), nullable=True),
        sa.Column("issuer_metadata", trust_json, nullable=False),
        sa.Column("serial_metadata", trust_json, nullable=False),
        sa.Column("chain_validation_state", sa.String(length=32), nullable=False),
        sa.Column("revocation_state", sa.String(length=32), nullable=False),
        sa.Column("superseded_by_credential_id", sa.UUID(), nullable=True),
        sa.Column("row_version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "state IN ('pending','active','superseded','expired','revoked','untrusted')",
            name="ck_trusted_credential_state",
        ),
        sa.CheckConstraint(
            "row_version >= 1",
            name="ck_trusted_credential_row_version",
        ),
        sa.ForeignKeyConstraint(
            ["principal_id"],
            ["trusted_principals.id"],
            name="fk_trusted_credentials_principal_id",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["superseded_by_credential_id"],
            ["trusted_credentials.id"],
            name="fk_trusted_credentials_superseded_by",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_trusted_credentials"),
        sa.UniqueConstraint(
            "credential_type",
            "fingerprint_sha256",
            name="uq_trusted_credential_type_fingerprint",
        ),
    )
    op.create_index(
        "ix_trusted_credentials_principal_state",
        "trusted_credentials",
        ["principal_id", "state"],
        unique=False,
    )

    op.create_table(
        "trusted_role_bindings",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("principal_id", sa.UUID(), nullable=False),
        sa.Column("role_name", sa.String(length=96), nullable=False),
        sa.Column("state", sa.String(length=16), nullable=False),
        sa.Column("granted_by_principal_id", sa.UUID(), nullable=True),
        sa.Column("reason_code", sa.String(length=64), nullable=False),
        sa.Column("valid_from", sa.DateTime(timezone=True), nullable=False),
        sa.Column("valid_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "state IN ('active','revoked','expired')",
            name="ck_trusted_role_binding_state",
        ),
        sa.ForeignKeyConstraint(
            ["principal_id"],
            ["trusted_principals.id"],
            name="fk_trusted_role_bindings_principal_id",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["granted_by_principal_id"],
            ["trusted_principals.id"],
            name="fk_trusted_role_bindings_granted_by_principal_id",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_trusted_role_bindings"),
    )
    op.create_index(
        "ix_trusted_role_bindings_principal_state",
        "trusted_role_bindings",
        ["principal_id", "state"],
        unique=False,
    )

    op.create_table(
        "trusted_devices",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("principal_id", sa.UUID(), nullable=False),
        sa.Column("credential_id", sa.UUID(), nullable=True),
        sa.Column("device_binding_hash", sa.String(length=64), nullable=True),
        sa.Column("state", sa.String(length=16), nullable=False),
        sa.Column("state_reason_code", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "state IN ('unknown','trusted','restricted','revoked')",
            name="ck_trusted_device_state",
        ),
        sa.ForeignKeyConstraint(
            ["principal_id"],
            ["trusted_principals.id"],
            name="fk_trusted_devices_principal_id",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["credential_id"],
            ["trusted_credentials.id"],
            name="fk_trusted_devices_credential_id",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_trusted_devices"),
    )
    op.create_index(
        "ix_trusted_devices_principal_state",
        "trusted_devices",
        ["principal_id", "state"],
        unique=False,
    )

    op.create_table(
        "trusted_sessions",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("principal_id", sa.UUID(), nullable=False),
        sa.Column("browser_session_binding_hash", sa.String(length=64), nullable=False),
        sa.Column("state", sa.String(length=16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "state IN ('ordinary','elevated','expired','revoked')",
            name="ck_trusted_session_state",
        ),
        sa.ForeignKeyConstraint(
            ["principal_id"],
            ["trusted_principals.id"],
            name="fk_trusted_sessions_principal_id",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_trusted_sessions"),
    )
    op.create_index(
        "ix_trusted_sessions_principal_state",
        "trusted_sessions",
        ["principal_id", "state"],
        unique=False,
    )

    op.create_table(
        "trusted_elevation_challenges",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("browser_session_binding_hash", sa.String(length=64), nullable=False),
        sa.Column("requested_operation_class", sa.String(length=96), nullable=False),
        sa.Column("required_capability", sa.String(length=128), nullable=False),
        sa.Column("return_target", sa.String(length=1024), nullable=False),
        sa.Column("state", sa.String(length=24), nullable=False),
        sa.Column("issued_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("consumed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "state IN ('pending','redirected','verified','denied','expired','consumed')",
            name="ck_trusted_challenge_state",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_trusted_elevation_challenges"),
    )
    op.create_index(
        "ix_trusted_challenges_state_expiry",
        "trusted_elevation_challenges",
        ["state", "expires_at"],
        unique=False,
    )

    op.create_table(
        "trusted_access_handoffs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("challenge_id", sa.UUID(), nullable=False),
        sa.Column("credential_id", sa.UUID(), nullable=False),
        sa.Column("code_hash", sa.String(length=64), nullable=False),
        sa.Column("audience", sa.String(length=128), nullable=False),
        sa.Column("browser_session_binding_hash", sa.String(length=64), nullable=False),
        sa.Column("state", sa.String(length=24), nullable=False),
        sa.Column("issued_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("consumed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "state IN ('issued','consumed','expired','revoked')",
            name="ck_trusted_handoff_state",
        ),
        sa.ForeignKeyConstraint(
            ["challenge_id"],
            ["trusted_elevation_challenges.id"],
            name="fk_trusted_access_handoffs_challenge_id",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["credential_id"],
            ["trusted_credentials.id"],
            name="fk_trusted_access_handoffs_credential_id",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_trusted_access_handoffs"),
        sa.UniqueConstraint("code_hash", name="uq_trusted_access_handoff_code_hash"),
    )
    op.create_index(
        "ix_trusted_handoffs_state_expiry",
        "trusted_access_handoffs",
        ["state", "expires_at"],
        unique=False,
    )

    op.create_table(
        "trusted_elevation_grants",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("trusted_session_id", sa.UUID(), nullable=False),
        sa.Column("challenge_id", sa.UUID(), nullable=False),
        sa.Column("principal_id", sa.UUID(), nullable=False),
        sa.Column("credential_id", sa.UUID(), nullable=False),
        sa.Column("operation_class", sa.String(length=96), nullable=False),
        sa.Column("required_capability", sa.String(length=128), nullable=False),
        sa.Column("policy_version", sa.String(length=64), nullable=False),
        sa.Column("state", sa.String(length=24), nullable=False),
        sa.Column("issued_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("reason_code", sa.String(length=64), nullable=True),
        sa.CheckConstraint(
            "state IN ('pending','active','expired','revoked','consumed')",
            name="ck_trusted_grant_state",
        ),
        sa.ForeignKeyConstraint(
            ["trusted_session_id"],
            ["trusted_sessions.id"],
            name="fk_trusted_elevation_grants_session_id",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["challenge_id"],
            ["trusted_elevation_challenges.id"],
            name="fk_trusted_elevation_grants_challenge_id",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["principal_id"],
            ["trusted_principals.id"],
            name="fk_trusted_elevation_grants_principal_id",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["credential_id"],
            ["trusted_credentials.id"],
            name="fk_trusted_elevation_grants_credential_id",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_trusted_elevation_grants"),
    )
    op.create_index(
        "ix_trusted_grants_principal_state",
        "trusted_elevation_grants",
        ["principal_id", "state"],
        unique=False,
    )

    op.create_table(
        "trusted_trust_events",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("principal_id", sa.UUID(), nullable=True),
        sa.Column("actor_principal_id", sa.UUID(), nullable=True),
        sa.Column("credential_id", sa.UUID(), nullable=True),
        sa.Column("device_id", sa.UUID(), nullable=True),
        sa.Column("session_id", sa.UUID(), nullable=True),
        sa.Column("challenge_id", sa.UUID(), nullable=True),
        sa.Column("grant_id", sa.UUID(), nullable=True),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("reason_code", sa.String(length=64), nullable=True),
        sa.Column("policy_version", sa.String(length=64), nullable=True),
        sa.Column("event_metadata", trust_json, nullable=False),
        sa.Column("previous_event_hash", sa.String(length=64), nullable=True),
        sa.Column("event_hash", sa.String(length=64), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["principal_id"],
            ["trusted_principals.id"],
            name="fk_trusted_trust_events_principal_id",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["actor_principal_id"],
            ["trusted_principals.id"],
            name="fk_trusted_trust_events_actor_principal_id",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["credential_id"],
            ["trusted_credentials.id"],
            name="fk_trusted_trust_events_credential_id",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["device_id"],
            ["trusted_devices.id"],
            name="fk_trusted_trust_events_device_id",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_trusted_trust_events"),
        sa.UniqueConstraint("event_hash", name="uq_trusted_trust_event_hash"),
    )
    op.create_index(
        "ix_trusted_events_principal_time",
        "trusted_trust_events",
        ["principal_id", "occurred_at"],
        unique=False,
    )
    op.create_index(
        "ix_trusted_events_type_time",
        "trusted_trust_events",
        ["event_type", "occurred_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_table("trusted_trust_events")
    op.drop_table("trusted_elevation_grants")
    op.drop_table("trusted_access_handoffs")
    op.drop_table("trusted_elevation_challenges")
    op.drop_table("trusted_sessions")
    op.drop_table("trusted_devices")
    op.drop_table("trusted_role_bindings")
    op.drop_table("trusted_credentials")
    op.drop_table("trusted_identity_bindings")
    op.drop_table("trusted_principals")
