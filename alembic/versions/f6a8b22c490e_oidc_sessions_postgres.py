"""Dedicated Keycloak OIDC session table, on existing Azure PostgreSQL.

Revision ID: f6a8b22c490e
Revises: d2e33e73c6d2

This migration deliberately does NOT create database users or databases.
The DBA-controlled login role `electionpulse_oidc_session` must exist before
an approved schema apply, with tightly constrained PostgreSQL role flags.
"""
from __future__ import annotations

from typing import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "f6a8b22c490e"
down_revision: str | None = "d2e33e73c6d2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SESSION_ROLE = "electionpulse_oidc_session"


def upgrade() -> None:
    # Production is PostgreSQL; other backends aren't an authorized deployment.
    if op.get_bind().dialect.name != "postgresql":
        raise RuntimeError("OIDC sessions migration requires PostgreSQL")
    op.create_table(
        "electionpulse_oidc_sessions",
        sa.Column("key_sha256", sa.CHAR(length=64), primary_key=True),
        sa.Column("value_json", sa.Text(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
        sa.CheckConstraint("key_sha256 ~ '^[0-9a-f]{64}$'",
                           name="ck_oidc_session_key_sha256"),
        sa.CheckConstraint("octet_length(value_json) BETWEEN 1 AND 2048",
                           name="ck_oidc_session_value_length"),
        schema="public",
    )
    op.create_index("electionpulse_oidc_sessions_expiry_idx",
                    "electionpulse_oidc_sessions", ["expires_at"], unique=False,
                    schema="public")
    # No CREATE/ALTER/TRUNCATE/UPDATE on the session table for this login.
    # Existing governed preflight validates role safety before applying.
    op.execute("REVOKE ALL ON TABLE public.electionpulse_oidc_sessions FROM PUBLIC")
    op.execute("GRANT USAGE ON SCHEMA public TO electionpulse_oidc_session")
    op.execute("GRANT SELECT, INSERT, DELETE ON TABLE public.electionpulse_oidc_sessions TO electionpulse_oidc_session")


def downgrade() -> None:
    # Session tables must never be dropped automatically by release rollback.
    raise RuntimeError("OIDC session schema rollback requires a separate DBA review")
