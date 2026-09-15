"""Bind trusted elevation to an exact resource and make grants single-use.

Revision ID: a19c7e4b2d60
Revises: f18a7c3d4e92
Create Date: 2026-09-15
"""
from __future__ import annotations
from typing import Sequence
from alembic import op
import sqlalchemy as sa

revision: str = "a19c7e4b2d60"
down_revision: str | None = "f18a7c3d4e92"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("trusted_elevation_challenges", sa.Column("resource_type", sa.String(length=64), nullable=False))
    op.add_column("trusted_elevation_challenges", sa.Column("resource_id", sa.String(length=128), nullable=False))
    op.add_column("trusted_elevation_challenges", sa.Column("resource_version", sa.Integer(), nullable=False))
    op.create_check_constraint("ck_trusted_challenge_resource_version", "trusted_elevation_challenges", "resource_version >= 1")
    op.create_index("ix_trusted_challenges_resource_state", "trusted_elevation_challenges", ["resource_type", "resource_id", "state"], unique=False)

    op.add_column("trusted_elevation_grants", sa.Column("resource_type", sa.String(length=64), nullable=False))
    op.add_column("trusted_elevation_grants", sa.Column("resource_id", sa.String(length=128), nullable=False))
    op.add_column("trusted_elevation_grants", sa.Column("resource_version", sa.Integer(), nullable=False))
    op.add_column("trusted_elevation_grants", sa.Column("consumed_at", sa.DateTime(timezone=True), nullable=True))
    op.create_check_constraint("ck_trusted_grant_resource_version", "trusted_elevation_grants", "resource_version >= 1")
    op.create_index("ix_trusted_grants_resource_state", "trusted_elevation_grants", ["resource_type", "resource_id", "state"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_trusted_grants_resource_state", table_name="trusted_elevation_grants")
    op.drop_constraint("ck_trusted_grant_resource_version", "trusted_elevation_grants", type_="check")
    op.drop_column("trusted_elevation_grants", "consumed_at")
    op.drop_column("trusted_elevation_grants", "resource_version")
    op.drop_column("trusted_elevation_grants", "resource_id")
    op.drop_column("trusted_elevation_grants", "resource_type")
    op.drop_index("ix_trusted_challenges_resource_state", table_name="trusted_elevation_challenges")
    op.drop_constraint("ck_trusted_challenge_resource_version", "trusted_elevation_challenges", type_="check")
    op.drop_column("trusted_elevation_challenges", "resource_version")
    op.drop_column("trusted_elevation_challenges", "resource_id")
    op.drop_column("trusted_elevation_challenges", "resource_type")
