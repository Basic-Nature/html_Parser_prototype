"""J1 project-owned investigation context; no Registry/Workflow/execution mutation.

Revision ID: ab8a7b16e24f
Revises: d2e33e73c6d2
"""
from __future__ import annotations
from alembic import op
import sqlalchemy as sa

revision = "ab8a7b16e24f"
down_revision = "d2e33e73c6d2"
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.create_table(
        "election_projects",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("owner_key", sa.String(64), nullable=False),
        sa.Column("creation_key", sa.Uuid(), nullable=False),
        sa.Column("title", sa.String(160), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("lifecycle", sa.String(16), nullable=False),
        sa.Column("row_version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("lifecycle IN ('active','archived')", name="ck_election_project_lifecycle"),
        sa.CheckConstraint("row_version >= 1", name="ck_election_project_version"),
        sa.UniqueConstraint("owner_key", "creation_key", name="uq_election_project_create"),
    )
    op.create_index("ix_election_projects_owner_update", "election_projects", ["owner_key", "updated_at"])
    op.create_table(
        "election_project_scopes",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("election_year", sa.Integer(), nullable=True),
        sa.Column("election_date", sa.Date(), nullable=True),
        sa.Column("state_code", sa.String(2), nullable=True),
        sa.Column("jurisdiction", sa.String(160), nullable=True),
        sa.Column("contest", sa.String(200), nullable=True),
        sa.Column("granularity", sa.String(24), nullable=True),
        sa.CheckConstraint("election_year IS NULL OR (election_year BETWEEN 1788 AND 2100)", name="ck_election_project_scope_year"),
        sa.CheckConstraint("granularity IS NULL OR granularity IN ('statewide','county','municipality','precinct','unknown')", name="ck_election_project_scope_granularity"),
        sa.ForeignKeyConstraint(["project_id"], ["election_projects.id"], name="fk_election_project_scope_project", ondelete="CASCADE"),
    )
    op.create_index("ix_election_project_scopes_project", "election_project_scopes", ["project_id"])
    op.create_table(
        "election_project_source_refs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("registry_binding_id", sa.Uuid(), nullable=False),
        sa.Column("registry_revision_id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["election_projects.id"], name="fk_election_project_ref_project", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["registry_binding_id"], ["source_registry_bindings.id"], name="fk_election_project_ref_binding", ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["registry_revision_id"], ["source_registry_revisions.id"], name="fk_election_project_ref_revision", ondelete="RESTRICT"),
        sa.UniqueConstraint("project_id", "registry_binding_id", "registry_revision_id", name="uq_election_project_ref_revision"),
    )
    op.create_index("ix_election_project_refs_project", "election_project_source_refs", ["project_id"])

def downgrade() -> None:
    op.drop_index("ix_election_project_refs_project", table_name="election_project_source_refs")
    op.drop_table("election_project_source_refs")
    op.drop_index("ix_election_project_scopes_project", table_name="election_project_scopes")
    op.drop_table("election_project_scopes")
    op.drop_index("ix_election_projects_owner_update", table_name="election_projects")
    op.drop_table("election_projects")
