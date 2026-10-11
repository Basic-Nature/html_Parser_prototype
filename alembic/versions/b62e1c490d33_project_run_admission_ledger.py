"""J2B1 admission-only run ledger: no dispatcher, evidence writer or execution.

Revision ID: b62e1c490d33
Revises: ab8a7b16e24f
"""
from __future__ import annotations
from alembic import op
import sqlalchemy as sa

revision = "b62e1c490d33"
down_revision = "ab8a7b16e24f"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "election_project_runs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("owner_key", sa.String(64), nullable=False),
        sa.Column("idempotency_key", sa.Uuid(), nullable=False),
        sa.Column("request_fingerprint", sa.String(64), nullable=False),
        sa.Column("source_ref_id", sa.Uuid(), nullable=False),
        sa.Column("registry_binding_id", sa.Uuid(), nullable=False),
        sa.Column("registry_revision_id", sa.Uuid(), nullable=False),
        sa.Column("workflow_item_id", sa.Uuid(), nullable=False),
        sa.Column("workflow_pass_id", sa.Uuid(), nullable=False),
        sa.Column("project_row_version", sa.Integer(), nullable=False),
        sa.Column("workflow_row_version", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("row_version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["project_id"],["election_projects.id"],name="fk_project_run_project",ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["source_ref_id"],["election_project_source_refs.id"],name="fk_project_run_source_ref",ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["registry_binding_id"],["source_registry_bindings.id"],name="fk_project_run_binding",ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["registry_revision_id"],["source_registry_revisions.id"],name="fk_project_run_revision",ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["workflow_item_id"],["workflow_items.id"],name="fk_project_run_workflow_item",ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["workflow_pass_id"],["workflow_passes.id"],name="fk_project_run_workflow_pass",ondelete="RESTRICT"),
        sa.UniqueConstraint("project_id","owner_key","idempotency_key",name="uq_election_project_run_intent"),
        sa.CheckConstraint("status = 'admitted'",name="ck_election_project_run_status"),
        sa.CheckConstraint("row_version >= 1 AND project_row_version >= 1 AND workflow_row_version >= 0",name="ck_election_project_run_versions"),
    )
    op.create_index("ix_election_project_runs_project_created","election_project_runs",["project_id","created_at"])


def downgrade() -> None:
    op.drop_index("ix_election_project_runs_project_created",table_name="election_project_runs")
    op.drop_table("election_project_runs")
