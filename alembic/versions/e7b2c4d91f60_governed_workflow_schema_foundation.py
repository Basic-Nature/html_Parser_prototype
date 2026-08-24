"""Add governed noncanonical workflow schema foundation.

Revision ID: e7b2c4d91f60
Revises: c2a3f7e91b4d
Create Date: 2026-08-23

The workflow_* tables are operational review/process state. They are not an
alternate election-result authority and do not write canonical election truth.
"""

from __future__ import annotations

from typing import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "e7b2c4d91f60"
down_revision: str | None = "c2a3f7e91b4d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _workflow_json_type() -> sa.types.TypeEngine:
    """Portable JSON in migration SQL; PostgreSQL materializes native JSONB."""
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        from sqlalchemy.dialects.postgresql import JSONB

        return JSONB()
    return sa.JSON()


def upgrade() -> None:
    workflow_json = _workflow_json_type()

    op.create_table(
        "workflow_items",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("lifecycle_state", sa.String(length=32), nullable=False),
        sa.Column("current_stage", sa.String(length=48), nullable=False),
        sa.Column("stage_condition", sa.String(length=32), nullable=False),
        sa.Column("priority", sa.Integer(), nullable=False),
        sa.Column("election_year", sa.Integer(), nullable=True),
        sa.Column("election_date", sa.Date(), nullable=True),
        sa.Column("state", sa.String(length=64), nullable=True),
        sa.Column("jurisdiction_name", sa.String(length=256), nullable=True),
        sa.Column("jurisdiction_type", sa.String(length=32), nullable=True),
        sa.Column("contest", sa.String(length=256), nullable=True),
        sa.Column("office_basic", sa.String(length=64), nullable=True),
        sa.Column("election_type", sa.String(length=64), nullable=True),
        sa.Column("source_race_id", sa.String(length=128), nullable=True),
        sa.Column("source_url", sa.String(length=2048), nullable=True),
        sa.Column("canonical_race_id", sa.UUID(), nullable=True),
        sa.Column("blocked_reason_code", sa.String(length=64), nullable=True),
        sa.Column("blocker_detail", sa.Text(), nullable=True),
        sa.Column("created_by_principal", sa.String(length=256), nullable=True),
        sa.Column("workflow_metadata", workflow_json, nullable=False),
        sa.Column("row_version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "priority >= 0",
            name="ck_workflow_items_priority_nonnegative",
        ),
        sa.CheckConstraint(
            "row_version >= 1",
            name="ck_workflow_items_row_version_positive",
        ),
        sa.ForeignKeyConstraint(
            ["canonical_race_id"],
            ["canonical_election_races.id"],
            name="fk_workflow_items_canonical_race_id",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_workflow_items"),
    )
    op.create_index(
        "ix_workflow_items_lifecycle_stage",
        "workflow_items",
        ["lifecycle_state", "current_stage", "stage_condition"],
        unique=False,
    )
    op.create_index(
        "ix_workflow_items_year_state",
        "workflow_items",
        ["election_year", "state"],
        unique=False,
    )
    op.create_index(
        "ix_workflow_items_canonical_race",
        "workflow_items",
        ["canonical_race_id"],
        unique=False,
    )

    op.create_table(
        "workflow_passes",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("workflow_item_id", sa.UUID(), nullable=False),
        sa.Column("pass_number", sa.Integer(), nullable=False),
        sa.Column("pass_label", sa.String(length=16), nullable=False),
        sa.Column("revision_number", sa.Integer(), nullable=False),
        sa.Column("is_current", sa.Boolean(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("assigned_principal", sa.String(length=256), nullable=True),
        sa.Column("source_evidence_ref", sa.String(length=512), nullable=True),
        sa.Column("staging_batch_id", sa.UUID(), nullable=True),
        sa.Column("candidate_check_status", sa.String(length=32), nullable=True),
        sa.Column("candidate_check_result", workflow_json, nullable=True),
        sa.Column("semantic_validation_status", sa.String(length=32), nullable=True),
        sa.Column("semantic_validation_result", workflow_json, nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("superseded_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "pass_number >= 1",
            name="ck_workflow_pass_number_positive",
        ),
        sa.CheckConstraint(
            "revision_number >= 1",
            name="ck_workflow_pass_revision_positive",
        ),
        sa.ForeignKeyConstraint(
            ["workflow_item_id"],
            ["workflow_items.id"],
            name="fk_workflow_passes_workflow_item_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["staging_batch_id"],
            ["batch_metadata.batch_id"],
            name="fk_workflow_passes_staging_batch_id",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_workflow_passes"),
        sa.UniqueConstraint(
            "workflow_item_id",
            "pass_number",
            "revision_number",
            name="uq_workflow_pass_item_number_revision",
        ),
    )
    op.create_index(
        "ix_workflow_pass_item_current",
        "workflow_passes",
        ["workflow_item_id", "is_current"],
        unique=False,
    )
    op.create_index(
        "ix_workflow_pass_status",
        "workflow_passes",
        ["status"],
        unique=False,
    )
    op.create_index(
        "ix_workflow_pass_assignee",
        "workflow_passes",
        ["assigned_principal"],
        unique=False,
    )

    op.create_table(
        "workflow_comparisons",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("workflow_item_id", sa.UUID(), nullable=False),
        sa.Column("left_pass_id", sa.UUID(), nullable=False),
        sa.Column("right_pass_id", sa.UUID(), nullable=False),
        sa.Column("comparison_version", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("strict_equality_passed", sa.Boolean(), nullable=True),
        sa.Column("difference_count", sa.Integer(), nullable=True),
        sa.Column("difference_summary", workflow_json, nullable=True),
        sa.Column("checked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("checked_by_service_version", sa.String(length=128), nullable=True),
        sa.Column("reviewed_by_principal", sa.String(length=256), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "left_pass_id <> right_pass_id",
            name="ck_workflow_comparison_distinct_passes",
        ),
        sa.CheckConstraint(
            "comparison_version >= 1",
            name="ck_workflow_comparison_version_positive",
        ),
        sa.CheckConstraint(
            "difference_count IS NULL OR difference_count >= 0",
            name="ck_workflow_comparison_difference_count_nonnegative",
        ),
        sa.ForeignKeyConstraint(
            ["workflow_item_id"],
            ["workflow_items.id"],
            name="fk_workflow_comparisons_workflow_item_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["left_pass_id"],
            ["workflow_passes.id"],
            name="fk_workflow_comparisons_left_pass_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["right_pass_id"],
            ["workflow_passes.id"],
            name="fk_workflow_comparisons_right_pass_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_workflow_comparisons"),
        sa.UniqueConstraint(
            "workflow_item_id",
            "left_pass_id",
            "right_pass_id",
            "comparison_version",
            name="uq_workflow_comparison_pair_version",
        ),
    )
    op.create_index(
        "ix_workflow_comparison_item_status",
        "workflow_comparisons",
        ["workflow_item_id", "status"],
        unique=False,
    )

    op.create_table(
        "workflow_discrepancies",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("comparison_id", sa.UUID(), nullable=False),
        sa.Column("workflow_item_id", sa.UUID(), nullable=False),
        sa.Column("category", sa.String(length=64), nullable=False),
        sa.Column("semantic_key", workflow_json, nullable=False),
        sa.Column("left_value", workflow_json, nullable=True),
        sa.Column("right_value", workflow_json, nullable=True),
        sa.Column("left_value_state", sa.String(length=32), nullable=True),
        sa.Column("right_value_state", sa.String(length=32), nullable=True),
        sa.Column("severity", sa.String(length=32), nullable=True),
        sa.Column("resolution_status", sa.String(length=32), nullable=False),
        sa.Column("resolution_code", sa.String(length=64), nullable=True),
        sa.Column("resolution_notes", sa.Text(), nullable=True),
        sa.Column("resolved_by_principal", sa.String(length=256), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["comparison_id"],
            ["workflow_comparisons.id"],
            name="fk_workflow_discrepancies_comparison_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["workflow_item_id"],
            ["workflow_items.id"],
            name="fk_workflow_discrepancies_workflow_item_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_workflow_discrepancies"),
    )
    op.create_index(
        "ix_workflow_discrepancy_item_status",
        "workflow_discrepancies",
        ["workflow_item_id", "resolution_status"],
        unique=False,
    )
    op.create_index(
        "ix_workflow_discrepancy_comparison",
        "workflow_discrepancies",
        ["comparison_id"],
        unique=False,
    )

    op.create_table(
        "workflow_reviews",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("workflow_item_id", sa.UUID(), nullable=False),
        sa.Column("review_stage", sa.String(length=48), nullable=False),
        sa.Column("reviewer_principal", sa.String(length=256), nullable=False),
        sa.Column("decision", sa.String(length=32), nullable=False),
        sa.Column("selected_pass_id", sa.UUID(), nullable=True),
        sa.Column("selected_staging_batch_id", sa.UUID(), nullable=True),
        sa.Column("checklist_version", sa.String(length=64), nullable=True),
        sa.Column("checklist_result", workflow_json, nullable=True),
        sa.Column("reason_codes", workflow_json, nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["workflow_item_id"],
            ["workflow_items.id"],
            name="fk_workflow_reviews_workflow_item_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["selected_pass_id"],
            ["workflow_passes.id"],
            name="fk_workflow_reviews_selected_pass_id",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["selected_staging_batch_id"],
            ["batch_metadata.batch_id"],
            name="fk_workflow_reviews_selected_staging_batch_id",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_workflow_reviews"),
    )
    op.create_index(
        "ix_workflow_review_item_stage",
        "workflow_reviews",
        ["workflow_item_id", "review_stage"],
        unique=False,
    )
    op.create_index(
        "ix_workflow_review_decision",
        "workflow_reviews",
        ["decision"],
        unique=False,
    )
    op.create_index(
        "ix_workflow_review_principal",
        "workflow_reviews",
        ["reviewer_principal"],
        unique=False,
    )

    op.create_table(
        "workflow_artifact_links",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("workflow_item_id", sa.UUID(), nullable=False),
        sa.Column("pass_id", sa.UUID(), nullable=True),
        sa.Column("relation_type", sa.String(length=64), nullable=False),
        sa.Column("artifact_type", sa.String(length=64), nullable=False),
        sa.Column("artifact_ref", sa.String(length=512), nullable=False),
        sa.Column("artifact_sha256", sa.String(length=64), nullable=True),
        sa.Column("canonical_source_artifact_id", sa.UUID(), nullable=True),
        sa.Column("staging_batch_id", sa.UUID(), nullable=True),
        sa.Column("artifact_metadata", workflow_json, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["workflow_item_id"],
            ["workflow_items.id"],
            name="fk_workflow_artifact_links_workflow_item_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["pass_id"],
            ["workflow_passes.id"],
            name="fk_workflow_artifact_links_pass_id",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["canonical_source_artifact_id"],
            ["canonical_source_artifacts.id"],
            name="fk_workflow_artifact_links_canonical_source_artifact_id",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["staging_batch_id"],
            ["batch_metadata.batch_id"],
            name="fk_workflow_artifact_links_staging_batch_id",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_workflow_artifact_links"),
    )
    op.create_index(
        "ix_workflow_artifact_item_relation",
        "workflow_artifact_links",
        ["workflow_item_id", "relation_type"],
        unique=False,
    )
    op.create_index(
        "ix_workflow_artifact_pass",
        "workflow_artifact_links",
        ["pass_id"],
        unique=False,
    )

    op.create_table(
        "workflow_events",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("workflow_item_id", sa.UUID(), nullable=False),
        sa.Column("actor_type", sa.String(length=32), nullable=False),
        sa.Column("actor_principal", sa.String(length=256), nullable=True),
        sa.Column("actor_service", sa.String(length=128), nullable=True),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("stage", sa.String(length=48), nullable=True),
        sa.Column("prior_state", workflow_json, nullable=True),
        sa.Column("new_state", workflow_json, nullable=True),
        sa.Column("related_pass_id", sa.UUID(), nullable=True),
        sa.Column("related_comparison_id", sa.UUID(), nullable=True),
        sa.Column("related_review_id", sa.UUID(), nullable=True),
        sa.Column("related_staging_batch_id", sa.UUID(), nullable=True),
        sa.Column("related_canonical_race_id", sa.UUID(), nullable=True),
        sa.Column("reason_code", sa.String(length=64), nullable=True),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("event_metadata", workflow_json, nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["workflow_item_id"],
            ["workflow_items.id"],
            name="fk_workflow_events_workflow_item_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["related_pass_id"],
            ["workflow_passes.id"],
            name="fk_workflow_events_related_pass_id",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["related_comparison_id"],
            ["workflow_comparisons.id"],
            name="fk_workflow_events_related_comparison_id",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["related_review_id"],
            ["workflow_reviews.id"],
            name="fk_workflow_events_related_review_id",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["related_staging_batch_id"],
            ["batch_metadata.batch_id"],
            name="fk_workflow_events_related_staging_batch_id",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["related_canonical_race_id"],
            ["canonical_election_races.id"],
            name="fk_workflow_events_related_canonical_race_id",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_workflow_events"),
    )
    op.create_index(
        "ix_workflow_event_item_time",
        "workflow_events",
        ["workflow_item_id", "occurred_at"],
        unique=False,
    )
    op.create_index(
        "ix_workflow_event_type",
        "workflow_events",
        ["event_type"],
        unique=False,
    )
    op.create_index(
        "ix_workflow_event_actor",
        "workflow_events",
        ["actor_principal"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_workflow_event_actor", table_name="workflow_events")
    op.drop_index("ix_workflow_event_type", table_name="workflow_events")
    op.drop_index("ix_workflow_event_item_time", table_name="workflow_events")
    op.drop_table("workflow_events")

    op.drop_index("ix_workflow_artifact_pass", table_name="workflow_artifact_links")
    op.drop_index(
        "ix_workflow_artifact_item_relation",
        table_name="workflow_artifact_links",
    )
    op.drop_table("workflow_artifact_links")

    op.drop_index("ix_workflow_review_principal", table_name="workflow_reviews")
    op.drop_index("ix_workflow_review_decision", table_name="workflow_reviews")
    op.drop_index("ix_workflow_review_item_stage", table_name="workflow_reviews")
    op.drop_table("workflow_reviews")

    op.drop_index(
        "ix_workflow_discrepancy_comparison",
        table_name="workflow_discrepancies",
    )
    op.drop_index(
        "ix_workflow_discrepancy_item_status",
        table_name="workflow_discrepancies",
    )
    op.drop_table("workflow_discrepancies")

    op.drop_index(
        "ix_workflow_comparison_item_status",
        table_name="workflow_comparisons",
    )
    op.drop_table("workflow_comparisons")

    op.drop_index("ix_workflow_pass_assignee", table_name="workflow_passes")
    op.drop_index("ix_workflow_pass_status", table_name="workflow_passes")
    op.drop_index("ix_workflow_pass_item_current", table_name="workflow_passes")
    op.drop_table("workflow_passes")

    op.drop_index("ix_workflow_items_canonical_race", table_name="workflow_items")
    op.drop_index("ix_workflow_items_year_state", table_name="workflow_items")
    op.drop_index("ix_workflow_items_lifecycle_stage", table_name="workflow_items")
    op.drop_table("workflow_items")
