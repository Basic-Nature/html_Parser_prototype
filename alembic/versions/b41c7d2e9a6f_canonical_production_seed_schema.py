"""Canonical production seed schema.

Revision ID: b41c7d2e9a6f
Revises: a2305f825683

This revision creates a canonical publication layer for the verified seed.
It also repairs six warehouse provenance columns on legacy databases that
were created outside Alembic before those columns existed physically.

DL1 and DL2 remain independent QA comparison lanes. Canonical rows are not
the union of both lanes; selected_dl_source records which lane QA approved
for each production race.
"""

from __future__ import annotations

from alembic import context, op
import sqlalchemy as sa


revision: str = "b41c7d2e9a6f"
down_revision: str | None = "a2305f825683"
branch_labels = None
depends_on = None

WAREHOUSE_TABLE = "warehouse_election_results"


def _repair_legacy_warehouse_columns() -> None:
    # Offline SQL represents a normal fresh upgrade chain. The initial
    # revision already creates these columns, so no repair SQL is needed.
    if context.is_offline_mode():
        return

    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if not inspector.has_table(WAREHOUSE_TABLE):
        return

    existing = {
        column["name"]
        for column in inspector.get_columns(WAREHOUSE_TABLE)
    }

    required = [
        sa.Column(
            "verification_status",
            sa.String(length=16),
            nullable=True,
        ),
        sa.Column(
            "source_url",
            sa.String(length=2048),
            nullable=True,
        ),
        sa.Column(
            "source_principal",
            sa.String(length=256),
            nullable=True,
        ),
        sa.Column(
            "verification_notes",
            sa.Text(),
            nullable=True,
        ),
        sa.Column(
            "verified_at",
            sa.DateTime(),
            nullable=True,
        ),
        sa.Column(
            "verified_by",
            sa.String(length=256),
            nullable=True,
        ),
    ]

    for column in required:
        if column.name not in existing:
            op.add_column(WAREHOUSE_TABLE, column)


def upgrade() -> None:
    _repair_legacy_warehouse_columns()

    op.create_table(
        "canonical_source_artifacts",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("artifact_role", sa.String(length=64), nullable=False),
        sa.Column("filename", sa.String(length=512), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("row_count", sa.Integer(), nullable=True),
        sa.Column("race_count", sa.Integer(), nullable=True),
        sa.Column(
            "imported_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column("provenance", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "sha256",
            name="uq_canonical_source_artifact_sha256",
        ),
    )
    op.create_index(
        "ix_canonical_source_artifacts_role",
        "canonical_source_artifacts",
        ["artifact_role"],
        unique=False,
    )

    op.create_table(
        "canonical_election_races",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("source_race_id", sa.String(length=64), nullable=False),
        sa.Column("election_year", sa.Integer(), nullable=False),
        sa.Column("election_date", sa.Date(), nullable=True),
        sa.Column("date_precision", sa.String(length=16), nullable=False),
        sa.Column("state", sa.String(length=64), nullable=False),
        sa.Column("contest", sa.String(length=128), nullable=False),
        sa.Column("office_basic", sa.String(length=64), nullable=True),
        sa.Column(
            "production_status",
            sa.String(length=32),
            nullable=False,
        ),
        sa.Column(
            "selected_dl_source",
            sa.String(length=3),
            nullable=False,
        ),
        sa.Column("source_url", sa.String(length=2048), nullable=True),
        sa.Column(
            "verification_status",
            sa.String(length=32),
            nullable=False,
        ),
        sa.Column(
            "verified_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column("payload_artifact_id", sa.UUID(), nullable=False),
        sa.Column("approval_artifact_id", sa.UUID(), nullable=False),
        sa.Column("qa_metadata", sa.JSON(), nullable=False),
        sa.CheckConstraint(
            "selected_dl_source IN ('DL1', 'DL2')",
            name="ck_canonical_race_selected_dl",
        ),
        sa.CheckConstraint(
            "date_precision IN ('year', 'date')",
            name="ck_canonical_race_date_precision",
        ),
        sa.ForeignKeyConstraint(
            ["approval_artifact_id"],
            ["canonical_source_artifacts.id"],
        ),
        sa.ForeignKeyConstraint(
            ["payload_artifact_id"],
            ["canonical_source_artifacts.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "source_race_id",
            name="uq_canonical_race_source_race_id",
        ),
    )
    op.create_index(
        "ix_canonical_races_year_state",
        "canonical_election_races",
        ["election_year", "state"],
        unique=False,
    )
    op.create_index(
        "ix_canonical_races_selected_dl",
        "canonical_election_races",
        ["selected_dl_source"],
        unique=False,
    )
    op.create_index(
        "ix_canonical_races_verification",
        "canonical_election_races",
        ["verification_status"],
        unique=False,
    )

    op.create_table(
        "canonical_election_results",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("race_id", sa.UUID(), nullable=False),
        sa.Column("source_row_index", sa.Integer(), nullable=False),
        sa.Column("source_row_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "source_jurisdiction_label",
            sa.String(length=256),
            nullable=False,
        ),
        sa.Column(
            "jurisdiction_key",
            sa.String(length=384),
            nullable=False,
        ),
        sa.Column(
            "jurisdiction_name",
            sa.String(length=256),
            nullable=False,
        ),
        sa.Column(
            "jurisdiction_type",
            sa.String(length=32),
            nullable=True,
        ),
        sa.Column(
            "aggregation_scope",
            sa.String(length=32),
            nullable=False,
        ),
        sa.Column("precinct", sa.String(length=256), nullable=True),
        sa.Column(
            "ballot_candidate_name",
            sa.String(length=512),
            nullable=True,
        ),
        sa.Column("candidate", sa.String(length=512), nullable=False),
        sa.Column("ballot_party", sa.String(length=128), nullable=True),
        sa.Column("party", sa.String(length=64), nullable=True),
        sa.Column("fec_id", sa.String(length=64), nullable=True),
        sa.Column("is_write_in", sa.Boolean(), nullable=False),
        sa.Column("total_votes", sa.Integer(), nullable=False),
        sa.Column("source_url", sa.String(length=2048), nullable=True),
        sa.Column("provenance", sa.JSON(), nullable=False),
        sa.CheckConstraint(
            "aggregation_scope IN ('jurisdiction', 'precinct')",
            name="ck_canonical_result_aggregation_scope",
        ),
        sa.ForeignKeyConstraint(
            ["race_id"],
            ["canonical_election_races.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "race_id",
            "source_row_hash",
            name="uq_canonical_result_race_source_row_hash",
        ),
        sa.UniqueConstraint(
            "race_id",
            "source_row_index",
            name="uq_canonical_result_race_source_row_index",
        ),
    )
    op.create_index(
        "ix_canonical_results_race_jurisdiction",
        "canonical_election_results",
        ["race_id", "jurisdiction_key"],
        unique=False,
    )
    op.create_index(
        "ix_canonical_results_candidate",
        "canonical_election_results",
        ["candidate"],
        unique=False,
    )
    op.create_index(
        "ix_canonical_results_fec_id",
        "canonical_election_results",
        ["fec_id"],
        unique=False,
    )

    op.create_table(
        "canonical_vote_components",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("result_id", sa.UUID(), nullable=False),
        sa.Column("vote_method", sa.String(length=32), nullable=False),
        sa.Column("votes", sa.Integer(), nullable=False),
        sa.Column("source_column", sa.String(length=128), nullable=False),
        sa.ForeignKeyConstraint(
            ["result_id"],
            ["canonical_election_results.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "result_id",
            "vote_method",
            name="uq_canonical_vote_component_method",
        ),
    )
    op.create_index(
        "ix_canonical_vote_components_result",
        "canonical_vote_components",
        ["result_id"],
        unique=False,
    )

    op.create_table(
        "canonical_verification_events",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("race_id", sa.UUID(), nullable=False),
        sa.Column("stage", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=64), nullable=False),
        sa.Column(
            "selected_dl_source",
            sa.String(length=3),
            nullable=True,
        ),
        sa.Column("actor", sa.String(length=256), nullable=True),
        sa.Column(
            "occurred_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("event_metadata", sa.JSON(), nullable=False),
        sa.CheckConstraint(
            (
                "selected_dl_source IS NULL OR "
                "selected_dl_source IN ('DL1', 'DL2')"
            ),
            name="ck_canonical_verification_selected_dl",
        ),
        sa.ForeignKeyConstraint(
            ["race_id"],
            ["canonical_election_races.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_canonical_verification_race_stage",
        "canonical_verification_events",
        ["race_id", "stage"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_canonical_verification_race_stage",
        table_name="canonical_verification_events",
    )
    op.drop_table("canonical_verification_events")

    op.drop_index(
        "ix_canonical_vote_components_result",
        table_name="canonical_vote_components",
    )
    op.drop_table("canonical_vote_components")

    op.drop_index(
        "ix_canonical_results_fec_id",
        table_name="canonical_election_results",
    )
    op.drop_index(
        "ix_canonical_results_candidate",
        table_name="canonical_election_results",
    )
    op.drop_index(
        "ix_canonical_results_race_jurisdiction",
        table_name="canonical_election_results",
    )
    op.drop_table("canonical_election_results")

    op.drop_index(
        "ix_canonical_races_verification",
        table_name="canonical_election_races",
    )
    op.drop_index(
        "ix_canonical_races_selected_dl",
        table_name="canonical_election_races",
    )
    op.drop_index(
        "ix_canonical_races_year_state",
        table_name="canonical_election_races",
    )
    op.drop_table("canonical_election_races")

    op.drop_index(
        "ix_canonical_source_artifacts_role",
        table_name="canonical_source_artifacts",
    )
    op.drop_table("canonical_source_artifacts")

    # The six repaired warehouse columns logically belong to the initial
    # revision and are intentionally preserved on downgrade to a2305f825683.
