"""Contracts for the canonical ElectionPulse seed schema."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from sqlalchemy import CheckConstraint

from webapp.parser.persistence.alembic_filters import include_object
from webapp.parser.utils.models import Base


REPO_ROOT = Path(__file__).resolve().parents[2]
MIGRATION = (
    REPO_ROOT
    / "alembic/versions/"
    "b41c7d2e9a6f_canonical_production_seed_schema.py"
)

CANONICAL_TABLES = {
    "canonical_source_artifacts",
    "canonical_election_races",
    "canonical_election_results",
    "canonical_vote_components",
    "canonical_verification_events",
}


def test_canonical_tables_are_registered_in_base_metadata():
    assert CANONICAL_TABLES <= set(Base.metadata.tables)


def test_race_schema_records_one_selected_dl_as_provenance():
    table = Base.metadata.tables["canonical_election_races"]

    required = {
        "source_race_id",
        "election_year",
        "election_date",
        "date_precision",
        "state",
        "contest",
        "production_status",
        "selected_dl_source",
        "payload_artifact_id",
        "approval_artifact_id",
        "verification_status",
        "qa_metadata",
    }
    assert required <= set(table.c.keys())

    check_text = " ".join(
        str(constraint.sqltext)
        for constraint in table.constraints
        if isinstance(constraint, CheckConstraint)
    )
    assert "DL1" in check_text
    assert "DL2" in check_text


def test_result_preserves_source_and_normalized_jurisdiction():
    table = Base.metadata.tables["canonical_election_results"]

    required = {
        "source_row_index",
        "source_row_hash",
        "source_jurisdiction_label",
        "jurisdiction_key",
        "jurisdiction_name",
        "jurisdiction_type",
        "aggregation_scope",
        "precinct",
        "candidate",
        "total_votes",
        "source_url",
        "provenance",
    }
    assert required <= set(table.c.keys())


def test_vote_components_allow_signed_source_adjustments():
    table = Base.metadata.tables["canonical_vote_components"]

    vote_checks = [
        constraint
        for constraint in table.constraints
        if isinstance(constraint, CheckConstraint)
    ]
    assert vote_checks == []


def test_unmanaged_reflected_tables_are_excluded_from_autogenerate():
    tiger_table = SimpleNamespace(schema="tiger", name="county")
    assert include_object(
        tiger_table,
        "county",
        "table",
        reflected=True,
        compare_to=None,
    ) is False

    public_support = SimpleNamespace(
        schema=None,
        name="spatial_ref_sys",
    )
    assert include_object(
        public_support,
        "spatial_ref_sys",
        "table",
        reflected=True,
        compare_to=None,
    ) is False

    workflow_table = SimpleNamespace(
        schema="workflow",
        name="contests",
    )
    assert include_object(
        workflow_table,
        "contests",
        "table",
        reflected=True,
        compare_to=None,
    ) is False

    model_backed_table = SimpleNamespace(
        schema=None,
        name="canonical_election_races",
    )
    assert include_object(
        model_backed_table,
        "canonical_election_races",
        "table",
        reflected=True,
        compare_to=object(),
    ) is True

    metadata_only_table = SimpleNamespace(
        schema=None,
        name="canonical_source_artifacts",
    )
    assert include_object(
        metadata_only_table,
        "canonical_source_artifacts",
        "table",
        reflected=False,
        compare_to=None,
    ) is True

    modeled_table = SimpleNamespace(
        schema=None,
        name="warehouse_election_results",
    )
    reflected_index = SimpleNamespace(
        table=modeled_table,
        name="ix_extra_modeled_index",
    )
    assert include_object(
        reflected_index,
        "ix_extra_modeled_index",
        "index",
        reflected=True,
        compare_to=None,
    ) is True


def test_migration_is_chained_and_legacy_repair_is_conditional():
    source = MIGRATION.read_text(encoding="utf-8")

    assert 'revision: str = "b41c7d2e9a6f"' in source
    assert 'down_revision: str | None = "a2305f825683"' in source

    for column in (
        "verification_status",
        "source_url",
        "source_principal",
        "verification_notes",
        "verified_at",
        "verified_by",
    ):
        assert column in source

    assert "context.is_offline_mode()" in source
    assert "inspector.has_table" in source

    for table in CANONICAL_TABLES:
        assert f'"{table}"' in source


def test_models_document_dl1_dl2_as_selection_not_union():
    models_path = REPO_ROOT / "webapp/parser/utils/models.py"
    source = models_path.read_text(encoding="utf-8")

    assert "DL1 and DL2 are independent comparison lanes." in source
    assert "They are NOT additive" in source
