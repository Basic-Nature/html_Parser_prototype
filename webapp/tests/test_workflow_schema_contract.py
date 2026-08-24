"""Contracts for the governed noncanonical ElectionPulse workflow schema."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import CheckConstraint, String
from sqlalchemy.dialects import postgresql, sqlite

from webapp.parser.utils.models import Base


REPO_ROOT = Path(__file__).resolve().parents[2]
MIGRATION = (
    REPO_ROOT
    / "alembic"
    / "versions"
    / "e7b2c4d91f60_governed_workflow_schema_foundation.py"
)

WORKFLOW_TABLES = {
    "workflow_items",
    "workflow_passes",
    "workflow_comparisons",
    "workflow_discrepancies",
    "workflow_reviews",
    "workflow_events",
    "workflow_artifact_links",
}

LEGACY_WORKFLOW_TABLES = {
    "download_records",
    "validation_records_dl1",
    "validation_records_dl2",
    "preqc_comparisons",
    "qc1_checkpoints",
}

WORKFLOW_JSON_FIELDS = {
    ("workflow_items", "workflow_metadata"),
    ("workflow_passes", "candidate_check_result"),
    ("workflow_passes", "semantic_validation_result"),
    ("workflow_comparisons", "difference_summary"),
    ("workflow_discrepancies", "semantic_key"),
    ("workflow_discrepancies", "left_value"),
    ("workflow_discrepancies", "right_value"),
    ("workflow_reviews", "checklist_result"),
    ("workflow_reviews", "reason_codes"),
    ("workflow_events", "prior_state"),
    ("workflow_events", "new_state"),
    ("workflow_events", "event_metadata"),
    ("workflow_artifact_links", "artifact_metadata"),
}


def _compiled_type(table_name: str, column_name: str, dialect) -> str:
    column = Base.metadata.tables[table_name].c[column_name]
    return column.type.compile(dialect=dialect).upper()


def _fk_targets(table_name: str, column_name: str) -> set[str]:
    column = Base.metadata.tables[table_name].c[column_name]
    return {fk.target_fullname for fk in column.foreign_keys}


def _column_names(table_name: str) -> set[str]:
    return set(Base.metadata.tables[table_name].c.keys())


def test_workflow_tables_are_registered_without_reusing_legacy_tables() -> None:
    tables = set(Base.metadata.tables)

    assert WORKFLOW_TABLES <= tables
    assert WORKFLOW_TABLES.isdisjoint(LEGACY_WORKFLOW_TABLES)


def test_workflow_item_preserves_precise_scope_and_optional_canonical_link() -> None:
    table = Base.metadata.tables["workflow_items"]

    required = {
        "election_year",
        "election_date",
        "state",
        "jurisdiction_name",
        "jurisdiction_type",
        "contest",
        "office_basic",
        "election_type",
        "source_race_id",
        "source_url",
        "canonical_race_id",
        "created_by_principal",
        "workflow_metadata",
        "row_version",
    }
    assert required <= _column_names("workflow_items")

    assert table.c.canonical_race_id.nullable is True
    assert _fk_targets(
        "workflow_items",
        "canonical_race_id",
    ) == {"canonical_election_races.id"}


def test_workflow_principals_are_strings_not_invented_user_foreign_keys() -> None:
    principal_fields = {
        ("workflow_items", "created_by_principal"),
        ("workflow_passes", "assigned_principal"),
        ("workflow_comparisons", "reviewed_by_principal"),
        ("workflow_discrepancies", "resolved_by_principal"),
        ("workflow_reviews", "reviewer_principal"),
        ("workflow_events", "actor_principal"),
    }

    for table_name, column_name in principal_fields:
        column = Base.metadata.tables[table_name].c[column_name]
        assert isinstance(column.type, String)
        assert column.type.length == 256
        assert not column.foreign_keys


def test_pass_model_is_generic_and_does_not_schema_limit_dl_count() -> None:
    table = Base.metadata.tables["workflow_passes"]

    assert {
        "pass_number",
        "pass_label",
        "revision_number",
        "is_current",
        "assigned_principal",
        "source_evidence_ref",
        "staging_batch_id",
    } <= _column_names("workflow_passes")

    assert _fk_targets(
        "workflow_passes",
        "staging_batch_id",
    ) == {"batch_metadata.batch_id"}

    check_text = " ".join(
        str(constraint.sqltext)
        for constraint in table.constraints
        if isinstance(constraint, CheckConstraint)
    )
    assert "pass_number >= 1" in check_text
    assert "revision_number >= 1" in check_text
    assert "DL1" not in check_text
    assert "DL2" not in check_text


def test_comparison_unknown_values_are_nullable_not_synthesized_zero() -> None:
    table = Base.metadata.tables["workflow_comparisons"]

    assert table.c.strict_equality_passed.nullable is True
    assert table.c.difference_count.nullable is True
    assert table.c.strict_equality_passed.default is None
    assert table.c.difference_count.default is None


def test_discrepancy_contract_preserves_value_and_presence_state_separately() -> None:
    table = Base.metadata.tables["workflow_discrepancies"]

    assert {
        "semantic_key",
        "left_value",
        "right_value",
        "left_value_state",
        "right_value_state",
        "resolution_status",
    } <= _column_names("workflow_discrepancies")

    assert table.c.left_value.nullable is True
    assert table.c.right_value.nullable is True
    assert table.c.left_value_state.nullable is True
    assert table.c.right_value_state.nullable is True


def test_reviews_select_operational_pass_or_staging_not_canonical_truth() -> None:
    table = Base.metadata.tables["workflow_reviews"]

    assert _fk_targets(
        "workflow_reviews",
        "selected_pass_id",
    ) == {"workflow_passes.id"}
    assert _fk_targets(
        "workflow_reviews",
        "selected_staging_batch_id",
    ) == {"batch_metadata.batch_id"}

    assert "canonical_race_id" not in table.c


def test_artifact_links_are_typed_references_not_inferred_lineage() -> None:
    table = Base.metadata.tables["workflow_artifact_links"]

    assert {
        "relation_type",
        "artifact_type",
        "artifact_ref",
        "artifact_sha256",
        "canonical_source_artifact_id",
        "staging_batch_id",
        "artifact_metadata",
    } <= _column_names("workflow_artifact_links")

    assert _fk_targets(
        "workflow_artifact_links",
        "canonical_source_artifact_id",
    ) == {"canonical_source_artifacts.id"}
    assert _fk_targets(
        "workflow_artifact_links",
        "staging_batch_id",
    ) == {"batch_metadata.batch_id"}


def test_workflow_json_is_jsonb_on_postgresql_and_json_on_sqlite() -> None:
    pg = postgresql.dialect()
    sq = sqlite.dialect()

    for table_name, column_name in sorted(WORKFLOW_JSON_FIELDS):
        assert _compiled_type(table_name, column_name, pg) == "JSONB"
        assert _compiled_type(table_name, column_name, sq) == "JSON"


def test_migration_is_exactly_chained_and_noncanonical() -> None:
    source = MIGRATION.read_text(encoding="utf-8")

    assert 'revision: str = "e7b2c4d91f60"' in source
    assert 'down_revision: str | None = "c2a3f7e91b4d"' in source

    for table_name in sorted(WORKFLOW_TABLES):
        assert f'"{table_name}"' in source

    for table_name in sorted(LEGACY_WORKFLOW_TABLES):
        assert table_name not in source

    assert "canonical_election_results" not in source
    assert "canonical_vote_components" not in source
    assert "google_sheets" not in source.lower()
    assert "db_lite" not in source.lower()


def test_migration_downgrade_drops_only_new_workflow_tables() -> None:
    source = MIGRATION.read_text(encoding="utf-8")
    downgrade_source = source.split("def downgrade() -> None:", 1)[1]

    for table_name in WORKFLOW_TABLES:
        assert f'op.drop_table("{table_name}")' in downgrade_source

    assert 'op.drop_table("canonical_' not in downgrade_source
    assert 'op.drop_table("batch_metadata")' not in downgrade_source
