from __future__ import annotations

from pathlib import Path

from sqlalchemy.dialects import postgresql, sqlite

from webapp.parser.utils.models import Base


REPO_ROOT = Path(__file__).resolve().parents[2]
MIGRATION = (
    REPO_ROOT
    / "alembic"
    / "versions"
    / "c2a3f7e91b4d_reconcile_legacy_jsonb_evidence_contract.py"
)

RECOVERED_JSONB_TARGETS = {
    ("alerts", "context"),
    ("ballot_types", "metastats"),
    ("batch_metadata", "metastats"),
    ("buttons", "metastats"),
    ("candidate_panels", "metastats"),
    ("candidates", "metastats"),
    ("contests", "metastats"),
    ("entities", "metastats"),
    ("headings", "metastats"),
    ("location_panels", "metastats"),
    ("misc_entities", "metastats"),
    ("panels", "metastats"),
    ("party_labels", "metastats"),
    ("results", "metastats"),
    ("results_timestamps", "metastats"),
    ("staging_election_results", "metastats"),
    ("vote_methods", "metastats"),
    ("warehouse_election_results", "metastats"),
}

NON_TARGET_JSON_FIELDS = {
    ("canonical_source_artifacts", "provenance"),
    ("canonical_election_races", "qa_metadata"),
    ("canonical_election_results", "provenance"),
    ("canonical_verification_events", "event_metadata"),
    ("data_framework_preview_cache", "payload"),
}


def _compiled_type(table_name: str, column_name: str, dialect) -> str:
    column = Base.metadata.tables[table_name].c[column_name]
    return column.type.compile(dialect=dialect)


def test_recovered_evidence_fields_use_jsonb_on_postgresql() -> None:
    dialect = postgresql.dialect()

    assert len(RECOVERED_JSONB_TARGETS) == 18

    for table_name, column_name in sorted(RECOVERED_JSONB_TARGETS):
        assert _compiled_type(table_name, column_name, dialect).upper() == "JSONB"


def test_recovered_evidence_fields_remain_json_on_sqlite() -> None:
    dialect = sqlite.dialect()

    for table_name, column_name in sorted(RECOVERED_JSONB_TARGETS):
        assert _compiled_type(table_name, column_name, dialect).upper() == "JSON"


def test_canonical_and_preview_json_fields_are_not_swept_into_jsonb() -> None:
    pg = postgresql.dialect()
    sq = sqlite.dialect()

    assert len(NON_TARGET_JSON_FIELDS) == 5

    for table_name, column_name in sorted(NON_TARGET_JSON_FIELDS):
        assert _compiled_type(table_name, column_name, pg).upper() == "JSON"
        assert _compiled_type(table_name, column_name, sq).upper() == "JSON"


def test_reconciliation_revision_is_exactly_chained_and_scoped() -> None:
    source = MIGRATION.read_text(encoding="utf-8")

    assert 'revision: str = "c2a3f7e91b4d"' in source
    assert 'down_revision: str | None = "b41c7d2e9a6f"' in source

    for table_name, column_name in sorted(RECOVERED_JSONB_TARGETS):
        assert f'("{table_name}", "{column_name}")' in source

    for table_name, column_name in sorted(NON_TARGET_JSON_FIELDS):
        assert f'("{table_name}", "{column_name}")' not in source

    assert "current_udt = 'jsonb'" in source
    assert "current_udt = 'json'" in source
    assert "TYPE jsonb" in source
    assert "unexpected type" in source
    assert "downgrade is intentionally blocked" in source