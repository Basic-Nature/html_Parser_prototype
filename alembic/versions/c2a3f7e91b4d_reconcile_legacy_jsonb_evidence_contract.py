"""Reconcile legacy parser/context evidence fields to PostgreSQL JSONB.

Revision ID: c2a3f7e91b4d
Revises: b41c7d2e9a6f

Historical contract
-------------------
The legacy parser/context evidence fields were deliberately modeled as JSONB
before a later portability refactor flattened them to SQLAlchemy JSON.

Production PostgreSQL retained JSONB. Fresh PostgreSQL databases created from
the current Alembic root receive JSON. This revision reconciles both histories:

* PostgreSQL JSON  -> JSONB
* PostgreSQL JSONB -> no-op
* unexpected physical type -> fail closed
* non-PostgreSQL -> no-op

The canonical publication JSON columns are intentionally outside this revision.
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision: str = "c2a3f7e91b4d"
down_revision: str | None = "b41c7d2e9a6f"
branch_labels: str | None = None
depends_on: str | None = None


EVIDENCE_JSON_COLUMNS: tuple[tuple[str, str], ...] = (
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
)


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _jsonb_reconciliation_block(table_name: str, column_name: str) -> str:
    table_literal = table_name.replace("'", "''")
    column_literal = column_name.replace("'", "''")
    table_ident = _quote_identifier(table_name)
    column_ident = _quote_identifier(column_name)

    return f"""
DO $electionpulse$
DECLARE
    current_udt text;
BEGIN
    SELECT c.udt_name
      INTO current_udt
      FROM information_schema.columns AS c
     WHERE c.table_schema = 'public'
       AND c.table_name = '{table_literal}'
       AND c.column_name = '{column_literal}';

    IF current_udt IS NULL THEN
        RAISE EXCEPTION
            'ElectionPulse JSONB reconciliation: missing %.%',
            '{table_literal}',
            '{column_literal}';
    ELSIF current_udt = 'jsonb' THEN
        NULL;
    ELSIF current_udt = 'json' THEN
        ALTER TABLE public.{table_ident}
            ALTER COLUMN {column_ident}
            TYPE jsonb
            USING {column_ident}::jsonb;
    ELSE
        RAISE EXCEPTION
            'ElectionPulse JSONB reconciliation: unexpected type %.% = %',
            '{table_literal}',
            '{column_literal}',
            current_udt;
    END IF;
END
$electionpulse$
""".strip()


def upgrade() -> None:
    bind = op.get_bind()

    if bind.dialect.name != "postgresql":
        return

    for table_name, column_name in EVIDENCE_JSON_COLUMNS:
        op.execute(
            sa.text(
                _jsonb_reconciliation_block(
                    table_name,
                    column_name,
                )
            )
        )


def downgrade() -> None:
    bind = op.get_bind()

    if bind.dialect.name != "postgresql":
        return

    raise RuntimeError(
        "c2a3f7e91b4d PostgreSQL downgrade is intentionally blocked: "
        "legacy production JSONB predates this reconciliation revision."
    )