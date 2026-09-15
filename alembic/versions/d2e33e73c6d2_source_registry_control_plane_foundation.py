"""Add Source Registry control-plane foundation and frozen legacy bootstrap.

Revision ID: d2e33e73c6d2
Revises: a19c7e4b2d60
Create Date: 2026-09-15
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Sequence
import uuid

from alembic import op
import sqlalchemy as sa

revision: str = "d2e33e73c6d2"
down_revision: str | None = "a19c7e4b2d60"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

BOOTSTRAP_RELATIVE_PATH = "scripts/production/source_registry_bootstrap_v1.json"
BOOTSTRAP_SHA256 = "2420c00728ac4ba42b00165d2000f15aa5a8f15428578a292062a29ae90833b0"
BOOTSTRAP_AT = datetime(2026, 9, 15, 8, 23, 29, tzinfo=timezone.utc)


def _registry_json_type() -> sa.types.TypeEngine:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        from sqlalchemy.dialects.postgresql import JSONB
        return JSONB()
    return sa.JSON()


def _bootstrap_path() -> Path:
    return Path(__file__).resolve().parents[2] / BOOTSTRAP_RELATIVE_PATH


def _load_bootstrap() -> dict:
    path = _bootstrap_path()
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != BOOTSTRAP_SHA256:
        raise RuntimeError(
            f"Source Registry bootstrap SHA mismatch: expected={BOOTSTRAP_SHA256} actual={actual}"
        )
    payload = json.loads(data.decode("utf-8"))
    if payload.get("schema") != "electionpulse_source_registry_bootstrap_snapshot_v1":
        raise RuntimeError("Unexpected Source Registry bootstrap schema.")
    return payload


def _u(value: str | None):
    return uuid.UUID(value) if value else None


def _ensure_offline_utf8_output() -> None:
    """Keep Alembic --sql output Unicode-safe on Windows legacy consoles.

    Online migrations are untouched. StringIO-style buffers already accept
    Unicode, while TextIOWrapper buffers can be switched to UTF-8 explicitly.
    """
    context = op.get_context()
    if not getattr(context, "as_sql", False):
        return

    impl = getattr(context, "impl", None)
    output_buffer = getattr(impl, "output_buffer", None)
    reconfigure = getattr(output_buffer, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8")


def upgrade() -> None:
    _ensure_offline_utf8_output()
    registry_json = _registry_json_type()

    op.create_table(
        "source_registry_sources",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("lifecycle_state", sa.String(length=16), nullable=False),
        sa.Column("row_version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_by_principal_id", sa.UUID(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "lifecycle_state IN ('active','quarantined','deprecated')",
            name="ck_source_registry_source_lifecycle",
        ),
        sa.CheckConstraint("row_version >= 1", name="ck_source_registry_source_version"),
        sa.ForeignKeyConstraint(
            ["created_by_principal_id"], ["trusted_principals.id"],
            name="fk_source_registry_sources_created_by", ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_source_registry_sources"),
    )
    op.create_index(
        "ix_source_registry_sources_lifecycle_state",
        "source_registry_sources", ["lifecycle_state"], unique=False,
    )

    op.create_table(
        "source_registry_revisions",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("source_id", sa.UUID(), nullable=False),
        sa.Column("revision_number", sa.Integer(), nullable=False),
        sa.Column("exact_url", sa.Text(), nullable=False),
        sa.Column("normalized_url", sa.Text(), nullable=False),
        sa.Column("host", sa.String(length=255), nullable=False),
        sa.Column("url_sha256", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_by_principal_id", sa.UUID(), nullable=True),
        sa.CheckConstraint("revision_number >= 1", name="ck_source_registry_revision_number"),
        sa.CheckConstraint("length(url_sha256) = 64", name="ck_source_registry_revision_url_hash"),
        sa.ForeignKeyConstraint(
            ["source_id"], ["source_registry_sources.id"],
            name="fk_source_registry_revisions_source", ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["created_by_principal_id"], ["trusted_principals.id"],
            name="fk_source_registry_revisions_created_by", ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_source_registry_revisions"),
        sa.UniqueConstraint("source_id", "revision_number", name="uq_source_registry_revision_number"),
        sa.UniqueConstraint("source_id", "url_sha256", name="uq_source_registry_revision_url_hash"),
    )
    op.create_index("ix_source_registry_revisions_source", "source_registry_revisions", ["source_id"], unique=False)
    op.create_index("ix_source_registry_revisions_host", "source_registry_revisions", ["host"], unique=False)

    op.create_table(
        "source_registry_bindings",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("source_id", sa.UUID(), nullable=False),
        sa.Column("current_revision_id", sa.UUID(), nullable=False),
        sa.Column("year", sa.String(length=16), nullable=False),
        sa.Column("contest", sa.String(length=512), nullable=False),
        sa.Column("state", sa.String(length=64), nullable=False),
        sa.Column("scope", sa.String(length=512), nullable=False),
        sa.Column("format", sa.String(length=64), nullable=False),
        sa.Column("notes", sa.Text(), nullable=False),
        sa.Column("review_state", sa.String(length=24), nullable=False),
        sa.Column("parser_eligible", sa.Boolean(), nullable=False),
        sa.Column("public_eligible", sa.Boolean(), nullable=False),
        sa.Column("workflow_eligible", sa.Boolean(), nullable=False),
        sa.Column("row_version", sa.Integer(), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("published_by_principal_id", sa.UUID(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "review_state IN ('backlog','approved','quarantined','deprecated')",
            name="ck_source_registry_binding_review_state",
        ),
        sa.CheckConstraint("row_version >= 1", name="ck_source_registry_binding_version"),
        sa.CheckConstraint(
            "public_eligible = false OR parser_eligible = true",
            name="ck_source_registry_binding_public_implies_parser",
        ),
        sa.CheckConstraint(
            "workflow_eligible = false OR parser_eligible = true",
            name="ck_source_registry_binding_workflow_implies_parser",
        ),
        sa.CheckConstraint(
            "public_eligible = false OR review_state = 'approved'",
            name="ck_source_registry_binding_public_requires_approved",
        ),
        sa.CheckConstraint(
            "workflow_eligible = false OR review_state = 'approved'",
            name="ck_source_registry_binding_workflow_requires_approved",
        ),
        sa.CheckConstraint(
            "review_state NOT IN ('quarantined','deprecated') OR "
            "(parser_eligible=false AND public_eligible=false AND workflow_eligible=false)",
            name="ck_source_registry_binding_disabled_states",
        ),
        sa.ForeignKeyConstraint(
            ["source_id"], ["source_registry_sources.id"],
            name="fk_source_registry_bindings_source", ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["current_revision_id"], ["source_registry_revisions.id"],
            name="fk_source_registry_bindings_revision", ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["published_by_principal_id"], ["trusted_principals.id"],
            name="fk_source_registry_bindings_published_by", ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_source_registry_bindings"),
    )
    op.create_index("ix_source_registry_bindings_review", "source_registry_bindings", ["review_state"], unique=False)
    op.create_index("ix_source_registry_bindings_public", "source_registry_bindings", ["public_eligible", "review_state"], unique=False)
    op.create_index("ix_source_registry_bindings_workflow", "source_registry_bindings", ["workflow_eligible", "review_state"], unique=False)
    op.create_index("ix_source_registry_bindings_source", "source_registry_bindings", ["source_id"], unique=False)

    op.create_table(
        "source_registry_aliases",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("binding_id", sa.UUID(), nullable=False),
        sa.Column("alias_type", sa.String(length=32), nullable=False),
        sa.Column("alias_value", sa.String(length=96), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "alias_type IN ('legacy_blsrc_v1','stable_blsrc_v2')",
            name="ck_source_registry_alias_type",
        ),
        sa.ForeignKeyConstraint(
            ["binding_id"], ["source_registry_bindings.id"],
            name="fk_source_registry_aliases_binding", ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_source_registry_aliases"),
        sa.UniqueConstraint("alias_value", name="uq_source_registry_alias_value"),
    )
    op.create_index("ix_source_registry_aliases_binding_active", "source_registry_aliases", ["binding_id", "active"], unique=False)

    op.create_table(
        "source_registry_proposals",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("operation", sa.String(length=32), nullable=False),
        sa.Column("target_binding_id", sa.UUID(), nullable=True),
        sa.Column("expected_target_row_version", sa.Integer(), nullable=True),
        sa.Column("proposed_payload", registry_json, nullable=False),
        sa.Column("payload_sha256", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("proposer_principal_id", sa.UUID(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("row_version", sa.Integer(), nullable=False),
        sa.CheckConstraint(
            "operation IN ('create','revise_url','revise_metadata','change_eligibility',"
            "'quarantine','deprecate','restore')",
            name="ck_source_registry_proposal_operation",
        ),
        sa.CheckConstraint(
            "status IN ('submitted','review_approved','review_rejected','published','cancelled')",
            name="ck_source_registry_proposal_status",
        ),
        sa.CheckConstraint("row_version >= 1", name="ck_source_registry_proposal_version"),
        sa.CheckConstraint(
            "expected_target_row_version IS NULL OR expected_target_row_version >= 1",
            name="ck_source_registry_proposal_target_version",
        ),
        sa.ForeignKeyConstraint(
            ["target_binding_id"], ["source_registry_bindings.id"],
            name="fk_source_registry_proposals_binding", ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["proposer_principal_id"], ["trusted_principals.id"],
            name="fk_source_registry_proposals_proposer", ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_source_registry_proposals"),
    )
    op.create_index("ix_source_registry_proposals_status_created", "source_registry_proposals", ["status", "created_at"], unique=False)
    op.create_index("ix_source_registry_proposals_target", "source_registry_proposals", ["target_binding_id"], unique=False)

    op.create_table(
        "source_registry_reviews",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("proposal_id", sa.UUID(), nullable=False),
        sa.Column("proposal_row_version", sa.Integer(), nullable=False),
        sa.Column("decision", sa.String(length=16), nullable=False),
        sa.Column("reviewer_principal_id", sa.UUID(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("proposal_row_version >= 1", name="ck_source_registry_review_version"),
        sa.CheckConstraint("decision IN ('approve','reject')", name="ck_source_registry_review_decision"),
        sa.ForeignKeyConstraint(
            ["proposal_id"], ["source_registry_proposals.id"],
            name="fk_source_registry_reviews_proposal", ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["reviewer_principal_id"], ["trusted_principals.id"],
            name="fk_source_registry_reviews_reviewer", ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_source_registry_reviews"),
        sa.UniqueConstraint("proposal_id", "proposal_row_version", name="uq_source_registry_review_proposal_version"),
    )

    op.create_table(
        "source_registry_events",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("source_id", sa.UUID(), nullable=True),
        sa.Column("binding_id", sa.UUID(), nullable=True),
        sa.Column("proposal_id", sa.UUID(), nullable=True),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("principal_id", sa.UUID(), nullable=True),
        sa.Column("credential_id", sa.UUID(), nullable=True),
        sa.Column("session_id", sa.UUID(), nullable=True),
        sa.Column("resource_version", sa.Integer(), nullable=False),
        sa.Column("before_sha256", sa.String(length=64), nullable=True),
        sa.Column("after_sha256", sa.String(length=64), nullable=True),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("event_metadata", registry_json, nullable=False),
        sa.Column("previous_event_hash", sa.String(length=64), nullable=True),
        sa.Column("event_hash", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("resource_version >= 1", name="ck_source_registry_event_version"),
        sa.CheckConstraint("length(event_hash) = 64", name="ck_source_registry_event_hash"),
        sa.ForeignKeyConstraint(["source_id"], ["source_registry_sources.id"], name="fk_source_registry_events_source", ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["binding_id"], ["source_registry_bindings.id"], name="fk_source_registry_events_binding", ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["proposal_id"], ["source_registry_proposals.id"], name="fk_source_registry_events_proposal", ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["principal_id"], ["trusted_principals.id"], name="fk_source_registry_events_principal", ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["credential_id"], ["trusted_credentials.id"], name="fk_source_registry_events_credential", ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id", name="pk_source_registry_events"),
        sa.UniqueConstraint("event_hash", name="uq_source_registry_event_hash"),
    )
    op.create_index("ix_source_registry_events_binding_time", "source_registry_events", ["binding_id", "created_at"], unique=False)
    op.create_index("ix_source_registry_events_type_time", "source_registry_events", ["event_type", "created_at"], unique=False)
    op.create_index("ix_source_registry_events_principal_time", "source_registry_events", ["principal_id", "created_at"], unique=False)

    snapshot = _load_bootstrap()
    source_table = sa.table(
        "source_registry_sources",
        sa.column("id", sa.UUID()),
        sa.column("lifecycle_state", sa.String()),
        sa.column("row_version", sa.Integer()),
        sa.column("created_at", sa.DateTime(timezone=True)),
        sa.column("created_by_principal_id", sa.UUID()),
        sa.column("updated_at", sa.DateTime(timezone=True)),
    )
    revision_table = sa.table(
        "source_registry_revisions",
        sa.column("id", sa.UUID()),
        sa.column("source_id", sa.UUID()),
        sa.column("revision_number", sa.Integer()),
        sa.column("exact_url", sa.Text()),
        sa.column("normalized_url", sa.Text()),
        sa.column("host", sa.String()),
        sa.column("url_sha256", sa.String()),
        sa.column("created_at", sa.DateTime(timezone=True)),
        sa.column("created_by_principal_id", sa.UUID()),
    )
    binding_table = sa.table(
        "source_registry_bindings",
        sa.column("id", sa.UUID()),
        sa.column("source_id", sa.UUID()),
        sa.column("current_revision_id", sa.UUID()),
        sa.column("year", sa.String()),
        sa.column("contest", sa.String()),
        sa.column("state", sa.String()),
        sa.column("scope", sa.String()),
        sa.column("format", sa.String()),
        sa.column("notes", sa.Text()),
        sa.column("review_state", sa.String()),
        sa.column("parser_eligible", sa.Boolean()),
        sa.column("public_eligible", sa.Boolean()),
        sa.column("workflow_eligible", sa.Boolean()),
        sa.column("row_version", sa.Integer()),
        sa.column("published_at", sa.DateTime(timezone=True)),
        sa.column("published_by_principal_id", sa.UUID()),
        sa.column("created_at", sa.DateTime(timezone=True)),
        sa.column("updated_at", sa.DateTime(timezone=True)),
    )
    alias_table = sa.table(
        "source_registry_aliases",
        sa.column("id", sa.UUID()),
        sa.column("binding_id", sa.UUID()),
        sa.column("alias_type", sa.String()),
        sa.column("alias_value", sa.String()),
        sa.column("active", sa.Boolean()),
        sa.column("created_at", sa.DateTime(timezone=True)),
    )
    event_table = sa.table(
        "source_registry_events",
        sa.column("id", sa.UUID()),
        sa.column("source_id", sa.UUID()),
        sa.column("binding_id", sa.UUID()),
        sa.column("proposal_id", sa.UUID()),
        sa.column("event_type", sa.String()),
        sa.column("principal_id", sa.UUID()),
        sa.column("credential_id", sa.UUID()),
        sa.column("session_id", sa.UUID()),
        sa.column("resource_version", sa.Integer()),
        sa.column("before_sha256", sa.String()),
        sa.column("after_sha256", sa.String()),
        sa.column("reason", sa.Text()),
        sa.column("event_metadata", registry_json),
        sa.column("previous_event_hash", sa.String()),
        sa.column("event_hash", sa.String()),
        sa.column("created_at", sa.DateTime(timezone=True)),
    )

    op.bulk_insert(source_table, [{
        "id": _u(item["source_id"]),
        "lifecycle_state": item["lifecycle_state"],
        "row_version": int(item["row_version"]),
        "created_at": BOOTSTRAP_AT,
        "created_by_principal_id": None,
        "updated_at": BOOTSTRAP_AT,
    } for item in snapshot["sources"]])

    op.bulk_insert(revision_table, [{
        "id": _u(item["revision_id"]),
        "source_id": _u(item["source_id"]),
        "revision_number": int(item["revision_number"]),
        "exact_url": item["exact_url"],
        "normalized_url": item["normalized_url"],
        "host": item["host"],
        "url_sha256": item["url_sha256"],
        "created_at": BOOTSTRAP_AT,
        "created_by_principal_id": None,
    } for item in snapshot["revisions"]])

    op.bulk_insert(binding_table, [{
        "id": _u(item["binding_id"]),
        "source_id": _u(item["source_id"]),
        "current_revision_id": _u(item["current_revision_id"]),
        "year": item["year"],
        "contest": item["contest"],
        "state": item["state"],
        "scope": item["scope"],
        "format": item["format"],
        "notes": item["notes"],
        "review_state": item["review_state"],
        "parser_eligible": bool(item["parser_eligible"]),
        "public_eligible": bool(item["public_eligible"]),
        "workflow_eligible": bool(item["workflow_eligible"]),
        "row_version": int(item["row_version"]),
        "published_at": BOOTSTRAP_AT if item["review_state"] == "approved" else None,
        "published_by_principal_id": None,
        "created_at": BOOTSTRAP_AT,
        "updated_at": BOOTSTRAP_AT,
    } for item in snapshot["bindings"]])

    op.bulk_insert(alias_table, [{
        "id": _u(item["alias_id"]),
        "binding_id": _u(item["binding_id"]),
        "alias_type": item["alias_type"],
        "alias_value": item["alias_value"],
        "active": bool(item["active"]),
        "created_at": BOOTSTRAP_AT,
    } for item in snapshot["aliases"]])

    event_metadata_literal = op.inline_literal(
        json.dumps(
            {"bootstrap_provenance": "legacy_urls_txt_v1"},
            sort_keys=True,
            separators=(",", ":"),
        ),
        type_=sa.Text(),
    )
    op.bulk_insert(
        event_table,
        [{
            "id": _u(item["event_id"]),
            "source_id": _u(item["source_id"]),
            "binding_id": _u(item["binding_id"]),
            "proposal_id": _u(item.get("proposal_id")),
            "event_type": item["event_type"],
            "principal_id": None,
            "credential_id": None,
            "session_id": None,
            "resource_version": int(item["resource_version"]),
            "before_sha256": item.get("before_sha256"),
            "after_sha256": item.get("after_sha256"),
            "reason": item["reason"],
            "event_metadata": event_metadata_literal,
            "previous_event_hash": item.get("previous_event_hash"),
            "event_hash": item["event_hash"],
            "created_at": BOOTSTRAP_AT,
        } for item in snapshot["events"]],
        multiinsert=False,
    )


def downgrade() -> None:
    op.drop_index("ix_source_registry_events_principal_time", table_name="source_registry_events")
    op.drop_index("ix_source_registry_events_type_time", table_name="source_registry_events")
    op.drop_index("ix_source_registry_events_binding_time", table_name="source_registry_events")
    op.drop_table("source_registry_events")
    op.drop_table("source_registry_reviews")
    op.drop_index("ix_source_registry_proposals_target", table_name="source_registry_proposals")
    op.drop_index("ix_source_registry_proposals_status_created", table_name="source_registry_proposals")
    op.drop_table("source_registry_proposals")
    op.drop_index("ix_source_registry_aliases_binding_active", table_name="source_registry_aliases")
    op.drop_table("source_registry_aliases")
    op.drop_index("ix_source_registry_bindings_source", table_name="source_registry_bindings")
    op.drop_index("ix_source_registry_bindings_workflow", table_name="source_registry_bindings")
    op.drop_index("ix_source_registry_bindings_public", table_name="source_registry_bindings")
    op.drop_index("ix_source_registry_bindings_review", table_name="source_registry_bindings")
    op.drop_table("source_registry_bindings")
    op.drop_index("ix_source_registry_revisions_host", table_name="source_registry_revisions")
    op.drop_index("ix_source_registry_revisions_source", table_name="source_registry_revisions")
    op.drop_table("source_registry_revisions")
    op.drop_index("ix_source_registry_sources_lifecycle_state", table_name="source_registry_sources")
    op.drop_table("source_registry_sources")
