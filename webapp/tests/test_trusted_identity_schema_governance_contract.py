"""Governed production migration contract for trusted identity foundation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = "f18a7c3d4e92"
FROM = "e7b2c4d91f60"
MIGRATION = (
    ROOT
    / "alembic"
    / "versions"
    / "f18a7c3d4e92_trusted_identity_authority_foundation.py"
)
REGISTRY = ROOT / "scripts" / "production" / "schema_migration_registry.json"
WORKFLOW = ROOT / ".github" / "workflows" / "production_schema_migration.yml"

EXPECTED_NEW_TABLES = {
    "trusted_principals",
    "trusted_identity_bindings",
    "trusted_credentials",
    "trusted_role_bindings",
    "trusted_devices",
    "trusted_trust_events",
    "trusted_sessions",
    "trusted_elevation_challenges",
    "trusted_access_handoffs",
    "trusted_elevation_grants",
}


def test_registry_exactly_governs_trusted_identity_migration() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    spec = registry["migrations"][TARGET]

    assert spec["target_revision"] == TARGET
    assert spec["from_revision"] == FROM
    assert spec["kind"] == "trusted_identity_authority_foundation"
    assert set(spec["expected_new_tables"]) == EXPECTED_NEW_TABLES
    assert spec["expect_new_tables_empty"] is True
    assert spec["preserve_canonical_publication_metrics"] is True
    assert spec["rollback_policy"] == "manual_review_only_no_automatic_downgrade"

    actual = hashlib.sha256(MIGRATION.read_bytes()).hexdigest()
    assert spec["migration_sha256"] == actual


def test_manual_workflow_exposes_new_allowlisted_target() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")

    assert f"          - {FROM}" in source
    assert f"          - {TARGET}" in source
    assert source.count(f"          - {TARGET}") == 1


def test_migration_is_schema_only_and_empty_by_default() -> None:
    source = MIGRATION.read_text(encoding="utf-8")

    assert "op.create_table(" in source
    assert "op.bulk_insert(" not in source
    assert "ROOT_ADMIN_CERT_FINGERPRINTS" not in source
    assert "ADMIN_JWT_TOKEN" not in source
