"""Governance contract for elevation resource-binding hardening."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = "a19c7e4b2d60"
MIGRATION = ROOT / "alembic/versions/a19c7e4b2d60_trusted_elevation_resource_binding.py"
REGISTRY = ROOT / "scripts/production/schema_migration_registry.json"
CONTROLLER = ROOT / "scripts/production/governed_schema_migration.py"
WORKFLOW = ROOT / ".github/workflows/production_schema_migration.yml"


def test_registry_binds_migration_columns_and_empty_precondition() -> None:
    spec = json.loads(REGISTRY.read_text(encoding="utf-8"))["migrations"][TARGET]
    assert spec["from_revision"] == "f18a7c3d4e92"
    assert spec["expected_new_tables"] == []
    assert spec["expected_new_columns"] == {
        "trusted_elevation_challenges": ["resource_type", "resource_id", "resource_version"],
        "trusted_elevation_grants": ["resource_type", "resource_id", "resource_version", "consumed_at"],
    }
    assert set(spec["required_empty_tables"]) == {"trusted_elevation_challenges", "trusted_elevation_grants"}
    assert spec["migration_sha256"] == hashlib.sha256(MIGRATION.read_bytes()).hexdigest()


def test_controller_checks_columns_and_required_empty_tables() -> None:
    source = CONTROLLER.read_text(encoding="utf-8")
    for token in ("expected_new_columns", "required_empty_tables", "expected_column_state", "required_empty_table_counts", "Unexpected pre-existing target column", "Required empty table is not empty", "Expected new column(s) missing"):
        assert token in source


def test_manual_workflow_allowlists_target_once() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")
    assert source.count("          - a19c7e4b2d60") == 1
