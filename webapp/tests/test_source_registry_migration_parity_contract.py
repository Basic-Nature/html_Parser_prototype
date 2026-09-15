from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect, text

ROOT = Path(__file__).resolve().parents[2]
MIGRATION = ROOT / "alembic/versions/d2e33e73c6d2_source_registry_control_plane_foundation.py"
SNAPSHOT = ROOT / "scripts/production/source_registry_bootstrap_v1.json"
REGISTRY = ROOT / "scripts/production/schema_migration_registry.json"


def _load_migration():
    spec = importlib.util.spec_from_file_location("w20_source_registry_migration", MIGRATION)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bootstrap_snapshot_exact_counts_and_identity_authority() -> None:
    raw = SNAPSHOT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == "2420c00728ac4ba42b00165d2000f15aa5a8f15428578a292062a29ae90833b0"
    payload = json.loads(raw)
    assert payload["counts"] == {
        "active_public_aliases": 128,
        "aliases": 277,
        "bindings": 213,
        "events": 213,
        "legacy_v1_aliases": 64,
        "proposals": 0,
        "reviews": 0,
        "revisions": 211,
        "sources": 207,
        "stable_v2_aliases": 213,
    }


def test_registry_allowlist_binds_seeded_post_counts() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    spec = registry["migrations"]["d2e33e73c6d2"]
    assert spec["from_revision"] == "a19c7e4b2d60"
    assert spec["bootstrap_snapshot_sha256"] == hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest()
    assert spec["migration_sha256"] == hashlib.sha256(MIGRATION.read_bytes()).hexdigest()
    assert spec["expected_post_table_counts"] == {
        "source_registry_aliases": 277,
        "source_registry_bindings": 213,
        "source_registry_events": 213,
        "source_registry_proposals": 0,
        "source_registry_reviews": 0,
        "source_registry_revisions": 211,
        "source_registry_sources": 207,
    }


def test_sqlite_upgrade_and_downgrade_portability() -> None:
    migration = _load_migration()
    engine = create_engine("sqlite:///:memory:", future=True)
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE trusted_principals (id UUID PRIMARY KEY)")
        conn.exec_driver_sql("CREATE TABLE trusted_credentials (id UUID PRIMARY KEY)")
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            migration.upgrade()

        inspector = inspect(conn)
        expected = {
            "source_registry_sources": 207,
            "source_registry_revisions": 211,
            "source_registry_bindings": 213,
            "source_registry_aliases": 277,
            "source_registry_proposals": 0,
            "source_registry_reviews": 0,
            "source_registry_events": 213,
        }
        for table, count in expected.items():
            assert inspector.has_table(table)
            actual = conn.execute(text(f'SELECT COUNT(*) FROM "{table}"')).scalar_one()
            assert actual == count

        with Operations.context(ctx):
            migration.downgrade()

        inspector = inspect(conn)
        for table in expected:
            assert not inspector.has_table(table)
