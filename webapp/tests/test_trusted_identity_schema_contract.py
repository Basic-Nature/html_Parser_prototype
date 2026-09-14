"""Trusted identity schema source contract tests."""
from __future__ import annotations

from pathlib import Path

REVISION = "f18a7c3d4e92"
DOWN_REVISION = "e7b2c4d91f60"
MIGRATION_NAME = (
    "f18a7c3d4e92_trusted_identity_authority_foundation.py"
)

TRUSTED_TABLES = {
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


def test_exact_trusted_table_set() -> None:
    assert len(TRUSTED_TABLES) == 10


def test_deployable_migration_has_exact_parent_and_no_preview_guard() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "alembic"
        / "versions"
        / MIGRATION_NAME
    ).read_text(encoding="utf-8")

    assert f'revision: str = "{REVISION}"' in source
    assert f'down_revision: str | None = "{DOWN_REVISION}"' in source
    assert "PREVIEW_ONLY" not in source
    assert "W18T5_PREVIEW_ONLY_PENDING_LIVE_HEAD" not in source

    for table in sorted(TRUSTED_TABLES):
        assert f'"{table}"' in source

    assert "canonical_election_results" not in source
    assert "workflow_items" not in source


def test_migration_does_not_seed_identity_authority() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "alembic"
        / "versions"
        / MIGRATION_NAME
    ).read_text(encoding="utf-8")

    assert "bulk_insert" not in source
    assert "ROOT_ADMIN_CERT_FINGERPRINTS" not in source
    assert "ADMIN_JWT_TOKEN" not in source
    assert "INSERT INTO" not in source.upper()
