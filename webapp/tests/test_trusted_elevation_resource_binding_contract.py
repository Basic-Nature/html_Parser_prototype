"""Resource-bound elevation schema/evaluator source contract."""
from __future__ import annotations
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "webapp/parser/auth/trusted_identity_models.py"
PROTECTED = ROOT / "webapp/parser/auth/protected_operation.py"
MIGRATION = ROOT / "alembic/versions/a19c7e4b2d60_trusted_elevation_resource_binding.py"


def _fields(source: str, name: str) -> set[str]:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            found = set()
            for item in node.body:
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name):
                            found.add(target.id)
            return found
    raise AssertionError(name)


def test_models_are_resource_bound_and_grant_is_consumable() -> None:
    source = MODELS.read_text(encoding="utf-8")
    assert {"resource_type", "resource_id", "resource_version"} <= _fields(source, "TrustedElevationChallenge")
    assert {"resource_type", "resource_id", "resource_version", "consumed_at"} <= _fields(source, "TrustedElevationGrant")


def test_migration_is_additive_schema_only() -> None:
    source = MIGRATION.read_text(encoding="utf-8")
    assert 'revision: str = "a19c7e4b2d60"' in source
    assert 'down_revision: str | None = "f18a7c3d4e92"' in source
    assert source.count("op.add_column(") == 7
    assert "bulk_insert" not in source
    assert "INSERT INTO" not in source.upper()


def test_evaluator_is_current_credential_session_resource_and_single_use_bound() -> None:
    source = PROTECTED.read_text(encoding="utf-8")
    for token in (
        "authorize_and_consume_protected_operation",
        "current_credential_id",
        "trusted_session_id",
        "browser_session_binding_hash",
        "resource_type",
        "resource_id",
        "resource_version",
        "with_for_update()",
        'trusted_session.state != "elevated"',
        "grant.credential_id != current_credential_id",
        "credential.principal_id != principal_id",
        'challenge.state != "consumed"',
        'grant.state = "consumed"',
        "grant.consumed_at = instant",
        'event_type="protected_operation_grant_consumed"',
    ):
        assert token in source
