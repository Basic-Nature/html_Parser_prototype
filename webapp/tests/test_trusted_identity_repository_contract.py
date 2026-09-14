"""Preview static repository contract tests."""
from __future__ import annotations
from pathlib import Path

def _source() -> str:
    return (
        Path(__file__).resolve().parents[1]
        / "parser" / "auth" / "trusted_identity_repository.py"
    ).read_text(encoding="utf-8")

def test_fingerprint_migration_is_not_automatic() -> None:
    source = _source()
    assert "automatic migration is forbidden" in source
    assert "migrate_fingerprint_candidate" in source

def test_capabilities_are_server_derived_from_roles() -> None:
    source = _source()
    assert "capabilities_for_roles" in source
    tail = source.split("def capabilities_for_principal", 1)[1]
    assert "fingerprint_sha256" not in tail
