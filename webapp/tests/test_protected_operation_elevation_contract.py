"""Preview static protected-operation contract tests."""
from __future__ import annotations
from pathlib import Path

def _source() -> str:
    return (
        Path(__file__).resolve().parents[1]
        / "parser" / "auth" / "protected_operation.py"
    ).read_text(encoding="utf-8")

def test_fail_closed_codes_are_present() -> None:
    source = _source()
    for code in (
        "deny_identity",
        "deny_principal_trust",
        "deny_capability",
        "deny_elevation",
        "deny_credential",
    ):
        assert code in source

def test_allow_occurs_after_live_rechecks() -> None:
    source = _source()
    allow_index = source.rindex('return ProtectedOperationDecision(True, "allow", grant_id)')
    for marker in (
        "principal.state",
        "capabilities_for_principal",
        "grant.state",
        "credential.state",
        "credential.revocation_state",
        "credential.not_after",
    ):
        assert source.index(marker) < allow_index
