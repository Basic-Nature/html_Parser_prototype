"""Structural contract for Tranche 1C authority-status extraction."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP_PATH = ROOT / "webapp" / "Smart_Elections_Parser_Webapp.py"
STATUS_PATH = ROOT / "webapp" / "parser" / "auth" / "status.py"

TARGETS = (
    "_sanitize_cert_metadata_for_status",
    "api_auth_status",
    "api_auth_certificate_info",
)

OUT_OF_SCOPE = (
    "_require_health_auth",
    "_health_auth_response",
    "auth_welcome",
    "auth_challenge",
    "handle_connect",
    "handle_ack_cert_reauth",
)

REQUIRED_STATUS_FIELDS = (
    '"authenticated"',
    '"certificate_present"',
    '"certificate_required_for_mutations"',
    '"certificate_action_required"',
    '"principal"',
    '"principal_source"',
    '"cert_metadata"',
    '"privilege_tier"',
    '"privilege_level"',
    '"privilege_display"',
    '"certificate_policy"',
    '"azure_client_cert_mode"',
    '"challenge_url"',
    '"auth_url"',
    '"status_source"',
    '"session_context"',
    '"certificate_proof_cached"',
    '"timestamp"',
)


def _top_level_functions(path: Path) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


def test_status_module_owns_primary_1c_implementations():
    functions = _top_level_functions(STATUS_PATH)
    assert set(TARGETS).issubset(functions)
    assert "configure_runtime" in functions
    assert "_runtime_binding" in functions


def test_composition_root_keeps_status_compatibility_wrappers():
    functions = _top_level_functions(APP_PATH)
    source = APP_PATH.read_text(encoding="utf-8")
    assert "_configure_authority_status_runtime" in functions

    for name in TARGETS:
        node = functions[name]
        segment = ast.get_source_segment(source, node)
        assert segment is not None
        assert "_configure_authority_status_runtime()" in segment
        assert "_authority_status." in segment


def test_status_module_preserves_canonical_contract_fields():
    source = STATUS_PATH.read_text(encoding="utf-8")
    for field in REQUIRED_STATUS_FIELDS:
        assert field in source


def test_status_module_does_not_absorb_later_tranches():
    functions = _top_level_functions(STATUS_PATH)
    for name in OUT_OF_SCOPE:
        assert name not in functions

    source = STATUS_PATH.read_text(encoding="utf-8")
    assert "class PrivilegeTier" not in source
    assert "def get_principal_tier" not in source


def test_status_module_does_not_import_composition_root():
    source = STATUS_PATH.read_text(encoding="utf-8")
    assert "from webapp.Smart_Elections_Parser_Webapp" not in source
    assert "import webapp.Smart_Elections_Parser_Webapp" not in source


def test_certificate_info_remains_status_compatibility_alias():
    functions = _top_level_functions(STATUS_PATH)
    source = STATUS_PATH.read_text(encoding="utf-8")
    node = functions["api_auth_certificate_info"]
    segment = ast.get_source_segment(source, node)
    assert segment is not None
    assert '_runtime_binding("api_auth_status")' in segment
    assert "return api_auth_status()" in segment
