# Structural contract for Tranche 1B authority-policy extraction.

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
APP_PATH = ROOT / "webapp" / "Smart_Elections_Parser_Webapp.py"
POLICY_PATH = ROOT / "webapp" / "parser" / "auth" / "policy.py"

TARGETS = (
    "_auth_mode_requires_certificate",
    "_cert_required_response",
    "_require_client_cert",
    "_require_cert_for_socket_action",
)

OUT_OF_SCOPE = (
    "_sanitize_cert_metadata_for_status",
    "_require_health_auth",
    "api_auth_certificate_info",
    "handle_ack_cert_reauth",
)


def _top_level_functions(path: Path) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


def test_extracted_policy_module_owns_target_implementations():
    functions = _top_level_functions(POLICY_PATH)

    assert set(TARGETS).issubset(functions)
    assert "configure_runtime" in functions
    assert "_runtime_binding" in functions


def test_composition_root_keeps_policy_compatibility_wrappers():
    functions = _top_level_functions(APP_PATH)
    app_source = APP_PATH.read_text(encoding="utf-8")

    assert "_configure_authority_policy_runtime" in functions

    for name in TARGETS:
        node = functions[name]
        source = ast.get_source_segment(app_source, node)
        assert source is not None
        assert "_configure_authority_policy_runtime()" in source
        assert "_authority_policy." in source
        assert len(node.body) == 2


def test_extracted_policy_does_not_absorb_later_tranches():
    functions = _top_level_functions(POLICY_PATH)

    for name in OUT_OF_SCOPE:
        assert name not in functions


def test_extracted_policy_does_not_import_composition_root():
    source = POLICY_PATH.read_text(encoding="utf-8")

    assert "from webapp.Smart_Elections_Parser_Webapp" not in source
    assert "import webapp.Smart_Elections_Parser_Webapp" not in source
