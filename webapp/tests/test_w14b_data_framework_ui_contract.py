from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "webapp" / "Smart_Elections_Parser_Webapp.py"
TEMPLATE = ROOT / "webapp" / "templates" / "data_framework.html"
JS = ROOT / "webapp" / "static" / "js" / "data_framework.js"
SERVICE = ROOT / "webapp" / "parser" / "services" / "data_framework_operator_access.py"


def _function_source(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    lines = source.splitlines()
    matches = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    assert len(matches) == 1
    node = matches[0]
    return "\n".join(lines[node.lineno - 1 : node.end_lineno])


def test_data_framework_page_uses_server_projected_operator_access():
    body = _function_source(MAIN, "data_framework")
    template = TEMPLATE.read_text(encoding="utf-8")
    assert "get_request_principal()" in body
    assert "resolve_data_framework_operator_access(" in body
    assert "data_framework_operator_access=operator_access" in body
    assert "bool(principal)" not in body
    assert "data-operator-access=" in template
    assert "{% if data_framework_operator_access.can_upload_input %}" in template
    assert 'id="dataFrameworkUploadRestricted"' in template
    assert "Public Data Framework remains view-only." in template
    assert "Operator Access" in template


def test_upload_server_gate_is_unchanged_and_browser_does_not_infer_permission():
    main = MAIN.read_text(encoding="utf-8")
    upload = _function_source(MAIN, "upload_to_input")
    service = SERVICE.read_text(encoding="utf-8")
    template = TEMPLATE.read_text(encoding="utf-8")
    assert '_require_client_cert("upload_input")' in upload
    assert "assert_trusted_action(" in service
    assert "classify_authority(" in service
    assert '"principal_disclosed": False' in service
    assert "can_upload_input = bool(principal)" not in main
    assert "can_upload_input = bool(principal)" not in service
    assert "data_framework_operator_access.can_upload_input" in template


def test_restrictions_are_surface_scoped_not_global():
    src = JS.read_text(encoding="utf-8")
    assert "function enterSurfaceRestrictedMode(" in src
    assert "restrictionSurface" in src
    assert re.search(r"enterSurfaceRestrictedMode\(\s*restrictionSurface,", src)
    assert "authRestrictedMode" not in src
    assert "_authRestrictionReason" not in src
    assert "authRestrictionNotified" not in src
    assert "if (authRestrictedMode) return;" not in src
    for surface in ("'priority'", "'curated'", "'analysis'", "'canonical-record'", "'scaffold'"):
        assert surface in src


def test_public_canonical_semantics_and_deferred_dropoff_are_preserved():
    template = TEMPLATE.read_text(encoding="utf-8")
    src = JS.read_text(encoding="utf-8")
    assert "DB-Lite may retain legacy County/District labels" not in template
    assert "Canonical rows preserve jurisdiction name and type separately" in template
    assert "Drop-off — Pending" in template
    assert "Governed canonical drop-off derivation is pending" in template
    assert "G3.1C2: Data Framework election-result consumers are canonical-only." in src
    assert "payload.semantic_contract?.null === 'preserved_null'" in src
    assert "payload.semantic_contract?.facet_mode === 'self_excluding'" in src
    assert "const VIZ_INTERACTION_PREVIEW = 'preview';" in src
    assert "const VIZ_INTERACTION_EXPLORE = 'explore';" in src
