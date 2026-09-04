from __future__ import annotations

import ast
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

APP = REPO_ROOT / "webapp" / "Smart_Elections_Parser_Webapp.py"
WORKLIST_TEMPLATE = REPO_ROOT / "webapp" / "templates" / "worklist.html"
WORKLIST_CSS = REPO_ROOT / "webapp" / "static" / "css" / "workflow_public.css"
WORKLIST_JS = REPO_ROOT / "webapp" / "static" / "js" / "workflow_public.js"


def _function_source(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            segment = ast.get_source_segment(source, node)
            assert segment is not None
            return segment

    raise AssertionError(f"function not found: {name}")


def test_worklist_public_projection_and_overview_keep_distinct_boundaries():
    public_body = _function_source(APP, "api_election_data_worklist")

    principal_index = public_body.index("get_request_principal()")
    projection_index = public_body.index("public_projection = not bool(principal)")
    capability_index = public_body.index("assert_public_read_surface(")
    surface_index = public_body.index(
        '"election_data_worklist_public_projection"'
    )

    assert principal_index < projection_index < capability_index < surface_index
    assert "not principal and not ALLOW_DEV_NO_PRINCIPAL" not in public_body
    assert 'public_record[sensitive_field] = None' in public_body
    assert 'public_record["visibility"] = "public_projection"' in public_body

    for marker in ("create_engine", "fetch_worklist_overview"):
        if marker in public_body:
            assert surface_index < public_body.index(marker)

    overview_body = _function_source(
        APP,
        "api_election_data_worklist_overview",
    )
    overview_principal_index = overview_body.index("get_request_principal()")
    overview_guard_index = overview_body.index(
        "not principal and not ALLOW_DEV_NO_PRINCIPAL"
    )
    overview_unauthorized_index = min(
        value
        for value in (
            overview_body.find('"Unauthorized"'),
            overview_body.find("'Unauthorized'"),
        )
        if value >= 0
    )

    assert ", 403" in overview_body
    assert (
        overview_principal_index
        < overview_guard_index
        < overview_unauthorized_index
    )
    assert "assert_public_read_surface(" not in overview_body
    assert overview_unauthorized_index < overview_body.index(
        "fetch_worklist_overview"
    )


def test_raw_workflow_identity_is_not_present_in_public_runtime():
    source = WORKLIST_JS.read_text(encoding="utf-8")

    assert "/api/workflow/v1/public/items" in source
    assert "/api/election_data/worklist" not in source

    for identity_token in (
        "created_by_principal",
        "assigned_principal",
        "reviewer_principal",
        "resolved_by_principal",
        "actor_principal",
        "workflow_metadata",
    ):
        assert identity_token not in source


def test_worklist_template_is_csp_clean_and_accessibility_polished():
    template = WORKLIST_TEMPLATE.read_text(encoding="utf-8")

    assert ' style="' not in template
    assert "?v={{ static_version }}" in template
    assert "ElectionPulse Workflow" in template
    assert "identity-safe public projection" in template
    assert 'aria-live="polite"' in template
    assert "filename='js/workflow_public.js'" in template
    assert "smart_elections_worklist.js" not in template

    assert "<thead>" in template
    assert "</thead>" in template
    assert '<th scope="col"ead>' not in template

    headers = re.findall(r"<th\b[^>]*>", template)
    assert headers
    assert all('scope="col"' in header for header in headers)


def test_worklist_css_has_prelaunch_responsive_accessibility_layer():
    css = WORKLIST_CSS.read_text(encoding="utf-8")

    assert "W1 PUBLIC WORKFLOW PARTICIPATION FOUNDATION" in css
    assert ":focus-visible" in css
    assert "prefers-reduced-motion: reduce" in css
    assert "scrollbar-gutter: stable" in css
    assert "@media (max-width: 620px)" in css
    assert css.count("{") == css.count("}")


