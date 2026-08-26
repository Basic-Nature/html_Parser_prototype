from __future__ import annotations

import ast
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

APP = REPO_ROOT / "webapp" / "Smart_Elections_Parser_Webapp.py"
WORKLIST_TEMPLATE = REPO_ROOT / "webapp" / "templates" / "worklist.html"
WORKLIST_CSS = REPO_ROOT / "webapp" / "static" / "css" / "smart_elections.css"
WORKLIST_JS = REPO_ROOT / "webapp" / "static" / "js" / "smart_elections_worklist.js"
BALLOT_JS = REPO_ROOT / "webapp" / "static" / "js" / "ballot_lens_modern.js"


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


def test_raw_worklist_identity_is_not_a_public_row_render():
    source = WORKLIST_JS.read_text(encoding="utf-8")

    start = source.index("renderRaceRow(race) {")
    end = source.index("Render status badge", start)
    block = source[start:end]

    assert "publicOperatorId(value, prefix = 'DT')" in source
    assert "this.publicOperatorId(race.dl1_assigned_to, 'DT')" in block
    assert "this.publicOperatorId(race.dl2_assigned_to, 'DT')" in block
    assert "this.escapeHtml(race.dl1_assigned_to" not in block
    assert "this.escapeHtml(race.dl2_assigned_to" not in block


def test_worklist_template_is_csp_clean_and_accessibility_polished():
    template = WORKLIST_TEMPLATE.read_text(encoding="utf-8")

    assert ' style="' not in template
    assert "?v={{ static_version }}" in template
    assert "DL1 Operator" in template
    assert "DL2 Operator" in template
    assert "stable aliases" in template
    assert 'aria-live="polite"' in template

    assert "<thead>" in template
    assert "</thead>" in template
    assert '<th scope="col"ead>' not in template

    headers = re.findall(r"<th\b[^>]*>", template)
    assert headers
    assert all('scope="col"' in header for header in headers)


def test_worklist_css_has_prelaunch_responsive_accessibility_layer():
    css = WORKLIST_CSS.read_text(encoding="utf-8")

    assert "G3.1C2.14 PRE-LAUNCH POLISH" in css
    assert "min-width: 1480px" in css
    assert ".btn:focus-visible" in css
    assert "prefers-reduced-motion: reduce" in css
    assert "scrollbar-gutter: stable" in css
    assert css.count("{") == css.count("}")


def test_ballot_lens_worklist_import_matches_server_records_contract():
    source = BALLOT_JS.read_text(encoding="utf-8")

    start = source.index("function openWorklistModal()")
    end = source.index("function populateWorklistTable", start)
    block = source[start:end]

    assert "Array.isArray(data.records)" in block
    assert "worklistData = data.records;" in block
    assert "Array.isArray(data.urls)" not in block
    assert "worklistData = data.urls;" not in block


def test_ballot_lens_worklist_records_map_real_source_url_fields():
    source = BALLOT_JS.read_text(encoding="utf-8")

    start = source.index("function importSelectedUrls()")
    end = source.index("// Event listeners", start)
    block = source[start:end]

    assert "rowData['Download 1']" in block
    assert "rowData['Download 2']" in block
    assert "rowData['Source Link']" in block


def test_ballot_lens_previews_never_fabricate_candidate_or_vote_rows():
    source = BALLOT_JS.read_text(encoding="utf-8")

    # Parser-session table previews remain sourced from parser_output payloads.
    assert "rows_preview" in source
    assert "TablePreviewManager.record" in source

    # Result-card modal consumes only the real canonical preview payload.
    assert "function displayExtractedResultPreview(result)" in source
    assert "displayExtractedResultPreview(result);" in source
    assert "result?.preview" in source
    assert (
        "No extracted preview payload is available for this canonical result."
        in source
    )

    for fabricated in (
        "Alice Johnson",
        "Bob Smith",
        "Alice Brown",
        "County Attorney",
        "sampleData",
        "sampleJson",
        "displayTablePreview(",
        "displayJsonPreview(",
    ):
        assert fabricated not in source


def test_known_ballot_lens_mojibake_em_dash_is_retired():
    source = BALLOT_JS.read_text(encoding="utf-8")
    assert "\u00e2\u20ac\u201d" not in source
