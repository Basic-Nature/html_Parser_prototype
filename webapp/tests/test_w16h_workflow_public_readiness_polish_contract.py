from __future__ import annotations

from pathlib import Path
from jinja2 import Environment

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "webapp" / "templates" / "worklist.html"
CSS = ROOT / "webapp" / "static" / "css" / "workflow_public.css"
JS = ROOT / "webapp" / "static" / "js" / "workflow_public.js"
RUNTIME = ROOT / "webapp" / "parser" / "services" / "public_read_runtime.py"
READER = ROOT / "webapp" / "parser" / "services" / "workflow_reader.py"


def test_w16h_public_operator_source_boundary_is_explicit_without_new_controls() -> None:
    template = TEMPLATE.read_text(encoding="utf-8")
    assert 'id="workflow-public-boundary"' in template
    assert "Public visibility and operator authority stay separate" in template
    assert "View the verification queue" in template
    assert "Approved registry links only" in template
    assert "Operator Workbench is separate" in template
    assert "raw workflow URLs stay withheld" in template
    assert "do not influence public search" in template
    assert "server-projected" in template
    assert "revalidated before any eligible Ballot Lens handoff" in template
    block = template.split('id="workflow-public-boundary"', 1)[1].split("</section>", 1)[0]
    for forbidden in ("<button", "<form", "<input", "<select", "source_url"):
        assert forbidden not in block
    Environment().parse(template)


def test_w16h_visual_state_labels_consume_existing_data_ui_state_only() -> None:
    css = CSS.read_text(encoding="utf-8")
    assert "/* W16H bounded Workflow public-readiness presentation polish." in css
    assert '#workflow-state[data-ui-state]:not([data-ui-state="idle"])::before' in css
    assert '#workflow-empty-state[data-ui-state]:not([data-ui-state="idle"])::before' in css
    assert "content: attr(data-ui-state);" in css
    for state in ("ready", "partial", "stale", "restricted", "unavailable", "error"):
        assert f'#workflow-state[data-ui-state="{state}"]::before' in css
    assert "@media (max-width: 980px)" in css
    assert "@media (max-width: 620px)" in css
    assert "@media (prefers-reduced-motion: reduce)" in css
    assert "@media (forced-colors: active)" in css
    assert css.count("{") == css.count("}")


def test_w16h_does_not_expand_public_browser_authority() -> None:
    source = JS.read_text(encoding="utf-8")
    assert "Anonymous governed Workflow visibility. GET-only by design." in source
    assert "method: 'GET'" in source
    for token in ("method: 'POST'", "method: 'PUT'", "method: 'PATCH'", "method: 'DELETE'"):
        assert token not in source
    assert "this.pageLimit = 200;" in source
    assert "this.activeController.abort();" in source
    assert "requestSeq !== this.requestSeq" in source
    assert "option.disabled = !available && value !== current;" in source
    assert "workflow_operator_access_v1" in source
    assert "workflow_ballot_lens_handoff_v1" in source
    assert "payload.source_url_disclosed !== false" in source


def test_w16h_redaction_registry_and_hidden_url_search_boundaries_remain_frozen() -> None:
    runtime = RUNTIME.read_text(encoding="utf-8")
    reader = READER.read_text(encoding="utf-8")
    template = TEMPLATE.read_text(encoding="utf-8")
    assert '"raw_source_url": "omitted"' in runtime
    assert '"source_link_policy": "approved_registry_only"' in runtime
    assert '"null": "preserved_null"' in runtime
    assert runtime.count("include_source_url_search=False") >= 3
    assert "include_source_url_search: bool = True" in reader
    assert "if include_source_url_search:" in reader
    assert "WorkflowItem.source_url" in reader
    assert "This public surface remains view-only." in template
    assert "authenticated contributor authority" in template
    assert "smart_elections_worklist.js" not in template


def test_w16h_provenance_and_narrow_layout_hierarchy_are_presentation_only() -> None:
    css = CSS.read_text(encoding="utf-8")
    template = TEMPLATE.read_text(encoding="utf-8")
    assert ".workflow-public-boundary-grid" in css
    assert ".workflow-source-details > div" in css
    assert ".workflow-source-details dd" in css
    assert ".workflow-filter-summary" in css
    assert "#workflow-pagination-summary" in css
    assert "Governed Workflow Plane" in template
    assert "Approved registry sources only · raw workflow URLs withheld" in template
    assert 'placeholder="Contest, jurisdiction, source race ID"' in template
    assert 'class="workflow-table-wrap"' in template
    assert 'role="region"' in template
