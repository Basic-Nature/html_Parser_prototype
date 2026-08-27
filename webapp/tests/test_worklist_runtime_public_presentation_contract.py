from __future__ import annotations

import re
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME = REPO_ROOT / "webapp" / "templates" / "worklist.html"
PUBLIC_JS = REPO_ROOT / "webapp" / "static" / "js" / "workflow_public.js"
PUBLIC_CSS = REPO_ROOT / "webapp" / "static" / "css" / "workflow_public.css"


def test_runtime_worklist_is_w1_public_workflow_surface():
    runtime = RUNTIME.read_text(encoding="utf-8")

    assert "<h1>ElectionPulse Workflow</h1>" in runtime
    assert "Public verification workflow" in runtime
    assert "Explore Published Data" in runtime
    assert "Governed Workflow Plane" in runtime
    assert "workflow-empty-state" in runtime

    assert "filename='js/workflow_public.js'" in runtime
    assert "filename='css/workflow_public.css'" in runtime
    assert "smart_elections_worklist.js" not in runtime

    for retired_runtime_ui in (
        "Assign DL Owner",
        "DL Editor",
        "QC1 Checkpoint Review",
        "QC2 Final Review",
        "DL1 Operator",
        "DL2 Operator",
        "Worklist Source",
    ):
        assert retired_runtime_ui not in runtime


def test_runtime_workflow_has_no_duplicate_ids_and_is_csp_clean():
    runtime = RUNTIME.read_text(encoding="utf-8")
    counts = Counter(
        re.findall(r'\bid=["\']([^"\']+)["\']', runtime)
    )
    duplicates = {
        key: value
        for key, value in counts.items()
        if value > 1
    }

    assert duplicates == {}
    assert ' style="' not in runtime
    assert "nonce=\"{{ g.csp_nonce }}\"" in runtime
    assert "?v={{ static_version }}" in runtime

    headers = re.findall(r"<th\b[^>]*>", runtime)
    assert headers
    assert all('scope="col"' in header for header in headers)


def test_runtime_workflow_navigation_explains_surface_boundaries():
    runtime = RUNTIME.read_text(encoding="utf-8")

    assert "{{ url_for('ballot_lens') }}" in runtime
    assert "{{ url_for('data_framework') }}" in runtime
    assert "Work being verified and work still needed." in runtime
    assert "Published and reference election data" in runtime
    assert "Explore and analyze election results." in runtime


def test_runtime_public_workflow_js_uses_only_governed_get_reads():
    source = PUBLIC_JS.read_text(encoding="utf-8")

    for endpoint in (
        "/api/workflow/v1/public/items",
        "/api/workflow/v1/facets",
        "/api/workflow/v1/stats",
    ):
        assert endpoint in source

    for legacy_endpoint in (
        "/api/election_data/worklist",
        "/api/election_data/worklist/overview",
    ):
        assert legacy_endpoint not in source

    assert "method: 'GET'" in source
    assert "method: 'POST'" not in source
    assert "method: 'PUT'" not in source
    assert "method: 'DELETE'" not in source


def test_runtime_public_workflow_assets_have_accessibility_layer():
    css = PUBLIC_CSS.read_text(encoding="utf-8")

    assert "W1 PUBLIC WORKFLOW PARTICIPATION FOUNDATION" in css
    assert ":focus-visible" in css
    assert "prefers-reduced-motion: reduce" in css
    assert "scrollbar-gutter: stable" in css
    assert css.count("{") == css.count("}")
