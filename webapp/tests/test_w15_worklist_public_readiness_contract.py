from __future__ import annotations

from pathlib import Path

from webapp.parser.services.public_read_runtime import (
    _project_public_workflow_item,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = REPO_ROOT / "webapp" / "templates" / "worklist.html"
PUBLIC_JS = REPO_ROOT / "webapp" / "static" / "js" / "workflow_public.js"
PUBLIC_CSS = REPO_ROOT / "webapp" / "static" / "css" / "workflow_public.css"
RUNTIME = REPO_ROOT / "webapp" / "parser" / "services" / "public_read_runtime.py"
PRESENTATION = (
    REPO_ROOT / "webapp" / "tests"
    / "test_worklist_runtime_public_presentation_contract.py"
)
JEST = (
    REPO_ROOT / "webapp" / "static" / "js" / "__tests__"
    / "workflow_public.contract.test.js"
)


def test_w15_public_projection_exposes_safe_provenance_not_raw_url():
    raw = {
        "id": "workflow-123",
        "lifecycle_state": "active",
        "current_stage": "independent_acquisition",
        "stage_condition": "in_progress",
        "priority": 7,
        "scope": {
            "election_year": 2024,
            "state": "Arizona",
            "jurisdiction_name": "Pima",
            "jurisdiction_type": "county",
            "contest": "President",
            "source_race_id": "AZ-2024-PRES",
        },
        "source_url": "https://secret.example/raw",
        "canonical_reference": {
            "race_id": "internal-race-id",
            "linked": True,
        },
        "updated_at": "2026-09-11T00:00:00+00:00",
    }

    public = _project_public_workflow_item(raw)

    assert public["provenance"] == {
        "source_race_id": "AZ-2024-PRES",
        "canonical_linked": True,
        "lineage_inferred": False,
        "source_link_available": False,
    }
    assert "source_url" not in public
    assert "https://secret.example/raw" not in repr(public)
    assert "internal-race-id" not in repr(public)


def test_w15_template_has_public_safe_source_panel_and_keyboard_table_region():
    source = TEMPLATE.read_text(encoding="utf-8")

    assert 'class="workflow-skip-link"' in source
    assert 'href="#workflowMain"' in source
    assert 'id="workflowMain"' in source
    assert 'tabindex="-1"' in source
    assert '<h2 id="workflow-authority-title">Governed Workflow Plane</h2>' in source
    assert "Worklist Source" in source
    assert "Source Link" in source
    assert "raw workflow URLs withheld" in source
    assert 'class="workflow-table-wrap"' in source
    assert 'tabindex="0"' in source
    assert 'role="region"' in source
    assert 'aria-describedby="workflow-table-description"' in source
    assert '<caption class="workflow-sr-only">' in source
    assert '<th scope="col">Source record</th>' in source


def test_w15_client_decouples_auxiliary_reads_and_supports_explicit_states():
    source = PUBLIC_JS.read_text(encoding="utf-8")

    assert "Promise.allSettled([" in source
    assert "await Promise.all([" not in source
    assert "statsResult.status !== 'fulfilled'" in source
    assert "facetsResult.status !== 'fulfilled'" in source
    assert "itemsResult.status !== 'fulfilled'" in source
    assert "'restricted'" in source
    assert "'partial'" in source
    assert "'stale'" in source
    assert "items?.stale === true" in source
    assert "panel.dataset.uiState" in source
    assert "task?.provenance?.source_race_id" in source


def test_w15_client_preserves_trusted_handoff_without_raw_source_url_access():
    source = PUBLIC_JS.read_text(encoding="utf-8")

    assert "payload.source_url_disclosed !== false" in source
    assert "workflow_item_id" in source
    assert "workflow_pass_id" in source
    assert "expected_row_version" in source
    assert "task.source_url" not in source
    assert "payload['source_url']" not in source
    assert "source_url:" not in source


def test_w15_css_has_accessibility_target_and_forced_color_contract():
    source = PUBLIC_CSS.read_text(encoding="utf-8")

    assert ".workflow-skip-link" in source
    assert ".workflow-sr-only" in source
    assert ".workflow-table-wrap:focus-visible" in source
    assert "#workflowMain:focus-visible" in source
    assert "min-height: 44px" in source
    assert "@media (forced-colors: active)" in source
    assert "prefers-reduced-motion: reduce" in source
    assert source.count("{") == source.count("}")


def test_w15_retires_stale_source_substring_contract_without_weakening_redaction():
    presentation = PRESENTATION.read_text(encoding="utf-8")
    jest = JEST.read_text(encoding="utf-8")
    runtime = RUNTIME.read_text(encoding="utf-8")

    retired_tuple = presentation.split(
        "for retired_runtime_ui in (", 1
    )[1].split("):", 1)[0]
    assert '"Worklist Source"' not in retired_tuple
    assert "payload.source_url_disclosed !== false" in jest
    assert "'source_url'," not in jest
    assert '"raw_source_url": "omitted"' in runtime
    assert '"source_link_policy": "approved_registry_only"' in runtime
