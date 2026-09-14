from __future__ import annotations

import hashlib
from pathlib import Path

from jinja2 import Environment


ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "webapp" / "templates" / "data_framework.html"
CSS = ROOT / "webapp" / "static" / "css" / "data_framework.css"
JS = ROOT / "webapp" / "static" / "js" / "data_framework.js"


def test_w16d_status_key_documents_existing_eight_state_model() -> None:
    template = TEMPLATE.read_text(encoding="utf-8")

    assert 'id="readinessStateGuide"' in template
    assert "<summary>" in template
    assert "Status key" in template
    assert 'role="list"' in template
    assert 'aria-label="Data Framework readiness states"' in template

    expected = {
        "idle": "Waiting for a scoped operation.",
        "loading": "A governed request is in progress.",
        "ready": "The current authoritative view is loaded.",
        "empty": "The scope is valid; no rows match.",
        "partial": "The bounded result may be incomplete.",
        "stale": "Cached evidence is shown while refresh is needed.",
        "restricted": "This surface requires governed operator access.",
        "error": "The request failed; prior authority is not silently replaced.",
    }
    for state, meaning in expected.items():
        assert f'data-readiness-state="{state}"' in template
        assert meaning in template


def test_w16d_live_status_surfaces_gain_consistent_visual_state_labels() -> None:
    css = CSS.read_text(encoding="utf-8")

    assert "/* W16D bounded readiness-presentation polish." in css
    assert '[data-ui-state]:not([data-ui-state="idle"])::before' in css
    assert "content:attr(data-ui-state);" in css

    for state in (
        "loading",
        "ready",
        "empty",
        "partial",
        "stale",
        "restricted",
        "error",
    ):
        assert f'[data-ui-state="{state}"]' in css

    assert "@media (max-width:760px)" in css
    assert "@media (max-width:520px)" in css
    assert "@media (prefers-reduced-motion:reduce)" in css
    assert "@media (forced-colors:active)" in css
    assert css.count("{") == css.count("}")


def test_w16d_polish_does_not_change_runtime_state_or_authority_model() -> None:
    source = JS.read_text(encoding="utf-8")

    for state in (
        "idle",
        "loading",
        "ready",
        "empty",
        "partial",
        "stale",
        "restricted",
        "error",
    ):
        assert f"'{state}'" in source

    assert "function enterSurfaceRestrictedMode(" in source
    assert "canonicalFacetAbortController.abort()" in source
    assert "canonicalDataAbortController.abort()" in source
    assert "canonicalRecordFacetAbortController.abort()" in source
    assert "canonicalRecordAbortController.abort()" in source
    assert "payload.semantic_contract?.null === 'preserved_null'" in source
    assert "payload.semantic_contract?.facet_mode === 'self_excluding'" in source


def test_w16d_evidence_hierarchy_and_cache_token_are_safe() -> None:
    template = TEMPLATE.read_text(encoding="utf-8")
    css = CSS.read_text(encoding="utf-8")

    assert 'id="evidenceContextBar"' in template
    assert "No lineage is inferred." in template
    assert ".evidence-context-summary > strong" in css
    assert ".evidence-relation > strong" in css

    css_token = hashlib.sha256(CSS.read_bytes()).hexdigest()[:16]
    assert (
        "filename='css/data_framework.css', v='" + css_token + "'"
        in template
    )
    Environment().parse(template)
