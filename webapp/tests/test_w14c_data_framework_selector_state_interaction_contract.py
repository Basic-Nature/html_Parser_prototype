from __future__ import annotations

import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JS = ROOT / "webapp" / "static" / "js" / "data_framework.js"
CSS = ROOT / "webapp" / "static" / "css" / "data_framework.css"
TEMPLATE = ROOT / "webapp" / "templates" / "data_framework.html"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _block(source: str, start: str, end: str) -> str:
    begin = source.find(start)
    assert begin >= 0, f"start marker missing: {start!r}"
    finish = source.find(end, begin + len(start))
    assert finish > begin, (
        f"end marker missing or not after start: "
        f"start={start!r} end={end!r}"
    )
    return source[begin:finish]


def test_all_canonical_analysis_selectors_preserve_universe_and_current_availability():
    source = _read(JS)

    counties = _block(
        source,
        "function updateVizCounties()",
        "function updateTopRaces()",
    )
    contests = _block(
        source,
        "function updateTopRaces()",
        "function ensureVizSelectionHasData()",
    )

    for block in (counties, contests):
        assert "canonicalFacetUniversePayload" in block
        assert "canonicalFacetPayload" in block
        assert "replaceCanonicalOptions(" in block

    assert "universePayload.jurisdictions" in counties
    assert "facetPayload.jurisdictions" in counties
    assert "universePayload.contests" in contests
    assert "facetPayload.contests" in contests

    # Bounded result rows may rank Preview frames, but they must not own
    # browser selector validity once canonical facet authority is available.
    assert "setSelectOptions(el.vizCounty" not in counties
    assert "setSelectOptions(el.vizContest" not in contests


def test_partial_and_stale_are_first_class_ui_states():
    source = _read(JS)
    css = _read(CSS)

    states = _block(
        source,
        "const UI_STATES = new Set([",
        "function normalizeUiState",
    )
    assert "'partial'" in states
    assert "'stale'" in states
    assert '[data-ui-state="partial"]' in css
    assert '[data-ui-state="stale"]' in css

    assert "analysisRowsPossiblyTruncated" in source
    assert "API cap reached; totals may be partial" in source
    assert re.search(
        r"API cap reached; totals may be partial`,\s*'partial'",
        source,
    )
    assert re.search(
        r"API cap reached, result may be partial\.`,\s*'partial'",
        source,
    )

    priority = _block(
        source,
        "function applyPriorityPayload(payload",
        "async function fetchPriorityStatus()",
    )
    assert "fromCache ? 'stale' : 'ready'" in priority


def test_request_authority_prevents_stale_response_overwrite():
    source = _read(JS)

    analysis_facets = _block(
        source,
        "async function fetchCanonicalFacets",
        "function refreshCanonicalExploreScope",
    )
    analysis_data = _block(
        source,
        "function fetchData(showLoading = false)",
        "function fetchCanonicalRecordData",
    )
    record_facets = _block(
        source,
        "async function fetchCanonicalRecordFacets",
        "async function fetchCanonicalFacets",
    )
    record_data = _block(
        source,
        "function fetchCanonicalRecordData",
        "// ---------- Init ----------",
    )

    assert "const requestSeq = ++canonicalFacetRequestSeq;" in analysis_facets
    assert "requestSeq !== canonicalFacetRequestSeq" in analysis_facets
    assert "canonicalFacetAbortController.abort()" in analysis_facets

    assert "const requestSeq = ++canonicalDataRequestSeq;" in analysis_data
    assert "requestSeq !== canonicalDataRequestSeq" in analysis_data
    assert "canonicalDataAbortController.abort()" in analysis_data

    assert "const requestSeq = ++canonicalRecordFacetRequestSeq;" in record_facets
    assert "requestSeq !== canonicalRecordFacetRequestSeq" in record_facets
    assert "canonicalRecordFacetAbortController.abort()" in record_facets

    assert "const requestSeq = ++canonicalRecordRequestSeq;" in record_data
    assert "requestSeq !== canonicalRecordRequestSeq" in record_data
    assert "canonicalRecordAbortController.abort()" in record_data


def test_preview_explore_separation_remains_authoritative():
    source = _read(JS)

    assert "const VIZ_INTERACTION_PREVIEW = 'preview';" in source
    assert "const VIZ_INTERACTION_EXPLORE = 'explore';" in source
    assert "vizInteractionMode !== VIZ_INTERACTION_EXPLORE" in source
    assert "vizInteractionMode !== VIZ_INTERACTION_PREVIEW" in source
    assert "vizAutoLocked" in source
    assert "stopVizAutoRotation()" in source


def test_asset_cache_tokens_match_exact_candidate_bytes():
    template = _read(TEMPLATE)
    js_token = hashlib.sha256(JS.read_bytes()).hexdigest()[:16]
    css_token = hashlib.sha256(CSS.read_bytes()).hexdigest()[:16]

    assert "filename='js/data_framework.js', v='" + js_token + "'" in template
    assert "filename='css/data_framework.css', v='" + css_token + "'" in template
