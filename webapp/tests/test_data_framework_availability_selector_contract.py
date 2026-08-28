from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
JS = REPO_ROOT / "webapp" / "static" / "js" / "data_framework.js"


def _read() -> str:
    return JS.read_text(encoding="utf-8")


def _block(source: str, start: str, end: str) -> str:
    begin = source.index(start)
    finish = source.index(end, begin)
    return source[begin:finish]


def test_analysis_retains_canonical_universe_and_marks_current_availability():
    source = _read()

    assert "let canonicalFacetUniversePayload = null;" in source
    assert (
        "const universePayload = "
        "isCanonicalFacetPayload(canonicalFacetUniversePayload)"
        in source
    )
    assert (
        "option.dataset.availability = "
        "isAvailable ? 'available' : 'unavailable';"
        in source
    )
    assert "option.disabled = !isAvailable && value !== desired;" in source
    assert (
        "Valid canonical option; no rows match the other active filters."
        in source
    )


def test_warehouse_status_no_longer_owns_canonical_record_selectors():
    source = _read()

    assert "function hydratePriorityStates(payload)" not in source
    assert "function hydratePriorityYears(payload)" not in source

    priority = _block(
        source,
        "function applyPriorityPayload(payload",
        "async function fetchPriorityStatus()",
    )

    assert "hydratePriorityStates" not in priority
    assert "hydratePriorityYears" not in priority
    assert (
        "warehouse-status availability mutate Canonical Record scope"
        in priority
    )


def test_canonical_record_has_independent_self_excluding_facet_lane():
    source = _read()

    assert "let canonicalRecordFacetRequestSeq = 0;" in source
    assert "let canonicalRecordFacetAbortController = null;" in source
    assert "function getCanonicalRecordFacetFilters()" in source
    assert "function applyCanonicalRecordFacetPayload(payload)" in source
    assert (
        "async function fetchCanonicalRecordFacets"
        "({ useUniverse = false } = {})"
        in source
    )

    record_facets = _block(
        source,
        "async function fetchCanonicalRecordFacets",
        "async function fetchCanonicalFacets",
    )

    assert "canonicalRecordFacetRequestSeq" in record_facets
    assert "canonicalRecordFacetAbortController" in record_facets
    assert "AbortController()" in record_facets
    assert (
        "buildCanonicalFacetUrl(getCanonicalRecordFacetFilters())"
        in record_facets
    )
    assert "applyCanonicalRecordFacetPayload(result.data)" in record_facets


def test_canonical_universe_precedes_record_load_and_events_refresh_availability():
    source = _read()

    bootstrap = _block(
        source,
        "async function bootstrapProtectedFeeds()",
        "// Canonical publication rows are the sole election-result Analysis feed.",
    )

    assert (
        bootstrap.index("await fetchCanonicalFacets({ universe: true });")
        < bootstrap.index("fetchCanonicalRecordData(true);")
    )
    assert "await fetchCanonicalRecordFacets({ useUniverse: true });" in bootstrap
    assert "hydratePriorityStates" not in bootstrap
    assert "hydratePriorityYears" not in bootstrap

    events = _block(
        source,
        "el.priorityStateSelect?.addEventListener('change'",
        "el.curatedSearch?.addEventListener('input'",
    )

    assert events.count("fetchCanonicalRecordFacets();") == 2
    assert events.count("fetchCanonicalRecordData(true);") == 2

def test_bounded_result_rows_never_replace_canonical_state_universe():
    source = _read()
    update_states = _block(
        source,
        "function updateVizStates()",
        "function setVizState",
    )

    assert "Canonical facets own selector validity." in update_states
    assert "canonicalFacetUniversePayload" in update_states
    assert "universePayload.states" in update_states
    assert "facetPayload.states" in update_states
    assert "replaceCanonicalOptions(" in update_states

    # The HTML State selector must never be repopulated from the bounded
    # result rows. Those rows may be a 1,000-row subset spanning only a
    # fraction of the valid canonical State universe.
    assert "setSelectOptions(el.vizState" not in update_states


def test_data_framework_missing_value_token_is_not_mojibake():
    source = _read()

    assert "â€”" not in source
    assert r"\u2014" in source
