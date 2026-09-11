from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "webapp" / "templates" / "worklist.html"
JS = ROOT / "webapp" / "static" / "js" / "workflow_public.js"
CSS = ROOT / "webapp" / "static" / "css" / "workflow_public.css"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_w15c_live_state_and_pagination_controls_are_accessible():
    source = _read(TEMPLATE)

    state = source.split('id="workflow-state"', 1)[1].split(">", 1)[0]
    assert 'tabindex="-1"' in state
    assert 'role="status"' in state
    assert 'aria-live="polite"' in state
    assert 'aria-atomic="true"' in state

    pagination = source.split('id="workflow-pagination-summary"', 1)[1].split(
        ">", 1
    )[0]
    assert 'role="status"' in pagination
    assert 'aria-live="polite"' in pagination
    assert 'aria-atomic="true"' in pagination

    assert 'class="workflow-pagination"' in source
    for button_id in ("workflow-page-prev", "workflow-page-next"):
        pos = source.index(f'id="{button_id}"')
        tag_start = source.rfind("<button", 0, pos)
        tag_end = source.index(">", pos)
        tag = source[tag_start:tag_end]
        assert 'type="button"' in tag
        assert 'aria-controls="workflow-table"' in tag
        assert 'aria-disabled="true"' in tag
        assert "disabled" in tag


def test_w15c_filters_describe_live_summary_and_control_results():
    source = _read(TEMPLATE)

    for control_id in (
        "workflow-filter-state",
        "workflow-filter-year",
        "workflow-filter-lifecycle",
        "workflow-filter-search",
    ):
        pos = source.index(f'id="{control_id}"')
        tag_start = max(
            source.rfind("<input", 0, pos),
            source.rfind("<select", 0, pos),
        )
        tag_end = source.index(">", pos)
        tag = source[tag_start:tag_end]
        assert 'aria-describedby="workflow-filter-summary"' in tag

    for button_id in ("workflow-filter-apply", "workflow-filter-reset"):
        pos = source.index(f'id="{button_id}"')
        tag_start = source.rfind("<button", 0, pos)
        tag_end = source.index(">", pos)
        tag = source[tag_start:tag_end]
        assert 'type="button"' in tag
        assert (
            'aria-controls="workflow-table workflow-state '
            'workflow-pagination-summary"'
            in tag
        )


def test_w15c_client_has_page_authority_and_interaction_focus_contract():
    source = _read(JS)

    assert "this.pageOffset = 0;" in source
    assert "this.pageLimit = 200;" in source
    assert "renderPagination(payload)" in source
    assert "focusResultContext()" in source
    assert "applyFilters()" in source
    assert "changePage(direction)" in source
    assert "itemParams.set('offset', String(this.pageOffset));" in source
    assert "/api/workflow/v1/public/items?${itemQuery}" in source
    assert "this.load({ syncUrl: true, focusResults: true });" in source
    assert "this.load({ focusResults: true });" in source


def test_w15c_unavailable_facets_disable_alternatives_not_current_selection():
    source = _read(JS)

    assert "option.disabled = !available && value !== current;" in source
    assert "option.disabled ? 'true' : 'false'" in source
    assert "workflow-option-unavailable" in source


def test_w15c_abort_and_sequence_authority_remain_first_class():
    source = _read(JS)

    assert "const requestSeq = ++this.requestSeq;" in source
    assert "this.activeController.abort();" in source
    assert "new AbortController()" in source
    assert "requestSeq !== this.requestSeq" in source
    assert "this.activeController === controller" in source
    assert "Promise.allSettled([" in source
    assert "await Promise.all([" not in source

    assert "method: 'GET'" in source
    for token in (
        "method: 'POST'",
        "method: 'PUT'",
        "method: 'PATCH'",
        "method: 'DELETE'",
    ):
        assert token not in source


def test_w15c_css_supports_focus_disabled_pagination_and_mobile_reflow():
    source = _read(CSS)

    assert "/* W15C state interaction and accessible pagination. */" in source
    assert ".workflow-pagination {" in source
    assert ".workflow-button:disabled" in source
    assert '.workflow-button[aria-disabled="true"]' in source
    assert ".workflow-option-unavailable" in source
    assert "overscroll-behavior-x: contain" in source
    assert ".workflow-state:focus-visible" in source
    assert "@media (max-width: 620px)" in source
    assert "@media (forced-colors: active)" in source
    assert "min-height: 44px" in source
    assert source.count("{") == source.count("}")
