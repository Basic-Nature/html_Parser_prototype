from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "webapp" / "templates" / "data_framework.html"
CSS = ROOT / "webapp" / "static" / "css" / "data_framework.css"
JS = ROOT / "webapp" / "static" / "js" / "data_framework.js"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_skip_link_targets_a_focusable_main_landmark():
    template = _read(TEMPLATE)

    assert (
        '<a class="df-skip-link" href="#dataFrameworkMain">'
        'Skip to Data Framework content</a>'
        in template
    )
    assert (
        '<main class="container-main data-workbench-shell" '
        'id="dataFrameworkMain" tabindex="-1">'
        in template
    )
    assert template.count("<main ") == 1
    assert template.count("</main>") == 1
    assert template.index('class="df-skip-link"') < template.index(
        'class="ep-app-nav"'
    )


def test_canonical_record_table_has_keyboard_scroll_and_structural_context():
    template = _read(TEMPLATE)

    assert (
        'id="warehouseTableSection" role="region" '
        'aria-label="Canonical record table and controls" tabindex="0"'
        in template
    )
    assert (
        '<caption class="sr-only">'
        'Canonical election records matching the current filters.'
        '</caption>'
        in template
    )
    assert (
        'id="data-table" class="neon-grid" aria-live="polite" '
        'aria-describedby="tableStatus warehousePriorityMeta"'
        in template
    )
    assert (
        'id="pageInfo" class="page-info" role="status" '
        'aria-live="polite" aria-atomic="true"'
        in template
    )


def test_action_buttons_are_explicit_and_pagination_declares_table_control():
    template = _read(TEMPLATE)

    for button_id in (
        "refreshBtn",
        "exportCsvBtn",
        "scaffoldJsonBtn",
        "scaffoldCsvBtn",
        "resetFiltersBtn",
        "columnChooserBtn",
    ):
        marker = f'id="{button_id}"'
        pos = template.index(marker)
        tag_start = template.rfind("<button", 0, pos)
        tag_end = template.index(">", pos)
        tag = template[tag_start:tag_end]
        assert 'type="button"' in tag

    for button_id in (
        "firstPageBtn",
        "prevPageBtn",
        "nextPageBtn",
        "lastPageBtn",
    ):
        marker = f'id="{button_id}"'
        pos = template.index(marker)
        tag_start = template.rfind("<button", 0, pos)
        tag_end = template.index(">", pos)
        tag = template[tag_start:tag_end]
        assert 'type="button"' in tag
        assert 'aria-controls="data-table"' in tag

    assert (
        'id="uploadPipelineDetail" role="status" aria-live="polite" '
        'aria-atomic="true"'
        in template
    )
    assert (
        'id="uploadStatus" class="status status-info" role="status" '
        'aria-live="polite" aria-atomic="true"'
        in template
    )


def test_responsive_focus_touch_and_contrast_polish_is_present():
    css = _read(CSS)

    assert "/* W14D responsive/accessibility/presentation polish */" in css
    assert ".df-skip-link:focus" in css
    assert '.db-table-section[tabindex="0"]:focus-visible' in css
    assert ".curated-item:focus-visible" in css
    assert ".warehouse-upload-summary:focus-visible" in css
    assert "min-height:2.75rem" in css
    assert "@media (max-width:760px)" in css
    assert "@media (max-width:520px)" in css
    assert ".table-toolbar .right-controls" in css
    assert ".curated-metrics" in css
    assert ".warehouse-priority-controls" in css
    assert "overscroll-behavior-x:contain" in css
    assert "color:var(--df-text)" in css
    assert css.count("{") == css.count("}")


def test_motion_forced_colors_and_exact_cache_tokens_remain_safe():
    template = _read(TEMPLATE)
    css = _read(CSS)
    js_token = hashlib.sha256(JS.read_bytes()).hexdigest()[:16]
    css_token = hashlib.sha256(CSS.read_bytes()).hexdigest()[:16]

    assert "@media (prefers-reduced-motion:reduce)" in css
    assert "@media (forced-colors:active)" in css
    assert "transition:none !important" in css
    assert "animation:none !important" in css
    assert (
        "filename='js/data_framework.js', v='" + js_token + "'"
        in template
    )
    assert (
        "filename='css/data_framework.css', v='" + css_token + "'"
        in template
    )
