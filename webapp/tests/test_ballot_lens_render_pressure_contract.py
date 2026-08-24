from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BALLOT_JS = REPO_ROOT / "webapp" / "static" / "js" / "ballot_lens_modern.js"
BALLOT_TEMPLATE = REPO_ROOT / "webapp" / "templates" / "ballot_lens.html"


def test_results_grid_does_not_rewrap_itself_for_card_height_sync():
    source = BALLOT_JS.read_text(encoding="utf-8")

    assert "function syncCardHeights()" not in source
    assert "debouncedSyncCardHeights" not in source
    assert "_cardSizeObserver" not in source
    assert "grid.appendChild(wrapper)" not in source
    assert "wrapper.appendChild(c)" not in source


def test_results_preview_observer_remains_without_card_rewrap_observer():
    source = BALLOT_JS.read_text(encoding="utf-8")

    assert (
        "gridObserver.observe(resultsGrid, { childList: true, subtree: true });"
        in source
    )

    assert (
        "ro.observe(resultsGrid, "
        "{ childList: true, subtree: true, attributes: true });"
        not in source
    )


def test_canonical_result_loading_has_one_initial_and_one_manual_call_authority():
    source = BALLOT_JS.read_text(encoding="utf-8")

    # One function definition, one trusted refresh call, one initial call.
    assert len(re.findall(r"\bloadRealData\s*\(", source)) == 3
    assert source.count("await loadRealData();") == 1
    assert source.count("  loadRealData();") == 1


def test_ballot_lens_script_and_refresh_button_are_single_authorities():
    template = BALLOT_TEMPLATE.read_text(encoding="utf-8")

    assert template.count("ballot_lens_modern.js") == 1
    assert len(
        re.findall(
            r'id=["\']btnRefreshResults["\']',
            template,
        )
    ) == 1
