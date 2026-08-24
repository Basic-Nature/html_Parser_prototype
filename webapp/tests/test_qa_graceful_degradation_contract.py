from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
INTEGRATION = (
    REPO_ROOT
    / "webapp"
    / "static"
    / "js"
    / "quality_assurance_integration.js"
)
PANEL = (
    REPO_ROOT
    / "webapp"
    / "static"
    / "js"
    / "quality_assurance_panel.js"
)


def _classify_function(source: str) -> str:
    start = source.index("  async function classifyVisibleResults() {")
    end = source.index(
        "  /**\n   * Clear classified results cache",
        start,
    )
    return source[start:end]


def test_visible_result_classification_is_sequential_and_fail_fast():
    source = INTEGRATION.read_text(encoding="utf-8")
    function = _classify_function(source)

    assert "setTimeout(async () =>" not in function
    assert "await qaPanel.classifyAndInject(card, metadata)" in function
    assert "qaClassificationAvailable = false" in function
    assert "error.status === 503" in function
    assert "break;" in function


def test_unavailable_qa_is_latched_for_page_session():
    source = INTEGRATION.read_text(encoding="utf-8")

    assert "let qaClassificationAvailable = true;" in source
    assert "if (!qaClassificationAvailable)" in source
    assert "QA Unavailable" in source


def test_panel_preserves_structured_unavailable_error_metadata():
    source = PANEL.read_text(encoding="utf-8")

    assert "apiError.status = response.status" in source
    assert "apiError.code = errorData.code || null" in source
    assert "apiError.qaUnavailable = Boolean(" in source
    assert "response.status === 503" in source
    assert "errorData.code === 'qa_database_unavailable'" in source
