from unittest.mock import patch

from webapp.parser.navigator.navigation_runner import NavigationInstructionRunner


def test_is_enhanced_voting_page_detects_rockland_url(mock_page):
    mock_page.url = "https://results.rocklandny.gov/enhancedvoting"
    mock_page.content.return_value = "<html><body>View results by election district</body></html>"

    runner = NavigationInstructionRunner()
    assert runner._is_enhanced_voting_page(mock_page)


def test_execute_step_skips_generic_county_selector_on_enhanced_voting_page(mock_page):
    mock_page.url = "https://results.rocklandny.gov/enhancedvoting"
    mock_page.content.return_value = "<html><body>View results by election district</body></html>"
    runner = NavigationInstructionRunner()
    trace = []
    context = {"url": mock_page.url}
    step = {"action": "wait_for_selector", "selector": "button:has-text('County')"}

    result = runner._execute_step(step, mock_page, context, None, None, trace)

    assert result is None
    assert trace, "Expected a trace entry when skipping a selector"
    assert trace[-1]["action"] == "wait_for_selector"
    assert trace[-1]["status"] == "skipped"
    assert trace[-1].get("details", {}).get("reason") == "enhanced_voting_generic_selector"


def test_execute_step_skips_generic_click_without_text_discovery_on_enhanced_voting_page(mock_page):
    mock_page.url = "https://results.rocklandny.gov/enhancedvoting"
    mock_page.content.return_value = "<html><body>View results by election district</body></html>"
    runner = NavigationInstructionRunner()
    trace = []
    context = {"url": mock_page.url}
    step = {"action": "click", "selector": "button:has-text('County')"}

    with patch.object(NavigationInstructionRunner, "_click_by_text_discovery", autospec=True) as mock_text_discovery:
        result = runner._execute_step(step, mock_page, context, None, None, trace)

    assert result is None
    assert not mock_text_discovery.called
    assert trace[-1]["action"] == "click"
    assert trace[-1]["status"] == "skipped"
    assert trace[-1].get("details", {}).get("reason") == "enhanced_voting_generic_selector"
