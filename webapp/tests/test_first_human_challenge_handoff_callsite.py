from __future__ import annotations

import ast
from pathlib import Path

import webapp.parser.html_election_parser as parser


def test_handoff_helper_records_only_known_human_policy_facts(monkeypatch):
    calls = []

    def fake_adapter(detected, **kwargs):
        calls.append((detected, kwargs))
        return object()

    monkeypatch.setattr(
        parser,
        "adapt_legacy_challenge_detection",
        fake_adapter,
    )

    result = parser._observe_legacy_human_handoff_noncanonical(
        session_id="session-288",
        browser_context_ref="playwright-navigation-attempt-1",
        vendor_hint="cloudflare",
        challenge_type_hint="legacy-cloudflare-detection",
        indicators=("user_selected_captcha_assist",),
    )

    assert result is not None
    assert len(calls) == 1

    detected, kwargs = calls[0]
    assert detected is True
    assert kwargs["session_id"] == "session-288"
    assert (
        kwargs["browser_context_ref"]
        == "playwright-navigation-attempt-1"
    )
    assert kwargs["vendor_hint"] == "cloudflare"
    assert (
        kwargs["challenge_type_hint"]
        == "legacy-cloudflare-detection"
    )
    assert kwargs["indicators"] == (
        "user_selected_captcha_assist",
    )

    assert kwargs["human_intervention_required"] is True
    assert kwargs["human_intervention_completed"] is False
    assert kwargs["create_human_handoff"] is True
    assert kwargs["observed_at"] is None
    assert kwargs["evidence_refs"] == ()
    assert kwargs["browser_engine_hint"] is None


def test_handoff_adapter_failure_is_noncontrolling(monkeypatch):
    def failing_adapter(*args, **kwargs):
        raise RuntimeError("synthetic handoff adapter failure")

    monkeypatch.setattr(
        parser,
        "adapt_legacy_challenge_detection",
        failing_adapter,
    )

    assert (
        parser._observe_legacy_human_handoff_noncanonical(
            session_id="session-288",
            browser_context_ref="playwright-navigation-attempt-1",
            vendor_hint="cloudflare",
            challenge_type_hint="legacy-cloudflare-detection",
            indicators=("user_selected_captcha_assist",),
        )
        is None
    )


def _parser_tree():
    path = Path(parser.__file__).resolve()
    text = path.read_text(encoding="utf-8-sig")
    return path, text, ast.parse(text, filename=str(path))


def test_exactly_one_explicit_handoff_runtime_call_exists():
    _, _, tree = _parser_tree()

    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id
        == "_observe_legacy_human_handoff_noncanonical"
    ]

    assert len(calls) == 1


def test_handoff_call_is_only_inside_explicit_decision_assist_branch():
    _, _, tree = _parser_tree()

    parent = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node

    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id
        == "_observe_legacy_human_handoff_noncanonical"
    ]

    assert len(calls) == 1
    expression = parent[calls[0]]
    assert isinstance(expression, ast.Expr)

    enclosing = parent[expression]
    assert isinstance(enclosing, ast.If)
    assert isinstance(enclosing.test, ast.Compare)
    assert isinstance(enclosing.test.left, ast.Name)
    assert enclosing.test.left.id == "decision"
    assert len(enclosing.test.comparators) == 1
    assert isinstance(enclosing.test.comparators[0], ast.Constant)
    assert enclosing.test.comparators[0].value == "assist"


def test_handoff_result_is_not_assigned_or_used_for_control_flow():
    _, _, tree = _parser_tree()

    parent = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node

    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id
        == "_observe_legacy_human_handoff_noncanonical"
    ]

    assert len(calls) == 1
    assert isinstance(parent[calls[0]], ast.Expr)


def test_auto_assist_branch_is_not_instrumented_in_this_checkpoint():
    _, text, _ = _parser_tree()

    marker = (
        'indicators=("user_selected_captcha_assist",),'
    )
    assert text.count(marker) == 1
    assert "auto_assist_handoff" not in text


def test_original_observation_seam_remains_exactly_once():
    _, _, tree = _parser_tree()

    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_observe_legacy_challenge_noncanonical"
    ]

    assert len(calls) == 1


def test_current_presentation_is_still_cross_browser_and_not_claimed_compliant():
    _, text, _ = _parser_tree()

    # Current behavior remains intentionally untouched:
    # Playwright closes, Selenium fallback later presents the assist browser.
    assert "_close_browser_quietly(browser, session_id)" in text
    assert 'if agent == "selenium":' in text
    assert "relaunch_browser_fullscreen_if_needed" in text
    assert 'user_agent=nav_meta.get("user_agent")' in text

    # The parser does not claim that same-context continuation was achieved.
    assert "same_context_resume_completed" not in text
    assert "same_context_resume_satisfied" not in text


def test_skip_branch_still_returns_before_handoff_creation():
    _, text, _ = _parser_tree()

    skip_index = text.index('if decision == "skip":')
    assist_index = text.index('if decision == "assist":', skip_index)
    handoff_index = text.index(
        "_observe_legacy_human_handoff_noncanonical(",
        assist_index,
    )

    assert skip_index < assist_index < handoff_index


def test_legacy_browser_identity_and_presentation_markers_remain():
    _, text, _ = _parser_tree()

    required = [
        "relaunch_browser_fullscreen_if_needed",
        'user_agent=nav_meta.get("user_agent")',
        "launch_browser()",
        "captcha_assist_requested = True",
    ]

    for marker in required:
        assert marker in text