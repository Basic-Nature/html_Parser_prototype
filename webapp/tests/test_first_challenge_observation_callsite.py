from __future__ import annotations

import ast
from pathlib import Path

import webapp.parser.html_election_parser as parser


def test_observation_helper_adapts_only_known_detection_facts(monkeypatch):
    calls = []

    def fake_adapter(detected, **kwargs):
        calls.append((detected, kwargs))
        return object()

    monkeypatch.setattr(parser, "adapt_legacy_challenge_detection", fake_adapter)

    result = parser._observe_legacy_challenge_noncanonical(
        session_id="session-287",
        browser_context_ref="playwright-navigation-attempt-1",
        vendor_hint="cloudflare",
        challenge_type_hint="legacy-cloudflare-detection",
        browser_engine_hint=None,
        indicators=("nav_meta.cloudflare_detected",),
    )

    assert result is not None
    assert len(calls) == 1
    detected, kwargs = calls[0]
    assert detected is True
    assert kwargs["session_id"] == "session-287"
    assert kwargs["browser_context_ref"] == "playwright-navigation-attempt-1"
    assert kwargs["vendor_hint"] == "cloudflare"
    assert kwargs["challenge_type_hint"] == "legacy-cloudflare-detection"
    assert kwargs["browser_engine_hint"] is None
    assert kwargs["indicators"] == ("nav_meta.cloudflare_detected",)
    assert kwargs["evidence_refs"] == ()
    assert kwargs["human_intervention_required"] is None
    assert kwargs["human_intervention_completed"] is None
    assert kwargs["observed_at"] is None
    assert kwargs["create_human_handoff"] is False


def test_observation_adapter_failure_is_noncontrolling(monkeypatch):
    def failing_adapter(*args, **kwargs):
        raise RuntimeError("synthetic adapter failure")

    monkeypatch.setattr(parser, "adapt_legacy_challenge_detection", failing_adapter)

    assert parser._observe_legacy_challenge_noncanonical(
        session_id="session-287",
        browser_context_ref="playwright-navigation-attempt-1",
        vendor_hint="cloudflare",
        challenge_type_hint="legacy-cloudflare-detection",
        indicators=("nav_meta.cloudflare_detected",),
    ) is None


def test_missing_session_is_bounded_without_changing_control_contract(monkeypatch):
    captured = {}

    def fake_adapter(detected, **kwargs):
        captured["detected"] = detected
        captured.update(kwargs)
        return "observation"

    monkeypatch.setattr(parser, "adapt_legacy_challenge_detection", fake_adapter)

    assert parser._observe_legacy_challenge_noncanonical(
        session_id=None,
        browser_context_ref="playwright-navigation-attempt-2",
        vendor_hint="cloudflare",
        challenge_type_hint="legacy-cloudflare-detection",
    ) == "observation"
    assert captured["session_id"] == "no_session"


def test_runtime_has_exactly_one_observation_call_site():
    path = Path(parser.__file__).resolve()
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))

    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_observe_legacy_challenge_noncanonical"
    ]

    assert len(calls) == 1


def test_first_call_site_is_additive_between_existing_detection_and_threshold():
    path = Path(parser.__file__).resolve()
    text = path.read_text(encoding="utf-8-sig")

    condition = (
        'if nav_meta.get("cloudflare_detected") '
        "and ENABLE_SELENIUM_FALLBACK:"
    )
    detection = (
        'detection_count = _register_cloudflare_detection('
        'session_id, target_url, "playwright")'
    )
    observation = "_observe_legacy_challenge_noncanonical("
    threshold = "if detection_count < 2:"

    condition_index = text.index(condition)
    detection_index = text.index(detection, condition_index)
    observation_index = text.index(observation, detection_index)
    threshold_index = text.index(threshold, observation_index)

    assert condition_index < detection_index < observation_index < threshold_index


def test_observation_result_has_no_downstream_assignment():
    path = Path(parser.__file__).resolve()
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    parent_map = {}

    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parent_map[child] = parent

    runtime_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_observe_legacy_challenge_noncanonical"
    ]

    assert len(runtime_calls) == 1
    assert isinstance(parent_map[runtime_calls[0]], ast.Expr)


def test_parser_does_not_create_challenge_handoff_at_first_call_site():
    path = Path(parser.__file__).resolve()
    text = path.read_text(encoding="utf-8-sig")

    assert "ChallengeHandoff" not in text
    assert "build_human_challenge_handoff" not in text


def test_existing_browser_decision_markers_remain_present():
    path = Path(parser.__file__).resolve()
    text = path.read_text(encoding="utf-8-sig")

    required = [
        "if detection_count < 2:",
        'if decision == "skip":',
        'if decision == "assist":',
        "captcha_assist_requested = True",
        "_close_browser_quietly(browser, session_id)",
        'if agent == "selenium":',
        "relaunch_browser_fullscreen_if_needed",
        'user_agent=nav_meta.get("user_agent")',
        "launch_browser()",
    ]

    for marker in required:
        assert marker in text