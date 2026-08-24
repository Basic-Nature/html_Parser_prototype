from __future__ import annotations

import ast
from pathlib import Path

from webapp.parser.handlers.states.pennsylvania import pennsylvania as pa


EXPECTED_MESSAGES = [
    "Do you want to continue parsing this election's contests? (y/n):",
    "Select an election to load by index:",
    "Select CSV file index:",
]


def _install_responses(monkeypatch, responses):
    calls = []
    queue = list(responses)

    def fake_prompt_input(message, **kwargs):
        calls.append((message, dict(kwargs)))
        if not queue:
            raise AssertionError("unexpected extra prompt")
        return queue.pop(0)

    monkeypatch.setattr(pa.prompt, "prompt_input", fake_prompt_input)
    return calls


def _assert_call(call, expected_message, session_id):
    message, kwargs = call
    assert message == expected_message
    assert kwargs == {
        "session_id": session_id,
        "allow_cancel": False,
    }


def test_pennsylvania_source_has_zero_raw_input_and_three_mediated_calls():
    path = Path(pa.__file__).resolve()
    tree = ast.parse(
        path.read_text(encoding="utf-8-sig"),
        filename=str(path),
    )

    raw_input = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "input"
    ]

    mediated = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "prompt"
        and node.func.attr == "prompt_input"
    ]

    assert raw_input == []
    assert len(mediated) == 3


def test_first_continue_prompt_uses_session_and_preserves_y_normalization(
    monkeypatch,
):
    session_id = "pa-session-continue"
    calls = _install_responses(monkeypatch, ["  Y  "])
    monkeypatch.setattr(pa.os, "listdir", lambda path: [])

    result = pa.parse(
        page=None,
        html_context={"config": {}, "selected_race": "Race"},
        session_id=session_id,
    )

    assert result[2] == "Pennsylvania (CSV not found)"
    assert len(calls) == 1
    _assert_call(calls[0], EXPECTED_MESSAGES[0], session_id)


def test_election_selection_prompt_preserves_raw_index_semantics(monkeypatch):
    session_id = "pa-session-election"
    calls = _install_responses(monkeypatch, ["n", " 0 "])

    election_toggle = object()
    election_link = object()

    def fake_query(page, selector):
        if selector == "a[aria-label='Elections']":
            return [election_toggle]
        if selector == "ul.dropdown-menu li a":
            return [election_link]
        return []

    clicks = []

    monkeypatch.setattr(pa, "safe_query_selector_all", fake_query)
    monkeypatch.setattr(
        pa,
        "safe_click",
        lambda element, logger: clicks.append(element),
    )
    monkeypatch.setattr(
        pa,
        "safe_wait_for_timeout",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        pa,
        "safe_inner_text",
        lambda *args, **kwargs: "Election A",
    )
    monkeypatch.setattr(pa.os, "listdir", lambda path: [])

    result = pa.parse(
        page=object(),
        html_context={"config": {}, "selected_race": "Race"},
        session_id=session_id,
    )

    assert result[2] == "Pennsylvania (CSV not found)"
    assert len(calls) == 2
    _assert_call(calls[0], EXPECTED_MESSAGES[0], session_id)
    _assert_call(calls[1], EXPECTED_MESSAGES[1], session_id)
    assert election_toggle in clicks
    assert election_link in clicks


def test_csv_selection_prompt_preserves_invalid_non_digit_return(monkeypatch):
    session_id = "pa-session-csv"
    calls = _install_responses(monkeypatch, ["y", "not-a-digit"])
    monkeypatch.setattr(
        pa.os,
        "listdir",
        lambda path: ["one.csv", "two.csv"],
    )

    result = pa.parse(
        page=None,
        html_context={"config": {}, "selected_race": "Race"},
        session_id=session_id,
    )

    assert result == (
        [],
        [],
        "Pennsylvania (CSV selection error)",
        {},
    )
    assert len(calls) == 2
    _assert_call(calls[0], EXPECTED_MESSAGES[0], session_id)
    _assert_call(calls[1], EXPECTED_MESSAGES[2], session_id)


def test_literal_cancel_is_not_reinterpreted_as_prompt_cancellation(monkeypatch):
    session_id = "pa-session-cancel"
    calls = _install_responses(monkeypatch, ["cancel"])

    monkeypatch.setattr(
        pa,
        "safe_query_selector_all",
        lambda page, selector: [],
    )
    monkeypatch.setattr(pa.os, "listdir", lambda path: [])

    result = pa.parse(
        page=object(),
        html_context={"config": {}, "selected_race": "Race"},
        session_id=session_id,
    )

    assert result[2] == "Pennsylvania (CSV not found)"
    assert len(calls) == 1
    _assert_call(calls[0], EXPECTED_MESSAGES[0], session_id)


def test_mediated_calls_do_not_add_default_validator_or_timeout():
    path = Path(pa.__file__).resolve()
    tree = ast.parse(
        path.read_text(encoding="utf-8-sig"),
        filename=str(path),
    )

    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "prompt"
        and node.func.attr == "prompt_input"
    ]

    assert len(calls) == 3

    for call in calls:
        keywords = {kw.arg for kw in call.keywords}
        assert keywords == {"session_id", "allow_cancel"}