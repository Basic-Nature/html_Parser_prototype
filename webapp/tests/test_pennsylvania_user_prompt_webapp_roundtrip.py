from __future__ import annotations

import json
import queue
import threading
from typing import Any

import pytest

from webapp.parser.handlers.states.pennsylvania import pennsylvania as pa
from webapp.parser.utils import user_prompt as user_prompt_module


@pytest.fixture
def isolated_webapp_prompt(monkeypatch):
    prompt = pa.prompt

    old_mode = prompt.mode
    old_emit = prompt.socketio_emit_func
    old_timeout = user_prompt_module.DEFAULT_WEBAPP_PROMPT_TIMEOUT_SEC

    prompt.prompt_sessions.clear()
    prompt.clear_queued_responses()
    prompt.set_mode("webapp")
    monkeypatch.setattr(
        user_prompt_module,
        "DEFAULT_WEBAPP_PROMPT_TIMEOUT_SEC",
        2,
    )

    emitted: queue.Queue[dict[str, Any]] = queue.Queue()

    def capture_emit(payload):
        if isinstance(payload, str):
            item = json.loads(payload)
        elif isinstance(payload, dict):
            item = dict(payload)
        else:
            raise AssertionError(
                f"unexpected UserPrompt emission type: {type(payload)!r}"
            )
        emitted.put(item)

    prompt.set_socketio_emit_func(capture_emit)

    try:
        yield prompt, emitted
    finally:
        prompt.prompt_sessions.clear()
        prompt.clear_queued_responses()
        prompt.set_mode(old_mode)
        prompt.set_socketio_emit_func(old_emit)
        user_prompt_module.DEFAULT_WEBAPP_PROMPT_TIMEOUT_SEC = old_timeout


def _next_prompt(emitted, timeout=3):
    while True:
        payload = emitted.get(timeout=timeout)
        if payload.get("type") == "prompt":
            return payload


def _respond(prompt, payload, response):
    session_id = payload["session_id"]
    prompt_session = prompt.prompt_sessions.get(session_id)
    assert prompt_session is not None
    assert prompt_session.status == "pending"
    prompt_session.set_response(response)


def _run_in_thread(target):
    result = {}
    error = {}

    def runner():
        try:
            result["value"] = target()
        except BaseException as exc:
            error["value"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    return thread, result, error


def _finish(thread, error, timeout=5):
    thread.join(timeout)
    assert not thread.is_alive(), "Pennsylvania parser remained blocked"
    if "value" in error:
        raise error["value"]


def test_first_pennsylvania_prompt_roundtrips_through_real_prompt_session(
    monkeypatch,
    isolated_webapp_prompt,
):
    prompt, emitted = isolated_webapp_prompt
    session_id = "pa-webapp-roundtrip-first"

    monkeypatch.setattr(pa.os, "listdir", lambda path: [])

    thread, result, error = _run_in_thread(
        lambda: pa.parse(
            page=None,
            html_context={"config": {}, "selected_race": "Race"},
            session_id=session_id,
        )
    )

    payload = _next_prompt(emitted)

    assert payload["session_id"] == session_id
    assert payload["message"] == (
        "Do you want to continue parsing this election's contests? (y/n): "
    )

    _respond(prompt, payload, "y")
    _finish(thread, error)

    assert result["value"][2] == "Pennsylvania (CSV not found)"


def test_election_selection_uses_second_prompt_session_roundtrip(
    monkeypatch,
    isolated_webapp_prompt,
):
    prompt, emitted = isolated_webapp_prompt
    session_id = "pa-webapp-roundtrip-election"

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

    thread, result, error = _run_in_thread(
        lambda: pa.parse(
            page=object(),
            html_context={"config": {}, "selected_race": "Race"},
            session_id=session_id,
        )
    )

    first = _next_prompt(emitted)
    assert first["session_id"] == session_id
    _respond(prompt, first, "n")

    second = _next_prompt(emitted)
    assert second["session_id"] == session_id
    assert second["message"] == "Select an election to load by index: "
    _respond(prompt, second, "0")

    _finish(thread, error)

    assert result["value"][2] == "Pennsylvania (CSV not found)"
    assert election_toggle in clicks
    assert election_link in clicks


def test_csv_selection_uses_real_prompt_session_and_preserves_error_contract(
    monkeypatch,
    isolated_webapp_prompt,
):
    prompt, emitted = isolated_webapp_prompt
    session_id = "pa-webapp-roundtrip-csv"

    monkeypatch.setattr(
        pa.os,
        "listdir",
        lambda path: ["one.csv", "two.csv"],
    )

    thread, result, error = _run_in_thread(
        lambda: pa.parse(
            page=None,
            html_context={"config": {}, "selected_race": "Race"},
            session_id=session_id,
        )
    )

    first = _next_prompt(emitted)
    _respond(prompt, first, "y")

    second = _next_prompt(emitted)
    assert second["session_id"] == session_id
    assert second["message"] == "Select CSV file index: "
    _respond(prompt, second, "not-a-digit")

    _finish(thread, error)

    assert result["value"] == (
        [],
        [],
        "Pennsylvania (CSV selection error)",
        {},
    )


def test_roundtrip_uses_userprompt_webapp_wait_path_not_raw_input(
    monkeypatch,
    isolated_webapp_prompt,
):
    prompt, emitted = isolated_webapp_prompt
    session_id = "pa-webapp-roundtrip-no-stdin"

    monkeypatch.setattr(pa.os, "listdir", lambda path: [])
    monkeypatch.setattr(
        "builtins.input",
        lambda *args, **kwargs: pytest.fail(
            "raw input() must not run in Pennsylvania webapp mode"
        ),
    )

    thread, result, error = _run_in_thread(
        lambda: pa.parse(
            page=None,
            html_context={"config": {}, "selected_race": "Race"},
            session_id=session_id,
        )
    )

    payload = _next_prompt(emitted)
    _respond(prompt, payload, "y")
    _finish(thread, error)

    assert result["value"][2] == "Pennsylvania (CSV not found)"