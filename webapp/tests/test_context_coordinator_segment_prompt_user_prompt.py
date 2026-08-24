from __future__ import annotations

import ast
import json
import queue
import threading
from pathlib import Path

import pytest

from webapp.parser.Context_Integration import context_coordinator as cc
from webapp.parser.utils import user_prompt as user_prompt_module


MESSAGE = "Please enter the semantic label for this segment:"


def _coordinator_with_log_capture():
    coordinator = cc.ContextCoordinator.__new__(cc.ContextCoordinator)
    captured = []

    def capture_log_field_selection(**kwargs):
        captured.append(dict(kwargs))

    coordinator.log_field_selection = capture_log_field_selection
    return coordinator, captured


def test_segment_prompt_source_has_zero_raw_input_and_one_mediated_call():
    path = Path(cc.__file__).resolve()
    tree = ast.parse(
        path.read_text(encoding="utf-8-sig"),
        filename=str(path),
    )

    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "ContextCoordinator"
    ]
    assert len(classes) == 1

    methods = [
        node
        for node in classes[0].body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "segment_prompt"
    ]
    assert len(methods) == 1

    raw_input = [
        node
        for node in ast.walk(methods[0])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "input"
    ]

    mediated = [
        node
        for node in ast.walk(methods[0])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "prompt"
        and node.func.attr == "prompt_input"
    ]

    assert raw_input == []
    assert len(mediated) == 1

    call = mediated[0]
    assert len(call.args) == 1
    assert isinstance(call.args[0], ast.Constant)
    assert call.args[0].value == MESSAGE

    keywords = {kw.arg: kw.value for kw in call.keywords}
    assert set(keywords) == {"session_id", "allow_cancel"}
    assert isinstance(keywords["session_id"], ast.Name)
    assert keywords["session_id"].id == "session_id"
    assert isinstance(keywords["allow_cancel"], ast.Constant)
    assert keywords["allow_cancel"].value is False


def test_segment_prompt_preserves_strip_log_and_return_contract(monkeypatch):
    coordinator, captured = _coordinator_with_log_capture()
    calls = []

    def fake_prompt_input(message, **kwargs):
        calls.append((message, dict(kwargs)))
        return "  semantic_label  "

    monkeypatch.setattr(cc.prompt, "prompt_input", fake_prompt_input)

    result = coordinator.segment_prompt(
        {"html": "<div>example</div>"},
        session_id="segment-session-1",
        reason="ambiguous segment",
    )

    assert result == "semantic_label"
    assert calls == [
        (
            MESSAGE,
            {
                "session_id": "segment-session-1",
                "allow_cancel": False,
            },
        )
    ]

    assert len(captured) == 1
    record = captured[0]
    assert record["field_type"] == "segment"
    assert record["field_name"] == "segment_prompt"
    assert record["extracted_value"] == "<div>example</div>"
    assert record["method"] == "interactive"
    assert record["score"] == 1.0
    assert record["result"] == "ambiguous segment"
    assert record["context"] == {
        "session_id": "segment-session-1",
        "reason": "ambiguous segment",
    }
    assert record["user_feedback"] == "semantic_label"


def test_segment_prompt_preserves_broad_exception_fallback_to_unknown(monkeypatch):
    coordinator, captured = _coordinator_with_log_capture()

    def raise_prompt(*args, **kwargs):
        raise RuntimeError("simulated mediator failure")

    monkeypatch.setattr(cc.prompt, "prompt_input", raise_prompt)

    result = coordinator.segment_prompt(
        {"html": "<span>needs review</span>"},
        session_id="segment-session-error",
        reason="test failure",
    )

    assert result == "unknown"
    assert len(captured) == 1
    assert captured[0]["user_feedback"] == "unknown"


def test_literal_cancel_remains_literal_when_allow_cancel_false(monkeypatch):
    coordinator, captured = _coordinator_with_log_capture()

    monkeypatch.setattr(
        cc.prompt,
        "prompt_input",
        lambda *args, **kwargs: "cancel",
    )

    result = coordinator.segment_prompt(
        {"html": "<div>cancel literal</div>"},
        session_id="segment-session-cancel",
        reason="literal cancel test",
    )

    assert result == "cancel"
    assert captured[0]["user_feedback"] == "cancel"


def test_segment_prompt_real_userprompt_webapp_roundtrip_without_stdin(monkeypatch):
    coordinator, captured = _coordinator_with_log_capture()
    prompt = cc.prompt

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

    emitted = queue.Queue()

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

    monkeypatch.setattr(
        "builtins.input",
        lambda *args, **kwargs: pytest.fail(
            "segment_prompt webapp path must not use raw input()"
        ),
    )

    result = {}
    error = {}

    def runner():
        try:
            result["value"] = coordinator.segment_prompt(
                {"html": "<section>roundtrip</section>"},
                session_id="segment-webapp-roundtrip",
                reason="roundtrip test",
            )
        except BaseException as exc:
            error["value"] = exc

    thread = threading.Thread(target=runner, daemon=True)

    try:
        thread.start()

        prompt_payload = None
        for _ in range(5):
            payload = emitted.get(timeout=3)
            if payload.get("type") == "prompt":
                prompt_payload = payload
                break

        assert prompt_payload is not None
        assert prompt_payload["session_id"] == "segment-webapp-roundtrip"
        assert prompt_payload["message"] == MESSAGE + " "

        prompt_session = prompt.prompt_sessions.get(
            "segment-webapp-roundtrip"
        )
        assert prompt_session is not None
        assert prompt_session.status == "pending"

        prompt_session.set_response("  webapp_semantic_label  ")

        thread.join(5)
        assert not thread.is_alive(), "segment_prompt remained blocked"

        if "value" in error:
            raise error["value"]

        assert result["value"] == "webapp_semantic_label"
        assert captured[0]["user_feedback"] == "webapp_semantic_label"
    finally:
        prompt.prompt_sessions.clear()
        prompt.clear_queued_responses()
        prompt.set_mode(old_mode)
        prompt.set_socketio_emit_func(old_emit)
        user_prompt_module.DEFAULT_WEBAPP_PROMPT_TIMEOUT_SEC = old_timeout