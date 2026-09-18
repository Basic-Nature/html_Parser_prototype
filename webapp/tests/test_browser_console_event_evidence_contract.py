from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

import webapp.parser.services.browser_console_event_evidence as console_evidence
from webapp.parser.services.browser_console_event_evidence import (
    BROWSER_CONSOLE_EVENT_OBSERVATION_AUTHORITY,
    BROWSER_CONSOLE_EVENT_OBSERVATION_CONTRACT,
    EVENT_LIST_IDENTITY_SEMANTICS,
    FINAL_URL_IDENTITY_SEMANTICS,
    REQUESTED_URL_IDENTITY_SEMANTICS,
    observe_browser_console_event_if_requested,
)


def _canonical_events(events):
    return (
        json.dumps(
            events,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def test_dormant_none_returns_exact_object_before_hashing_or_validation(monkeypatch):
    sentinel = object()
    events = [sentinel]

    def explode(*_args, **_kwargs):
        raise AssertionError("dormant path must not hash")

    monkeypatch.setattr(console_evidence.hashlib, "sha256", explode)

    returned = observe_browser_console_event_if_requested(
        events,
        requested_url="https://example.invalid/?token=secret",
        final_url="https://example.invalid/final?secret=1",
        capture_role="url_glimpse",
        observation_emit_func=None,
    )

    assert returned is events
    assert returned[0] is sentinel


def test_active_observation_is_hash_only_and_deterministic():
    events = [
        {"type": "error", "text": "SECRET_TOKEN=abc"},
        {"type": "warning", "text": "https://private.example/?token=xyz"},
        {"type": "error", "text": "candidate debug value"},
    ]
    requested = "https://example.org/results?auth=SECRET_REQUEST"
    final = "https://example.org/final?session=SECRET_FINAL"
    captured = []

    returned = observe_browser_console_event_if_requested(
        events,
        requested_url=requested,
        final_url=final,
        capture_role="url_glimpse",
        observation_emit_func=captured.append,
    )

    assert returned is events
    assert len(captured) == 1
    payload = captured[0]

    assert payload["contract"] == BROWSER_CONSOLE_EVENT_OBSERVATION_CONTRACT
    assert payload["authority"] == BROWSER_CONSOLE_EVENT_OBSERVATION_AUTHORITY
    assert payload["canonical"] is False
    assert payload["event_count"] == 3
    assert payload["event_type_counts"] == {"error": 2, "warning": 1}
    assert payload["event_list_sha256"] == hashlib.sha256(
        _canonical_events(events)
    ).hexdigest()
    assert payload["requested_url_sha256"] == hashlib.sha256(
        requested.encode("utf-8")
    ).hexdigest()
    assert payload["final_url_sha256"] == hashlib.sha256(
        final.encode("utf-8")
    ).hexdigest()
    assert payload["event_list_identity_semantics"] == EVENT_LIST_IDENTITY_SEMANTICS
    assert payload["requested_url_identity_semantics"] == REQUESTED_URL_IDENTITY_SEMANTICS
    assert payload["final_url_identity_semantics"] == FINAL_URL_IDENTITY_SEMANTICS

    assert payload["raw_console_text_included"] is False
    assert payload["raw_requested_url_included"] is False
    assert payload["raw_final_url_included"] is False
    assert payload["filesystem_path_included"] is False
    assert payload["automatic_timestamp"] is False

    serialized = json.dumps(payload, sort_keys=True)
    assert "SECRET_TOKEN" not in serialized
    assert "private.example" not in serialized
    assert "SECRET_REQUEST" not in serialized
    assert "SECRET_FINAL" not in serialized
    assert requested not in serialized
    assert final not in serialized


def test_active_callback_exception_propagates():
    events = [{"type": "error", "text": "x"}]

    def fail(_payload):
        raise RuntimeError("observer failed")

    with pytest.raises(RuntimeError, match="observer failed"):
        observe_browser_console_event_if_requested(
            events,
            requested_url="https://example.org/a",
            final_url="https://example.org/b",
            capture_role="url_glimpse",
            observation_emit_func=fail,
        )


def test_url_glimpse_integration_preserves_capture_and_persistence_contract():
    text = Path("webapp/parser/utils/url_glimpse.py").read_text(encoding="utf-8")

    assert 'page.on("console", on_console)' in text
    assert '{"type": msg.type, "text": msg.text}' in text
    assert "console_msgs[-50:]" in text
    assert 'result["console"]' in text
    assert 'json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")' in text

    assert "console_observation_emit_func=None" in text
    assert "observe_browser_console_event_if_requested" in text
    assert "if console_observation_emit_func is not None:" in text
    assert '"final_url": page.url' in text
    assert '"capture_role": "url_glimpse"' in text

    assert "screenshot_observation_emit_func=None" in text
    assert "finalize_screenshot_image_evidence" in text
    assert "observation_emit_func=screenshot_observation_emit_func" in text

    report_write = text.index(
        'json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")'
    )
    observation_call = text.rindex(
        "observe_browser_console_event_if_requested(**console_observation_args)"
    )
    assert report_write < observation_call

    assert "console_path" not in text
    assert "console.json" not in text


def test_service_has_no_playwright_or_filesystem_write_surface():
    service = Path(
        "webapp/parser/services/browser_console_event_evidence.py"
    ).read_text(encoding="utf-8")

    tree = ast.parse(service)
    imported_modules = []
    executable_names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name.lower() for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_modules.append((node.module or "").lower())
        elif isinstance(node, ast.Name):
            executable_names.append(node.id.lower())

    assert not any("playwright" in name for name in imported_modules)
    assert "playwright" not in executable_names
    assert ".write_text(" not in service
    assert ".write_bytes(" not in service
    assert "open(" not in service
    assert "Path(" not in service
    assert "raw_console_text_included" in service
    assert "raw_requested_url_included" in service
    assert "raw_final_url_included" in service
