from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

import webapp.parser.socket_ballot_lens_orchestration as orchestration
import webapp.parser.services.public_ballot_lens_policy as public_policy
import webapp.parser.utils.url_registry as url_registry


SOURCE_PATH = Path(
    "webapp/parser/socket_ballot_lens_orchestration.py"
)


class FakeLogger:
    def __init__(self):
        self.records = []

    def warning(self, payload):
        self.records.append(("warning", payload))


class FakeSocketIO:
    def __init__(self):
        self.sleeps = []

    def sleep(self, seconds):
        self.sleeps.append(seconds)


def make_public_hooks():
    emitted = []
    session_calls = []
    rooms = []
    hooks = {
        "logger": FakeLogger(),
        "normalize_log_obj": lambda value: value,
        "emit": lambda *args, **kwargs: emitted.append(
            (args, kwargs)
        ),
        "resolve_session_id": (
            lambda payload, create_if_missing=True:
            session_calls.append(
                (payload, create_if_missing)
            )
            or "sess_server_generated_123456"
        ),
        "join_room": lambda room: rooms.append(room),
        "socketio": FakeSocketIO(),
    }
    return hooks, emitted, session_calls, rooms


def curated_source():
    return SimpleNamespace(
        registry_source_id="blsrc_v1_" + ("a" * 64),
        year="2024",
        contest="President",
        state="Example",
        registry_scope="statewide",
        registry_format="HTML",
        registry_category="curated",
        url="https://results.example.gov/elections/2024",
    )


def test_registry_source_id_presence_is_public_intent_even_if_mixed():
    assert orchestration._is_public_registry_intent(
        {"registry_source_id": "bad", "direct_urls": ["https://x"]}
    ) is True
    assert orchestration._is_public_registry_intent(
        {"direct_urls": ["https://x"]}
    ) is False


def test_public_handler_never_falls_through_to_trusted_initialization(
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(
        orchestration,
        "_handle_public_registry_authority_split",
        lambda payload, hooks: calls.append(
            ("public", dict(payload))
        ),
    )
    monkeypatch.setattr(
        orchestration,
        "_initialize_session_and_auth",
        lambda payload, hooks: (_ for _ in ()).throw(
            AssertionError("trusted init must not run")
        ),
    )

    orchestration.run_ballot_lens_socket_handler(
        {
            "registry_source_id": "blsrc_v1_" + ("a" * 64),
            "direct_urls": ["https://attacker.invalid"],
        },
        hooks={},
    )

    assert calls == [
        (
            "public",
            {
                "registry_source_id":
                    "blsrc_v1_" + ("a" * 64),
                "direct_urls":
                    ["https://attacker.invalid"],
            },
        )
    ]


def test_legacy_payload_still_uses_trusted_certificate_path(
    monkeypatch,
):
    calls = []

    monkeypatch.setattr(
        orchestration,
        "_initialize_session_and_auth",
        lambda payload, hooks: (
            calls.append(("trusted_init", dict(payload)))
            or {
                "session_id": "sess_trusted_123456",
                "dev_isolation_bypass": False,
                "principal": "cert:abc",
                "principal_source": "fresh_certificate",
            }
        ),
    )
    monkeypatch.setattr(
        orchestration,
        "_prepare_run_inputs",
        lambda payload, session_id, bypass, hooks: (
            calls.append(("prepare", dict(payload)))
            or {"run": "cfg"}
        ),
    )
    monkeypatch.setattr(
        orchestration,
        "_configure_logging_and_prompt",
        lambda session_id, hooks:
        calls.append(("logging", session_id)),
    )
    monkeypatch.setattr(
        orchestration,
        "_start_pipeline_worker",
        lambda *args:
        calls.append(("worker", args[0])),
    )

    orchestration.run_ballot_lens_socket_handler(
        {
            "direct_urls":
                ["https://trusted.example.gov/results"]
        },
        hooks={},
    )

    assert calls[0][0] == "trusted_init"
    assert ("worker", "sess_trusted_123456") in calls


def test_public_authority_uses_server_session_not_caller_payload(
    monkeypatch,
):
    source = curated_source()
    monkeypatch.setattr(
        url_registry,
        "resolve_public_registry_source",
        lambda path, source_id: source,
    )
    monkeypatch.setattr(
        public_policy,
        "validate_public_start_payload",
        lambda payload: source.registry_source_id,
    )
    monkeypatch.setattr(
        public_policy,
        "authorize_public_registry_parse",
        lambda payload, registry_source_resolved: (
            "ballot_lens_public_registry_parse",
            source.registry_source_id,
        ),
    )

    hooks, emitted, session_calls, rooms = make_public_hooks()

    context = orchestration._initialize_public_registry_authority(
        {
            "registry_source_id": source.registry_source_id,
        },
        hooks,
    )

    assert context is not None
    assert context["runtime_dispatch_enabled"] is False
    assert context["resolved_url"] == source.url
    assert session_calls == [({}, True)]
    assert context["source_projection"] == {
        "registry_source_id": source.registry_source_id,
        "year": "2024",
        "contest": "President",
        "state": "Example",
        "scope": "statewide",
        "format": "HTML",
        "registry_category": "curated",
    }
    assert emitted == []
    assert rooms == []


def test_public_authority_denial_does_not_create_session(
    monkeypatch,
):
    source = curated_source()

    monkeypatch.setattr(
        public_policy,
        "validate_public_start_payload",
        lambda payload: source.registry_source_id,
    )
    monkeypatch.setattr(
        url_registry,
        "resolve_public_registry_source",
        lambda path, source_id: None,
    )

    hooks, emitted, session_calls, _ = make_public_hooks()

    context = orchestration._initialize_public_registry_authority(
        {
            "registry_source_id": source.registry_source_id,
        },
        hooks,
    )

    assert context is None
    assert session_calls == []
    assert emitted
    rendered = repr(emitted)
    assert source.url not in rendered


def test_public_authority_status_projection_never_exposes_raw_url(
    monkeypatch,
):
    source = curated_source()
    monkeypatch.setattr(
        orchestration,
        "_initialize_public_registry_authority",
        lambda payload, hooks: {
            "mode": "public_registry",
            "session_id": "sess_server_generated_123456",
            "capability": "ballot_lens_public_registry_parse",
            "registry_source_id": source.registry_source_id,
            "resolved_url": source.url,
            "source_projection":
                orchestration._public_registry_source_projection(
                    source
                ),
            "runtime_dispatch_enabled": False,
        },
    )

    hooks, emitted, _, rooms = make_public_hooks()
    orchestration._handle_public_registry_authority_split(
        {"registry_source_id": source.registry_source_id},
        hooks,
    )

    rendered = repr(emitted)
    assert source.url not in rendered
    assert "direct_urls" not in rendered
    assert rooms == ["sess_server_generated_123456"]
    assert "runtime_pending" in rendered


def test_public_split_source_contains_no_cert_gate_or_worker_dispatch():
    source = SOURCE_PATH.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(SOURCE_PATH))

    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        == "_initialize_public_registry_authority"
    )
    helper_text = ast.get_source_segment(source, helper) or ""

    handler = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "run_ballot_lens_socket_handler"
    )
    handler_text = ast.get_source_segment(source, handler) or ""

    assert "require_cert_for_socket_action" not in helper_text
    assert "_start_pipeline_worker" not in helper_text
    assert 'resolve_session_id"](' in helper_text
    assert "payload," not in (
        helper_text.split('resolve_session_id"](', 1)[1]
        .split(")", 1)[0]
    )

    public_branch = handler_text.index(
        "if _is_public_registry_intent(payload):"
    )
    trusted_init = handler_text.index(
        "_initialize_session_and_auth(payload, hooks)"
    )
    assert public_branch < trusted_init
    assert "_handle_public_registry_authority_split" in handler_text


def test_mixed_public_payload_is_rejected_by_exact_public_payload_contract():
    source = curated_source()
    with pytest.raises(public_policy.PublicBallotLensPolicyError):
        public_policy.validate_public_start_payload(
            {
                "registry_source_id": source.registry_source_id,
                "direct_urls": [source.url],
            }
        )
