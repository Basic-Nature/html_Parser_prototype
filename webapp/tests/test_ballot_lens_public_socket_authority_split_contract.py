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


class FakePublicSessionManager:
    def __init__(self):
        self.sessions = {}
        self.socket_map = {}

    def resolve_socket(self, sid):
        return self.socket_map.get(sid)

    def has_session(self, sid):
        return sid in self.sessions

    def ensure_session(self, sid):
        self.sessions.setdefault(
            sid,
            {"session_id": sid, "principal": None},
        )
        return dict(self.sessions[sid])

    def update_metadata(self, sid, **updates):
        self.sessions.setdefault(sid, {"session_id": sid}).update(updates)
        return dict(self.sessions[sid])

    def get_metadata(self, sid):
        value = self.sessions.get(sid)
        return dict(value) if value else None

    def bind_socket(self, socket_sid, session_id):
        self.socket_map[socket_sid] = session_id


class FakePublicRequest:
    sid = "socket_public_test"
    remote_addr = "203.0.113.10"


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
        "session_manager": FakePublicSessionManager(),
        "request": FakePublicRequest(),
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

    monkeypatch.setenv(
        public_policy.PUBLIC_REGISTRY_RATE_HMAC_SECRET_ENV,
        "s" * 32,
    )
    hooks, emitted, session_calls, rooms = make_public_hooks()

    context = orchestration._initialize_public_registry_authority(
        {
            "registry_source_id": source.registry_source_id,
        },
        hooks,
    )

    assert context is not None
    assert context["runtime_dispatch_enabled"] is True
    assert context["resolved_url"] == source.url
    assert context["client_key"].startswith("client:")
    assert session_calls == []
    assert hooks["session_manager"].sessions
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
    assert 'resolve_session_id"](' not in helper_text
    assert "_create_public_server_session" in helper_text

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

def test_feature_disabled_capability_error_emits_public_denial_without_session(
    monkeypatch,
):
    source = curated_source()

    monkeypatch.delenv(
        public_policy.PUBLIC_REGISTRY_PARSE_ENV,
        raising=False,
    )
    monkeypatch.setattr(
        url_registry,
        "resolve_public_registry_source",
        lambda path, source_id: source,
    )

    hooks, emitted, session_calls, rooms = make_public_hooks()

    context = orchestration._initialize_public_registry_authority(
        {
            "registry_source_id": source.registry_source_id,
        },
        hooks,
    )

    assert context is None
    assert session_calls == []
    assert rooms == []

    rendered = repr(emitted)
    assert "public_registry_authority_denied" in rendered
    assert (
        "public_registry_authority_accepted_runtime_pending"
        not in rendered
    )
    assert source.url not in rendered

def test_public_runtime_dispatch_source_is_isolated_from_trusted_worker():
    source = SOURCE_PATH.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(SOURCE_PATH))
    worker = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_start_public_registry_runtime"
    )
    worker_text = ast.get_source_segment(source, worker) or ""

    for token in (
        "activate_admitted_public_runtime",
        "output_bypass=True",
        "emit_func=None",
        "urls=[resolved_url]",
        "principal=None",
        "principal_source=None",
        "inspection_emit_func=None",
    ):
        assert token in worker_text

    assert "_start_pipeline_worker" not in worker_text
    assert "log_run_event" not in worker_text
    assert "_snapshot_output_artifacts" not in worker_text
