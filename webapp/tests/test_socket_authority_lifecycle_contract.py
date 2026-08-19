# Contracts for Tranche 1G-A socket authority lifecycle extraction.

from __future__ import annotations

import inspect
from pathlib import Path

from webapp.parser.auth.socket_lifecycle import (
    acknowledge_certificate_reauth,
    derive_certificate_fingerprint,
    disconnect_socket_authority,
    process_certificate_session_authority,
)


ROOT = Path(__file__).resolve().parents[2]
MONOLITH = ROOT / "webapp" / "Smart_Elections_Parser_Webapp.py"
LIFECYCLE = ROOT / "webapp" / "parser" / "auth" / "socket_lifecycle.py"


class _Logger:
    def __init__(self):
        self.info_events = []
        self.warning_events = []

    def info(self, payload):
        self.info_events.append(payload)

    def warning(self, payload):
        self.warning_events.append(payload)


class _SessionManager:
    def __init__(self, *, changed=False, expired=False):
        self.changed = changed
        self.expired = expired
        self.cached = []
        self.resolved = {}
        self.unbound = {}
        self.popped = []

    def cert_changed(self, session_id, fingerprint, metadata):
        return self.changed

    def cache_cert(self, session_id, fingerprint, metadata, principal):
        self.cached.append((session_id, fingerprint, metadata, principal))

    def cert_expired(self, session_id):
        return self.expired

    def resolve_socket(self, sid):
        return self.resolved.get(sid)

    def unbind_socket(self, sid):
        return self.unbound.get(sid)

    def pop_emitter(self, session_id):
        self.popped.append(session_id)


def test_certificate_fingerprint_preserves_cert_principal_behavior():
    logger = _Logger()
    fingerprint = derive_certificate_fingerprint(
        "cert:abc123",
        {
            "cn": "Example",
            "is_expired": True,
            "expiry_date": "2030-01-01T00:00:00Z",
        },
        "ignored-header",
        logger=logger,
    )
    assert fingerprint == "abc123"
    assert logger.warning_events == []


def test_certificate_fingerprint_preserves_arr_header_hash_behavior():
    logger = _Logger()
    fingerprint = derive_certificate_fingerprint(
        "sso:user@example.org",
        {
            "cn": "Example",
            "is_expired": False,
        },
        "certificate-material",
        logger=logger,
    )
    assert isinstance(fingerprint, str)
    assert len(fingerprint) == 16


def test_certificate_change_locks_and_emits_before_cache():
    logger = _Logger()
    manager = _SessionManager(changed=True, expired=False)
    transitions = []
    emits = []

    def transition(*args, **kwargs):
        transitions.append((args, kwargs))

    def emit(event, payload, **kwargs):
        emits.append((event, payload, kwargs))

    result = process_certificate_session_authority(
        "sid-1",
        "cert:abc",
        {
            "cn": "Example",
            "expiry_date": "2030-01-01T00:00:00Z",
        },
        "abc",
        session_manager=manager,
        transition_session=transition,
        idle_state="IDLE",
        prepare_phase="PREPARE",
        emit_event=emit,
        logger=logger,
    )

    assert result == {"changed": True, "expired": False}
    assert len(transitions) == 1
    assert transitions[0][1]["locked"] is True
    assert transitions[0][1]["extras"] == {
        "auth_blocked": True,
        "auth_block_reason": "cert_changed",
    }
    assert [item[0] for item in emits] == [
        "cert_changed",
        "auth_blocked",
    ]
    assert len(manager.cached) == 1


def test_certificate_expiry_emits_without_changing_lock_state():
    logger = _Logger()
    manager = _SessionManager(changed=False, expired=True)
    transitions = []
    emits = []

    result = process_certificate_session_authority(
        "sid-2",
        "cert:def",
        {"expiry_date": "2000-01-01T00:00:00Z"},
        "def",
        session_manager=manager,
        transition_session=lambda *a, **k: transitions.append((a, k)),
        idle_state="IDLE",
        prepare_phase="PREPARE",
        emit_event=lambda e, p, **k: emits.append((e, p, k)),
        logger=logger,
    )

    assert result == {"changed": False, "expired": True}
    assert transitions == []
    assert [item[0] for item in emits] == ["cert_expired"]


def test_disconnect_preserves_resolve_unbind_and_pop_behavior():
    logger = _Logger()
    manager = _SessionManager()
    manager.resolved["socket-1"] = "session-1"
    manager.unbound["socket-1"] = "session-1"

    logical = disconnect_socket_authority(
        safe_sid=lambda: "socket-1",
        request_sid=None,
        session_manager=manager,
        logger=logger,
    )

    assert logical == "session-1"
    assert manager.popped == ["session-1"]


def test_reauth_ack_preserves_unlock_payload_and_emit_contract():
    logger = _Logger()
    transitions = []
    emits = []

    session_id = acknowledge_certificate_reauth(
        {"session_id": "session-2"},
        resolve_session_id=lambda payload, create_if_missing: (
            payload.get("session_id")
        ),
        transition_session=lambda *a, **k: transitions.append((a, k)),
        idle_state="IDLE",
        prepare_phase="PREPARE",
        emit_event=lambda e, p, **k: emits.append((e, p, k)),
        logger=logger,
    )

    assert session_id == "session-2"
    assert transitions[0][1]["locked"] is False
    assert transitions[0][1]["extras"] == {
        "auth_blocked": False,
        "auth_block_reason": None,
    }
    assert emits == [
        (
            "auth_unblocked",
            {"session_id": "session-2"},
            {"room": "session-2"},
        )
    ]


def test_monolith_handlers_delegate_without_changing_public_names():
    source = MONOLITH.read_text(encoding="utf-8")

    assert "def handle_connect(auth=None):" in source
    assert "def handle_disconnect(arg=None) -> None:" in source
    assert "def handle_ack_cert_reauth(data=None) -> None:" in source
    assert "_socket_lifecycle.derive_certificate_fingerprint(" in source
    assert "_socket_lifecycle.process_certificate_session_authority(" in source
    assert "_socket_lifecycle.disconnect_socket_authority(" in source
    assert "_socket_lifecycle.acknowledge_certificate_reauth(" in source


def test_lifecycle_module_does_not_absorb_storage_routes_or_frontend():
    source = LIFECYCLE.read_text(encoding="utf-8")

    forbidden = (
        "Flask",
        "socketio =",
        "@socketio.on",
        "SessionManager(",
        "ALLOW_AUTO_SESSION_REUSE",
        "last_contest_options",
        "cleanup_sessions",
        "monitor_db_for_alerts",
    )

    for token in forbidden:
        assert token not in source


def test_helper_signatures_are_explicit_dependency_injection():
    assert "logger" in inspect.signature(
        derive_certificate_fingerprint
    ).parameters
    assert "session_manager" in inspect.signature(
        process_certificate_session_authority
    ).parameters
    assert "session_manager" in inspect.signature(
        disconnect_socket_authority
    ).parameters
    assert "resolve_session_id" in inspect.signature(
        acknowledge_certificate_reauth
    ).parameters
