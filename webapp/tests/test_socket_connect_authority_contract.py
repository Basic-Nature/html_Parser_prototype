# Contracts for Tranche 1G-B connect authority delegation.

from __future__ import annotations

from pathlib import Path

from webapp.parser.auth.context import resolve_session_reuse_policy
from webapp.parser.auth.socket_lifecycle import socket_connection_admitted


ROOT = Path(__file__).resolve().parents[2]
MONOLITH = ROOT / "webapp" / "Smart_Elections_Parser_Webapp.py"
CONTEXT = ROOT / "webapp" / "parser" / "auth" / "context.py"
SOCKET_LIFECYCLE = ROOT / "webapp" / "parser" / "auth" / "socket_lifecycle.py"


class _Mapping:
    def __init__(self, values=None, *, raises=False):
        self.values = dict(values or {})
        self.raises = raises

    def get(self, key, default=None):
        if self.raises:
            raise RuntimeError("mapping unavailable")
        return self.values.get(key, default)


def test_reuse_policy_preserves_auto_reuse_default():
    assert resolve_session_reuse_policy(
        {},
        "sso",
        allow_auto_session_reuse=True,
        request_args=_Mapping(),
        request_headers=_Mapping(),
    ) == (False, True)


def test_reuse_policy_preserves_explicit_payload_hint():
    assert resolve_session_reuse_policy(
        {"reuse_session": "yes"},
        "sso",
        allow_auto_session_reuse=False,
        request_args=_Mapping(),
        request_headers=_Mapping(),
    ) == (True, True)


def test_reuse_policy_preserves_query_and_header_hints():
    assert resolve_session_reuse_policy(
        {},
        "sso",
        allow_auto_session_reuse=False,
        request_args=_Mapping({"reuse_session": "reuse"}),
        request_headers=_Mapping(),
    ) == (True, True)

    assert resolve_session_reuse_policy(
        {},
        "sso",
        allow_auto_session_reuse=False,
        request_args=_Mapping(),
        request_headers=_Mapping({"X-Reuse-Session": "on"}),
    ) == (True, True)


def test_dev_bypass_requires_explicit_reuse_hint():
    assert resolve_session_reuse_policy(
        {},
        "dev_bypass",
        allow_auto_session_reuse=True,
        request_args=_Mapping(),
        request_headers=_Mapping(),
    ) == (False, False)

    assert resolve_session_reuse_policy(
        {"reuse": True},
        "dev_bypass",
        allow_auto_session_reuse=True,
        request_args=_Mapping(),
        request_headers=_Mapping(),
    ) == (True, True)


def test_reuse_policy_preserves_mapping_failure_fallback():
    assert resolve_session_reuse_policy(
        {},
        "sso",
        allow_auto_session_reuse=False,
        request_args=_Mapping(raises=True),
        request_headers=_Mapping(),
    ) == (False, False)


def test_socket_principal_admission_matrix():
    assert socket_connection_admitted(
        "cert:abc",
        allow_anonymous=False,
    )
    assert socket_connection_admitted(
        "sso:user@example.org",
        allow_anonymous=False,
    )
    assert socket_connection_admitted(
        None,
        allow_anonymous=True,
    )
    assert not socket_connection_admitted(
        None,
        allow_anonymous=False,
    )


def test_context_resolver_uses_canonical_reuse_policy():
    source = CONTEXT.read_text(encoding="utf-8")
    assert "resolve_session_reuse_policy(" in source
    assert "def _truthy(val):" not in source
    assert "ALLOW_AUTO_SESSION_REUSE or reuse_hint" not in source


def test_connect_handler_delegates_authority_decisions():
    source = MONOLITH.read_text(encoding="utf-8")
    assert "_socket_lifecycle.socket_connection_admitted(" in source
    assert "_authority_context.resolve_session_reuse_policy(" in source
    assert "def _truthy(val):" not in source
    assert "ALLOW_AUTO_SESSION_REUSE or reuse_hint" not in source


def test_connect_transport_compatibility_remains_in_handler():
    source = MONOLITH.read_text(encoding="utf-8")
    required = (
        "requested_session_id",
        "prev_session_id",
        "logical_session_id",
        "cancellation_manager.remove",
        "_recover_stale_session",
        "last_contest_options",
        "list_active_metadata",
        "_socket_lifecycle.process_certificate_session_authority",
    )
    for token in required:
        assert token in source


def test_join_clone_and_session_list_handlers_remain_in_monolith():
    source = MONOLITH.read_text(encoding="utf-8")
    for function_name in (
        "on_join",
        "handle_clone_session",
        "handle_get_session_history",
        "handle_get_sessions",
    ):
        assert f"def {function_name}" in source


def test_socket_lifecycle_does_not_absorb_general_session_reuse():
    source = SOCKET_LIFECYCLE.read_text(encoding="utf-8")
    forbidden = (
        "ALLOW_AUTO_SESSION_REUSE",
        "requested_session_id",
        "prev_session_id",
        "logical_session_id",
        "cleanup_sessions",
        "last_contest_options",
        "SessionManager(",
    )
    for token in forbidden:
        assert token not in source
