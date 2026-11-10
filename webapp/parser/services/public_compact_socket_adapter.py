"""P6C5B2E REVIEW CANDIDATE ONLY — not installed.

Process-local compact result delivery for server-derived public registry sessions.
Requires a separate activation and worker-topology decision before any public use.
"""
from __future__ import annotations

import secrets
import threading
from typing import Any

from webapp.parser.services.public_contest_page_broker import (
    PublicContestPageBroker, PublicPageDenied, TrustedSocketIdentity,
)

_broker_lock = threading.Lock()
_broker: PublicContestPageBroker | None = None


def broker_instance() -> PublicContestPageBroker:
    global _broker
    with _broker_lock:
        if _broker is None:
            _broker = PublicContestPageBroker(secrets.token_bytes(32))
        return _broker


def trusted_peer(*, manager: Any, socket_sid: str, source_id: str,
                 origin_allowed: bool) -> TrustedSocketIdentity:
    if not isinstance(socket_sid, str) or not socket_sid:
        raise PublicPageDenied("PUBLIC_COMPACT_SESSION_UNAVAILABLE")
    server_sid = manager.resolve_socket(socket_sid)
    if not isinstance(server_sid, str):
        raise PublicPageDenied("PUBLIC_COMPACT_SESSION_UNAVAILABLE")
    meta = manager.get_metadata(server_sid) or {}
    if meta.get("public_compact_registry_source_id") != source_id:
        raise PublicPageDenied("PUBLIC_COMPACT_SOURCE_MISMATCH")
    return TrustedSocketIdentity(
        server_session_id=server_sid,
        socket_sid=socket_sid,
        registry_source_id=source_id,
        public_registry_session=meta.get("public_registry_session") is True,
        origin_authorized=origin_allowed is True,
        principal_present=bool(meta.get("principal")),
    )


def request_peer(*, manager: Any, socket_sid: str, origin_allowed: bool) -> TrustedSocketIdentity:
    server_sid = manager.resolve_socket(socket_sid)
    if not isinstance(server_sid, str):
        raise PublicPageDenied("PUBLIC_COMPACT_SESSION_UNAVAILABLE")
    meta = manager.get_metadata(server_sid) or {}
    source_id = meta.get("public_compact_registry_source_id")
    if not isinstance(source_id, str):
        raise PublicPageDenied("PUBLIC_COMPACT_SOURCE_UNAVAILABLE")
    return trusted_peer(manager=manager, socket_sid=socket_sid,
                        source_id=source_id, origin_allowed=origin_allowed)


def exact_origin_allowed(*, origin: str | None, allowed: Any) -> bool:
    """Only explicit origins qualify for this opt-in candidate; wildcard is not used."""
    if not isinstance(origin, str) or not origin:
        return False
    if isinstance(allowed, str):
        return origin == allowed
    if isinstance(allowed, (tuple, list, set)):
        return origin in allowed
    return False


def process_action(*, action: str, data: Any, manager: Any, request: Any,
                   allowed_origins: Any, emit: Any) -> None:
    """Called only by separately opted-in event handlers, not by client labels."""
    event_map = {
        'next': 'public_registry_compact_page',
        'finish': 'public_registry_compact_complete',
        'cancel': 'public_registry_compact_cancelled',
    }
    if action not in event_map:
        raise ValueError('Unknown compact action')
    socket_sid = getattr(request, 'sid', None)
    headers = getattr(request, 'headers', {})
    origin = headers.get('Origin') if hasattr(headers,'get') else None
    try:
        peer = request_peer(manager=manager, socket_sid=socket_sid,
                            origin_allowed=exact_origin_allowed(
                                origin=origin, allowed=allowed_origins))
        broker = broker_instance()
        if action == 'next': result = broker.next_page(peer, data)
        elif action == 'finish': result = broker.finish(peer, data)
        else: result = broker.cancel(peer, data)
        emit(event_map[action], result, room=peer.server_session_id)
    except (PublicPageDenied, TypeError, ValueError):
        # Bound result status only; the source preview remains undisclosed.
        if isinstance(socket_sid, str) and socket_sid:
            emit('public_registry_compact_unavailable',
                 {'state': 'unavailable'}, room=socket_sid)
