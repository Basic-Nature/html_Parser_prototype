# Socket authority lifecycle helpers.
#
# This module owns certificate/session authority transitions for Socket.IO
# connections while leaving generic session reuse, parser orchestration,
# Socket.IO route registration, and SessionManager storage in their existing
# owners.

from __future__ import annotations

import hashlib
from typing import Any, Callable, Mapping


def derive_certificate_fingerprint(
    principal: str | None,
    cert_metadata: Mapping[str, Any] | None,
    cert_header: str,
    *,
    logger: Any,
) -> str | None:
    """Preserve the existing socket-connect certificate fingerprint behavior."""
    cert_fingerprint = None
    cert_expired = False

    if cert_metadata and isinstance(cert_metadata, dict):
        try:
            if isinstance(principal, str) and principal.startswith("cert:"):
                cert_fingerprint = principal.split(":", 1)[1]
            else:
                if cert_header:
                    cert_fingerprint = hashlib.sha256(
                        cert_header.encode("utf-8")
                    ).hexdigest()[:16]

                cert_expired = cert_metadata.get("is_expired", False)

                if cert_expired:
                    logger.warning({
                        "level": "WARNING",
                        "type": "auth",
                        "message": "Client certificate is expired.",
                        "session_id": None,
                        "principal": principal,
                        "cert_cn": cert_metadata.get("cn"),
                        "expiry_date": cert_metadata.get("expiry_date"),
                    })
        except Exception:
            pass

    return cert_fingerprint


def process_certificate_session_authority(
    session_id: str,
    principal: str | None,
    cert_metadata: Mapping[str, Any],
    cert_fingerprint: str,
    *,
    session_manager: Any,
    transition_session: Callable[..., Any],
    idle_state: Any,
    prepare_phase: Any,
    emit_event: Callable[..., Any],
    logger: Any,
) -> dict[str, bool]:
    """Apply certificate-change and expiry authority transitions to a session."""
    changed = bool(
        session_manager.cert_changed(
            session_id,
            cert_fingerprint,
            cert_metadata,
        )
    )

    if changed:
        logger.info({
            "level": "INFO",
            "type": "auth",
            "message": "Certificate changed or new for session.",
            "session_id": session_id,
            "principal": principal,
            "cert_cn": cert_metadata.get("cn"),
            "fingerprint": cert_fingerprint,
        })

        transition_session(
            session_id,
            idle_state,
            locked=True,
            phase=prepare_phase,
            broadcast=False,
            extras={
                "auth_blocked": True,
                "auth_block_reason": "cert_changed",
            },
        )

        try:
            emit_event(
                "cert_changed",
                {
                    "session_id": session_id,
                    "fingerprint": cert_fingerprint,
                    "cert_metadata": cert_metadata,
                    "principal": principal,
                },
                room=session_id,
            )
        except Exception:
            pass

        try:
            emit_event(
                "auth_blocked",
                {
                    "session_id": session_id,
                    "reason": "cert_changed",
                },
                room=session_id,
            )
        except Exception:
            pass

    session_manager.cache_cert(
        session_id,
        cert_fingerprint,
        cert_metadata,
        principal,
    )

    expired = bool(
        session_manager.cert_expired(session_id)
    )

    if expired:
        logger.warning({
            "level": "WARNING",
            "type": "auth",
            "message": "Cached certificate is expired.",
            "session_id": session_id,
            "principal": principal,
            "expiry_date": cert_metadata.get("expiry_date"),
        })

        try:
            emit_event(
                "cert_expired",
                {
                    "session_id": session_id,
                    "principal": principal,
                    "expiry_date": cert_metadata.get("expiry_date"),
                },
                room=session_id,
            )
        except Exception:
            pass

    return {
        "changed": changed,
        "expired": expired,
    }


def disconnect_socket_authority(
    *,
    safe_sid: Callable[[], Any],
    request_sid: Any,
    session_manager: Any,
    logger: Any,
) -> str | None:
    """Unbind Socket.IO transport identity from its logical session."""
    try:
        req_sid = safe_sid()
    except Exception:
        req_sid = request_sid
        if not isinstance(req_sid, str):
            req_sid = None

    logical = None

    if req_sid:
        logical = session_manager.resolve_socket(req_sid)

    unbound_session = (
        session_manager.unbind_socket(req_sid)
        if req_sid
        else None
    )
    logical = logical or unbound_session

    logger.info({
        "level": "INFO",
        "type": "status",
        "message": (
            "Client disconnected "
            f"(socket_sid={req_sid}, session_id={logical})"
        ),
        "session_id": logical,
    })

    if logical:
        session_manager.pop_emitter(logical)

    return logical


def acknowledge_certificate_reauth(
    data: Any,
    *,
    resolve_session_id: Callable[..., str | None],
    transition_session: Callable[..., Any],
    idle_state: Any,
    prepare_phase: Any,
    emit_event: Callable[..., Any],
    logger: Any,
) -> str | None:
    """Acknowledge certificate reauthentication and unblock a logical session."""
    payload = data or {}
    session_id = resolve_session_id(
        payload,
        create_if_missing=False,
    )

    if not session_id:
        logger.warning({
            "level": "WARNING",
            "type": "auth",
            "message": "No session_id provided for cert reauth ack.",
            "session_id": None,
        })
        return None

    transition_session(
        session_id,
        idle_state,
        locked=False,
        phase=prepare_phase,
        broadcast=False,
        extras={
            "auth_blocked": False,
            "auth_block_reason": None,
        },
    )

    try:
        emit_event(
            "auth_unblocked",
            {
                "session_id": session_id,
            },
            room=session_id,
        )
    except Exception:
        pass

    logger.info({
        "level": "INFO",
        "type": "auth",
        "message": (
            "Certificate reauth acknowledged; "
            "session unblocked."
        ),
        "session_id": session_id,
    })

    return session_id

def socket_connection_admitted(
    principal: str | None,
    *,
    allow_anonymous: bool,
) -> bool:
    """Return whether the Socket.IO connection passes principal admission."""
    return bool(principal) or bool(allow_anonymous)
