"""Principal/session authority context extracted from the composition root.

Runtime dependencies are rebound by compatibility wrappers in
Smart_Elections_Parser_Webapp.py so request behavior and existing
monkeypatch seams remain intact during Tranche 1.
"""

from __future__ import annotations

from contextvars import ContextVar
import time


_RUNTIME_BINDINGS: ContextVar[dict[str, object]] = ContextVar(
    "electionpulse_authority_context_runtime",
    default={},
)


def configure_runtime(**bindings: object) -> None:
    """Bind current composition-root dependencies for this call context."""
    _RUNTIME_BINDINGS.set(dict(bindings))


def _runtime_binding(name: str):
    bindings = _RUNTIME_BINDINGS.get()
    if name not in bindings:
        raise RuntimeError(f"Authority context runtime binding missing: {name}")
    return bindings[name]


def _clear_certificate_session_authority(session) -> None:
    session.pop("certificate_session_principal", None)
    session.pop("certificate_session_established_at", None)


def _cache_certificate_session_authority(session, principal: str) -> None:
    if not isinstance(principal, str) or not principal.startswith("cert:"):
        return
    session["certificate_session_principal"] = principal
    session["certificate_session_established_at"] = int(time.time())


def _get_certificate_session_authority(
    session,
    *,
    ttl_seconds: int,
) -> str | None:
    principal = session.get("certificate_session_principal")
    established_at = session.get("certificate_session_established_at")

    if (
        not isinstance(principal, str)
        or not principal.startswith("cert:")
        or not isinstance(established_at, (int, float))
    ):
        _clear_certificate_session_authority(session)
        return None

    age_seconds = max(0, int(time.time() - float(established_at)))

    if age_seconds > ttl_seconds:
        _clear_certificate_session_authority(session)
        return None

    return principal


def get_request_principal():
    """
    Resolve effective request authority.

    Current client-certificate proof always wins. A successful certificate
    request establishes a bounded signed Flask-session authority so later
    HTTP/Socket.IO requests can retain the same pseudonymous principal without
    falsely claiming that the certificate is physically present on that later
    TLS request.
    """
    ALLOW_DEV_NO_PRINCIPAL = _runtime_binding('ALLOW_DEV_NO_PRINCIPAL')
    CERT_SESSION_AUTH_TTL_SECONDS = _runtime_binding(
        'CERT_SESSION_AUTH_TTL_SECONDS'
    )
    _is_local_host = _runtime_binding('_is_local_host')
    extract_client_principal = _runtime_binding('extract_client_principal')
    request = _runtime_binding('request')
    session = _runtime_binding('session')

    principal, source, cert_meta = extract_client_principal(request.headers)

    if principal:
        if isinstance(principal, str) and principal.startswith("cert:"):
            _cache_certificate_session_authority(session, principal)
        return (principal, source, cert_meta)

    # A certificate that is physically present on this request but rejected by
    # the canonical production trust decision must remain authoritative as a
    # rejection. Preserve its source/trust metadata, clear any older bounded
    # certificate-session authority, and fail closed instead of silently
    # falling through to cached session or lower-authority identity sources.
    presented_certificate_rejected = bool(
        source
        and isinstance(cert_meta, dict)
        and cert_meta.get("trust_required") is True
        and cert_meta.get("trust_valid") is False
    )

    if presented_certificate_rejected:
        _clear_certificate_session_authority(session)
        return (None, source, cert_meta)

    cached_certificate_principal = _get_certificate_session_authority(
        session,
        ttl_seconds=CERT_SESSION_AUTH_TTL_SECONDS,
    )

    if cached_certificate_principal:
        return (
            cached_certificate_principal,
            "certificate_session",
            None,
        )

    host = (request.host or '').lower()
    is_local = _is_local_host(host)

    if ALLOW_DEV_NO_PRINCIPAL and is_local:
        remote = request.remote_addr or 'local'
        return (f'dev:{remote}', 'dev_bypass', None)

    return (None, None, None)


def _resolve_cert_session_id(principal: str | None) -> str | None:
    if not principal or not principal.startswith('cert:'):
        return None
    fingerprint = principal.split(':', 1)[1].strip()
    if not fingerprint:
        return None
    return f'cert_{fingerprint[:32]}'


def _derive_auth_context(principal: str | None, principal_source: str | None) -> dict:
    if principal and principal.startswith('cert:'):
        return {'auth_tier': 'cert', 'auth_trusted': True, 'principal_source': principal_source}
    if principal and principal.startswith('sso:'):
        return {'auth_tier': 'sso', 'auth_trusted': True, 'principal_source': principal_source}
    if principal and principal.startswith('dev:'):
        return {'auth_tier': 'dev', 'auth_trusted': True, 'principal_source': principal_source}
    return {'auth_tier': 'anon', 'auth_trusted': False, 'principal_source': principal_source}


def _apply_auth_context(session_id: str, principal: str | None, principal_source: str | None) -> None:
    _derive_auth_context = _runtime_binding('_derive_auth_context')
    session_manager = _runtime_binding('session_manager')
    if not session_id:
        return
    updates = _derive_auth_context(principal, principal_source)
    if principal:
        updates['principal'] = principal
    session_manager.update_metadata(session_id, **updates)


def _session_has_principal(session_id: str) -> bool:
    session_manager = _runtime_binding('session_manager')
    meta = session_manager.get_metadata(session_id)
    return bool(meta and meta.get('principal'))


def _truthy_reuse_value(value) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    try:
        normalized = str(value).strip().lower()
    except Exception:
        return False
    return normalized in {
        "1",
        "true",
        "yes",
        "y",
        "on",
        "reuse",
        "reuse_session",
    }


def resolve_session_reuse_policy(
    data,
    principal_source: str | None,
    *,
    allow_auto_session_reuse: bool,
    request_args=None,
    request_headers=None,
) -> tuple[bool, bool]:
    """Return (reuse_hint, allow_reuse) for request/session resolution."""
    reuse_hint = False

    if isinstance(data, dict):
        reuse_hint = _truthy_reuse_value(
            data.get("reuse_session") or data.get("reuse")
        )

    try:
        arg_reuse = (
            request_args.get("reuse_session")
            if request_args is not None
            else None
        )
        header_reuse = (
            request_headers.get("X-Reuse-Session")
            if request_headers is not None
            else None
        )
        reuse_hint = bool(
            reuse_hint
            or _truthy_reuse_value(arg_reuse)
            or _truthy_reuse_value(header_reuse)
        )
    except Exception:
        pass

    allow_reuse = bool(
        allow_auto_session_reuse
        or reuse_hint
        or principal_source == "certificate_session"
    )

    if (
        principal_source == "dev_bypass"
        and not reuse_hint
    ):
        allow_reuse = False

    return reuse_hint, allow_reuse

def resolve_session_id(data=None, create_if_missing=True):
    ALLOW_AUTO_SESSION_REUSE = _runtime_binding('ALLOW_AUTO_SESSION_REUSE')
    CERT_SESSION_BINDING = _runtime_binding('CERT_SESSION_BINDING')
    CERT_SESSION_CAP = _runtime_binding('CERT_SESSION_CAP')
    ENABLE_FINGERPRINT_SESSION_RECOVERY = _runtime_binding('ENABLE_FINGERPRINT_SESSION_RECOVERY')
    _apply_auth_context = _runtime_binding('_apply_auth_context')
    _ensure_quick_copy_dir = _runtime_binding('_ensure_quick_copy_dir')
    _resolve_cert_session_id = _runtime_binding('_resolve_cert_session_id')
    _session_has_principal = _runtime_binding('_session_has_principal')
    client_fingerprint = _runtime_binding('client_fingerprint')
    get_request_principal = _runtime_binding('get_request_principal')
    request = _runtime_binding('request')
    safe_get = _runtime_binding('safe_get')
    safe_sid = _runtime_binding('safe_sid')
    secrets = _runtime_binding('secrets')
    session = _runtime_binding('session')
    session_manager = _runtime_binding('session_manager')

    def _log_resolution(decision: str, sid_val: str | None, reason: str | None=None):
        try:
            logger.info({'level': 'INFO', 'type': 'auth', 'message': f'Session resolution: {decision}', 'session_id': sid_val, 'reason': reason})
        except Exception:
            pass

    try:
        socket_sid = safe_sid()
    except Exception:
        socket_sid = getattr(request, 'sid', None)

    if not isinstance(socket_sid, str) or not socket_sid:
        socket_sid = None

    def _bind_socket_if_available(resolved_session_id: str) -> None:
        if socket_sid:
            session_manager.bind_socket(socket_sid, resolved_session_id)

    principal, principal_source, _ = get_request_principal()
    reuse_hint, allow_reuse = resolve_session_reuse_policy(
        data,
        principal_source,
        allow_auto_session_reuse=ALLOW_AUTO_SESSION_REUSE,
        request_args=getattr(request, "args", None),
        request_headers=getattr(request, "headers", None),
    )
    sid = None
    if isinstance(data, dict):
        sid = safe_get(data, 'session_id')
    if isinstance(sid, str) and sid:
        _bind_socket_if_available(sid)
        if principal:
            session_manager.set_principal(sid, principal, principal_source)
        _apply_auth_context(sid, principal, principal_source)
        _log_resolution('explicit_session_id', sid, 'payload session_id')
        return sid
    if allow_reuse and principal:
        mapped_principal = session_manager.resolve_principal(principal)
        if isinstance(mapped_principal, str) and mapped_principal:
            _bind_socket_if_available(mapped_principal)
            session['logical_session_id'] = mapped_principal
            _apply_auth_context(mapped_principal, principal, principal_source)
            _log_resolution('reuse_principal', mapped_principal, principal_source)
            return mapped_principal
    if allow_reuse:
        mapped = session_manager.resolve_socket(socket_sid)
        if isinstance(mapped, str) and mapped:
            if not principal and _session_has_principal(mapped):
                _log_resolution('reuse_socket_blocked', mapped, 'principal_required')
            else:
                if principal:
                    session_manager.set_principal(mapped, principal, principal_source)
                _apply_auth_context(mapped, principal, principal_source)
                _log_resolution('reuse_socket', mapped, 'socket bind')
                return mapped
    if allow_reuse:
        cookie_sid = session.get('logical_session_id')
        if isinstance(cookie_sid, str) and cookie_sid:
            if principal_source == "certificate_session" and principal:
                # The Flask session is signed and the bounded certificate
                # authority TTL was validated by get_request_principal().
                # Re-hydrate SessionManager on this worker if necessary.
                session_manager.ensure_session(cookie_sid)
                session_manager.set_principal(
                    cookie_sid,
                    principal,
                    principal_source,
                )
                _bind_socket_if_available(cookie_sid)
                _apply_auth_context(
                    cookie_sid,
                    principal,
                    principal_source,
                )
                _log_resolution(
                    'reuse_certificate_session',
                    cookie_sid,
                    'signed bounded certificate session',
                )
                return cookie_sid

            if not principal and _session_has_principal(cookie_sid):
                _log_resolution(
                    'reuse_cookie_blocked',
                    cookie_sid,
                    'principal_required',
                )
            else:
                _bind_socket_if_available(cookie_sid)
                if principal:
                    session_manager.set_principal(
                        cookie_sid,
                        principal,
                        principal_source,
                    )
                _apply_auth_context(
                    cookie_sid,
                    principal,
                    principal_source,
                )
                _log_resolution(
                    'reuse_cookie',
                    cookie_sid,
                    'logical_session_id cookie',
                )
                return cookie_sid
    fingerprint = client_fingerprint() if ENABLE_FINGERPRINT_SESSION_RECOVERY else None
    if allow_reuse and ENABLE_FINGERPRINT_SESSION_RECOVERY and fingerprint:
        fp_sid = session_manager.resolve_fingerprint(fingerprint)
        if isinstance(fp_sid, str) and fp_sid:
            if not principal and _session_has_principal(fp_sid):
                _log_resolution('reuse_fingerprint_blocked', fp_sid, 'principal_required')
            else:
                _bind_socket_if_available(fp_sid)
                session['logical_session_id'] = fp_sid
                if principal:
                    session_manager.set_principal(fp_sid, principal, principal_source)
                _apply_auth_context(fp_sid, principal, principal_source)
                _log_resolution('reuse_fingerprint', fp_sid, 'fingerprint')
                return fp_sid
    if principal and CERT_SESSION_CAP > 0:
        active_sessions = session_manager.list_principal_sessions(principal, active_only=True)
        if len(active_sessions) >= CERT_SESSION_CAP:
            reuse_sid = session_manager.select_principal_session(principal, active_only=True) or active_sessions[0]
            _bind_socket_if_available(reuse_sid)
            session['logical_session_id'] = reuse_sid
            session_manager.set_principal(reuse_sid, principal, principal_source)
            _apply_auth_context(reuse_sid, principal, principal_source)
            _log_resolution('reuse_principal_cap', reuse_sid, f'cap={CERT_SESSION_CAP}')
            return reuse_sid
    if not create_if_missing:
        return None
    cert_session_id = _resolve_cert_session_id(principal) if CERT_SESSION_BINDING else None
    if cert_session_id:
        mapped_principal = session_manager.resolve_principal(principal) if principal else None
        if mapped_principal and mapped_principal != cert_session_id:
            cert_session_id = mapped_principal
        if ENABLE_FINGERPRINT_SESSION_RECOVERY and fingerprint:
            session_manager.bind_fingerprint(fingerprint, cert_session_id)
        _bind_socket_if_available(cert_session_id)
        session['logical_session_id'] = cert_session_id
        session_manager.ensure_session(cert_session_id)
        _ensure_quick_copy_dir(cert_session_id)
        session_manager.set_principal(cert_session_id, principal, principal_source)
        _apply_auth_context(cert_session_id, principal, principal_source)
        _log_resolution('new_cert_session', cert_session_id, 'cert_binding')
        return cert_session_id
    new_sid = 'sess_' + secrets.token_urlsafe(16)
    if ENABLE_FINGERPRINT_SESSION_RECOVERY and fingerprint:
        session_manager.bind_fingerprint(fingerprint, new_sid)
    _bind_socket_if_available(new_sid)
    session['logical_session_id'] = new_sid
    session_manager.ensure_session(new_sid)
    _ensure_quick_copy_dir(new_sid)
    if principal:
        session_manager.set_principal(new_sid, principal, principal_source)
    _apply_auth_context(new_sid, principal, principal_source)
    _log_resolution('new_session', new_sid, 'created')
    return new_sid
