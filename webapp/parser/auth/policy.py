# Certificate-enforcement policy extracted from the composition root.
#
# Runtime dependencies are rebound by compatibility wrappers in
# Smart_Elections_Parser_Webapp.py so request behavior and existing
# monkeypatch seams remain intact during Tranche 1.

from __future__ import annotations

from contextvars import ContextVar


_RUNTIME_BINDINGS: ContextVar[dict[str, object]] = ContextVar(
    "electionpulse_authority_policy_runtime",
    default={},
)


def configure_runtime(**bindings: object) -> None:
    # Bind current composition-root dependencies for this call context.
    _RUNTIME_BINDINGS.set(dict(bindings))


def _runtime_binding(name: str):
    bindings = _RUNTIME_BINDINGS.get()
    if name not in bindings:
        raise RuntimeError(f"Authority policy runtime binding missing: {name}")
    return bindings[name]


def _auth_mode_requires_certificate() -> bool:
    CERT_ENFORCEMENT_MODE = _runtime_binding("CERT_ENFORCEMENT_MODE")
    return CERT_ENFORCEMENT_MODE == "mutations"


def _cert_required_response(reason: str):
    AZURE_CLIENT_CERT_MODE = _runtime_binding("AZURE_CLIENT_CERT_MODE")
    CERT_ENFORCEMENT_MODE = _runtime_binding("CERT_ENFORCEMENT_MODE")
    _request_wants_json = _runtime_binding("_request_wants_json")
    jsonify = _runtime_binding("jsonify")
    redirect = _runtime_binding("redirect")
    request = _runtime_binding("request")
    sanitize_internal_next = _runtime_binding("sanitize_internal_next")
    url_for = _runtime_binding("url_for")

    wants_json = _request_wants_json()

    fallback_path = (
        request.path
        or "/"
    )

    if request.query_string:
        query_string = (
            request.query_string
            .decode(
                "utf-8",
                "ignore",
            )
        )

        if query_string:
            fallback_path += (
                f"?{query_string}"
            )

    auth_next = sanitize_internal_next(
        request.args.get("next"),
        fallback=fallback_path,
    )

    auth_url = url_for(
        "auth_welcome",
        next=auth_next,
    )

    challenge_url = url_for(
        "auth_challenge",
        next=auth_next,
    )

    if wants_json:
        return jsonify({
            "error": "certificate_required",
            "reason": reason,
            "auth_url": auth_url,
            "challenge_url": challenge_url,
            "certificate_policy": CERT_ENFORCEMENT_MODE,
            "azure_client_cert_mode": AZURE_CLIENT_CERT_MODE,
        }), 401

    return redirect(
        auth_url
    )


def _require_client_cert(reason: str):
    DEPLOY_ENV = _runtime_binding("DEPLOY_ENV")
    REQUIRE_CERT_FOR_MUTATIONS = _runtime_binding("REQUIRE_CERT_FOR_MUTATIONS")
    _auth_mode_requires_certificate = _runtime_binding(
        "_auth_mode_requires_certificate"
    )
    _cert_required_response = _runtime_binding("_cert_required_response")
    _is_local_request = _runtime_binding("_is_local_request")
    get_request_principal = _runtime_binding("get_request_principal")
    hmac = _runtime_binding("hmac")
    os = _runtime_binding("os")
    request = _runtime_binding("request")

    if not _auth_mode_requires_certificate() or not REQUIRE_CERT_FOR_MUTATIONS:
        return None
    if DEPLOY_ENV == "local" and _is_local_request():
        return None
    admin_token = os.environ.get("ADMIN_JWT_TOKEN")
    auth_hdr = (request.headers.get("Authorization") or "").strip()
    if admin_token and auth_hdr.lower().startswith("bearer "):
        try:
            if hmac.compare_digest(
                auth_hdr.split(None, 1)[1].strip(),
                admin_token,
            ):
                return None
        except Exception:
            pass
    principal, _, _ = get_request_principal()
    if principal and principal.startswith("cert:"):
        return None
    return _cert_required_response(reason)


def _require_cert_for_socket_action(
    action: str,
    session_id: str | None = None,
) -> bool:
    DEPLOY_ENV = _runtime_binding("DEPLOY_ENV")
    REQUIRE_CERT_FOR_MUTATIONS = _runtime_binding("REQUIRE_CERT_FOR_MUTATIONS")
    _auth_mode_requires_certificate = _runtime_binding(
        "_auth_mode_requires_certificate"
    )
    _is_local_request = _runtime_binding("_is_local_request")
    emit = _runtime_binding("emit")
    get_request_principal = _runtime_binding("get_request_principal")
    hmac = _runtime_binding("hmac")
    normalize_log_obj = _runtime_binding("normalize_log_obj")
    os = _runtime_binding("os")
    request = _runtime_binding("request")
    session_manager = _runtime_binding("session_manager")

    if not _auth_mode_requires_certificate() or not REQUIRE_CERT_FOR_MUTATIONS:
        return True
    if DEPLOY_ENV == "local" and _is_local_request():
        return True
    admin_token = os.environ.get("ADMIN_JWT_TOKEN")
    auth_hdr = (request.headers.get("Authorization") or "").strip()
    if admin_token and auth_hdr.lower().startswith("bearer "):
        try:
            if hmac.compare_digest(
                auth_hdr.split(None, 1)[1].strip(),
                admin_token,
            ):
                return True
        except Exception:
            pass
    # Preserve existing socket-event behavior: a cached authority principal
    # can satisfy this gate when the original HTTP headers are unavailable.
    if session_id:
        try:
            meta = session_manager.get_metadata(session_id)
            if meta and meta.get("principal"):
                cached_principal = meta.get("principal")
                if isinstance(cached_principal, str) and (
                    cached_principal.startswith("cert:")
                    or cached_principal.startswith("sso:")
                    or cached_principal.startswith("dev:")
                ):
                    return True
        except Exception:
            pass
    principal, _, _ = get_request_principal()
    if principal and principal.startswith("cert:"):
        return True
    if session_id:
        try:
            session_manager.update_metadata(
                session_id,
                auth_blocked=True,
            )
        except Exception:
            pass
    emit(
        "parser_output",
        normalize_log_obj({
            "level": "WARNING",
            "type": "auth",
            "message": f"Certificate required for {action}.",
            "session_id": session_id,
        }),
        room=session_id,
    )
    return False
