"""W24AUTHOIDC1E — verified roles with localhost-only development sessions.

Disabled by default. Anonymous parsing and certificate capabilities are
independent. No production session backend is authorized or implemented here.
MemorySessionStore is explicit opt-in, localhost and single-process ONLY;
production/Azure access must fail closed. Do not reuse Doors credentials.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import time
import threading
from urllib.parse import urlsplit

from flask import abort, current_app, jsonify, redirect, request, session

from .electionpulse_client_roles import CLIENT_ID, ISSUER, roles_from_verified_access_token_claims
from .keycloak_tokens import verify_keycloak_access_token
from .oidc_session_store import issue_session, read_session, revoke_session

_BACKEND_INITIALIZATION_LOCK = threading.Lock()
SESSION_KEY = "electionpulse_keycloak_sid_v2"
_LEGACY_KEY = "electionpulse_keycloak_session_v1"
_PREFIX = "oidc:keycloak:"
_CALLBACK_PATH = "/auth/oidc/callback"


def _enabled() -> bool:
    return os.environ.get("ELECTIONPULSE_KEYCLOAK_OIDC_ENABLED", "").strip().lower() == "true"


def _required_configuration() -> tuple[str, str, str]:
    if not _enabled():
        abort(404)
    uri = os.environ.get("ELECTIONPULSE_KEYCLOAK_REDIRECT_URI", "").strip()
    parsed = urlsplit(uri)
    host = request.host.lower()
    is_local = host in {"localhost:5000", "127.0.0.1:5000"}
    correct_uri = (parsed.scheme in {"http", "https"} and parsed.netloc.lower() == host
                   and parsed.path == _CALLBACK_PATH and not parsed.query
                   and not parsed.fragment and not parsed.username and not parsed.password)
    # The captured production root redirects the apex host to www. Do not
    # accept apex callback, HTTP on a public host, or a spoofed Host header.
    # An unapproved Host is not an ElectionPulse OIDC endpoint. Fail closed
    # as not found; keep 503 for configuration mismatch on allowed hosts.
    if not is_local and host != "www.electionpulse.org":
        abort(404)
    if (not correct_uri or (is_local and parsed.scheme != "http") or
        (not is_local and parsed.scheme != "https")):
        abort(503)
    # Behind TLS termination, the deployment must verify trusted forwarding
    # semantics before activation; do not trust arbitrary X-Forwarded-* here.
    if not is_local and (current_app.config.get("SESSION_COOKIE_SECURE") is not True or
                         current_app.config.get("SESSION_COOKIE_HTTPONLY") is not True or
                         current_app.config.get("SESSION_COOKIE_SAMESITE") != "Lax"):
        abort(503)
    client_id = os.environ.get("ELECTIONPULSE_KEYCLOAK_CLIENT_ID", "").strip()
    client_secret = os.environ.get("ELECTIONPULSE_KEYCLOAK_CLIENT_SECRET", "")
    hmac_key = os.environ.get("ELECTIONPULSE_OIDC_SUBJECT_HMAC_KEY", "")
    if (client_id != CLIENT_ID or not client_secret or len(client_secret) < 16 or
        len(hmac_key.encode("utf-8")) < 32):
        abort(503)
    return uri, client_secret, hmac_key


def _session_backend():
    """Fail closed unless explicitly opted into a localhost, one-process dev store.

    This is not a production backend. Azure markers reject even a forged
    localhost Host header. A later production store requires separate design,
    security/load tests, database approval, and explicit activation.
    """
    if request.host.lower() not in {"localhost:5000", "127.0.0.1:5000"}:
        raise RuntimeError("oidc_dev_backend_nonlocal_host")
    if os.environ.get("ELECTIONPULSE_OIDC_SESSION_BACKEND", "") != "memory_dev":
        raise RuntimeError("oidc_production_backend_not_implemented")
    if os.environ.get("ELECTIONPULSE_OIDC_ALLOW_DEV_MEMORY_SESSIONS", "").lower() != "true":
        raise RuntimeError("oidc_dev_backend_not_explicitly_allowed")
    if any(os.environ.get(key) for key in (
        "WEBSITE_SITE_NAME", "WEBSITE_INSTANCE_ID", "WEBSITE_HOSTNAME",
        "CONTAINER_APP_NAME", "K_SERVICE",
    )):
        raise RuntimeError("oidc_dev_backend_forbidden_in_cloud")
    for key in ("DEPLOY_ENV", "ELECTIONPULSE_ENVIRONMENT", "ELECTIONPULSE_DEPLOY_ENV", "FLASK_ENV"):
        if os.environ.get(key, "").lower() in {"production", "prod", "staging", "azure"}:
            raise RuntimeError("oidc_dev_backend_forbidden_in_deployment")
    # Flask app-scoped (not global) and in-process only, with a lock guarding
    # first access. Each process gets a distinct store. No persistence.
    from .oidc_session_store import MemorySessionStore
    app = current_app._get_current_object()
    with _BACKEND_INITIALIZATION_LOCK:
        return app.extensions.setdefault("electionpulse_oidc_dev_memory", MemorySessionStore())


def _remote():
    """Authlib verifies ID token, OIDC nonce, PKCE, callback state and signature."""
    from authlib.integrations.flask_client import OAuth
    _, secret, _ = _required_configuration()
    oauth = OAuth(current_app._get_current_object())
    oauth.register(
        name="smart_elections_keycloak", client_id=CLIENT_ID, client_secret=secret,
        server_metadata_url=ISSUER + "/.well-known/openid-configuration",
        client_kwargs={"scope": "openid profile email", "code_challenge_method": "S256"},
    )
    return oauth.smart_elections_keycloak


def _safe_next(value: object) -> str:
    return value if isinstance(value, str) and value in {
        "/projects", "/ballot_lens", "/worklist", "/data_framework"
    } else "/projects"


def _validated_id_token(claims: object, now: int) -> tuple[str, int]:
    """Defense in depth; caller MUST use Authlib's verified UserInfo, never JSON."""
    if not isinstance(claims, dict):
        raise ValueError("missing_verified_id_token")
    subject = claims.get("sub")
    audiences = claims.get("aud")
    audiences = audiences if isinstance(audiences, list) else [audiences]
    exp, iat = claims.get("exp"), claims.get("iat")
    if (claims.get("iss") != ISSUER or not isinstance(subject, str) or
        not re.fullmatch(r"[\x21-\x7e]{1,256}", subject) or
        not audiences or any(not isinstance(v, str) for v in audiences) or
        CLIENT_ID not in audiences or
        (len(audiences) > 1 and claims.get("azp") != CLIENT_ID) or
        (claims.get("azp") is not None and claims["azp"] != CLIENT_ID) or
        type(exp) is not int or exp <= now or
        type(iat) is not int or iat > now + 30 or iat < now - 86400 or
        not isinstance(claims.get("nonce"), str) or not claims["nonce"]):
        raise ValueError("invalid_verified_id_token_claims")
    return subject, exp


def _clear_oidc_cookie():
    session.pop(SESSION_KEY, None)
    session.pop(_LEGACY_KEY, None)


def start_login():
    uri, _, _ = _required_configuration()
    try:
        store = _session_backend()
        old_sid = session.get(SESSION_KEY)
        if old_sid:
            revoke_session(store, old_sid)
    except Exception:
        abort(503)
    _clear_oidc_cookie()
    session["electionpulse_oidc_next"] = _safe_next(request.args.get("next"))
    try:
        return _remote().authorize_redirect(uri)
    except Exception:
        session.pop("electionpulse_oidc_next", None)
        abort(503)


def finish_login():
    _, _, hmac_key = _required_configuration()
    _clear_oidc_cookie()
    try:
        # Authlib owns OIDC state, nonce, code exchange and ID-token signature.
        # Independently verify the signed *access token* with pinned JWKS.
        token = _remote().authorize_access_token()
        if (not isinstance(token, dict) or not isinstance(token.get("id_token"), str)
            or not isinstance(token.get("access_token"), str)):
            raise ValueError("missing_oidc_tokens")
        now = int(time.time())
        subject, id_expiry = _validated_id_token(token.get("userinfo"), now)
        access = verify_keycloak_access_token(token["access_token"])
        if access["sub"] != subject:
            raise ValueError("id_access_subject_mismatch")
        decision = roles_from_verified_access_token_claims(access)
        if not decision.authenticated:
            raise ValueError("unverified_role_claims")
        opaque = hmac.new(hmac_key.encode("utf-8"), (ISSUER + "\0" + subject).encode("utf-8"),
                          hashlib.sha256).hexdigest()
        store = _session_backend()
        sid = issue_session(store, principal=_PREFIX + opaque,
                            project_access=decision.project_access,
                            token_expires_at=min(id_expiry, access["exp"]), now=now)
    except Exception:
        session.pop("electionpulse_oidc_next", None)
        abort(401)
    session[SESSION_KEY] = sid
    destination = _safe_next(session.pop("electionpulse_oidc_next", "/projects"))
    response = redirect(destination, code=303)
    response.headers["Cache-Control"] = "no-store"
    return response


def federated_session() -> dict | None:
    if not _enabled():
        return None
    sid = session.get(SESSION_KEY)
    if sid is None:
        return None
    try:
        _required_configuration()
        value = read_session(_session_backend(), sid)
    except Exception:
        # No authorized backend = no privilege, regardless of a browser cookie.
        # Anonymous approved-source parsing remains unaffected.
        return None
    if value is None:
        _clear_oidc_cookie()
    return value


def project_permission(principal: object, source: object) -> bool:
    value = federated_session()
    return bool(value and source == "keycloak_oidc" and principal == value["principal"]
                and value["project_access"] is True)


def status():
    if not _enabled():
        abort(404)
    value = federated_session()
    response = jsonify({"provider": "smart_elections_keycloak",
                        "authenticated": bool(value),
                        "project_access": bool(value and value["project_access"] is True),
                        "trusted_certificate": False,
                        "parser_upload_authorized": False})
    response.headers["Cache-Control"] = "no-store"
    return response


def logout():
    if not _enabled():
        abort(404)
    from webapp.parser.auth.workflow_csrf import assert_workflow_csrf_token, WorkflowCsrfError
    try:
        assert_workflow_csrf_token(request.headers.get("X-CSRFToken"))
    except WorkflowCsrfError:
        abort(403)
    sid = session.get(SESSION_KEY)
    _clear_oidc_cookie()
    session.pop("electionpulse_oidc_next", None)
    try:
        if sid:
            revoke_session(_session_backend(), sid)
    except Exception:
        # Backend unavailable: future access fails closed, but deletion is
        # unconfirmed. Report failure rather than claiming global logout.
        abort(503)
    response = jsonify({"local_session_closed": True,
                        "keycloak_sso_session_closed": False})
    response.headers["Cache-Control"] = "no-store"
    return response
