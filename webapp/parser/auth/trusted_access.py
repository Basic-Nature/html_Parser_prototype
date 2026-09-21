"""Public-to-dedicated-mTLS trusted-access entry seam.

The public ElectionPulse host must not depend on global client-certificate
negotiation. When explicitly enabled, a deliberate user action creates a
browser-bound one-time state and redirects to a separately configured HTTPS
mTLS boundary.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
from urllib.parse import urlencode, urlparse, urlunparse

from flask import jsonify, redirect, request, session

_CERTIFICATE_AUTH_ENABLED_ENV = "CERTIFICATE_AUTH_ENABLED"
_TRUSTED_ACCESS_BASE_URL_ENV = "TRUSTED_ACCESS_BASE_URL"
_TRUSTED_ACCESS_VERIFY_PATH = "/auth/certificate/verify"
_TRUSTED_ACCESS_STATE_HASH_SESSION_KEY = "trusted_access_state_sha256"
_TRUSTED_ACCESS_STATE_ISSUED_AT_SESSION_KEY = "trusted_access_state_issued_at"
_TRUSTED_ACCESS_STATE_TTL_SECONDS_ENV = "TRUSTED_ACCESS_STATE_TTL_SECONDS"
_DEFAULT_STATE_TTL_SECONDS = 300
_TRUE_VALUES = {"1", "true", "yes", "on"}

def _sha256_text(value: object) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()

def _bounded_positive_int(value: object, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(str(value or "").strip())
    except (TypeError, ValueError):
        return default
    return min(max(parsed, minimum), maximum)

def trusted_access_state_ttl_seconds() -> int:
    return _bounded_positive_int(
        os.environ.get(_TRUSTED_ACCESS_STATE_TTL_SECONDS_ENV),
        default=_DEFAULT_STATE_TTL_SECONDS,
        minimum=30,
        maximum=900,
    )

def certificate_auth_feature_enabled() -> bool:
    return os.environ.get(_CERTIFICATE_AUTH_ENABLED_ENV, "").strip().lower() in _TRUE_VALUES

def _normalized_trusted_access_base_url() -> str | None:
    raw = os.environ.get(_TRUSTED_ACCESS_BASE_URL_ENV, "").strip()
    if not raw:
        return None
    parsed = urlparse(raw)
    if (
        parsed.scheme.lower() != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        return None
    return f"https://{parsed.netloc}".rstrip("/")

def certificate_auth_public_state() -> dict[str, bool]:
    enabled = certificate_auth_feature_enabled()
    configured = _normalized_trusted_access_base_url() is not None
    return {"enabled": enabled, "configured": configured, "available": bool(enabled and configured)}

def sanitize_trusted_return_target(value: str | None, *, fallback: str = "/ballot_lens") -> str:
    candidate = str(value or "").strip()
    if not candidate.startswith("/") or candidate.startswith("//"):
        candidate = fallback
    parsed = urlparse(candidate)
    if parsed.scheme or parsed.netloc:
        candidate = fallback
        parsed = urlparse(candidate)
    path = parsed.path or fallback
    if path in {
        "/auth/welcome",
        "/auth/challenge",
        "/auth/certificate/start",
        "/auth/certificate/verify",
        "/auth/certificate/complete",
    }:
        return fallback
    return urlunparse(("", "", path, "", parsed.query, ""))

def issue_trusted_access_browser_state() -> str:
    state = secrets.token_urlsafe(32)
    session[_TRUSTED_ACCESS_STATE_HASH_SESSION_KEY] = _sha256_text(state)
    session[_TRUSTED_ACCESS_STATE_ISSUED_AT_SESSION_KEY] = int(time.time())
    return state

def consume_trusted_access_browser_state(state: object, *, now: int | None = None) -> bool:
    expected_hash = session.pop(_TRUSTED_ACCESS_STATE_HASH_SESSION_KEY, None)
    issued_at = session.pop(_TRUSTED_ACCESS_STATE_ISSUED_AT_SESSION_KEY, None)
    candidate = str(state or "").strip()
    if (
        not candidate
        or not isinstance(expected_hash, str)
        or len(expected_hash) != 64
        or not isinstance(issued_at, (int, float))
    ):
        return False
    instant = int(time.time()) if now is None else int(now)
    if max(0, instant - int(issued_at)) > trusted_access_state_ttl_seconds():
        return False
    return hmac.compare_digest(expected_hash, _sha256_text(candidate))

def build_trusted_access_entry_url(return_target: str | None, *, state: str | None = None) -> str | None:
    base_url = _normalized_trusted_access_base_url()
    if not certificate_auth_feature_enabled() or base_url is None:
        return None
    values = {"return_to": sanitize_trusted_return_target(return_target)}
    if state:
        values["state"] = str(state)
    return f"{base_url}{_TRUSTED_ACCESS_VERIFY_PATH}?{urlencode(values)}"

def begin_trusted_certificate_access():
    state = certificate_auth_public_state()
    if not state["enabled"]:
        return jsonify({"error": "certificate_auth_disabled", "certificate_auth_available": False}), 404
    if not state["configured"]:
        return jsonify({"error": "trusted_access_not_configured", "certificate_auth_available": False}), 503

    browser_state = issue_trusted_access_browser_state()
    destination = build_trusted_access_entry_url(request.args.get("next"), state=browser_state)
    if not destination:
        consume_trusted_access_browser_state(browser_state)
        return jsonify({"error": "trusted_access_unavailable", "certificate_auth_available": False}), 503

    response = redirect(destination, code=302)
    response.headers["Cache-Control"] = "no-store"
    return response
