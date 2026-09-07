"""Dormant public-to-mTLS trusted-access entry seam.

The public ElectionPulse host must not depend on global client-certificate
negotiation. When explicitly enabled later, this module redirects a deliberate
user action to a separately configured HTTPS mTLS boundary.

This module does not grant authority, validate a certificate, create a trusted
session, or mutate Azure configuration. It only owns the fail-closed navigation
boundary from the public application.
"""

from __future__ import annotations

import os
from urllib.parse import urlencode, urlparse, urlunparse

from flask import jsonify, redirect, request


_CERTIFICATE_AUTH_ENABLED_ENV = "CERTIFICATE_AUTH_ENABLED"
_TRUSTED_ACCESS_BASE_URL_ENV = "TRUSTED_ACCESS_BASE_URL"
_TRUSTED_ACCESS_VERIFY_PATH = "/auth/certificate/verify"
_TRUE_VALUES = {"1", "true", "yes", "on"}


def certificate_auth_feature_enabled() -> bool:
    return (
        os.environ.get(_CERTIFICATE_AUTH_ENABLED_ENV, "")
        .strip()
        .lower()
        in _TRUE_VALUES
    )


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
    return {
        "enabled": enabled,
        "configured": configured,
        "available": bool(enabled and configured),
    }


def sanitize_trusted_return_target(
    value: str | None,
    *,
    fallback: str = "/ballot_lens",
) -> str:
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
    }:
        return fallback

    return urlunparse(("", "", path, "", parsed.query, ""))


def build_trusted_access_entry_url(
    return_target: str | None,
) -> str | None:
    base_url = _normalized_trusted_access_base_url()
    if not certificate_auth_feature_enabled() or base_url is None:
        return None

    safe_return_target = sanitize_trusted_return_target(return_target)
    query = urlencode({"return_to": safe_return_target})
    return f"{base_url}{_TRUSTED_ACCESS_VERIFY_PATH}?{query}"


def begin_trusted_certificate_access():
    state = certificate_auth_public_state()

    if not state["enabled"]:
        return jsonify({
            "error": "certificate_auth_disabled",
            "certificate_auth_available": False,
        }), 404

    if not state["configured"]:
        return jsonify({
            "error": "trusted_access_not_configured",
            "certificate_auth_available": False,
        }), 503

    destination = build_trusted_access_entry_url(
        request.args.get("next"),
    )
    if not destination:
        return jsonify({
            "error": "trusted_access_unavailable",
            "certificate_auth_available": False,
        }), 503

    response = redirect(destination, code=302)
    response.headers["Cache-Control"] = "no-store"
    return response
