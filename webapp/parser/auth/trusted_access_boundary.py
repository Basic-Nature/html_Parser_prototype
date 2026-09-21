"""Dedicated mTLS trusted-access verification and handoff boundary.

Dormant unless TRUSTED_ACCESS_BOUNDARY_ENABLED=true on a separately
provisioned App Service. Fresh client certificate proof resolves against the
durable trusted identity authority in a read-only DB transaction. A short-lived
opaque handoff is browser-state-bound and redeemed once on the public host into
the existing certificate_session authority.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from urllib.parse import urlencode, urlparse

from cryptography.fernet import Fernet, InvalidToken
from flask import jsonify, redirect, request, session

from webapp.parser.auth.trusted_access import (
    consume_trusted_access_browser_state,
    sanitize_trusted_return_target,
)
from webapp.parser.utils.cert_utils import extract_client_cert_fingerprint

_BOUNDARY_ENABLED_ENV = "TRUSTED_ACCESS_BOUNDARY_ENABLED"
_PUBLIC_BASE_URL_ENV = "TRUSTED_ACCESS_PUBLIC_BASE_URL"
_HANDOFF_SECRET_ENV = "TRUSTED_ACCESS_HANDOFF_SECRET"
_HANDOFF_TTL_SECONDS_ENV = "TRUSTED_ACCESS_HANDOFF_TTL_SECONDS"
_COMPLETE_PATH = "/auth/certificate/complete"
_TRUE_VALUES = {"1", "true", "yes", "on"}
_DEFAULT_HANDOFF_TTL_SECONDS = 120

def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE_VALUES

def trusted_access_boundary_enabled() -> bool:
    return _truthy_env(_BOUNDARY_ENABLED_ENV)

def _bounded_positive_int(value: object, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(str(value or "").strip())
    except (TypeError, ValueError):
        return default
    return min(max(parsed, minimum), maximum)

def trusted_access_handoff_ttl_seconds() -> int:
    return _bounded_positive_int(
        os.environ.get(_HANDOFF_TTL_SECONDS_ENV),
        default=_DEFAULT_HANDOFF_TTL_SECONDS,
        minimum=30,
        maximum=300,
    )

def _normalized_https_root(name: str) -> str | None:
    raw = os.environ.get(name, "").strip()
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

def _public_base_url() -> str | None:
    return _normalized_https_root(_PUBLIC_BASE_URL_ENV)

def _handoff_secret() -> str | None:
    secret = os.environ.get(_HANDOFF_SECRET_ENV, "")
    return secret if len(secret.encode("utf-8")) >= 32 else None

def _handoff_fernet() -> Fernet | None:
    secret = _handoff_secret()
    if secret is None:
        return None
    key = base64.urlsafe_b64encode(hashlib.sha256(secret.encode("utf-8")).digest())
    return Fernet(key)

def _sha256_text(value: object) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()

def _request_is_public_origin() -> bool:
    base = _public_base_url()
    if base is None:
        return False
    expected = urlparse(base).hostname
    actual = (request.host or "").split(":", 1)[0].lower()
    return bool(expected and actual == expected.lower())

def _resolve_fresh_durable_certificate(headers):
    fingerprint, source_header, metadata = extract_client_cert_fingerprint(headers)
    if not fingerprint or str(source_header or "").lower() != "x-arr-clientcert":
        return None
    if isinstance(metadata, dict) and metadata.get("is_expired") is True:
        return None

    from sqlalchemy import text
    from webapp.parser.auth.trusted_identity_repository import TrustedIdentityRepository
    from webapp.parser.auth.trusted_principal_authority import resolve_enrolled_mtls_principal
    from webapp.parser.utils.db_utils import SessionLocal

    db_session = None
    try:
        db_session = SessionLocal()
        db_session.execute(text("SET TRANSACTION READ ONLY"))
        repository = TrustedIdentityRepository(db_session)
        decision = resolve_enrolled_mtls_principal(repository, fingerprint)
        db_session.rollback()
        return decision
    except Exception:
        if db_session is not None:
            try:
                db_session.rollback()
            except Exception:
                pass
        return None
    finally:
        if db_session is not None:
            try:
                db_session.close()
            except Exception:
                pass

def _decision_value(decision, name: str):
    if decision is None:
        return None
    if isinstance(decision, dict):
        return decision.get(name)
    return getattr(decision, name, None)

def _issue_handoff(principal: str, *, state: str, return_target: str) -> str | None:
    fernet = _handoff_fernet()
    if fernet is None:
        return None
    payload = {
        "v": 1,
        "purpose": "trusted_access_certificate_handoff",
        "principal": principal,
        "state_sha256": _sha256_text(state),
        "return_to": sanitize_trusted_return_target(return_target),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return fernet.encrypt(raw).decode("ascii")

def _consume_handoff(token: object) -> dict | None:
    fernet = _handoff_fernet()
    if fernet is None:
        return None
    raw = str(token or "").strip()
    if not raw:
        return None
    try:
        plaintext = fernet.decrypt(
            raw.encode("ascii"),
            ttl=trusted_access_handoff_ttl_seconds(),
        )
        payload = json.loads(plaintext.decode("utf-8"))
    except (InvalidToken, ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("v") != 1 or payload.get("purpose") != "trusted_access_certificate_handoff":
        return None
    return payload

def verify_trusted_certificate_access():
    if not trusted_access_boundary_enabled():
        return jsonify({"error": "trusted_access_boundary_disabled"}), 404
    public_base = _public_base_url()
    if public_base is None or _handoff_secret() is None:
        return jsonify({"error": "trusted_access_boundary_not_configured"}), 503

    state = str(request.args.get("state") or "").strip()
    if len(state) < 32:
        return jsonify({"error": "trusted_access_state_required"}), 400

    decision = _resolve_fresh_durable_certificate(request.headers)
    if not bool(_decision_value(decision, "resolved")):
        return jsonify({"error": "trusted_certificate_not_enrolled"}), 403
    if not bool(_decision_value(decision, "protected_operation_eligible")):
        return jsonify({"error": "trusted_principal_not_eligible"}), 403

    principal = str(_decision_value(decision, "compatibility_principal") or "").strip()
    if not principal.startswith("cert:") or len(principal) != 69:
        return jsonify({"error": "trusted_principal_invalid"}), 403

    return_target = sanitize_trusted_return_target(request.args.get("return_to"))
    token = _issue_handoff(principal, state=state, return_target=return_target)
    if token is None:
        return jsonify({"error": "trusted_access_handoff_unavailable"}), 503

    destination = f"{public_base}{_COMPLETE_PATH}?" + urlencode({"handoff": token, "state": state})
    response = redirect(destination, code=302)
    response.headers["Cache-Control"] = "no-store"
    return response

def complete_trusted_certificate_access():
    if trusted_access_boundary_enabled():
        return jsonify({"error": "trusted_access_complete_not_public"}), 404
    if not _request_is_public_origin():
        return jsonify({"error": "trusted_access_public_origin_mismatch"}), 403

    state = str(request.args.get("state") or "").strip()
    if not consume_trusted_access_browser_state(state):
        return jsonify({"error": "trusted_access_state_invalid"}), 400

    payload = _consume_handoff(request.args.get("handoff"))
    if payload is None:
        return jsonify({"error": "trusted_access_handoff_invalid"}), 400
    if payload.get("state_sha256") != _sha256_text(state):
        return jsonify({"error": "trusted_access_handoff_state_mismatch"}), 400

    principal = str(payload.get("principal") or "").strip()
    if not principal.startswith("cert:") or len(principal) != 69:
        return jsonify({"error": "trusted_access_handoff_principal_invalid"}), 400

    session["certificate_session_principal"] = principal
    session["certificate_session_established_at"] = int(time.time())

    response = redirect(
        sanitize_trusted_return_target(payload.get("return_to")),
        code=302,
    )
    response.headers["Cache-Control"] = "no-store"
    return response
