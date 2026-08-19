"""Client-certificate trust authority for ElectionPulse.

App Service forwards X-ARR-ClientCert but does not decide whether the presented
certificate is trusted. This module owns the fail-closed production trust
decision while keeping certificate parsing/fingerprinting in cert_utils.py.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hmac
import os
import re
from typing import Mapping


_PRODUCTION_ENVS = frozenset({"azure", "prod", "production"})
_AZURE_CERT_HEADER = "X-ARR-ClientCert"

_TRUST_FINGERPRINT_ENV_KEYS = (
    "TRUSTED_CLIENT_CERT_FINGERPRINTS",
    "ROOT_ADMIN_CERT_FINGERPRINTS",
    "ADMIN_FULL_TRUST_CERT_FINGERPRINTS",
    "ADMIN_REVIEWER_CERT_FINGERPRINTS",
)

# Compatibility names are accepted only when their entries are full SHA-256
# fingerprints. Legacy CN/substring values never become production trust pins.
_LEGACY_CERT_ENV_KEYS = (
    "ROOT_ADMIN_CERTS",
    "ADMIN_FULL_TRUST_CERTS",
    "ADMIN_REVIEWER_CERTS",
)


def normalize_sha256_fingerprint(value: object) -> str | None:
    """Return canonical lowercase SHA-256 hex, or None for non-fingerprints."""
    if value is None:
        return None

    try:
        text = str(value).strip().lower()
    except Exception:
        return None

    text = text.replace(":", "").replace(" ", "")

    if not re.fullmatch(r"[0-9a-f]{64}", text):
        return None

    return text


def _parse_env_values(
    name: str,
    *,
    environ: Mapping[str, str] | None = None,
) -> list[str]:
    env = os.environ if environ is None else environ
    raw = env.get(name, "")

    if not raw:
        return []

    values = []

    for item in re.split(r"[,;\n]+", str(raw)):
        normalized = item.strip()
        if normalized:
            values.append(normalized)

    return values


def configured_trusted_fingerprints(
    *,
    environ: Mapping[str, str] | None = None,
) -> frozenset[str]:
    """Return the exact SHA-256 fingerprints trusted for client-cert authority."""
    trusted: set[str] = set()

    for name in (*_TRUST_FINGERPRINT_ENV_KEYS, *_LEGACY_CERT_ENV_KEYS):
        for value in _parse_env_values(name, environ=environ):
            fingerprint = normalize_sha256_fingerprint(value)
            if fingerprint:
                trusted.add(fingerprint)

    return frozenset(trusted)


def client_certificate_trust_required(
    *,
    deploy_env: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Production/Azure requests require explicit certificate trust enrollment."""
    env = os.environ if environ is None else environ

    resolved = (
        deploy_env
        if deploy_env is not None
        else env.get("DEPLOY_ENV", "")
    )

    return str(resolved or "").strip().lower() in _PRODUCTION_ENVS


def _parse_cert_datetime(value: object) -> datetime | None:
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


def evaluate_client_certificate_trust(
    fingerprint: object,
    source_header: object,
    metadata: object,
    *,
    deploy_env: str | None = None,
    environ: Mapping[str, str] | None = None,
    now: datetime | None = None,
) -> dict[str, object]:
    """Evaluate whether a parsed client certificate may become authority.

    Production/Azure policy:
      * certificate must arrive through X-ARR-ClientCert;
      * identity must be a canonical SHA-256 fingerprint;
      * X.509 metadata parsing must succeed;
      * validity window must contain the current time;
      * fingerprint must be explicitly enrolled.

    Non-production environments preserve the historical parsing behavior so
    local development/tests do not require production enrollment.
    """
    required = client_certificate_trust_required(
        deploy_env=deploy_env,
        environ=environ,
    )

    normalized = normalize_sha256_fingerprint(fingerprint)

    if not required:
        return {
            "required": False,
            "trusted": True,
            "reason": "trust_not_required",
            "fingerprint": normalized,
        }

    source = str(source_header or "").strip()

    if source.lower() != _AZURE_CERT_HEADER.lower():
        return {
            "required": True,
            "trusted": False,
            "reason": "unexpected_certificate_source",
            "fingerprint": normalized,
        }

    if normalized is None:
        return {
            "required": True,
            "trusted": False,
            "reason": "invalid_sha256_fingerprint",
            "fingerprint": None,
        }

    if not isinstance(metadata, dict):
        return {
            "required": True,
            "trusted": False,
            "reason": "certificate_metadata_missing",
            "fingerprint": normalized,
        }

    if metadata.get("error"):
        return {
            "required": True,
            "trusted": False,
            "reason": "certificate_metadata_error",
            "fingerprint": normalized,
        }

    if metadata.get("is_expired") is True:
        return {
            "required": True,
            "trusted": False,
            "reason": "certificate_expired",
            "fingerprint": normalized,
        }

    issued_at = _parse_cert_datetime(metadata.get("issued_date"))
    expires_at = _parse_cert_datetime(metadata.get("expiry_date"))

    if issued_at is None:
        return {
            "required": True,
            "trusted": False,
            "reason": "certificate_issued_date_invalid",
            "fingerprint": normalized,
        }

    if expires_at is None:
        return {
            "required": True,
            "trusted": False,
            "reason": "certificate_expiry_date_invalid",
            "fingerprint": normalized,
        }

    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    else:
        current = current.astimezone(timezone.utc)

    if current < issued_at:
        return {
            "required": True,
            "trusted": False,
            "reason": "certificate_not_yet_valid",
            "fingerprint": normalized,
        }

    if current >= expires_at:
        return {
            "required": True,
            "trusted": False,
            "reason": "certificate_expired",
            "fingerprint": normalized,
        }

    trusted = configured_trusted_fingerprints(environ=environ)

    if not trusted:
        return {
            "required": True,
            "trusted": False,
            "reason": "no_trusted_fingerprints_configured",
            "fingerprint": normalized,
        }

    matched = any(
        hmac.compare_digest(normalized, candidate)
        for candidate in trusted
    )

    if not matched:
        return {
            "required": True,
            "trusted": False,
            "reason": "fingerprint_not_enrolled",
            "fingerprint": normalized,
        }

    return {
        "required": True,
        "trusted": True,
        "reason": "fingerprint_enrolled",
        "fingerprint": normalized,
    }
