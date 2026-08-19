from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional, Tuple

try:
    from cryptography import x509
    from cryptography.hazmat.backends import default_backend
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False

CERT_HEADER_CANDIDATES = [
    "X-ARR-ClientCert",  # Azure App Service / ARR
    "X-Client-Cert",     # Generic reverse proxy
    "SSL_CLIENT_CERT",   # Apache/nginx/IIS style
    "SSL_CLIENT_S_DN",   # DN string only
]

SSO_OID_HEADER = "X-MS-CLIENT-PRINCIPAL-ID"
SSO_PRINCIPAL_HEADER = "X-MS-CLIENT-PRINCIPAL"
SSO_OID_CLAIM = "http://schemas.microsoft.com/identity/claims/objectidentifier"


def _sha256_hex(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _decode_base64(value: str) -> Optional[bytes]:
    try:
        return base64.b64decode(value, validate=False)
    except Exception:
        return None


def _extract_cert_metadata(der_bytes: bytes) -> Dict[str, Any]:
    """Parse X.509 certificate DER bytes and extract safe metadata.

    Certificate validity values in cryptography are properties, not methods.
    Prefer the timezone-aware *_utc properties when available and fall back to
    legacy na?ve-UTC properties for older cryptography versions.
    """
    if not CRYPTOGRAPHY_AVAILABLE:
        return {
            "cn": None,
            "issuer": None,
            "issued_date": None,
            "expiry_date": None,
            "expiry_days": None,
            "serial_number": None,
            "key_algorithm": None,
            "subject_dn": None,
            "is_expired": None,
            "error": "cryptography library not available",
        }

    try:
        cert = x509.load_der_x509_certificate(
            der_bytes,
            default_backend(),
        )

        cn = None
        try:
            cn_attr = cert.subject.get_attributes_for_oid(
                x509.oid.NameOID.COMMON_NAME
            )
            if cn_attr:
                cn = cn_attr[0].value
        except Exception:
            pass

        subject_dn = None
        try:
            subject_dn = cert.subject.rfc4514_string()
        except Exception:
            pass

        issuer_dn = None
        try:
            issuer_dn = cert.issuer.rfc4514_string()
        except Exception:
            pass

        not_before = getattr(
            cert,
            "not_valid_before_utc",
            None,
        )

        if not_before is None:
            not_before = cert.not_valid_before

            if not_before.tzinfo is None:
                not_before = not_before.replace(
                    tzinfo=timezone.utc
                )
            else:
                not_before = not_before.astimezone(
                    timezone.utc
                )

        not_after = getattr(
            cert,
            "not_valid_after_utc",
            None,
        )

        if not_after is None:
            not_after = cert.not_valid_after

            if not_after.tzinfo is None:
                not_after = not_after.replace(
                    tzinfo=timezone.utc
                )
            else:
                not_after = not_after.astimezone(
                    timezone.utc
                )

        issued_iso = not_before.isoformat()
        expiry_iso = not_after.isoformat()

        now = datetime.now(timezone.utc)
        expiry_days = (not_after - now).days

        serial_number = f"{cert.serial_number:X}"
        key_algorithm = cert.public_key().__class__.__name__
        is_expired = now > not_after

        return {
            "cn": cn,
            "issuer": issuer_dn,
            "issued_date": issued_iso,
            "expiry_date": expiry_iso,
            "expiry_days": expiry_days,
            "serial_number": serial_number,
            "key_algorithm": key_algorithm,
            "subject_dn": subject_dn,
            "is_expired": is_expired,
        }

    except Exception as exc:
        return {
            "cn": None,
            "issuer": None,
            "issued_date": None,
            "expiry_date": None,
            "expiry_days": None,
            "serial_number": None,
            "key_algorithm": None,
            "subject_dn": None,
            "is_expired": None,
            "error": str(exc),
        }


def observe_client_certificate_transport(
    headers: Mapping[str, str],
) -> Dict[str, Any]:
    """Return safe request-transport evidence for Azure client certificates.

    This is observability only. It does not grant authority and does not expose
    the certificate header value. Production authority remains owned by
    extract_client_principal() + auth.cert_trust.
    """
    raw = None

    try:
        raw = headers.get("X-ARR-ClientCert")
    except Exception:
        raw = None

    if raw is None:
        try:
            raw = headers.get("x-arr-clientcert")
        except Exception:
            raw = None

    if raw is None:
        try:
            for key, candidate in headers.items():
                if str(key).lower() == "x-arr-clientcert":
                    raw = candidate
                    break
        except Exception:
            raw = None

    observation: Dict[str, Any] = {
        "header_present": False,
        "header_length": 0,
        "decode_ok": False,
        "x509_parse_ok": False,
        "der_sha256": None,
        "parse_error": None,
    }

    if raw is None:
        return observation

    try:
        value = str(raw).strip()
    except Exception:
        observation["header_present"] = True
        observation["parse_error"] = "header_value_unreadable"
        return observation

    observation["header_present"] = True
    observation["header_length"] = len(value)

    if not value:
        observation["parse_error"] = "empty_header"
        return observation

    decoded = _decode_base64(value)

    if not decoded:
        observation["parse_error"] = "base64_decode_failed"
        return observation

    observation["decode_ok"] = True
    observation["der_sha256"] = _sha256_hex(decoded)

    metadata = _extract_cert_metadata(decoded)

    if (
        not isinstance(metadata, dict)
        or bool(metadata.get("error"))
    ):
        observation["parse_error"] = "x509_parse_failed"
        return observation

    observation["x509_parse_ok"] = True
    return observation


def extract_client_cert_fingerprint(headers: Mapping[str, str]) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
    """Return (fingerprint, source_header, cert_metadata) if a client certificate header is present.

    - For PEM/DER values (ARR and many proxies), we base64-decode and hash the DER bytes.
    - For DN-only values (SSL_CLIENT_S_DN), we hash the raw string (no metadata available).
    - Metadata is extracted from DER bytes if cryptography library is available.
    """
    for header in CERT_HEADER_CANDIDATES:
        raw = headers.get(header) or headers.get(header.lower())
        if not raw:
            continue
        value = raw.strip()
        if not value:
            continue
        if header == "SSL_CLIENT_S_DN":
            fp = _sha256_hex(value.encode("utf-8", "replace"))
            return fp, header, None  # No metadata for DN-only header
        decoded = _decode_base64(value)
        if decoded:
            fp = _sha256_hex(decoded)
            metadata = _extract_cert_metadata(decoded)
            return fp, header, metadata
        # If not base64, hash the raw string (no metadata)
        fp = _sha256_hex(value.encode("utf-8", "replace"))
        return fp, header, None
    return None, None, None


def extract_sso_principal(headers: Mapping[str, str]) -> Tuple[Optional[str], Optional[str]]:
    """Return (oid, source) using Azure Easy Auth headers if available."""
    oid = headers.get(SSO_OID_HEADER) or headers.get(SSO_OID_HEADER.lower())
    if oid:
        return oid.strip() or None, SSO_OID_HEADER
    principal_blob = headers.get(SSO_PRINCIPAL_HEADER) or headers.get(SSO_PRINCIPAL_HEADER.lower())
    if not principal_blob:
        return None, None
    decoded = _decode_base64(principal_blob.strip())
    if not decoded:
        return None, None
    try:
        payload = json.loads(decoded.decode("utf-8", "replace"))
        if isinstance(payload, dict):
            claims = payload.get("claims") or {}
            if isinstance(claims, list):
                for claim in claims:
                    if not isinstance(claim, dict):
                        continue
                    if claim.get("typ") == SSO_OID_CLAIM and claim.get("val"):
                        return str(claim.get("val")), SSO_PRINCIPAL_HEADER
    except Exception:
        return None, None
    return None, None


def extract_client_principal(headers: Mapping[str, str]) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
    """Return the authoritative client principal for request headers.

    Client certificates are fingerprinted by this module, then pass through
    the canonical production trust decision before a ``cert:`` principal is
    created. SSO remains the fallback only when no certificate was presented.
    """
    cert_fp, cert_src, cert_meta = extract_client_cert_fingerprint(headers)

    if cert_fp:
        from webapp.parser.auth.cert_trust import (
            evaluate_client_certificate_trust,
        )

        decision = evaluate_client_certificate_trust(
            cert_fp,
            cert_src,
            cert_meta,
        )

        if isinstance(cert_meta, dict):
            cert_meta = dict(cert_meta)
        else:
            cert_meta = {}

        cert_meta["trust_required"] = bool(decision["required"])
        cert_meta["trust_valid"] = bool(decision["trusted"])
        cert_meta["trust_reason"] = str(decision["reason"])

        # A presented-but-untrusted production certificate is fail-closed.
        # Do not silently downgrade/fallback to SSO.
        if decision["required"] and not decision["trusted"]:
            return None, cert_src, cert_meta

        return f"cert:{cert_fp}", cert_src, cert_meta

    oid, sso_src = extract_sso_principal(headers)
    if oid:
        return f"sso:{oid}", sso_src, None

    return None, None, None
