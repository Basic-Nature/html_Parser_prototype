from __future__ import annotations

import base64
import hashlib
import json
from typing import Mapping, Optional, Tuple

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


def extract_client_cert_fingerprint(headers: Mapping[str, str]) -> Tuple[Optional[str], Optional[str]]:
    """Return (fingerprint, source_header) if a client certificate header is present.

    - For PEM/DER values (ARR and many proxies), we base64-decode and hash the DER bytes.
    - For DN-only values (SSL_CLIENT_S_DN), we hash the raw string.
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
            return fp, header
        decoded = _decode_base64(value)
        if decoded:
            return _sha256_hex(decoded), header
        # If not base64, hash the raw string
        return _sha256_hex(value.encode("utf-8", "replace")), header
    return None, None


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


def extract_client_principal(headers: Mapping[str, str]) -> Tuple[Optional[str], Optional[str]]:
    """Preferred principal: client cert fingerprint; fallback: SSO object ID.

    Returns (principal, source), where principal is prefixed to denote source.
    """
    cert_fp, cert_src = extract_client_cert_fingerprint(headers)
    if cert_fp:
        return f"cert:{cert_fp}", cert_src
    oid, sso_src = extract_sso_principal(headers)
    if oid:
        return f"sso:{oid}", sso_src
    return None, None
