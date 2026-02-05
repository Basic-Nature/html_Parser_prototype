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
    """Parse X.509 certificate DER bytes and extract human-readable metadata.
    
    Returns dict with keys:
    - cn: Common Name from Subject
    - issuer: Issuer DN string
    - expiry_date: ISO 8601 datetime string
    - expiry_days: Days until expiration (negative if expired)
    - serial_number: Certificate serial number (hex)
    - key_algorithm: Public key algorithm name
    - subject_dn: Full Subject Distinguished Name
    - is_expired: Boolean indicating if cert has expired
    """
    if not CRYPTOGRAPHY_AVAILABLE:
        return {
            "cn": None,
            "issuer": None,
            "expiry_date": None,
            "expiry_days": None,
            "serial_number": None,
            "key_algorithm": None,
            "subject_dn": None,
            "is_expired": None,
            "error": "cryptography library not available",
        }
    
    try:
        cert = x509.load_der_x509_certificate(der_bytes, default_backend())
        
        # Extract Common Name
        cn = None
        try:
            cn_attr = cert.subject.get_attributes_for_oid(x509.oid.NameOID.COMMON_NAME)
            if cn_attr:
                cn = cn_attr[0].value
        except Exception:
            pass
        
        # Extract Subject DN
        subject_dn = None
        try:
            subject_dn = cert.subject.rfc4514_string()
        except Exception:
            pass
        
        # Extract Issuer DN
        issuer_dn = None
        try:
            issuer_dn = cert.issuer.rfc4514_string()
        except Exception:
            pass
        
        # Extract Expiry Date
        expiry_date = cert.not_valid_after()
        expiry_iso = expiry_date.isoformat()
        
        # Calculate days until expiry
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        delta = expiry_date - now
        expiry_days = delta.days
        
        # Extract Serial Number
        serial_number = f"{cert.serial_number:X}"
        
        # Extract Key Algorithm
        key_algorithm = cert.public_key().__class__.__name__
        
        # Is expired?
        is_expired = expiry_days < 0
        
        return {
            "cn": cn,
            "issuer": issuer_dn,
            "expiry_date": expiry_iso,
            "expiry_days": expiry_days,
            "serial_number": serial_number,
            "key_algorithm": key_algorithm,
            "subject_dn": subject_dn,
            "is_expired": is_expired,
        }
    except Exception as e:
        return {
            "cn": None,
            "issuer": None,
            "expiry_date": None,
            "expiry_days": None,
            "serial_number": None,
            "key_algorithm": None,
            "subject_dn": None,
            "is_expired": None,
            "error": str(e),
        }


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
    """Preferred principal: client cert fingerprint; fallback: SSO object ID.

    Returns (principal, source, cert_metadata), where principal is prefixed to denote source.
    cert_metadata is populated only for cert-based auth; None for SSO.
    """
    cert_fp, cert_src, cert_meta = extract_client_cert_fingerprint(headers)
    if cert_fp:
        return f"cert:{cert_fp}", cert_src, cert_meta
    oid, sso_src = extract_sso_principal(headers)
    if oid:
        return f"sso:{oid}", sso_src, None
    return None, None, None
