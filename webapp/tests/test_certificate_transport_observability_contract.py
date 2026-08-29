"""Certificate transport observability contract.

Transport evidence is deliberately distinct from certificate authority:
certificate_transport.header_present answers whether X-ARR-ClientCert reached
this request, while certificate_present remains trusted current-request
cert-principal proof. The certificate body/header is never returned.
"""

from __future__ import annotations

import base64
import hashlib
from datetime import datetime, timedelta, timezone

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from webapp.parser.auth import status as authority_status
from webapp.parser.utils import cert_utils


EXPECTED_TRANSPORT_KEYS = {
    "header_present",
    "header_length",
    "decode_ok",
    "x509_parse_ok",
    "der_sha256",
    "parse_error",
}


class _IterationOnlyCaseVariantHeaders:
    """Headers whose get() cannot resolve the iterated case variant."""

    def __init__(self, name: str, value: str):
        self._name = name
        self._value = value

    def get(self, _name, default=None):
        return default

    def items(self):
        return [(self._name, self._value)]


def _build_der_certificate() -> bytes:
    key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )

    name = x509.Name(
        [
            x509.NameAttribute(
                NameOID.COMMON_NAME,
                "ElectionPulse Transport Contract",
            )
        ]
    )

    now = datetime.now(timezone.utc)

    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=1))
        .sign(key, hashes.SHA256())
    )

    return cert.public_bytes(serialization.Encoding.DER)


def test_transport_observation_absent_is_safe():
    observed = cert_utils.observe_client_certificate_transport({})

    assert set(observed) == EXPECTED_TRANSPORT_KEYS
    assert observed == {
        "header_present": False,
        "header_length": 0,
        "decode_ok": False,
        "x509_parse_ok": False,
        "der_sha256": None,
        "parse_error": None,
    }


def test_transport_observation_malformed_base64_is_safe():
    raw_value = "%%%not-a-certificate%%%"

    observed = cert_utils.observe_client_certificate_transport(
        {"X-ARR-ClientCert": raw_value}
    )

    assert set(observed) == EXPECTED_TRANSPORT_KEYS
    assert observed["header_present"] is True
    assert observed["header_length"] == len(raw_value)
    assert observed["decode_ok"] is False
    assert observed["x509_parse_ok"] is False
    assert observed["der_sha256"] is None
    assert observed["parse_error"] == "base64_decode_failed"
    assert raw_value not in repr(observed)


def test_transport_observation_valid_der_reports_only_safe_evidence():
    der = _build_der_certificate()
    encoded = base64.b64encode(der).decode("ascii")

    observed = cert_utils.observe_client_certificate_transport(
        {"x-arr-clientcert": encoded}
    )

    assert set(observed) == EXPECTED_TRANSPORT_KEYS
    assert observed["header_present"] is True
    assert observed["header_length"] == len(encoded)
    assert observed["decode_ok"] is True
    assert observed["x509_parse_ok"] is True
    assert observed["der_sha256"] == hashlib.sha256(der).hexdigest()
    assert observed["parse_error"] is None
    assert encoded not in repr(observed)

    metadata = cert_utils._extract_cert_metadata(der)

    assert "error" not in metadata
    assert metadata["issued_date"]
    assert metadata["expiry_date"]
    assert metadata["is_expired"] is False


def test_transport_and_authority_share_case_insensitive_azure_header_resolution(
    monkeypatch,
):
    der = _build_der_certificate()
    encoded = base64.b64encode(der).decode("ascii")
    expected_fp = hashlib.sha256(der).hexdigest()
    headers = _IterationOnlyCaseVariantHeaders(
        "X-ARR-CLIENTCERT",
        encoded,
    )

    observed = cert_utils.observe_client_certificate_transport(headers)
    assert observed["header_present"] is True
    assert observed["decode_ok"] is True
    assert observed["x509_parse_ok"] is True
    assert observed["der_sha256"] == expected_fp

    fingerprint, source, metadata = (
        cert_utils.extract_client_cert_fingerprint(headers)
    )
    assert fingerprint == expected_fp
    assert source == "X-ARR-ClientCert"
    assert isinstance(metadata, dict)
    assert "error" not in metadata

    monkeypatch.setenv("DEPLOY_ENV", "azure")
    monkeypatch.setenv(
        "TRUSTED_CLIENT_CERT_FINGERPRINTS",
        expected_fp,
    )

    principal, principal_source, principal_metadata = (
        cert_utils.extract_client_principal(headers)
    )
    assert principal == f"cert:{expected_fp}"
    assert principal_source == "X-ARR-ClientCert"
    assert principal_metadata["trust_required"] is True
    assert principal_metadata["trust_valid"] is True
    assert principal_metadata["trust_reason"] == "fingerprint_enrolled"


def test_status_sanitizer_exposes_trust_decision_but_not_parser_error():
    sanitized = authority_status._sanitize_cert_metadata_for_status(
        {
            "cn": "ElectionPulse Interactive Client",
            "issuer": "CN=ElectionPulse Interactive Client",
            "trust_required": True,
            "trust_valid": False,
            "trust_reason": "fingerprint_not_enrolled",
            "error": "DO NOT EXPOSE RAW PARSER ERROR",
            "subject_dn": "DO NOT EXPOSE EXTRA DN",
        }
    )

    assert sanitized["trust_required"] is True
    assert sanitized["trust_valid"] is False
    assert sanitized["trust_reason"] == "fingerprint_not_enrolled"
    assert "error" not in sanitized
    assert "subject_dn" not in sanitized


def test_status_endpoint_exposes_absent_transport_observation():
    from webapp import Smart_Elections_Parser_Webapp as appmod

    with appmod.app.test_client() as client:
        response = client.get(
            "/api/auth/status",
            headers={"Accept": "application/json"},
        )

    assert response.status_code == 200

    payload = response.get_json()
    assert isinstance(payload, dict)

    transport = payload["certificate_transport"]

    assert set(transport) == EXPECTED_TRANSPORT_KEYS
    assert transport["header_present"] is False
    assert transport["header_length"] == 0
    assert transport["decode_ok"] is False
    assert transport["x509_parse_ok"] is False
    assert transport["der_sha256"] is None
    assert transport["parse_error"] is None

    assert payload["certificate_present"] is False


def test_status_source_keeps_transport_and_authority_semantics_distinct():
    from pathlib import Path

    status_path = (
        Path(__file__).resolve().parents[1]
        / "parser"
        / "auth"
        / "status.py"
    )

    source = status_path.read_text(encoding="utf-8")

    assert "observe_client_certificate_transport" in source
    assert '"certificate_transport"' in source
    assert '"certificate_present"' in source
    # Transport evidence remains a separate diagnostic surface; status now
    # consumes the centralized provider-neutral authority classifier instead
    # of re-implementing certificate recognition inline.
    assert "authority = classify_authority(" in source
    assert '"authority": authority' in source


def test_cert_metadata_validity_members_are_properties_not_calls():
    from pathlib import Path

    cert_utils_path = (
        Path(__file__).resolve().parents[1]
        / "parser"
        / "utils"
        / "cert_utils.py"
    )

    source = cert_utils_path.read_text(encoding="utf-8")

    assert "cert.not_valid_after()" not in source
    assert "cert.not_valid_before()" not in source
    assert "not_valid_after_utc" in source
    assert "not_valid_before_utc" in source
