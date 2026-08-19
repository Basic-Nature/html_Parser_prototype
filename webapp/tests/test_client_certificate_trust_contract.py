from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from webapp.parser.auth.cert_trust import (
    configured_trusted_fingerprints,
    evaluate_client_certificate_trust,
    normalize_sha256_fingerprint,
)
from webapp.parser.utils import cert_utils
from webapp.parser.utils.privilege_tiers import (
    PrivilegeTier,
    get_principal_tier,
)


FP = "a" * 64
OTHER_FP = "b" * 64


def _valid_metadata():
    now = datetime.now(timezone.utc)
    return {
        "cn": "ElectionPulse Test Client",
        "issuer": "ElectionPulse Test CA",
        "issued_date": (now - timedelta(days=1)).isoformat(),
        "expiry_date": (now + timedelta(days=30)).isoformat(),
        "expiry_days": 30,
        "is_expired": False,
        "serial_number": "1234",
        "key_algorithm": "RSA",
    }


def test_fingerprint_normalization_is_exact_sha256():
    assert normalize_sha256_fingerprint(FP.upper()) == FP
    assert normalize_sha256_fingerprint(":".join([FP[i:i+2] for i in range(0, 64, 2)])) == FP
    assert normalize_sha256_fingerprint("abc123") is None
    assert normalize_sha256_fingerprint(None) is None


def test_azure_requires_explicit_enrollment():
    decision = evaluate_client_certificate_trust(
        FP,
        "X-ARR-ClientCert",
        _valid_metadata(),
        deploy_env="azure",
        environ={},
    )

    assert decision["required"] is True
    assert decision["trusted"] is False
    assert decision["reason"] == "no_trusted_fingerprints_configured"


def test_azure_accepts_exact_enrolled_fingerprint():
    decision = evaluate_client_certificate_trust(
        FP,
        "X-ARR-ClientCert",
        _valid_metadata(),
        deploy_env="azure",
        environ={"TRUSTED_CLIENT_CERT_FINGERPRINTS": FP},
    )

    assert decision["trusted"] is True
    assert decision["reason"] == "fingerprint_enrolled"


def test_azure_rejects_wrong_source_header():
    decision = evaluate_client_certificate_trust(
        FP,
        "SSL_CLIENT_CERT",
        _valid_metadata(),
        deploy_env="azure",
        environ={"TRUSTED_CLIENT_CERT_FINGERPRINTS": FP},
    )

    assert decision["trusted"] is False
    assert decision["reason"] == "unexpected_certificate_source"


def test_azure_rejects_unparsed_certificate_metadata():
    decision = evaluate_client_certificate_trust(
        FP,
        "X-ARR-ClientCert",
        None,
        deploy_env="azure",
        environ={"TRUSTED_CLIENT_CERT_FINGERPRINTS": FP},
    )

    assert decision["trusted"] is False
    assert decision["reason"] == "certificate_metadata_missing"


def test_azure_rejects_expired_certificate():
    metadata = _valid_metadata()
    metadata["is_expired"] = True

    decision = evaluate_client_certificate_trust(
        FP,
        "X-ARR-ClientCert",
        metadata,
        deploy_env="azure",
        environ={"TRUSTED_CLIENT_CERT_FINGERPRINTS": FP},
    )

    assert decision["trusted"] is False
    assert decision["reason"] == "certificate_expired"


def test_azure_rejects_not_yet_valid_certificate():
    now = datetime.now(timezone.utc)
    metadata = _valid_metadata()
    metadata["issued_date"] = (now + timedelta(days=1)).isoformat()
    metadata["expiry_date"] = (now + timedelta(days=30)).isoformat()

    decision = evaluate_client_certificate_trust(
        FP,
        "X-ARR-ClientCert",
        metadata,
        deploy_env="azure",
        environ={"TRUSTED_CLIENT_CERT_FINGERPRINTS": FP},
        now=now,
    )

    assert decision["trusted"] is False
    assert decision["reason"] == "certificate_not_yet_valid"


def test_nonproduction_preserves_legacy_certificate_parsing_behavior():
    decision = evaluate_client_certificate_trust(
        FP,
        "SSL_CLIENT_CERT",
        None,
        deploy_env="local",
        environ={},
    )

    assert decision["required"] is False
    assert decision["trusted"] is True
    assert decision["reason"] == "trust_not_required"


def test_tier_fingerprint_settings_are_also_trust_enrollment():
    trusted = configured_trusted_fingerprints(
        environ={
            "ADMIN_REVIEWER_CERT_FINGERPRINTS": FP,
            "ROOT_ADMIN_CERTS": OTHER_FP,
            "ADMIN_FULL_TRUST_CERTS": "legacy-cn-not-a-fingerprint",
        }
    )

    assert FP in trusted
    assert OTHER_FP in trusted
    assert len(trusted) == 2


def test_extract_client_principal_fails_closed_in_azure(monkeypatch):
    monkeypatch.setenv("DEPLOY_ENV", "azure")
    monkeypatch.delenv("TRUSTED_CLIENT_CERT_FINGERPRINTS", raising=False)
    monkeypatch.delenv("ROOT_ADMIN_CERT_FINGERPRINTS", raising=False)
    monkeypatch.delenv("ADMIN_FULL_TRUST_CERT_FINGERPRINTS", raising=False)
    monkeypatch.delenv("ADMIN_REVIEWER_CERT_FINGERPRINTS", raising=False)
    monkeypatch.delenv("ROOT_ADMIN_CERTS", raising=False)
    monkeypatch.delenv("ADMIN_FULL_TRUST_CERTS", raising=False)
    monkeypatch.delenv("ADMIN_REVIEWER_CERTS", raising=False)

    monkeypatch.setattr(
        cert_utils,
        "extract_client_cert_fingerprint",
        lambda _headers: (
            FP,
            "X-ARR-ClientCert",
            _valid_metadata(),
        ),
    )

    principal, source, metadata = cert_utils.extract_client_principal(
        {"X-ARR-ClientCert": "placeholder"}
    )

    assert principal is None
    assert source == "X-ARR-ClientCert"
    assert metadata["trust_required"] is True
    assert metadata["trust_valid"] is False
    assert metadata["trust_reason"] == "no_trusted_fingerprints_configured"


def test_extract_client_principal_promotes_only_enrolled_azure_cert(monkeypatch):
    monkeypatch.setenv("DEPLOY_ENV", "azure")
    monkeypatch.setenv("TRUSTED_CLIENT_CERT_FINGERPRINTS", FP)

    monkeypatch.setattr(
        cert_utils,
        "extract_client_cert_fingerprint",
        lambda _headers: (
            FP,
            "X-ARR-ClientCert",
            _valid_metadata(),
        ),
    )

    principal, source, metadata = cert_utils.extract_client_principal(
        {"X-ARR-ClientCert": "placeholder"}
    )

    assert principal == f"cert:{FP}"
    assert source == "X-ARR-ClientCert"
    assert metadata["trust_valid"] is True
    assert metadata["trust_reason"] == "fingerprint_enrolled"


def test_presented_untrusted_cert_does_not_fallback_to_sso(monkeypatch):
    monkeypatch.setenv("DEPLOY_ENV", "azure")
    monkeypatch.delenv("TRUSTED_CLIENT_CERT_FINGERPRINTS", raising=False)

    monkeypatch.setattr(
        cert_utils,
        "extract_client_cert_fingerprint",
        lambda _headers: (
            FP,
            "X-ARR-ClientCert",
            _valid_metadata(),
        ),
    )
    monkeypatch.setattr(
        cert_utils,
        "extract_sso_principal",
        lambda _headers: ("oid-should-not-win", "X-MS-CLIENT-PRINCIPAL-ID"),
    )

    principal, source, metadata = cert_utils.extract_client_principal({})

    assert principal is None
    assert source == "X-ARR-ClientCert"
    assert metadata["trust_valid"] is False


def test_sso_fallback_still_works_when_no_certificate(monkeypatch):
    monkeypatch.setattr(
        cert_utils,
        "extract_client_cert_fingerprint",
        lambda _headers: (None, None, None),
    )
    monkeypatch.setattr(
        cert_utils,
        "extract_sso_principal",
        lambda _headers: ("oid-123", "X-MS-CLIENT-PRINCIPAL-ID"),
    )

    principal, source, metadata = cert_utils.extract_client_principal({})

    assert principal == "sso:oid-123"
    assert source == "X-MS-CLIENT-PRINCIPAL-ID"
    assert metadata is None


def test_real_fingerprint_tier_matching_is_exact(monkeypatch):
    monkeypatch.setenv("ROOT_ADMIN_CERT_FINGERPRINTS", FP)
    monkeypatch.setenv("ADMIN_FULL_TRUST_CERT_FINGERPRINTS", "")
    monkeypatch.setenv("ADMIN_REVIEWER_CERT_FINGERPRINTS", "")

    assert get_principal_tier(
        f"cert:{FP}",
        "X-ARR-ClientCert",
    ) == PrivilegeTier.ROOT_ADMIN

    assert get_principal_tier(
        f"cert:{OTHER_FP}",
        "X-ARR-ClientCert",
    ) == PrivilegeTier.STANDARD_USER


def test_partial_fingerprint_cannot_elevate_real_cert(monkeypatch):
    monkeypatch.setenv("ROOT_ADMIN_CERTS", FP[:12])
    monkeypatch.delenv("ROOT_ADMIN_CERT_FINGERPRINTS", raising=False)

    assert get_principal_tier(
        f"cert:{FP}",
        "X-ARR-ClientCert",
    ) == PrivilegeTier.STANDARD_USER


def test_legacy_nonfingerprint_cert_principal_keeps_compatibility(monkeypatch):
    monkeypatch.setenv("ADMIN_REVIEWER_CERTS", "reviewer-example")

    assert get_principal_tier(
        "cert:CN=reviewer-example-client",
        "cert",
    ) == PrivilegeTier.ADMIN_REVIEWER
