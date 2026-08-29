from __future__ import annotations

from dataclasses import replace
import pytest

from webapp.parser.auth.capability_policy import (
    Capability,
    CapabilityPolicyError,
    assert_ballot_lens_public_registry_parse,
)
from webapp.parser.services.public_ballot_lens_policy import (
    DEFAULT_PUBLIC_RUN_POLICY,
    PublicBallotLensPolicyError,
    PublicRunAdmissionController,
    PublicRunAdmissionError,
    authorize_public_registry_parse,
    configured_public_registry_pilot_source_id,
    derive_pseudonymous_client_rate_key,
    public_registry_parse_feature_enabled,
    public_registry_rate_hmac_secret,
    validate_public_start_payload,
)

VALID_SOURCE_ID = "blsrc_v1_" + ("a" * 64)
SECRET = b"s" * 32


def test_ballot_lens_public_registry_parse_is_distinct_capability():
    assert Capability.BALLOT_LENS_PUBLIC_REGISTRY_PARSE.value == (
        "ballot_lens_public_registry_parse"
    )
    assert Capability.BALLOT_LENS_PUBLIC_REGISTRY_PARSE is not Capability.PUBLIC_READ


def test_capability_fails_closed_until_all_public_authority_is_proven():
    with pytest.raises(CapabilityPolicyError):
        assert_ballot_lens_public_registry_parse(
            feature_enabled=False,
            payload_validated=True,
            registry_source_resolved=True,
        )
    with pytest.raises(CapabilityPolicyError):
        assert_ballot_lens_public_registry_parse(
            feature_enabled=True,
            payload_validated=False,
            registry_source_resolved=True,
        )
    with pytest.raises(CapabilityPolicyError):
        assert_ballot_lens_public_registry_parse(
            feature_enabled=True,
            payload_validated=True,
            registry_source_resolved=False,
        )
    assert assert_ballot_lens_public_registry_parse(
        feature_enabled=True,
        payload_validated=True,
        registry_source_resolved=True,
    ) is Capability.BALLOT_LENS_PUBLIC_REGISTRY_PARSE


def test_public_feature_defaults_disabled():
    assert public_registry_parse_feature_enabled({}) is False
    assert public_registry_parse_feature_enabled({
        "BALLOT_LENS_PUBLIC_REGISTRY_PARSE_ENABLED": "false"
    }) is False
    assert public_registry_parse_feature_enabled({
        "BALLOT_LENS_PUBLIC_REGISTRY_PARSE_ENABLED": "true"
    }) is True


def test_public_start_payload_accepts_only_registry_source_id():
    assert validate_public_start_payload({
        "registry_source_id": VALID_SOURCE_ID
    }) == VALID_SOURCE_ID
    forbidden = (
        {},
        {"registry_source_id": VALID_SOURCE_ID, "session_id": "sess_attacker_selected"},
        {"registry_source_id": VALID_SOURCE_ID, "direct_urls": ["https://example.gov/results"]},
        {"registry_source_id": VALID_SOURCE_ID, "manual_upload_path": "input.csv"},
        {"registry_source_id": VALID_SOURCE_ID, "warehouse_override_url": "https://example.gov/results"},
        {"registry_source_id": VALID_SOURCE_ID, "unknown_key": "x"},
        {"registry_source_id": "https://example.gov/results"},
    )
    for payload in forbidden:
        with pytest.raises(PublicBallotLensPolicyError):
            validate_public_start_payload(payload)


def test_public_start_payload_has_hard_byte_cap():
    tiny = replace(DEFAULT_PUBLIC_RUN_POLICY, start_payload_max_bytes=32)
    with pytest.raises(PublicBallotLensPolicyError):
        validate_public_start_payload(
            {"registry_source_id": VALID_SOURCE_ID},
            policy=tiny,
        )


def test_authorization_remains_disabled_by_default():
    payload = {"registry_source_id": VALID_SOURCE_ID}
    with pytest.raises(CapabilityPolicyError):
        authorize_public_registry_parse(
            payload,
            registry_source_resolved=True,
            environ={},
        )
    with pytest.raises(PublicBallotLensPolicyError):
        authorize_public_registry_parse(
            payload,
            registry_source_resolved=True,
            environ={
                "BALLOT_LENS_PUBLIC_REGISTRY_PARSE_ENABLED": "true",
            },
        )

    with pytest.raises(PublicBallotLensPolicyError):
        authorize_public_registry_parse(
            payload,
            registry_source_resolved=True,
            environ={
                "BALLOT_LENS_PUBLIC_REGISTRY_PARSE_ENABLED": "true",
                "BALLOT_LENS_PUBLIC_REGISTRY_PILOT_SOURCE_ID":
                    "blsrc_v1_" + ("b" * 64),
            },
        )

    enabled = {
        "BALLOT_LENS_PUBLIC_REGISTRY_PARSE_ENABLED": "true",
        "BALLOT_LENS_PUBLIC_REGISTRY_PILOT_SOURCE_ID":
            VALID_SOURCE_ID,
    }
    capability, source_id = authorize_public_registry_parse(
        payload,
        registry_source_resolved=True,
        environ=enabled,
    )
    assert capability is Capability.BALLOT_LENS_PUBLIC_REGISTRY_PARSE
    assert source_id == VALID_SOURCE_ID
    assert configured_public_registry_pilot_source_id(enabled) == VALID_SOURCE_ID


def test_client_rate_key_is_pseudonymous_and_secret_bound():
    key_a = derive_pseudonymous_client_rate_key(
        "203.0.113.10",
        secret=SECRET,
    )
    key_b = derive_pseudonymous_client_rate_key(
        "203.0.113.10",
        secret=b"t" * 32,
    )
    assert key_a.startswith("client:")
    assert "203.0.113.10" not in key_a
    assert key_a != key_b
    with pytest.raises(PublicRunAdmissionError):
        derive_pseudonymous_client_rate_key(
            "203.0.113.10",
            secret=b"short",
        )


def test_process_local_admission_enforces_global_and_session_limits():
    policy = replace(
        DEFAULT_PUBLIC_RUN_POLICY,
        global_concurrent_runs=1,
        session_rate_max_runs=2,
        session_rate_window_seconds=600,
        client_rate_max_runs=6,
        client_rate_window_seconds=3600,
    )
    controller = PublicRunAdmissionController(policy)
    client_key = derive_pseudonymous_client_rate_key(
        "203.0.113.10",
        secret=SECRET,
    )
    session_id = "sess_server_generated_12345"
    lease = controller.acquire(
        client_key=client_key,
        server_session_id=session_id,
        now=1000.0,
    )
    assert controller.active_count() == 1
    with pytest.raises(PublicRunAdmissionError):
        controller.acquire(
            client_key=client_key,
            server_session_id="sess_server_generated_other",
            now=1001.0,
        )
    lease.release()
    second = controller.acquire(
        client_key=client_key,
        server_session_id=session_id,
        now=1002.0,
    )
    second.release()
    with pytest.raises(PublicRunAdmissionError):
        controller.acquire(
            client_key=client_key,
            server_session_id=session_id,
            now=1003.0,
        )
    after_window = controller.acquire(
        client_key=client_key,
        server_session_id=session_id,
        now=1601.0,
    )
    after_window.release()


def test_admission_rejects_raw_client_or_non_server_session_keys():
    controller = PublicRunAdmissionController()
    with pytest.raises(PublicRunAdmissionError):
        controller.acquire(
            client_key="203.0.113.10",
            server_session_id="sess_server_generated_12345",
            now=1000.0,
        )
    client_key = derive_pseudonymous_client_rate_key(
        "203.0.113.10",
        secret=SECRET,
    )
    with pytest.raises(PublicRunAdmissionError):
        controller.acquire(
            client_key=client_key,
            server_session_id="caller-selected",
            now=1000.0,
        )

def test_public_rate_hmac_secret_requires_explicit_32_byte_value():
    with pytest.raises(PublicRunAdmissionError):
        public_registry_rate_hmac_secret({})
    with pytest.raises(PublicRunAdmissionError):
        public_registry_rate_hmac_secret({
            "BALLOT_LENS_PUBLIC_RATE_HMAC_SECRET": "short",
        })
    assert public_registry_rate_hmac_secret({
        "BALLOT_LENS_PUBLIC_RATE_HMAC_SECRET": "x" * 32,
    }) == b"x" * 32
