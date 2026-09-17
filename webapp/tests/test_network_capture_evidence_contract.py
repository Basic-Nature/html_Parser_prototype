from __future__ import annotations

import hashlib
import json
from pathlib import Path

from webapp.parser.contracts.artifact_identity import ArtifactIdentityHandoff
from webapp.parser.services.network_capture_evidence import (
    NETWORK_CAPTURE_FORMAT,
    NETWORK_CAPTURE_OBSERVATION_CONTRACT,
    NETWORK_CAPTURE_RAW_SEAL_CONTRACT,
    NETWORK_CAPTURE_SANITIZED_DERIVATIVE_CONTRACT,
    finalize_network_capture_evidence,
    project_network_capture_observation,
    sanitize_network_capture_records,
)


def _raw_records() -> list[dict[str, object]]:
    return [
        {
            "type": "request",
            "url": "https://example.test/api/results?token=SECRET_QUERY",
            "method": "post",
            "headers": {
                "Authorization": "Bearer SECRET_AUTH",
                "Cookie": "session=SECRET_COOKIE",
            },
            "post_data": '{"password":"SECRET_POST"}',
            "timestamp": 123456,
        },
        {
            "type": "response",
            "url": "https://example.test/api/results?token=SECRET_QUERY",
            "status": 200,
            "headers": {"Set-Cookie": "session=SECRET_SET_COOKIE"},
            "body": "SECRET_BODY",
            "timestamp": 123457,
        },
    ]


def _assert_no_sensitive_material(raw: bytes) -> None:
    for token in (
        b"SECRET_QUERY",
        b"SECRET_AUTH",
        b"SECRET_COOKIE",
        b"SECRET_POST",
        b"SECRET_SET_COOKIE",
        b"SECRET_BODY",
    ):
        assert token not in raw


def test_sanitized_derivative_is_allow_list_only():
    derivative = sanitize_network_capture_records(_raw_records())
    assert derivative["contract"] == NETWORK_CAPTURE_SANITIZED_DERIVATIVE_CONTRACT
    assert derivative["capture_format"] == NETWORK_CAPTURE_FORMAT
    assert derivative["raw_sensitive_fields_included"] is False
    assert derivative["automatic_timestamp"] is False
    encoded = json.dumps(derivative, sort_keys=True).encode()
    _assert_no_sensitive_material(encoded)
    decoded = encoded.decode()
    for forbidden_key in (
        '"url"',
        '"headers"',
        '"cookies"',
        '"query"',
        '"post_data"',
        '"body"',
        '"timestamp"',
    ):
        assert forbidden_key not in decoded


def test_finalize_preserves_raw_and_writes_seal_and_derivative(tmp_path):
    raw_path = tmp_path / "network_capture.json"
    raw_bytes = (json.dumps(_raw_records(), indent=2) + "\n").encode()
    raw_path.write_bytes(raw_bytes)

    artifacts = finalize_network_capture_evidence(raw_path)

    assert raw_path.read_bytes() == raw_bytes
    assert artifacts.raw_original_sha256 == hashlib.sha256(raw_bytes).hexdigest()
    assert artifacts.raw_original_bytes == len(raw_bytes)
    assert artifacts.observation_emitted is False

    derivative_raw = artifacts.sanitized_derivative_path.read_bytes()
    _assert_no_sensitive_material(derivative_raw)
    derivative = json.loads(derivative_raw)
    assert derivative["contract"] == NETWORK_CAPTURE_SANITIZED_DERIVATIVE_CONTRACT

    seal_raw = artifacts.seal_path.read_bytes()
    _assert_no_sensitive_material(seal_raw)
    seal = json.loads(seal_raw)
    assert seal["contract"] == NETWORK_CAPTURE_RAW_SEAL_CONTRACT
    assert seal["raw_original_sha256"] == hashlib.sha256(raw_bytes).hexdigest()
    assert seal["raw_original_bytes"] == len(raw_bytes)
    assert seal["raw_original_mutated"] is False
    assert seal["automatic_timestamp"] is False
    assert seal["sanitized_derivative_sha256"] == hashlib.sha256(
        derivative_raw
    ).hexdigest()


def test_explicit_callback_receives_noncanonical_summary_only(tmp_path):
    raw_path = tmp_path / "network_capture.json"
    raw_bytes = json.dumps(_raw_records()).encode()
    raw_path.write_bytes(raw_bytes)

    seen: list[dict[str, object]] = []
    artifacts = finalize_network_capture_evidence(
        raw_path,
        observation_emit_func=seen.append,
    )
    assert artifacts.observation_emitted is True
    assert len(seen) == 1
    payload = seen[0]
    assert payload["contract"] == NETWORK_CAPTURE_OBSERVATION_CONTRACT
    assert payload["authority"]["canonical"] is False
    assert payload["automatic_timestamp"] is False
    assert payload["record_count"] == 2
    assert payload["request_count"] == 1
    assert payload["response_count"] == 1
    assert payload["request_method_counts"] == {"POST": 1}
    assert payload["response_status_counts"] == {"200": 1}
    assert payload["raw_network_capture_included"] is False
    assert payload["raw_urls_included"] is False
    assert payload["raw_headers_included"] is False
    assert payload["raw_cookies_included"] is False
    assert payload["raw_query_included"] is False
    assert payload["raw_post_data_included"] is False
    assert payload["raw_response_body_included"] is False
    assert payload["artifact_identity"]["document_sha256"] == hashlib.sha256(
        raw_bytes
    ).hexdigest()
    _assert_no_sensitive_material(json.dumps(payload, sort_keys=True).encode())


def test_missing_identity_remains_unknown():
    payload = project_network_capture_observation(
        artifact_identity=None,
        sanitized_derivative_sha256="a" * 64,
        record_count=0,
        request_count=0,
        response_count=0,
        response_status_counts={},
        request_method_counts={},
    )
    assert "artifact_identity" not in payload
    assert payload["automatic_timestamp"] is False


def test_invalid_identity_fails_closed():
    try:
        project_network_capture_observation(
            artifact_identity="b" * 64,  # type: ignore[arg-type]
            sanitized_derivative_sha256="a" * 64,
            record_count=0,
            request_count=0,
            response_count=0,
            response_status_counts={},
            request_method_counts={},
        )
    except TypeError as exc:
        assert "ArtifactIdentityHandoff" in str(exc)
    else:
        raise AssertionError("invalid identity must fail closed")


def test_existing_artifact_identity_contract_is_used():
    identity = ArtifactIdentityHandoff(document_sha256="c" * 64)
    payload = project_network_capture_observation(
        artifact_identity=identity,
        sanitized_derivative_sha256="d" * 64,
        record_count=0,
        request_count=0,
        response_count=0,
        response_status_counts={},
        request_method_counts={},
    )
    assert payload["artifact_identity"]["document_sha256"] == "c" * 64
    assert payload["artifact_identity"]["algorithm"] == "sha256"
    assert payload["artifact_identity"]["semantics"] == "SHA256_OF_IMMUTABLE_CONTENT_BYTES"


def test_producer_wiring_is_after_existing_raw_write():
    repo = Path(__file__).resolve().parents[2]
    for relative in (
        "tools/capture_har_smoke.py",
        "tools/capture_har_smoke_v2.py",
    ):
        text = (repo / relative).read_text(encoding="utf-8")
        assert "network_capture.json" in text
        assert "json.dump(records" in text
        assert "finalize_network_capture_evidence(out_path)" in text
        assert text.index("json.dump(records") < text.index(
            "finalize_network_capture_evidence(out_path)"
        )


def test_downstream_consumers_remain_raw_capture_readers_only():
    repo = Path(__file__).resolve().parents[2]
    for relative in (
        "tools/parse_api_capture.py",
        "tools/summarize_network_capture.py",
    ):
        text = (repo / relative).read_text(encoding="utf-8")
        assert "network_capture.json" in text
        assert "json.load(" in text
        assert "network_capture_evidence" not in text
