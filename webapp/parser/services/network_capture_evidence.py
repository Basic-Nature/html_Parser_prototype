# Trusted evidence handling for custom browser network captures.
#
# Existing capture tools own the raw network_capture.json bytes. This module is
# called only after that raw write succeeds. It never mutates the raw capture.
from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

from webapp.parser.contracts.artifact_identity import (
    ARTIFACT_IDENTITY_HANDOFF_CONTRACT,
    ArtifactIdentityHandoff,
)

NETWORK_CAPTURE_FORMAT = "CUSTOM_HAR_LIKE_NETWORK_JSON"
NETWORK_CAPTURE_RAW_SEAL_CONTRACT = "network_capture_raw_seal_v1"
NETWORK_CAPTURE_SANITIZED_DERIVATIVE_CONTRACT = "network_capture_sanitized_derivative_v1"
NETWORK_CAPTURE_OBSERVATION_CONTRACT = "network_capture_observation_v1"
NETWORK_CAPTURE_OBSERVATION_AUTHORITY = "NETWORK_CAPTURE_EVIDENCE_INSPECTION"

NetworkCaptureObservationEmitFunc = Callable[[dict[str, Any]], Any]
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class NetworkCaptureEvidenceArtifacts:
    raw_original_path: Path
    raw_original_sha256: str
    raw_original_bytes: int
    sanitized_derivative_path: Path
    sanitized_derivative_sha256: str
    seal_path: Path
    observation_emitted: bool


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require_sha256(label: str, value: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase 64-character SHA-256")
    return value


def _atomic_write_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp-network-capture-evidence")
    try:
        temp_path.write_bytes(raw)
        os.replace(temp_path, path)
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def _require_records(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        raise TypeError("network capture root must be a JSON list")
    checked: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise TypeError(f"network capture record {index} must be an object")
        checked.append(item)
    return checked


def _safe_record_type(value: Any) -> str:
    if value == "request":
        return "request"
    if value == "response":
        return "response"
    return "other"


def sanitize_network_capture_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    sanitized: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("every network capture record must be a mapping")
        record_type = _safe_record_type(record.get("type"))
        item: dict[str, Any] = {
            "type": record_type,
            "url_included": False,
            "headers_included": False,
            "cookies_included": False,
            "query_included": False,
            "post_data_included": False,
            "response_body_included": False,
            "timestamp_included": False,
        }
        if record_type == "request":
            method = record.get("method")
            if isinstance(method, str) and 0 < len(method) <= 32:
                item["method"] = method.upper()
        elif record_type == "response":
            status = record.get("status")
            if isinstance(status, int) and not isinstance(status, bool):
                item["status"] = status
        sanitized.append(item)

    return {
        "contract": NETWORK_CAPTURE_SANITIZED_DERIVATIVE_CONTRACT,
        "capture_format": NETWORK_CAPTURE_FORMAT,
        "authority": {
            "inspection": NETWORK_CAPTURE_OBSERVATION_AUTHORITY,
            "canonical": False,
        },
        "record_count": len(sanitized),
        "request_count": sum(1 for item in sanitized if item["type"] == "request"),
        "response_count": sum(1 for item in sanitized if item["type"] == "response"),
        "raw_sensitive_fields_included": False,
        "automatic_timestamp": False,
        "records": sanitized,
    }


def project_network_capture_observation(
    *,
    artifact_identity: ArtifactIdentityHandoff | None,
    sanitized_derivative_sha256: str,
    record_count: int,
    request_count: int,
    response_count: int,
    response_status_counts: Mapping[int, int],
    request_method_counts: Mapping[str, int],
) -> dict[str, Any]:
    derivative_sha256 = _require_sha256(
        "sanitized_derivative_sha256",
        sanitized_derivative_sha256,
    )
    for label, value in (
        ("record_count", record_count),
        ("request_count", request_count),
        ("response_count", response_count),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{label} must be a non-negative integer")

    payload: dict[str, Any] = {
        "contract": NETWORK_CAPTURE_OBSERVATION_CONTRACT,
        "capture_format": NETWORK_CAPTURE_FORMAT,
        "authority": {
            "inspection": NETWORK_CAPTURE_OBSERVATION_AUTHORITY,
            "canonical": False,
        },
        "record_count": record_count,
        "request_count": request_count,
        "response_count": response_count,
        "response_status_counts": {
            str(int(key)): int(value)
            for key, value in sorted(response_status_counts.items())
        },
        "request_method_counts": {
            str(key): int(value)
            for key, value in sorted(request_method_counts.items())
        },
        "sanitized_derivative_sha256": derivative_sha256,
        "raw_network_capture_included": False,
        "raw_urls_included": False,
        "raw_headers_included": False,
        "raw_cookies_included": False,
        "raw_query_included": False,
        "raw_post_data_included": False,
        "raw_response_body_included": False,
        "automatic_timestamp": False,
    }
    if artifact_identity is not None:
        if not isinstance(artifact_identity, ArtifactIdentityHandoff):
            raise TypeError("artifact_identity must be ArtifactIdentityHandoff or None")
        payload["artifact_identity"] = {
            "contract": ARTIFACT_IDENTITY_HANDOFF_CONTRACT,
            "algorithm": artifact_identity.algorithm,
            "semantics": artifact_identity.semantics,
            "document_sha256": artifact_identity.document_sha256,
        }
    return payload


def emit_network_capture_observation_if_requested(
    payload: dict[str, Any],
    *,
    observation_emit_func: NetworkCaptureObservationEmitFunc | None = None,
) -> bool:
    if observation_emit_func is None:
        return False
    if not callable(observation_emit_func):
        raise TypeError("observation_emit_func must be callable")
    observation_emit_func(payload)
    return True


def finalize_network_capture_evidence(
    raw_capture_path: str | os.PathLike[str],
    *,
    observation_emit_func: NetworkCaptureObservationEmitFunc | None = None,
) -> NetworkCaptureEvidenceArtifacts:
    raw_path = Path(raw_capture_path)
    raw_bytes = raw_path.read_bytes()
    raw_sha256 = _sha256(raw_bytes)
    identity = ArtifactIdentityHandoff(document_sha256=raw_sha256)

    try:
        decoded = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("raw network capture must be UTF-8 JSON") from exc

    records = _require_records(decoded)
    sanitized = sanitize_network_capture_records(records)
    sanitized_bytes = _canonical_json_bytes(sanitized)
    sanitized_sha256 = _sha256(sanitized_bytes)

    sanitized_path = raw_path.with_name(raw_path.stem + ".sanitized.json")
    seal_path = raw_path.with_name(raw_path.stem + ".seal.json")
    if sanitized_path == raw_path or seal_path == raw_path:
        raise ValueError("evidence derivative paths must differ from raw capture")

    _atomic_write_bytes(sanitized_path, sanitized_bytes)

    request_methods: Counter[str] = Counter()
    response_statuses: Counter[int] = Counter()
    for record in records:
        if record.get("type") == "request":
            method = record.get("method")
            if isinstance(method, str) and method:
                request_methods[method.upper()] += 1
        elif record.get("type") == "response":
            status = record.get("status")
            if isinstance(status, int) and not isinstance(status, bool):
                response_statuses[status] += 1

    seal = {
        "contract": NETWORK_CAPTURE_RAW_SEAL_CONTRACT,
        "capture_format": NETWORK_CAPTURE_FORMAT,
        "algorithm": identity.algorithm,
        "semantics": identity.semantics,
        "raw_original_filename": raw_path.name,
        "raw_original_sha256": identity.document_sha256,
        "raw_original_bytes": len(raw_bytes),
        "sanitized_derivative_filename": sanitized_path.name,
        "sanitized_derivative_sha256": sanitized_sha256,
        "raw_original_mutated": False,
        "automatic_timestamp": False,
    }
    _atomic_write_bytes(seal_path, _canonical_json_bytes(seal))

    observation = project_network_capture_observation(
        artifact_identity=identity,
        sanitized_derivative_sha256=sanitized_sha256,
        record_count=len(records),
        request_count=sum(1 for r in records if r.get("type") == "request"),
        response_count=sum(1 for r in records if r.get("type") == "response"),
        response_status_counts=response_statuses,
        request_method_counts=request_methods,
    )
    emitted = emit_network_capture_observation_if_requested(
        observation,
        observation_emit_func=observation_emit_func,
    )
    return NetworkCaptureEvidenceArtifacts(
        raw_original_path=raw_path,
        raw_original_sha256=identity.document_sha256,
        raw_original_bytes=len(raw_bytes),
        sanitized_derivative_path=sanitized_path,
        sanitized_derivative_sha256=sanitized_sha256,
        seal_path=seal_path,
        observation_emitted=emitted,
    )
