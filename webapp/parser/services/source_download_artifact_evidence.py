"""Trusted observation contract for persisted source-download artifacts.

Transport-free and dormant unless an explicit observer callback is supplied.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Callable

CONTRACT = "source_download_artifact_observation_v1"
TRANSPORTS = frozenset({
    "requests_stream",
    "playwright_api_request",
    "raw_requests_temp_last_resort",
})
_SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")


class SourceDownloadArtifactObservationError(RuntimeError):
    pass


def _sha256_utf8(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise SourceDownloadArtifactObservationError(
            f"{field} must be a non-empty string."
        )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_utf8_optional(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise SourceDownloadArtifactObservationError(
            "final_response_url must be a string when present."
        )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _hash_persisted_payload(path_value) -> tuple[str, int]:
    path = Path(path_value)
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size


def _validate_precomputed(
    payload_sha256: str | None,
    payload_size: int | None,
) -> tuple[str, int] | None:
    if payload_sha256 is None:
        return None
    if not isinstance(payload_sha256, str) or not _SHA256_RE.fullmatch(payload_sha256):
        raise SourceDownloadArtifactObservationError(
            "precomputed payload SHA-256 is invalid."
        )
    if isinstance(payload_size, bool) or not isinstance(payload_size, int) or payload_size < 0:
        raise SourceDownloadArtifactObservationError(
            "precomputed payload size is invalid."
        )
    return payload_sha256, payload_size


def observe_source_download_artifact_if_requested(
    *,
    emit_func: Callable[[dict[str, object]], object] | None,
    persisted_path,
    requested_url: str,
    effective_request_url: str,
    final_response_url: str | None,
    transport: str,
    precomputed_payload_sha256: str | None = None,
    precomputed_payload_size: int | None = None,
) -> dict[str, object] | None:
    if emit_func is None:
        return None

    if not callable(emit_func):
        raise SourceDownloadArtifactObservationError(
            "source download artifact observer must be callable."
        )
    if transport not in TRANSPORTS:
        raise SourceDownloadArtifactObservationError(
            "unsupported source download transport."
        )

    precomputed = _validate_precomputed(
        precomputed_payload_sha256,
        precomputed_payload_size,
    )
    if precomputed is None:
        payload_sha256, payload_size = _hash_persisted_payload(persisted_path)
    else:
        payload_sha256, payload_size = precomputed

    core: dict[str, object] = {
        "contract": CONTRACT,
        "transport": transport,
        "payload_hash_algorithm": "sha256",
        "payload_sha256": payload_sha256,
        "payload_size": payload_size,
        "url_hash_algorithm": "sha256_utf8",
        "requested_url_sha256": _sha256_utf8(requested_url, field="requested_url"),
        "effective_request_url_sha256": _sha256_utf8(
            effective_request_url,
            field="effective_request_url",
        ),
        "final_response_url_sha256": _sha256_utf8_optional(final_response_url),
        "final_response_url_present": bool(final_response_url),
        "operational_manifest_trusted_authority": False,
    }
    canonical = json.dumps(
        core,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    observation: dict[str, object] = {
        **core,
        "observation_sha256": hashlib.sha256(canonical).hexdigest(),
    }

    try:
        emit_func(dict(observation))
    except Exception as exc:
        raise SourceDownloadArtifactObservationError(
            "source download artifact observation callback failed."
        ) from exc

    return observation
