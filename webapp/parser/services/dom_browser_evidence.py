"""Trusted DOM-browser evidence boundary.

Persists the exact UTF-8 bytes of an explicitly requested browser DOM
serialization, writes deterministic safe metadata/seal sidecars, and may emit
a noncanonical summary through an explicit same-run callback.

This service does not claim that Playwright ``page.content()`` bytes are the
origin HTTP response body.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from ..contracts.artifact_identity import ArtifactIdentityHandoff


DOM_BROWSER_METADATA_CONTRACT = "dom_browser_metadata_v1"
DOM_BROWSER_SEAL_CONTRACT = "dom_browser_seal_v1"
DOM_BROWSER_OBSERVATION_CONTRACT = "dom_browser_observation_v1"
DOM_BROWSER_OBSERVATION_AUTHORITY = "NONCANONICAL_OBSERVATION"
DOM_BROWSER_MEDIA_TYPE = "text/html; charset=utf-8"
DOM_BROWSER_SERIALIZATION_SEMANTICS = (
    "UTF8_OF_EXACT_BROWSER_SERIALIZED_DOM_RETURNED_BY_CAPTURE_CALL;"
    "NOT_ORIGIN_HTTP_RESPONSE_BYTES"
)


def _canonical_json_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _identity_payload(
    artifact_identity: ArtifactIdentityHandoff | None,
    *,
    sealed_sha256: str,
) -> dict[str, str] | None:
    if artifact_identity is None:
        return None
    if artifact_identity.document_sha256 != sealed_sha256:
        raise ValueError(
            "artifact_identity must match exact immutable DOM serialization bytes"
        )
    return {
        "document_sha256": artifact_identity.document_sha256,
        "algorithm": artifact_identity.algorithm,
        "semantics": artifact_identity.semantics,
    }


def _exclusive_write_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)


def _require_fresh_targets(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            raise FileExistsError(
                f"DOM evidence target already exists; refusing overwrite: {path.name}"
            )


def project_dom_browser_observation(
    *,
    artifact_sha256: str,
    byte_count: int,
    capture_role: str,
    derivative_sha256: str,
    artifact_identity: ArtifactIdentityHandoff | None = None,
) -> dict[str, Any]:
    return {
        "contract": DOM_BROWSER_OBSERVATION_CONTRACT,
        "authority": DOM_BROWSER_OBSERVATION_AUTHORITY,
        "canonical": False,
        "artifact_identity": _identity_payload(
            artifact_identity,
            sealed_sha256=artifact_sha256,
        ),
        "artifact_sha256": artifact_sha256,
        "byte_count": int(byte_count),
        "media_type": DOM_BROWSER_MEDIA_TYPE,
        "capture_role": str(capture_role),
        "serialization_semantics": DOM_BROWSER_SERIALIZATION_SEMANTICS,
        "derivative_sha256": derivative_sha256,
        "raw_dom_bytes_included": False,
        "raw_dom_text_included": False,
        "raw_dom_path_included": False,
        "dom_content_summary_included": False,
        "automatic_timestamp": False,
    }


def emit_dom_browser_observation_if_requested(
    *,
    observation_emit_func: Callable[[dict[str, Any]], Any] | None,
    payload: dict[str, Any],
) -> bool:
    if observation_emit_func is None:
        return False
    if not callable(observation_emit_func):
        raise TypeError("observation_emit_func must be callable or None")
    observation_emit_func(payload)
    return True


def finalize_dom_browser_evidence(
    html_content: str,
    evidence_path: str | Path,
    *,
    capture_role: str,
    artifact_identity: ArtifactIdentityHandoff | None = None,
    observation_emit_func: Callable[[dict[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(html_content, str):
        raise TypeError("html_content must be str")

    raw = html_content.encode("utf-8")
    artifact_sha256 = _sha256(raw)
    identity = _identity_payload(
        artifact_identity,
        sealed_sha256=artifact_sha256,
    )

    path = Path(evidence_path)
    metadata_path = path.with_suffix(path.suffix + ".metadata.json")
    seal_path = path.with_suffix(path.suffix + ".seal.json")
    _require_fresh_targets(path, metadata_path, seal_path)

    metadata = {
        "contract": DOM_BROWSER_METADATA_CONTRACT,
        "artifact_sha256": artifact_sha256,
        "byte_count": len(raw),
        "media_type": DOM_BROWSER_MEDIA_TYPE,
        "capture_role": str(capture_role),
        "serialization_semantics": DOM_BROWSER_SERIALIZATION_SEMANTICS,
        "sealed_original_persisted": True,
        "raw_original_mutated": False,
        "raw_dom_payload_included": False,
        "filesystem_path_included": False,
        "dom_content_summary_included": False,
        "automatic_timestamp": False,
    }
    metadata_bytes = _canonical_json_bytes(metadata)
    derivative_sha256 = _sha256(metadata_bytes)

    seal = {
        "contract": DOM_BROWSER_SEAL_CONTRACT,
        "algorithm": "sha256",
        "semantics": "SHA256_OF_IMMUTABLE_CONTENT_BYTES",
        "artifact_sha256": artifact_sha256,
        "artifact_identity": identity,
        "byte_count": len(raw),
        "media_type": DOM_BROWSER_MEDIA_TYPE,
        "capture_role": str(capture_role),
        "serialization_semantics": DOM_BROWSER_SERIALIZATION_SEMANTICS,
        "metadata_derivative_sha256": derivative_sha256,
        "sealed_original_persisted": True,
        "raw_original_mutated": False,
        "raw_dom_payload_included": False,
        "filesystem_path_included": False,
        "dom_content_summary_included": False,
        "automatic_timestamp": False,
    }

    _exclusive_write_bytes(path, raw)
    _exclusive_write_bytes(metadata_path, metadata_bytes)
    _exclusive_write_bytes(seal_path, _canonical_json_bytes(seal))

    payload = project_dom_browser_observation(
        artifact_sha256=artifact_sha256,
        byte_count=len(raw),
        capture_role=str(capture_role),
        derivative_sha256=derivative_sha256,
        artifact_identity=artifact_identity,
    )
    emitted = emit_dom_browser_observation_if_requested(
        observation_emit_func=observation_emit_func,
        payload=payload,
    )

    return {
        "contract": DOM_BROWSER_METADATA_CONTRACT,
        "artifact_sha256": artifact_sha256,
        "artifact_identity": identity,
        "byte_count": len(raw),
        "media_type": DOM_BROWSER_MEDIA_TYPE,
        "capture_role": str(capture_role),
        "serialization_semantics": DOM_BROWSER_SERIALIZATION_SEMANTICS,
        "derivative_sha256": derivative_sha256,
        "sealed_original_persisted": True,
        "raw_original_mutated": False,
        "raw_dom_payload_included": False,
        "filesystem_path_included": False,
        "dom_content_summary_included": False,
        "automatic_timestamp": False,
        "observation_emitted": emitted,
    }
