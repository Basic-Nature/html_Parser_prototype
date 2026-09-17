"""Trusted screenshot-image evidence boundary.

Seals an already-persisted PNG by exact immutable bytes, writes deterministic
metadata/seal sidecars without copying pixel payloads, and may emit a
noncanonical summary through an explicit same-run callback.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct
from typing import Any, Callable

from ..contracts.artifact_identity import ArtifactIdentityHandoff

SCREENSHOT_IMAGE_METADATA_CONTRACT = "screenshot_image_metadata_v1"
SCREENSHOT_IMAGE_SEAL_CONTRACT = "screenshot_image_seal_v1"
SCREENSHOT_IMAGE_OBSERVATION_CONTRACT = "screenshot_image_observation_v1"
SCREENSHOT_IMAGE_OBSERVATION_AUTHORITY = "NONCANONICAL_OBSERVATION"
PNG_MEDIA_TYPE = "image/png"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


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


def _atomic_write_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(raw)
    os.replace(tmp, path)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _png_dimensions(raw: bytes) -> tuple[int, int]:
    if len(raw) < 24 or raw[:8] != PNG_SIGNATURE:
        raise ValueError("screenshot artifact must be PNG bytes")
    if raw[12:16] != b"IHDR":
        raise ValueError("PNG first chunk must be IHDR")
    if struct.unpack(">I", raw[8:12])[0] != 13:
        raise ValueError("PNG IHDR length must be 13")
    width, height = struct.unpack(">II", raw[16:24])
    if width <= 0 or height <= 0:
        raise ValueError("PNG width and height must be positive")
    return width, height


def _identity_payload(
    artifact_identity: ArtifactIdentityHandoff | None,
    *,
    sealed_sha256: str,
) -> dict[str, str] | None:
    if artifact_identity is None:
        return None
    if artifact_identity.document_sha256 != sealed_sha256:
        raise ValueError(
            "artifact_identity must match exact immutable screenshot bytes"
        )
    return {
        "document_sha256": artifact_identity.document_sha256,
        "algorithm": artifact_identity.algorithm,
        "semantics": artifact_identity.semantics,
    }


def project_screenshot_image_observation(
    *,
    artifact_sha256: str,
    byte_count: int,
    pixel_width: int,
    pixel_height: int,
    capture_role: str,
    derivative_sha256: str,
    artifact_identity: ArtifactIdentityHandoff | None = None,
) -> dict[str, Any]:
    return {
        "contract": SCREENSHOT_IMAGE_OBSERVATION_CONTRACT,
        "authority": SCREENSHOT_IMAGE_OBSERVATION_AUTHORITY,
        "canonical": False,
        "artifact_identity": _identity_payload(
            artifact_identity,
            sealed_sha256=artifact_sha256,
        ),
        "artifact_sha256": artifact_sha256,
        "byte_count": int(byte_count),
        "media_type": PNG_MEDIA_TYPE,
        "pixel_width": int(pixel_width),
        "pixel_height": int(pixel_height),
        "capture_role": str(capture_role),
        "derivative_sha256": derivative_sha256,
        "raw_screenshot_bytes_included": False,
        "raw_screenshot_path_included": False,
        "ocr_text_included": False,
        "visual_content_summary_included": False,
        "automatic_timestamp": False,
    }


def emit_screenshot_image_observation_if_requested(
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


def finalize_screenshot_image_evidence(
    screenshot_path: str | Path,
    *,
    capture_role: str,
    artifact_identity: ArtifactIdentityHandoff | None = None,
    observation_emit_func: Callable[[dict[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    path = Path(screenshot_path)
    raw = path.read_bytes()
    artifact_sha256 = _sha256(raw)
    pixel_width, pixel_height = _png_dimensions(raw)

    identity = _identity_payload(
        artifact_identity,
        sealed_sha256=artifact_sha256,
    )

    metadata = {
        "contract": SCREENSHOT_IMAGE_METADATA_CONTRACT,
        "artifact_sha256": artifact_sha256,
        "byte_count": len(raw),
        "media_type": PNG_MEDIA_TYPE,
        "pixel_width": pixel_width,
        "pixel_height": pixel_height,
        "capture_role": str(capture_role),
        "raw_original_mutated": False,
        "pixel_payload_included": False,
        "filesystem_path_included": False,
        "ocr_text_included": False,
        "visual_content_summary_included": False,
        "automatic_timestamp": False,
    }
    metadata_bytes = _canonical_json_bytes(metadata)
    derivative_sha256 = _sha256(metadata_bytes)

    metadata_path = path.with_suffix(path.suffix + ".metadata.json")
    seal_path = path.with_suffix(path.suffix + ".seal.json")
    seal = {
        "contract": SCREENSHOT_IMAGE_SEAL_CONTRACT,
        "algorithm": "sha256",
        "semantics": "SHA256_OF_IMMUTABLE_CONTENT_BYTES",
        "artifact_sha256": artifact_sha256,
        "artifact_identity": identity,
        "byte_count": len(raw),
        "media_type": PNG_MEDIA_TYPE,
        "pixel_width": pixel_width,
        "pixel_height": pixel_height,
        "metadata_derivative_sha256": derivative_sha256,
        "raw_original_mutated": False,
        "pixel_payload_included": False,
        "filesystem_path_included": False,
        "automatic_timestamp": False,
    }

    _atomic_write_bytes(metadata_path, metadata_bytes)
    _atomic_write_bytes(seal_path, _canonical_json_bytes(seal))

    payload = project_screenshot_image_observation(
        artifact_sha256=artifact_sha256,
        byte_count=len(raw),
        pixel_width=pixel_width,
        pixel_height=pixel_height,
        capture_role=str(capture_role),
        derivative_sha256=derivative_sha256,
        artifact_identity=artifact_identity,
    )
    emitted = emit_screenshot_image_observation_if_requested(
        observation_emit_func=observation_emit_func,
        payload=payload,
    )

    return {
        "contract": SCREENSHOT_IMAGE_METADATA_CONTRACT,
        "artifact_sha256": artifact_sha256,
        "artifact_identity": identity,
        "byte_count": len(raw),
        "media_type": PNG_MEDIA_TYPE,
        "pixel_width": pixel_width,
        "pixel_height": pixel_height,
        "capture_role": str(capture_role),
        "derivative_sha256": derivative_sha256,
        "raw_original_mutated": False,
        "pixel_payload_included": False,
        "filesystem_path_included": False,
        "ocr_text_included": False,
        "visual_content_summary_included": False,
        "automatic_timestamp": False,
        "observation_emitted": emitted,
    }
