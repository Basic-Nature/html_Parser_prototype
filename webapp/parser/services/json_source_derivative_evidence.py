"""Noncanonical trusted observation for decoded JSON source derivatives.

Dormant unless an explicit callback is supplied. This module never reads files,
URLs, parser state, databases, or persistent storage.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Callable

CONTRACT = "json_source_derivative_observation_v1"
FINGERPRINT_SCHEME = "SHA256_CANONICAL_JSON_OF_DECODED_JSON_SOURCE_V1"
SOURCE_IDENTITY_ALGORITHM = "sha256"
SOURCE_IDENTITY_SEMANTICS = "SHA256_OF_IMMUTABLE_CONTENT_BYTES"
SUPPORTED_PRODUCERS = frozenset({"json_handler.parse"})
_SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")


class JsonSourceDerivativeObservationError(RuntimeError):
    pass


def _validate_source_identity(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise JsonSourceDerivativeObservationError(
            "source_document_sha256 must be lowercase SHA-256 or None."
        )
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        rendered = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise JsonSourceDerivativeObservationError(
            "decoded JSON value is not canonicalizable."
        ) from exc
    return rendered.encode("utf-8")


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    raise JsonSourceDerivativeObservationError(
        "decoded JSON contains an unsupported value type."
    )


def _walk_counts(value: Any) -> dict[str, int]:
    counts = {
        "object_count": 0,
        "array_count": 0,
        "scalar_count": 0,
        "key_count": 0,
        "null_count": 0,
        "string_count": 0,
        "number_count": 0,
        "boolean_count": 0,
        "max_depth": 0,
    }

    def visit(node: Any, depth: int) -> None:
        counts["max_depth"] = max(counts["max_depth"], depth)
        if node is None:
            counts["scalar_count"] += 1
            counts["null_count"] += 1
            return
        if isinstance(node, bool):
            counts["scalar_count"] += 1
            counts["boolean_count"] += 1
            return
        if isinstance(node, str):
            counts["scalar_count"] += 1
            counts["string_count"] += 1
            return
        if isinstance(node, int) and not isinstance(node, bool):
            counts["scalar_count"] += 1
            counts["number_count"] += 1
            return
        if isinstance(node, float):
            if not math.isfinite(node):
                raise JsonSourceDerivativeObservationError(
                    "decoded JSON contains a non-finite number."
                )
            counts["scalar_count"] += 1
            counts["number_count"] += 1
            return
        if isinstance(node, list):
            counts["array_count"] += 1
            for child in node:
                visit(child, depth + 1)
            return
        if isinstance(node, dict):
            counts["object_count"] += 1
            for key, child in node.items():
                if not isinstance(key, str):
                    raise JsonSourceDerivativeObservationError(
                        "decoded JSON object keys must be strings."
                    )
                counts["key_count"] += 1
                visit(child, depth + 1)
            return
        raise JsonSourceDerivativeObservationError(
            "decoded JSON contains an unsupported value type."
        )

    visit(value, 0)
    return counts


def observe_json_source_derivative_if_requested(
    *,
    emit_func: Callable[[dict[str, object]], object] | None,
    decoded_json: Any,
    producer: str,
    source_document_sha256: str | None,
) -> dict[str, object] | None:
    if emit_func is None:
        return None
    if not callable(emit_func):
        raise JsonSourceDerivativeObservationError(
            "json source derivative observer must be callable."
        )
    if producer not in SUPPORTED_PRODUCERS:
        raise JsonSourceDerivativeObservationError(
            "unsupported json source derivative producer."
        )

    source_sha = _validate_source_identity(source_document_sha256)
    counts = _walk_counts(decoded_json)
    canonical = _canonical_json_bytes(decoded_json)
    core: dict[str, object] = {
        "contract": CONTRACT,
        "producer": producer,
        "fingerprint_scheme": FINGERPRINT_SCHEME,
        "derivative_hash_algorithm": "sha256",
        "derivative_sha256": hashlib.sha256(canonical).hexdigest(),
        "derivative_byte_count": len(canonical),
        "source_identity_algorithm": SOURCE_IDENTITY_ALGORITHM,
        "source_identity_semantics": SOURCE_IDENTITY_SEMANTICS,
        "source_document_sha256": source_sha,
        "source_identity_present": source_sha is not None,
        "top_level_type": _type_name(decoded_json),
        **counts,
        "raw_json_values_included": False,
        "raw_object_keys_included": False,
        "operational_manifest_trusted_authority": False,
        "canonical_election_authority": False,
        "automatic_timestamp": False,
    }
    observation_bytes = json.dumps(
        core,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    observation: dict[str, object] = {
        **core,
        "observation_sha256": hashlib.sha256(observation_bytes).hexdigest(),
    }
    try:
        emit_func(dict(observation))
    except Exception as exc:
        raise JsonSourceDerivativeObservationError(
            "json source derivative observation callback failed."
        ) from exc
    return observation
