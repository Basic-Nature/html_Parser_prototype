"""Dormant trusted observation for decoded TXT/delimited source derivatives.

Transport-neutral and noncanonical. The service fingerprints an already-decoded
TXT/delimited representation held by a parser and emits only hashes/counts when
an explicit callback is supplied. It never opens the source file, hashes a path,
persists decoded content, or changes parser decisions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import re
from typing import Any, Callable


CONTRACT = "txt_source_derivative_observation_v1"
SURFACE = "TXT_DELIMITED_SOURCE_DERIVATIVE"
DERIVATIVE_FINGERPRINT_SCHEME = (
    "SHA256_CANONICAL_JSON_OF_DECODED_TXT_DELIMITED_SOURCE_V1"
)
SOURCE_IDENTITY_SEMANTICS = "SHA256_OF_IMMUTABLE_CONTENT_BYTES"
SUPPORTED_PRODUCERS = frozenset(
    {
        "txt_handler.parse_txt_election_results",
    }
)
_SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")


class TxtSourceDerivativeObservationError(RuntimeError):
    """Requested TXT source evidence could not be formed safely."""


def _source_identity(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise TxtSourceDerivativeObservationError(
            "source document SHA-256 is invalid."
        )
    return value


def _header(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TxtSourceDerivativeObservationError(
            "decoded TXT headers must contain strings or null."
        )
    return value


def _cell(value: Any) -> object:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        items: list[object] = []
        for item in value:
            if item is not None and not isinstance(item, str):
                raise TxtSourceDerivativeObservationError(
                    "decoded TXT overflow cells must contain strings or null."
                )
            items.append(item)
        return items
    raise TxtSourceDerivativeObservationError(
        "decoded TXT cells must be strings, string lists, or null."
    )


def _projection(
    decoded_headers: Any,
    decoded_rows: Any,
    *,
    encoding: str,
    delimiter: str,
) -> tuple[dict[str, object], int]:
    if (
        isinstance(decoded_headers, (str, bytes))
        or not isinstance(decoded_headers, Sequence)
    ):
        raise TxtSourceDerivativeObservationError(
            "decoded_headers must be a sequence."
        )
    if (
        isinstance(decoded_rows, (str, bytes))
        or not isinstance(decoded_rows, Sequence)
    ):
        raise TxtSourceDerivativeObservationError(
            "decoded_rows must be a sequence."
        )

    headers = [_header(item) for item in decoded_headers]
    rows: list[list[list[object]]] = []
    nonempty = 0
    for row in decoded_rows:
        if not isinstance(row, Mapping):
            raise TxtSourceDerivativeObservationError(
                "decoded_rows entries must be mappings."
            )
        projected_row: list[list[object]] = []
        row_nonempty = False
        for key, value in row.items():
            projected_key = _header(key)
            projected_value = _cell(value)
            projected_row.append([projected_key, projected_value])
            if isinstance(projected_value, list):
                row_nonempty = row_nonempty or any(
                    item not in (None, "") for item in projected_value
                )
            else:
                row_nonempty = row_nonempty or projected_value not in (
                    None,
                    "",
                )
        rows.append(projected_row)
        if row_nonempty:
            nonempty += 1

    return {
        "encoding": encoding,
        "delimiter": delimiter,
        "headers": headers,
        "rows": rows,
    }, nonempty


def observe_txt_source_derivative_if_requested(
    *,
    emit_func: Callable[[dict[str, object]], object] | None,
    decoded_headers: Any,
    decoded_rows: Any,
    encoding: Any,
    delimiter: Any,
    producer: Any,
    source_document_sha256: Any = None,
) -> dict[str, object] | None:
    """Emit one hash-only TXT/delimited source observation, or remain dormant."""
    if emit_func is None:
        return None
    if not callable(emit_func):
        raise TxtSourceDerivativeObservationError(
            "TXT source derivative observer must be callable."
        )
    if not isinstance(producer, str) or producer not in SUPPORTED_PRODUCERS:
        raise TxtSourceDerivativeObservationError(
            "TXT source derivative producer is unsupported."
        )
    if not isinstance(encoding, str) or not encoding or len(encoding) > 64:
        raise TxtSourceDerivativeObservationError(
            "TXT source derivative encoding is invalid."
        )
    if not isinstance(delimiter, str) or len(delimiter) != 1:
        raise TxtSourceDerivativeObservationError(
            "TXT source derivative delimiter is invalid."
        )

    source_identity = _source_identity(source_document_sha256)
    decoded, nonempty = _projection(
        decoded_headers,
        decoded_rows,
        encoding=encoding,
        delimiter=delimiter,
    )
    derivative_bytes = json.dumps(
        decoded,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    core: dict[str, object] = {
        "contract": CONTRACT,
        "surface": SURFACE,
        "producer": producer,
        "canonical_authority": False,
        "source_identity_algorithm": "sha256",
        "source_identity_semantics": SOURCE_IDENTITY_SEMANTICS,
        "source_document_sha256": source_identity,
        "encoding": encoding,
        "delimiter": delimiter,
        "derivative_fingerprint_scheme": DERIVATIVE_FINGERPRINT_SCHEME,
        "derivative_hash_algorithm": "sha256",
        "derivative_sha256": hashlib.sha256(derivative_bytes).hexdigest(),
        "derivative_byte_count": len(derivative_bytes),
        "header_count": len(decoded["headers"]),
        "row_count": len(decoded["rows"]),
        "nonempty_row_count": nonempty,
        "raw_header_values_included": False,
        "raw_row_values_included": False,
        "new_persistence": False,
    }
    canonical_core = json.dumps(
        core,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    observation = {
        **core,
        "observation_sha256": hashlib.sha256(canonical_core).hexdigest(),
    }

    try:
        emit_func(dict(observation))
    except Exception as exc:
        raise TxtSourceDerivativeObservationError(
            "TXT source derivative observation callback failed."
        ) from exc

    return observation
