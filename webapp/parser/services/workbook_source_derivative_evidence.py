"""Dormant trusted observation for decoded workbook source derivatives.

The service is transport-neutral and noncanonical. It fingerprints the decoded
workbook representation already held by a parser and emits only hashes/counts
when an explicit callback is supplied. It never opens the source file, hashes a
path, persists decoded content, or changes parser decisions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime, time
import hashlib
import json
import math
import re
from typing import Any, Callable


CONTRACT = "workbook_source_derivative_observation_v1"
SURFACE = "WORKBOOK_SOURCE_DERIVATIVE"
DERIVATIVE_FINGERPRINT_SCHEME = (
    "SHA256_CANONICAL_JSON_OF_DECODED_WORKBOOK_SOURCE_V1"
)
SOURCE_IDENTITY_SEMANTICS = "SHA256_OF_IMMUTABLE_CONTENT_BYTES"
SUPPORTED_PRODUCERS = frozenset(
    {
        "fec_handler.parse",
        "xlsx_handler.parse_xlsx_election_results",
        "format_router.prompt_and_handle_download",
    }
)
_SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")


class WorkbookSourceDerivativeObservationError(RuntimeError):
    """Requested workbook source evidence could not be formed safely."""


def _source_identity(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise WorkbookSourceDerivativeObservationError(
            "source document SHA-256 is invalid."
        )
    return value


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if type(value).__name__ in {"NAType", "NaTType"}:
        return True
    try:
        probe = value != value
        if isinstance(probe, bool):
            return probe
        item = getattr(probe, "item", None)
        if callable(item):
            scalar = item()
            if isinstance(scalar, bool):
                return scalar
    except Exception:
        pass
    return False


def _value(value: Any) -> object:
    if _is_missing(value):
        return None
    if isinstance(value, str):
        return {"type": "str", "value": value}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": value}
    if isinstance(value, float):
        if math.isfinite(value):
            return {"type": "float", "value": value}
        return {"type": "float", "value": str(value).lower()}
    if isinstance(value, bytes):
        return {
            "type": "bytes",
            "sha256": hashlib.sha256(value).hexdigest(),
            "byte_count": len(value),
        }
    if isinstance(value, (datetime, date, time)):
        return {"type": type(value).__name__, "value": value.isoformat()}
    if isinstance(value, Mapping):
        pairs: list[list[object]] = []
        for key, item in value.items():
            pairs.append([_value(key), _value(item)])
        pairs.sort(
            key=lambda pair: json.dumps(
                pair[0], sort_keys=True, separators=(",", ":"),
                ensure_ascii=False, allow_nan=False,
            )
        )
        return {"type": "mapping", "value": pairs}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return {"type": "sequence", "value": [_value(item) for item in value]}
    item = getattr(value, "item", None)
    if callable(item):
        try:
            scalar = item()
        except Exception:
            scalar = value
        if scalar is not value:
            return _value(scalar)
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "value": str(value),
    }


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")


def _is_nonempty(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, dict):
        if value.get("type") == "str":
            return value.get("value") != ""
        if value.get("type") in {"sequence", "mapping"}:
            return bool(value.get("value"))
    return True


def _projection(decoded_frame: Any, sheet_name: Any) -> tuple[dict[str, object], int]:
    if decoded_frame is None:
        raise WorkbookSourceDerivativeObservationError(
            "decoded workbook frame is unavailable."
        )
    if not hasattr(decoded_frame, "columns") or not callable(
        getattr(decoded_frame, "to_dict", None)
    ):
        raise WorkbookSourceDerivativeObservationError(
            "decoded workbook frame must expose columns and to_dict()."
        )
    try:
        columns_raw = list(decoded_frame.columns)
        records_raw = decoded_frame.to_dict(orient="records")
    except Exception as exc:
        raise WorkbookSourceDerivativeObservationError(
            "decoded workbook frame projection failed."
        ) from exc
    if not isinstance(records_raw, list):
        raise WorkbookSourceDerivativeObservationError(
            "decoded workbook records must be a list."
        )
    columns = [_value(value) for value in columns_raw]
    rows: list[list[object]] = []
    nonempty = 0
    for record in records_raw:
        if not isinstance(record, Mapping):
            raise WorkbookSourceDerivativeObservationError(
                "decoded workbook rows must be mappings."
            )
        row = [_value(record.get(column)) for column in columns_raw]
        rows.append(row)
        if any(_is_nonempty(value) for value in row):
            nonempty += 1
    return {"sheet": _value(sheet_name), "columns": columns, "rows": rows}, nonempty


def observe_workbook_source_derivative_if_requested(
    *,
    emit_func: Callable[[dict[str, object]], object] | None,
    decoded_frame: Any,
    producer: Any,
    sheet_name: Any = None,
    source_document_sha256: Any = None,
) -> dict[str, object] | None:
    """Emit one hash-only workbook derivative observation, or remain dormant."""
    if emit_func is None:
        return None
    if not callable(emit_func):
        raise WorkbookSourceDerivativeObservationError(
            "workbook source derivative observer must be callable."
        )
    if not isinstance(producer, str) or producer not in SUPPORTED_PRODUCERS:
        raise WorkbookSourceDerivativeObservationError(
            "workbook source derivative producer is unsupported."
        )
    source_identity = _source_identity(source_document_sha256)
    decoded, nonempty = _projection(decoded_frame, sheet_name)
    derivative_bytes = _canonical(decoded)
    sheet_selector_bytes = _canonical(decoded["sheet"])
    core: dict[str, object] = {
        "contract": CONTRACT,
        "surface": SURFACE,
        "producer": producer,
        "canonical_authority": False,
        "source_identity_algorithm": "sha256",
        "source_identity_semantics": SOURCE_IDENTITY_SEMANTICS,
        "source_document_sha256": source_identity,
        "derivative_fingerprint_scheme": DERIVATIVE_FINGERPRINT_SCHEME,
        "derivative_hash_algorithm": "sha256",
        "derivative_sha256": hashlib.sha256(derivative_bytes).hexdigest(),
        "derivative_byte_count": len(derivative_bytes),
        "sheet_selector_sha256": hashlib.sha256(sheet_selector_bytes).hexdigest(),
        "column_count": len(decoded["columns"]),
        "row_count": len(decoded["rows"]),
        "nonempty_row_count": nonempty,
        "raw_sheet_name_included": False,
        "raw_column_values_included": False,
        "raw_cell_values_included": False,
        "new_persistence": False,
    }
    observation = {
        **core,
        "observation_sha256": hashlib.sha256(_canonical(core)).hexdigest(),
    }
    try:
        emit_func(dict(observation))
    except Exception as exc:
        raise WorkbookSourceDerivativeObservationError(
            "workbook source derivative observation callback failed."
        ) from exc
    return observation
