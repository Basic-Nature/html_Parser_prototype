from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

Observer = Callable[[dict[str, Any]], None]

_OBSERVER: ContextVar[Observer | None] = ContextVar(
    "electionpulse_ocr_derivative_observer",
    default=None,
)
_ROOT_SOURCE_IDENTITY: ContextVar[Any | None] = ContextVar(
    "electionpulse_ocr_derivative_root_source_identity",
    default=None,
)

_SAFE_CONTEXT_KEYS = frozenset({"page_index", "page_number", "ocr_pass", "purpose_code"})


@contextmanager
def ocr_derivative_observer(observer: Observer | None) -> Iterator[None]:
    token = _OBSERVER.set(observer)
    try:
        yield
    finally:
        _OBSERVER.reset(token)


@contextmanager
def ocr_derivative_source_identity(identity: Any | None) -> Iterator[None]:
    token = _ROOT_SOURCE_IDENTITY.set(identity)
    try:
        yield
    finally:
        _ROOT_SOURCE_IDENTITY.reset(token)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_context(context: Mapping[str, Any] | None) -> dict[str, Any]:
    if not context:
        return {}
    out: dict[str, Any] = {}
    for key in _SAFE_CONTEXT_KEYS:
        if key not in context:
            continue
        value = context[key]
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
    return out


def _source_fingerprint(value: Any) -> dict[str, Any]:
    type_name = f"{type(value).__module__}.{type(value).__qualname__}"

    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        return {
            "source_type": type_name,
            "fingerprint_scheme": "OCR_INPUT_BYTES_V1",
            "sha256": _sha256(raw),
            "byte_count": len(raw),
        }

    tobytes = getattr(value, "tobytes", None)
    mode = getattr(value, "mode", None)
    size = getattr(value, "size", None)
    if callable(tobytes) and isinstance(mode, str) and isinstance(size, tuple):
        raw = tobytes()
        header = json.dumps(
            {"mode": mode, "size": list(size)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        canonical = header + b"\0" + raw
        return {
            "source_type": type_name,
            "fingerprint_scheme": "OCR_INPUT_PIL_PIXELS_V1",
            "sha256": _sha256(canonical),
            "byte_count": len(raw),
            "mode": mode,
            "size": list(size),
        }

    dtype = getattr(value, "dtype", None)
    shape = getattr(value, "shape", None)
    if callable(tobytes) and dtype is not None and shape is not None:
        raw = value.tobytes(order="C")
        header = json.dumps(
            {"dtype": str(dtype), "shape": list(shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        canonical = header + b"\0" + raw
        return {
            "source_type": type_name,
            "fingerprint_scheme": "OCR_INPUT_NUMPY_C_ORDER_V1",
            "sha256": _sha256(canonical),
            "byte_count": len(raw),
            "dtype": str(dtype),
            "shape": list(shape),
        }

    return {
        "source_type": type_name,
        "fingerprint_scheme": "UNSUPPORTED_TYPE_PRESERVE_UNKNOWN_HASH",
        "sha256": None,
        "byte_count": None,
    }


def _result_fingerprint(value: Any) -> dict[str, Any]:
    type_name = f"{type(value).__module__}.{type(value).__qualname__}"

    if isinstance(value, str):
        raw = value.encode("utf-8")
        return {
            "result_type": type_name,
            "fingerprint_scheme": "OCR_RESULT_UTF8_V1",
            "sha256": _sha256(raw),
            "byte_count": len(raw),
        }

    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        return {
            "result_type": type_name,
            "fingerprint_scheme": "OCR_RESULT_BYTES_V1",
            "sha256": _sha256(raw),
            "byte_count": len(raw),
        }

    if isinstance(value, (dict, list)):
        try:
            raw = json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        except (TypeError, ValueError):
            pass
        else:
            return {
                "result_type": type_name,
                "fingerprint_scheme": "OCR_RESULT_CANONICAL_JSON_V1",
                "sha256": _sha256(raw),
                "byte_count": len(raw),
                "row_count": len(value) if isinstance(value, list) else None,
            }

    if type(value).__module__.startswith("pandas.") and type(value).__name__ == "DataFrame":
        raw = value.to_csv(index=False, lineterminator="\n").encode("utf-8")
        return {
            "result_type": type_name,
            "fingerprint_scheme": "OCR_RESULT_DATAFRAME_CSV_V1",
            "sha256": _sha256(raw),
            "byte_count": len(raw),
            "row_count": int(len(value.index)),
            "column_count": int(len(value.columns)),
        }

    return {
        "result_type": type_name,
        "fingerprint_scheme": "UNSUPPORTED_TYPE_PRESERVE_UNKNOWN_HASH",
        "sha256": None,
        "byte_count": None,
    }


def _existing_persistence_binding(persisted_path: str | Path | None) -> dict[str, Any]:
    if persisted_path is None:
        return {
            "logical_persistence_mode": "EPHEMERAL_DERIVATIVE_OBSERVATION",
            "persisted_sha256": None,
            "persisted_byte_count": None,
        }

    path = Path(persisted_path)
    if not path.is_file():
        return {
            "logical_persistence_mode": "EXISTING_DIAGNOSTIC_PERSISTENCE_BINDING",
            "persisted_sha256": None,
            "persisted_byte_count": None,
        }

    raw = path.read_bytes()
    return {
        "logical_persistence_mode": "EXISTING_DIAGNOSTIC_PERSISTENCE_BINDING",
        "persisted_sha256": _sha256(raw),
        "persisted_byte_count": len(raw),
    }


def observe_ocr_derivative(
    result: Any,
    *,
    source_input: Any,
    method: str,
    producer: str,
    context: Mapping[str, Any] | None = None,
    persisted_path: str | Path | None = None,
) -> Any:
    observer = _OBSERVER.get()
    if observer is None:
        return result

    observation = {
        "schema": "electionpulse.ocr_derivative_observation.v1",
        "surface": "OCR_DERIVATIVE",
        "producer": str(producer),
        "method": str(method),
        "root_source_artifact_identity": _ROOT_SOURCE_IDENTITY.get(),
        "source": _source_fingerprint(source_input),
        "derivative": _result_fingerprint(result),
        "context": _safe_context(context),
        "persistence": _existing_persistence_binding(persisted_path),
    }
    observer(observation)
    return result
