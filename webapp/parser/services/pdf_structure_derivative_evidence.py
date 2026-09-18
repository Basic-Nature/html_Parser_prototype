"""Dormant trusted observation for bounded PDF structure derivatives.

This contract is transport-neutral and noncanonical. It emits only a canonical
fingerprint of an already-bounded structure-phase summary when an explicit
callback is supplied. It never reads the source PDF, writes files, acquires
geometry, or changes parser decisions. Source identity is an already-known
scalar derived by the outer PDF wrapper.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import re
from typing import Any, Callable

CONTRACT = "pdf_structure_derivative_observation_v1"
SURFACE = "PDF_STRUCTURE_DERIVATIVE"
DERIVATIVE_FINGERPRINT_SCHEME = (
    "SHA256_CANONICAL_JSON_OF_BOUNDED_STRUCTURE_PHASE_V1"
)
SOURCE_IDENTITY_SEMANTICS = "SHA256_OF_IMMUTABLE_CONTENT_BYTES"
SUPPORTED_PHASES = (
    "page_text_structure",
    "contest_hint_structure",
    "columnar_structure",
)
_SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")

_PHASE_FIELDS = {
    "page_text_structure": (
        "page_count",
        "page_line_total",
        "page_line_pages",
        "page_line_source",
        "page_line_index_available",
        "page_lines_fallback",
        "page_text_map_entries",
        "fitz_mode",
    ),
    "contest_hint_structure": (
        "contest_detection_available",
        "detected_title_count",
        "selection_mode_if_already_present",
        "contest_segment_hint_count_if_already_present",
    ),
    "columnar_structure": (
        "attempted",
        "attempt_count_if_already_present",
        "failure_present",
        "result_present",
        "segment_count_if_already_present",
    ),
}


class PdfStructureDerivativeObservationError(RuntimeError):
    """Raised only when explicitly enabled structure observation is invalid."""


def _source_identity(source_document_sha256: Any) -> str | None:
    if source_document_sha256 is None:
        return None
    if (
        not isinstance(source_document_sha256, str)
        or not _SHA256_RE.fullmatch(source_document_sha256)
    ):
        raise PdfStructureDerivativeObservationError(
            "source document SHA-256 is invalid."
        )
    return source_document_sha256


def _nonnegative_int(
    name: str,
    value: Any,
    *,
    allow_none: bool = False,
) -> int | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PdfStructureDerivativeObservationError(
            f"{name} must be a non-negative integer"
            + (" or None." if allow_none else ".")
        )
    return value


def _bounded_text(
    name: str,
    value: Any,
    *,
    allow_none: bool = True,
) -> str | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or len(value) > 80:
        raise PdfStructureDerivativeObservationError(
            f"{name} must be bounded text"
            + (" or None." if allow_none else ".")
        )
    return value


def _exact_bool(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise PdfStructureDerivativeObservationError(
            f"{name} must be a boolean."
        )
    return value


def _normalize_summary(
    phase: Any,
    bounded_summary: Any,
) -> tuple[str, dict[str, object]]:
    if not isinstance(phase, str) or phase not in SUPPORTED_PHASES:
        raise PdfStructureDerivativeObservationError(
            "structure phase is unsupported."
        )
    if not isinstance(bounded_summary, Mapping):
        raise PdfStructureDerivativeObservationError(
            "bounded_summary must be a mapping."
        )

    expected = _PHASE_FIELDS[phase]
    if set(bounded_summary) != set(expected):
        raise PdfStructureDerivativeObservationError(
            "bounded_summary fields do not match the structure phase contract."
        )

    if phase == "page_text_structure":
        page_line_source = bounded_summary["page_line_source"]
        if page_line_source not in {"fallback", "page_map"}:
            raise PdfStructureDerivativeObservationError(
                "page_line_source is unsupported."
            )
        normalized = {
            "page_count": _nonnegative_int(
                "page_count",
                bounded_summary["page_count"],
                allow_none=True,
            ),
            "page_line_total": _nonnegative_int(
                "page_line_total",
                bounded_summary["page_line_total"],
            ),
            "page_line_pages": _nonnegative_int(
                "page_line_pages",
                bounded_summary["page_line_pages"],
            ),
            "page_line_source": page_line_source,
            "page_line_index_available": _exact_bool(
                "page_line_index_available",
                bounded_summary["page_line_index_available"],
            ),
            "page_lines_fallback": _exact_bool(
                "page_lines_fallback",
                bounded_summary["page_lines_fallback"],
            ),
            "page_text_map_entries": _nonnegative_int(
                "page_text_map_entries",
                bounded_summary["page_text_map_entries"],
            ),
            "fitz_mode": _bounded_text(
                "fitz_mode",
                bounded_summary["fitz_mode"],
            ),
        }
    elif phase == "contest_hint_structure":
        normalized = {
            "contest_detection_available": _exact_bool(
                "contest_detection_available",
                bounded_summary["contest_detection_available"],
            ),
            "detected_title_count": _nonnegative_int(
                "detected_title_count",
                bounded_summary["detected_title_count"],
            ),
            "selection_mode_if_already_present": _bounded_text(
                "selection_mode_if_already_present",
                bounded_summary["selection_mode_if_already_present"],
            ),
            "contest_segment_hint_count_if_already_present": _nonnegative_int(
                "contest_segment_hint_count_if_already_present",
                bounded_summary[
                    "contest_segment_hint_count_if_already_present"
                ],
                allow_none=True,
            ),
        }
    else:
        normalized = {
            "attempted": _exact_bool(
                "attempted",
                bounded_summary["attempted"],
            ),
            "attempt_count_if_already_present": _nonnegative_int(
                "attempt_count_if_already_present",
                bounded_summary["attempt_count_if_already_present"],
                allow_none=True,
            ),
            "failure_present": _exact_bool(
                "failure_present",
                bounded_summary["failure_present"],
            ),
            "result_present": _exact_bool(
                "result_present",
                bounded_summary["result_present"],
            ),
            "segment_count_if_already_present": _nonnegative_int(
                "segment_count_if_already_present",
                bounded_summary["segment_count_if_already_present"],
                allow_none=True,
            ),
        }

    return phase, normalized


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def observe_pdf_structure_derivative_if_requested(
    *,
    emit_func: Callable[[dict[str, object]], object] | None,
    phase: Any,
    bounded_summary: Any,
    source_document_sha256: Any = None,
) -> dict[str, object] | None:
    """Emit one hash-only bounded phase observation, or return dormant."""
    if emit_func is None:
        return None

    if not callable(emit_func):
        raise PdfStructureDerivativeObservationError(
            "structure observation callback must be callable."
        )

    source_identity = _source_identity(source_document_sha256)
    normalized_phase, normalized_summary = _normalize_summary(
        phase,
        bounded_summary,
    )

    derivative_core = {
        "phase": normalized_phase,
        "bounded_summary": normalized_summary,
    }
    derivative_sha256 = hashlib.sha256(
        _canonical_bytes(derivative_core)
    ).hexdigest()

    core: dict[str, object] = {
        "contract": CONTRACT,
        "surface": SURFACE,
        "producer": "pdf_handler.parse_pdf_election_results",
        "canonical_authority": False,
        "source_identity_algorithm": "sha256",
        "source_identity_semantics": SOURCE_IDENTITY_SEMANTICS,
        "source_document_sha256": source_identity,
        "phase": normalized_phase,
        "derivative_fingerprint_scheme": DERIVATIVE_FINGERPRINT_SCHEME,
        "derivative_hash_algorithm": "sha256_canonical_json",
        "derivative_sha256": derivative_sha256,
        "bounded_summary": normalized_summary,
        "raw_content_included": False,
        "new_persistence": False,
    }
    observation: dict[str, object] = {
        **core,
        "observation_sha256": hashlib.sha256(
            _canonical_bytes(core)
        ).hexdigest(),
    }

    try:
        emit_func(dict(observation))
    except Exception as exc:
        raise PdfStructureDerivativeObservationError(
            "PDF structure derivative observation callback failed."
        ) from exc

    return observation
