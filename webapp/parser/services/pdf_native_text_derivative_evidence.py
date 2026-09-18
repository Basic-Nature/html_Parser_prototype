"""Dormant trusted observation for selected native PDF text derivatives.

The contract is transport-neutral and noncanonical.  It emits only immutable
content fingerprints and bounded counts when an explicit callback is supplied.
It never reads the source PDF, writes a file, or persists raw extracted text.
Source identity is an already-known scalar derived by the wrapper; the inner
PDF parser does not consume the artifact-identity handoff object.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import re
from typing import Any, Callable

CONTRACT = "pdf_native_text_derivative_observation_v1"
SURFACE = "PDF_NATIVE_TEXT_DERIVATIVE"
DERIVATIVE_FINGERPRINT_SCHEME = "SHA256_UTF8_OF_SELECTED_NATIVE_TEXT_V1"
PAGE_FINGERPRINT_SCHEME = "SHA256_UTF8_OF_NATIVE_PAGE_TEXT_V1"
SOURCE_IDENTITY_SEMANTICS = "SHA256_OF_IMMUTABLE_CONTENT_BYTES"
_ALLOWED_MODES = frozenset({"text", "raw", "html", "xhtml"})
_SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")


class PdfNativeTextDerivativeObservationError(RuntimeError):
    """Raised only when explicitly enabled native-text observation is invalid."""


def _sha256_utf8(value: str) -> tuple[str, int]:
    raw = value.encode("utf-8")
    return hashlib.sha256(raw).hexdigest(), len(raw)


def _source_identity(source_document_sha256: Any) -> str | None:
    if source_document_sha256 is None:
        return None
    if (
        not isinstance(source_document_sha256, str)
        or not _SHA256_RE.fullmatch(source_document_sha256)
    ):
        raise PdfNativeTextDerivativeObservationError(
            "source document SHA-256 is invalid."
        )
    return source_document_sha256


def _page_fingerprints(
    page_text_map: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, object]], int]:
    if isinstance(page_text_map, (str, bytes)) or not isinstance(
        page_text_map,
        Sequence,
    ):
        raise PdfNativeTextDerivativeObservationError(
            "page_text_map must be a sequence."
        )

    pages: list[dict[str, object]] = []
    nonempty = 0
    seen: set[int] = set()

    for entry in page_text_map:
        if not isinstance(entry, Mapping):
            raise PdfNativeTextDerivativeObservationError(
                "page_text_map entries must be mappings."
            )

        page_index = entry.get("page")
        if (
            isinstance(page_index, bool)
            or not isinstance(page_index, int)
            or page_index < 0
            or page_index in seen
        ):
            raise PdfNativeTextDerivativeObservationError(
                "page index must be a unique non-negative integer."
            )
        seen.add(page_index)

        raw_text = entry.get("raw_text")
        if not isinstance(raw_text, str):
            raise PdfNativeTextDerivativeObservationError(
                "native page text must be a string."
            )

        char_count = entry.get("char_count")
        if (
            isinstance(char_count, bool)
            or not isinstance(char_count, int)
            or char_count < 0
            or char_count != len(raw_text)
        ):
            raise PdfNativeTextDerivativeObservationError(
                "native page char_count must match exact text length."
            )

        page_sha256, byte_count = _sha256_utf8(raw_text)
        if raw_text.strip():
            nonempty += 1

        pages.append(
            {
                "page_index": page_index,
                "fingerprint_scheme": PAGE_FINGERPRINT_SCHEME,
                "sha256": page_sha256,
                "byte_count": byte_count,
                "char_count": char_count,
            }
        )

    return pages, nonempty


def observe_pdf_native_text_derivative_if_requested(
    *,
    emit_func: Callable[[dict[str, object]], object] | None,
    selected_text: Any,
    page_text_map: Any,
    native_text_mode: Any,
    source_document_sha256: Any = None,
) -> dict[str, object] | None:
    """Emit one hash-only observation, or do nothing before inspecting inputs."""
    if emit_func is None:
        return None

    if not callable(emit_func):
        raise PdfNativeTextDerivativeObservationError(
            "native-text observation callback must be callable."
        )
    if not isinstance(selected_text, str):
        raise PdfNativeTextDerivativeObservationError(
            "selected native text must be a string."
        )
    if not isinstance(native_text_mode, str) or native_text_mode not in _ALLOWED_MODES:
        raise PdfNativeTextDerivativeObservationError(
            "native-text extraction mode is unsupported."
        )

    source_identity = _source_identity(source_document_sha256)
    derivative_sha256, derivative_byte_count = _sha256_utf8(selected_text)
    page_fingerprints, nonempty_page_count = _page_fingerprints(page_text_map)

    core: dict[str, object] = {
        "contract": CONTRACT,
        "surface": SURFACE,
        "producer": "pdf_handler.parse_pdf_election_results",
        "canonical_authority": False,
        "source_identity_algorithm": "sha256",
        "source_identity_semantics": SOURCE_IDENTITY_SEMANTICS,
        "source_document_sha256": source_identity,
        "native_text_mode": native_text_mode,
        "derivative_fingerprint_scheme": DERIVATIVE_FINGERPRINT_SCHEME,
        "derivative_hash_algorithm": "sha256_utf8",
        "derivative_sha256": derivative_sha256,
        "derivative_byte_count": derivative_byte_count,
        "derivative_char_count": len(selected_text),
        "page_count": len(page_fingerprints),
        "nonempty_page_count": nonempty_page_count,
        "page_fingerprints": page_fingerprints,
        "raw_text_included": False,
        "new_persistence": False,
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
        raise PdfNativeTextDerivativeObservationError(
            "native-text derivative observation callback failed."
        ) from exc

    return observation
