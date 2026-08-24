"""Pure table-structure review mutations with legacy pre-harmonization parity.

The functions in this module are deterministic and side-effect free. They
return new headers/rows and never invoke harmonization, learning persistence,
transport, parser control flow, or canonical promotion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple


SERVICE_VERSION = "table_structure_review_mutations_v1"

RUNTIME_CALLER_WIRED = False
STATE_MACHINE_WIRED = False
HARMONIZATION_INCLUDED = False
LEARNING_SIDE_EFFECT_AUTHORITY = False
TRANSPORT_AUTHORITY = False
INPUT_OBJECT_MUTATION = False
CANONICAL_AUTHORITY = False


class TableStructureReviewMutationError(ValueError):
    """Raised when typed mutation input cannot be applied safely."""


@dataclass(frozen=True)
class TableStructureMutationResult:
    """Immutable pure mutation output."""

    headers: Tuple[str, ...]
    rows: Tuple[Mapping[str, Any], ...]


def _require_headers(
    headers: Sequence[str],
) -> tuple[str, ...]:
    if isinstance(headers, (str, bytes)):
        raise TableStructureReviewMutationError(
            "headers must be a sequence of strings"
        )

    typed = tuple(headers)

    if not typed or not all(isinstance(header, str) for header in typed):
        raise TableStructureReviewMutationError(
            "headers must contain at least one string"
        )

    return typed


def _copy_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    if isinstance(rows, (str, bytes)):
        raise TableStructureReviewMutationError(
            "rows must be a sequence of mappings"
        )

    copied = []

    for row in rows:
        if not isinstance(row, Mapping):
            raise TableStructureReviewMutationError(
                "each row must be a mapping"
            )
        if not all(isinstance(key, str) for key in row.keys()):
            raise TableStructureReviewMutationError(
                "row keys must be strings"
            )
        copied.append(dict(row))

    return tuple(copied)


def _project_rows(
    rows: Sequence[Mapping[str, Any]],
    headers: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            header: row[header] if header in row else ""
            for header in headers
        }
        for row in rows
    )


def _require_index_sequence(
    indices: Sequence[int],
    *,
    field_name: str,
) -> tuple[int, ...]:
    if isinstance(indices, (str, bytes)):
        raise TableStructureReviewMutationError(
            f"{field_name} must be a sequence of integers"
        )

    typed = tuple(indices)

    if not typed:
        raise TableStructureReviewMutationError(
            f"{field_name} must not be empty"
        )

    for index in typed:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TableStructureReviewMutationError(
                f"{field_name} must contain integers only"
            )

    return typed


def apply_remove_columns(
    headers: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
) -> TableStructureMutationResult:
    """Mirror legacy remove branch before harmonize_headers_and_data()."""

    typed_headers = _require_headers(headers)
    copied_rows = _copy_rows(rows)
    typed_indices = _require_index_sequence(
        indices,
        field_name="indices",
    )

    # Legacy removal uses membership against the parsed zero-based index list.
    # Invalid/negative indices therefore have no effect on enumerate(headers).
    new_headers = tuple(
        header
        for index, header in enumerate(typed_headers)
        if index not in typed_indices
    )

    if not new_headers:
        # The legacy branch can remove every header. Preserve that result.
        projected_rows = tuple({} for _ in copied_rows)
    else:
        projected_rows = _project_rows(copied_rows, new_headers)

    return TableStructureMutationResult(
        headers=new_headers,
        rows=projected_rows,
    )


def apply_reorder_columns(
    headers: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    order: Sequence[int],
) -> TableStructureMutationResult:
    """Apply an already-filtered legacy reorder payload without harmonization."""

    typed_headers = _require_headers(headers)
    copied_rows = _copy_rows(rows)
    typed_order = _require_index_sequence(
        order,
        field_name="order",
    )

    for index in typed_order:
        if not 0 <= index < len(typed_headers):
            raise TableStructureReviewMutationError(
                "reorder index is outside current headers"
            )

    # Legacy reorder preserves subset and duplicate selections.
    new_headers = tuple(
        typed_headers[index]
        for index in typed_order
    )

    projected_rows = _project_rows(
        copied_rows,
        new_headers,
    )

    return TableStructureMutationResult(
        headers=new_headers,
        rows=projected_rows,
    )


def apply_rename_columns(
    headers: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    renames: Mapping[int, str],
) -> TableStructureMutationResult:
    """Mirror legacy sequential header rename + row reprojection."""

    typed_headers = list(_require_headers(headers))
    copied_rows = _copy_rows(rows)

    if not isinstance(renames, Mapping) or not renames:
        raise TableStructureReviewMutationError(
            "renames must be a non-empty mapping"
        )

    for index, raw_name in renames.items():
        if isinstance(index, bool) or not isinstance(index, int):
            raise TableStructureReviewMutationError(
                "rename indices must be integers"
            )
        if not 0 <= index < len(typed_headers):
            raise TableStructureReviewMutationError(
                "rename index is outside current headers"
            )
        if not isinstance(raw_name, str):
            raise TableStructureReviewMutationError(
                "rename values must be strings"
            )

        new_name = raw_name.strip()

        # Legacy interactive branch ignores an empty rename response.
        if new_name:
            typed_headers[index] = new_name

    new_headers = tuple(typed_headers)

    # This intentionally mirrors legacy behavior: reprojection uses final header
    # names against the original row keys. Renaming a header may therefore yield
    # "" unless that new key already existed in the row.
    projected_rows = _project_rows(
        copied_rows,
        new_headers,
    )

    return TableStructureMutationResult(
        headers=new_headers,
        rows=projected_rows,
    )


def apply_add_columns(
    headers: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    names: Sequence[str],
) -> TableStructureMutationResult:
    """Mirror legacy add/fill behavior before harmonization."""

    typed_headers = list(_require_headers(headers))
    copied_rows = [
        dict(row)
        for row in _copy_rows(rows)
    ]

    if isinstance(names, (str, bytes)):
        raise TableStructureReviewMutationError(
            "names must be a sequence of strings"
        )

    for raw_name in names:
        if not isinstance(raw_name, str):
            raise TableStructureReviewMutationError(
                "added column names must be strings"
            )

        name = raw_name.strip()

        if name and name not in typed_headers:
            # safe_append(..., deduplicate=False) is an ordinary append here,
            # guarded by the legacy "col not in candidate_headers" condition.
            typed_headers.append(name)

            for row in copied_rows:
                row[name] = row[name] if name in row else ""

    # Legacy add branch then ensures every candidate header exists in every row.
    for row in copied_rows:
        for header in typed_headers:
            if header not in row:
                row[header] = ""

    return TableStructureMutationResult(
        headers=tuple(typed_headers),
        rows=tuple(copied_rows),
    )