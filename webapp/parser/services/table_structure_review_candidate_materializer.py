"""Pure off-runtime table-structure candidate materializer.

This service deliberately performs no row projection.

It selects:
    - one typed header proposal from the catalog
    - one explicitly declared row basis

Then it validates the selected row source BEFORE conversion/copy, deep-copies
those mappings, and freezes the copied mappings.

No missing header keys are synthesized. No extra keys are dropped.
No harmonization, session integration, runtime wiring, or canonical writes
occur here.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

from ..contracts.table_structure_review_candidate_materialization import (
    TableStructureReviewCandidateMaterializationResult,
)
from ..contracts.table_structure_review_candidates import (
    TableStructureReviewCandidateCatalog,
    TableStructureReviewCandidateContractError,
    TableStructureReviewCandidateRowBasis,
)


SERVICE_VERSION = "table_structure_review_candidate_materializer_v1"
SESSION_INTEGRATION = False
STATE_MACHINE_WIRED = False
RUNTIME_CALLER_WIRED = False
TRANSPORT_WIRED = False
FRONTEND_WIRED = False
HARMONIZATION_APPLIED = False
CANONICAL_AUTHORITY = False


class TableStructureReviewCandidateMaterializationError(ValueError):
    """Raised when pure candidate materialization cannot proceed safely."""


def _validate_row_source(
    rows: Any,
    *,
    source_name: str,
) -> Sequence[Mapping[str, Any]]:
    """Validate before any list()/dict() normalization."""

    if isinstance(rows, (str, bytes, bytearray)):
        raise TableStructureReviewCandidateMaterializationError(
            f"{source_name} must be a sequence of mappings, not text/bytes"
        )

    if not isinstance(rows, Sequence):
        raise TableStructureReviewCandidateMaterializationError(
            f"{source_name} must be a sequence of mappings"
        )

    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TableStructureReviewCandidateMaterializationError(
                f"{source_name}[{index}] must be a mapping"
            )

    return rows


def _freeze_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Deep-copy validated rows and freeze their top-level mappings."""

    return tuple(
        MappingProxyType(copy.deepcopy(dict(row)))
        for row in rows
    )


def materialize_table_structure_candidate(
    catalog: TableStructureReviewCandidateCatalog,
    candidate_index: int,
    *,
    immutable_original_rows: Sequence[Mapping[str, Any]] | None = None,
    current_working_rows: Sequence[Mapping[str, Any]] | None = None,
) -> TableStructureReviewCandidateMaterializationResult:
    """Materialize headers + one explicit row basis without row transformation."""

    if not isinstance(catalog, TableStructureReviewCandidateCatalog):
        raise TableStructureReviewCandidateMaterializationError(
            "catalog must be TableStructureReviewCandidateCatalog"
        )

    try:
        proposal = catalog.candidate_at(candidate_index)
    except TableStructureReviewCandidateContractError as exc:
        raise TableStructureReviewCandidateMaterializationError(
            str(exc)
        ) from exc

    if (
        catalog.row_basis
        is TableStructureReviewCandidateRowBasis.IMMUTABLE_ORIGINAL_ROWS
    ):
        if immutable_original_rows is None:
            raise TableStructureReviewCandidateMaterializationError(
                "immutable_original_rows is required by catalog row_basis"
            )

        selected_rows = _validate_row_source(
            immutable_original_rows,
            source_name="immutable_original_rows",
        )

    elif (
        catalog.row_basis
        is TableStructureReviewCandidateRowBasis.CURRENT_WORKING_ROWS
    ):
        if current_working_rows is None:
            raise TableStructureReviewCandidateMaterializationError(
                "current_working_rows is required by catalog row_basis"
            )

        selected_rows = _validate_row_source(
            current_working_rows,
            source_name="current_working_rows",
        )

    else:
        raise TableStructureReviewCandidateMaterializationError(
            "unsupported catalog row_basis"
        )

    frozen_rows = _freeze_rows(selected_rows)

    return TableStructureReviewCandidateMaterializationResult(
        review_id=catalog.review_id,
        candidate_index=proposal.candidate_index,
        candidates_total=catalog.candidates_total,
        row_basis=catalog.row_basis,
        headers=proposal.headers,
        rows=frozen_rows,
    )
