"""Result contract for pure table-structure candidate materialization.

"Candidate" means an alternate TABLE HEADER STRUCTURE, not an election
candidate.  Materialization in this checkpoint selects an explicit row basis
and isolates those rows unchanged. It does not project, synthesize, harmonize,
or mutate source rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping, Tuple

from .table_structure_review_candidates import (
    TableStructureReviewCandidateRowBasis,
)


CONTRACT_VERSION = "table_structure_review_candidate_materialization_v1"
RUNTIME_TRANSPORT_WIRED = False
CANONICAL_AUTHORITY = False


@dataclass(frozen=True)
class TableStructureReviewCandidateMaterializationResult:
    """Pure materialization output with explicit non-transform provenance."""

    contract_version: ClassVar[str] = CONTRACT_VERSION
    runtime_transport_wired: ClassVar[bool] = RUNTIME_TRANSPORT_WIRED
    canonical_authority: ClassVar[bool] = CANONICAL_AUTHORITY

    row_projection_applied: ClassVar[bool] = False
    missing_header_values_synthesized: ClassVar[bool] = False
    extra_row_keys_dropped: ClassVar[bool] = False
    harmonization_applied: ClassVar[bool] = False
    source_rows_mutated: ClassVar[bool] = False
    session_integration: ClassVar[bool] = False

    review_id: str
    candidate_index: int
    candidates_total: int
    row_basis: TableStructureReviewCandidateRowBasis
    headers: Tuple[str, ...]
    rows: Tuple[Mapping[str, Any], ...]
