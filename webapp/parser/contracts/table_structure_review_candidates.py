"""Typed table-structure candidate catalog contract.

IMPORTANT: "candidate" here means an alternate TABLE HEADER STRUCTURE.
It does not mean an election/ballot candidate.

C2G 2.8.29 introduces no candidate materializer.  The catalog stores header
proposals only and requires an explicit row-basis policy so future
materialization cannot silently inherit the raw-Git shared-mutable-data model.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Tuple


CONTRACT_VERSION = "table_structure_review_candidates_v1"
RUNTIME_TRANSPORT_WIRED = False
CANONICAL_AUTHORITY = False
CATALOG_STORES_ROWS = False
MATERIALIZER_IMPLEMENTED = False


class TableStructureReviewCandidateContractError(ValueError):
    """Raised when a table-structure candidate catalog is malformed."""


class TableStructureReviewCandidateRowBasis(str, Enum):
    """Explicit source-row basis for a future candidate materializer.

    IMMUTABLE_ORIGINAL_ROWS:
        Candidate headers would be applied to the review session's immutable
        baseline rows.

    CURRENT_WORKING_ROWS:
        Candidate headers would be applied to the review session's current
        working rows at materialization time.

    No default is provided. A caller constructing a catalog must choose one
    explicitly. This enum defines vocabulary only; C2G 2.8.29 does not decide
    which policy the future GUI/runtime should use.
    """

    IMMUTABLE_ORIGINAL_ROWS = "IMMUTABLE_ORIGINAL_ROWS"
    CURRENT_WORKING_ROWS = "CURRENT_WORKING_ROWS"


@dataclass(frozen=True)
class TableStructureReviewCandidateHeaderProposal:
    """One 1-based alternate header proposal.

    Header order, duplicate header strings, and blank string values are
    preserved as supplied. This contract does not silently normalize or
    deduplicate legacy-compatible structural evidence.
    """

    candidate_index: int
    headers: Tuple[str, ...]

    def __post_init__(self) -> None:
        if isinstance(self.candidate_index, bool) or not isinstance(
            self.candidate_index,
            int,
        ):
            raise TableStructureReviewCandidateContractError(
                "candidate_index must be an integer"
            )

        if self.candidate_index < 1:
            raise TableStructureReviewCandidateContractError(
                "candidate_index must be at least 1"
            )

        if not isinstance(self.headers, tuple):
            raise TableStructureReviewCandidateContractError(
                "headers must be a tuple"
            )

        if not self.headers:
            raise TableStructureReviewCandidateContractError(
                "headers must not be empty"
            )

        if not all(isinstance(header, str) for header in self.headers):
            raise TableStructureReviewCandidateContractError(
                "headers must contain strings only"
            )


@dataclass(frozen=True)
class TableStructureReviewCandidateCatalog:
    """Header-proposal catalog with an explicit row-basis policy.

    The catalog deliberately carries no rows.  It therefore cannot alias,
    mutate, or own candidate-specific datasets.
    """

    contract_version: ClassVar[str] = CONTRACT_VERSION
    runtime_transport_wired: ClassVar[bool] = RUNTIME_TRANSPORT_WIRED
    canonical_authority: ClassVar[bool] = CANONICAL_AUTHORITY
    catalog_stores_rows: ClassVar[bool] = CATALOG_STORES_ROWS
    materializer_implemented: ClassVar[bool] = MATERIALIZER_IMPLEMENTED

    review_id: str
    row_basis: TableStructureReviewCandidateRowBasis
    candidates: Tuple[TableStructureReviewCandidateHeaderProposal, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.review_id, str) or not self.review_id:
            raise TableStructureReviewCandidateContractError(
                "review_id must be a non-empty string"
            )

        if not isinstance(
            self.row_basis,
            TableStructureReviewCandidateRowBasis,
        ):
            raise TableStructureReviewCandidateContractError(
                "row_basis must be TableStructureReviewCandidateRowBasis"
            )

        if not isinstance(self.candidates, tuple):
            raise TableStructureReviewCandidateContractError(
                "candidates must be a tuple"
            )

        if not self.candidates:
            raise TableStructureReviewCandidateContractError(
                "candidates must not be empty"
            )

        for candidate in self.candidates:
            if not isinstance(
                candidate,
                TableStructureReviewCandidateHeaderProposal,
            ):
                raise TableStructureReviewCandidateContractError(
                    "candidates must contain typed header proposals"
                )

        expected_indices = tuple(
            range(1, len(self.candidates) + 1)
        )
        actual_indices = tuple(
            candidate.candidate_index
            for candidate in self.candidates
        )

        if actual_indices != expected_indices:
            raise TableStructureReviewCandidateContractError(
                "candidate indices must be contiguous, ordered, and 1-based"
            )

    @property
    def candidates_total(self) -> int:
        """Number of structural header proposals in this catalog."""

        return len(self.candidates)

    def candidate_at(
        self,
        candidate_index: int,
    ) -> TableStructureReviewCandidateHeaderProposal:
        """Return a 1-based proposal; fail closed for out-of-range indices."""

        if isinstance(candidate_index, bool) or not isinstance(
            candidate_index,
            int,
        ):
            raise TableStructureReviewCandidateContractError(
                "candidate_index must be an integer"
            )

        if not 1 <= candidate_index <= self.candidates_total:
            raise TableStructureReviewCandidateContractError(
                "candidate_index is outside this catalog"
            )

        return self.candidates[candidate_index - 1]
