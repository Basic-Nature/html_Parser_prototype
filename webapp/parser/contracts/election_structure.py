"""Typed Smart Elections structural-analysis contracts.

Behavior-neutral and noncanonical. Missing/unknown is distinct from numeric
zero; signed evidence is preserved; dynamic vote methods are allowed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from .table_pipeline import TableStage

STRUCTURE_AUTHORITY = "noncanonical_parser_evidence"


class ElectionStructureIssueCode(str, Enum):
    MISSING_PRECINCT_HEADER = "missing_precinct_header"
    MISSING_PRECINCT_VALUE = "missing_precinct_value"
    DUPLICATE_PRECINCT = "duplicate_precinct"
    MISSING_CANDIDATE_TOTAL_COLUMN = "missing_candidate_total_column"
    MISSING_METHOD_COLUMN = "missing_method_column"
    DUPLICATE_METHOD_COLUMN = "duplicate_method_column"
    MISSING_METHOD_TOTAL_COLUMN = "missing_method_total_column"
    MISSING_VOTE_VALUE = "missing_vote_value"
    NON_NUMERIC_VOTE_VALUE = "non_numeric_vote_value"
    NEGATIVE_VOTE_VALUE = "negative_vote_value"
    CANDIDATE_TOTAL_MISSING = "candidate_total_missing"
    CANDIDATE_TOTAL_NON_NUMERIC = "candidate_total_non_numeric"
    CANDIDATE_TOTAL_MISMATCH = "candidate_total_mismatch"
    METHOD_TOTAL_MISSING = "method_total_missing"
    METHOD_TOTAL_NON_NUMERIC = "method_total_non_numeric"
    METHOD_TOTAL_MISMATCH = "method_total_mismatch"
    GRAND_TOTAL_MISSING = "grand_total_missing"
    GRAND_TOTAL_NON_NUMERIC = "grand_total_non_numeric"
    PRECINCT_TOTAL_MISMATCH = "precinct_total_mismatch"


class ElectionStructureSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class CandidateMethodBlock:
    candidate: str
    method_headers: tuple[tuple[str, str], ...] = ()
    total_header: str | None = None
    party_header: str | None = None
    percent_header: str | None = None

    def __post_init__(self) -> None:
        if not str(self.candidate).strip():
            raise ValueError("candidate must be non-empty")
        methods = [method for method, _ in self.method_headers]
        if any(not str(method).strip() for method in methods):
            raise ValueError("method names must be non-empty")
        if len(methods) != len(set(methods)):
            raise ValueError("method names must be unique per candidate")

    @property
    def methods(self) -> tuple[str, ...]:
        return tuple(method for method, _ in self.method_headers)

    def header_for_method(self, method: str) -> str | None:
        for observed, header in self.method_headers:
            if observed == method:
                return header
        return None


@dataclass(frozen=True)
class ElectionStructureIssue:
    code: ElectionStructureIssueCode
    message: str
    severity: ElectionStructureSeverity = ElectionStructureSeverity.WARNING
    row_index: int | None = None
    precinct: str | None = None
    candidate: str | None = None
    method: str | None = None
    header: str | None = None
    observed: Any = None
    expected: Any = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.code, ElectionStructureIssueCode):
            object.__setattr__(self, "code", ElectionStructureIssueCode(str(self.code)))
        if not isinstance(self.severity, ElectionStructureSeverity):
            object.__setattr__(
                self, "severity", ElectionStructureSeverity(str(self.severity))
            )
        if not self.message.strip():
            raise ValueError("message must be non-empty")
        if self.row_index is not None and self.row_index < 0:
            raise ValueError("row_index must be non-negative or None")


@dataclass(frozen=True)
class ElectionStructureAnalysis:
    source_stage: TableStage
    row_count: int
    precinct_header: str | None
    reporting_header: str | None
    grand_total_header: str | None
    method_total_headers: tuple[tuple[str, str], ...]
    candidates: tuple[str, ...]
    vote_methods: tuple[str, ...]
    candidate_blocks: tuple[CandidateMethodBlock, ...]
    duplicate_precincts: tuple[str, ...]
    issues: tuple[ElectionStructureIssue, ...]
    candidate_method_matrix_complete: bool
    schema_complete: bool
    candidate_totals_reconciled: bool | None
    method_totals_reconciled: bool | None
    precinct_totals_reconciled: bool | None
    observed_zero_count: int
    observed_null_count: int
    authority: str = STRUCTURE_AUTHORITY

    def __post_init__(self) -> None:
        if not isinstance(self.source_stage, TableStage):
            object.__setattr__(self, "source_stage", TableStage(str(self.source_stage)))
        if self.row_count < 0:
            raise ValueError("row_count must be non-negative")
        if self.observed_zero_count < 0 or self.observed_null_count < 0:
            raise ValueError("observed counts must be non-negative")
        if self.authority != STRUCTURE_AUTHORITY:
            raise ValueError("analysis authority must remain noncanonical")

    @property
    def issue_codes(self) -> tuple[ElectionStructureIssueCode, ...]:
        return tuple(issue.code for issue in self.issues)
