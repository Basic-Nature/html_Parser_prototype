"""Behavior-neutral typed contracts for the ElectionPulse table pipeline.

C2G 1.1 intentionally introduces types only.  No parser call sites are wired
to these contracts yet.

Semantic stages:
    EXTRACTED   - what did we observe?
    NORMALIZED  - how did we structurally represent it?
    INTERPRETED - what do we think it means?
    VALIDATED   - does the interpretation reconcile?
    LEARNED     - should the interpretation help future parsing?

CANONICAL is deliberately not a TableStage.  It is a separate governed
authority boundary and cannot be reached by automatic parser progression.

Value semantics:
    * None is not numeric zero.
    * Signed numeric evidence remains signed.
    * Unknown completeness remains unknown.
    * Presentation tokens such as "NA" do not replace typed internal absence.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from webapp.parser.Context_Integration.context_write_policy import ContextWriteKind


class TableStage(str, Enum):
    """Automatic/controlled parser stages below the canonical boundary."""

    EXTRACTED = "extracted"
    NORMALIZED = "normalized"
    INTERPRETED = "interpreted"
    VALIDATED = "validated"
    LEARNED = "learned"


class GovernedBoundary(str, Enum):
    """Authority boundaries that automatic parser stages may not cross."""

    CANONICAL = "canonical"


class WarningSeverity(str, Enum):
    """Severity for reviewable pipeline warnings."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class CompletenessState(str, Enum):
    """Whether a derived/observed collection is known to be complete."""

    UNKNOWN = "unknown"
    PARTIAL = "partial"
    COMPLETE = "complete"


_STAGE_ORDER: dict[TableStage, int] = {
    TableStage.EXTRACTED: 10,
    TableStage.NORMALIZED: 20,
    TableStage.INTERPRETED: 30,
    TableStage.VALIDATED: 40,
    TableStage.LEARNED: 50,
}


_TRANSFORMATION_COLLECTOR: ContextVar[list[TransformationRecord] | None] = ContextVar(
    "electionpulse_table_transformation_collector",
    default=None,
)


@contextmanager
def collect_transformations():
    """Collect TransformationRecord objects for one typed parser invocation.

    ContextVar keeps the collector task/thread local. Legacy parser calls that do
    not open this context collect nothing and retain their existing behavior.
    """

    records: list[TransformationRecord] = []
    token = _TRANSFORMATION_COLLECTOR.set(records)
    try:
        yield records
    finally:
        _TRANSFORMATION_COLLECTOR.reset(token)


def current_transformation_sequence() -> int:
    """Return the next sequence number inside the active collector."""

    records = _TRANSFORMATION_COLLECTOR.get()
    return len(records) if records is not None else 0


def record_transformation(record: TransformationRecord) -> bool:
    """Append one record when observability is active; otherwise no-op."""

    records = _TRANSFORMATION_COLLECTOR.get()
    if records is None:
        return False
    records.append(record)
    return True


def _coerce_table_stage(value: TableStage | str) -> TableStage:
    if isinstance(value, TableStage):
        return value
    return TableStage(str(value).strip().lower())


def _coerce_context_write_kind(
    value: ContextWriteKind | str,
) -> ContextWriteKind:
    if isinstance(value, ContextWriteKind):
        return value
    return ContextWriteKind(str(value).strip().lower())


def is_forward_stage_transition(
    from_stage: TableStage | str,
    to_stage: TableStage | str,
) -> bool:
    """Return True for same-stage or forward parser progression.

    CANONICAL cannot be passed here because GovernedBoundary is a distinct type.
    """

    source = _coerce_table_stage(from_stage)
    target = _coerce_table_stage(to_stage)
    return _STAGE_ORDER[target] >= _STAGE_ORDER[source]


@dataclass(frozen=True)
class SourceLocation:
    """Optional location of evidence inside a source artifact.

    Index fields are intentionally nullable because a source may not expose the
    corresponding coordinate.  Missing coordinates are not converted to zero.
    """

    page_number: int | None = None
    table_index: int | None = None
    row_index: int | None = None
    column_index: int | None = None
    selector: str | None = None

    def __post_init__(self) -> None:
        for name in ("page_number", "table_index", "row_index", "column_index"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative or None")


@dataclass(frozen=True)
class SourceProvenance:
    """Traceability for an observed/derived table result."""

    source_type: str
    source_uri: str | None = None
    source_sha256: str | None = None
    artifact_id: str | None = None
    evidence_ref: str | None = None
    location: SourceLocation | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.source_type).strip():
            raise ValueError("source_type must be non-empty")

        if self.source_sha256 is not None:
            sha = self.source_sha256.strip().lower()
            if len(sha) != 64 or any(ch not in "0123456789abcdef" for ch in sha):
                raise ValueError("source_sha256 must be a 64-character SHA256 hex string")


@dataclass(frozen=True)
class TransformationRecord:
    """One explainable transformation between parser semantic stages."""

    sequence: int
    from_stage: TableStage
    to_stage: TableStage
    operation: str
    rule_source: str | None = None
    confidence: float | None = None
    evidence_refs: tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "from_stage", _coerce_table_stage(self.from_stage))
        object.__setattr__(self, "to_stage", _coerce_table_stage(self.to_stage))

        if self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        if not str(self.operation).strip():
            raise ValueError("operation must be non-empty")
        if not is_forward_stage_transition(self.from_stage, self.to_stage):
            raise ValueError(
                f"backward stage transition is not allowed: "
                f"{self.from_stage.value} -> {self.to_stage.value}"
            )
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be between 0.0 and 1.0 or None")


@dataclass(frozen=True)
class PipelineWarning:
    """Reviewable warning without silently changing underlying evidence."""

    code: str
    message: str
    stage: TableStage
    severity: WarningSeverity = WarningSeverity.WARNING
    requires_review: bool = False
    evidence_refs: tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", _coerce_table_stage(self.stage))
        if not isinstance(self.severity, WarningSeverity):
            object.__setattr__(
                self,
                "severity",
                WarningSeverity(str(self.severity).strip().lower()),
            )

        if not str(self.code).strip():
            raise ValueError("warning code must be non-empty")
        if not str(self.message).strip():
            raise ValueError("warning message must be non-empty")


@dataclass(frozen=True)
class CompletenessInfo:
    """Completeness metadata for observed or derived values.

    All counts are optional.  Unknown information remains None and is never
    inferred as zero merely because a count is unavailable.
    """

    state: CompletenessState = CompletenessState.UNKNOWN
    expected_count: int | None = None
    observed_count: int | None = None
    missing_count: int | None = None
    null_value_count: int | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.state, CompletenessState):
            object.__setattr__(
                self,
                "state",
                CompletenessState(str(self.state).strip().lower()),
            )

        for name in (
            "expected_count",
            "observed_count",
            "missing_count",
            "null_value_count",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative or None")

        if self.state is CompletenessState.COMPLETE:
            if self.missing_count not in (None, 0):
                raise ValueError(
                    "COMPLETE completeness cannot declare a positive missing_count"
                )

        if self.state is CompletenessState.PARTIAL:
            if self.missing_count == 0:
                raise ValueError(
                    "PARTIAL completeness cannot declare missing_count=0"
                )

    @property
    def is_complete(self) -> bool | None:
        if self.state is CompletenessState.COMPLETE:
            return True
        if self.state is CompletenessState.PARTIAL:
            return False
        return None


@dataclass(frozen=True)
class TablePipelineResult:
    """Typed table state below the governed canonical boundary.

    This contract stores the stage, values, provenance, transformation history,
    warnings and completeness without mutating election values.

    `write_kind` reuses the existing parser context authority vocabulary.
    ContextWriteKind.CANONICAL is always rejected here.  Canonical promotion
    belongs to a separate governed service and is not a parser result state.
    """

    stage: TableStage
    headers: tuple[str, ...]
    rows: tuple[Mapping[str, Any], ...]
    source_provenance: SourceProvenance
    transformations: tuple[TransformationRecord, ...] = ()
    warnings: tuple[PipelineWarning, ...] = ()
    completeness: CompletenessInfo = field(default_factory=CompletenessInfo)
    semantic_annotations: Mapping[str, Any] = field(default_factory=dict)
    write_kind: ContextWriteKind = ContextWriteKind.NONE

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", _coerce_table_stage(self.stage))
        object.__setattr__(
            self,
            "write_kind",
            _coerce_context_write_kind(self.write_kind),
        )

        if self.write_kind is ContextWriteKind.CANONICAL:
            raise PermissionError(
                "TablePipelineResult cannot request canonical persistence; "
                "CANONICAL is a separate governed authority boundary."
            )

        # Normalize containers without altering cell values.  In particular,
        # None, 0 and negative signed values are preserved exactly.
        object.__setattr__(self, "headers", tuple(self.headers))
        object.__setattr__(
            self,
            "rows",
            tuple(dict(row) for row in self.rows),
        )
        object.__setattr__(
            self,
            "transformations",
            tuple(self.transformations),
        )
        object.__setattr__(self, "warnings", tuple(self.warnings))

        for transformation in self.transformations:
            if _STAGE_ORDER[transformation.to_stage] > _STAGE_ORDER[self.stage]:
                raise ValueError(
                    "transformation history cannot advance beyond result stage"
                )

    @classmethod
    def from_sequences(
        cls,
        *,
        stage: TableStage | str,
        headers: Sequence[str],
        rows: Sequence[Mapping[str, Any]],
        source_provenance: SourceProvenance,
        transformations: Sequence[TransformationRecord] = (),
        warnings: Sequence[PipelineWarning] = (),
        completeness: CompletenessInfo | None = None,
        semantic_annotations: Mapping[str, Any] | None = None,
        write_kind: ContextWriteKind | str = ContextWriteKind.NONE,
    ) -> "TablePipelineResult":
        """Convenience constructor for current list-based parser boundaries.

        This performs container adaptation only; it does not transform values,
        infer semantics, calculate totals, fill blanks, or serialize output.
        """

        return cls(
            stage=_coerce_table_stage(stage),
            headers=tuple(headers),
            rows=tuple(dict(row) for row in rows),
            source_provenance=source_provenance,
            transformations=tuple(transformations),
            warnings=tuple(warnings),
            completeness=completeness or CompletenessInfo(),
            semantic_annotations=semantic_annotations or {},
            write_kind=_coerce_context_write_kind(write_kind),
        )