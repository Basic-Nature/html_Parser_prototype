"""Provider-neutral PDF geometry and structure observation contracts.

Native selectable text and OCR-derived text boxes feed the same structure
profile. These observations are not canonical election truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from ._contract_validation import (
    require_finite_number,
    require_nonempty_text,
    require_nonnegative_int,
    require_positive_int,
    require_sha256,
)
from .parser_confidence import ConfidenceVector


PDF_STRUCTURE_CONTRACT = "pdf_structure_profile_v1"


class TextProvider(str, Enum):
    NATIVE = "native"
    OCR = "ocr"
    MIXED = "mixed"
    NONE = "none"


class CoordinateSpace(str, Enum):
    PDF_POINTS = "pdf_points"
    RASTER_PIXELS = "raster_pixels"
    NORMALIZED = "normalized"


class ObservedCellState(str, Enum):
    NUMERIC = "numeric"
    EXPLICIT_ZERO = "explicit_zero"
    TEXT = "text"
    BLANK = "blank"
    NOT_APPLICABLE = "not_applicable"
    PROTECTED = "protected"
    UNKNOWN = "unknown"


class StructureFindingKind(str, Enum):
    ROTATED_HEADER = "rotated_header"
    AMBIGUOUS_HEADER = "ambiguous_header"
    COVERAGE_CHANGE = "coverage_change"
    CONTINUATION_FRAGMENT = "continuation_fragment"
    PROTECTED_CELL = "protected_cell"
    SPARSE_ROW = "sparse_row"
    COLUMN_ANCHOR_DRIFT = "column_anchor_drift"
    CONTEST_BOUNDARY_AMBIGUITY = "contest_boundary_ambiguity"


@dataclass(frozen=True)
class BoundingBox:
    x0: float
    y0: float
    x1: float
    y1: float
    coordinate_space: CoordinateSpace

    def __post_init__(self) -> None:
        if not isinstance(self.coordinate_space, CoordinateSpace):
            raise ValueError("coordinate_space must be a CoordinateSpace")
        x0 = require_finite_number("x0", self.x0)
        y0 = require_finite_number("y0", self.y0)
        x1 = require_finite_number("x1", self.x1)
        y1 = require_finite_number("y1", self.y1)
        if x1 < x0:
            raise ValueError("x1 must be >= x0")
        if y1 < y0:
            raise ValueError("y1 must be >= y0")
        if self.coordinate_space is CoordinateSpace.NORMALIZED:
            for name, value in (
                ("x0", x0),
                ("y0", y0),
                ("x1", x1),
                ("y1", y1),
            ):
                if value < 0.0 or value > 1.0:
                    raise ValueError(
                        f"{name} must be within [0, 1] for normalized coordinates"
                    )


@dataclass(frozen=True)
class PageRegion:
    page_number: int
    box: BoundingBox

    def __post_init__(self) -> None:
        require_positive_int("page_number", self.page_number)
        if not isinstance(self.box, BoundingBox):
            raise ValueError("box must be a BoundingBox")


@dataclass(frozen=True)
class BallotLineObservation:
    """Observed ballot line; not canonical candidate/person identity."""

    candidate_text: str
    party_text: Optional[str] = None
    x_anchor: Optional[float] = None
    source_pages: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        require_nonempty_text("candidate_text", self.candidate_text)
        if self.party_text is not None:
            require_nonempty_text("party_text", self.party_text)
        require_finite_number(
            "x_anchor",
            self.x_anchor,
            allow_none=True,
        )
        for page in self.source_pages:
            require_positive_int("source_page", page)


@dataclass(frozen=True)
class CellStateObservation:
    page_number: int
    row_label: Optional[str]
    column_anchor: Optional[float]
    state: ObservedCellState
    token_text: Optional[str] = None

    def __post_init__(self) -> None:
        require_positive_int("page_number", self.page_number)
        if not isinstance(self.state, ObservedCellState):
            raise ValueError("state must be an ObservedCellState")
        require_finite_number(
            "column_anchor",
            self.column_anchor,
            allow_none=True,
        )


@dataclass(frozen=True)
class StructureFinding:
    kind: StructureFindingKind
    page_numbers: tuple[int, ...]
    detail: str
    confidence: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, StructureFindingKind):
            raise ValueError("kind must be a StructureFindingKind")
        if not self.page_numbers:
            raise ValueError("page_numbers must be non-empty")
        for page in self.page_numbers:
            require_positive_int("page_number", page)
        require_nonempty_text("detail", self.detail)
        require_finite_number(
            "confidence",
            self.confidence,
            minimum=0.0,
            maximum=1.0,
            allow_none=True,
        )


@dataclass(frozen=True)
class PDFPageProfile:
    page_number: int
    width_points: float
    height_points: float
    text_provider: TextProvider
    text_box_count: int
    header_angle_hypotheses: tuple[float, ...] = ()
    column_anchors: tuple[float, ...] = ()
    protected_token_count: int = 0
    structure_confidence: ConfidenceVector = ConfidenceVector()

    def __post_init__(self) -> None:
        require_positive_int("page_number", self.page_number)
        require_finite_number(
            "width_points",
            self.width_points,
            minimum=0.000001,
        )
        require_finite_number(
            "height_points",
            self.height_points,
            minimum=0.000001,
        )
        if not isinstance(self.text_provider, TextProvider):
            raise ValueError("text_provider must be a TextProvider")
        require_nonnegative_int("text_box_count", self.text_box_count)
        require_nonnegative_int(
            "protected_token_count",
            self.protected_token_count,
        )
        for angle in self.header_angle_hypotheses:
            require_finite_number("header_angle_hypothesis", angle)
        for anchor in self.column_anchors:
            require_finite_number("column_anchor", anchor)
        if not isinstance(self.structure_confidence, ConfidenceVector):
            raise ValueError(
                "structure_confidence must be a ConfidenceVector"
            )


@dataclass(frozen=True)
class ContestSegmentProfile:
    start_page: int
    end_page: int
    title_observation: Optional[str] = None
    vote_for_observation: Optional[str] = None
    header_signature: Optional[str] = None
    column_anchors: tuple[float, ...] = ()
    coverage_fingerprint: Optional[str] = None
    ballot_line_observations: tuple[BallotLineObservation, ...] = ()
    continuation_findings: tuple[StructureFinding, ...] = ()

    def __post_init__(self) -> None:
        require_positive_int("start_page", self.start_page)
        require_positive_int("end_page", self.end_page)
        if self.end_page < self.start_page:
            raise ValueError("end_page must be >= start_page")
        for anchor in self.column_anchors:
            require_finite_number("column_anchor", anchor)
        for observation in self.ballot_line_observations:
            if not isinstance(observation, BallotLineObservation):
                raise ValueError(
                    "ballot_line_observations must contain BallotLineObservation"
                )
        for finding in self.continuation_findings:
            if not isinstance(finding, StructureFinding):
                raise ValueError(
                    "continuation_findings must contain StructureFinding"
                )


@dataclass(frozen=True)
class PDFStructureProfile:
    document_sha256: str
    page_count: int
    native_text_page_ratio: Optional[float]
    page_profiles: tuple[PDFPageProfile, ...] = ()
    contest_segments: tuple[ContestSegmentProfile, ...] = ()
    coverage_fingerprints: tuple[str, ...] = ()
    structural_findings: tuple[StructureFinding, ...] = ()
    cell_state_observations: tuple[CellStateObservation, ...] = ()

    def __post_init__(self) -> None:
        require_sha256("document_sha256", self.document_sha256)
        require_positive_int("page_count", self.page_count)
        require_finite_number(
            "native_text_page_ratio",
            self.native_text_page_ratio,
            minimum=0.0,
            maximum=1.0,
            allow_none=True,
        )

        page_numbers: list[int] = []
        for page in self.page_profiles:
            if not isinstance(page, PDFPageProfile):
                raise ValueError(
                    "page_profiles must contain PDFPageProfile"
                )
            if page.page_number > self.page_count:
                raise ValueError(
                    "page profile exceeds document page_count"
                )
            page_numbers.append(page.page_number)
        if len(page_numbers) != len(set(page_numbers)):
            raise ValueError("page_profiles must not repeat a page number")

        for segment in self.contest_segments:
            if not isinstance(segment, ContestSegmentProfile):
                raise ValueError(
                    "contest_segments must contain ContestSegmentProfile"
                )
            if segment.end_page > self.page_count:
                raise ValueError(
                    "contest segment exceeds document page_count"
                )

        for fingerprint in self.coverage_fingerprints:
            require_nonempty_text("coverage_fingerprint", fingerprint)
        for finding in self.structural_findings:
            if not isinstance(finding, StructureFinding):
                raise ValueError(
                    "structural_findings must contain StructureFinding"
                )
        for observation in self.cell_state_observations:
            if not isinstance(observation, CellStateObservation):
                raise ValueError(
                    "cell_state_observations must contain CellStateObservation"
                )
            if observation.page_number > self.page_count:
                raise ValueError(
                    "cell state observation exceeds document page_count"
                )
