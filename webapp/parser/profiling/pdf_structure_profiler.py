"""Pure PDF structure observation accumulation.

This module is deliberately acquisition-free and decision-free.

It accepts observations that other parser phases have already made and builds
immutable :class:`PDFStructureProfile` snapshots. It does not open PDFs, run
OCR, choose parse strategies, alter confidence thresholds, or decide whether a
parse succeeds.

A snapshot reports exactly which evidence phases have been observed. It does
not expose an ``is_complete`` or scalar confidence authority because the PDF
pipeline has multiple valid terminal paths and no single existing seam contains
all structure evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from ..contracts.pdf_structure import (
    CellStateObservation,
    ContestSegmentProfile,
    PDFPageProfile,
    PDFStructureProfile,
    StructureFinding,
    TextProvider,
)


PDF_STRUCTURE_ACCUMULATOR_CONTRACT = "pdf_structure_profile_accumulator_v1"


class StructureObservationPhase(str, Enum):
    PAGE_TEXT_STRUCTURE = "page_text_structure"
    GEOMETRY = "geometry"
    CONTEST_HINT_STRUCTURE = "contest_hint_structure"
    COLUMNAR_STRUCTURE = "columnar_structure"


@dataclass(frozen=True)
class PageTextEvidence:
    """Page-level text provenance that does not pretend to be geometry."""

    page_number: int
    text_provider: TextProvider
    char_count: int
    line_count: int

    def __post_init__(self) -> None:
        if isinstance(self.page_number, bool) or not isinstance(
            self.page_number, int
        ):
            raise ValueError("page_number must be an integer")
        if self.page_number < 1:
            raise ValueError("page_number must be >= 1")
        if not isinstance(self.text_provider, TextProvider):
            raise ValueError("text_provider must be a TextProvider")
        for name, value in (
            ("char_count", self.char_count),
            ("line_count", self.line_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be >= 0")


@dataclass(frozen=True)
class PDFStructureProfileSnapshot:
    """Immutable observation snapshot with explicit phase coverage.

    The absence of ``is_complete`` is intentional. Different terminal parser
    paths observe different structural phases, so completeness must not be
    invented by this provider-neutral accumulator.
    """

    profile: PDFStructureProfile
    observed_phases: tuple[StructureObservationPhase, ...]
    page_text_evidence: tuple[PageTextEvidence, ...]
    missing_page_profiles: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.profile, PDFStructureProfile):
            raise ValueError("profile must be a PDFStructureProfile")
        for phase in self.observed_phases:
            if not isinstance(phase, StructureObservationPhase):
                raise ValueError(
                    "observed_phases must contain StructureObservationPhase"
                )
        for evidence in self.page_text_evidence:
            if not isinstance(evidence, PageTextEvidence):
                raise ValueError(
                    "page_text_evidence must contain PageTextEvidence"
                )
        for page_number in self.missing_page_profiles:
            if (
                isinstance(page_number, bool)
                or not isinstance(page_number, int)
                or page_number < 1
            ):
                raise ValueError(
                    "missing_page_profiles must contain positive integers"
                )

    def has_phase(self, phase: StructureObservationPhase) -> bool:
        if not isinstance(phase, StructureObservationPhase):
            raise ValueError("phase must be a StructureObservationPhase")
        return phase in self.observed_phases


class PDFStructureProfileAccumulator:
    """Accumulate already-observed structure without acquiring new evidence."""

    def __init__(self, document_sha256: str, page_count: int) -> None:
        # Delegates document/page validation to the accepted immutable contract
        # without performing I/O.
        PDFStructureProfile(
            document_sha256=document_sha256,
            page_count=page_count,
            native_text_page_ratio=None,
        )
        self._document_sha256 = document_sha256
        self._page_count = page_count
        self._native_text_page_ratio: Optional[float] = None
        self._page_text: dict[int, PageTextEvidence] = {}
        self._page_profiles: dict[int, PDFPageProfile] = {}
        self._contest_segments: list[ContestSegmentProfile] = []
        self._coverage_fingerprints: list[str] = []
        self._structural_findings: list[StructureFinding] = []
        self._cell_states: list[CellStateObservation] = []
        self._phases: list[StructureObservationPhase] = []

    @property
    def document_sha256(self) -> str:
        return self._document_sha256

    @property
    def page_count(self) -> int:
        return self._page_count

    def mark_phase(self, phase: StructureObservationPhase) -> None:
        if not isinstance(phase, StructureObservationPhase):
            raise ValueError("phase must be a StructureObservationPhase")
        if phase not in self._phases:
            self._phases.append(phase)

    def set_native_text_page_ratio(self, value: Optional[float]) -> None:
        """Record an explicit ratio without erasing stronger prior evidence."""
        PDFStructureProfile(
            document_sha256=self._document_sha256,
            page_count=self._page_count,
            native_text_page_ratio=value,
        )
        if value is None:
            if self._native_text_page_ratio is not None:
                raise ValueError(
                    "known native_text_page_ratio cannot be cleared to unknown"
                )
            return
        if (
            self._native_text_page_ratio is not None
            and self._native_text_page_ratio != value
        ):
            raise ValueError(
                "native_text_page_ratio already has a conflicting observation"
            )
        self._native_text_page_ratio = value

    def observe_page_text(self, evidence: PageTextEvidence) -> None:
        if not isinstance(evidence, PageTextEvidence):
            raise ValueError("evidence must be PageTextEvidence")
        if evidence.page_number > self._page_count:
            raise ValueError("page text evidence exceeds document page_count")
        existing = self._page_text.get(evidence.page_number)
        if existing is not None and existing != evidence:
            raise ValueError(
                "conflicting page text evidence for the same page"
            )
        self._page_text[evidence.page_number] = evidence

    def observe_page_profile(self, profile: PDFPageProfile) -> None:
        """Record contract-ready geometry for a page without merging it."""
        if not isinstance(profile, PDFPageProfile):
            raise ValueError("profile must be a PDFPageProfile")
        if profile.page_number > self._page_count:
            raise ValueError("page profile exceeds document page_count")
        existing = self._page_profiles.get(profile.page_number)
        if existing is not None and existing != profile:
            raise ValueError(
                "conflicting page profiles require explicit upstream review"
            )
        self._page_profiles[profile.page_number] = profile

    def observe_contest_segment(
        self,
        segment: ContestSegmentProfile,
    ) -> None:
        if not isinstance(segment, ContestSegmentProfile):
            raise ValueError("segment must be a ContestSegmentProfile")
        if segment.end_page > self._page_count:
            raise ValueError("contest segment exceeds document page_count")
        if segment not in self._contest_segments:
            self._contest_segments.append(segment)

    def observe_coverage_fingerprint(self, fingerprint: str) -> None:
        if not isinstance(fingerprint, str) or not fingerprint.strip():
            raise ValueError("coverage fingerprint must be non-empty text")
        if fingerprint not in self._coverage_fingerprints:
            self._coverage_fingerprints.append(fingerprint)

    def observe_structural_finding(
        self,
        finding: StructureFinding,
    ) -> None:
        if not isinstance(finding, StructureFinding):
            raise ValueError("finding must be a StructureFinding")
        if any(page > self._page_count for page in finding.page_numbers):
            raise ValueError("structural finding exceeds document page_count")
        if finding not in self._structural_findings:
            self._structural_findings.append(finding)

    def observe_cell_state(
        self,
        observation: CellStateObservation,
    ) -> None:
        """Preserve observations exactly; do not reconcile conflicting states."""
        if not isinstance(observation, CellStateObservation):
            raise ValueError("observation must be a CellStateObservation")
        if observation.page_number > self._page_count:
            raise ValueError("cell-state observation exceeds document page_count")
        if observation not in self._cell_states:
            self._cell_states.append(observation)

    def snapshot(self) -> PDFStructureProfileSnapshot:
        page_profiles = tuple(
            self._page_profiles[page_number]
            for page_number in sorted(self._page_profiles)
        )
        page_text_evidence = tuple(
            self._page_text[page_number]
            for page_number in sorted(self._page_text)
        )
        missing_page_profiles = tuple(
            page_number
            for page_number in range(1, self._page_count + 1)
            if page_number not in self._page_profiles
        )
        contest_segments = tuple(
            sorted(
                self._contest_segments,
                key=lambda segment: (
                    segment.start_page,
                    segment.end_page,
                    segment.title_observation or "",
                    segment.coverage_fingerprint or "",
                ),
            )
        )
        findings = tuple(
            sorted(
                self._structural_findings,
                key=lambda finding: (
                    finding.page_numbers,
                    finding.kind.value,
                    finding.detail,
                    -1.0
                    if finding.confidence is None
                    else finding.confidence,
                ),
            )
        )
        cell_states = tuple(
            sorted(
                self._cell_states,
                key=lambda observation: (
                    observation.page_number,
                    observation.row_label or "",
                    float("-inf")
                    if observation.column_anchor is None
                    else observation.column_anchor,
                    observation.state.value,
                    observation.token_text or "",
                ),
            )
        )
        profile = PDFStructureProfile(
            document_sha256=self._document_sha256,
            page_count=self._page_count,
            native_text_page_ratio=self._native_text_page_ratio,
            page_profiles=page_profiles,
            contest_segments=contest_segments,
            coverage_fingerprints=tuple(
                sorted(self._coverage_fingerprints)
            ),
            structural_findings=findings,
            cell_state_observations=cell_states,
        )
        return PDFStructureProfileSnapshot(
            profile=profile,
            observed_phases=tuple(self._phases),
            page_text_evidence=page_text_evidence,
            missing_page_profiles=missing_page_profiles,
        )
