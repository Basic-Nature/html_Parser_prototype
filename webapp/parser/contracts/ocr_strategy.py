"""Behavior-neutral adaptive OCR planning and outcome contracts.

Existing OCR implementation/configuration remains authoritative until later
integration. These types make strategy, parameters, expected gain and cost
auditable without executing OCR or choosing final confidence weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ._contract_validation import (
    require_finite_number,
    require_nonempty_text,
    require_nonnegative_int,
    require_positive_int,
)
from .parser_confidence import (
    ConfidenceDelta,
    ConfidenceDimension,
    ConfidenceVector,
)
from .pdf_structure import PageRegion


OCR_STRATEGY_CONTRACT = "ocr_strategy_plan_v1"


class OCRScope(str, Enum):
    DOCUMENT = "document"
    PAGE = "page"
    REGION = "region"


class OCRDecision(str, Enum):
    ACCEPT_CURRENT_EVIDENCE = "accept_current_evidence"
    ESCALATE_TARGETED_REGION = "escalate_targeted_region"
    ESCALATE_FULL_PAGE = "escalate_full_page"
    TRY_ROTATION_HYPOTHESIS = "try_rotation_hypothesis"
    HOLD_UNCERTAIN = "hold_uncertain"
    BUDGET_EXHAUSTED = "budget_exhausted"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class OCRExecutionParameters:
    """Auditable OCR engine/raster parameters for one pass.

    ``word_confidence_threshold`` is engine-native (Tesseract currently uses
    a 0-100 scale) and is deliberately separate from ConfidenceVector's 0-1
    parser-confidence dimensions.
    """

    engine: str
    dpi: int
    preprocessing: str
    oem: Optional[int] = None
    psm: Optional[int] = None
    word_confidence_threshold: Optional[float] = None

    def __post_init__(self) -> None:
        require_nonempty_text("engine", self.engine)
        require_positive_int("dpi", self.dpi)
        require_nonempty_text("preprocessing", self.preprocessing)
        if self.oem is not None:
            require_nonnegative_int("oem", self.oem)
        if self.psm is not None:
            require_nonnegative_int("psm", self.psm)
        require_finite_number(
            "word_confidence_threshold",
            self.word_confidence_threshold,
            minimum=0.0,
            maximum=100.0,
            allow_none=True,
        )


@dataclass(frozen=True)
class OCRBudget:
    max_parameter_trials: int
    max_sample_search_seconds: float
    max_full_page_passes: int
    max_region_passes: int
    max_rotation_hypotheses: int
    max_seconds_per_page: float
    max_seconds_per_document: float
    max_pixels_per_document: int

    def __post_init__(self) -> None:
        require_nonnegative_int(
            "max_parameter_trials",
            self.max_parameter_trials,
        )
        require_finite_number(
            "max_sample_search_seconds",
            self.max_sample_search_seconds,
            minimum=0.0,
        )
        require_nonnegative_int(
            "max_full_page_passes",
            self.max_full_page_passes,
        )
        require_nonnegative_int(
            "max_region_passes",
            self.max_region_passes,
        )
        require_nonnegative_int(
            "max_rotation_hypotheses",
            self.max_rotation_hypotheses,
        )
        require_finite_number(
            "max_seconds_per_page",
            self.max_seconds_per_page,
            minimum=0.0,
        )
        require_finite_number(
            "max_seconds_per_document",
            self.max_seconds_per_document,
            minimum=0.0,
        )
        require_nonnegative_int(
            "max_pixels_per_document",
            self.max_pixels_per_document,
        )


@dataclass(frozen=True)
class OCRPassPlan:
    pass_id: str
    scope: OCRScope
    parameters: OCRExecutionParameters
    target_dimensions: tuple[ConfidenceDimension, ...]
    expected_gain: ConfidenceDelta
    estimated_pixels: int
    page_numbers: tuple[int, ...] = ()
    regions: tuple[PageRegion, ...] = ()
    rotation_hypotheses: tuple[float, ...] = ()
    trigger_reasons: tuple[str, ...] = ()
    estimated_seconds: Optional[float] = None

    def __post_init__(self) -> None:
        require_nonempty_text("pass_id", self.pass_id)
        if not isinstance(self.scope, OCRScope):
            raise ValueError("scope must be an OCRScope")
        if not isinstance(self.parameters, OCRExecutionParameters):
            raise ValueError(
                "parameters must be OCRExecutionParameters"
            )
        if not self.target_dimensions:
            raise ValueError("target_dimensions must be non-empty")
        if len(self.target_dimensions) != len(set(self.target_dimensions)):
            raise ValueError("target_dimensions must not contain duplicates")
        for dimension in self.target_dimensions:
            if not isinstance(dimension, ConfidenceDimension):
                raise ValueError(
                    "target_dimensions must contain ConfidenceDimension"
                )
        if not isinstance(self.expected_gain, ConfidenceDelta):
            raise ValueError("expected_gain must be a ConfidenceDelta")
        if not self.expected_gain.has_nonnegative_values_for(
            self.target_dimensions
        ):
            raise ValueError(
                "every target dimension requires explicit non-negative expected gain"
            )
        require_positive_int("estimated_pixels", self.estimated_pixels)
        require_finite_number(
            "estimated_seconds",
            self.estimated_seconds,
            minimum=0.0,
            allow_none=True,
        )

        for page in self.page_numbers:
            require_positive_int("page_number", page)
        if len(self.page_numbers) != len(set(self.page_numbers)):
            raise ValueError("page_numbers must not contain duplicates")

        for region in self.regions:
            if not isinstance(region, PageRegion):
                raise ValueError("regions must contain PageRegion")
        for angle in self.rotation_hypotheses:
            require_finite_number("rotation_hypothesis", angle)

        if not self.trigger_reasons:
            raise ValueError("trigger_reasons must be non-empty")
        for reason in self.trigger_reasons:
            require_nonempty_text("trigger_reason", reason)

        if self.scope is OCRScope.DOCUMENT:
            if self.page_numbers or self.regions:
                raise ValueError(
                    "DOCUMENT scope must not include page_numbers or regions"
                )
        elif self.scope is OCRScope.PAGE:
            if not self.page_numbers or self.regions:
                raise ValueError(
                    "PAGE scope requires page_numbers and forbids regions"
                )
        elif self.scope is OCRScope.REGION:
            if not self.regions or self.page_numbers:
                raise ValueError(
                    "REGION scope requires regions and forbids page_numbers"
                )


@dataclass(frozen=True)
class OCRPassOutcome:
    pass_id: str
    executed: bool
    pixels_processed: int
    seconds_elapsed: float
    before: ConfidenceVector
    after: ConfidenceVector
    findings: tuple[str, ...] = ()
    stop_reason: Optional[str] = None
    delta: ConfidenceDelta = field(init=False)

    def __post_init__(self) -> None:
        require_nonempty_text("pass_id", self.pass_id)
        if not isinstance(self.executed, bool):
            raise ValueError("executed must be bool")
        require_nonnegative_int(
            "pixels_processed",
            self.pixels_processed,
        )
        require_finite_number(
            "seconds_elapsed",
            self.seconds_elapsed,
            minimum=0.0,
        )
        if not isinstance(self.before, ConfidenceVector):
            raise ValueError("before must be a ConfidenceVector")
        if not isinstance(self.after, ConfidenceVector):
            raise ValueError("after must be a ConfidenceVector")
        object.__setattr__(
            self,
            "delta",
            self.after.delta_from(self.before),
        )

    @classmethod
    def from_observations(
        cls,
        *,
        pass_id: str,
        executed: bool,
        pixels_processed: int,
        seconds_elapsed: float,
        before: ConfidenceVector,
        after: ConfidenceVector,
        findings: tuple[str, ...] = (),
        stop_reason: Optional[str] = None,
    ) -> "OCRPassOutcome":
        return cls(
            pass_id=pass_id,
            executed=executed,
            pixels_processed=pixels_processed,
            seconds_elapsed=seconds_elapsed,
            before=before,
            after=after,
            findings=findings,
            stop_reason=stop_reason,
        )


@dataclass(frozen=True)
class OCRDecisionRecord:
    decision: OCRDecision
    rationale: tuple[str, ...]
    deficient_dimensions: tuple[ConfidenceDimension, ...] = ()
    hard_validation_findings: tuple[str, ...] = ()
    protected_value_present: bool = False
    remaining_pixel_budget: Optional[int] = None
    remaining_seconds_budget: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.decision, OCRDecision):
            raise ValueError("decision must be an OCRDecision")
        if not self.rationale:
            raise ValueError("OCR decisions require at least one rationale")
        for reason in self.rationale:
            require_nonempty_text("rationale", reason)
        for dimension in self.deficient_dimensions:
            if not isinstance(dimension, ConfidenceDimension):
                raise ValueError(
                    "deficient_dimensions must contain ConfidenceDimension"
                )
        if not isinstance(self.protected_value_present, bool):
            raise ValueError("protected_value_present must be bool")
        if self.remaining_pixel_budget is not None:
            require_nonnegative_int(
                "remaining_pixel_budget",
                self.remaining_pixel_budget,
            )
        require_finite_number(
            "remaining_seconds_budget",
            self.remaining_seconds_budget,
            minimum=0.0,
            allow_none=True,
        )
