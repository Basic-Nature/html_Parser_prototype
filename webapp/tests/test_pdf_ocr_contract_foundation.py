from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest

from webapp.parser.contracts.ocr_strategy import (
    OCRBudget,
    OCRDecision,
    OCRDecisionRecord,
    OCRExecutionParameters,
    OCRPassOutcome,
    OCRPassPlan,
    OCRScope,
)
from webapp.parser.contracts.parser_confidence import (
    CONFIDENCE_DIMENSION_NAMES,
    ConfidenceDelta,
    ConfidenceDimension,
    ConfidenceVector,
)
from webapp.parser.contracts.pdf_structure import (
    BallotLineObservation,
    BoundingBox,
    CellStateObservation,
    ContestSegmentProfile,
    CoordinateSpace,
    ObservedCellState,
    PDFPageProfile,
    PDFStructureProfile,
    PageRegion,
    TextProvider,
)


EXPECTED_DIMENSIONS = (
    "acquisition",
    "text_recognition",
    "geometry",
    "structure",
    "semantics",
    "context",
    "coverage",
    "normalization",
    "reconciliation",
)


def _params() -> OCRExecutionParameters:
    return OCRExecutionParameters(
        engine="tesseract",
        dpi=300,
        preprocessing="gray",
        oem=1,
        psm=6,
        word_confidence_threshold=30.0,
    )


def test_confidence_vector_preserves_dimensions_without_scalar_authority():
    vector = ConfidenceVector(
        acquisition=1.0,
        text_recognition=0.8,
        geometry=0.6,
        structure=0.5,
        semantics=0.4,
        context=0.7,
        coverage=0.3,
        normalization=0.9,
        reconciliation=0.2,
    )
    assert CONFIDENCE_DIMENSION_NAMES == EXPECTED_DIMENSIONS
    assert tuple(vector.as_dict()) == EXPECTED_DIMENSIONS
    assert not hasattr(vector, "overall_confidence")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_confidence_rejects_nonfinite_values(value):
    with pytest.raises(ValueError):
        ConfidenceVector(geometry=value)


def test_confidence_delta_is_signed_and_dimension_specific():
    before = ConfidenceVector(
        text_recognition=0.4,
        geometry=0.5,
        structure=0.7,
    )
    after = ConfidenceVector(
        text_recognition=0.8,
        geometry=0.4,
        structure=0.7,
    )
    delta = after.delta_from(before)
    assert delta.text_recognition == pytest.approx(0.4)
    assert delta.geometry == pytest.approx(-0.1)
    assert delta.structure == pytest.approx(0.0)


def test_geometry_rejects_nonfinite_coordinates():
    with pytest.raises(ValueError):
        BoundingBox(
            0.0,
            0.0,
            float("nan"),
            1.0,
            CoordinateSpace.NORMALIZED,
        )


def test_document_hash_is_exact_sha256_shape():
    page = PDFPageProfile(
        page_number=1,
        width_points=792.0,
        height_points=612.0,
        text_provider=TextProvider.NATIVE,
        text_box_count=10,
    )
    PDFStructureProfile(
        document_sha256="a" * 64,
        page_count=1,
        native_text_page_ratio=1.0,
        page_profiles=(page,),
    )
    with pytest.raises(ValueError):
        PDFStructureProfile(
            document_sha256="abc123",
            page_count=1,
            native_text_page_ratio=1.0,
        )


def test_native_and_ocr_pages_use_same_structure_contract():
    native = PDFPageProfile(
        page_number=1,
        width_points=792.0,
        height_points=612.0,
        text_provider=TextProvider.NATIVE,
        text_box_count=150,
    )
    ocr = PDFPageProfile(
        page_number=2,
        width_points=792.0,
        height_points=612.0,
        text_provider=TextProvider.OCR,
        text_box_count=145,
    )
    profile = PDFStructureProfile(
        document_sha256="b" * 64,
        page_count=2,
        native_text_page_ratio=0.5,
        page_profiles=(native, ocr),
    )
    assert [page.text_provider for page in profile.page_profiles] == [
        TextProvider.NATIVE,
        TextProvider.OCR,
    ]


def test_protected_cell_is_distinct_from_zero_blank_and_unknown():
    states = {
        ObservedCellState.EXPLICIT_ZERO,
        ObservedCellState.BLANK,
        ObservedCellState.PROTECTED,
        ObservedCellState.UNKNOWN,
    }
    assert len(states) == 4
    protected = CellStateObservation(
        page_number=7,
        row_label="Example Precinct",
        column_anchor=412.0,
        state=ObservedCellState.PROTECTED,
        token_text="**",
    )
    assert protected.state is ObservedCellState.PROTECTED


def test_ballot_lines_do_not_force_same_name_into_one_party():
    dem = BallotLineObservation(
        candidate_text="Example Candidate",
        party_text="DEM",
        source_pages=(1,),
    )
    wor = BallotLineObservation(
        candidate_text="Example Candidate",
        party_text="WOR",
        source_pages=(1,),
    )
    segment = ContestSegmentProfile(
        start_page=1,
        end_page=15,
        ballot_line_observations=(dem, wor),
        coverage_fingerprint="coverage-a",
    )
    assert segment.ballot_line_observations[0].party_text == "DEM"
    assert segment.ballot_line_observations[1].party_text == "WOR"


def test_pass_parameters_capture_current_adaptive_search_axes():
    params = _params()
    assert params.engine == "tesseract"
    assert params.dpi == 300
    assert params.preprocessing == "gray"
    assert params.oem == 1
    assert params.psm == 6
    assert params.word_confidence_threshold == 30.0


def test_targeted_region_plan_requires_gain_and_cost_evidence():
    region = PageRegion(
        page_number=4,
        box=BoundingBox(
            0.2, 0.1, 0.8, 0.3,
            CoordinateSpace.NORMALIZED,
        ),
    )
    plan = OCRPassPlan(
        pass_id="pass-2-header-region",
        scope=OCRScope.REGION,
        parameters=_params(),
        target_dimensions=(
            ConfidenceDimension.TEXT_RECOGNITION,
            ConfidenceDimension.GEOMETRY,
        ),
        expected_gain=ConfidenceDelta(
            text_recognition=0.2,
            geometry=0.3,
        ),
        estimated_pixels=1_000_000,
        regions=(region,),
        rotation_hypotheses=(-60.0,),
        trigger_reasons=("header_geometry_low",),
    )
    assert plan.scope is OCRScope.REGION
    assert plan.estimated_pixels == 1_000_000


def test_target_dimension_requires_explicit_expected_gain():
    with pytest.raises(ValueError):
        OCRPassPlan(
            pass_id="bad-plan",
            scope=OCRScope.PAGE,
            parameters=_params(),
            target_dimensions=(ConfidenceDimension.GEOMETRY,),
            expected_gain=ConfidenceDelta(),
            estimated_pixels=1000,
            page_numbers=(1,),
            trigger_reasons=("geometry_low",),
        )


def test_scope_locators_are_coherent():
    with pytest.raises(ValueError):
        OCRPassPlan(
            pass_id="page-without-page",
            scope=OCRScope.PAGE,
            parameters=_params(),
            target_dimensions=(ConfidenceDimension.STRUCTURE,),
            expected_gain=ConfidenceDelta(structure=0.1),
            estimated_pixels=1000,
            trigger_reasons=("structure_low",),
        )

    with pytest.raises(ValueError):
        OCRPassPlan(
            pass_id="document-with-page",
            scope=OCRScope.DOCUMENT,
            parameters=_params(),
            target_dimensions=(ConfidenceDimension.TEXT_RECOGNITION,),
            expected_gain=ConfidenceDelta(text_recognition=0.1),
            estimated_pixels=1000,
            page_numbers=(1,),
            trigger_reasons=("native_text_missing",),
        )


def test_ocr_outcome_derives_delta_and_cannot_accept_conflicting_delta():
    before = ConfidenceVector(text_recognition=0.5, geometry=0.4)
    after = ConfidenceVector(text_recognition=0.8, geometry=0.75)
    outcome = OCRPassOutcome(
        pass_id="targeted-pass",
        executed=True,
        pixels_processed=500_000,
        seconds_elapsed=1.2,
        before=before,
        after=after,
    )
    assert outcome.delta.text_recognition == pytest.approx(0.3)
    assert outcome.delta.geometry == pytest.approx(0.35)

    with pytest.raises(TypeError):
        OCRPassOutcome(
            pass_id="conflicting",
            executed=True,
            pixels_processed=1,
            seconds_elapsed=0.1,
            before=before,
            after=after,
            delta=ConfidenceDelta(geometry=-1.0),
        )


def test_budget_captures_parameter_search_and_processing_caps():
    budget = OCRBudget(
        max_parameter_trials=20,
        max_sample_search_seconds=12.0,
        max_full_page_passes=1,
        max_region_passes=3,
        max_rotation_hypotheses=4,
        max_seconds_per_page=15.0,
        max_seconds_per_document=180.0,
        max_pixels_per_document=20_000_000,
    )
    assert budget.max_parameter_trials == 20
    assert budget.max_sample_search_seconds == 12.0

    with pytest.raises(ValueError):
        OCRBudget(
            max_parameter_trials=20,
            max_sample_search_seconds=float("nan"),
            max_full_page_passes=1,
            max_region_passes=3,
            max_rotation_hypotheses=4,
            max_seconds_per_page=15.0,
            max_seconds_per_document=180.0,
            max_pixels_per_document=20_000_000,
        )


def test_decision_contract_preserves_uncertain_and_budget_terminals():
    assert OCRDecision.HOLD_UNCERTAIN.value == "hold_uncertain"
    assert OCRDecision.BUDGET_EXHAUSTED.value == "budget_exhausted"
    decision = OCRDecisionRecord(
        decision=OCRDecision.HOLD_UNCERTAIN,
        rationale=("confidence_gain_below_cost_floor",),
        deficient_dimensions=(ConfidenceDimension.STRUCTURE,),
    )
    assert decision.decision is OCRDecision.HOLD_UNCERTAIN


def test_contracts_are_immutable_observations():
    vector = ConfidenceVector(geometry=0.5)
    with pytest.raises(FrozenInstanceError):
        vector.geometry = 0.9
