from __future__ import annotations

import pytest

from webapp.parser.contracts.pdf_structure import (
    BallotLineObservation,
    CellStateObservation,
    ContestSegmentProfile,
    ObservedCellState,
    PDFPageProfile,
    StructureFinding,
    StructureFindingKind,
    TextProvider,
)
from webapp.parser.profiling.pdf_structure_profiler import (
    PDFStructureProfileAccumulator,
    PageTextEvidence,
    StructureObservationPhase,
)


DOC_SHA = "a" * 64


def test_partial_snapshot_never_claims_completion():
    accumulator = PDFStructureProfileAccumulator(DOC_SHA, 3)
    accumulator.mark_phase(
        StructureObservationPhase.PAGE_TEXT_STRUCTURE
    )
    accumulator.observe_page_text(
        PageTextEvidence(
            page_number=1,
            text_provider=TextProvider.NATIVE,
            char_count=120,
            line_count=8,
        )
    )
    snapshot = accumulator.snapshot()
    assert not hasattr(snapshot, "is_complete")
    assert snapshot.observed_phases == (
        StructureObservationPhase.PAGE_TEXT_STRUCTURE,
    )
    assert snapshot.missing_page_profiles == (1, 2, 3)
    assert snapshot.profile.native_text_page_ratio is None


def test_native_text_ratio_is_not_inferred_from_page_text():
    accumulator = PDFStructureProfileAccumulator(DOC_SHA, 2)
    accumulator.observe_page_text(
        PageTextEvidence(
            page_number=1,
            text_provider=TextProvider.NATIVE,
            char_count=50,
            line_count=4,
        )
    )
    accumulator.observe_page_text(
        PageTextEvidence(
            page_number=2,
            text_provider=TextProvider.OCR,
            char_count=80,
            line_count=5,
        )
    )
    assert accumulator.snapshot().profile.native_text_page_ratio is None
    accumulator.set_native_text_page_ratio(0.5)
    assert accumulator.snapshot().profile.native_text_page_ratio == 0.5


def test_conflicting_native_text_ratio_is_rejected():
    accumulator = PDFStructureProfileAccumulator(DOC_SHA, 2)
    accumulator.set_native_text_page_ratio(1.0)
    with pytest.raises(ValueError):
        accumulator.set_native_text_page_ratio(0.5)


def test_known_native_text_ratio_cannot_be_cleared_to_unknown():
    accumulator = PDFStructureProfileAccumulator(DOC_SHA, 2)
    accumulator.set_native_text_page_ratio(1.0)

    with pytest.raises(ValueError):
        accumulator.set_native_text_page_ratio(None)

    assert accumulator.snapshot().profile.native_text_page_ratio == 1.0


def test_page_text_is_provenance_not_fake_geometry():
    evidence = PageTextEvidence(
        page_number=7,
        text_provider=TextProvider.NATIVE,
        char_count=900,
        line_count=31,
    )
    assert not hasattr(evidence, "width_points")
    assert not hasattr(evidence, "height_points")
    assert not hasattr(evidence, "header_angle_hypotheses")


def test_rotated_geometry_can_be_recorded_without_acquisition():
    accumulator = PDFStructureProfileAccumulator(DOC_SHA, 1)
    accumulator.mark_phase(StructureObservationPhase.GEOMETRY)
    page = PDFPageProfile(
        page_number=1,
        width_points=792.0,
        height_points=612.0,
        text_provider=TextProvider.NATIVE,
        text_box_count=150,
        header_angle_hypotheses=(-60.0,),
        column_anchors=(50.0, 110.0, 170.0),
    )
    accumulator.observe_page_profile(page)
    snapshot = accumulator.snapshot()
    assert snapshot.profile.page_profiles == (page,)
    assert snapshot.missing_page_profiles == ()
    assert snapshot.profile.page_profiles[0].header_angle_hypotheses == (
        -60.0,
    )


def test_conflicting_page_profile_is_not_silently_merged():
    accumulator = PDFStructureProfileAccumulator(DOC_SHA, 1)
    first = PDFPageProfile(
        page_number=1,
        width_points=792.0,
        height_points=612.0,
        text_provider=TextProvider.NATIVE,
        text_box_count=100,
    )
    conflicting = PDFPageProfile(
        page_number=1,
        width_points=792.0,
        height_points=612.0,
        text_provider=TextProvider.NATIVE,
        text_box_count=101,
    )
    accumulator.observe_page_profile(first)
    with pytest.raises(ValueError):
        accumulator.observe_page_profile(conflicting)


def test_duplicate_identical_page_profile_is_idempotent():
    accumulator = PDFStructureProfileAccumulator(DOC_SHA, 1)
    page = PDFPageProfile(
        page_number=1,
        width_points=792.0,
        height_points=612.0,
        text_provider=TextProvider.NATIVE,
        text_box_count=100,
    )
    accumulator.observe_page_profile(page)
    accumulator.observe_page_profile(page)
    assert accumulator.snapshot().profile.page_profiles == (page,)


def test_fusion_ballot_lines_remain_separate_observations():
    accumulator = PDFStructureProfileAccumulator(DOC_SHA, 15)
    segment = ContestSegmentProfile(
        start_page=1,
        end_page=15,
        title_observation="Example Contest",
        ballot_line_observations=(
            BallotLineObservation(
                candidate_text="Example Candidate",
                party_text="DEM",
                source_pages=(1,),
            ),
            BallotLineObservation(
                candidate_text="Example Candidate",
                party_text="WOR",
                source_pages=(1,),
            ),
        ),
    )
    accumulator.observe_contest_segment(segment)
    result = accumulator.snapshot().profile.contest_segments[0]
    assert len(result.ballot_line_observations) == 2
    assert result.ballot_line_observations[0].party_text == "DEM"
    assert result.ballot_line_observations[1].party_text == "WOR"


def test_cell_states_are_preserved_without_reconciliation():
    accumulator = PDFStructureProfileAccumulator(DOC_SHA, 7)
    observations = (
        CellStateObservation(
            page_number=7,
            row_label="Precinct A",
            column_anchor=412.0,
            state=ObservedCellState.PROTECTED,
            token_text="**",
        ),
        CellStateObservation(
            page_number=7,
            row_label="Precinct A",
            column_anchor=412.0,
            state=ObservedCellState.EXPLICIT_ZERO,
            token_text="0",
        ),
        CellStateObservation(
            page_number=7,
            row_label="Precinct B",
            column_anchor=412.0,
            state=ObservedCellState.BLANK,
            token_text=None,
        ),
        CellStateObservation(
            page_number=7,
            row_label="Precinct C",
            column_anchor=412.0,
            state=ObservedCellState.UNKNOWN,
            token_text=None,
        ),
    )
    for observation in observations:
        accumulator.observe_cell_state(observation)
    states = {
        item.state
        for item in accumulator.snapshot().profile.cell_state_observations
    }
    assert states == {
        ObservedCellState.PROTECTED,
        ObservedCellState.EXPLICIT_ZERO,
        ObservedCellState.BLANK,
        ObservedCellState.UNKNOWN,
    }


def test_structural_findings_and_coverage_remain_observations():
    accumulator = PDFStructureProfileAccumulator(DOC_SHA, 3)
    finding = StructureFinding(
        kind=StructureFindingKind.CONTINUATION_FRAGMENT,
        page_numbers=(2, 3),
        detail="header continues across pages",
        confidence=0.8,
    )
    accumulator.observe_structural_finding(finding)
    accumulator.observe_coverage_fingerprint("coverage-family-a")
    profile = accumulator.snapshot().profile
    assert profile.structural_findings == (finding,)
    assert profile.coverage_fingerprints == ("coverage-family-a",)


def test_phase_marking_is_explicit_unique_and_ordered():
    accumulator = PDFStructureProfileAccumulator(DOC_SHA, 1)
    accumulator.mark_phase(
        StructureObservationPhase.PAGE_TEXT_STRUCTURE
    )
    accumulator.mark_phase(StructureObservationPhase.GEOMETRY)
    accumulator.mark_phase(StructureObservationPhase.GEOMETRY)
    accumulator.mark_phase(
        StructureObservationPhase.CONTEST_HINT_STRUCTURE
    )
    assert accumulator.snapshot().observed_phases == (
        StructureObservationPhase.PAGE_TEXT_STRUCTURE,
        StructureObservationPhase.GEOMETRY,
        StructureObservationPhase.CONTEST_HINT_STRUCTURE,
    )


def test_out_of_document_observations_are_rejected():
    accumulator = PDFStructureProfileAccumulator(DOC_SHA, 1)
    with pytest.raises(ValueError):
        accumulator.observe_page_text(
            PageTextEvidence(
                page_number=2,
                text_provider=TextProvider.NATIVE,
                char_count=1,
                line_count=1,
            )
        )
    with pytest.raises(ValueError):
        accumulator.observe_page_profile(
            PDFPageProfile(
                page_number=2,
                width_points=792.0,
                height_points=612.0,
                text_provider=TextProvider.NATIVE,
                text_box_count=1,
            )
        )


def test_snapshot_order_is_deterministic_without_merging_semantics():
    accumulator = PDFStructureProfileAccumulator(DOC_SHA, 3)
    accumulator.observe_page_text(
        PageTextEvidence(
            page_number=3,
            text_provider=TextProvider.NATIVE,
            char_count=30,
            line_count=3,
        )
    )
    accumulator.observe_page_text(
        PageTextEvidence(
            page_number=1,
            text_provider=TextProvider.NATIVE,
            char_count=10,
            line_count=1,
        )
    )
    snapshot = accumulator.snapshot()
    assert [item.page_number for item in snapshot.page_text_evidence] == [1, 3]
    assert snapshot.missing_page_profiles == (1, 2, 3)


def test_invalid_document_hash_is_rejected_by_frozen_contract():
    with pytest.raises(ValueError):
        PDFStructureProfileAccumulator("not-a-sha256", 1)
