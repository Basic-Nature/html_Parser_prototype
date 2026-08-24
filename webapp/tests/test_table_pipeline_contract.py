"""C2G 1.1 behavior-neutral contracts for the parser table pipeline."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from webapp.parser.Context_Integration.context_write_policy import ContextWriteKind
from webapp.parser.contracts.table_pipeline import (
    CompletenessInfo,
    CompletenessState,
    GovernedBoundary,
    PipelineWarning,
    SourceLocation,
    SourceProvenance,
    TablePipelineResult,
    TableStage,
    TransformationRecord,
    WarningSeverity,
    is_forward_stage_transition,
)


def _provenance() -> SourceProvenance:
    return SourceProvenance(
        source_type="test_fixture",
        source_uri="fixture://c2g/null-zero-signed",
        source_sha256="a" * 64,
        location=SourceLocation(page_number=None, row_index=0),
    )


def test_canonical_is_governed_boundary_not_table_stage() -> None:
    assert GovernedBoundary.CANONICAL.value == "canonical"
    assert "canonical" not in {stage.value for stage in TableStage}
    assert [stage.value for stage in TableStage] == [
        "extracted",
        "normalized",
        "interpreted",
        "validated",
        "learned",
    ]


def test_stage_transition_is_forward_only() -> None:
    assert is_forward_stage_transition(
        TableStage.EXTRACTED,
        TableStage.EXTRACTED,
    )
    assert is_forward_stage_transition(
        TableStage.EXTRACTED,
        TableStage.NORMALIZED,
    )
    assert is_forward_stage_transition(
        TableStage.INTERPRETED,
        TableStage.VALIDATED,
    )
    assert not is_forward_stage_transition(
        TableStage.VALIDATED,
        TableStage.INTERPRETED,
    )


def test_pipeline_result_preserves_null_zero_and_signed_values() -> None:
    row = {
        "unknown_votes": None,
        "confirmed_zero_votes": 0,
        "signed_adjustment": -4,
        "candidate": "Example Candidate",
    }

    result = TablePipelineResult.from_sequences(
        stage=TableStage.EXTRACTED,
        headers=list(row),
        rows=[row],
        source_provenance=_provenance(),
    )

    assert result.rows[0]["unknown_votes"] is None
    assert result.rows[0]["confirmed_zero_votes"] == 0
    assert result.rows[0]["signed_adjustment"] == -4
    assert result.rows[0]["candidate"] == "Example Candidate"


def test_unknown_completeness_remains_unknown_not_zero() -> None:
    completeness = CompletenessInfo()

    assert completeness.state is CompletenessState.UNKNOWN
    assert completeness.expected_count is None
    assert completeness.observed_count is None
    assert completeness.missing_count is None
    assert completeness.null_value_count is None
    assert completeness.is_complete is None


def test_complete_and_partial_completeness_are_explicit() -> None:
    complete = CompletenessInfo(
        state=CompletenessState.COMPLETE,
        expected_count=4,
        observed_count=4,
        missing_count=0,
        null_value_count=0,
    )
    partial = CompletenessInfo(
        state=CompletenessState.PARTIAL,
        expected_count=4,
        observed_count=3,
        missing_count=1,
        null_value_count=1,
    )

    assert complete.is_complete is True
    assert partial.is_complete is False


def test_invalid_completeness_contracts_fail_closed() -> None:
    with pytest.raises(ValueError):
        CompletenessInfo(
            state=CompletenessState.COMPLETE,
            missing_count=1,
        )

    with pytest.raises(ValueError):
        CompletenessInfo(
            state=CompletenessState.PARTIAL,
            missing_count=0,
        )

    with pytest.raises(ValueError):
        CompletenessInfo(observed_count=-1)


def test_transformation_records_are_explainable_and_forward_only() -> None:
    record = TransformationRecord(
        sequence=0,
        from_stage=TableStage.EXTRACTED,
        to_stage=TableStage.NORMALIZED,
        operation="merge_page_spanning_header",
        rule_source="shared_structural_normalization",
        confidence=0.97,
        evidence_refs=("fixture://page/1", "fixture://page/2"),
        details={"break_sensitive": True},
    )

    assert record.operation == "merge_page_spanning_header"
    assert record.confidence == 0.97
    assert record.details["break_sensitive"] is True

    with pytest.raises(ValueError):
        TransformationRecord(
            sequence=1,
            from_stage=TableStage.VALIDATED,
            to_stage=TableStage.INTERPRETED,
            operation="illegal_backward_transition",
        )

    with pytest.raises(ValueError):
        TransformationRecord(
            sequence=1,
            from_stage=TableStage.EXTRACTED,
            to_stage=TableStage.NORMALIZED,
            operation="bad_confidence",
            confidence=1.1,
        )


def test_pipeline_warning_carries_review_state_without_mutating_data() -> None:
    warning = PipelineWarning(
        code="MISSING_ABSENTEE_MAIL",
        message="Source did not establish an absentee-mail value.",
        stage=TableStage.INTERPRETED,
        severity=WarningSeverity.WARNING,
        requires_review=True,
        evidence_refs=("fixture://row/4",),
    )

    assert warning.requires_review is True
    assert warning.stage is TableStage.INTERPRETED
    assert warning.code == "MISSING_ABSENTEE_MAIL"


def test_canonical_write_kind_is_always_rejected_from_pipeline_result() -> None:
    with pytest.raises(PermissionError, match="separate governed authority boundary"):
        TablePipelineResult.from_sequences(
            stage=TableStage.VALIDATED,
            headers=["Precinct"],
            rows=[{"Precinct": "P-1"}],
            source_provenance=_provenance(),
            write_kind=ContextWriteKind.CANONICAL,
        )


def test_learned_stage_does_not_imply_canonical_write_authority() -> None:
    result = TablePipelineResult.from_sequences(
        stage=TableStage.LEARNED,
        headers=["Precinct"],
        rows=[{"Precinct": "P-1"}],
        source_provenance=_provenance(),
        write_kind=ContextWriteKind.LEARNED,
    )

    assert result.stage is TableStage.LEARNED
    assert result.write_kind is ContextWriteKind.LEARNED
    assert result.write_kind is not ContextWriteKind.CANONICAL


def test_transformation_history_cannot_advance_beyond_result_stage() -> None:
    future_transform = TransformationRecord(
        sequence=0,
        from_stage=TableStage.EXTRACTED,
        to_stage=TableStage.VALIDATED,
        operation="future_validation",
    )

    with pytest.raises(ValueError, match="beyond result stage"):
        TablePipelineResult.from_sequences(
            stage=TableStage.NORMALIZED,
            headers=["Precinct"],
            rows=[{"Precinct": "P-1"}],
            source_provenance=_provenance(),
            transformations=[future_transform],
        )


def test_contract_records_are_frozen_at_top_level() -> None:
    result = TablePipelineResult.from_sequences(
        stage=TableStage.EXTRACTED,
        headers=["Precinct"],
        rows=[{"Precinct": "P-1"}],
        source_provenance=_provenance(),
    )

    with pytest.raises(FrozenInstanceError):
        result.stage = TableStage.NORMALIZED  # type: ignore[misc]


def test_contract_constructor_does_not_add_timestamp_or_serialize_null() -> None:
    result = TablePipelineResult.from_sequences(
        stage=TableStage.EXTRACTED,
        headers=["Votes"],
        rows=[{"Votes": None}],
        source_provenance=_provenance(),
    )

    assert not hasattr(result, "timestamp")
    assert result.rows[0]["Votes"] is None