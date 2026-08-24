"""C2G 1.7 noncanonical pipeline inspection projection contract."""

from __future__ import annotations

import json
import math

import pytest

from webapp.parser.Context_Integration.context_write_policy import ContextWriteKind
from webapp.parser.contracts.table_pipeline import (
    CompletenessInfo,
    CompletenessState,
    PipelineWarning,
    SourceLocation,
    SourceProvenance,
    TablePipelineResult,
    TableStage,
    TransformationRecord,
    WarningSeverity,
)
from webapp.parser.services.pipeline_inspection import (
    INSPECTION_AUTHORITY,
    INSPECTION_CONTRACT,
    project_pipeline_inspection,
)


def _result() -> TablePipelineResult:
    transformations = (
        TransformationRecord(
            sequence=0,
            from_stage=TableStage.INTERPRETED,
            to_stage=TableStage.INTERPRETED,
            operation="vote_method_header_canonicalization",
            rule_source=(
                "Context_Integration.Context_Library.constants."
                "BALLOT_NAME_CANON_MAP"
            ),
            confidence=None,
            evidence_refs=("fixture://header/2",),
            details={
                "before_header": "election day",
                "after_header": "Election Day",
                "vote_value_mutation": False,
                "confirmed_zero_example": 0,
                "unknown_example": None,
                "signed_example": -4,
            },
        ),
    )

    warnings = (
        PipelineWarning(
            code="EXAMPLE_REVIEW",
            message="Example warning for projection contract.",
            stage=TableStage.INTERPRETED,
            severity=WarningSeverity.WARNING,
            requires_review=True,
            evidence_refs=("fixture://warning/1",),
            details={"missing_value": None, "confirmed_zero": 0},
        ),
    )

    return TablePipelineResult.from_sequences(
        stage=TableStage.INTERPRETED,
        headers=("Precinct", "Election Day"),
        rows=(
            {"Precinct": "P-1", "Election Day": None},
            {"Precinct": "P-2", "Election Day": 0},
            {"Precinct": "P-3", "Election Day": -4},
        ),
        source_provenance=SourceProvenance(
            source_type="csv",
            source_uri="https://example.test/results?secret=not-for-v1",
            source_sha256="a" * 64,
            artifact_id="artifact-1",
            evidence_ref="fixture://source",
            location=SourceLocation(
                page_number=None,
                table_index=0,
                row_index=None,
                column_index=None,
                selector="table.results",
            ),
            metadata={
                "local_path": "C:/not-for-v1.csv",
                "token": "not-for-v1",
            },
        ),
        transformations=transformations,
        warnings=warnings,
        completeness=CompletenessInfo(
            state=CompletenessState.PARTIAL,
            expected_count=4,
            observed_count=3,
            missing_count=1,
            null_value_count=1,
        ),
        write_kind=ContextWriteKind.EVIDENCE,
    )


def test_projection_is_explicitly_noncanonical() -> None:
    payload = project_pipeline_inspection(_result())

    assert payload["contract"] == INSPECTION_CONTRACT == "pipeline_inspection_v1"
    assert payload["authority"] == {
        "inspection": INSPECTION_AUTHORITY,
        "canonical": False,
        "write_kind": "evidence",
    }
    assert payload["stage"] == "interpreted"


def test_projection_excludes_rows_headers_uri_and_source_metadata() -> None:
    payload = project_pipeline_inspection(_result())

    assert payload["rows_included"] is False
    assert payload["headers_included"] is False
    assert "rows" not in payload
    assert "headers" not in payload

    provenance = payload["source_provenance"]
    assert "source_uri" not in provenance
    assert "metadata" not in provenance
    assert provenance["source_uri_included"] is False
    assert provenance["source_metadata_included"] is False

    encoded = json.dumps(payload, sort_keys=True)
    assert "secret=not-for-v1" not in encoded
    assert "C:/not-for-v1.csv" not in encoded
    assert "not-for-v1" not in encoded


def test_true_semantic_before_after_record_is_projected_exactly() -> None:
    payload = project_pipeline_inspection(_result())
    record = payload["transformations"][0]

    assert record["operation"] == "vote_method_header_canonicalization"
    assert record["from_stage"] == "interpreted"
    assert record["to_stage"] == "interpreted"
    assert record["confidence"] is None
    assert record["details"]["before_header"] == "election day"
    assert record["details"]["after_header"] == "Election Day"
    assert record["details"]["vote_value_mutation"] is False


def test_projection_preserves_null_zero_and_signed_values_in_evidence_details() -> None:
    payload = project_pipeline_inspection(_result())
    details = payload["transformations"][0]["details"]

    assert details["unknown_example"] is None
    assert details["confirmed_zero_example"] == 0
    assert details["signed_example"] == -4

    warning_details = payload["warnings"][0]["details"]
    assert warning_details["missing_value"] is None
    assert warning_details["confirmed_zero"] == 0


def test_projection_contains_counts_not_election_rows() -> None:
    payload = project_pipeline_inspection(_result())

    assert payload["summary"] == {
        "header_count": 2,
        "row_count": 3,
        "transformation_count": 1,
        "warning_count": 1,
    }
    assert payload["completeness"]["state"] == "partial"
    assert payload["completeness"]["is_complete"] is False


def test_projection_has_no_automatic_timestamp() -> None:
    payload = project_pipeline_inspection(_result())

    assert payload["automatic_timestamp"] is False
    assert "timestamp" not in payload
    assert "generated_at" not in payload
    assert "created_at" not in payload


@pytest.mark.parametrize(
    "bad_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        object(),
        {1: "non-string-key"},
    ],
)
def test_projection_fails_closed_on_unsupported_json_values(bad_value) -> None:
    result = _result()
    bad_record = TransformationRecord(
        sequence=0,
        from_stage=TableStage.INTERPRETED,
        to_stage=TableStage.INTERPRETED,
        operation="bad_inspection_value",
        details={"bad": bad_value},
    )

    bad_result = TablePipelineResult.from_sequences(
        stage=result.stage,
        headers=result.headers,
        rows=result.rows,
        source_provenance=result.source_provenance,
        transformations=(bad_record,),
        completeness=result.completeness,
        write_kind=result.write_kind,
    )

    with pytest.raises((TypeError, ValueError)):
        project_pipeline_inspection(bad_result)


def test_projection_is_standard_json_serializable() -> None:
    payload = project_pipeline_inspection(_result())

    encoded = json.dumps(payload, allow_nan=False, sort_keys=True)
    decoded = json.loads(encoded)

    assert decoded["contract"] == "pipeline_inspection_v1"
    assert decoded["authority"]["canonical"] is False