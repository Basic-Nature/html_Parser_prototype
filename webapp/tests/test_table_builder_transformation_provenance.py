"""C2G 1.4 first factual TransformationRecord provenance contract."""

from __future__ import annotations

import pytest

from webapp.parser.contracts.table_pipeline import (
    TableStage,
    TransformationRecord,
)
from webapp.parser.utils import table_builder


def _legacy_result():
    headers = [
        "Precinct",
        "Candidate - Election Day",
        "Candidate - Absentee Mail",
        "Candidate - Total Votes",
    ]
    rows = [
        {
            "Precinct": "P-1",
            "Candidate - Election Day": 0,
            "Candidate - Absentee Mail": None,
            "Candidate - Total Votes": -4,
        }
    ]
    entity_info = {"candidate_columns": ["Candidate"]}
    return headers, rows, entity_info


def test_typed_adapter_emits_one_factual_same_stage_boundary_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        table_builder,
        "build_table_noninteractive",
        lambda **kwargs: _legacy_result(),
    )

    result = table_builder.build_table_noninteractive_result(
        domain="example.gov",
        headers=[],
        data=[],
        pivot_to_wide=True,
        source_type="csv",
        evidence_ref="fixture://csv/primary",
    )

    assert len(result.transformations) == 1

    record = result.transformations[0]
    assert isinstance(record, TransformationRecord)
    assert record.sequence == 0
    assert record.from_stage is TableStage.INTERPRETED
    assert record.to_stage is TableStage.INTERPRETED
    assert record.operation == "typed_boundary_adaptation"
    assert record.rule_source == "table_builder.build_table_noninteractive_result"
    assert record.confidence is None
    assert record.evidence_refs == ("fixture://csv/primary",)

    assert record.details == {
        "adapter": "build_table_noninteractive_result",
        "legacy_boundary": "build_table_noninteractive",
        "source_type": "csv",
        "domain": "example.gov",
        "pivot_to_wide": True,
        "semantic_value_mutation": False,
    }


def test_boundary_record_does_not_claim_election_semantic_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        table_builder,
        "build_table_noninteractive",
        lambda **kwargs: _legacy_result(),
    )

    result = table_builder.build_table_noninteractive_result(
        domain="example.gov",
        headers=[],
        data=[],
        pivot_to_wide=False,
        source_type="xlsx",
    )

    record = result.transformations[0]

    assert record.confidence is None
    assert record.evidence_refs == ()
    assert record.details["semantic_value_mutation"] is False
    assert record.details["pivot_to_wide"] is False
    assert record.details["source_type"] == "xlsx"


def test_transformation_provenance_does_not_change_null_zero_or_signed_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        table_builder,
        "build_table_noninteractive",
        lambda **kwargs: _legacy_result(),
    )

    result = table_builder.build_table_noninteractive_result(
        domain="example.gov",
        headers=[],
        data=[],
        source_type="csv",
    )

    row = result.rows[0]
    assert row["Candidate - Election Day"] == 0
    assert row["Candidate - Absentee Mail"] is None
    assert row["Candidate - Total Votes"] == -4


def test_transformation_record_has_no_automatic_timestamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        table_builder,
        "build_table_noninteractive",
        lambda **kwargs: _legacy_result(),
    )

    result = table_builder.build_table_noninteractive_result(
        domain="example.gov",
        headers=[],
        data=[],
        source_type="csv",
    )

    assert not hasattr(result.transformations[0], "timestamp")