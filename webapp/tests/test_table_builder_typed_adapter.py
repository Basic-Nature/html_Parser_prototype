"""C2G 1.2 typed adapter around build_table_noninteractive."""

from __future__ import annotations

import inspect

import pytest

from webapp.parser.Context_Integration.context_write_policy import ContextWriteKind
from webapp.parser.contracts.table_pipeline import (
    CompletenessState,
    TablePipelineResult,
    TableStage,
)
from webapp.parser.utils import table_builder


def _legacy_fixture_result():
    headers = [
        "Precinct",
        "Example Candidate - Election Day",
        "Example Candidate - Absentee Mail",
        "Example Candidate - Total Votes",
    ]
    rows = [
        {
            "Precinct": "P-1",
            "Example Candidate - Election Day": 0,
            "Example Candidate - Absentee Mail": None,
            "Example Candidate - Total Votes": -4,
        }
    ]
    entity_info = {
        "candidate_columns": ["Example Candidate"],
        "null_reason_inferred": False,
    }
    return headers, rows, entity_info


def test_typed_adapter_is_additive_and_legacy_signature_remains_available() -> None:
    legacy_signature = inspect.signature(table_builder.build_table_noninteractive)
    typed_signature = inspect.signature(table_builder.build_table_noninteractive_result)

    assert "source_type" not in legacy_signature.parameters
    assert "source_type" in typed_signature.parameters
    assert typed_signature.parameters["source_type"].default is inspect.Parameter.empty


def test_typed_adapter_delegates_to_legacy_boundary_and_preserves_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_headers, expected_rows, expected_entity_info = _legacy_fixture_result()
    calls = []

    def fake_legacy(**kwargs):
        calls.append(kwargs)
        return expected_headers, expected_rows, expected_entity_info

    monkeypatch.setattr(
        table_builder,
        "build_table_noninteractive",
        fake_legacy,
    )

    result = table_builder.build_table_noninteractive_result(
        domain="example.gov",
        headers=["ignored-input-header"],
        data=[{"ignored-input-header": "ignored-input-value"}],
        coordinator=None,
        context={"session_id": "c2g-test"},
        pivot_to_wide=True,
        debug=False,
        source_type="html",
        source_uri="https://example.gov/results",
        source_sha256="b" * 64,
        artifact_id="artifact-123",
        evidence_ref="fixture://typed-adapter",
    )

    assert isinstance(result, TablePipelineResult)
    assert len(calls) == 1

    call = calls[0]
    assert call["domain"] == "example.gov"
    assert call["headers"] == ["ignored-input-header"]
    assert call["data"] == [{"ignored-input-header": "ignored-input-value"}]
    assert call["context"] == {"session_id": "c2g-test"}
    assert call["pivot_to_wide"] is True
    assert call["debug"] is False

    assert result.stage is TableStage.INTERPRETED
    assert list(result.headers) == expected_headers
    assert [dict(row) for row in result.rows] == expected_rows
    assert result.semantic_annotations["entity_info"] == expected_entity_info

    # The typed boundary must preserve the election-value distinctions exactly.
    assert result.rows[0]["Example Candidate - Election Day"] == 0
    assert result.rows[0]["Example Candidate - Absentee Mail"] is None
    assert result.rows[0]["Example Candidate - Total Votes"] == -4


def test_typed_adapter_does_not_claim_completeness_or_write_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        table_builder,
        "build_table_noninteractive",
        lambda **kwargs: _legacy_fixture_result(),
    )

    result = table_builder.build_table_noninteractive_result(
        domain="example.gov",
        headers=[],
        data=[],
        source_type="json",
    )

    assert result.completeness.state is CompletenessState.UNKNOWN
    assert result.completeness.expected_count is None
    assert result.completeness.observed_count is None
    assert result.completeness.missing_count is None
    assert result.completeness.is_complete is None

    assert result.write_kind is ContextWriteKind.NONE
    assert result.write_kind is not ContextWriteKind.CANONICAL


def test_typed_adapter_provenance_is_caller_declared_not_invented(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        table_builder,
        "build_table_noninteractive",
        lambda **kwargs: _legacy_fixture_result(),
    )

    result = table_builder.build_table_noninteractive_result(
        domain="county.example.gov",
        headers=[],
        data=[],
        source_type="csv",
        source_uri="file:///fixture/results.csv",
        evidence_ref="fixture://results.csv",
    )

    provenance = result.source_provenance
    assert provenance.source_type == "csv"
    assert provenance.source_uri == "file:///fixture/results.csv"
    assert provenance.evidence_ref == "fixture://results.csv"
    assert provenance.source_sha256 is None
    assert provenance.artifact_id is None
    assert provenance.metadata["domain"] == "county.example.gov"
    assert provenance.metadata["legacy_boundary"] == "build_table_noninteractive"


def test_typed_adapter_requires_explicit_source_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        table_builder,
        "build_table_noninteractive",
        lambda **kwargs: _legacy_fixture_result(),
    )

    with pytest.raises(TypeError):
        table_builder.build_table_noninteractive_result(  # type: ignore[call-arg]
            domain="example.gov",
            headers=[],
            data=[],
        )


def test_typed_adapter_does_not_add_serialization_or_timestamp_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        table_builder,
        "build_table_noninteractive",
        lambda **kwargs: _legacy_fixture_result(),
    )

    result = table_builder.build_table_noninteractive_result(
        domain="example.gov",
        headers=[],
        data=[],
        source_type="xlsx",
    )

    assert not hasattr(result, "timestamp")
    assert not hasattr(result, "csv_path")
    assert result.rows[0]["Example Candidate - Absentee Mail"] is None