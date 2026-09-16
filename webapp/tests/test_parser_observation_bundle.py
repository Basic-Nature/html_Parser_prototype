from __future__ import annotations

from copy import deepcopy
import json

import pytest

from webapp.parser.contracts.table_pipeline import (
    SourceProvenance,
    TablePipelineResult,
    TableStage,
)
from webapp.parser.services import parser_observation_bundle as bundle_module
from webapp.parser.services.election_structure_observation import (
    ELECTION_STRUCTURE_OBSERVATION_CONTRACT,
    project_election_structure_observation,
)
from webapp.parser.services.parser_observation_bundle import (
    PARSER_OBSERVATION_BUNDLE_AUTHORITY,
    PARSER_OBSERVATION_BUNDLE_CONTRACT,
    project_parser_observation_bundle,
)
from webapp.parser.services.pipeline_inspection import (
    INSPECTION_AUTHORITY,
    INSPECTION_CONTRACT,
    project_pipeline_inspection,
)


def _headers():
    return (
        "Precinct",
        "% Precincts Reporting",
        "Election Day Total",
        "Curbside Total",
        "Candidate A - Election Day",
        "Candidate A - Curbside",
        "Candidate A - Total Vote",
        "Candidate B - Election Day",
        "Candidate B - Curbside",
        "Candidate B - Total Vote",
        "Grand Total",
    )


def _row():
    return {
        "Precinct": "P-001",
        "% Precincts Reporting": "100.00%",
        "Election Day Total": 2,
        "Curbside Total": 3,
        "Candidate A - Election Day": 0,
        "Candidate A - Curbside": 3,
        "Candidate A - Total Vote": 3,
        "Candidate B - Election Day": 2,
        "Candidate B - Curbside": 0,
        "Candidate B - Total Vote": 2,
        "Grand Total": 5,
    }


def _result(rows=None):
    return TablePipelineResult.from_sequences(
        stage=TableStage.NORMALIZED,
        headers=_headers(),
        rows=rows or [_row()],
        source_provenance=SourceProvenance(
            source_type="contract_test_fixture",
            source_uri="https://example.invalid/private/source",
            metadata={"private": "must-not-project"},
            evidence_ref="w22x-observation-composition",
        ),
    )


def test_bundle_exactly_composes_existing_projectors_for_same_result():
    result = _result()
    payload = project_parser_observation_bundle(result)

    assert payload["pipeline_inspection"] == project_pipeline_inspection(result)
    assert payload["election_structure"] == project_election_structure_observation(
        result
    )
    assert payload["pipeline_inspection"]["contract"] == INSPECTION_CONTRACT
    assert (
        payload["election_structure"]["contract"]
        == ELECTION_STRUCTURE_OBSERVATION_CONTRACT
    )


def test_bundle_is_noncanonical_deterministic_json_safe_and_timestamp_free():
    result = _result()
    first = project_parser_observation_bundle(result)
    second = project_parser_observation_bundle(result)

    assert first == second
    assert first["contract"] == PARSER_OBSERVATION_BUNDLE_CONTRACT
    assert PARSER_OBSERVATION_BUNDLE_AUTHORITY == INSPECTION_AUTHORITY
    assert first["authority"] == {
        "inspection": INSPECTION_AUTHORITY,
        "canonical": False,
    }
    assert first["source_stage"] == "normalized"
    assert first["raw_rows_included"] is False
    assert first["raw_headers_included"] is False
    assert "rows" not in first
    assert "headers" not in first
    assert first["automatic_timestamp"] is False

    encoded = json.dumps(first, allow_nan=False, sort_keys=True)
    assert json.loads(encoded) == first


def test_bundle_preserves_zero_null_signed_dynamic_and_issue_evidence():
    row = _row()
    row["Candidate A - Curbside"] = None
    row["Candidate B - Election Day"] = -2
    result = _result([row])

    payload = project_parser_observation_bundle(result)
    structure = payload["election_structure"]

    assert structure["vote_methods"] == ["Election Day", "Curbside"]
    assert structure["observed_counts"]["zero"] == 2
    assert structure["observed_counts"]["null"] == 1

    issues = structure["issues"]
    assert "missing_vote_value" in {issue["code"] for issue in issues}
    negatives = [
        issue for issue in issues
        if issue["code"] == "negative_vote_value"
    ]
    assert negatives
    assert negatives[0]["observed"] == -2


def test_bundle_preserves_pipeline_provenance_redaction_contract():
    payload = project_parser_observation_bundle(_result())
    provenance = payload["pipeline_inspection"]["source_provenance"]

    assert provenance["source_uri_included"] is False
    assert provenance["source_metadata_included"] is False
    assert "source_uri" not in provenance
    assert "metadata" not in provenance
    assert "https://example.invalid/private/source" not in repr(payload)
    assert "must-not-project" not in repr(payload)


def test_bundle_does_not_mutate_typed_result():
    result = _result()
    before = deepcopy(result)
    before_rows = tuple(dict(row) for row in result.rows)

    project_parser_observation_bundle(result)

    assert result == before
    assert tuple(dict(row) for row in result.rows) == before_rows


def test_bundle_fails_closed_on_wrong_pipeline_subcontract(monkeypatch):
    monkeypatch.setattr(
        bundle_module,
        "project_pipeline_inspection",
        lambda result: {
            "contract": "wrong_contract",
            "authority": {
                "inspection": INSPECTION_AUTHORITY,
                "canonical": False,
            },
            "rows_included": False,
            "headers_included": False,
            "automatic_timestamp": False,
        },
    )

    with pytest.raises(ValueError, match="contract mismatch"):
        project_parser_observation_bundle(_result())


def test_bundle_fails_closed_on_wrong_election_structure_authority(monkeypatch):
    monkeypatch.setattr(
        bundle_module,
        "project_election_structure_observation",
        lambda result: {
            "contract": ELECTION_STRUCTURE_OBSERVATION_CONTRACT,
            "authority": {
                "inspection": "wrong_authority",
                "canonical": False,
            },
            "raw_rows_included": False,
            "raw_headers_included": False,
            "automatic_timestamp": False,
        },
    )

    with pytest.raises(ValueError, match="authority mismatch"):
        project_parser_observation_bundle(_result())


def test_bundle_rejects_raw_evidence_from_subprojectors(monkeypatch):
    monkeypatch.setattr(
        bundle_module,
        "project_pipeline_inspection",
        lambda result: {
            "contract": INSPECTION_CONTRACT,
            "authority": {
                "inspection": INSPECTION_AUTHORITY,
                "canonical": False,
            },
            "rows_included": True,
            "rows": [{"Precinct": "forbidden"}],
            "headers_included": False,
            "automatic_timestamp": False,
        },
    )

    with pytest.raises(ValueError, match="raw rows"):
        project_parser_observation_bundle(_result())


def test_bundle_requires_table_pipeline_result():
    with pytest.raises(TypeError):
        project_parser_observation_bundle(object())
