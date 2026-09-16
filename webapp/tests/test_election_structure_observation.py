from __future__ import annotations

from copy import deepcopy
import json

import pytest

from webapp.parser.contracts.election_structure import (
    ElectionStructureAnalysis,
    ElectionStructureIssue,
    ElectionStructureIssueCode,
    ElectionStructureSeverity,
)
from webapp.parser.contracts.table_pipeline import (
    SourceProvenance,
    TablePipelineResult,
    TableStage,
)
from webapp.parser.services.election_structure_analysis import (
    analyze_election_structure,
)
from webapp.parser.services.election_structure_observation import (
    ELECTION_STRUCTURE_OBSERVATION_CONTRACT,
    project_election_structure_analysis,
    project_election_structure_observation,
)
from webapp.parser.services.pipeline_inspection import (
    INSPECTION_AUTHORITY,
    inspection_json_safe_value,
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
            evidence_ref="w22m-observation-projection",
        ),
    )


def test_observation_is_noncanonical_deterministic_and_json_safe():
    payload = project_election_structure_observation(_result())
    assert payload["contract"] == ELECTION_STRUCTURE_OBSERVATION_CONTRACT
    assert payload["authority"] == {
        "inspection": INSPECTION_AUTHORITY,
        "analysis": "noncanonical_parser_evidence",
        "canonical": False,
    }
    assert payload["source_stage"] == "normalized"
    assert payload["candidates"] == ["Candidate A", "Candidate B"]
    assert payload["vote_methods"] == ["Election Day", "Curbside"]
    assert payload["observed_counts"] == {"zero": 2, "null": 0}
    assert payload["reconciliation"] == {
        "candidate_totals_reconciled": True,
        "method_totals_reconciled": True,
        "precinct_totals_reconciled": True,
    }
    assert payload["raw_rows_included"] is False
    assert payload["raw_headers_included"] is False
    assert "rows" not in payload
    assert "headers" not in payload
    assert payload["automatic_timestamp"] is False
    encoded = json.dumps(payload, allow_nan=False, sort_keys=True)
    assert json.loads(encoded) == payload


def test_none_remains_missing_not_zero_in_observation():
    row = _row()
    row["Candidate A - Curbside"] = None
    payload = project_election_structure_observation(_result([row]))
    assert payload["observed_counts"] == {"zero": 2, "null": 1}
    assert payload["reconciliation"] == {
        "candidate_totals_reconciled": None,
        "method_totals_reconciled": None,
        "precinct_totals_reconciled": None,
    }
    assert "missing_vote_value" in {
        issue["code"] for issue in payload["issues"]
    }


def test_signed_negative_evidence_is_preserved_and_flagged():
    row = _row()
    row["Candidate A - Election Day"] = -1
    row["Candidate A - Total Vote"] = 2
    row["Election Day Total"] = 1
    row["Grand Total"] = 4
    result = _result([row])
    payload = project_election_structure_observation(result)
    matching = [
        issue
        for issue in payload["issues"]
        if issue["code"] == "negative_vote_value"
    ]
    assert matching
    assert matching[0]["observed"] == -1
    assert result.rows[0]["Candidate A - Election Day"] == -1
    assert payload["reconciliation"] == {
        "candidate_totals_reconciled": True,
        "method_totals_reconciled": True,
        "precinct_totals_reconciled": True,
    }


def test_dynamic_candidate_method_structure_is_preserved():
    payload = project_election_structure_observation(_result())
    blocks = {block["candidate"]: block for block in payload["candidate_blocks"]}
    assert [item["method"] for item in blocks["Candidate A"]["methods"]] == [
        "Election Day",
        "Curbside",
    ]
    assert blocks["Candidate A"]["total_header"] == "Candidate A - Total Vote"
    assert payload["schema"]["method_total_headers"] == [
        {"method": "Election Day", "header": "Election Day Total"},
        {"method": "Curbside", "header": "Curbside Total"},
    ]


def test_issue_codes_evidence_and_duplicate_precincts_are_preserved():
    row1 = _row()
    row2 = deepcopy(row1)
    payload = project_election_structure_observation(_result([row1, row2]))
    assert payload["summary"]["row_count"] == 2
    assert payload["duplicate_precincts"] == ["P-001"]
    dup = [
        issue for issue in payload["issues"]
        if issue["code"] == "duplicate_precinct"
    ]
    assert dup
    assert dup[0]["precinct"] == "P-001"
    assert dup[0]["details"]["first_row_index"] == 0


def test_projection_does_not_mutate_result_or_analysis():
    result = _result()
    before_rows = tuple(dict(row) for row in result.rows)
    payload = project_election_structure_observation(result)
    assert tuple(dict(row) for row in result.rows) == before_rows

    analysis = analyze_election_structure(result)
    before_analysis = deepcopy(analysis)
    direct = project_election_structure_analysis(analysis)
    assert analysis == before_analysis
    assert direct == payload


def test_shared_inspection_json_safety_authority_fails_closed():
    assert inspection_json_safe_value(None, path="x") is None
    assert inspection_json_safe_value(0, path="x") == 0
    assert inspection_json_safe_value(-4, path="x") == -4

    with pytest.raises(ValueError):
        inspection_json_safe_value(float("nan"), path="x")
    with pytest.raises(TypeError):
        inspection_json_safe_value(object(), path="x")
    with pytest.raises(TypeError):
        inspection_json_safe_value({1: "bad-key"}, path="x")


def test_analysis_projection_fails_closed_on_unsupported_issue_details():
    base = analyze_election_structure(_result())
    bad_issue = ElectionStructureIssue(
        code=ElectionStructureIssueCode.NON_NUMERIC_VOTE_VALUE,
        severity=ElectionStructureSeverity.ERROR,
        message="Synthetic unsupported detail.",
        details={"bad": object()},
    )
    bad = ElectionStructureAnalysis(
        source_stage=base.source_stage,
        row_count=base.row_count,
        precinct_header=base.precinct_header,
        reporting_header=base.reporting_header,
        grand_total_header=base.grand_total_header,
        method_total_headers=base.method_total_headers,
        candidates=base.candidates,
        vote_methods=base.vote_methods,
        candidate_blocks=base.candidate_blocks,
        duplicate_precincts=base.duplicate_precincts,
        issues=base.issues + (bad_issue,),
        candidate_method_matrix_complete=base.candidate_method_matrix_complete,
        schema_complete=base.schema_complete,
        candidate_totals_reconciled=base.candidate_totals_reconciled,
        method_totals_reconciled=base.method_totals_reconciled,
        precinct_totals_reconciled=base.precinct_totals_reconciled,
        observed_zero_count=base.observed_zero_count,
        observed_null_count=base.observed_null_count,
    )
    with pytest.raises(TypeError):
        project_election_structure_analysis(bad)
