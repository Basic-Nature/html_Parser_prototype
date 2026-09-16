from __future__ import annotations

from copy import deepcopy

from webapp.parser.contracts.election_structure import (
    ElectionStructureIssueCode,
    STRUCTURE_AUTHORITY,
)
from webapp.parser.contracts.table_pipeline import (
    SourceProvenance,
    TablePipelineResult,
    TableStage,
)
from webapp.parser.services.election_structure_analysis import analyze_election_structure


def _result(headers, rows):
    return TablePipelineResult.from_sequences(
        stage=TableStage.NORMALIZED,
        headers=headers,
        rows=rows,
        source_provenance=SourceProvenance(
            source_type="contract_test_fixture",
            evidence_ref="w22b-structure-contract",
        ),
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


def test_dynamic_method_zero_and_totals_reconcile():
    analysis = analyze_election_structure(_result(_headers(), [_row()]))
    assert analysis.authority == STRUCTURE_AUTHORITY
    assert analysis.candidates == ("Candidate A", "Candidate B")
    assert "Election Day" in analysis.vote_methods
    assert "Curbside" in analysis.vote_methods
    assert analysis.candidate_method_matrix_complete is True
    assert analysis.schema_complete is True
    assert analysis.candidate_totals_reconciled is True
    assert analysis.method_totals_reconciled is True
    assert analysis.precinct_totals_reconciled is True
    assert analysis.observed_zero_count == 2
    assert analysis.observed_null_count == 0


def test_none_is_missing_not_zero():
    row = _row()
    row["Candidate A - Curbside"] = None
    analysis = analyze_election_structure(_result(_headers(), [row]))
    assert analysis.observed_null_count == 1
    assert analysis.observed_zero_count == 2
    assert ElectionStructureIssueCode.MISSING_VOTE_VALUE in analysis.issue_codes
    assert analysis.candidate_totals_reconciled is None
    assert analysis.method_totals_reconciled is None
    assert analysis.precinct_totals_reconciled is None


def test_cross_candidate_method_gap_is_explicit():
    headers = tuple(h for h in _headers() if h != "Candidate B - Curbside")
    row = _row()
    row.pop("Candidate B - Curbside")
    analysis = analyze_election_structure(_result(headers, [row]))
    assert analysis.candidate_method_matrix_complete is False
    assert analysis.schema_complete is False
    assert ElectionStructureIssueCode.MISSING_METHOD_COLUMN in analysis.issue_codes


def test_reconciliation_mismatches_are_separate():
    row = _row()
    row["Candidate A - Total Vote"] = 99
    row["Election Day Total"] = 77
    row["Grand Total"] = 88
    analysis = analyze_election_structure(_result(_headers(), [row]))
    assert analysis.candidate_totals_reconciled is False
    assert analysis.method_totals_reconciled is False
    assert analysis.precinct_totals_reconciled is False
    assert ElectionStructureIssueCode.CANDIDATE_TOTAL_MISMATCH in analysis.issue_codes
    assert ElectionStructureIssueCode.METHOD_TOTAL_MISMATCH in analysis.issue_codes
    assert ElectionStructureIssueCode.PRECINCT_TOTAL_MISMATCH in analysis.issue_codes


def test_duplicate_precinct_flagged_without_dedup():
    row1 = _row()
    row2 = deepcopy(row1)
    analysis = analyze_election_structure(_result(_headers(), [row1, row2]))
    assert analysis.row_count == 2
    assert analysis.duplicate_precincts == ("P-001",)
    assert ElectionStructureIssueCode.DUPLICATE_PRECINCT in analysis.issue_codes


def test_analysis_does_not_mutate_input():
    headers, rows = _headers(), [_row()]
    before_headers, before_rows = deepcopy(headers), deepcopy(rows)
    result = _result(headers, rows)
    typed_headers = tuple(result.headers)
    typed_rows = tuple(dict(r) for r in result.rows)
    analyze_election_structure(result)
    assert headers == before_headers
    assert rows == before_rows
    assert result.headers == typed_headers
    assert tuple(dict(r) for r in result.rows) == typed_rows


def test_negative_signed_evidence_preserved_and_flagged():
    row = _row()
    row["Candidate A - Election Day"] = -1
    row["Candidate A - Total Vote"] = 2
    row["Election Day Total"] = 1
    row["Grand Total"] = 4
    result = _result(_headers(), [row])
    analysis = analyze_election_structure(result)
    assert result.rows[0]["Candidate A - Election Day"] == -1
    assert ElectionStructureIssueCode.NEGATIVE_VOTE_VALUE in analysis.issue_codes
    assert analysis.candidate_totals_reconciled is True
    assert analysis.method_totals_reconciled is True
    assert analysis.precinct_totals_reconciled is True


def test_missing_method_total_is_schema_gap():
    headers = tuple(h for h in _headers() if h != "Curbside Total")
    row = _row()
    row.pop("Curbside Total")
    analysis = analyze_election_structure(_result(headers, [row]))
    assert analysis.schema_complete is False
    assert ElectionStructureIssueCode.MISSING_METHOD_TOTAL_COLUMN in analysis.issue_codes
    assert analysis.method_totals_reconciled is None
