"""Pure JSON-safe observation projection for election-structure analysis.

This service is noncanonical and behavior-neutral. It does not expose raw
TablePipelineResult rows or the raw header sequence, does not persist or publish
data, and does not create handler/runtime wiring.
"""
from __future__ import annotations

from typing import Any

from webapp.parser.contracts.election_structure import (
    CandidateMethodBlock,
    ElectionStructureAnalysis,
    ElectionStructureIssue,
    STRUCTURE_AUTHORITY,
)
from webapp.parser.contracts.table_pipeline import TablePipelineResult
from webapp.parser.services.election_structure_analysis import (
    analyze_election_structure,
)
from webapp.parser.services.pipeline_inspection import (
    INSPECTION_AUTHORITY,
    inspection_json_safe_value,
)

ELECTION_STRUCTURE_OBSERVATION_CONTRACT = "election_structure_observation_v1"


def _project_block(block: CandidateMethodBlock) -> dict[str, Any]:
    return {
        "candidate": block.candidate,
        "methods": [
            {"method": method, "header": header}
            for method, header in block.method_headers
        ],
        "total_header": block.total_header,
        "party_header": block.party_header,
        "percent_header": block.percent_header,
    }


def _project_issue(
    issue: ElectionStructureIssue,
    *,
    index: int,
) -> dict[str, Any]:
    return {
        "code": issue.code.value,
        "severity": issue.severity.value,
        "message": issue.message,
        "row_index": issue.row_index,
        "precinct": issue.precinct,
        "candidate": issue.candidate,
        "method": issue.method,
        "header": issue.header,
        "observed": inspection_json_safe_value(
            issue.observed,
            path=f"issues[{index}].observed",
        ),
        "expected": inspection_json_safe_value(
            issue.expected,
            path=f"issues[{index}].expected",
        ),
        "details": inspection_json_safe_value(
            issue.details,
            path=f"issues[{index}].details",
        ),
    }


def project_election_structure_analysis(
    analysis: ElectionStructureAnalysis,
) -> dict[str, Any]:
    """Project typed structure analysis to deterministic noncanonical evidence."""
    if not isinstance(analysis, ElectionStructureAnalysis):
        raise TypeError("analysis must be an ElectionStructureAnalysis")
    if analysis.authority != STRUCTURE_AUTHORITY:
        raise ValueError("analysis authority must remain noncanonical")

    blocks = [_project_block(block) for block in analysis.candidate_blocks]
    issues = [
        _project_issue(issue, index=index)
        for index, issue in enumerate(analysis.issues)
    ]

    return {
        "contract": ELECTION_STRUCTURE_OBSERVATION_CONTRACT,
        "authority": {
            "inspection": INSPECTION_AUTHORITY,
            "analysis": analysis.authority,
            "canonical": False,
        },
        "source_stage": analysis.source_stage.value,
        "summary": {
            "row_count": analysis.row_count,
            "candidate_count": len(analysis.candidates),
            "vote_method_count": len(analysis.vote_methods),
            "issue_count": len(issues),
            "duplicate_precinct_count": len(analysis.duplicate_precincts),
        },
        "schema": {
            "precinct_header": analysis.precinct_header,
            "reporting_header": analysis.reporting_header,
            "grand_total_header": analysis.grand_total_header,
            "method_total_headers": [
                {"method": method, "header": header}
                for method, header in analysis.method_total_headers
            ],
            "candidate_method_matrix_complete":
                analysis.candidate_method_matrix_complete,
            "schema_complete": analysis.schema_complete,
        },
        "candidates": list(analysis.candidates),
        "vote_methods": list(analysis.vote_methods),
        "candidate_blocks": blocks,
        "duplicate_precincts": list(analysis.duplicate_precincts),
        "reconciliation": {
            "candidate_totals_reconciled":
                analysis.candidate_totals_reconciled,
            "method_totals_reconciled":
                analysis.method_totals_reconciled,
            "precinct_totals_reconciled":
                analysis.precinct_totals_reconciled,
        },
        "observed_counts": {
            "zero": analysis.observed_zero_count,
            "null": analysis.observed_null_count,
        },
        "issues": issues,
        "raw_rows_included": False,
        "raw_headers_included": False,
        "selected_structure_headers_included": True,
        "automatic_timestamp": False,
    }


def project_election_structure_observation(
    result: TablePipelineResult,
) -> dict[str, Any]:
    """Analyze one typed parser result and project noncanonical observation."""
    if not isinstance(result, TablePipelineResult):
        raise TypeError("result must be a TablePipelineResult")
    return project_election_structure_analysis(
        analyze_election_structure(result)
    )
