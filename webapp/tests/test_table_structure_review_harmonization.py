"""C2G 2.8.23 contracts for isolated legacy harmonization parity."""

from __future__ import annotations

import ast
import copy
from dataclasses import fields
from pathlib import Path

import pytest

from webapp.parser.contracts.table_structure_review_harmonization import (
    LEGACY_HARMONIZATION_KNOWN_HAZARDS,
    TableStructureHarmonizationRequest,
    TableStructureHarmonizationResult,
)
from webapp.parser.services.table_structure_review_harmonization import (
    harmonize_table_structure_review,
)
from webapp.parser.utils.detect import harmonize_headers_and_data


WEBAPP_ROOT = Path(__file__).resolve().parents[1]


def _legacy_output(headers, rows, context=None):
    working_headers = copy.deepcopy(headers)
    working_rows = copy.deepcopy(rows)
    working_context = copy.deepcopy(context)
    return harmonize_headers_and_data(
        working_headers,
        working_rows,
        working_context,
    )


def _typed_output(headers, rows, context=None, source_location_label=None):
    request = TableStructureHarmonizationRequest(
        headers=tuple(headers),
        rows=tuple(copy.deepcopy(rows)),
        context=copy.deepcopy(context),
        source_location_label=source_location_label,
    )
    result = harmonize_table_structure_review(request)
    return list(result.headers), [dict(row) for row in result.rows], result


@pytest.mark.parametrize(
    "headers,rows,context",
    [
        (
            ["District", "Candidate", "Value"],
            [
                {
                    "District": "__LOCATION_A__",
                    "Candidate": "__ENTITY_A__",
                    "Value": "__VALUE_A__",
                }
            ],
            None,
        ),
        (
            ["District", "Candidate"],
            [
                {
                    "District": "__LOCATION_A__",
                    "Candidate": "__ENTITY_A__",
                    "RowOnlyAlpha": "__A__",
                    "RowOnlyBeta": "__B__",
                }
            ],
            None,
        ),
        (
            ["District", "Candidate", "Percent Reported"],
            [
                {
                    "District": "__LOCATION_A__",
                    "Candidate": "__ENTITY_A__",
                    "Percent Reported": "",
                }
            ],
            {"percent_reported": "77%"},
        ),
        (
            ["District", "Candidate", "Ballot Type"],
            [
                {
                    "District": "__LOCATION_A__",
                    "Candidate": "__ENTITY_A__",
                    "Ballot Type": "__METHOD_A__",
                },
                {
                    "District": "__LOCATION_A__",
                    "Candidate": "__ENTITY_A__",
                    "Ballot Type": "__METHOD_A__",
                },
            ],
            None,
        ),
    ],
)
def test_typed_boundary_preserves_legacy_output_semantics(headers, rows, context):
    legacy_headers, legacy_rows = _legacy_output(headers, rows, context)
    typed_headers, typed_rows, result = _typed_output(headers, rows, context)

    assert typed_headers == legacy_headers
    assert typed_rows == legacy_rows
    assert result.legacy_output_semantics_preserved is True
    assert result.caller_input_mutation_preserved is False
    assert result.normalized_internal_schema_authority is False
    assert result.canonical_authority is False


def test_boundary_does_not_propagate_legacy_caller_row_mutation():
    headers = ["District", "Candidate", "Value"]
    rows = [
        {
            "District": "__LOCATION_A__",
            "Candidate": "__ENTITY_A__",
            "Value": "__VALUE_A__",
        }
    ]
    headers_before = copy.deepcopy(headers)
    rows_before = copy.deepcopy(rows)

    typed_headers, typed_rows, result = _typed_output(
        headers,
        rows,
        source_location_label="District",
    )

    assert headers == headers_before
    assert rows == rows_before
    assert result.source_location_label == "District"

    legacy_headers, legacy_rows = _legacy_output(headers_before, rows_before)
    assert typed_headers == legacy_headers
    assert typed_rows == legacy_rows


def test_source_location_label_survives_as_evidence_metadata():
    _, _, result = _typed_output(
        ["District", "Candidate"],
        [{"District": "__LOCATION_A__", "Candidate": "__ENTITY_A__"}],
        source_location_label="District",
    )
    assert result.source_location_label == "District"


def test_null_zero_and_signed_values_survive_when_legacy_output_preserves_their_keys():
    headers = [
        "District",
        "Candidate",
        "Null Evidence",
        "Zero Evidence",
        "Signed Evidence",
    ]
    rows = [
        {
            "District": "__LOCATION_A__",
            "Candidate": "__ENTITY_A__",
            "Null Evidence": None,
            "Zero Evidence": 0,
            "Signed Evidence": -4,
        }
    ]

    legacy_headers, legacy_rows = _legacy_output(headers, rows)
    typed_headers, typed_rows, _ = _typed_output(headers, rows)

    assert typed_headers == legacy_headers
    assert typed_rows == legacy_rows
    assert typed_rows[0]["Null Evidence"] is None
    assert typed_rows[0]["Zero Evidence"] == 0
    assert typed_rows[0]["Signed Evidence"] == -4


def test_numeric_zero_percent_truthiness_hazard_is_not_repaired_in_parity_checkpoint():
    headers = ["District", "Candidate", "Percent Reported"]
    rows = [
        {
            "District": "__LOCATION_A__",
            "Candidate": "__ENTITY_A__",
            "Percent Reported": 0,
        }
    ]
    context = {"percent_reported": "100%"}

    legacy_headers, legacy_rows = _legacy_output(headers, rows, context)
    typed_headers, typed_rows, _ = _typed_output(headers, rows, context)

    assert typed_headers == legacy_headers
    assert typed_rows == legacy_rows
    # This pins the accepted legacy hazard for parity only; it is not endorsement.
    assert typed_rows[0]["Percent Reported"] == "100%"


def test_known_hazards_remain_explicit_and_complete():
    assert LEGACY_HARMONIZATION_KNOWN_HAZARDS == (
        "PERCENT_REPORTED_ZERO_TRUTHINESS",
        "PERCENT_ACCUMULATOR_TRUTHINESS",
        "LEGACY_LOCATION_LABEL_COLLAPSE",
        "LEGACY_CALLER_OWNED_ROW_MUTATION",
        "ROW_DEDUP_CAN_CHANGE_CARDINALITY",
        "ROW_ONLY_EXTRA_HEADER_SET_ORDER",
        "CASE_INSENSITIVE_HEADER_DEDUP_CAN_DROP_COLUMNS",
    )


def test_new_request_contract_uses_jurisdiction_neutral_field_names():
    request_fields = {field.name for field in fields(TableStructureHarmonizationRequest)}
    result_fields = {field.name for field in fields(TableStructureHarmonizationResult)}

    assert "source_location_label" in request_fields
    assert "source_location_label" in result_fields

    forbidden_specific_location_fields = {
        "precinct",
        "county",
        "district",
        "borough",
        "ward",
        "municipality",
    }
    assert request_fields.isdisjoint(forbidden_specific_location_fields)
    assert result_fields.isdisjoint(forbidden_specific_location_fields)


def test_contract_does_not_encode_historical_smart_wide_vote_method_slots():
    contract_path = (
        WEBAPP_ROOT
        / "parser"
        / "contracts"
        / "table_structure_review_harmonization.py"
    )
    source = contract_path.read_text(encoding="utf-8")
    forbidden = (
        "Election Day",
        "Mail-In",
        "Absentee Mail",
        "Early Voting",
        "Provisional Votes",
    )
    for token in forbidden:
        assert token not in source


def test_service_has_no_state_machine_transport_or_frontend_imports():
    service_path = (
        WEBAPP_ROOT
        / "parser"
        / "services"
        / "table_structure_review_harmonization.py"
    )
    source = service_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    forbidden_fragments = (
        "table_structure_review_state_machine",
        "socket",
        "flask",
        "websocket",
        "javascript",
        "static",
        "templates",
    )
    for module_name in imported:
        lowered = module_name.lower()
        assert not any(fragment in lowered for fragment in forbidden_fragments)


def test_boundary_has_no_canonical_or_learning_side_effect_calls():
    service_path = (
        WEBAPP_ROOT
        / "parser"
        / "services"
        / "table_structure_review_harmonization.py"
    )
    source = service_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    called = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            called.add(func.id)
        elif isinstance(func, ast.Attribute):
            called.add(func.attr)

    forbidden_calls = {
        "commit",
        "flush",
        "add",
        "execute",
        "save_table_structure_to_db",
        "log_table_structure",
        "cache_table_structure",
        "finalize_election_output",
    }
    assert called.isdisjoint(forbidden_calls)
