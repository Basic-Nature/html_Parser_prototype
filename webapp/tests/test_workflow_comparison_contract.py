from __future__ import annotations

import copy

import pytest

from webapp.parser.contracts.workflow_comparison import (
    CANONICAL_JSON_RULE,
    DISCREPANCY_CATEGORIES,
    KNOWN_VOTE_METHOD_ORDER,
    VALUE_STATES,
    VOTE_METHOD_ALIASES,
    WORKFLOW_COMPARISON_DIFFERENCE_SUMMARY_SCHEMA,
    WORKFLOW_COMPARISON_PAYLOAD_CONTRACT,
    WORKFLOW_COMPARISON_PAYLOAD_SCHEMA_VERSION,
    WORKFLOW_COMPARISON_VERSION,
    WorkflowComparisonContractError,
    assert_comparable_payloads,
    build_difference_summary,
    canonical_percent_string,
    canonical_semantic_json,
    normalize_text,
    normalize_vote_method,
    ordered_vote_methods,
    semantic_sha256,
    strict_semantic_equality,
    validate_comparison_payload,
)


def _semantic():
    return {
        "scope": {
            "election_year": 2024,
            "election_date": "2024-11-05",
            "state": "TX",
            "jurisdiction_name": None,
            "jurisdiction_type": None,
            "contest": "President",
        },
        "records": [
            {
                "reporting_unit": {
                    "name": "Precinct 1",
                    "type": "precinct",
                },
                "percent_reporting": {
                    "state": "value",
                    "value": "100",
                },
                "vote_methods": [
                    "Election Day",
                    "Early Voting",
                    "Absentee Mail",
                    "Provisional",
                    "Curbside",
                ],
                "method_totals": [
                    {"method": "Election Day", "state": "value", "votes": 11},
                    {"method": "Early Voting", "state": "value", "votes": 7},
                    {"method": "Absentee Mail", "state": "null", "votes": None},
                    {"method": "Provisional", "state": "value", "votes": 0},
                    {"method": "Curbside", "state": "missing", "votes": None},
                ],
                "candidates": [
                    {
                        "name": "Jane Doe",
                        "party": "DEM",
                        "method_votes": [
                            {"method": "Election Day", "state": "value", "votes": 6},
                            {"method": "Early Voting", "state": "value", "votes": 4},
                            {"method": "Absentee Mail", "state": "null", "votes": None},
                            {"method": "Provisional", "state": "value", "votes": 0},
                            {"method": "Curbside", "state": "missing", "votes": None},
                        ],
                        "total_votes": {"state": "value", "votes": 10},
                    },
                    {
                        "name": "John Smith",
                        "party": "REP",
                        "method_votes": [
                            {"method": "Election Day", "state": "value", "votes": 5},
                            {"method": "Early Voting", "state": "value", "votes": 3},
                            {"method": "Absentee Mail", "state": "null", "votes": None},
                            {"method": "Provisional", "state": "value", "votes": 0},
                            {"method": "Curbside", "state": "missing", "votes": None},
                        ],
                        "total_votes": {"state": "value", "votes": 8},
                    },
                ],
                "grand_total": {"state": "value", "votes": 18},
            }
        ],
    }


def _payload(*, pass_id: str):
    semantic = _semantic()
    return {
        "schema": WORKFLOW_COMPARISON_PAYLOAD_CONTRACT,
        "schema_version": WORKFLOW_COMPARISON_PAYLOAD_SCHEMA_VERSION,
        "comparison_version": WORKFLOW_COMPARISON_VERSION,
        "binding": {
            "workflow_item_id": "00000000-0000-0000-0000-000000000001",
            "workflow_pass_id": pass_id,
            "pass_number": 1,
            "revision_number": 1,
            "source_evidence_ref": "official-source:example",
            "staging_batch_id": "00000000-0000-0000-0000-000000000010",
            "normalized_artifact_ref": "staging://normalized/example.json",
            "normalized_artifact_sha256": "1" * 64,
        },
        "semantic": semantic,
        "semantic_sha256": semantic_sha256(semantic),
    }


def test_exact_w4a_contract_vocabulary():
    assert WORKFLOW_COMPARISON_PAYLOAD_CONTRACT == (
        "workflow_normalized_semantic_comparison_payload_v1"
    )
    assert WORKFLOW_COMPARISON_PAYLOAD_SCHEMA_VERSION == 1
    assert WORKFLOW_COMPARISON_VERSION == 1
    assert WORKFLOW_COMPARISON_DIFFERENCE_SUMMARY_SCHEMA == (
        "workflow_comparison_difference_summary_v1"
    )
    assert VALUE_STATES == ("value", "null", "missing")
    assert KNOWN_VOTE_METHOD_ORDER == (
        "Election Day",
        "Early Voting",
        "Absentee Mail",
        "Provisional",
    )
    assert DISCREPANCY_CATEGORIES == (
        "scope_mismatch",
        "missing_left",
        "missing_right",
        "null_vs_zero",
        "null_vs_value",
        "value_mismatch",
    )
    assert CANONICAL_JSON_RULE.endswith("SEMANTIC_CONTENT_ONLY")


def test_name_and_method_normalization_is_deterministic_not_fuzzy():
    assert normalize_text("  Jane   Doe  ") == "Jane Doe"
    assert normalize_vote_method(" election   day ") == "Election Day"
    assert normalize_vote_method("Curbside") == "Curbside"
    assert VOTE_METHOD_ALIASES["early voting"] == "Early Voting"
    assert ordered_vote_methods([
        "Curbside",
        "Provisional",
        "Election Day",
        "Early Voting",
        "Absentee Mail",
    ]) == (
        "Election Day",
        "Early Voting",
        "Absentee Mail",
        "Provisional",
        "Curbside",
    )
    with pytest.raises(WorkflowComparisonContractError):
        ordered_vote_methods(["Election Day", " election day "])


def test_percent_reporting_has_canonical_decimal_string_and_no_float():
    assert canonical_percent_string("100.00") == "100"
    assert canonical_percent_string("12.500") == "12.5"
    assert canonical_percent_string(0) == "0"
    with pytest.raises(WorkflowComparisonContractError):
        canonical_percent_string(12.5)
    with pytest.raises(WorkflowComparisonContractError):
        canonical_percent_string("50%")


def test_canonical_semantic_hash_is_stable_and_excludes_binding_provenance():
    left = _payload(
        pass_id="00000000-0000-0000-0000-000000000101",
    )
    right = _payload(
        pass_id="00000000-0000-0000-0000-000000000102",
    )
    right["binding"]["source_evidence_ref"] = "independent-source:other"
    right["binding"]["normalized_artifact_ref"] = "staging://other.json"
    right["binding"]["normalized_artifact_sha256"] = "2" * 64

    assert canonical_semantic_json(left["semantic"]) == canonical_semantic_json(
        right["semantic"]
    )
    assert left["semantic_sha256"] == right["semantic_sha256"]
    assert strict_semantic_equality(left, right) is True


def test_valid_payload_preserves_zero_null_missing_and_all_methods():
    payload = _payload(
        pass_id="00000000-0000-0000-0000-000000000101",
    )
    validated = validate_comparison_payload(payload)
    record = validated["semantic"]["records"][0]

    assert record["vote_methods"] == [
        "Election Day",
        "Early Voting",
        "Absentee Mail",
        "Provisional",
        "Curbside",
    ]
    assert record["method_totals"][2] == {
        "method": "Absentee Mail",
        "state": "null",
        "votes": None,
    }
    assert record["method_totals"][3] == {
        "method": "Provisional",
        "state": "value",
        "votes": 0,
    }
    assert record["method_totals"][4] == {
        "method": "Curbside",
        "state": "missing",
        "votes": None,
    }
    for candidate in record["candidates"]:
        assert len(candidate["method_votes"]) == len(record["vote_methods"])


def test_payload_hash_mismatch_fails_closed():
    payload = _payload(
        pass_id="00000000-0000-0000-0000-000000000101",
    )
    payload["semantic_sha256"] = "0" * 64
    with pytest.raises(WorkflowComparisonContractError):
        validate_comparison_payload(payload)


def test_missing_candidate_method_is_invalid_not_silently_zero_filled():
    payload = _payload(
        pass_id="00000000-0000-0000-0000-000000000101",
    )
    payload["semantic"]["records"][0]["candidates"][0]["method_votes"].pop()
    payload["semantic_sha256"] = semantic_sha256(payload["semantic"])
    with pytest.raises(WorkflowComparisonContractError):
        validate_comparison_payload(payload)


def test_duplicate_semantic_keys_fail_closed():
    payload = _payload(
        pass_id="00000000-0000-0000-0000-000000000101",
    )
    duplicate = copy.deepcopy(
        payload["semantic"]["records"][0]["candidates"][0]
    )
    payload["semantic"]["records"][0]["candidates"].append(duplicate)
    payload["semantic_sha256"] = semantic_sha256(payload["semantic"])
    with pytest.raises(WorkflowComparisonContractError):
        validate_comparison_payload(payload)

    payload = _payload(
        pass_id="00000000-0000-0000-0000-000000000101",
    )
    payload["semantic"]["records"].append(
        copy.deepcopy(payload["semantic"]["records"][0])
    )
    payload["semantic_sha256"] = semantic_sha256(payload["semantic"])
    with pytest.raises(WorkflowComparisonContractError):
        validate_comparison_payload(payload)


def test_noncanonical_order_fails_closed():
    payload = _payload(
        pass_id="00000000-0000-0000-0000-000000000101",
    )
    record = payload["semantic"]["records"][0]
    record["vote_methods"] = list(reversed(record["vote_methods"]))
    payload["semantic_sha256"] = semantic_sha256(payload["semantic"])
    with pytest.raises(WorkflowComparisonContractError):
        validate_comparison_payload(payload)

    payload = _payload(
        pass_id="00000000-0000-0000-0000-000000000101",
    )
    payload["semantic"]["records"][0]["candidates"].reverse()
    payload["semantic_sha256"] = semantic_sha256(payload["semantic"])
    with pytest.raises(WorkflowComparisonContractError):
        validate_comparison_payload(payload)


def test_comparable_payloads_require_same_item_and_distinct_pass_revisions():
    left = _payload(
        pass_id="00000000-0000-0000-0000-000000000101",
    )
    right = _payload(
        pass_id="00000000-0000-0000-0000-000000000102",
    )
    assert_comparable_payloads(left, right)

    wrong_item = copy.deepcopy(right)
    wrong_item["binding"]["workflow_item_id"] = (
        "00000000-0000-0000-0000-000000000002"
    )
    with pytest.raises(WorkflowComparisonContractError):
        assert_comparable_payloads(left, wrong_item)

    same_pass = copy.deepcopy(right)
    same_pass["binding"]["workflow_pass_id"] = (
        left["binding"]["workflow_pass_id"]
    )
    with pytest.raises(WorkflowComparisonContractError):
        assert_comparable_payloads(left, same_pass)


def test_semantic_change_breaks_strict_equality_without_provenance_dependency():
    left = _payload(
        pass_id="00000000-0000-0000-0000-000000000101",
    )
    right = _payload(
        pass_id="00000000-0000-0000-0000-000000000102",
    )
    right["semantic"]["records"][0]["candidates"][0]["method_votes"][0][
        "votes"
    ] = 7
    right["semantic_sha256"] = semantic_sha256(right["semantic"])

    assert strict_semantic_equality(left, right) is False


def test_difference_summary_is_exact_and_category_counts_sum():
    summary = build_difference_summary(
        left_semantic_sha256="1" * 64,
        right_semantic_sha256="2" * 64,
        category_counts={
            "scope_mismatch": 0,
            "missing_left": 1,
            "missing_right": 2,
            "null_vs_zero": 3,
            "null_vs_value": 4,
            "value_mismatch": 5,
        },
    )
    assert summary == {
        "schema": "workflow_comparison_difference_summary_v1",
        "comparison_version": 1,
        "left_semantic_sha256": "1" * 64,
        "right_semantic_sha256": "2" * 64,
        "difference_count": 15,
        "category_counts": {
            "scope_mismatch": 0,
            "missing_left": 1,
            "missing_right": 2,
            "null_vs_zero": 3,
            "null_vs_value": 4,
            "value_mismatch": 5,
        },
    }

    with pytest.raises(WorkflowComparisonContractError):
        build_difference_summary(
            left_semantic_sha256="1" * 64,
            right_semantic_sha256="2" * 64,
            category_counts={"value_mismatch": 1},
        )
