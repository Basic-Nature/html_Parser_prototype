from __future__ import annotations

import copy
from pathlib import Path

import pytest

from webapp.parser.contracts.workflow_canonical_writer import (
    CANONICAL_WRITER_ERROR_CODES,
    CANONICAL_WRITER_FAILURE_STATUSES,
    CANONICAL_WRITER_IDEMPOTENCY_FIELDS,
    CANONICAL_WRITER_IDEMPOTENCY_JSON_RULE,
    CANONICAL_WRITER_RESULT_STATUSES,
    CANONICAL_WRITER_SUCCESS_STATUSES,
    WORKFLOW_CANONICAL_WRITER_CALLBACK_INTERFACE,
    WORKFLOW_CANONICAL_WRITER_CONTRACT_VERSION,
    WORKFLOW_CANONICAL_WRITER_REQUEST_SCHEMA,
    WORKFLOW_CANONICAL_WRITER_RESULT_SCHEMA,
    WORKFLOW_CANONICAL_WRITER_TRANSPORT,
    WorkflowCanonicalWriterContractError,
    assert_canonical_writer_result_matches_request,
    derive_canonical_writer_idempotency_key,
    validate_canonical_writer_request,
    validate_canonical_writer_result,
)
from webapp.parser.contracts.workflow_comparison import (
    WORKFLOW_COMPARISON_PAYLOAD_CONTRACT,
    WORKFLOW_COMPARISON_PAYLOAD_SCHEMA_VERSION,
    WORKFLOW_COMPARISON_VERSION,
    semantic_sha256,
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


def _payload():
    semantic = _semantic()
    return {
        "schema": WORKFLOW_COMPARISON_PAYLOAD_CONTRACT,
        "schema_version": WORKFLOW_COMPARISON_PAYLOAD_SCHEMA_VERSION,
        "comparison_version": WORKFLOW_COMPARISON_VERSION,
        "binding": {
            "workflow_item_id": "00000000-0000-0000-0000-000000000001",
            "workflow_pass_id": "00000000-0000-0000-0000-000000000101",
            "pass_number": 2,
            "revision_number": 1,
            "source_evidence_ref": "official-source:example",
            "staging_batch_id": "00000000-0000-0000-0000-000000000010",
            "normalized_artifact_ref": "staging://normalized/example.json",
            "normalized_artifact_sha256": "1" * 64,
        },
        "semantic": semantic,
        "semantic_sha256": semantic_sha256(semantic),
    }


def _request():
    request = {
        "schema": WORKFLOW_CANONICAL_WRITER_REQUEST_SCHEMA,
        "schema_version": WORKFLOW_CANONICAL_WRITER_CONTRACT_VERSION,
        "request_id": "00000000-0000-0000-0000-000000000501",
        "idempotency_key": "0" * 64,
        "workflow": {
            "workflow_item_id": "00000000-0000-0000-0000-000000000001",
            "workflow_row_version": 7,
            "publication_handoff_event_id":
                "00000000-0000-0000-0000-000000000502",
            "publication_operator_principal": "principal:publication",
        },
        "approval": {
            "qc1_review_id": "00000000-0000-0000-0000-000000000201",
            "qc2_review_id": "00000000-0000-0000-0000-000000000202",
            "qc1_decision": "approved",
            "qc2_decision": "approved",
            "selected_pass_id": "00000000-0000-0000-0000-000000000101",
            "selected_staging_batch_id":
                "00000000-0000-0000-0000-000000000010",
        },
        "comparison": {
            "comparison_id": "00000000-0000-0000-0000-000000000301",
            "comparison_version": 1,
            "status": "complete",
            "strict_equality_passed": False,
            "open_discrepancy_count": 0,
        },
        "payload": _payload(),
    }
    request["idempotency_key"] = derive_canonical_writer_idempotency_key(
        request
    )
    return request


def _published_result(request, *, status="published"):
    return {
        "schema": WORKFLOW_CANONICAL_WRITER_RESULT_SCHEMA,
        "schema_version": WORKFLOW_CANONICAL_WRITER_CONTRACT_VERSION,
        "request_id": request["request_id"],
        "idempotency_key": request["idempotency_key"],
        "status": status,
        "success": True,
        "publication": {
            "canonical_race_id":
                "00000000-0000-0000-0000-000000000601",
            "canonical_source_artifact_id":
                "00000000-0000-0000-0000-000000000602",
            "canonical_result_count": 2,
            "canonical_vote_component_count": 10,
            "semantic_sha256": request["payload"]["semantic_sha256"],
            "committed_at": "2026-09-08T18:30:00Z",
            "writer_service_version": "canonical-writer:v1",
        },
        "error": None,
    }


def _failure_result(request, *, status, code, retryable):
    return {
        "schema": WORKFLOW_CANONICAL_WRITER_RESULT_SCHEMA,
        "schema_version": WORKFLOW_CANONICAL_WRITER_CONTRACT_VERSION,
        "request_id": request["request_id"],
        "idempotency_key": request["idempotency_key"],
        "status": status,
        "success": False,
        "publication": None,
        "error": {
            "code": code,
            "message": "governed publication rejected",
            "retryable": retryable,
        },
    }


def test_exact_w5a_contract_vocabulary():
    assert WORKFLOW_CANONICAL_WRITER_REQUEST_SCHEMA == (
        "workflow_canonical_writer_request_v1"
    )
    assert WORKFLOW_CANONICAL_WRITER_RESULT_SCHEMA == (
        "workflow_canonical_writer_result_v1"
    )
    assert WORKFLOW_CANONICAL_WRITER_CONTRACT_VERSION == 1
    assert WORKFLOW_CANONICAL_WRITER_CALLBACK_INTERFACE == (
        "canonical_writer(request) -> result"
    )
    assert WORKFLOW_CANONICAL_WRITER_TRANSPORT == (
        "TRANSPORT_NEUTRAL_SYNCHRONOUS_APPLICATION_CALLBACK"
    )
    assert CANONICAL_WRITER_SUCCESS_STATUSES == (
        "published", "already_published",
    )
    assert CANONICAL_WRITER_FAILURE_STATUSES == ("rejected", "failed")
    assert CANONICAL_WRITER_RESULT_STATUSES == (
        "published", "already_published", "rejected", "failed",
    )
    assert CANONICAL_WRITER_ERROR_CODES == (
        "invalid_request",
        "precondition_failed",
        "idempotency_conflict",
        "canonical_conflict",
        "write_failed",
    )


def test_idempotency_vocabulary_and_material_are_exact():
    assert CANONICAL_WRITER_IDEMPOTENCY_JSON_RULE == (
        "UTF8_JSON_SORTED_OBJECT_KEYS_COMPACT_SEPARATORS_"
        "ENSURE_ASCII_FALSE_ALLOW_NAN_FALSE_NO_TRAILING_NEWLINE"
    )
    assert CANONICAL_WRITER_IDEMPOTENCY_FIELDS == (
        "workflow_item_id",
        "qc1_review_id",
        "qc2_review_id",
        "selected_pass_id",
        "selected_staging_batch_id",
        "comparison_id",
        "comparison_version",
        "normalized_artifact_sha256",
        "semantic_sha256",
    )
    request = _request()
    assert len(request["idempotency_key"]) == 64
    assert request["idempotency_key"] == (
        derive_canonical_writer_idempotency_key(request)
    )


def test_valid_request_accepts_resolved_discrepancy_path_and_preserves_w4_payload():
    request = _request()
    assert request["comparison"]["strict_equality_passed"] is False
    validated = validate_canonical_writer_request(request)
    assert validated["comparison"]["open_discrepancy_count"] == 0
    assert validated["approval"]["qc1_decision"] == "approved"
    assert validated["approval"]["qc2_decision"] == "approved"
    assert validated["payload"]["semantic"]["scope"]["jurisdiction_name"] is None
    method_totals = validated["payload"]["semantic"]["records"][0][
        "method_totals"
    ]
    assert method_totals[2] == {
        "method": "Absentee Mail",
        "state": "null",
        "votes": None,
    }
    assert method_totals[3] == {
        "method": "Provisional",
        "state": "value",
        "votes": 0,
    }
    assert method_totals[4] == {
        "method": "Curbside",
        "state": "missing",
        "votes": None,
    }


def test_idempotency_excludes_request_id_and_publication_operator_principal():
    left = _request()
    right = copy.deepcopy(left)
    right["request_id"] = "00000000-0000-0000-0000-000000000599"
    right["workflow"]["publication_operator_principal"] = (
        "principal:other-publication"
    )
    assert derive_canonical_writer_idempotency_key(left) == (
        derive_canonical_writer_idempotency_key(right)
    )
    right["idempotency_key"] = left["idempotency_key"]
    validate_canonical_writer_request(right)


def test_request_rejects_wrong_idempotency_key():
    request = _request()
    request["idempotency_key"] = "f" * 64
    with pytest.raises(WorkflowCanonicalWriterContractError):
        validate_canonical_writer_request(request)


def test_request_rejects_nonapproved_qc_and_open_discrepancies():
    request = _request()
    request["approval"]["qc2_decision"] = "returned"
    request["idempotency_key"] = derive_canonical_writer_idempotency_key(
        request
    )
    with pytest.raises(WorkflowCanonicalWriterContractError):
        validate_canonical_writer_request(request)

    request = _request()
    request["comparison"]["open_discrepancy_count"] = 1
    request["idempotency_key"] = derive_canonical_writer_idempotency_key(
        request
    )
    with pytest.raises(WorkflowCanonicalWriterContractError):
        validate_canonical_writer_request(request)


def test_request_rejects_noncomplete_comparison():
    request = _request()
    request["comparison"]["status"] = "pending"
    request["idempotency_key"] = derive_canonical_writer_idempotency_key(
        request
    )
    with pytest.raises(WorkflowCanonicalWriterContractError):
        validate_canonical_writer_request(request)


def test_request_requires_exact_selected_pass_and_staging_bindings():
    request = _request()
    request["approval"]["selected_pass_id"] = (
        "00000000-0000-0000-0000-000000000199"
    )
    request["idempotency_key"] = derive_canonical_writer_idempotency_key(
        request
    )
    with pytest.raises(WorkflowCanonicalWriterContractError):
        validate_canonical_writer_request(request)

    request = _request()
    request["approval"]["selected_staging_batch_id"] = (
        "00000000-0000-0000-0000-000000000019"
    )
    request["idempotency_key"] = derive_canonical_writer_idempotency_key(
        request
    )
    with pytest.raises(WorkflowCanonicalWriterContractError):
        validate_canonical_writer_request(request)


def test_published_result_is_valid_and_matches_request():
    request = _request()
    result = _published_result(request)
    validated = validate_canonical_writer_result(result)
    assert validated["status"] == "published"
    assert validated["success"] is True
    assert validated["error"] is None
    assert_canonical_writer_result_matches_request(request, result)


def test_already_published_replay_is_success_with_same_authority_shape():
    request = _request()
    result = _published_result(request, status="already_published")
    validated = validate_canonical_writer_result(result)
    assert validated["status"] == "already_published"
    assert validated["success"] is True
    assert_canonical_writer_result_matches_request(request, result)


def test_rejected_result_is_nonretryable_and_has_no_publication():
    request = _request()
    result = _failure_result(
        request,
        status="rejected",
        code="canonical_conflict",
        retryable=False,
    )
    validated = validate_canonical_writer_result(result)
    assert validated["success"] is False
    assert validated["publication"] is None
    assert validated["error"]["retryable"] is False
    assert_canonical_writer_result_matches_request(request, result)


def test_retryable_true_is_only_failed_write_failed():
    request = _request()
    valid = _failure_result(
        request,
        status="failed",
        code="write_failed",
        retryable=True,
    )
    validate_canonical_writer_result(valid)

    invalid = _failure_result(
        request,
        status="rejected",
        code="canonical_conflict",
        retryable=True,
    )
    with pytest.raises(WorkflowCanonicalWriterContractError):
        validate_canonical_writer_result(invalid)


def test_success_result_must_echo_request_ids_and_semantic_hash():
    request = _request()
    result = _published_result(request)
    result["request_id"] = "00000000-0000-0000-0000-000000000598"
    with pytest.raises(WorkflowCanonicalWriterContractError):
        assert_canonical_writer_result_matches_request(request, result)

    result = _published_result(request)
    result["publication"]["semantic_sha256"] = "f" * 64
    with pytest.raises(WorkflowCanonicalWriterContractError):
        assert_canonical_writer_result_matches_request(request, result)


def test_contract_module_is_pure_and_has_no_runtime_writer_dependencies():
    source = Path(
        "webapp/parser/contracts/workflow_canonical_writer.py"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "import flask",
        "from flask",
        "import sqlalchemy",
        "from sqlalchemy",
        "get_engine",
        "Session(",
        "session.add",
        "session.commit",
        "register_blueprint",
        "requests.post",
        "httpx",
        "import keycloak",
        "from keycloak",
        "KeycloakOpenID(",
    ):
        assert forbidden not in source
