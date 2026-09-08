"""Pure canonical-writer request/result contract for governed Workflow publication.

This module freezes the accepted W5A application callback vocabulary, exact
request/result shapes, idempotency derivation, and fail-closed validation.

It may import the pure W4 normalized comparison contract. It does NOT register
routes, access the database, mutate Workflow or canonical rows, activate
Keycloak, or invoke a canonical writer.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
import hashlib
import json
import re
import uuid

from webapp.parser.contracts.workflow_comparison import (
    WORKFLOW_COMPARISON_PAYLOAD_CONTRACT,
    WORKFLOW_COMPARISON_VERSION,
    validate_comparison_payload,
)


WORKFLOW_CANONICAL_WRITER_REQUEST_SCHEMA = "workflow_canonical_writer_request_v1"
WORKFLOW_CANONICAL_WRITER_RESULT_SCHEMA = "workflow_canonical_writer_result_v1"
WORKFLOW_CANONICAL_WRITER_CONTRACT_VERSION = 1
WORKFLOW_CANONICAL_WRITER_CALLBACK_INTERFACE = "canonical_writer(request) -> result"
WORKFLOW_CANONICAL_WRITER_TRANSPORT = (
    "TRANSPORT_NEUTRAL_SYNCHRONOUS_APPLICATION_CALLBACK"
)

CANONICAL_WRITER_SUCCESS_STATUSES = ("published", "already_published")
CANONICAL_WRITER_FAILURE_STATUSES = ("rejected", "failed")
CANONICAL_WRITER_RESULT_STATUSES = (
    CANONICAL_WRITER_SUCCESS_STATUSES + CANONICAL_WRITER_FAILURE_STATUSES
)
CANONICAL_WRITER_ERROR_CODES = (
    "invalid_request",
    "precondition_failed",
    "idempotency_conflict",
    "canonical_conflict",
    "write_failed",
)

CANONICAL_WRITER_IDEMPOTENCY_JSON_RULE = (
    "UTF8_JSON_SORTED_OBJECT_KEYS_COMPACT_SEPARATORS_"
    "ENSURE_ASCII_FALSE_ALLOW_NAN_FALSE_NO_TRAILING_NEWLINE"
)
CANONICAL_WRITER_IDEMPOTENCY_FIELDS = (
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

REQUEST_TOP_LEVEL_KEYS = frozenset({
    "schema",
    "schema_version",
    "request_id",
    "idempotency_key",
    "workflow",
    "approval",
    "comparison",
    "payload",
})
REQUEST_WORKFLOW_KEYS = frozenset({
    "workflow_item_id",
    "workflow_row_version",
    "publication_handoff_event_id",
    "publication_operator_principal",
})
REQUEST_APPROVAL_KEYS = frozenset({
    "qc1_review_id",
    "qc2_review_id",
    "qc1_decision",
    "qc2_decision",
    "selected_pass_id",
    "selected_staging_batch_id",
})
REQUEST_COMPARISON_KEYS = frozenset({
    "comparison_id",
    "comparison_version",
    "status",
    "strict_equality_passed",
    "open_discrepancy_count",
})
RESULT_TOP_LEVEL_KEYS = frozenset({
    "schema",
    "schema_version",
    "request_id",
    "idempotency_key",
    "status",
    "success",
    "publication",
    "error",
})
RESULT_PUBLICATION_KEYS = frozenset({
    "canonical_race_id",
    "canonical_source_artifact_id",
    "canonical_result_count",
    "canonical_vote_component_count",
    "semantic_sha256",
    "committed_at",
    "writer_service_version",
})
RESULT_ERROR_KEYS = frozenset({
    "code",
    "message",
    "retryable",
})

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class WorkflowCanonicalWriterContractError(ValueError):
    """Fail-closed canonical writer contract violation."""


def _require_mapping(name: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise WorkflowCanonicalWriterContractError(f"{name} must be an object")
    return value


def _require_exact_keys(
    name: str,
    value: Mapping[str, object],
    expected: frozenset[str],
) -> None:
    actual = frozenset(str(key) for key in value.keys())
    if actual != expected:
        raise WorkflowCanonicalWriterContractError(
            f"{name} keys must be exactly {sorted(expected)!r}; "
            f"observed {sorted(actual)!r}"
        )


def _require_uuid(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorkflowCanonicalWriterContractError(
            f"{name} must be a UUID string"
        )
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise WorkflowCanonicalWriterContractError(
            f"{name} must be a valid UUID"
        ) from exc
    canonical = str(parsed)
    if value != canonical:
        raise WorkflowCanonicalWriterContractError(
            f"{name} must use canonical lowercase UUID text"
        )
    return canonical


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise WorkflowCanonicalWriterContractError(
            f"{name} must be lowercase 64-character SHA-256 hex"
        )
    return value


def _require_nonempty(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorkflowCanonicalWriterContractError(
            f"{name} must be a non-empty string"
        )
    if value != value.strip():
        raise WorkflowCanonicalWriterContractError(
            f"{name} must not contain surrounding whitespace"
        )
    return value


def _require_nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise WorkflowCanonicalWriterContractError(
            f"{name} must be a nonnegative integer"
        )
    return value


def _require_positive_int(name: str, value: object) -> int:
    value = _require_nonnegative_int(name, value)
    if value < 1:
        raise WorkflowCanonicalWriterContractError(
            f"{name} must be a positive integer"
        )
    return value


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise WorkflowCanonicalWriterContractError(
            "idempotency material must be canonical JSON serializable"
        ) from exc


def canonical_writer_idempotency_material(
    request: Mapping[str, object],
) -> dict[str, object]:
    workflow = _require_mapping("workflow", request.get("workflow"))
    approval = _require_mapping("approval", request.get("approval"))
    comparison = _require_mapping("comparison", request.get("comparison"))
    payload = _require_mapping("payload", request.get("payload"))
    binding = _require_mapping("payload.binding", payload.get("binding"))

    return {
        "workflow_item_id": workflow.get("workflow_item_id"),
        "qc1_review_id": approval.get("qc1_review_id"),
        "qc2_review_id": approval.get("qc2_review_id"),
        "selected_pass_id": approval.get("selected_pass_id"),
        "selected_staging_batch_id":
            approval.get("selected_staging_batch_id"),
        "comparison_id": comparison.get("comparison_id"),
        "comparison_version": comparison.get("comparison_version"),
        "normalized_artifact_sha256":
            binding.get("normalized_artifact_sha256"),
        "semantic_sha256": payload.get("semantic_sha256"),
    }


def derive_canonical_writer_idempotency_key(
    request: Mapping[str, object],
) -> str:
    material = canonical_writer_idempotency_material(request)
    raw = _canonical_json(material).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _validate_request_workflow(raw: object) -> dict[str, object]:
    value = _require_mapping("workflow", raw)
    _require_exact_keys("workflow", value, REQUEST_WORKFLOW_KEYS)
    return {
        "workflow_item_id":
            _require_uuid("workflow.workflow_item_id", value["workflow_item_id"]),
        "workflow_row_version":
            _require_positive_int(
                "workflow.workflow_row_version",
                value["workflow_row_version"],
            ),
        "publication_handoff_event_id":
            _require_uuid(
                "workflow.publication_handoff_event_id",
                value["publication_handoff_event_id"],
            ),
        "publication_operator_principal":
            _require_nonempty(
                "workflow.publication_operator_principal",
                value["publication_operator_principal"],
            ),
    }


def _validate_request_approval(raw: object) -> dict[str, object]:
    value = _require_mapping("approval", raw)
    _require_exact_keys("approval", value, REQUEST_APPROVAL_KEYS)
    qc1 = value["qc1_decision"]
    qc2 = value["qc2_decision"]
    if qc1 != "approved" or qc2 != "approved":
        raise WorkflowCanonicalWriterContractError(
            "QC1 and QC2 decisions must both be approved"
        )
    return {
        "qc1_review_id":
            _require_uuid("approval.qc1_review_id", value["qc1_review_id"]),
        "qc2_review_id":
            _require_uuid("approval.qc2_review_id", value["qc2_review_id"]),
        "qc1_decision": qc1,
        "qc2_decision": qc2,
        "selected_pass_id":
            _require_uuid(
                "approval.selected_pass_id",
                value["selected_pass_id"],
            ),
        "selected_staging_batch_id":
            _require_uuid(
                "approval.selected_staging_batch_id",
                value["selected_staging_batch_id"],
            ),
    }


def _validate_request_comparison(raw: object) -> dict[str, object]:
    value = _require_mapping("comparison", raw)
    _require_exact_keys("comparison", value, REQUEST_COMPARISON_KEYS)

    version = _require_positive_int(
        "comparison.comparison_version",
        value["comparison_version"],
    )
    if version != WORKFLOW_COMPARISON_VERSION:
        raise WorkflowCanonicalWriterContractError(
            "comparison.comparison_version must match the W4 comparison version"
        )
    if value["status"] != "complete":
        raise WorkflowCanonicalWriterContractError(
            "comparison.status must be complete"
        )
    strict = value["strict_equality_passed"]
    if not isinstance(strict, bool):
        raise WorkflowCanonicalWriterContractError(
            "comparison.strict_equality_passed must be bool"
        )
    open_count = _require_nonnegative_int(
        "comparison.open_discrepancy_count",
        value["open_discrepancy_count"],
    )
    if open_count != 0:
        raise WorkflowCanonicalWriterContractError(
            "comparison.open_discrepancy_count must be zero"
        )

    return {
        "comparison_id":
            _require_uuid(
                "comparison.comparison_id",
                value["comparison_id"],
            ),
        "comparison_version": version,
        "status": "complete",
        "strict_equality_passed": strict,
        "open_discrepancy_count": 0,
    }


def validate_canonical_writer_request(
    raw: Mapping[str, object],
) -> dict[str, object]:
    request = _require_mapping("request", raw)
    _require_exact_keys("request", request, REQUEST_TOP_LEVEL_KEYS)

    if request["schema"] != WORKFLOW_CANONICAL_WRITER_REQUEST_SCHEMA:
        raise WorkflowCanonicalWriterContractError(
            "request.schema does not match the canonical writer request schema"
        )
    if request["schema_version"] != WORKFLOW_CANONICAL_WRITER_CONTRACT_VERSION:
        raise WorkflowCanonicalWriterContractError(
            "request.schema_version must equal 1"
        )

    request_id = _require_uuid("request.request_id", request["request_id"])
    idempotency_key = _require_sha256(
        "request.idempotency_key",
        request["idempotency_key"],
    )
    workflow = _validate_request_workflow(request["workflow"])
    approval = _validate_request_approval(request["approval"])
    comparison = _validate_request_comparison(request["comparison"])

    payload = validate_comparison_payload(request["payload"])
    if payload["schema"] != WORKFLOW_COMPARISON_PAYLOAD_CONTRACT:
        raise WorkflowCanonicalWriterContractError(
            "request.payload must use the W4 normalized comparison payload schema"
        )

    binding = payload["binding"]
    if binding["workflow_item_id"] != workflow["workflow_item_id"]:
        raise WorkflowCanonicalWriterContractError(
            "payload workflow item binding must match request workflow item"
        )
    if binding["workflow_pass_id"] != approval["selected_pass_id"]:
        raise WorkflowCanonicalWriterContractError(
            "payload pass binding must match the QC-selected pass"
        )
    if binding["staging_batch_id"] != approval["selected_staging_batch_id"]:
        raise WorkflowCanonicalWriterContractError(
            "payload staging binding must match the QC-selected staging batch"
        )
    if payload["comparison_version"] != comparison["comparison_version"]:
        raise WorkflowCanonicalWriterContractError(
            "payload comparison_version must match request comparison"
        )

    normalized = {
        "schema": WORKFLOW_CANONICAL_WRITER_REQUEST_SCHEMA,
        "schema_version": WORKFLOW_CANONICAL_WRITER_CONTRACT_VERSION,
        "request_id": request_id,
        "idempotency_key": idempotency_key,
        "workflow": workflow,
        "approval": approval,
        "comparison": comparison,
        "payload": payload,
    }

    expected_key = derive_canonical_writer_idempotency_key(normalized)
    if idempotency_key != expected_key:
        raise WorkflowCanonicalWriterContractError(
            "request.idempotency_key does not match deterministic material"
        )
    return normalized


def _validate_publication(raw: object) -> dict[str, object]:
    value = _require_mapping("result.publication", raw)
    _require_exact_keys(
        "result.publication",
        value,
        RESULT_PUBLICATION_KEYS,
    )

    committed_at = _require_nonempty(
        "result.publication.committed_at",
        value["committed_at"],
    )
    try:
        parsed = datetime.fromisoformat(
            committed_at.replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise WorkflowCanonicalWriterContractError(
            "result.publication.committed_at must be ISO-8601"
        ) from exc
    if parsed.tzinfo is None:
        raise WorkflowCanonicalWriterContractError(
            "result.publication.committed_at must be timezone-aware"
        )

    return {
        "canonical_race_id":
            _require_uuid(
                "result.publication.canonical_race_id",
                value["canonical_race_id"],
            ),
        "canonical_source_artifact_id":
            _require_uuid(
                "result.publication.canonical_source_artifact_id",
                value["canonical_source_artifact_id"],
            ),
        "canonical_result_count":
            _require_nonnegative_int(
                "result.publication.canonical_result_count",
                value["canonical_result_count"],
            ),
        "canonical_vote_component_count":
            _require_nonnegative_int(
                "result.publication.canonical_vote_component_count",
                value["canonical_vote_component_count"],
            ),
        "semantic_sha256":
            _require_sha256(
                "result.publication.semantic_sha256",
                value["semantic_sha256"],
            ),
        "committed_at": committed_at,
        "writer_service_version":
            _require_nonempty(
                "result.publication.writer_service_version",
                value["writer_service_version"],
            ),
    }


def _validate_error(
    raw: object,
    *,
    status: str,
) -> dict[str, object]:
    value = _require_mapping("result.error", raw)
    _require_exact_keys("result.error", value, RESULT_ERROR_KEYS)

    code = value["code"]
    if code not in CANONICAL_WRITER_ERROR_CODES:
        raise WorkflowCanonicalWriterContractError(
            "result.error.code is not an accepted canonical writer error code"
        )
    message = _require_nonempty(
        "result.error.message",
        value["message"],
    )
    retryable = value["retryable"]
    if not isinstance(retryable, bool):
        raise WorkflowCanonicalWriterContractError(
            "result.error.retryable must be bool"
        )

    if retryable and not (
        status == "failed" and code == "write_failed"
    ):
        raise WorkflowCanonicalWriterContractError(
            "retryable=true is allowed only for failed/write_failed"
        )
    if status == "rejected" and code == "write_failed":
        raise WorkflowCanonicalWriterContractError(
            "write_failed must use failed status"
        )
    if status == "failed" and code != "write_failed":
        raise WorkflowCanonicalWriterContractError(
            "failed status requires write_failed error code"
        )

    return {
        "code": code,
        "message": message,
        "retryable": retryable,
    }


def validate_canonical_writer_result(
    raw: Mapping[str, object],
) -> dict[str, object]:
    result = _require_mapping("result", raw)
    _require_exact_keys("result", result, RESULT_TOP_LEVEL_KEYS)

    if result["schema"] != WORKFLOW_CANONICAL_WRITER_RESULT_SCHEMA:
        raise WorkflowCanonicalWriterContractError(
            "result.schema does not match canonical writer result schema"
        )
    if result["schema_version"] != WORKFLOW_CANONICAL_WRITER_CONTRACT_VERSION:
        raise WorkflowCanonicalWriterContractError(
            "result.schema_version must equal 1"
        )

    request_id = _require_uuid(
        "result.request_id",
        result["request_id"],
    )
    idempotency_key = _require_sha256(
        "result.idempotency_key",
        result["idempotency_key"],
    )
    status = result["status"]
    if status not in CANONICAL_WRITER_RESULT_STATUSES:
        raise WorkflowCanonicalWriterContractError(
            "result.status is not accepted"
        )
    success = result["success"]
    if not isinstance(success, bool):
        raise WorkflowCanonicalWriterContractError(
            "result.success must be bool"
        )

    expected_success = status in CANONICAL_WRITER_SUCCESS_STATUSES
    if success is not expected_success:
        raise WorkflowCanonicalWriterContractError(
            "result.success must match result.status"
        )

    if expected_success:
        if result["error"] is not None:
            raise WorkflowCanonicalWriterContractError(
                "successful canonical writer result must have error=null"
            )
        publication = _validate_publication(result["publication"])
        error = None
    else:
        if result["publication"] is not None:
            raise WorkflowCanonicalWriterContractError(
                "failed canonical writer result must have publication=null"
            )
        publication = None
        error = _validate_error(result["error"], status=status)

    return {
        "schema": WORKFLOW_CANONICAL_WRITER_RESULT_SCHEMA,
        "schema_version": WORKFLOW_CANONICAL_WRITER_CONTRACT_VERSION,
        "request_id": request_id,
        "idempotency_key": idempotency_key,
        "status": status,
        "success": success,
        "publication": publication,
        "error": error,
    }


def assert_canonical_writer_result_matches_request(
    request_raw: Mapping[str, object],
    result_raw: Mapping[str, object],
) -> None:
    request = validate_canonical_writer_request(request_raw)
    result = validate_canonical_writer_result(result_raw)

    if result["request_id"] != request["request_id"]:
        raise WorkflowCanonicalWriterContractError(
            "result.request_id must echo request.request_id"
        )
    if result["idempotency_key"] != request["idempotency_key"]:
        raise WorkflowCanonicalWriterContractError(
            "result.idempotency_key must echo request.idempotency_key"
        )
    if result["success"]:
        publication = result["publication"]
        assert isinstance(publication, Mapping)
        if publication["semantic_sha256"] != request["payload"]["semantic_sha256"]:
            raise WorkflowCanonicalWriterContractError(
                "successful result semantic_sha256 must echo request payload"
            )
