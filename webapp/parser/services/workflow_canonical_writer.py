"""Internal governed canonical writer for Workflow publication.

This module implements the W5 application callback boundary against canonical
publication tables only. It does not authorize HTTP callers, register routes,
read or mutate Workflow tables, activate feature flags, or infer client
publication authority.

The runtime entry point owns one atomic canonical SQLAlchemy transaction.
Workflow publication orchestration is deliberately separate and may recover
after a canonical commit by replaying the deterministic W5 idempotency key.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import date, datetime, timezone
import hashlib
import json
import re
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from webapp.parser.contracts.workflow_canonical_writer import (
    WORKFLOW_CANONICAL_WRITER_CONTRACT_VERSION,
    WORKFLOW_CANONICAL_WRITER_REQUEST_SCHEMA,
    WORKFLOW_CANONICAL_WRITER_RESULT_SCHEMA,
    WorkflowCanonicalWriterContractError,
    assert_canonical_writer_result_matches_request,
    canonical_writer_idempotency_material,
    validate_canonical_writer_request,
    validate_canonical_writer_result,
)
from webapp.parser.utils.models import (
    CanonicalElectionRace,
    CanonicalElectionResult,
    CanonicalSourceArtifact,
    CanonicalVoteComponent,
)


WORKFLOW_CANONICAL_WRITER_RUNTIME_CONTRACT = (
    "workflow_canonical_writer_runtime_v1"
)
WORKFLOW_CANONICAL_WRITER_SERVICE_VERSION = "workflow-canonical-writer:v1"
WORKFLOW_CANONICAL_SOURCE_RACE_ID_RULE = (
    "CANONICAL_SOURCE_RACE_ID_EQUALS_WORKFLOW_ITEM_UUID"
)
WORKFLOW_CANONICAL_PAYLOAD_ARTIFACT_ROLE = "workflow_normalized_payload_v1"
WORKFLOW_CANONICAL_APPROVAL_ARTIFACT_ROLE = "workflow_publication_approval_v1"
WORKFLOW_CANONICAL_WRITE_SCOPE = (
    "CANONICAL_SOURCE_ARTIFACT_RACE_RESULT_VOTE_COMPONENT_TABLES_ONLY"
)
WORKFLOW_CANONICAL_NULL_MISSING_POLICY = (
    "REJECT_UNREPRESENTABLE_CANDIDATE_TOTAL_OR_METHOD_STATE_NO_COERCION"
)
WORKFLOW_CANONICAL_HANDOFF_EVENT_REQUIREMENT = (
    "CALLER_MUST_SUPPLY_DURABLE_WORKFLOW_HANDOFF_EVENT_ID_BEFORE_CANONICAL_COMMIT"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class WorkflowCanonicalWriterRuntimeError(RuntimeError):
    """Internal runtime canonical-writer failure."""


class WorkflowCanonicalWriterPrecondition(WorkflowCanonicalWriterRuntimeError):
    """Validated W5 request cannot be represented by canonical v1 tables."""


class WorkflowCanonicalWriterConflict(WorkflowCanonicalWriterRuntimeError):
    """Existing canonical state conflicts with governed publication authority."""


class WorkflowCanonicalWriterIdempotencyConflict(
    WorkflowCanonicalWriterRuntimeError
):
    """The Workflow item already published under different idempotency material."""


def _utc(value: datetime | None) -> datetime:
    observed = value or datetime.now(timezone.utc)
    if observed.tzinfo is None:
        observed = observed.replace(tzinfo=timezone.utc)
    return observed.astimezone(timezone.utc)


def _iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(domain: str, value: object) -> str:
    return hashlib.sha256(
        domain.encode("utf-8") + b"\x00" + _canonical_json(value)
    ).hexdigest()


def _failure_result(
    *,
    request_id: str,
    idempotency_key: str,
    code: str,
    message: str,
    status: str = "rejected",
    retryable: bool = False,
) -> dict[str, object]:
    result = {
        "schema": WORKFLOW_CANONICAL_WRITER_RESULT_SCHEMA,
        "schema_version": WORKFLOW_CANONICAL_WRITER_CONTRACT_VERSION,
        "request_id": request_id,
        "idempotency_key": idempotency_key,
        "status": status,
        "success": False,
        "publication": None,
        "error": {
            "code": code,
            "message": message,
            "retryable": retryable,
        },
    }
    validate_canonical_writer_result(result)
    return result


def _invalid_request_result(
    raw: Mapping[str, object],
    exc: Exception,
) -> dict[str, object]:
    request_id = str(
        raw.get("request_id")
        or "00000000-0000-0000-0000-000000000000"
    )
    try:
        import uuid
        request_id = str(uuid.UUID(request_id))
    except (TypeError, ValueError, AttributeError):
        request_id = "00000000-0000-0000-0000-000000000000"

    raw_key = str(raw.get("idempotency_key") or "")
    idempotency_key = (
        raw_key
        if _SHA256_RE.fullmatch(raw_key)
        else "0" * 64
    )
    return _failure_result(
        request_id=request_id,
        idempotency_key=idempotency_key,
        code="invalid_request",
        message=f"Canonical writer request rejected: {exc}",
    )


def _require_text(
    name: str,
    value: object,
    *,
    max_length: int,
) -> str:
    if not isinstance(value, str) or not value:
        raise WorkflowCanonicalWriterPrecondition(
            f"{name} must be present for canonical v1 publication."
        )
    if len(value) > max_length:
        raise WorkflowCanonicalWriterPrecondition(
            f"{name} exceeds canonical v1 maximum length {max_length}."
        )
    return value


def _require_optional_text(
    name: str,
    value: object,
    *,
    max_length: int,
) -> str | None:
    if value is None:
        return None
    return _require_text(name, value, max_length=max_length)


def _require_value_votes(name: str, raw: Mapping[str, object]) -> int:
    if raw.get("state") != "value":
        raise WorkflowCanonicalWriterPrecondition(
            f"{name} uses {raw.get('state')!r}; canonical v1 integer columns "
            "cannot represent null/missing without coercion."
        )
    votes = raw.get("votes")
    if isinstance(votes, bool) or not isinstance(votes, int) or votes < 0:
        raise WorkflowCanonicalWriterPrecondition(
            f"{name} must contain a nonnegative integer value."
        )
    return votes


def _scope_authority(
    request: Mapping[str, object],
) -> dict[str, object]:
    payload = request["payload"]
    assert isinstance(payload, Mapping)
    semantic = payload["semantic"]
    assert isinstance(semantic, Mapping)
    scope = semantic["scope"]
    assert isinstance(scope, Mapping)

    year = scope.get("election_year")
    if isinstance(year, bool) or not isinstance(year, int):
        raise WorkflowCanonicalWriterPrecondition(
            "semantic.scope.election_year is required by canonical v1."
        )
    state = _require_text(
        "semantic.scope.state",
        scope.get("state"),
        max_length=64,
    )
    contest = _require_text(
        "semantic.scope.contest",
        scope.get("contest"),
        max_length=128,
    )
    election_date = scope.get("election_date")
    parsed_date: date | None = None
    if election_date is not None:
        try:
            parsed_date = date.fromisoformat(str(election_date))
        except ValueError as exc:
            raise WorkflowCanonicalWriterPrecondition(
                "semantic.scope.election_date is not canonical YYYY-MM-DD."
            ) from exc

    return {
        "election_year": year,
        "election_date": parsed_date,
        "date_precision": "date" if parsed_date is not None else "year",
        "state": state,
        "contest": contest,
        "scope_jurisdiction_name": scope.get("jurisdiction_name"),
        "scope_jurisdiction_type": scope.get("jurisdiction_type"),
    }


def _flatten_rows(
    request: Mapping[str, object],
    *,
    state: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    payload = request["payload"]
    assert isinstance(payload, Mapping)
    semantic = payload["semantic"]
    assert isinstance(semantic, Mapping)
    records = semantic["records"]
    assert isinstance(records, list)

    rows: list[dict[str, object]] = []
    reporting_metadata: list[dict[str, object]] = []

    for record_index, record_raw in enumerate(records, start=1):
        assert isinstance(record_raw, Mapping)
        unit = record_raw["reporting_unit"]
        assert isinstance(unit, Mapping)
        unit_name = _require_text(
            f"semantic.records[{record_index - 1}].reporting_unit.name",
            unit.get("name"),
            max_length=256,
        )
        unit_type = _require_optional_text(
            f"semantic.records[{record_index - 1}].reporting_unit.type",
            unit.get("type"),
            max_length=32,
        )
        unit_type_key = unit_type if unit_type is not None else "jurisdiction"
        jurisdiction_key = f"{state}|{unit_type_key}|{unit_name}"
        if len(jurisdiction_key) > 384:
            raise WorkflowCanonicalWriterPrecondition(
                "Derived canonical jurisdiction_key exceeds v1 maximum length."
            )
        is_precinct = (
            unit_type is not None
            and unit_type.casefold() == "precinct"
        )

        vote_methods = record_raw["vote_methods"]
        candidates = record_raw["candidates"]
        assert isinstance(vote_methods, list)
        assert isinstance(candidates, list)

        for method_index, method in enumerate(vote_methods):
            _require_text(
                (
                    f"semantic.records[{record_index - 1}]."
                    f"vote_methods[{method_index}]"
                ),
                method,
                max_length=32,
            )

        reporting_metadata.append({
            "reporting_unit": dict(unit),
            "percent_reporting": record_raw["percent_reporting"],
            "method_totals": record_raw["method_totals"],
            "grand_total": record_raw["grand_total"],
        })

        for candidate_index, candidate_raw in enumerate(candidates):
            assert isinstance(candidate_raw, Mapping)
            candidate_name = _require_text(
                (
                    f"semantic.records[{record_index - 1}]."
                    f"candidates[{candidate_index}].name"
                ),
                candidate_raw.get("name"),
                max_length=512,
            )
            party = _require_optional_text(
                (
                    f"semantic.records[{record_index - 1}]."
                    f"candidates[{candidate_index}].party"
                ),
                candidate_raw.get("party"),
                max_length=64,
            )
            total_raw = candidate_raw["total_votes"]
            assert isinstance(total_raw, Mapping)
            total_votes = _require_value_votes(
                (
                    f"semantic.records[{record_index - 1}]."
                    f"candidates[{candidate_index}].total_votes"
                ),
                total_raw,
            )

            method_values = candidate_raw["method_votes"]
            assert isinstance(method_values, list)
            components: list[dict[str, object]] = []
            for method_index, component_raw in enumerate(method_values):
                assert isinstance(component_raw, Mapping)
                method = _require_text(
                    (
                        f"semantic.records[{record_index - 1}]."
                        f"candidates[{candidate_index}]."
                        f"method_votes[{method_index}].method"
                    ),
                    component_raw.get("method"),
                    max_length=32,
                )
                votes = _require_value_votes(
                    (
                        f"semantic.records[{record_index - 1}]."
                        f"candidates[{candidate_index}]."
                        f"method_votes[{method_index}]"
                    ),
                    component_raw,
                )
                components.append({
                    "vote_method": method,
                    "votes": votes,
                    "source_column": method,
                })

            row_material = {
                "scope": semantic["scope"],
                "reporting_unit": dict(unit),
                "candidate": dict(candidate_raw),
            }
            rows.append({
                "source_row_index": len(rows) + 1,
                "source_row_hash": _sha256_json(
                    "workflow-canonical-result-row-v1",
                    row_material,
                ),
                "source_jurisdiction_label": unit_name,
                "jurisdiction_key": jurisdiction_key,
                "jurisdiction_name": unit_name,
                "jurisdiction_type": unit_type,
                "aggregation_scope": (
                    "precinct" if is_precinct else "jurisdiction"
                ),
                "precinct": unit_name if is_precinct else None,
                "ballot_candidate_name": candidate_name,
                "candidate": candidate_name,
                "ballot_party": party,
                "party": party,
                "fec_id": None,
                "is_write_in": False,
                "total_votes": total_votes,
                "source_url": None,
                "provenance": {
                    "contract": WORKFLOW_CANONICAL_WRITER_RUNTIME_CONTRACT,
                    "source_record_index": record_index,
                    "source_candidate_index": candidate_index + 1,
                    "reporting_unit": dict(unit),
                    "percent_reporting": record_raw["percent_reporting"],
                    "method_totals": record_raw["method_totals"],
                    "grand_total": record_raw["grand_total"],
                    "write_in_authority": (
                        "W4_V1_HAS_NO_WRITE_IN_FIELD_COMPATIBILITY_FALSE_"
                        "IS_NONAUTHORITATIVE"
                    ),
                    "source_url_authority": (
                        "W5_REQUEST_HAS_NO_SOURCE_URL_NO_INFERENCE"
                    ),
                },
                "components": components,
            })

    if not rows:
        raise WorkflowCanonicalWriterPrecondition(
            "Canonical publication requires at least one candidate result row."
        )
    return rows, reporting_metadata


def _artifact_filename(
    artifact_ref: object,
) -> str:
    return _require_text(
        "payload.binding.normalized_artifact_ref",
        artifact_ref,
        max_length=512,
    )


def _get_existing_race(
    session: Session,
    workflow_item_id: str,
) -> CanonicalElectionRace | None:
    return session.execute(
        select(CanonicalElectionRace)
        .where(CanonicalElectionRace.source_race_id == workflow_item_id)
        .with_for_update()
    ).scalar_one_or_none()


def _publication_metadata(
    race: CanonicalElectionRace,
) -> Mapping[str, object] | None:
    metadata = race.qa_metadata
    if not isinstance(metadata, Mapping):
        return None
    publication = metadata.get("workflow_publication")
    return publication if isinstance(publication, Mapping) else None


def _payload_artifact_provenance(
    request: Mapping[str, object],
) -> dict[str, object]:
    workflow = request["workflow"]
    approval = request["approval"]
    payload = request["payload"]
    assert isinstance(workflow, Mapping)
    assert isinstance(approval, Mapping)
    assert isinstance(payload, Mapping)
    binding = payload["binding"]
    assert isinstance(binding, Mapping)
    return {
        "contract": WORKFLOW_CANONICAL_WRITER_RUNTIME_CONTRACT,
        "workflow_item_id": workflow["workflow_item_id"],
        "selected_pass_id": approval["selected_pass_id"],
        "selected_staging_batch_id": approval["selected_staging_batch_id"],
        "source_evidence_ref": binding["source_evidence_ref"],
        "normalized_artifact_ref": binding["normalized_artifact_ref"],
        "semantic_sha256": payload["semantic_sha256"],
    }


def _assert_artifact_identity(
    artifact: CanonicalSourceArtifact | None,
    *,
    role: str,
    filename: str,
    digest: str,
    row_count: int,
    provenance: Mapping[str, object],
) -> None:
    if artifact is None:
        raise WorkflowCanonicalWriterConflict(
            "Existing canonical race references a missing source artifact."
        )
    actual_provenance = (
        dict(artifact.provenance)
        if isinstance(artifact.provenance, Mapping)
        else None
    )
    if (
        artifact.artifact_role != role
        or artifact.filename != filename
        or artifact.sha256 != digest
        or artifact.row_count != row_count
        or artifact.race_count != 1
        or actual_provenance != dict(provenance)
    ):
        raise WorkflowCanonicalWriterConflict(
            "Existing canonical source artifact identity/provenance drifted."
        )


def _assert_existing_result_rows(
    session: Session,
    *,
    race: CanonicalElectionRace,
    expected_rows: list[dict[str, object]],
) -> int:
    actual_rows = session.execute(
        select(CanonicalElectionResult)
        .where(CanonicalElectionResult.race_id == race.id)
        .order_by(CanonicalElectionResult.source_row_index)
    ).scalars().all()
    if len(actual_rows) != len(expected_rows):
        raise WorkflowCanonicalWriterConflict(
            "Existing canonical result row count does not match publication."
        )

    component_count = 0
    scalar_fields = (
        "source_row_index",
        "source_row_hash",
        "source_jurisdiction_label",
        "jurisdiction_key",
        "jurisdiction_name",
        "jurisdiction_type",
        "aggregation_scope",
        "precinct",
        "ballot_candidate_name",
        "candidate",
        "ballot_party",
        "party",
        "fec_id",
        "is_write_in",
        "total_votes",
        "source_url",
    )
    for actual, expected in zip(actual_rows, expected_rows, strict=True):
        for field in scalar_fields:
            if getattr(actual, field) != expected[field]:
                raise WorkflowCanonicalWriterConflict(
                    f"Existing canonical result drifted at {field}."
                )
        actual_provenance = (
            dict(actual.provenance)
            if isinstance(actual.provenance, Mapping)
            else None
        )
        if actual_provenance != expected["provenance"]:
            raise WorkflowCanonicalWriterConflict(
                "Existing canonical result provenance drifted."
            )

        actual_components = session.execute(
            select(CanonicalVoteComponent)
            .where(CanonicalVoteComponent.result_id == actual.id)
            .order_by(CanonicalVoteComponent.vote_method)
        ).scalars().all()
        expected_components = sorted(
            expected["components"],
            key=lambda value: str(value["vote_method"]),
        )
        if len(actual_components) != len(expected_components):
            raise WorkflowCanonicalWriterConflict(
                "Existing canonical vote-component count drifted."
            )
        for actual_component, expected_component in zip(
            actual_components,
            expected_components,
            strict=True,
        ):
            if (
                actual_component.vote_method
                != expected_component["vote_method"]
                or actual_component.votes != expected_component["votes"]
                or actual_component.source_column
                != expected_component["source_column"]
            ):
                raise WorkflowCanonicalWriterConflict(
                    "Existing canonical vote-component semantics drifted."
                )
        component_count += len(actual_components)
    return component_count


def _existing_publication_result(
    session: Session,
    request: Mapping[str, object],
    race: CanonicalElectionRace,
) -> dict[str, object]:
    publication_metadata = _publication_metadata(race)
    if publication_metadata is None:
        raise WorkflowCanonicalWriterConflict(
            "Existing canonical race is not governed by Workflow writer v1."
        )

    idempotency_key = str(request["idempotency_key"])
    if publication_metadata.get("idempotency_key") != idempotency_key:
        raise WorkflowCanonicalWriterIdempotencyConflict(
            "Workflow item already has canonical publication under different "
            "idempotency material."
        )

    workflow = request["workflow"]
    payload = request["payload"]
    approval = request["approval"]
    comparison = request["comparison"]
    assert isinstance(workflow, Mapping)
    assert isinstance(payload, Mapping)
    assert isinstance(approval, Mapping)
    assert isinstance(comparison, Mapping)
    binding = payload["binding"]
    assert isinstance(binding, Mapping)

    scope = _scope_authority(request)
    expected_rows, reporting_metadata = _flatten_rows(
        request,
        state=str(scope["state"]),
    )
    pass_number = binding["pass_number"]
    if pass_number not in (1, 2):
        raise WorkflowCanonicalWriterConflict(
            "Existing publication request no longer maps to DL1/DL2."
        )
    expected_dl = f"DL{pass_number}"

    if (
        race.source_race_id != workflow["workflow_item_id"]
        or race.election_year != scope["election_year"]
        or race.election_date != scope["election_date"]
        or race.date_precision != scope["date_precision"]
        or race.state != scope["state"]
        or race.contest != scope["contest"]
        or race.office_basic is not None
        or race.production_status != "prod_loaded"
        or race.selected_dl_source != expected_dl
        or race.source_url is not None
        or race.verification_status != "verified"
        or race.verified_at is None
    ):
        raise WorkflowCanonicalWriterConflict(
            "Existing canonical race fields drifted from governed publication."
        )

    expected_metadata = {
        "contract": WORKFLOW_CANONICAL_WRITER_RUNTIME_CONTRACT,
        "idempotency_key": idempotency_key,
        "workflow_item_id": workflow["workflow_item_id"],
        "qc1_review_id": approval["qc1_review_id"],
        "qc2_review_id": approval["qc2_review_id"],
        "selected_pass_id": approval["selected_pass_id"],
        "selected_staging_batch_id": approval["selected_staging_batch_id"],
        "selected_pass_number": binding["pass_number"],
        "selected_pass_revision_number": binding["revision_number"],
        "source_evidence_ref": binding["source_evidence_ref"],
        "normalized_artifact_ref": binding["normalized_artifact_ref"],
        "comparison_id": comparison["comparison_id"],
        "comparison_version": comparison["comparison_version"],
        "strict_equality_passed": comparison["strict_equality_passed"],
        "open_discrepancy_count": comparison["open_discrepancy_count"],
        "semantic_sha256": payload["semantic_sha256"],
        "normalized_artifact_sha256": binding[
            "normalized_artifact_sha256"
        ],
        "source_race_id_rule": WORKFLOW_CANONICAL_SOURCE_RACE_ID_RULE,
        "source_url_authority": "W5_REQUEST_HAS_NO_SOURCE_URL_NO_INFERENCE",
        "office_basic_authority":
            "W5_REQUEST_HAS_NO_OFFICE_BASIC_NO_INFERENCE",
        "scope_jurisdiction_name": scope["scope_jurisdiction_name"],
        "scope_jurisdiction_type": scope["scope_jurisdiction_type"],
        "reporting_metadata": reporting_metadata,
    }
    for key, value in expected_metadata.items():
        if publication_metadata.get(key) != value:
            raise WorkflowCanonicalWriterConflict(
                f"Existing publication provenance mismatch for {key}."
            )

    attempt_keys = {
        "workflow_row_version",
        "publication_handoff_event_id",
        "publication_operator_principal",
        "committed_at",
        "writer_service_version",
    }
    if set(publication_metadata) != (
        set(expected_metadata) | attempt_keys
    ):
        raise WorkflowCanonicalWriterConflict(
            "Existing publication provenance key set drifted."
        )

    committed_at = publication_metadata.get("committed_at")
    writer_version = publication_metadata.get("writer_service_version")
    if not isinstance(committed_at, str) or not committed_at:
        raise WorkflowCanonicalWriterConflict(
            "Existing publication is missing committed_at provenance."
        )
    if writer_version != WORKFLOW_CANONICAL_WRITER_SERVICE_VERSION:
        raise WorkflowCanonicalWriterConflict(
            "Existing publication writer version is not v1."
        )

    payload_provenance = _payload_artifact_provenance(request)
    payload_artifact = session.get(
        CanonicalSourceArtifact,
        race.payload_artifact_id,
    )
    _assert_artifact_identity(
        payload_artifact,
        role=WORKFLOW_CANONICAL_PAYLOAD_ARTIFACT_ROLE,
        filename=_artifact_filename(binding["normalized_artifact_ref"]),
        digest=str(binding["normalized_artifact_sha256"]),
        row_count=len(expected_rows),
        provenance=payload_provenance,
    )

    approval_material = _approval_artifact_material(request)
    approval_digest = _sha256_json(
        "workflow-publication-approval-artifact-v1",
        approval_material,
    )
    approval_artifact = session.get(
        CanonicalSourceArtifact,
        race.approval_artifact_id,
    )
    _assert_artifact_identity(
        approval_artifact,
        role=WORKFLOW_CANONICAL_APPROVAL_ARTIFACT_ROLE,
        filename=(
            "workflow-publication-approval-"
            f"{idempotency_key}.json"
        ),
        digest=approval_digest,
        row_count=2,
        provenance=approval_material,
    )

    component_count = _assert_existing_result_rows(
        session,
        race=race,
        expected_rows=expected_rows,
    )

    result = {
        "schema": WORKFLOW_CANONICAL_WRITER_RESULT_SCHEMA,
        "schema_version": WORKFLOW_CANONICAL_WRITER_CONTRACT_VERSION,
        "request_id": request["request_id"],
        "idempotency_key": idempotency_key,
        "status": "already_published",
        "success": True,
        "publication": {
            "canonical_race_id": str(race.id),
            "canonical_source_artifact_id": str(race.payload_artifact_id),
            "canonical_result_count": len(expected_rows),
            "canonical_vote_component_count": component_count,
            "semantic_sha256": payload["semantic_sha256"],
            "committed_at": committed_at,
            "writer_service_version": WORKFLOW_CANONICAL_WRITER_SERVICE_VERSION,
        },
        "error": None,
    }
    validate_canonical_writer_result(result)
    assert_canonical_writer_result_matches_request(request, result)
    return result


def _get_or_create_artifact(
    session: Session,
    *,
    role: str,
    filename: str,
    digest: str,
    row_count: int,
    provenance: Mapping[str, object],
) -> CanonicalSourceArtifact:
    existing = session.execute(
        select(CanonicalSourceArtifact)
        .where(CanonicalSourceArtifact.sha256 == digest)
        .with_for_update()
    ).scalar_one_or_none()
    if existing is not None:
        if existing.artifact_role != role:
            raise WorkflowCanonicalWriterConflict(
                "Canonical source artifact hash already exists under a "
                "different immutable artifact role."
            )
        existing_provenance = (
            dict(existing.provenance)
            if isinstance(existing.provenance, Mapping)
            else None
        )
        if (
            existing.filename != filename
            or existing.race_count != 1
            or (
                existing.row_count is not None
                and int(existing.row_count) != int(row_count)
            )
            or existing_provenance != dict(provenance)
        ):
            raise WorkflowCanonicalWriterConflict(
                "Canonical source artifact hash exists with different immutable "
                "identity/provenance."
            )
        return existing

    artifact = CanonicalSourceArtifact(
        artifact_role=role,
        filename=filename,
        sha256=digest,
        row_count=row_count,
        race_count=1,
        provenance=dict(provenance),
    )
    session.add(artifact)
    session.flush()
    return artifact


def _approval_artifact_material(
    request: Mapping[str, object],
) -> dict[str, object]:
    return {
        "contract": WORKFLOW_CANONICAL_WRITER_RUNTIME_CONTRACT,
        "idempotency_material": canonical_writer_idempotency_material(request),
        "approval": request["approval"],
        "comparison": request["comparison"],
    }


def _publish_new(
    session: Session,
    request: Mapping[str, object],
    *,
    committed_at: datetime,
) -> dict[str, object]:
    workflow = request["workflow"]
    approval = request["approval"]
    comparison = request["comparison"]
    payload = request["payload"]
    assert isinstance(workflow, Mapping)
    assert isinstance(approval, Mapping)
    assert isinstance(comparison, Mapping)
    assert isinstance(payload, Mapping)

    workflow_item_id = str(workflow["workflow_item_id"])
    existing = _get_existing_race(session, workflow_item_id)
    if existing is not None:
        return _existing_publication_result(session, request, existing)

    binding = payload["binding"]
    assert isinstance(binding, Mapping)
    pass_number = binding["pass_number"]
    if pass_number not in (1, 2):
        raise WorkflowCanonicalWriterPrecondition(
            "Canonical v1 selected_dl_source supports only governed DL1/DL2."
        )
    selected_dl_source = f"DL{pass_number}"

    scope = _scope_authority(request)
    rows, reporting_metadata = _flatten_rows(
        request,
        state=str(scope["state"]),
    )
    result_count = len(rows)
    component_count = sum(
        len(row["components"])
        for row in rows
    )

    payload_digest = str(binding["normalized_artifact_sha256"])
    payload_artifact = _get_or_create_artifact(
        session,
        role=WORKFLOW_CANONICAL_PAYLOAD_ARTIFACT_ROLE,
        filename=_artifact_filename(binding["normalized_artifact_ref"]),
        digest=payload_digest,
        row_count=result_count,
        provenance=_payload_artifact_provenance(request),
    )

    approval_material = _approval_artifact_material(request)
    approval_digest = _sha256_json(
        "workflow-publication-approval-artifact-v1",
        approval_material,
    )
    approval_artifact = _get_or_create_artifact(
        session,
        role=WORKFLOW_CANONICAL_APPROVAL_ARTIFACT_ROLE,
        filename=(
            "workflow-publication-approval-"
            f"{request['idempotency_key']}.json"
        ),
        digest=approval_digest,
        row_count=2,
        provenance=approval_material,
    )

    committed_text = _iso_utc(committed_at)
    publication_metadata = {
        "contract": WORKFLOW_CANONICAL_WRITER_RUNTIME_CONTRACT,
        "idempotency_key": request["idempotency_key"],
        "workflow_item_id": workflow_item_id,
        "workflow_row_version": workflow["workflow_row_version"],
        "publication_handoff_event_id": workflow[
            "publication_handoff_event_id"
        ],
        "publication_operator_principal": workflow[
            "publication_operator_principal"
        ],
        "qc1_review_id": approval["qc1_review_id"],
        "qc2_review_id": approval["qc2_review_id"],
        "selected_pass_id": approval["selected_pass_id"],
        "selected_staging_batch_id": approval["selected_staging_batch_id"],
        "selected_pass_number": binding["pass_number"],
        "selected_pass_revision_number": binding["revision_number"],
        "source_evidence_ref": binding["source_evidence_ref"],
        "normalized_artifact_ref": binding["normalized_artifact_ref"],
        "comparison_id": comparison["comparison_id"],
        "comparison_version": comparison["comparison_version"],
        "strict_equality_passed": comparison["strict_equality_passed"],
        "open_discrepancy_count": comparison["open_discrepancy_count"],
        "normalized_artifact_sha256": payload_digest,
        "semantic_sha256": payload["semantic_sha256"],
        "committed_at": committed_text,
        "writer_service_version": WORKFLOW_CANONICAL_WRITER_SERVICE_VERSION,
        "source_race_id_rule": WORKFLOW_CANONICAL_SOURCE_RACE_ID_RULE,
        "source_url_authority": "W5_REQUEST_HAS_NO_SOURCE_URL_NO_INFERENCE",
        "office_basic_authority": (
            "W5_REQUEST_HAS_NO_OFFICE_BASIC_NO_INFERENCE"
        ),
        "scope_jurisdiction_name": scope["scope_jurisdiction_name"],
        "scope_jurisdiction_type": scope["scope_jurisdiction_type"],
        "reporting_metadata": reporting_metadata,
    }

    race = CanonicalElectionRace(
        source_race_id=workflow_item_id,
        election_year=scope["election_year"],
        election_date=scope["election_date"],
        date_precision=scope["date_precision"],
        state=scope["state"],
        contest=scope["contest"],
        office_basic=None,
        production_status="prod_loaded",
        selected_dl_source=selected_dl_source,
        source_url=None,
        verification_status="verified",
        verified_at=committed_at,
        payload_artifact_id=payload_artifact.id,
        approval_artifact_id=approval_artifact.id,
        qa_metadata={"workflow_publication": publication_metadata},
    )
    session.add(race)
    session.flush()

    for row in rows:
        components = row["components"]
        assert isinstance(components, list)
        canonical_result = CanonicalElectionResult(
            race_id=race.id,
            source_row_index=row["source_row_index"],
            source_row_hash=row["source_row_hash"],
            source_jurisdiction_label=row["source_jurisdiction_label"],
            jurisdiction_key=row["jurisdiction_key"],
            jurisdiction_name=row["jurisdiction_name"],
            jurisdiction_type=row["jurisdiction_type"],
            aggregation_scope=row["aggregation_scope"],
            precinct=row["precinct"],
            ballot_candidate_name=row["ballot_candidate_name"],
            candidate=row["candidate"],
            ballot_party=row["ballot_party"],
            party=row["party"],
            fec_id=row["fec_id"],
            is_write_in=row["is_write_in"],
            total_votes=row["total_votes"],
            source_url=row["source_url"],
            provenance=row["provenance"],
        )
        session.add(canonical_result)
        session.flush()
        for component in components:
            session.add(CanonicalVoteComponent(
                result_id=canonical_result.id,
                vote_method=component["vote_method"],
                votes=component["votes"],
                source_column=component["source_column"],
            ))

    session.flush()

    result = {
        "schema": WORKFLOW_CANONICAL_WRITER_RESULT_SCHEMA,
        "schema_version": WORKFLOW_CANONICAL_WRITER_CONTRACT_VERSION,
        "request_id": request["request_id"],
        "idempotency_key": request["idempotency_key"],
        "status": "published",
        "success": True,
        "publication": {
            "canonical_race_id": str(race.id),
            "canonical_source_artifact_id": str(payload_artifact.id),
            "canonical_result_count": result_count,
            "canonical_vote_component_count": component_count,
            "semantic_sha256": payload["semantic_sha256"],
            "committed_at": committed_text,
            "writer_service_version": WORKFLOW_CANONICAL_WRITER_SERVICE_VERSION,
        },
        "error": None,
    }
    validate_canonical_writer_result(result)
    assert_canonical_writer_result_matches_request(request, result)
    return result


def write_workflow_canonical_publication(
    request_raw: Mapping[str, object],
    *,
    session_factory: Callable[[], Session],
    now: datetime | None = None,
) -> dict[str, object]:
    """Validate and atomically publish one W5 request into canonical tables.

    The injected session_factory is the canonical database transaction boundary.
    The function never reads or mutates Workflow tables and never commits a
    partial canonical publication.
    """
    if not isinstance(request_raw, Mapping):
        raise WorkflowCanonicalWriterRuntimeError(
            "Canonical writer request must be a mapping."
        )
    if not callable(session_factory):
        raise WorkflowCanonicalWriterRuntimeError(
            "session_factory must be callable."
        )

    try:
        request = validate_canonical_writer_request(request_raw)
    except WorkflowCanonicalWriterContractError as exc:
        return _invalid_request_result(request_raw, exc)

    timestamp = _utc(now)
    try:
        with session_factory() as session:
            with session.begin():
                result = _publish_new(
                    session,
                    request,
                    committed_at=timestamp,
                )
        return result
    except WorkflowCanonicalWriterIdempotencyConflict as exc:
        return _failure_result(
            request_id=str(request["request_id"]),
            idempotency_key=str(request["idempotency_key"]),
            code="idempotency_conflict",
            message=str(exc),
        )
    except WorkflowCanonicalWriterConflict as exc:
        return _failure_result(
            request_id=str(request["request_id"]),
            idempotency_key=str(request["idempotency_key"]),
            code="canonical_conflict",
            message=str(exc),
        )
    except WorkflowCanonicalWriterPrecondition as exc:
        return _failure_result(
            request_id=str(request["request_id"]),
            idempotency_key=str(request["idempotency_key"]),
            code="precondition_failed",
            message=str(exc),
        )
    except SQLAlchemyError:
        return _failure_result(
            request_id=str(request["request_id"]),
            idempotency_key=str(request["idempotency_key"]),
            status="failed",
            code="write_failed",
            message=(
                "Canonical transaction failed and was rolled back; retry may "
                "reconcile an independently completed concurrent publication."
            ),
            retryable=True,
        )


def build_workflow_canonical_writer(
    session_factory: Callable[[], Session],
) -> Callable[[Mapping[str, object]], dict[str, object]]:
    """Return the exact W5 one-argument application callback."""
    if not callable(session_factory):
        raise WorkflowCanonicalWriterRuntimeError(
            "session_factory must be callable."
        )

    def canonical_writer(
        request: Mapping[str, object],
    ) -> dict[str, object]:
        return write_workflow_canonical_publication(
            request,
            session_factory=session_factory,
        )

    return canonical_writer
