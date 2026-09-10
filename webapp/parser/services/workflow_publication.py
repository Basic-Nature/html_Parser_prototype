"""Governed publication-handoff orchestration for canonical Workflow release.

This module is the application boundary between an already-approved Workflow
item and the internal W5 canonical-writer callback.  It deliberately does not
register HTTP routes or infer caller authorization.  The composition root must
first establish the publication-operator principal/capability and inject:

* a Workflow SQLAlchemy session factory,
* the internal one-argument ``canonical_writer(request)`` callback,
* the server-owned normalized-artifact loader, and
* the server-owned W4 comparison-payload adapter.

Publication is intentionally split into durable authorities rather than
pretending Workflow and canonical state share one atomic transaction:

1. a Workflow transaction freezes one append-only handoff-start event;
2. the canonical writer owns its independent canonical transaction;
3. a second Workflow transaction links the returned canonical identifiers.

If canonical persistence succeeds but the final Workflow link fails, canonical
state is never rolled back.  A later governed retry derives the same W5
idempotency material and the canonical writer reconciles it as
``already_published`` before Workflow linkage is retried.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
import hashlib
import json
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from webapp.parser.contracts.workflow_authorization import (
    assert_publication_operator_separation,
)
from webapp.parser.contracts.workflow_canonical_writer import (
    CANONICAL_WRITER_SUCCESS_STATUSES,
    WORKFLOW_CANONICAL_WRITER_CONTRACT_VERSION,
    WORKFLOW_CANONICAL_WRITER_REQUEST_SCHEMA,
    assert_canonical_writer_result_matches_request,
    derive_canonical_writer_idempotency_key,
    validate_canonical_writer_request,
    validate_canonical_writer_result,
)
from webapp.parser.services.workflow_pre_qc_validation import (
    WorkflowPreQCValidationError,
    validate_frozen_workflow_pre_qc_payload,
)
from webapp.parser.services.workflow_reviews import (
    WorkflowQCReviewConflict,
    load_workflow_publication_approval_authority,
)
from webapp.parser.services.workflow_staging_binding import (
    WorkflowStagingBindingError,
    validate_frozen_workflow_staging_binding,
)
from webapp.parser.utils.models import (
    WorkflowArtifactLink,
    WorkflowEvent,
    WorkflowItem,
)


WORKFLOW_PUBLICATION_HANDOFF_CONTRACT = "workflow_publication_handoff_v1"
WORKFLOW_PUBLICATION_SERVICE = "workflow_publication"
WORKFLOW_PUBLICATION_CANONICAL_RELATION = "canonical_publication"
WORKFLOW_PUBLICATION_CANONICAL_ARTIFACT_TYPE = "canonical_source_artifact"
WORKFLOW_PUBLICATION_TRANSACTION_MODEL = (
    "DURABLE_WORKFLOW_START_THEN_CANONICAL_TRANSACTION_THEN_WORKFLOW_LINK"
)
WORKFLOW_PUBLICATION_LINK_FAILURE_RECOVERY = (
    "IDEMPOTENT_WRITER_REPLAY_THEN_WORKFLOW_LINK_RETRY"
)
WORKFLOW_PUBLICATION_CLIENT_AUTHORITY = False

ArtifactLoader = Callable[[str], bytes]
ComparisonPayloadAdapter = Callable[[str, bytes, Mapping[str, object]], object]
CanonicalWriter = Callable[[Mapping[str, object]], Mapping[str, object]]
SessionFactory = Callable[[], Session]


class WorkflowPublicationError(RuntimeError):
    status_code = 400
    code = "workflow_publication_error"


class WorkflowPublicationNotFound(WorkflowPublicationError):
    status_code = 404
    code = "workflow_publication_not_found"


class WorkflowPublicationConflict(WorkflowPublicationError):
    status_code = 409
    code = "workflow_publication_conflict"


class WorkflowPublicationDependencyUnavailable(WorkflowPublicationError):
    status_code = 503
    code = "workflow_publication_dependency_unavailable"


class WorkflowPublicationLinkFailure(WorkflowPublicationError):
    """Canonical success is durable but Workflow linkage did not commit."""

    status_code = 503
    code = "workflow_publication_link_failure"

    def __init__(
        self,
        message: str,
        *,
        canonical_result: Mapping[str, object],
        idempotency_key: str,
    ) -> None:
        super().__init__(message)
        self.canonical_result = dict(canonical_result)
        self.idempotency_key = idempotency_key


class WorkflowPublicationWriterFailure(WorkflowPublicationError):
    """Canonical writer returned a valid governed failure result."""

    def __init__(self, payload: Mapping[str, object]) -> None:
        retryable = bool(payload.get("retryable"))
        self.status_code = 503 if retryable else 409
        self.code = "workflow_publication_writer_failed"
        self.payload = dict(payload)
        super().__init__(str(payload.get("message") or "Canonical writer failed."))


def _utc(now: datetime | None) -> datetime:
    value = now or datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _uuid(value: UUID | str, *, name: str) -> UUID:
    try:
        return value if isinstance(value, UUID) else UUID(str(value))
    except (TypeError, ValueError) as exc:
        raise WorkflowPublicationError(f"{name} must be a UUID.") from exc


def _expected_version(value: object) -> int:
    if isinstance(value, bool):
        raise WorkflowPublicationError("expected_row_version must be an integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise WorkflowPublicationError(
            "expected_row_version must be an integer."
        ) from exc
    if parsed < 1:
        raise WorkflowPublicationError("expected_row_version must be >= 1.")
    return parsed


def _principal(value: object) -> str:
    principal = str(value or "").strip()
    if not principal:
        raise WorkflowPublicationError(
            "Authenticated publication operator principal is required."
        )
    return principal


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    ).encode("utf-8")


def _request_sha256(request: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(request)).hexdigest()


def _require_dependencies(
    *,
    workflow_session_factory: SessionFactory,
    canonical_writer: CanonicalWriter,
    normalized_artifact_loader: ArtifactLoader,
    comparison_payload_adapter: ComparisonPayloadAdapter,
) -> None:
    for name, value in (
        ("workflow_session_factory", workflow_session_factory),
        ("canonical_writer", canonical_writer),
        ("normalized_artifact_loader", normalized_artifact_loader),
        ("comparison_payload_adapter", comparison_payload_adapter),
    ):
        if not callable(value):
            raise WorkflowPublicationDependencyUnavailable(
                f"{name} is not configured."
            )


def _snapshot(item: WorkflowItem) -> dict[str, object]:
    return {
        "lifecycle_state": item.lifecycle_state,
        "current_stage": item.current_stage,
        "stage_condition": item.stage_condition,
        "canonical_race_id": (
            str(item.canonical_race_id)
            if item.canonical_race_id is not None
            else None
        ),
        "row_version": int(item.row_version),
    }


def _validate_selected_payload(
    session: Session,
    authority: Mapping[str, object],
    *,
    normalized_artifact_loader: ArtifactLoader,
    comparison_payload_adapter: ComparisonPayloadAdapter,
) -> tuple[dict[str, object], dict[str, object]]:
    item = authority["item"]
    selected = authority["selected_pass"]
    assert isinstance(item, WorkflowItem)

    try:
        frozen = validate_frozen_workflow_staging_binding(
            session,
            item.id,
            selected.id,
            selected.staging_batch_id,
            required_pass_number=int(selected.pass_number),
            required_pass_label=str(selected.pass_label),
        )
    except WorkflowStagingBindingError as exc:
        raise WorkflowPublicationConflict(
            f"Selected publication staging authority failed: {exc}"
        ) from exc

    artifact_ref = str(frozen["artifact_ref"])
    expected_hash = str(frozen["artifact_sha256"])
    try:
        raw = normalized_artifact_loader(artifact_ref)
    except Exception as exc:
        raise WorkflowPublicationDependencyUnavailable(
            "Server-owned normalized artifact loader failed."
        ) from exc
    if not isinstance(raw, bytes):
        raise WorkflowPublicationDependencyUnavailable(
            "Server-owned normalized artifact loader must return bytes."
        )
    if hashlib.sha256(raw).hexdigest() != expected_hash:
        raise WorkflowPublicationConflict(
            "Selected normalized artifact bytes do not match frozen SHA-256."
        )

    binding = frozen.get("comparison_binding")
    if not isinstance(binding, Mapping):
        raise WorkflowPublicationConflict(
            "Selected staging authority is missing the W4 comparison binding."
        )
    try:
        adapted = comparison_payload_adapter(
            artifact_ref,
            raw,
            dict(binding),
        )
    except Exception as exc:
        raise WorkflowPublicationDependencyUnavailable(
            "Server-owned comparison payload adapter failed."
        ) from exc

    try:
        payload = validate_frozen_workflow_pre_qc_payload(
            selected,
            frozen,
            adapted,
        )
    except WorkflowPreQCValidationError as exc:
        raise WorkflowPublicationConflict(
            f"Selected frozen Pre-QC authority failed: {exc}"
        ) from exc
    return frozen, payload


def _build_w5_request(
    *,
    item: WorkflowItem,
    authority: Mapping[str, object],
    payload: Mapping[str, object],
    handoff_event_id: UUID,
    principal: str,
    expected_row_version: int,
) -> dict[str, object]:
    comparison = authority["comparison"]
    selected = authority["selected_pass"]
    qc1 = authority["qc1_review"]
    qc2 = authority["qc2_review"]

    request: dict[str, object] = {
        "schema": WORKFLOW_CANONICAL_WRITER_REQUEST_SCHEMA,
        "schema_version": WORKFLOW_CANONICAL_WRITER_CONTRACT_VERSION,
        "request_id": str(uuid4()),
        "idempotency_key": "0" * 64,
        "workflow": {
            "workflow_item_id": str(item.id),
            "workflow_row_version": expected_row_version,
            "publication_handoff_event_id": str(handoff_event_id),
            "publication_operator_principal": principal,
        },
        "approval": {
            "qc1_review_id": str(qc1.id),
            "qc2_review_id": str(qc2.id),
            "qc1_decision": "approved",
            "qc2_decision": "approved",
            "selected_pass_id": str(selected.id),
            "selected_staging_batch_id": str(selected.staging_batch_id),
        },
        "comparison": {
            "comparison_id": str(comparison.id),
            "comparison_version": int(comparison.comparison_version),
            "status": "complete",
            "strict_equality_passed": bool(comparison.strict_equality_passed),
            "open_discrepancy_count": 0,
        },
        "payload": dict(payload),
    }
    request["idempotency_key"] = derive_canonical_writer_idempotency_key(
        request
    )
    return validate_canonical_writer_request(request)


def _load_authority(
    session: Session,
    item_id: UUID,
    *,
    principal: str,
    require_ready_state: bool,
) -> dict[str, object]:
    try:
        authority = load_workflow_publication_approval_authority(
            session,
            item_id,
            require_ready_state=require_ready_state,
        )
    except WorkflowQCReviewConflict as exc:
        raise WorkflowPublicationConflict(str(exc)) from exc

    try:
        assert_publication_operator_separation(
            dl1_principal=str(authority["dl1_principal"]),
            dl2_principal=str(authority["dl2_principal"]),
            qc1_principal=str(authority["qc1_principal"]),
            qc2_principal=str(authority["qc2_principal"]),
            publication_operator_principal=principal,
        )
    except ValueError as exc:
        raise WorkflowPublicationConflict(str(exc)) from exc
    return authority


def _published_replay(
    workflow_session_factory: SessionFactory,
    item_id: UUID,
    *,
    principal: str,
    expected_row_version: int,
) -> dict[str, object] | None:
    with workflow_session_factory() as session:
        item = session.execute(
            select(WorkflowItem)
            .where(WorkflowItem.id == item_id)
            .with_for_update()
        ).scalar_one_or_none()
        if item is None:
            raise WorkflowPublicationNotFound("Workflow item was not found.")
        if (
            item.lifecycle_state,
            item.current_stage,
            item.stage_condition,
        ) != ("published", "publication_handoff", "complete"):
            return None
        if int(item.row_version) != expected_row_version:
            raise WorkflowPublicationConflict(
                "Workflow row_version changed before publication replay."
            )
        if item.canonical_race_id is None:
            raise WorkflowPublicationConflict(
                "Published Workflow item is missing canonical_race_id."
            )

        authority = _load_authority(
            session,
            item.id,
            principal=principal,
            require_ready_state=False,
        )
        completed = session.execute(
            select(WorkflowEvent).where(
                WorkflowEvent.workflow_item_id == item.id,
                WorkflowEvent.event_type == "publication_handoff_completed",
            )
        ).scalars().all()
        if len(completed) != 1:
            raise WorkflowPublicationConflict(
                "Published Workflow item does not have one exact completion event."
            )
        event = completed[0]
        metadata = (
            event.event_metadata
            if isinstance(event.event_metadata, Mapping)
            else {}
        )
        if (
            event.actor_type != "service"
            or event.actor_principal != principal
            or event.actor_service != WORKFLOW_PUBLICATION_SERVICE
            or event.stage != "publication_handoff"
            or event.related_pass_id != authority["selected_pass"].id
            or event.related_staging_batch_id
            != authority["selected_pass"].staging_batch_id
            or event.related_comparison_id != authority["comparison"].id
            or event.related_review_id != authority["qc2_review"].id
            or event.related_canonical_race_id != item.canonical_race_id
            or metadata.get("contract") != WORKFLOW_PUBLICATION_HANDOFF_CONTRACT
            or metadata.get("writer_status") not in CANONICAL_WRITER_SUCCESS_STATUSES
            or metadata.get("canonical_race_id") != str(item.canonical_race_id)
            or not metadata.get("canonical_source_artifact_id")
            or not metadata.get("idempotency_key")
            or not metadata.get("handoff_event_id")
        ):
            raise WorkflowPublicationConflict(
                "Published Workflow completion provenance does not reconcile."
            )

        links = session.execute(
            select(WorkflowArtifactLink).where(
                WorkflowArtifactLink.workflow_item_id == item.id,
                WorkflowArtifactLink.relation_type
                == WORKFLOW_PUBLICATION_CANONICAL_RELATION,
            )
        ).scalars().all()
        if len(links) != 1:
            raise WorkflowPublicationConflict(
                "Published Workflow item requires one canonical artifact link."
            )
        link = links[0]
        if (
            link.pass_id != authority["selected_pass"].id
            or link.staging_batch_id
            != authority["selected_pass"].staging_batch_id
            or str(link.canonical_source_artifact_id)
            != metadata.get("canonical_source_artifact_id")
            or link.artifact_sha256 != metadata.get("normalized_artifact_sha256")
        ):
            raise WorkflowPublicationConflict(
                "Published canonical artifact linkage does not reconcile."
            )

        return {
            "success": True,
            "contract": WORKFLOW_PUBLICATION_HANDOFF_CONTRACT,
            "task_id": str(item.id),
            "canonical_race_id": str(item.canonical_race_id),
            "canonical_source_artifact_id": metadata["canonical_source_artifact_id"],
            "canonical_result_count": metadata.get("canonical_result_count"),
            "canonical_vote_component_count": metadata.get(
                "canonical_vote_component_count"
            ),
            "semantic_sha256": metadata.get("semantic_sha256"),
            "idempotency_key": metadata["idempotency_key"],
            "handoff_event_id": metadata["handoff_event_id"],
            "completion_event_id": str(event.id),
            "row_version": int(item.row_version),
            "already_workflow_published": True,
            "canonical_writer_invoked": False,
            "committed": False,
        }


def _prepare_attempt(
    workflow_session_factory: SessionFactory,
    item_id: UUID,
    *,
    principal: str,
    expected_row_version: int,
    normalized_artifact_loader: ArtifactLoader,
    comparison_payload_adapter: ComparisonPayloadAdapter,
    now: datetime | None,
) -> tuple[dict[str, object], dict[str, object]]:
    timestamp = _utc(now)
    with workflow_session_factory() as session:
        with session.begin():
            authority = _load_authority(
                session,
                item_id,
                principal=principal,
                require_ready_state=True,
            )
            item = authority["item"]
            assert isinstance(item, WorkflowItem)
            if int(item.row_version) != expected_row_version:
                raise WorkflowPublicationConflict(
                    "Workflow row_version changed before publication handoff."
                )
            frozen, payload = _validate_selected_payload(
                session,
                authority,
                normalized_artifact_loader=normalized_artifact_loader,
                comparison_payload_adapter=comparison_payload_adapter,
            )

            prior_state = _snapshot(item)
            handoff = WorkflowEvent(
                workflow_item_id=item.id,
                actor_type="principal",
                actor_principal=principal,
                actor_service=None,
                event_type="publication_handoff_started",
                stage="publication_handoff",
                prior_state=prior_state,
                new_state=prior_state,
                related_pass_id=authority["selected_pass"].id,
                related_comparison_id=authority["comparison"].id,
                related_review_id=authority["qc2_review"].id,
                related_staging_batch_id=authority["selected_pass"].staging_batch_id,
                related_canonical_race_id=None,
                reason_code=None,
                summary="Governed canonical publication handoff started.",
                event_metadata={},
                occurred_at=timestamp,
            )
            session.add(handoff)
            session.flush()

            request = _build_w5_request(
                item=item,
                authority=authority,
                payload=payload,
                handoff_event_id=handoff.id,
                principal=principal,
                expected_row_version=expected_row_version,
            )
            binding = payload["binding"]
            handoff.event_metadata = {
                "contract": WORKFLOW_PUBLICATION_HANDOFF_CONTRACT,
                "transaction_model": WORKFLOW_PUBLICATION_TRANSACTION_MODEL,
                "link_failure_recovery": WORKFLOW_PUBLICATION_LINK_FAILURE_RECOVERY,
                "request_schema": WORKFLOW_CANONICAL_WRITER_REQUEST_SCHEMA,
                "request_id": request["request_id"],
                "request_sha256": _request_sha256(request),
                "idempotency_key": request["idempotency_key"],
                "workflow_row_version": expected_row_version,
                "qc1_review_id": str(authority["qc1_review"].id),
                "qc2_review_id": str(authority["qc2_review"].id),
                "selected_pass_id": str(authority["selected_pass"].id),
                "selected_staging_batch_id": str(
                    authority["selected_pass"].staging_batch_id
                ),
                "comparison_id": str(authority["comparison"].id),
                "normalized_artifact_ref": binding["normalized_artifact_ref"],
                "normalized_artifact_sha256": binding[
                    "normalized_artifact_sha256"
                ],
                "semantic_sha256": payload["semantic_sha256"],
                "canonical_writer_invoked": False,
                "canonical_link_completed": False,
                "client_publication_authority": False,
            }
            session.flush()

            return request, {
                "handoff_event_id": str(handoff.id),
                "request_sha256": handoff.event_metadata["request_sha256"],
                "selected_pass_id": str(authority["selected_pass"].id),
                "selected_staging_batch_id": str(
                    authority["selected_pass"].staging_batch_id
                ),
                "comparison_id": str(authority["comparison"].id),
                "qc1_review_id": str(authority["qc1_review"].id),
                "qc2_review_id": str(authority["qc2_review"].id),
                "normalized_artifact_sha256": str(frozen["artifact_sha256"]),
                "semantic_sha256": str(payload["semantic_sha256"]),
            }


def _verify_start_event(
    session: Session,
    *,
    item: WorkflowItem,
    authority: Mapping[str, object],
    principal: str,
    expected_row_version: int,
    request: Mapping[str, object],
    attempt: Mapping[str, object],
) -> WorkflowEvent:
    handoff_id = _uuid(attempt["handoff_event_id"], name="handoff_event_id")
    event = session.get(WorkflowEvent, handoff_id)
    if event is None:
        raise WorkflowPublicationConflict(
            "Durable publication handoff event was not found."
        )
    metadata = (
        event.event_metadata if isinstance(event.event_metadata, Mapping) else {}
    )
    if (
        event.workflow_item_id != item.id
        or event.event_type != "publication_handoff_started"
        or event.actor_type != "principal"
        or event.actor_principal != principal
        or event.actor_service is not None
        or event.stage != "publication_handoff"
        or event.related_pass_id != authority["selected_pass"].id
        or event.related_comparison_id != authority["comparison"].id
        or event.related_review_id != authority["qc2_review"].id
        or event.related_staging_batch_id
        != authority["selected_pass"].staging_batch_id
        or metadata.get("contract") != WORKFLOW_PUBLICATION_HANDOFF_CONTRACT
        or metadata.get("request_id") != request["request_id"]
        or metadata.get("request_sha256") != _request_sha256(request)
        or metadata.get("idempotency_key") != request["idempotency_key"]
        or metadata.get("workflow_row_version") != expected_row_version
        or metadata.get("canonical_link_completed") is not False
        or metadata.get("client_publication_authority") is not False
    ):
        raise WorkflowPublicationConflict(
            "Durable publication handoff event no longer matches W5 authority."
        )
    return event



def _record_untrusted_writer_failure(
    workflow_session_factory: SessionFactory,
    item_id: UUID,
    *,
    principal: str,
    expected_row_version: int,
    request: Mapping[str, object],
    attempt: Mapping[str, object],
    failure_code: str,
    message: str,
    now: datetime | None,
) -> str:
    """Persist fail-closed audit when writer durability/result is untrusted."""
    timestamp = _utc(now)
    with workflow_session_factory() as session:
        with session.begin():
            authority = _load_authority(
                session,
                item_id,
                principal=principal,
                require_ready_state=True,
            )
            item = authority["item"]
            assert isinstance(item, WorkflowItem)
            if int(item.row_version) != expected_row_version:
                raise WorkflowPublicationConflict(
                    "Workflow row_version changed before writer-failure audit."
                )
            handoff = _verify_start_event(
                session,
                item=item,
                authority=authority,
                principal=principal,
                expected_row_version=expected_row_version,
                request=request,
                attempt=attempt,
            )
            state = _snapshot(item)
            failed = WorkflowEvent(
                workflow_item_id=item.id,
                actor_type="service",
                actor_principal=principal,
                actor_service=WORKFLOW_PUBLICATION_SERVICE,
                event_type="publication_handoff_failed",
                stage="publication_handoff",
                prior_state=state,
                new_state=state,
                related_pass_id=authority["selected_pass"].id,
                related_comparison_id=authority["comparison"].id,
                related_review_id=authority["qc2_review"].id,
                related_staging_batch_id=authority["selected_pass"].staging_batch_id,
                related_canonical_race_id=None,
                reason_code=failure_code,
                summary=message,
                event_metadata={
                    "contract": WORKFLOW_PUBLICATION_HANDOFF_CONTRACT,
                    "handoff_event_id": str(handoff.id),
                    "request_id": request["request_id"],
                    "idempotency_key": request["idempotency_key"],
                    "writer_status": "untrusted",
                    "writer_error_code": failure_code,
                    "writer_retryable": False,
                    "canonical_durability": "unknown",
                    "canonical_writer_invoked": True,
                    "canonical_link_completed": False,
                    "automatic_retry_allowed": False,
                },
                occurred_at=timestamp,
            )
            session.add(failed)
            session.flush()
            return str(failed.id)

def _record_writer_failure(
    workflow_session_factory: SessionFactory,
    item_id: UUID,
    *,
    principal: str,
    expected_row_version: int,
    request: Mapping[str, object],
    attempt: Mapping[str, object],
    writer_result: Mapping[str, object],
    now: datetime | None,
) -> dict[str, object]:
    timestamp = _utc(now)
    error = writer_result.get("error")
    if not isinstance(error, Mapping):
        raise WorkflowPublicationConflict(
            "Canonical writer failure result is missing error authority."
        )
    retryable = bool(error.get("retryable"))
    if retryable != (
        writer_result.get("status") == "failed"
        and error.get("code") == "write_failed"
    ):
        raise WorkflowPublicationConflict(
            "Only failed/write_failed canonical results may be retryable."
        )

    with workflow_session_factory() as session:
        with session.begin():
            authority = _load_authority(
                session,
                item_id,
                principal=principal,
                require_ready_state=True,
            )
            item = authority["item"]
            assert isinstance(item, WorkflowItem)
            if int(item.row_version) != expected_row_version:
                raise WorkflowPublicationConflict(
                    "Workflow row_version changed before publication failure audit."
                )
            handoff = _verify_start_event(
                session,
                item=item,
                authority=authority,
                principal=principal,
                expected_row_version=expected_row_version,
                request=request,
                attempt=attempt,
            )
            prior_state = _snapshot(item)
            failed = WorkflowEvent(
                workflow_item_id=item.id,
                actor_type="service",
                actor_principal=principal,
                actor_service=WORKFLOW_PUBLICATION_SERVICE,
                event_type="publication_handoff_failed",
                stage="publication_handoff",
                prior_state=prior_state,
                new_state=prior_state,
                related_pass_id=authority["selected_pass"].id,
                related_comparison_id=authority["comparison"].id,
                related_review_id=authority["qc2_review"].id,
                related_staging_batch_id=authority["selected_pass"].staging_batch_id,
                related_canonical_race_id=None,
                reason_code=str(error.get("code") or "canonical_writer_failed"),
                summary="Canonical publication writer did not complete publication.",
                event_metadata={
                    "contract": WORKFLOW_PUBLICATION_HANDOFF_CONTRACT,
                    "handoff_event_id": str(handoff.id),
                    "request_id": request["request_id"],
                    "idempotency_key": request["idempotency_key"],
                    "writer_status": writer_result.get("status"),
                    "writer_error_code": error.get("code"),
                    "writer_retryable": retryable,
                    "canonical_writer_invoked": True,
                    "canonical_link_completed": False,
                },
                occurred_at=timestamp,
            )
            session.add(failed)
            session.flush()
            return {
                "success": False,
                "contract": WORKFLOW_PUBLICATION_HANDOFF_CONTRACT,
                "task_id": str(item.id),
                "handoff_event_id": str(handoff.id),
                "failure_event_id": str(failed.id),
                "idempotency_key": request["idempotency_key"],
                "row_version": int(item.row_version),
                "writer_status": writer_result.get("status"),
                "writer_error_code": error.get("code"),
                "retryable": retryable,
                "workflow_state_preserved": True,
                "canonical_writer_invoked": True,
                "canonical_link_completed": False,
                "message": str(error.get("message") or "Canonical writer failed."),
                "committed": True,
            }


def _finalize_success(
    workflow_session_factory: SessionFactory,
    item_id: UUID,
    *,
    principal: str,
    expected_row_version: int,
    request: Mapping[str, object],
    attempt: Mapping[str, object],
    writer_result: Mapping[str, object],
    now: datetime | None,
) -> dict[str, object]:
    timestamp = _utc(now)
    publication = writer_result.get("publication")
    if not isinstance(publication, Mapping):
        raise WorkflowPublicationConflict(
            "Successful canonical writer result is missing publication authority."
        )
    canonical_race_id = _uuid(
        publication["canonical_race_id"],
        name="canonical_race_id",
    )
    canonical_source_artifact_id = _uuid(
        publication["canonical_source_artifact_id"],
        name="canonical_source_artifact_id",
    )

    try:
        with workflow_session_factory() as session:
            with session.begin():
                authority = _load_authority(
                    session,
                    item_id,
                    principal=principal,
                    require_ready_state=True,
                )
                item = authority["item"]
                assert isinstance(item, WorkflowItem)
                if int(item.row_version) != expected_row_version:
                    raise WorkflowPublicationConflict(
                        "Workflow row_version changed before canonical linkage."
                    )
                if item.canonical_race_id is not None:
                    raise WorkflowPublicationConflict(
                        "Workflow item acquired canonical linkage before finalize."
                    )
                handoff = _verify_start_event(
                    session,
                    item=item,
                    authority=authority,
                    principal=principal,
                    expected_row_version=expected_row_version,
                    request=request,
                    attempt=attempt,
                )

                existing_links = session.execute(
                    select(WorkflowArtifactLink).where(
                        WorkflowArtifactLink.workflow_item_id == item.id,
                        WorkflowArtifactLink.relation_type
                        == WORKFLOW_PUBLICATION_CANONICAL_RELATION,
                    )
                ).scalars().all()
                if existing_links:
                    raise WorkflowPublicationConflict(
                        "Workflow item already contains canonical publication linkage."
                    )
                existing_completed = session.execute(
                    select(WorkflowEvent.id).where(
                        WorkflowEvent.workflow_item_id == item.id,
                        WorkflowEvent.event_type == "publication_handoff_completed",
                    )
                ).all()
                if existing_completed:
                    raise WorkflowPublicationConflict(
                        "Workflow item already contains publication completion authority."
                    )

                prior_state = _snapshot(item)
                selected = authority["selected_pass"]
                link = WorkflowArtifactLink(
                    workflow_item_id=item.id,
                    pass_id=selected.id,
                    relation_type=WORKFLOW_PUBLICATION_CANONICAL_RELATION,
                    artifact_type=WORKFLOW_PUBLICATION_CANONICAL_ARTIFACT_TYPE,
                    artifact_ref=(
                        "canonical://source_artifacts/"
                        f"{canonical_source_artifact_id}"
                    ),
                    artifact_sha256=str(attempt["normalized_artifact_sha256"]),
                    canonical_source_artifact_id=canonical_source_artifact_id,
                    staging_batch_id=selected.staging_batch_id,
                    artifact_metadata={
                        "contract": WORKFLOW_PUBLICATION_HANDOFF_CONTRACT,
                        "canonical_race_id": str(canonical_race_id),
                        "idempotency_key": request["idempotency_key"],
                        "handoff_event_id": str(handoff.id),
                        "semantic_sha256": publication["semantic_sha256"],
                        "writer_status": writer_result["status"],
                    },
                    created_at=timestamp,
                )
                session.add(link)
                session.flush()

                item.canonical_race_id = canonical_race_id
                item.lifecycle_state = "published"
                item.current_stage = "publication_handoff"
                item.stage_condition = "complete"
                item.blocked_reason_code = None
                item.blocker_detail = None
                item.row_version = expected_row_version + 1
                item.updated_at = timestamp
                new_state = _snapshot(item)

                completed = WorkflowEvent(
                    workflow_item_id=item.id,
                    actor_type="service",
                    actor_principal=principal,
                    actor_service=WORKFLOW_PUBLICATION_SERVICE,
                    event_type="publication_handoff_completed",
                    stage="publication_handoff",
                    prior_state=prior_state,
                    new_state=new_state,
                    related_pass_id=selected.id,
                    related_comparison_id=authority["comparison"].id,
                    related_review_id=authority["qc2_review"].id,
                    related_staging_batch_id=selected.staging_batch_id,
                    related_canonical_race_id=canonical_race_id,
                    reason_code=None,
                    summary="Governed canonical publication handoff completed.",
                    event_metadata={
                        "contract": WORKFLOW_PUBLICATION_HANDOFF_CONTRACT,
                        "handoff_event_id": str(handoff.id),
                        "request_id": request["request_id"],
                        "idempotency_key": request["idempotency_key"],
                        "writer_status": writer_result["status"],
                        "canonical_race_id": str(canonical_race_id),
                        "canonical_source_artifact_id": str(
                            canonical_source_artifact_id
                        ),
                        "canonical_result_count": publication[
                            "canonical_result_count"
                        ],
                        "canonical_vote_component_count": publication[
                            "canonical_vote_component_count"
                        ],
                        "semantic_sha256": publication["semantic_sha256"],
                        "normalized_artifact_sha256": attempt[
                            "normalized_artifact_sha256"
                        ],
                        "canonical_writer_invoked": True,
                        "canonical_link_completed": True,
                        "client_publication_authority": False,
                    },
                    occurred_at=timestamp,
                )
                session.add(completed)
                session.flush()

                return {
                    "success": True,
                    "contract": WORKFLOW_PUBLICATION_HANDOFF_CONTRACT,
                    "task_id": str(item.id),
                    "canonical_race_id": str(canonical_race_id),
                    "canonical_source_artifact_id": str(
                        canonical_source_artifact_id
                    ),
                    "canonical_result_count": publication[
                        "canonical_result_count"
                    ],
                    "canonical_vote_component_count": publication[
                        "canonical_vote_component_count"
                    ],
                    "semantic_sha256": publication["semantic_sha256"],
                    "idempotency_key": request["idempotency_key"],
                    "handoff_event_id": str(handoff.id),
                    "canonical_artifact_link_id": str(link.id),
                    "completion_event_id": str(completed.id),
                    "row_version": int(item.row_version),
                    "writer_status": writer_result["status"],
                    "already_canonical_published": (
                        writer_result["status"] == "already_published"
                    ),
                    "already_workflow_published": False,
                    "canonical_writer_invoked": True,
                    "canonical_link_completed": True,
                    "committed": True,
                }
    except WorkflowPublicationLinkFailure:
        raise
    except Exception as exc:
        raise WorkflowPublicationLinkFailure(
            "Canonical publication succeeded but Workflow linkage did not complete; "
            "canonical state must not be rolled back and recovery must replay the "
            "same idempotency material before Workflow linkage is retried.",
            canonical_result=writer_result,
            idempotency_key=str(request["idempotency_key"]),
        ) from exc


def publish_workflow_item(
    item_id: UUID | str,
    *,
    principal: str,
    expected_row_version: int,
    workflow_session_factory: SessionFactory,
    canonical_writer: CanonicalWriter,
    normalized_artifact_loader: ArtifactLoader,
    comparison_payload_adapter: ComparisonPayloadAdapter,
    now: datetime | None = None,
) -> dict[str, object]:
    """Publish one QC-approved Workflow item through the frozen W5 callback.

    No client-selected pass, staging batch, QC review, comparison, payload,
    artifact hash, idempotency key, canonical identifier, or canonical value is
    accepted.  Every publication authority is reconstructed server-side.
    """
    _require_dependencies(
        workflow_session_factory=workflow_session_factory,
        canonical_writer=canonical_writer,
        normalized_artifact_loader=normalized_artifact_loader,
        comparison_payload_adapter=comparison_payload_adapter,
    )
    normalized_item = _uuid(item_id, name="item_id")
    actor = _principal(principal)
    expected = _expected_version(expected_row_version)

    replay = _published_replay(
        workflow_session_factory,
        normalized_item,
        principal=actor,
        expected_row_version=expected,
    )
    if replay is not None:
        return replay

    request, attempt = _prepare_attempt(
        workflow_session_factory,
        normalized_item,
        principal=actor,
        expected_row_version=expected,
        normalized_artifact_loader=normalized_artifact_loader,
        comparison_payload_adapter=comparison_payload_adapter,
        now=now,
    )

    try:
        raw_result = canonical_writer(request)
    except Exception as exc:
        # The accepted internal writer returns contract-valid failures rather
        # than raising for governed write outcomes. A raised callback therefore
        # has unknown canonical durability and must never be blindly retried.
        _record_untrusted_writer_failure(
            workflow_session_factory,
            normalized_item,
            principal=actor,
            expected_row_version=expected,
            request=request,
            attempt=attempt,
            failure_code="writer_callback_exception",
            message="Canonical writer callback raised; durability is unknown.",
            now=now,
        )
        raise WorkflowPublicationDependencyUnavailable(
            "Canonical writer callback raised; publication durability is unknown."
        ) from exc
    if not isinstance(raw_result, Mapping):
        _record_untrusted_writer_failure(
            workflow_session_factory,
            normalized_item,
            principal=actor,
            expected_row_version=expected,
            request=request,
            attempt=attempt,
            failure_code="writer_result_invalid",
            message="Canonical writer returned a non-object result.",
            now=now,
        )
        raise WorkflowPublicationDependencyUnavailable(
            "Canonical writer callback returned a non-object result."
        )
    try:
        writer_result = validate_canonical_writer_result(raw_result)
        assert_canonical_writer_result_matches_request(request, writer_result)
    except Exception as exc:
        _record_untrusted_writer_failure(
            workflow_session_factory,
            normalized_item,
            principal=actor,
            expected_row_version=expected,
            request=request,
            attempt=attempt,
            failure_code="writer_result_invalid",
            message="Canonical writer returned invalid or mismatched authority.",
            now=now,
        )
        raise WorkflowPublicationDependencyUnavailable(
            "Canonical writer callback returned invalid or mismatched authority."
        ) from exc

    if writer_result["status"] not in CANONICAL_WRITER_SUCCESS_STATUSES:
        failure = _record_writer_failure(
            workflow_session_factory,
            normalized_item,
            principal=actor,
            expected_row_version=expected,
            request=request,
            attempt=attempt,
            writer_result=writer_result,
            now=now,
        )
        raise WorkflowPublicationWriterFailure(failure)

    return _finalize_success(
        workflow_session_factory,
        normalized_item,
        principal=actor,
        expected_row_version=expected,
        request=request,
        attempt=attempt,
        writer_result=writer_result,
        now=now,
    )
