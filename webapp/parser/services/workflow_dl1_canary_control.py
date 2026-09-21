from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import re
from typing import Mapping
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from webapp.parser.services.source_registry_runtime import lookup_exact_registry_entry
from webapp.parser.utils.models import WorkflowEvent, WorkflowItem, WorkflowPass

WORKFLOW_DL1_CANARY_CONTROL_CONTRACT = "workflow_dl1_canary_control_v1"
WORKFLOW_DL1_CANARY_RELEASE_CONTRACT = "workflow_dl1_canary_release_v1"

ENV_CLAIM_ENABLED = "WORKFLOW_DL1_CANARY_CLAIM_ENABLED"
ENV_COMPLETION_ENABLED = "WORKFLOW_DL1_CANARY_COMPLETION_ENABLED"
ENV_RELEASE_ENABLED = "WORKFLOW_DL1_CANARY_RELEASE_ENABLED"
ENV_ITEM_ID = "WORKFLOW_DL1_CANARY_ITEM_ID"
ENV_EXPECTED_ROW_VERSION = "WORKFLOW_DL1_CANARY_EXPECTED_ROW_VERSION"
ENV_PRINCIPAL_SHA256 = "WORKFLOW_DL1_CANARY_PRINCIPAL_SHA256"
ENV_SOURCE_SHA256 = "WORKFLOW_DL1_CANARY_SOURCE_SHA256"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class WorkflowDl1CanaryControlError(RuntimeError):
    status_code = 409
    code = "workflow_dl1_canary_control_error"


class WorkflowDl1CanaryConfigError(WorkflowDl1CanaryControlError):
    status_code = 503
    code = "workflow_dl1_canary_config_invalid"


class WorkflowDl1CanaryDisabled(WorkflowDl1CanaryControlError):
    status_code = 503
    code = "workflow_dl1_canary_disabled"


class WorkflowDl1CanaryDenied(WorkflowDl1CanaryControlError):
    status_code = 403
    code = "workflow_dl1_canary_denied"


class WorkflowDl1CanaryConflict(WorkflowDl1CanaryControlError):
    status_code = 409
    code = "workflow_dl1_canary_conflict"


@dataclass(frozen=True)
class WorkflowDl1CanaryConfig:
    claim_enabled: bool
    completion_enabled: bool
    release_enabled: bool
    item_id: UUID | None
    expected_row_version: int | None
    principal_sha256: str | None
    source_sha256: str | None

    @property
    def any_enabled(self) -> bool:
        return bool(self.claim_enabled or self.completion_enabled or self.release_enabled)

    @property
    def binding_complete(self) -> bool:
        return bool(
            self.item_id is not None
            and self.expected_row_version is not None
            and self.principal_sha256
            and self.source_sha256
        )


def _truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _sha256_text(value: object) -> str:
    return hashlib.sha256(str(value or "").strip().encode("utf-8")).hexdigest()


def _parse_uuid(raw: object, *, name: str) -> UUID:
    try:
        return UUID(str(raw or "").strip())
    except (TypeError, ValueError) as exc:
        raise WorkflowDl1CanaryConfigError(f"{name} must be a UUID.") from exc


def _parse_row_version(raw: object) -> int:
    if isinstance(raw, bool):
        raise WorkflowDl1CanaryConfigError(f"{ENV_EXPECTED_ROW_VERSION} must be an integer.")
    try:
        value = int(str(raw or "").strip())
    except (TypeError, ValueError) as exc:
        raise WorkflowDl1CanaryConfigError(f"{ENV_EXPECTED_ROW_VERSION} must be an integer.") from exc
    if value < 1:
        raise WorkflowDl1CanaryConfigError(f"{ENV_EXPECTED_ROW_VERSION} must be >= 1.")
    return value


def _parse_hash(raw: object, *, name: str) -> str:
    value = str(raw or "").strip().lower()
    if not _HEX64.fullmatch(value):
        raise WorkflowDl1CanaryConfigError(f"{name} must be lowercase SHA-256 hex.")
    return value


def load_dl1_canary_config(environ: Mapping[str, str] | None = None) -> WorkflowDl1CanaryConfig:
    source = os.environ if environ is None else environ
    claim_enabled = _truthy(source.get(ENV_CLAIM_ENABLED))
    completion_enabled = _truthy(source.get(ENV_COMPLETION_ENABLED))
    release_enabled = _truthy(source.get(ENV_RELEASE_ENABLED))
    if not (claim_enabled or completion_enabled or release_enabled):
        return WorkflowDl1CanaryConfig(False, False, False, None, None, None, None)
    config = WorkflowDl1CanaryConfig(
        claim_enabled=claim_enabled,
        completion_enabled=completion_enabled,
        release_enabled=release_enabled,
        item_id=_parse_uuid(source.get(ENV_ITEM_ID), name=ENV_ITEM_ID),
        expected_row_version=_parse_row_version(source.get(ENV_EXPECTED_ROW_VERSION)),
        principal_sha256=_parse_hash(source.get(ENV_PRINCIPAL_SHA256), name=ENV_PRINCIPAL_SHA256),
        source_sha256=_parse_hash(source.get(ENV_SOURCE_SHA256), name=ENV_SOURCE_SHA256),
    )
    if not config.binding_complete:
        raise WorkflowDl1CanaryConfigError("Enabled DL1 canary requires complete server-owned binding.")
    return config


def _assert_principal(config: WorkflowDl1CanaryConfig, principal: object) -> str:
    actor = str(principal or "").strip()
    if not actor:
        raise WorkflowDl1CanaryDenied("Authenticated canary principal is required.")
    if _sha256_text(actor) != config.principal_sha256:
        raise WorkflowDl1CanaryDenied("Workflow principal is not the bound DL1 canary principal.")
    return actor


def _assert_item_id(config: WorkflowDl1CanaryConfig, item_id: UUID | str) -> UUID:
    try:
        normalized = item_id if isinstance(item_id, UUID) else UUID(str(item_id))
    except (TypeError, ValueError) as exc:
        raise WorkflowDl1CanaryConflict("Workflow canary item id is invalid.") from exc
    if normalized != config.item_id:
        raise WorkflowDl1CanaryDenied("Workflow item is not the bound DL1 canary item.")
    return normalized


def _assert_source(item: WorkflowItem, *, config: WorkflowDl1CanaryConfig, registry_path: Path) -> None:
    entry = lookup_exact_registry_entry(str(item.source_url or ""), path=registry_path)
    if entry is None or entry.registry_category != "curated":
        raise WorkflowDl1CanaryConflict("Bound canary source is not an exact curated registry entry.")
    if _sha256_text(entry.url) != config.source_sha256:
        raise WorkflowDl1CanaryConflict("Bound canary source SHA-256 does not match registry authority.")


def assert_dl1_canary_claim_binding(
    session: Session,
    item_id: UUID | str,
    *,
    principal: str,
    expected_row_version: int,
    registry_path: Path,
    config: WorkflowDl1CanaryConfig | None = None,
    environ: Mapping[str, str] | None = None,
) -> WorkflowItem:
    cfg = config or load_dl1_canary_config(environ)
    if not cfg.claim_enabled:
        raise WorkflowDl1CanaryDisabled("DL1 canary claim mutation is disabled.")
    _assert_principal(cfg, principal)
    normalized = _assert_item_id(cfg, item_id)
    try:
        expected = int(expected_row_version)
    except (TypeError, ValueError) as exc:
        raise WorkflowDl1CanaryConflict("expected_row_version must be an integer.") from exc
    if expected != cfg.expected_row_version:
        raise WorkflowDl1CanaryConflict("Request row_version is not the bound pre-claim canary version.")
    item = session.execute(
        select(WorkflowItem).where(WorkflowItem.id == normalized).with_for_update()
    ).scalar_one_or_none()
    if item is None:
        raise WorkflowDl1CanaryConflict("Bound canary item was not found.")
    if int(item.row_version) != expected:
        raise WorkflowDl1CanaryConflict("Bound canary row_version changed before claim.")
    if (item.lifecycle_state, item.current_stage, item.stage_condition) != ("queued", "source_intake", "pending"):
        raise WorkflowDl1CanaryConflict("Bound canary item is no longer DL1-claimable.")
    _assert_source(item, config=cfg, registry_path=registry_path)
    return item


def assert_dl1_canary_completion_context(
    *,
    workflow_item_id: UUID | str,
    principal: str,
    expected_row_version: int,
    config: WorkflowDl1CanaryConfig | None = None,
    environ: Mapping[str, str] | None = None,
) -> None:
    cfg = config or load_dl1_canary_config(environ)
    if not cfg.completion_enabled:
        raise WorkflowDl1CanaryDisabled("DL1 canary completion mutation is disabled.")
    _assert_principal(cfg, principal)
    _assert_item_id(cfg, workflow_item_id)
    try:
        observed = int(expected_row_version)
    except (TypeError, ValueError) as exc:
        raise WorkflowDl1CanaryConflict("Workflow completion row_version must be an integer.") from exc
    if observed != int(cfg.expected_row_version) + 1:
        raise WorkflowDl1CanaryConflict("Workflow completion row_version is not the bound post-claim version.")


def release_dl1_canary_claim(
    session: Session,
    item_id: UUID | str,
    pass_id: UUID | str,
    *,
    principal: str,
    expected_row_version: int,
    registry_path: Path,
    config: WorkflowDl1CanaryConfig | None = None,
    environ: Mapping[str, str] | None = None,
    now: datetime | None = None,
) -> dict[str, object]:
    cfg = config or load_dl1_canary_config(environ)
    if not cfg.release_enabled:
        raise WorkflowDl1CanaryDisabled("DL1 canary claim release is disabled.")
    actor = _assert_principal(cfg, principal)
    normalized_item = _assert_item_id(cfg, item_id)
    try:
        normalized_pass = pass_id if isinstance(pass_id, UUID) else UUID(str(pass_id))
    except (TypeError, ValueError) as exc:
        raise WorkflowDl1CanaryConflict("Canary pass id is invalid.") from exc
    try:
        expected = int(expected_row_version)
    except (TypeError, ValueError) as exc:
        raise WorkflowDl1CanaryConflict("expected_row_version must be an integer.") from exc
    if expected != int(cfg.expected_row_version) + 1:
        raise WorkflowDl1CanaryConflict("Release requires exact bound post-claim row_version.")
    timestamp = now or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    timestamp = timestamp.astimezone(timezone.utc)
    item = session.execute(
        select(WorkflowItem).where(WorkflowItem.id == normalized_item).with_for_update()
    ).scalar_one_or_none()
    if item is None or int(item.row_version) != expected:
        raise WorkflowDl1CanaryConflict("Bound canary item/version changed before release.")
    if (item.lifecycle_state, item.current_stage, item.stage_condition) != ("active", "independent_acquisition", "in_progress"):
        raise WorkflowDl1CanaryConflict("Bound canary item is not in releasable DL1 state.")
    _assert_source(item, config=cfg, registry_path=registry_path)
    workflow_pass = session.execute(
        select(WorkflowPass).where(WorkflowPass.id == normalized_pass).with_for_update()
    ).scalar_one_or_none()
    if workflow_pass is None:
        raise WorkflowDl1CanaryConflict("Bound DL1 pass was not found.")
    if (
        workflow_pass.workflow_item_id != item.id
        or int(workflow_pass.pass_number) != 1
        or str(workflow_pass.pass_label) != "DL1"
        or workflow_pass.is_current is not True
        or workflow_pass.status != "in_progress"
        or str(workflow_pass.assigned_principal or "").strip() != actor
        or workflow_pass.submitted_at is not None
    ):
        raise WorkflowDl1CanaryConflict("DL1 canary release requires current assigned pre-submit pass.")
    if (
        workflow_pass.staging_batch_id is not None
        or workflow_pass.source_evidence_ref is not None
        or workflow_pass.candidate_check_status is not None
        or workflow_pass.candidate_check_result is not None
        or workflow_pass.semantic_validation_status is not None
        or workflow_pass.semantic_validation_result is not None
    ):
        raise WorkflowDl1CanaryConflict("DL1 canary release refuses a pass with staging or validation evidence.")
    prior_state = {
        "lifecycle_state": item.lifecycle_state,
        "current_stage": item.current_stage,
        "stage_condition": item.stage_condition,
        "row_version": int(item.row_version),
        "current_dl1_pass_id": str(workflow_pass.id),
        "current_dl1_revision": int(workflow_pass.revision_number),
    }
    workflow_pass.is_current = False
    workflow_pass.status = "released"
    workflow_pass.superseded_at = timestamp
    workflow_pass.updated_at = timestamp
    item.lifecycle_state = "queued"
    item.current_stage = "source_intake"
    item.stage_condition = "pending"
    item.row_version = expected + 1
    item.updated_at = timestamp
    new_state = {
        "lifecycle_state": item.lifecycle_state,
        "current_stage": item.current_stage,
        "stage_condition": item.stage_condition,
        "row_version": int(item.row_version),
        "released_dl1_pass_id": str(workflow_pass.id),
        "released_dl1_revision": int(workflow_pass.revision_number),
    }
    event = WorkflowEvent(
        workflow_item_id=item.id,
        actor_type="principal",
        actor_principal=actor,
        actor_service=None,
        event_type="pass_claim_released",
        stage="source_intake",
        prior_state=prior_state,
        new_state=new_state,
        related_pass_id=workflow_pass.id,
        related_comparison_id=None,
        related_review_id=None,
        related_staging_batch_id=None,
        related_canonical_race_id=item.canonical_race_id,
        reason_code="w23_dl1_canary_recovery",
        summary="DL1 canary claim released before staging; prior revision preserved.",
        event_metadata={
            "contract": WORKFLOW_DL1_CANARY_RELEASE_CONTRACT,
            "pass_number": 1,
            "pass_label": "DL1",
            "released_revision_number": int(workflow_pass.revision_number),
            "audit_history_preserved": True,
            "retry_requires_new_bound_row_version": True,
        },
        occurred_at=timestamp,
    )
    session.add(event)
    session.flush()
    return {
        "success": True,
        "contract": WORKFLOW_DL1_CANARY_RELEASE_CONTRACT,
        "task_id": str(item.id),
        "released_pass_id": str(workflow_pass.id),
        "released_revision_number": int(workflow_pass.revision_number),
        "lifecycle_state": item.lifecycle_state,
        "current_stage": item.current_stage,
        "stage_condition": item.stage_condition,
        "row_version": int(item.row_version),
        "audit_history_preserved": True,
        "committed": False,
    }
