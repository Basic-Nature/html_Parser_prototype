"""Server-owned completion bridge for a successful governed DL1 Ballot Lens run.

The browser never supplies staging identifiers, evidence references, artifact
paths/hashes, Pre-QC claims, or submit assertions. They are derived from the
same trusted parser run and committed in one Workflow database transaction.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from webapp.parser.services.workflow_actions import submit_first_workflow_pass
from webapp.parser.services.workflow_normalized_artifact import (
    build_comparison_payload,
    build_workflow_semantic,
    normalized_artifact_bytes,
    parser_observation_manifest_bytes,
    sha256_bytes,
    workflow_scope_from_item,
    write_exact_artifact,
)
from webapp.parser.services.workflow_pre_qc_validation import (
    validate_first_workflow_pass_pre_qc,
)
from webapp.parser.services.workflow_staging_binding import (
    begin_workflow_staging_binding,
    finalize_workflow_staging_binding,
)
from webapp.parser.utils.models import (
    StagingElectionResult,
    WorkflowItem,
    WorkflowPass,
)

WORKFLOW_DL1_RUNTIME_BRIDGE_CONTRACT = "workflow_dl1_runtime_bridge_v1"
WORKFLOW_DL1_STAGING_ROW_CONTRACT = "workflow_dl1_staging_row_v1"


class WorkflowDL1RuntimeBridgeError(RuntimeError):
    pass


def _uuid(value: object, *, name: str) -> UUID:
    try:
        return value if isinstance(value, UUID) else UUID(str(value))
    except (TypeError, ValueError, AttributeError) as exc:
        raise WorkflowDL1RuntimeBridgeError(f"{name} must be a UUID") from exc


def _principal(value: object) -> str:
    principal = str(value or "").strip()
    if not principal:
        raise WorkflowDL1RuntimeBridgeError(
            "authenticated Workflow principal is required"
        )
    return principal


def _exact_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise WorkflowDL1RuntimeBridgeError(f"{name} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise WorkflowDL1RuntimeBridgeError(f"{name} must be an integer") from exc
    if parsed < 0:
        raise WorkflowDL1RuntimeBridgeError(f"{name} must be >= 0")
    return parsed


def _safe_output_folder(csv_path: object, output_root: Path) -> Path:
    root = output_root.resolve()
    path = Path(str(csv_path or "")).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise WorkflowDL1RuntimeBridgeError(
            "trusted parser CSV escaped OUTPUT_DIR"
        ) from exc
    if path.name != "results.csv" or not path.is_file():
        raise WorkflowDL1RuntimeBridgeError(
            "trusted Workflow run requires persisted results.csv"
        )
    return path.parent


def _output_ref(path: Path, output_root: Path) -> str:
    relative = path.resolve().relative_to(output_root.resolve())
    return "output://" + relative.as_posix()


def _json_safe_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    materialized = [dict(row) for row in rows]
    try:
        json.dumps(
            materialized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise WorkflowDL1RuntimeBridgeError(
            "finalized Smart Elections rows must be deterministic JSON values"
        ) from exc
    return materialized


def complete_governed_dl1_from_trusted_run(
    *,
    session_factory: Callable[[], Session],
    workflow_item_id: object,
    workflow_pass_id: object,
    principal: object,
    expected_row_version: object,
    capture: Mapping[str, object],
    registry_path: Path,
    output_root: Path,
) -> dict[str, object]:
    item_id = _uuid(workflow_item_id, name="workflow_item_id")
    pass_id = _uuid(workflow_pass_id, name="workflow_pass_id")
    actor = _principal(principal)
    expected_version = _exact_int(
        expected_row_version,
        name="expected_row_version",
    )

    headers_raw = capture.get("headers")
    rows_raw = capture.get("rows")
    observations_raw = capture.get("observations")
    if (
        not isinstance(headers_raw, (list, tuple))
        or not isinstance(rows_raw, (list, tuple))
        or not isinstance(observations_raw, (list, tuple))
    ):
        raise WorkflowDL1RuntimeBridgeError(
            "trusted runtime capture is incomplete"
        )
    headers = [str(header) for header in headers_raw]
    rows = _json_safe_rows(rows_raw)
    observations = [
        dict(observation)
        for observation in observations_raw
        if isinstance(observation, Mapping)
    ]
    if len(observations) != len(observations_raw):
        raise WorkflowDL1RuntimeBridgeError(
            "parser observation capture is invalid"
        )

    output_folder = _safe_output_folder(capture.get("csv_path"), output_root)
    semantic_path = output_folder / "workflow_normalized_semantic.json"
    evidence_path = output_folder / "workflow_parser_observations.json"

    session = session_factory()
    try:
        item = session.execute(
            select(WorkflowItem).where(WorkflowItem.id == item_id)
        ).scalar_one_or_none()
        workflow_pass = session.execute(
            select(WorkflowPass).where(WorkflowPass.id == pass_id)
        ).scalar_one_or_none()
        if item is None or workflow_pass is None:
            raise WorkflowDL1RuntimeBridgeError(
                "Workflow item/pass was not found"
            )
        if int(item.row_version) != expected_version:
            raise WorkflowDL1RuntimeBridgeError(
                "Workflow row_version changed before parser completion"
            )
        if (
            workflow_pass.workflow_item_id != item.id
            or workflow_pass.is_current is not True
            or workflow_pass.status != "in_progress"
            or str(workflow_pass.assigned_principal or "").strip() != actor
        ):
            raise WorkflowDL1RuntimeBridgeError(
                "trusted parser completion is not bound to current pass authority"
            )
        if (
            workflow_pass.pass_number,
            workflow_pass.pass_label,
        ) != (1, "DL1"):
            return {
                "success": False,
                "contract": WORKFLOW_DL1_RUNTIME_BRIDGE_CONTRACT,
                "skipped": True,
                "reason": "dl1_only_bridge",
                "task_id": str(item.id),
                "pass_id": str(workflow_pass.id),
                "committed": False,
            }

        semantic = build_workflow_semantic(
            headers,
            rows,
            scope=workflow_scope_from_item(item),
        )
        semantic_bytes = normalized_artifact_bytes(semantic)
        semantic_sha = sha256_bytes(semantic_bytes)
        observation_bytes = parser_observation_manifest_bytes(observations)
        observation_sha = sha256_bytes(observation_bytes)

        write_exact_artifact(semantic_path, semantic_bytes)
        write_exact_artifact(evidence_path, observation_bytes)
        artifact_ref = _output_ref(semantic_path, output_root)
        source_evidence_ref = (
            _output_ref(evidence_path, output_root)
            + f"#sha256={observation_sha}"
        )

        with session.begin_nested():
            started = begin_workflow_staging_binding(
                session,
                item.id,
                workflow_pass.id,
                principal=actor,
                registry_path=registry_path,
            )
            staging_batch_id = _uuid(
                started["staging_batch_id"],
                name="staging_batch_id",
            )

            for index, row in enumerate(rows):
                session.add(
                    StagingElectionResult(
                        batch_id=staging_batch_id,
                        state=item.state,
                        county=item.jurisdiction_name,
                        source_url=item.source_url,
                        raw_html=None,
                        metastats={
                            "contract": WORKFLOW_DL1_STAGING_ROW_CONTRACT,
                            "row_index": index,
                            "normalized_row": row,
                        },
                    )
                )
            session.flush()

            frozen = finalize_workflow_staging_binding(
                session,
                item.id,
                workflow_pass.id,
                staging_batch_id,
                principal=actor,
                source_evidence_ref=source_evidence_ref,
                artifact_ref=artifact_ref,
                artifact_sha256=semantic_sha,
            )
            binding = {
                "workflow_item_id": str(item.id),
                "workflow_pass_id": str(workflow_pass.id),
                "pass_number": 1,
                "revision_number": int(workflow_pass.revision_number),
                "source_evidence_ref": source_evidence_ref,
                "staging_batch_id": str(staging_batch_id),
                "normalized_artifact_ref": artifact_ref,
                "normalized_artifact_sha256": semantic_sha,
            }
            comparison_payload = build_comparison_payload(
                semantic=semantic,
                binding=binding,
            )
            pre_qc = validate_first_workflow_pass_pre_qc(
                session,
                item.id,
                workflow_pass.id,
                staging_batch_id,
                principal=actor,
                normalized_payload=comparison_payload,
            )
            submitted = submit_first_workflow_pass(
                session,
                item.id,
                pass_id=workflow_pass.id,
                principal=actor,
                expected_row_version=expected_version,
                staging_batch_id=staging_batch_id,
                source_evidence_ref=source_evidence_ref,
                artifact_ref=artifact_ref,
                artifact_sha256=semantic_sha,
            )

        session.commit()
        return {
            "success": True,
            "contract": WORKFLOW_DL1_RUNTIME_BRIDGE_CONTRACT,
            "task_id": str(item.id),
            "pass_id": str(workflow_pass.id),
            "status": submitted["status"],
            "row_version": submitted["row_version"],
            "current_stage": submitted["current_stage"],
            "stage_condition": submitted["stage_condition"],
            "staging_batch_id": str(staging_batch_id),
            "source_evidence_ref": source_evidence_ref,
            "normalized_artifact_ref": artifact_ref,
            "normalized_artifact_sha256": semantic_sha,
            "semantic_sha256": comparison_payload["semantic_sha256"],
            "staging_row_count": len(rows),
            "pre_qc_complete": (
                pre_qc.get("candidate_check_status") == "complete"
                and pre_qc.get("semantic_validation_status") == "complete"
            ),
            "committed": True,
        }
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
