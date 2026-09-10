"""Read-only authenticated Workbench handoff into Workflow Ballot Lens.

The caller supplies only a Workflow item id from the public-safe worklist row.
The server resolves the requesting principal's exact current in-progress pass,
reconstructs the W13C server context, re-runs W13B execution authorization, and
returns only the three browser fields accepted by the Workflow socket intent.

No source URL, principal identity, role name, or capability claim is disclosed.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from webapp.parser.services.workflow_ballot_lens_execution import (
    WorkflowBallotLensExecutionDenied,
    authorize_workflow_ballot_lens_execution,
)
from webapp.parser.services.workflow_ballot_lens_runtime_context import (
    WorkflowBallotLensRuntimeContextDenied,
    build_workflow_ballot_lens_server_context,
)
from webapp.parser.utils.models import WorkflowItem, WorkflowPass


WORKFLOW_BALLOT_LENS_HANDOFF_CONTRACT = "workflow_ballot_lens_handoff_v1"
WORKFLOW_BALLOT_LENS_BROWSER_KEYS = (
    "workflow_item_id",
    "workflow_pass_id",
    "expected_row_version",
)


class WorkflowBallotLensHandoffDenied(PermissionError):
    """Generic fail-closed Workbench handoff denial."""


def _deny(exc: Exception | None = None):
    error = WorkflowBallotLensHandoffDenied(
        "Workflow Ballot Lens handoff denied."
    )
    if exc is None:
        raise error
    raise error from exc


def _uuid(value: object) -> UUID:
    try:
        return value if isinstance(value, UUID) else UUID(str(value))
    except (TypeError, ValueError, AttributeError) as exc:
        _deny(exc)


def project_workflow_ballot_lens_handoff(
    session: Session,
    item_id: UUID | str,
    *,
    principal: object,
    internal_roles: Iterable[str],
    registry_path: Path,
) -> dict[str, object]:
    """Return only the exact browser payload needed for W13C execution."""

    actor = str(principal or "").strip()
    if not actor:
        _deny()

    normalized_item_id = _uuid(item_id)
    item = session.get(WorkflowItem, normalized_item_id)
    if item is None:
        _deny()

    current_assigned = session.execute(
        select(WorkflowPass).where(
            WorkflowPass.workflow_item_id == item.id,
            WorkflowPass.is_current.is_(True),
            WorkflowPass.status == "in_progress",
            WorkflowPass.assigned_principal == actor,
        )
    ).scalars().all()
    if len(current_assigned) != 1:
        _deny()

    workflow_pass = current_assigned[0]
    request_payload = {
        "workflow_item_id": str(item.id),
        "workflow_pass_id": str(workflow_pass.id),
        "expected_row_version": int(item.row_version),
    }

    try:
        server_context = build_workflow_ballot_lens_server_context(
            session,
            request_payload,
            registry_path=registry_path,
        )
        authority = authorize_workflow_ballot_lens_execution(
            request_payload,
            principal=actor,
            internal_roles=internal_roles,
            server_context=server_context,
        )
    except (
        WorkflowBallotLensExecutionDenied,
        WorkflowBallotLensRuntimeContextDenied,
    ) as exc:
        _deny(exc)

    safe = authority.safe_projection()
    if (
        safe.get("execution_mode") != "workflow"
        or safe.get("source_url_disclosed") is not False
        or safe.get("workflow_item_id") != authority.workflow_item_id
        or safe.get("workflow_pass_id") != authority.workflow_pass_id
        or safe.get("expected_row_version")
        != authority.expected_row_version
        or authority.resolved_source_url in repr(safe)
    ):
        _deny()

    payload = {
        "success": True,
        "contract": WORKFLOW_BALLOT_LENS_HANDOFF_CONTRACT,
        "workflow_item_id": authority.workflow_item_id,
        "workflow_pass_id": authority.workflow_pass_id,
        "expected_row_version": authority.expected_row_version,
        "can_execute_ballot_lens": True,
        "principal_disclosed": False,
        "source_url_disclosed": False,
        "browser_payload_keys": list(WORKFLOW_BALLOT_LENS_BROWSER_KEYS),
    }
    if set(payload["browser_payload_keys"]) != set(
        WORKFLOW_BALLOT_LENS_BROWSER_KEYS
    ):
        _deny()
    return payload
