"""Workflow-bound Ballot Lens execution authority contract.

The browser may identify only a governed Workflow item/pass and optimistic row
version. It never supplies the execution URL or anonymous public-registry source
ID. The caller builds authoritative server context before invoking this policy.

This module performs no parser, network, filesystem, database, Flask or
Socket.IO operation and cannot dispatch a run.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from uuid import UUID

from webapp.parser.contracts.workflow_authorization import (
    CAP_BALLOT_LENS_EXECUTE,
    WorkflowAuthorizationError,
    assert_capability,
)
from webapp.parser.services.trusted_parser_source_policy import (
    TrustedParserSourceAuthority,
    TrustedParserSourceDenied,
    assert_trusted_parser_source,
)

WORKFLOW_BALLOT_LENS_EXECUTION_CONTRACT = (
    "workflow_ballot_lens_execution_authority_v1"
)
WORKFLOW_EXECUTION_REQUEST_KEYS = frozenset({
    "workflow_item_id",
    "workflow_pass_id",
    "expected_row_version",
})


class WorkflowBallotLensExecutionDenied(PermissionError):
    pass


def _uuid_text(value: object) -> str:
    try:
        return str(UUID(str(value)))
    except (TypeError, ValueError, AttributeError) as exc:
        raise WorkflowBallotLensExecutionDenied(
            "Workflow Ballot Lens execution denied."
        ) from exc


def validate_workflow_execution_request(
    payload: Mapping[str, object] | None,
) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise WorkflowBallotLensExecutionDenied(
            "Workflow Ballot Lens execution denied."
        )
    if frozenset(payload.keys()) != WORKFLOW_EXECUTION_REQUEST_KEYS:
        raise WorkflowBallotLensExecutionDenied(
            "Workflow Ballot Lens execution denied."
        )

    row_version = payload.get("expected_row_version")
    if (
        isinstance(row_version, bool)
        or not isinstance(row_version, int)
        or row_version < 0
    ):
        raise WorkflowBallotLensExecutionDenied(
            "Workflow Ballot Lens execution denied."
        )

    return {
        "workflow_item_id": _uuid_text(payload.get("workflow_item_id")),
        "workflow_pass_id": _uuid_text(payload.get("workflow_pass_id")),
        "expected_row_version": row_version,
    }


@dataclass(frozen=True)
class WorkflowBallotLensServerContext:
    workflow_item_id: str
    workflow_pass_id: str
    row_version: int
    assigned_principal: str
    pass_is_current: bool
    pass_status: str
    source_url: str
    qc_evidence: Mapping[str, object]
    registry_state: Mapping[str, object]


@dataclass(frozen=True)
class AuthorizedWorkflowBallotLensExecution:
    workflow_item_id: str
    workflow_pass_id: str
    expected_row_version: int
    resolved_source_url: str
    trusted_source: TrustedParserSourceAuthority
    contract: str = WORKFLOW_BALLOT_LENS_EXECUTION_CONTRACT

    def safe_projection(self) -> dict[str, object]:
        return {
            "contract": self.contract,
            "workflow_item_id": self.workflow_item_id,
            "workflow_pass_id": self.workflow_pass_id,
            "expected_row_version": self.expected_row_version,
            "execution_mode": "workflow",
            "source_url_disclosed": False,
            "trusted_source": self.trusted_source.safe_projection(),
        }


def authorize_workflow_ballot_lens_execution(
    payload: Mapping[str, object] | None,
    *,
    principal: object,
    internal_roles: Iterable[str],
    server_context: WorkflowBallotLensServerContext,
) -> AuthorizedWorkflowBallotLensExecution:
    request = validate_workflow_execution_request(payload)
    actor = str(principal or "").strip()
    if not actor:
        raise WorkflowBallotLensExecutionDenied(
            "Workflow Ballot Lens execution denied."
        )

    try:
        assert_capability(internal_roles, CAP_BALLOT_LENS_EXECUTE)
    except WorkflowAuthorizationError as exc:
        raise WorkflowBallotLensExecutionDenied(
            "Workflow Ballot Lens execution denied."
        ) from exc

    item_id = _uuid_text(server_context.workflow_item_id)
    pass_id = _uuid_text(server_context.workflow_pass_id)
    assigned = str(server_context.assigned_principal or "").strip()
    status = str(server_context.pass_status or "").strip().lower()

    if (
        request["workflow_item_id"] != item_id
        or request["workflow_pass_id"] != pass_id
        or request["expected_row_version"] != server_context.row_version
        or assigned != actor
        or server_context.pass_is_current is not True
        or status != "in_progress"
    ):
        raise WorkflowBallotLensExecutionDenied(
            "Workflow Ballot Lens execution denied."
        )

    try:
        trusted_source = assert_trusted_parser_source(
            source_url=server_context.source_url,
            qc_evidence=server_context.qc_evidence,
            registry_state=server_context.registry_state,
        )
    except TrustedParserSourceDenied as exc:
        raise WorkflowBallotLensExecutionDenied(
            "Workflow Ballot Lens execution denied."
        ) from exc

    source_url = str(server_context.source_url or "").strip()
    if not source_url:
        raise WorkflowBallotLensExecutionDenied(
            "Workflow Ballot Lens execution denied."
        )

    return AuthorizedWorkflowBallotLensExecution(
        workflow_item_id=item_id,
        workflow_pass_id=pass_id,
        expected_row_version=server_context.row_version,
        resolved_source_url=source_url,
        trusted_source=trusted_source,
    )
