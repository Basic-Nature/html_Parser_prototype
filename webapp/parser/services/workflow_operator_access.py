"""Safe server-derived Workflow operator capability projection.

This module performs no authentication and accepts no browser capability claims.
Callers must first resolve the authenticated principal and server-owned internal
Workflow roles. The projection deliberately omits principal identity.
"""

from __future__ import annotations

from collections.abc import Iterable

from webapp.parser.contracts.workflow_authorization import (
    CAP_BALLOT_LENS_EXECUTE,
    capabilities_for_roles,
)

WORKFLOW_OPERATOR_ACCESS_CONTRACT = "workflow_operator_access_v1"


class WorkflowOperatorAccessError(ValueError):
    pass


def project_workflow_operator_access(
    principal: object,
    internal_roles: Iterable[str],
) -> dict[str, object]:
    actor = str(principal or "").strip()
    if not actor:
        raise WorkflowOperatorAccessError(
            "Authenticated Workflow principal is required."
        )

    capabilities = capabilities_for_roles(internal_roles)
    return {
        "contract": WORKFLOW_OPERATOR_ACCESS_CONTRACT,
        "authenticated": True,
        "capabilities": sorted(capabilities),
        "can_execute_ballot_lens": (
            CAP_BALLOT_LENS_EXECUTE in capabilities
        ),
        "principal_disclosed": False,
    }
