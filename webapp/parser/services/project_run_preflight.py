"""J2A read-only Project -> governed Workflow eligibility preview.

No Project selection grants execution, no URL is accepted from a browser, and
no parser/session/admission/write operation is present in this module.
"""
from __future__ import annotations
from pathlib import Path
from uuid import UUID
from sqlalchemy import select
from sqlalchemy.orm import Session
from webapp.parser.auth.workflow_runtime_authorization import (
    WorkflowRuntimeAuthorizationDenied, assert_workflow_runtime_capability,
)
from webapp.parser.contracts.workflow_authorization import CAP_BALLOT_LENS_EXECUTE
from webapp.parser.models.election_project import ElectionProject, ElectionProjectSourceRef
from webapp.parser.services.election_projects import ProjectError
from webapp.parser.services.workflow_ballot_lens_handoff import (
    WorkflowBallotLensHandoffDenied, project_workflow_ballot_lens_handoff,
)
from webapp.parser.utils.models import (
    SourceRegistryBinding, SourceRegistryRevision, SourceRegistrySource, WorkflowItem,
)

CONTRACT = "project_run_preflight_v1"


def _deny() -> None:
    raise ProjectError("project_run_preflight_not_eligible", 403)


def _workflow_snapshot(
    session: Session, *, workflow_item_id: UUID, principal: str,
    roles: frozenset[str], registry_path: Path,
) -> tuple[str, int]:
    """Reuse existing handoff, including assignment, QC and Registry authority."""
    handoff = project_workflow_ballot_lens_handoff(
        session, workflow_item_id, principal=principal,
        internal_roles=roles, registry_path=registry_path,
    )
    item = session.get(WorkflowItem, workflow_item_id)
    if (item is None or handoff.get("contract") != "workflow_ballot_lens_handoff_v1"
        or handoff.get("success") is not True
        or handoff.get("can_execute_ballot_lens") is not True
        or handoff.get("source_url_disclosed") is not False
        or handoff.get("principal_disclosed") is not False
        or handoff.get("workflow_item_id") != str(workflow_item_id)
        or handoff.get("browser_payload_keys") != [
            "workflow_item_id", "workflow_pass_id", "expected_row_version"
        ] or not isinstance(handoff.get("expected_row_version"), int)
        or handoff["expected_row_version"] != item.row_version
        or not isinstance(item.source_url, str) or not item.source_url.strip()):
        _deny()
    return item.source_url, int(item.row_version)


def project_run_preflight(
    session: Session, *, project_id: UUID, source_ref_id: UUID,
    workflow_item_id: UUID, expected_project_version: int,
    owner_key: str, principal: str, registry_path: Path,
) -> dict[str, object]:
    """Return non-executable preview only; repeat ALL checks at J2B dispatch."""
    if (type(expected_project_version) is not int or expected_project_version < 1):
        raise ProjectError("invalid_project_version")
    project = session.execute(select(ElectionProject).where(
        ElectionProject.id == project_id,
        ElectionProject.owner_key == owner_key,
    )).scalar_one_or_none()
    if project is None:
        raise ProjectError("project_not_found", 404)
    if project.lifecycle != "active":
        raise ProjectError("project_archived", 409)
    if project.row_version != expected_project_version:
        raise ProjectError("project_version_conflict", 409)
    ref = session.execute(select(ElectionProjectSourceRef).where(
        ElectionProjectSourceRef.id == source_ref_id,
        ElectionProjectSourceRef.project_id == project_id,
    )).scalar_one_or_none()
    if ref is None:
        _deny()
    binding = session.get(SourceRegistryBinding, ref.registry_binding_id)
    revision = session.get(SourceRegistryRevision, ref.registry_revision_id)
    if (binding is None or revision is None
        or binding.current_revision_id != ref.registry_revision_id
        or revision.source_id != binding.source_id
        or binding.review_state != "approved"
        or binding.public_eligible is not True
        or binding.workflow_eligible is not True
        or binding.parser_eligible is not True):
        _deny()
    source = session.get(SourceRegistrySource, binding.source_id)
    if source is None or source.lifecycle_state != "active":
        _deny()
    try:
        roles = assert_workflow_runtime_capability(principal, CAP_BALLOT_LENS_EXECUTE)
        workflow_source_url, workflow_version = _workflow_snapshot(
            session, workflow_item_id=workflow_item_id,
            principal=principal, roles=roles, registry_path=registry_path,
        )
    except (WorkflowRuntimeAuthorizationDenied, WorkflowBallotLensHandoffDenied):
        _deny()
    if workflow_source_url != revision.exact_url:
        _deny()
    # No source URL, pass ID, confirmation token or runnable socket payload.
    return {
        "contract": CONTRACT,
        "project_id": str(project.id),
        "project_row_version": int(project.row_version),
        "source_ref_id": str(ref.id),
        "registry_binding_id": str(binding.id),
        "registry_revision_id": str(revision.id),
        "workflow_item_id": str(workflow_item_id),
        "workflow_row_version": workflow_version,
        "source_label": {
            "year": binding.year, "state": binding.state,
            "contest": binding.contest, "scope": binding.scope,
            "format": binding.format,
        },
        "eligible_for_confirmation_review": True,
        "confirmation_enabled": False,
        "execution_authorized": False,
        "run_dispatched": False,
        "source_url_disclosed": False,
    }
