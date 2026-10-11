"""J2B1 governed admission ledger. This module never starts a parser.

The J2A preflight is repeated under a DB transaction and locks. Idempotent
replays only return existing state; they never dispatch or issue credentials.
"""
from __future__ import annotations
import hashlib
import hmac
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from webapp.parser.models.election_project import ElectionProject, ElectionProjectRun, ElectionProjectSourceRef
from webapp.parser.services.election_projects import ProjectError
from webapp.parser.services.project_run_intent_contract import InvalidRunIntent, parse_browser_run_intent
from webapp.parser.services.project_run_preflight import project_run_preflight
from webapp.parser.services.workflow_ballot_lens_handoff import project_workflow_ballot_lens_handoff, WorkflowBallotLensHandoffDenied
from webapp.parser.utils.models import SourceRegistryBinding, SourceRegistryRevision, WorkflowItem, WorkflowPass

CONTRACT = "project_run_admission_v1"


def _safe(run: ElectionProjectRun) -> dict[str, object]:
    # J2B has no dispatcher; a future lifecycle value must not be misreported
    # as an admitted, never-dispatched request. Future tranche owns expansion.
    if run.status != "admitted":
        raise ProjectError("project_run_state_not_supported", 409)
    return {
        "contract": CONTRACT,
        "run_id": str(run.id),
        "project_id": str(run.project_id),
        "source_ref_id": str(run.source_ref_id),
        "workflow_item_id": str(run.workflow_item_id),
        "project_row_version": run.project_row_version,
        "workflow_row_version": run.workflow_row_version,
        "status": run.status,
        "execution_authorized": False,
        "run_dispatched": False,
        "evidence_available": False,
        "source_url_disclosed": False,
    }


def _project(session: Session, project_id: uuid.UUID, owner_key: str, *, lock: bool) -> ElectionProject:
    q = select(ElectionProject).where(ElectionProject.id == project_id, ElectionProject.owner_key == owner_key)
    if lock:
        q = q.with_for_update()
    project = session.execute(q).scalar_one_or_none()
    if project is None:
        raise ProjectError("project_not_found", 404)
    return project


def _fingerprint(project_id: uuid.UUID, owner_key: str, intent: object) -> str:
    secret = os.getenv("ELECTION_PROJECT_OWNER_HMAC_KEY", "")
    if len(secret.encode("utf-8")) < 32:
        raise ProjectError("project_identity_key_unconfigured", 503)
    payload = json.dumps([str(project_id),owner_key,*intent.safe_request_fingerprint_fields()],separators=(",",":"))
    return hmac.new(secret.encode("utf-8"), ("electionpulse-project-run-v1:"+payload).encode(), hashlib.sha256).hexdigest()


def admit(session: Session, *, project_id: uuid.UUID, owner_key: str,
          principal: str, body: object, registry_path: Path) -> dict[str, object]:
    try:
        intent = parse_browser_run_intent(body)
    except InvalidRunIntent as exc:
        raise ProjectError(str(exc),400) from exc
    project = _project(session,project_id,owner_key,lock=True)
    digest = _fingerprint(project_id,owner_key,intent)
    prior = session.execute(select(ElectionProjectRun).where(
        ElectionProjectRun.project_id == project.id,
        ElectionProjectRun.owner_key == owner_key,
        ElectionProjectRun.idempotency_key == intent.idempotency_key,
    )).scalar_one_or_none()
    if prior is not None:
        if prior.request_fingerprint != digest:
            raise ProjectError("run_idempotency_conflict",409)
        return _safe(prior)
    if project.lifecycle != "active":
        raise ProjectError("project_archived",409)
    if project.row_version != intent.expected_project_version:
        raise ProjectError("project_version_conflict",409)
    # Lock server-owned Registry/Workflow identities before the J2A recheck.
    ref = session.execute(select(ElectionProjectSourceRef).where(
        ElectionProjectSourceRef.id == intent.source_ref_id,
        ElectionProjectSourceRef.project_id == project.id).with_for_update()).scalar_one_or_none()
    if ref is None:
        raise ProjectError("project_run_not_eligible",403)
    binding = session.execute(select(SourceRegistryBinding).where(
        SourceRegistryBinding.id == ref.registry_binding_id).with_for_update()).scalar_one_or_none()
    revision = session.execute(select(SourceRegistryRevision).where(
        SourceRegistryRevision.id == ref.registry_revision_id).with_for_update()).scalar_one_or_none()
    item = session.execute(select(WorkflowItem).where(
        WorkflowItem.id == intent.workflow_item_id).with_for_update()).scalar_one_or_none()
    if binding is None or revision is None or item is None:
        raise ProjectError("project_run_not_eligible",403)
    # Always recheck J2A's single authority routine; never trust a previous GET.
    preview = project_run_preflight(session,
        project_id=project.id, source_ref_id=ref.id,
        workflow_item_id=item.id, expected_project_version=intent.expected_project_version,
        owner_key=owner_key,principal=principal,registry_path=registry_path)
    if (preview.get("eligible_for_confirmation_review") is not True
        or preview.get("run_dispatched") is not False
        or preview.get("execution_authorized") is not False
        or preview.get("registry_revision_id") != str(revision.id)
        or preview.get("workflow_row_version") != int(item.row_version)):
        raise ProjectError("project_run_not_eligible",403)
    # Resolve Workflow pass ONLY on the server. Lock it, then revalidate current
    # assignment/version. This ID is private ledger data, never a browser input.
    from webapp.parser.auth.workflow_runtime_authorization import assert_workflow_runtime_capability, WorkflowRuntimeAuthorizationDenied
    from webapp.parser.contracts.workflow_authorization import CAP_BALLOT_LENS_EXECUTE
    try:
        roles = assert_workflow_runtime_capability(principal,CAP_BALLOT_LENS_EXECUTE)
        handoff = project_workflow_ballot_lens_handoff(session,item.id,principal=principal,
                      internal_roles=roles,registry_path=registry_path)
    except (WorkflowBallotLensHandoffDenied,WorkflowRuntimeAuthorizationDenied) as exc:
        raise ProjectError("project_run_not_eligible",403) from exc
    if (handoff.get("workflow_item_id") != str(item.id)
        or handoff.get("expected_row_version") != item.row_version
        or handoff.get("browser_payload_keys") != ["workflow_item_id","workflow_pass_id","expected_row_version"]):
        raise ProjectError("project_run_not_eligible",403)
    try:
        pass_id = uuid.UUID(str(handoff["workflow_pass_id"]))
    except (KeyError,ValueError,TypeError) as exc:
        raise ProjectError("project_run_not_eligible",403) from exc
    current_pass = session.execute(select(WorkflowPass).where(
        WorkflowPass.id == pass_id,WorkflowPass.workflow_item_id == item.id,
        WorkflowPass.is_current.is_(True),WorkflowPass.status == "in_progress",
        WorkflowPass.assigned_principal == principal).with_for_update()).scalar_one_or_none()
    if current_pass is None or binding.current_revision_id != revision.id:
        raise ProjectError("project_run_not_eligible",403)
    now = datetime.now(timezone.utc)
    run = ElectionProjectRun(id=uuid.uuid4(), project_id=project.id,
        owner_key=owner_key,idempotency_key=intent.idempotency_key,
        request_fingerprint=digest,source_ref_id=ref.id,
        registry_binding_id=binding.id,registry_revision_id=revision.id,
        workflow_item_id=item.id,workflow_pass_id=current_pass.id,
        project_row_version=project.row_version,workflow_row_version=item.row_version,
        status="admitted",row_version=1,created_at=now,updated_at=now)
    # PostgreSQL READ COMMITTED + the locked Project row serializes requests
    # for the same project. The unique key remains the final idempotency guard.
    # A SAVEPOINT permits a losing insert to recover without poisoning the
    # caller-owned transaction; this is not a claim of SQLite race correctness.
    try:
        with session.begin_nested():
            session.add(run)
            session.flush()
    except IntegrityError:
        winner = session.execute(select(ElectionProjectRun).where(
            ElectionProjectRun.project_id == project.id,
            ElectionProjectRun.owner_key == owner_key,
            ElectionProjectRun.idempotency_key == intent.idempotency_key,
        )).scalar_one_or_none()
        if winner is None:
            raise  # Unrelated constraint failure; caller must roll back.
        if winner.request_fingerprint != digest:
            raise ProjectError("run_idempotency_conflict", 409)
        return _safe(winner)
    return _safe(run)  # Caller owns transaction commit; never dispatch here.


def list_runs(session: Session, *, project_id: uuid.UUID, owner_key: str) -> dict[str, object]:
    _project(session,project_id,owner_key,lock=False)
    records = session.execute(select(ElectionProjectRun).where(
        ElectionProjectRun.project_id == project_id,
        ElectionProjectRun.owner_key == owner_key).order_by(
        ElectionProjectRun.created_at.desc(),ElectionProjectRun.id).limit(50)).scalars().all()
    return {"contract":CONTRACT,"runs":[_safe(r) for r in records]}


def get_run(session: Session, *, project_id: uuid.UUID,run_id: uuid.UUID,owner_key: str) -> dict[str, object]:
    _project(session,project_id,owner_key,lock=False)
    run = session.execute(select(ElectionProjectRun).where(
        ElectionProjectRun.id == run_id,
        ElectionProjectRun.project_id == project_id,
        ElectionProjectRun.owner_key == owner_key)).scalar_one_or_none()
    if run is None:
        raise ProjectError("project_run_not_found",404)
    return _safe(run)
