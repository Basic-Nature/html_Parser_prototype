"""J1 projects: private bounded context; never authorizes parsing or publication."""
from __future__ import annotations
import hashlib
import hmac
import os
import re
import uuid
from datetime import date, datetime, timezone
from sqlalchemy import select, text, bindparam, Uuid
from sqlalchemy.orm import Session
from webapp.parser.models.election_project import ElectionProject, ElectionProjectScope, ElectionProjectSourceRef

class ProjectError(ValueError):
    def __init__(self, code: str, status: int = 400):
        super().__init__(code)
        self.code, self.status = code, status

def owner_key(principal: str) -> str:
    secret = os.getenv("ELECTION_PROJECT_OWNER_HMAC_KEY", "")
    if len(secret.encode("utf-8")) < 32:
        raise ProjectError("project_identity_key_unconfigured", 503)
    if not isinstance(principal, str) or not principal or len(principal) > 512:
        raise ProjectError("authenticated_principal_required", 401)
    return hmac.new(secret.encode("utf-8"), ("electionpulse-project-owner-v1:" + principal).encode(), hashlib.sha256).hexdigest()

def parse_uuid(value: object) -> uuid.UUID:
    try:
        return uuid.UUID(str(value))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ProjectError("invalid_uuid") from exc

def validated_text(raw: object, *, limit: int, required: bool = False) -> str:
    if not isinstance(raw, str):
        raise ProjectError("invalid_text")
    value = raw.strip()
    if len(value) > limit or (required and not value) or any(ord(x) < 32 for x in value):
        raise ProjectError("invalid_text")
    return value

def require_obj(value: object, allowed: set[str]) -> dict:
    if not isinstance(value, dict) or set(value) - allowed:
        raise ProjectError("invalid_fields")
    return value

def _now(): return datetime.now(timezone.utc)

def view(project: ElectionProject, scopes: list = (), refs: list = ()) -> dict:
    return {
        "id": str(project.id), "title": project.title, "description": project.description,
        "lifecycle": project.lifecycle, "row_version": project.row_version,
        "updated_at": project.updated_at.isoformat() if project.updated_at else None,
        "scopes": [
            {"id": str(s.id), "election_year": s.election_year,
             "election_date": s.election_date.isoformat() if s.election_date else None,
             "state_code": s.state_code, "jurisdiction": s.jurisdiction,
             "contest": s.contest, "granularity": s.granularity} for s in scopes
        ],
        "source_refs": [{"id": str(r.id), "registry_binding_id": str(r.registry_binding_id),
                         "registry_revision_id": str(r.registry_revision_id),
                         "execution_authorized": False, "canonical_publication": False} for r in refs],
        "authority": "project_context_only",
    }

def _owned(session: Session, project_id: uuid.UUID, owner: str, *, lock: bool = False) -> ElectionProject:
    stmt = select(ElectionProject).where(ElectionProject.id == project_id, ElectionProject.owner_key == owner)
    if lock: stmt = stmt.with_for_update()
    project = session.execute(stmt).scalar_one_or_none()
    if project is None: raise ProjectError("project_not_found", 404)
    return project

def detail(session: Session, project_id: uuid.UUID, owner: str) -> dict:
    project = _owned(session, project_id, owner)
    scopes = session.execute(select(ElectionProjectScope).where(ElectionProjectScope.project_id == project.id).order_by(ElectionProjectScope.id)).scalars().all()
    refs = session.execute(select(ElectionProjectSourceRef).where(ElectionProjectSourceRef.project_id == project.id).order_by(ElectionProjectSourceRef.id)).scalars().all()
    return view(project, scopes, refs)

def projects(session: Session, owner: str) -> list[dict]:
    rows = session.execute(select(ElectionProject).where(ElectionProject.owner_key == owner).order_by(ElectionProject.updated_at.desc()).limit(100)).scalars().all()
    return [view(row) for row in rows]

def create(session: Session, owner: str, body: object) -> dict:
    obj = require_obj(body, {"title", "description", "idempotency_key"})
    title = validated_text(obj.get("title"), limit=160, required=True)
    description = validated_text(obj.get("description", ""), limit=2000)
    key = parse_uuid(obj.get("idempotency_key"))
    existing = session.execute(select(ElectionProject).where(ElectionProject.owner_key == owner, ElectionProject.creation_key == key)).scalar_one_or_none()
    if existing is not None:
        if (existing.title, existing.description) != (title, description):
            raise ProjectError("idempotency_conflict", 409)
        return view(existing)
    item = ElectionProject(owner_key=owner, creation_key=key, title=title, description=description, created_at=_now(), updated_at=_now())
    session.add(item); session.flush()
    return view(item)

def _advance(project: ElectionProject, version: object) -> None:
    if type(version) is not int or version != project.row_version:
        raise ProjectError("project_version_conflict", 409)
    project.row_version += 1
    project.updated_at = _now()

def update(session: Session, project_id: uuid.UUID, owner: str, body: object) -> dict:
    obj = require_obj(body, {"title", "description", "lifecycle", "expected_version"})
    project = _owned(session, project_id, owner, lock=True)
    if project.lifecycle == "archived" and obj.get("lifecycle") != "active":
        raise ProjectError("project_archived", 409)
    if "title" in obj: project.title = validated_text(obj["title"], limit=160, required=True)
    if "description" in obj: project.description = validated_text(obj["description"], limit=2000)
    if "lifecycle" in obj:
        if obj["lifecycle"] not in ("active", "archived"): raise ProjectError("invalid_lifecycle")
        project.lifecycle = obj["lifecycle"]
    _advance(project, obj.get("expected_version")); session.flush()
    return detail(session, project_id, owner)

def validate_scopes(raw: object) -> list[dict]:
    if not isinstance(raw, list) or len(raw) > 20:
        raise ProjectError("invalid_scope_count")
    normalized = []
    seen = set()
    for item in raw:
        data = require_obj(item, {"election_year","election_date","state_code","jurisdiction","contest","granularity"})
        year = data.get("election_year")
        if year is not None and (type(year) is not int or not 1788 <= year <= 2100):
            raise ProjectError("invalid_year")
        raw_date = data.get("election_date")
        if raw_date is not None:
            if not isinstance(raw_date, str): raise ProjectError("invalid_date")
            try: parsed_date = date.fromisoformat(raw_date)
            except ValueError as exc: raise ProjectError("invalid_date") from exc
            if parsed_date.isoformat() != raw_date or (year is not None and parsed_date.year != year):
                raise ProjectError("invalid_date")
        else: parsed_date = None
        state = data.get("state_code")
        if state is not None and (not isinstance(state, str) or not re.fullmatch(r"[A-Z]{2}", state)):
            raise ProjectError("invalid_state_code")
        jurisdiction = validated_text(data["jurisdiction"], limit=160) if data.get("jurisdiction") is not None else None
        contest = validated_text(data["contest"], limit=200) if data.get("contest") is not None else None
        granularity = data.get("granularity")
        if granularity not in (None, "statewide", "county", "municipality", "precinct", "unknown"):
            raise ProjectError("invalid_granularity")
        if not any((year, parsed_date, state, jurisdiction, contest, granularity)):
            raise ProjectError("empty_scope")
        row = {"election_year":year,"election_date":parsed_date,"state_code":state,
               "jurisdiction":jurisdiction,"contest":contest,"granularity":granularity}
        sig = tuple(row.values())
        if sig in seen: raise ProjectError("duplicate_scope")
        seen.add(sig); normalized.append(row)
    return normalized

def replace_scopes(session: Session, project_id: uuid.UUID, owner: str, body: object) -> dict:
    obj = require_obj(body, {"scopes", "expected_version"})
    scopes = validate_scopes(obj.get("scopes"))
    project = _owned(session, project_id, owner, lock=True)
    if project.lifecycle != "active": raise ProjectError("project_archived", 409)
    _advance(project, obj.get("expected_version"))
    for prior in session.execute(select(ElectionProjectScope).where(ElectionProjectScope.project_id == project_id)).scalars().all():
        session.delete(prior)
    session.flush()
    for item in scopes: session.add(ElectionProjectScope(project_id=project_id, **item))
    session.flush()
    return detail(session, project_id, owner)

def options(session: Session) -> list[dict]:
    # Display-only: approved public binding metadata, without any executable URL.
    q = text("SELECT id,year,contest,state,scope,format FROM source_registry_bindings "
             "WHERE review_state='approved' AND public_eligible = true ORDER BY year DESC, state, contest LIMIT 100")
    return [{"registry_binding_id": str(parse_uuid(row.id)), "year":row.year,"contest":row.contest,
             "state":row.state,"scope":row.scope,"format":row.format} for row in session.execute(q)]

def add_source_ref(session: Session, project_id: uuid.UUID, owner: str, body: object) -> dict:
    obj = require_obj(body, {"registry_binding_id", "expected_version"})
    binding_id = parse_uuid(obj.get("registry_binding_id"))
    project = _owned(session, project_id, owner, lock=True)
    if project.lifecycle != "active": raise ProjectError("project_archived", 409)
    if session.query(ElectionProjectSourceRef).filter_by(project_id=project_id).count() >= 50:
        raise ProjectError("source_reference_limit", 409)
    binding = session.execute(text("SELECT id,source_id,current_revision_id FROM source_registry_bindings "
                                   "WHERE id=:id AND review_state='approved' AND public_eligible=true").bindparams(
                                  bindparam("id",type_=Uuid(as_uuid=True))),
                              {"id":binding_id}).first()
    if binding is None: raise ProjectError("approved_binding_unavailable", 404)
    revision = parse_uuid(binding.current_revision_id)
    duplicate = session.execute(select(ElectionProjectSourceRef).where(
        ElectionProjectSourceRef.project_id == project_id,
        ElectionProjectSourceRef.registry_binding_id == binding_id,
        ElectionProjectSourceRef.registry_revision_id == revision)).scalar_one_or_none()
    _advance(project, obj.get("expected_version"))
    if duplicate is None:
        session.add(ElectionProjectSourceRef(project_id=project_id,registry_binding_id=binding_id,registry_revision_id=revision))
    session.flush()
    return detail(session, project_id, owner)

def remove_source_ref(session: Session, project_id: uuid.UUID, owner: str, association_id: uuid.UUID, body: object) -> dict:
    obj = require_obj(body, {"expected_version"})
    project = _owned(session, project_id, owner, lock=True)
    if project.lifecycle != "active": raise ProjectError("project_archived", 409)
    ref = session.execute(select(ElectionProjectSourceRef).where(ElectionProjectSourceRef.id == association_id,
                                ElectionProjectSourceRef.project_id == project_id)).scalar_one_or_none()
    if ref is None: raise ProjectError("reference_not_found", 404)
    _advance(project, obj.get("expected_version")); session.delete(ref); session.flush()
    return detail(session, project_id, owner)
