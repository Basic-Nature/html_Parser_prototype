"""Private project context routes; no parser execution or Registry governance."""
from __future__ import annotations
import os
from flask import Blueprint, jsonify, render_template, request
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from webapp.parser.auth.authority_model import classify_authority
from webapp.parser.auth.capability_policy import assert_trusted_action, CapabilityPolicyError
from webapp.parser.auth.workflow_csrf import issue_workflow_csrf_token, assert_workflow_csrf_token, WorkflowCsrfError
from webapp.parser.utils.privilege_tiers import get_principal_tier, PrivilegeTier
from webapp.parser.services import election_projects as svc


def create_election_projects_blueprint(*, principal_resolver, session_factory) -> Blueprint:
    bp = Blueprint("election_projects_routes", __name__)

    @bp.get("/projects")
    @bp.get("/projects/<uuid:project_id>")
    def project_page(project_id=None):
        # Landing is public; all private data is fetched through authorized APIs.
        return render_template("projects.html", csrf_token=issue_workflow_csrf_token())

    def actor():
        if os.getenv("ELECTION_PROJECTS_ENABLED", "").lower() not in ("true", "1"):
            raise svc.ProjectError("projects_not_enabled", 503)
        principal, source, _ = principal_resolver()
        authority = classify_authority(principal, source)
        if not authority["authenticated"] or authority["state"] == "development_bypass":
            raise svc.ProjectError("contributor_access_required", 401)
        try:
            assert_trusted_action(authority, get_principal_tier(principal, source),
                                  minimum_tier=PrivilegeTier.STANDARD_USER)
        except CapabilityPolicyError:
            raise svc.ProjectError("contributor_access_required", 403)
        return svc.owner_key(principal)

    def operation(fn, *, write=False):
        db = None
        try:
            key = actor()
            if write:
                assert_workflow_csrf_token(request.headers.get("X-CSRFToken"))
            db = session_factory()
            payload = fn(db, key)
            if write: db.commit()
            else: db.rollback()
            response = jsonify(payload)
            response.headers["Cache-Control"] = "no-store"
            return response
        except WorkflowCsrfError:
            if db is not None: db.rollback()
            return jsonify({"error": "invalid_csrf"}), 403
        except svc.ProjectError as exc:
            if db is not None: db.rollback()
            return jsonify({"error":exc.code}), exc.status
        except IntegrityError:
            if db is not None: db.rollback()
            return jsonify({"error":"project_write_conflict"}), 409
        except SQLAlchemyError:
            if db is not None: db.rollback()
            # Schema, service and network failures are not proof the user lacks permission.
            return jsonify({"error":"project_store_unavailable"}), 503
        finally:
            if db is not None: db.close()

    def body():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            raise svc.ProjectError("json_object_required")
        return data

    @bp.get("/api/projects/v1")
    def list_projects():
        return operation(lambda db, key: {"projects":svc.projects(db,key), "authority":"owner_only"})

    @bp.post("/api/projects/v1")
    def create_project():
        return operation(lambda db,key: svc.create(db,key,body()),write=True)

    @bp.get("/api/projects/v1/source-options")
    def source_options():
        return operation(lambda db,key: {"sources":svc.options(db), "authority":"metadata_only"})

    @bp.get("/api/projects/v1/<uuid:project_id>")
    def get_project(project_id):
        return operation(lambda db,key: svc.detail(db,project_id,key))

    @bp.patch("/api/projects/v1/<uuid:project_id>")
    def update_project(project_id):
        return operation(lambda db,key: svc.update(db,project_id,key,body()),write=True)

    @bp.put("/api/projects/v1/<uuid:project_id>/scopes")
    def replace_scopes(project_id):
        return operation(lambda db,key: svc.replace_scopes(db,project_id,key,body()),write=True)

    @bp.post("/api/projects/v1/<uuid:project_id>/source-refs")
    def add_source_ref(project_id):
        return operation(lambda db,key: svc.add_source_ref(db,project_id,key,body()),write=True)

    @bp.delete("/api/projects/v1/<uuid:project_id>/source-refs/<uuid:association_id>")
    def remove_source_ref(project_id,association_id):
        return operation(lambda db,key: svc.remove_source_ref(db,project_id,key,association_id,body()),write=True)
    return bp
