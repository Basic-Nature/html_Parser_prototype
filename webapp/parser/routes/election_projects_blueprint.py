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
        return svc.owner_key(principal), principal

    def operation(fn, *, write=False, include_principal=False):
        db = None
        try:
            key, principal = actor()
            if write:
                assert_workflow_csrf_token(request.headers.get("X-CSRFToken"))
            db = session_factory()
            payload = fn(db, key, principal) if include_principal else fn(db, key)
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
    @bp.get("/api/projects/v1/<uuid:project_id>/run-preflight")
    def project_run_preflight_route(project_id):
        # GET produces a non-executable preview. A future J2B mutation must
        # independently reauthorize and consume a durable human intent.
        def review(db, owner, principal):
            if os.getenv("ELECTION_PROJECT_RUN_PREFLIGHT_ENABLED", "").lower() not in ("1", "true"):
                raise svc.ProjectError("project_run_preflight_disabled", 503)
            required = {"source_ref_id", "workflow_item_id", "expected_project_version"}
            if set(request.args.keys()) != required or any(
                len(request.args.getlist(key)) != 1 for key in required
            ):
                raise svc.ProjectError("invalid_preflight_selectors")
            def canonical_id(key):
                raw = request.args.get(key, "")
                parsed = svc.parse_uuid(raw)
                if str(parsed) != raw.lower():
                    raise svc.ProjectError("invalid_preflight_selectors")
                return parsed
            version = request.args.get("expected_project_version", "")
            if (len(version) > 10 or not version.isascii() or not version.isdigit()
                or str(int(version)) != version or not 1 <= int(version) <= 2147483647):
                raise svc.ProjectError("invalid_project_version")
            from webapp.parser.services.project_run_preflight import project_run_preflight
            from webapp.parser.config import URL_LIST_FILE
            return project_run_preflight(
                db, project_id=project_id,
                source_ref_id=canonical_id("source_ref_id"),
                workflow_item_id=canonical_id("workflow_item_id"),
                expected_project_version=int(version), owner_key=owner,
                principal=principal, registry_path=URL_LIST_FILE,
            )
        return operation(review, include_principal=True)


    def admission_enabled():
        if os.getenv("ELECTION_PROJECT_RUN_ADMISSION_ENABLED", "").lower() not in ("1", "true"):
            raise svc.ProjectError("project_run_admission_disabled", 503)

    @bp.post("/api/projects/v1/<uuid:project_id>/runs")
    def admit_project_run(project_id):
        def admit(db, owner, principal):
            admission_enabled()
            from webapp.parser.services.project_run_admission import admit as admit_run
            from webapp.parser.config import URL_LIST_FILE
            return admit_run(db,project_id=project_id,owner_key=owner,
                principal=principal,body=body(),registry_path=URL_LIST_FILE)
        return operation(admit,write=True,include_principal=True)

    @bp.get("/api/projects/v1/<uuid:project_id>/runs")
    def list_project_runs(project_id):
        def listing(db, owner):
            admission_enabled()
            from webapp.parser.services.project_run_admission import list_runs
            return list_runs(db,project_id=project_id,owner_key=owner)
        return operation(listing)

    @bp.get("/api/projects/v1/<uuid:project_id>/runs/<uuid:run_id>")
    def get_project_run(project_id,run_id):
        def get(db, owner):
            admission_enabled()
            from webapp.parser.services.project_run_admission import get_run
            return get_run(db,project_id=project_id,run_id=run_id,owner_key=owner)
        return operation(get)

    return bp
