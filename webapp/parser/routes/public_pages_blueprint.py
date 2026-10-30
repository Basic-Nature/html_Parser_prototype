from __future__ import annotations

from flask import Blueprint, abort, current_app, jsonify, render_template, make_response
from .public_support_snapshot import load_approved_public_snapshot

from .route_monitor import record_route_monitor_event

def _call_handler(handler_name: str, *args, **kwargs):
    handlers = current_app.config.get("_PUBLIC_PAGES_ROUTE_HANDLERS")
    if not isinstance(handlers, dict):
        record_route_monitor_event("public_pages", handler_name, "failure")
        return jsonify({"error": "Public page routes are not configured."}), 500
    handler = handlers.get(handler_name)
    if not callable(handler):
        record_route_monitor_event("public_pages", handler_name, "failure")
        return jsonify({"error": f"Missing public page handler: {handler_name}"}), 500
    try:
        response = handler(*args, **kwargs)
        record_route_monitor_event("public_pages", handler_name, "success")
        return response
    except Exception:
        record_route_monitor_event("public_pages", handler_name, "failure")
        raise

def create_public_pages_blueprint() -> Blueprint:
    bp = Blueprint("public_pages_routes", __name__)

    @bp.route("/", methods=["GET"], endpoint="index")
    def index_route():
        return _call_handler("index")

    @bp.route("/support", methods=["GET"], endpoint="support")
    def support_route():
        # Public support information and local preview are separate exact-True gates.
        # This does not authorize financial-data publication or any payment activity.
        public = current_app.config.get("EP_SUPPORT_PAGE_PUBLICATION_ENABLED") is True
        preview = current_app.config.get("EP_SUPPORT_LOCAL_PREVIEW_ENABLED") is True
        if not public and not preview:
            abort(404)
        # Reserved: a payment feature flag alone has no checkout capability.
        # The public page always renders without a payment provider integration.
        approved_financial = public and current_app.config.get("EP_SUPPORT_FINANCIAL_PUBLICATION_ENABLED") is True
        return render_template("support.html", support_is_public=public,
                               support_financial_publication_enabled=approved_financial)

    @bp.route("/support/financial-snapshot", methods=["GET"], endpoint="support_financial_snapshot")
    def support_financial_snapshot():
        # This is a separate server-side authority boundary, NOT a static JS asset.
        # Publication needs all three independent controls and a vetted instance file.
        if current_app.config.get("EP_SUPPORT_PAGE_PUBLICATION_ENABLED") is not True:
            abort(404)
        if current_app.config.get("EP_SUPPORT_FINANCIAL_PUBLICATION_ENABLED") is not True:
            abort(404)
        approval_id = current_app.config.get("EP_SUPPORT_APPROVED_FINANCIAL_RECORD_ID")
        snapshot = load_approved_public_snapshot(current_app.instance_path, approval_id)
        if snapshot is None:
            abort(404)
        response = make_response(jsonify(snapshot))
        response.headers["Cache-Control"] = "no-store, private, max-age=0"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @bp.route("/ballot_lens", methods=["GET", "POST"], endpoint="ballot_lens")
    def ballot_lens_route():
        return _call_handler("ballot_lens")

    @bp.route("/worklist", methods=["GET"], endpoint="worklist")
    def worklist_route():
        return _call_handler("worklist")

    @bp.route("/auth/welcome", methods=["GET"], endpoint="auth_welcome")
    def auth_welcome_route():
        return _call_handler("auth_welcome")

    @bp.route("/auth/challenge", methods=["GET"], endpoint="auth_challenge")
    def auth_challenge_route():
        return _call_handler("auth_challenge")

    @bp.route("/auth/certificate/start", methods=["GET"], endpoint="auth_certificate_start")
    def auth_certificate_start_route():
        from webapp.parser.auth.trusted_access import begin_trusted_certificate_access
        return begin_trusted_certificate_access()

    @bp.route("/auth/certificate/verify", methods=["GET"], endpoint="auth_certificate_verify")
    def auth_certificate_verify_route():
        from webapp.parser.auth.trusted_access_boundary import verify_trusted_certificate_access
        return verify_trusted_certificate_access()

    @bp.route("/auth/certificate/complete", methods=["GET"], endpoint="auth_certificate_complete")
    def auth_certificate_complete_route():
        from webapp.parser.auth.trusted_access_boundary import complete_trusted_certificate_access
        return complete_trusted_certificate_access()

    @bp.route("/ocr_diagnostics", methods=["GET"], endpoint="ocr_diagnostics")
    def ocr_diagnostics_route():
        return _call_handler("ocr_diagnostics")

    return bp
