from __future__ import annotations

from flask import Blueprint, current_app, jsonify

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

    @bp.route("/ocr_diagnostics", methods=["GET"], endpoint="ocr_diagnostics")
    def ocr_diagnostics_route():
        return _call_handler("ocr_diagnostics")

    return bp