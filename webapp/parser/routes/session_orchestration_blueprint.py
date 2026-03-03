from __future__ import annotations

from flask import Blueprint, current_app, jsonify

from .route_monitor import record_route_monitor_event


def _call_handler(handler_name: str, *args, **kwargs):
    handlers = current_app.config.get("_SESSION_ORCHESTRATION_ROUTE_HANDLERS")
    if not isinstance(handlers, dict):
        record_route_monitor_event("session_orchestration", handler_name, "failure")
        return jsonify({"error": "Session orchestration routes are not configured."}), 500
    handler = handlers.get(handler_name)
    if not callable(handler):
        record_route_monitor_event("session_orchestration", handler_name, "failure")
        return jsonify({"error": f"Missing session orchestration handler: {handler_name}"}), 500
    try:
        response = handler(*args, **kwargs)
        record_route_monitor_event("session_orchestration", handler_name, "success")
        return response
    except Exception:
        record_route_monitor_event("session_orchestration", handler_name, "failure")
        raise


def create_session_orchestration_blueprint() -> Blueprint:
    bp = Blueprint("session_orchestration_routes", __name__)

    @bp.get("/api/session/enums", endpoint="get_session_enums")
    def get_session_enums_route():
        return _call_handler("get_session_enums")

    @bp.route("/test/ui/prompt", methods=["POST"], endpoint="test_ui_prompt")
    def test_ui_prompt_route():
        return _call_handler("test_ui_prompt")

    return bp