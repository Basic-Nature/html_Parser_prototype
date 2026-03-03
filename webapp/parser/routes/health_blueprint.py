from __future__ import annotations

from flask import Blueprint, current_app, jsonify

from .route_monitor import record_route_monitor_event


def _call_handler(handler_name: str, *args, **kwargs):
    handlers = current_app.config.get("_HEALTH_ROUTE_HANDLERS")
    if not isinstance(handlers, dict):
        record_route_monitor_event("health", handler_name, "failure")
        return jsonify({"error": "Health routes are not configured."}), 500
    handler = handlers.get(handler_name)
    if not callable(handler):
        record_route_monitor_event("health", handler_name, "failure")
        return jsonify({"error": f"Missing health handler: {handler_name}"}), 500
    try:
        response = handler(*args, **kwargs)
        record_route_monitor_event("health", handler_name, "success")
        return response
    except Exception:
        record_route_monitor_event("health", handler_name, "failure")
        raise


def create_health_blueprint() -> Blueprint:
    bp = Blueprint("health_routes", __name__)

    @bp.route("/health_dashboard", methods=["GET"], endpoint="health_dashboard")
    def health_dashboard_route():
        return _call_handler("health_dashboard")

    @bp.route("/api/health_tasks", methods=["GET"], endpoint="api_list_health_tasks")
    def api_list_health_tasks_route():
        return _call_handler("api_list_health_tasks")

    @bp.route("/api/health_tasks", methods=["POST"], endpoint="api_start_health_task")
    def api_start_health_task_route():
        return _call_handler("api_start_health_task")

    @bp.route("/api/health_tasks/<task_id>", methods=["GET"], endpoint="api_health_task_detail")
    def api_health_task_detail_route(task_id: str):
        return _call_handler("api_health_task_detail", task_id)

    @bp.route("/api/health_socket_test", methods=["POST"], endpoint="api_health_socket_test")
    def api_health_socket_test_route():
        return _call_handler("api_health_socket_test")

    @bp.route("/health", methods=["GET"], endpoint="health")
    def health_route():
        return _call_handler("health")

    return bp