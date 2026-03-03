from __future__ import annotations

from flask import Blueprint, current_app, jsonify

from .route_monitor import record_route_monitor_event


def _call_handler(handler_name: str, *args, **kwargs):
    handlers = current_app.config.get("_DATA_FRAMEWORK_ROUTE_HANDLERS")
    if not isinstance(handlers, dict):
        record_route_monitor_event("data_framework", handler_name, "failure")
        return jsonify({"error": "Data framework routes are not configured."}), 500
    handler = handlers.get(handler_name)
    if not callable(handler):
        record_route_monitor_event("data_framework", handler_name, "failure")
        return jsonify({"error": f"Missing data framework handler: {handler_name}"}), 500
    try:
        response = handler(*args, **kwargs)
        record_route_monitor_event("data_framework", handler_name, "success")
        return response
    except Exception:
        record_route_monitor_event("data_framework", handler_name, "failure")
        raise


def create_data_framework_blueprint() -> Blueprint:
    bp = Blueprint("data_framework_routes", __name__)

    @bp.route("/data_framework", methods=["GET"], endpoint="data_framework")
    def data_framework_route():
        return _call_handler("data_framework")

    @bp.route("/api/data_framework/preview", methods=["GET"], endpoint="api_data_framework_preview")
    def api_data_framework_preview_route():
        return _call_handler("api_data_framework_preview")

    @bp.route("/api/data_framework/scaffold", methods=["GET"], endpoint="api_data_framework_scaffold")
    def api_data_framework_scaffold_route():
        return _call_handler("api_data_framework_scaffold")

    @bp.route("/api/data_framework/scaffold.csv", methods=["GET"], endpoint="api_data_framework_scaffold_csv")
    def api_data_framework_scaffold_csv_route():
        return _call_handler("api_data_framework_scaffold_csv")

    @bp.route("/api/data_framework/curated", methods=["GET"], endpoint="api_data_framework_curated")
    def api_data_framework_curated_route():
        return _call_handler("api_data_framework_curated")

    @bp.route("/api/data_framework/warehouse_status", methods=["GET"], endpoint="api_data_framework_warehouse_status")
    def api_data_framework_warehouse_status_route():
        return _call_handler("api_data_framework_warehouse_status")

    @bp.route("/api/data_framework/exports", methods=["GET"], endpoint="api_data_framework_exports")
    def api_data_framework_exports_route():
        return _call_handler("api_data_framework_exports")

    return bp