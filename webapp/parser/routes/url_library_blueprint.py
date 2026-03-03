from __future__ import annotations

from flask import Blueprint, current_app, jsonify

from .route_monitor import record_route_monitor_event


def _call_handler(handler_name: str, *args, **kwargs):
    handlers = current_app.config.get("_URL_LIBRARY_ROUTE_HANDLERS")
    if not isinstance(handlers, dict):
        record_route_monitor_event("url_library", handler_name, "failure")
        return jsonify({"error": "URL library routes are not configured."}), 500
    handler = handlers.get(handler_name)
    if not callable(handler):
        record_route_monitor_event("url_library", handler_name, "failure")
        return jsonify({"error": f"Missing URL library handler: {handler_name}"}), 500
    try:
        response = handler(*args, **kwargs)
        record_route_monitor_event("url_library", handler_name, "success")
        return response
    except Exception:
        record_route_monitor_event("url_library", handler_name, "failure")
        raise


def create_url_library_blueprint() -> Blueprint:
    bp = Blueprint("url_library_routes", __name__)

    @bp.route("/api/urls", methods=["GET", "POST"], endpoint="api_urls")
    def api_urls_route():
        return _call_handler("api_urls")

    @bp.route("/api/urls/parse", methods=["POST"], endpoint="api_urls_parse")
    def api_urls_parse_route():
        return _call_handler("api_urls_parse")

    @bp.route("/api/urls/training_data", methods=["GET"], endpoint="api_urls_training_data")
    def api_urls_training_data_route():
        return _call_handler("api_urls_training_data")

    @bp.route("/api/urls/parse_all", methods=["POST"], endpoint="api_urls_parse_all")
    def api_urls_parse_all_route():
        return _call_handler("api_urls_parse_all")

    @bp.route("/api/filename/parse", methods=["POST"], endpoint="api_filename_parse")
    def api_filename_parse_route():
        return _call_handler("api_filename_parse")

    @bp.route("/api/outputs/lookup", methods=["GET"], endpoint="api_outputs_lookup")
    def api_outputs_lookup_route():
        return _call_handler("api_outputs_lookup")

    @bp.route("/api/warehouse/match", methods=["GET"], endpoint="api_warehouse_match")
    def api_warehouse_match_route():
        return _call_handler("api_warehouse_match")

    @bp.route("/api/warehouse/export", methods=["GET"], endpoint="api_warehouse_export")
    def api_warehouse_export_route():
        return _call_handler("api_warehouse_export")

    @bp.route("/api/warehouse/coverage", methods=["GET"], endpoint="api_warehouse_coverage")
    def api_warehouse_coverage_route():
        return _call_handler("api_warehouse_coverage")

    return bp
