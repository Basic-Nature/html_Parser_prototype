from __future__ import annotations

from flask import Blueprint, current_app, jsonify

from .route_monitor import record_route_monitor_event


def _call_handler(handler_name: str, *args, **kwargs):
    handlers = current_app.config.get("_FEC_DATA_ASSURANCE_ROUTE_HANDLERS")
    if not isinstance(handlers, dict):
        record_route_monitor_event("fec_data_assurance", handler_name, "failure")
        return jsonify({"error": "FEC/Data Assurance routes are not configured."}), 500
    handler = handlers.get(handler_name)
    if not callable(handler):
        record_route_monitor_event("fec_data_assurance", handler_name, "failure")
        return jsonify({"error": f"Missing FEC/Data Assurance handler: {handler_name}"}), 500
    try:
        response = handler(*args, **kwargs)
        record_route_monitor_event("fec_data_assurance", handler_name, "success")
        return response
    except Exception:
        record_route_monitor_event("fec_data_assurance", handler_name, "failure")
        raise


def create_fec_data_assurance_blueprint() -> Blueprint:
    bp = Blueprint("fec_data_assurance_routes", __name__)

    @bp.route("/fec_mappings_review", methods=["GET"], endpoint="fec_mappings_review")
    def fec_mappings_review_route():
        return _call_handler("fec_mappings_review")

    @bp.route("/api/fec/problem_rows", methods=["GET"], endpoint="api_fec_problem_rows")
    def api_fec_problem_rows_route():
        return _call_handler("api_fec_problem_rows")

    @bp.route("/api/fec/save_mapping", methods=["POST"], endpoint="api_fec_save_mapping")
    def api_fec_save_mapping_route():
        return _call_handler("api_fec_save_mapping")

    @bp.route("/api/data-assurance/parse-and-classify", methods=["POST"], endpoint="api_data_assurance_classify")
    def api_data_assurance_classify_route():
        return _call_handler("api_data_assurance_classify")

    @bp.route("/api/data-assurance/verify-and-promote", methods=["POST"], endpoint="api_data_assurance_promote")
    def api_data_assurance_promote_route():
        return _call_handler("api_data_assurance_promote")

    @bp.route("/api/data-assurance/pending-dl2-reviews", methods=["GET"], endpoint="api_data_assurance_pending_reviews")
    def api_data_assurance_pending_reviews_route():
        return _call_handler("api_data_assurance_pending_reviews")

    return bp
