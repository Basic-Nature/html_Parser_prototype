from __future__ import annotations

from flask import Blueprint, current_app, jsonify

from .route_monitor import record_route_monitor_event


def _call_handler(handler_name: str, *args, **kwargs):
    handlers = current_app.config.get("_OBSERVABILITY_ROUTE_HANDLERS")
    if not isinstance(handlers, dict):
        record_route_monitor_event("observability", handler_name, "failure")
        return jsonify({"error": "Observability routes are not configured."}), 500
    handler = handlers.get(handler_name)
    if not callable(handler):
        record_route_monitor_event("observability", handler_name, "failure")
        return jsonify({"error": f"Missing observability handler: {handler_name}"}), 500
    try:
        response = handler(*args, **kwargs)
        record_route_monitor_event("observability", handler_name, "success")
        return response
    except Exception:
        record_route_monitor_event("observability", handler_name, "failure")
        raise


def create_observability_blueprint() -> Blueprint:
    bp = Blueprint("observability_routes", __name__)

    @bp.route("/api/integrity_trends", methods=["GET"], endpoint="api_integrity_trends")
    def api_integrity_trends_route():
        return _call_handler("api_integrity_trends")

    @bp.route("/api/integrity_signal", methods=["POST"], endpoint="api_integrity_signal")
    def api_integrity_signal_route():
        return _call_handler("api_integrity_signal")

    @bp.route("/api/integrity_export", methods=["GET"], endpoint="api_integrity_export")
    def api_integrity_export_route():
        return _call_handler("api_integrity_export")

    @bp.route("/api/quality_metrics", methods=["GET"], endpoint="api_quality_metrics")
    def api_quality_metrics_route():
        return _call_handler("api_quality_metrics")

    @bp.route("/api/ml_usage", methods=["GET"], endpoint="api_ml_usage")
    def api_ml_usage_route():
        return _call_handler("api_ml_usage")

    @bp.route("/api/ml_pipeline_profile", methods=["GET"], endpoint="api_ml_pipeline_profile")
    def api_ml_pipeline_profile_route():
        return _call_handler("api_ml_pipeline_profile")

    @bp.route("/api/ml_vocab_alignment", methods=["GET"], endpoint="api_ml_vocab_alignment")
    def api_ml_vocab_alignment_route():
        return _call_handler("api_ml_vocab_alignment")

    @bp.route("/api/ml_vocab_alignment_suggestions", methods=["GET"], endpoint="api_ml_vocab_alignment_suggestions")
    def api_ml_vocab_alignment_suggestions_route():
        return _call_handler("api_ml_vocab_alignment_suggestions")

    @bp.route("/api/ml_vocab_alignment_suggestions/export", methods=["GET"], endpoint="api_ml_vocab_alignment_suggestions_export")
    def api_ml_vocab_alignment_suggestions_export_route():
        return _call_handler("api_ml_vocab_alignment_suggestions_export")

    @bp.route("/api/preingest_url_glimpse", methods=["GET"], endpoint="api_preingest_url_glimpse")
    def api_preingest_url_glimpse_route():
        return _call_handler("api_preingest_url_glimpse")

    return bp
