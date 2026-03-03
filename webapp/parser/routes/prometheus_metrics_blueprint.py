from __future__ import annotations

from flask import Blueprint, current_app, jsonify

from .route_monitor import record_route_monitor_event


def _call_handler(handler_name: str, *args, **kwargs):
    handlers = current_app.config.get("_PROMETHEUS_METRICS_ROUTE_HANDLERS")
    if not isinstance(handlers, dict):
        record_route_monitor_event("prometheus_metrics", handler_name, "failure")
        return jsonify({"error": "Prometheus metrics routes are not configured."}), 500
    handler = handlers.get(handler_name)
    if not callable(handler):
        record_route_monitor_event("prometheus_metrics", handler_name, "failure")
        return jsonify({"error": f"Missing Prometheus metrics handler: {handler_name}"}), 500
    try:
        response = handler(*args, **kwargs)
        record_route_monitor_event("prometheus_metrics", handler_name, "success")
        return response
    except Exception:
        record_route_monitor_event("prometheus_metrics", handler_name, "failure")
        raise


def create_prometheus_metrics_blueprint(*, include_test_increment: bool = False) -> Blueprint:
    bp = Blueprint("prometheus_metrics_routes", __name__)

    @bp.route("/metrics", methods=["GET"], endpoint="metrics")
    def metrics_route():
        return _call_handler("metrics")

    if include_test_increment:
        @bp.route("/test/metrics/increment", methods=["POST"], endpoint="test_metrics_increment")
        def test_metrics_increment_route():
            return _call_handler("test_metrics_increment")

    return bp
