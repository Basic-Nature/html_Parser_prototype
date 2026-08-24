"""Routes for the read-only ElectionPulse workflow_v1 operational API."""

from __future__ import annotations

from flask import Blueprint, current_app, jsonify

from .route_monitor import record_route_monitor_event


_HANDLER_CONFIG_KEY = "_WORKFLOW_V1_ROUTE_HANDLERS"


def _call_handler(handler_name: str, *args, **kwargs):
    handlers = current_app.config.get(_HANDLER_CONFIG_KEY)
    if not isinstance(handlers, dict):
        record_route_monitor_event(
            "workflow_v1",
            handler_name,
            "failure",
        )
        return jsonify(
            {"error": "Workflow v1 routes are not configured."}
        ), 500

    handler = handlers.get(handler_name)
    if not callable(handler):
        record_route_monitor_event(
            "workflow_v1",
            handler_name,
            "failure",
        )
        return jsonify(
            {"error": f"Missing workflow v1 handler: {handler_name}"}
        ), 500

    try:
        response = handler(*args, **kwargs)
        record_route_monitor_event(
            "workflow_v1",
            handler_name,
            "success",
        )
        return response
    except Exception:
        record_route_monitor_event(
            "workflow_v1",
            handler_name,
            "failure",
        )
        raise


def create_workflow_v1_blueprint() -> Blueprint:
    bp = Blueprint("workflow_v1_routes", __name__)

    @bp.route(
        "/api/workflow/v1/items",
        methods=["GET"],
        endpoint="api_workflow_v1_items",
    )
    def items_route():
        return _call_handler("api_workflow_v1_items")

    @bp.route(
        "/api/workflow/v1/items/<uuid:item_id>",
        methods=["GET"],
        endpoint="api_workflow_v1_item_detail",
    )
    def item_detail_route(item_id):
        return _call_handler(
            "api_workflow_v1_item_detail",
            item_id=item_id,
        )

    @bp.route(
        "/api/workflow/v1/facets",
        methods=["GET"],
        endpoint="api_workflow_v1_facets",
    )
    def facets_route():
        return _call_handler("api_workflow_v1_facets")

    @bp.route(
        "/api/workflow/v1/stats",
        methods=["GET"],
        endpoint="api_workflow_v1_stats",
    )
    def stats_route():
        return _call_handler("api_workflow_v1_stats")

    return bp
