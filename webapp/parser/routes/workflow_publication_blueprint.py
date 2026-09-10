"""Protected publication-operator route for governed Workflow handoff."""

from __future__ import annotations

from flask import Blueprint, current_app, jsonify


_HANDLER_CONFIG_KEY = "_WORKFLOW_PUBLICATION_ROUTE_HANDLERS"


def _call_handler(handler_name: str, *args, **kwargs):
    handlers = current_app.config.get(_HANDLER_CONFIG_KEY)
    if not isinstance(handlers, dict):
        return jsonify(
            {"error": "Workflow publication routes are not configured."}
        ), 500

    handler = handlers.get(handler_name)
    if not callable(handler):
        return jsonify(
            {"error": f"Missing Workflow publication handler: {handler_name}"}
        ), 500

    return handler(*args, **kwargs)


def create_workflow_publication_blueprint() -> Blueprint:
    bp = Blueprint("workflow_publication_routes", __name__)

    @bp.route(
        "/api/workflow/v1/publication/items/<uuid:item_id>/handoff",
        methods=["POST"],
        endpoint="api_workflow_v1_publication_handoff",
    )
    def publication_handoff_route(item_id):
        return _call_handler(
            "api_workflow_v1_publication_handoff",
            item_id=item_id,
        )

    return bp
