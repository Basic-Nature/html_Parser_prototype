"""Protected contributor routes for the governed workflow plane."""

from __future__ import annotations

from flask import Blueprint, current_app, jsonify


_HANDLER_CONFIG_KEY = "_WORKFLOW_CONTRIBUTOR_ROUTE_HANDLERS"


def _call_handler(handler_name: str, *args, **kwargs):
    handlers = current_app.config.get(_HANDLER_CONFIG_KEY)
    if not isinstance(handlers, dict):
        return jsonify(
            {"error": "Workflow contributor routes are not configured."}
        ), 500

    handler = handlers.get(handler_name)
    if not callable(handler):
        return jsonify(
            {"error": f"Missing workflow contributor handler: {handler_name}"}
        ), 500

    return handler(*args, **kwargs)


def create_workflow_contributor_blueprint() -> Blueprint:
    bp = Blueprint("workflow_contributor_routes", __name__)

    @bp.route(
        "/api/workflow/v1/contributor/items/<uuid:item_id>/source",
        methods=["GET"],
        endpoint="api_workflow_v1_contributor_source",
    )
    def contributor_source_route(item_id):
        return _call_handler(
            "api_workflow_v1_contributor_source",
            item_id=item_id,
        )

    @bp.route(
        "/api/workflow/v1/contributor/items/<uuid:item_id>/passes/1/claim",
        methods=["POST"],
        endpoint="api_workflow_v1_claim_first_pass",
    )
    def claim_first_pass_route(item_id):
        return _call_handler(
            "api_workflow_v1_claim_first_pass",
            item_id=item_id,
        )

    return bp
