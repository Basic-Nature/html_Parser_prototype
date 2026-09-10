# Protected reviewer routes for the governed workflow plane.

from __future__ import annotations

from flask import Blueprint, current_app, jsonify


_HANDLER_CONFIG_KEY = "_WORKFLOW_REVIEWER_ROUTE_HANDLERS"


def _call_handler(handler_name: str, *args, **kwargs):
    handlers = current_app.config.get(_HANDLER_CONFIG_KEY)
    if not isinstance(handlers, dict):
        return jsonify(
            {"error": "Workflow reviewer routes are not configured."}
        ), 500

    handler = handlers.get(handler_name)
    if not callable(handler):
        return jsonify(
            {"error": f"Missing workflow reviewer handler: {handler_name}"}
        ), 500

    return handler(*args, **kwargs)


def create_workflow_reviewer_blueprint() -> Blueprint:
    bp = Blueprint("workflow_reviewer_routes", __name__)

    @bp.route(
        (
            "/api/workflow/v1/reviewer/items/<uuid:item_id>/"
            "comparisons/<uuid:comparison_id>/resolve"
        ),
        methods=["POST"],
        endpoint="api_workflow_v1_resolve_discrepancies",
    )
    def resolve_discrepancies_route(item_id, comparison_id):
        return _call_handler(
            "api_workflow_v1_resolve_discrepancies",
            item_id=item_id,
            comparison_id=comparison_id,
        )

    @bp.route(
        "/api/workflow/v1/reviewer/items/<uuid:item_id>/reviews/qc1",
        methods=["POST"],
        endpoint="api_workflow_v1_submit_qc1_review",
    )
    def submit_qc1_review_route(item_id):
        return _call_handler(
            "api_workflow_v1_submit_qc1_review",
            item_id=item_id,
        )

    @bp.route(
        "/api/workflow/v1/reviewer/items/<uuid:item_id>/reviews/qc2",
        methods=["POST"],
        endpoint="api_workflow_v1_submit_qc2_review",
    )
    def submit_qc2_review_route(item_id):
        return _call_handler(
            "api_workflow_v1_submit_qc2_review",
            item_id=item_id,
        )

    return bp
