from __future__ import annotations

from flask import Blueprint, current_app, jsonify

from .route_monitor import record_route_monitor_event


def _call_handler(handler_name: str, *args, **kwargs):
    handlers = current_app.config.get("_ELECTION_DATA_ROUTE_HANDLERS")
    if not isinstance(handlers, dict):
        record_route_monitor_event("election_data", handler_name, "failure")
        return jsonify({"error": "Election data routes are not configured."}), 500
    handler = handlers.get(handler_name)
    if not callable(handler):
        record_route_monitor_event("election_data", handler_name, "failure")
        return jsonify({"error": f"Missing election data handler: {handler_name}"}), 500
    try:
        response = handler(*args, **kwargs)
        record_route_monitor_event("election_data", handler_name, "success")
        return response
    except Exception:
        record_route_monitor_event("election_data", handler_name, "failure")
        raise


def create_election_data_blueprint() -> Blueprint:
    bp = Blueprint("election_data_routes", __name__)

    @bp.route("/api/election_data/worklist", methods=["GET"], endpoint="api_election_data_worklist")
    def api_election_data_worklist_route():
        return _call_handler("api_election_data_worklist")

    @bp.route("/api/election_data/worklist/overview", methods=["GET"], endpoint="api_election_data_worklist_overview")
    def api_election_data_worklist_overview_route():
        return _call_handler("api_election_data_worklist_overview")

    @bp.route("/api/election_data/db_lite/finalized", methods=["GET"], endpoint="api_election_data_db_lite_finalized")
    def api_election_data_db_lite_finalized_route():
        return _call_handler("api_election_data_db_lite_finalized")

    @bp.route("/api/election_data/db_lite/down_ballot", methods=["GET"], endpoint="api_election_data_db_lite_down_ballot")
    def api_election_data_db_lite_down_ballot_route():
        return _call_handler("api_election_data_db_lite_down_ballot")

    @bp.route("/api/election_data/google_sheets/health", methods=["GET"], endpoint="api_election_data_google_sheets_health")
    def api_election_data_google_sheets_health_route():
        return _call_handler("api_election_data_google_sheets_health")

    @bp.route("/api/election_data/states_counties", methods=["GET"], endpoint="api_election_data_states_counties")
    def api_election_data_states_counties_route():
        return _call_handler("api_election_data_states_counties")

    @bp.route("/api/election_data/worklist/<race_id>/assign", methods=["POST"], endpoint="api_assign_dl_owner")
    def api_assign_dl_owner_route(race_id):
        return _call_handler("api_assign_dl_owner", race_id)

    @bp.route("/api/election_data/preqc/<race_id>", methods=["POST"], endpoint="api_preqc_check")
    def api_preqc_check_route(race_id):
        return _call_handler("api_preqc_check", race_id)

    @bp.route("/api/election_data/qc1/<race_id>/submit", methods=["POST"], endpoint="api_qc1_submit")
    def api_qc1_submit_route(race_id):
        return _call_handler("api_qc1_submit", race_id)

    @bp.route("/api/election_data/stats", methods=["GET"], endpoint="api_election_data_stats")
    def api_election_data_stats_route():
        return _call_handler("api_election_data_stats")

    @bp.route("/api/ballotlens-database", methods=["GET"], endpoint="api_ballotlens_database")
    def api_ballotlens_database_route():
        return _call_handler("api_ballotlens_database")

    @bp.route("/api/warehouse_election_results", methods=["GET"], endpoint="api_warehouse_election_results")
    def api_warehouse_election_results_route():
        return _call_handler("api_warehouse_election_results")

    return bp
