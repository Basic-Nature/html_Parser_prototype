from __future__ import annotations

from flask import Blueprint, current_app, jsonify

from .route_monitor import record_route_monitor_event


def _call_handler(handler_name: str, *args, **kwargs):
    handlers = current_app.config.get("_UI_NAVIGATION_ROUTE_HANDLERS")
    if not isinstance(handlers, dict):
        record_route_monitor_event("ui_navigation", handler_name, "failure")
        return jsonify({"error": "UI/navigation routes are not configured."}), 500
    handler = handlers.get(handler_name)
    if not callable(handler):
        record_route_monitor_event("ui_navigation", handler_name, "failure")
        return jsonify({"error": f"Missing UI/navigation handler: {handler_name}"}), 500
    try:
        response = handler(*args, **kwargs)
        record_route_monitor_event("ui_navigation", handler_name, "success")
        return response
    except Exception:
        record_route_monitor_event("ui_navigation", handler_name, "failure")
        raise


def create_ui_navigation_blueprint() -> Blueprint:
    bp = Blueprint("ui_navigation_routes", __name__)

    @bp.route("/favicon.ico", methods=["GET"], endpoint="favicon")
    def favicon_route():
        return _call_handler("favicon")

    @bp.route("/robots.txt", methods=["GET"], endpoint="robots_txt")
    def robots_txt_route():
        return _call_handler("robots_txt")

    @bp.route("/.well-known/appspecific/<path:filename>", methods=["GET"], endpoint="serve_well_known_appspecific")
    def serve_well_known_appspecific_route(filename: str):
        return _call_handler("serve_well_known_appspecific", filename)

    @bp.route("/site.webmanifest", methods=["GET"], endpoint="site_webmanifest")
    def site_webmanifest_route():
        return _call_handler("site_webmanifest")

    @bp.route("/quality_dashboard", methods=["GET"], endpoint="quality_dashboard")
    def quality_dashboard_route():
        return _call_handler("quality_dashboard")

    @bp.route("/url_status_dashboard", methods=["GET"], endpoint="url_status_dashboard")
    def url_status_dashboard_route():
        return _call_handler("url_status_dashboard")

    @bp.route("/quick-reference", methods=["GET"], endpoint="quick_reference_page")
    def quick_reference_page_route():
        return _call_handler("quick_reference_page")

    @bp.route("/quick_reference", methods=["GET"], endpoint="quick_reference_page_legacy")
    def quick_reference_page_legacy_route():
        return _call_handler("quick_reference_page")

    return bp