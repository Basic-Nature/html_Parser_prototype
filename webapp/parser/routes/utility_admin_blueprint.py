from __future__ import annotations

from flask import Blueprint, current_app, jsonify

from .route_monitor import record_route_monitor_event


def _call_handler(handler_name: str, *args, **kwargs):
    handlers = current_app.config.get("_UTILITY_ADMIN_ROUTE_HANDLERS")
    if not isinstance(handlers, dict):
        record_route_monitor_event("utility_admin", handler_name, "failure")
        return jsonify({"error": "Utility/admin routes are not configured."}), 500
    handler = handlers.get(handler_name)
    if not callable(handler):
        record_route_monitor_event("utility_admin", handler_name, "failure")
        return jsonify({"error": f"Missing utility/admin handler: {handler_name}"}), 500
    try:
        response = handler(*args, **kwargs)
        record_route_monitor_event("utility_admin", handler_name, "success")
        return response
    except Exception:
        record_route_monitor_event("utility_admin", handler_name, "failure")
        raise


def create_utility_admin_blueprint() -> Blueprint:
    bp = Blueprint("utility_admin_routes", __name__)

    @bp.route("/api/fs/list", methods=["GET"], endpoint="api_fs_list")
    def api_fs_list_route():
        return _call_handler("api_fs_list")

    @bp.route("/api/list_dir", methods=["GET"], endpoint="api_list_dir_compat")
    def api_list_dir_compat_route():
        return _call_handler("api_list_dir_compat")

    @bp.route("/api/fs/mkdir", methods=["POST"], endpoint="api_fs_mkdir")
    def api_fs_mkdir_route():
        return _call_handler("api_fs_mkdir")

    @bp.route("/api/fs/delete", methods=["POST"], endpoint="api_fs_delete")
    def api_fs_delete_route():
        return _call_handler("api_fs_delete")

    @bp.route("/api/quick_copy", methods=["POST"], endpoint="api_quick_copy")
    def api_quick_copy_route():
        return _call_handler("api_quick_copy")

    @bp.route("/api/quick_copy/clear", methods=["POST"], endpoint="api_quick_copy_clear")
    def api_quick_copy_clear_route():
        return _call_handler("api_quick_copy_clear")

    @bp.route("/api/validate_urls", methods=["POST"], endpoint="api_validate_urls")
    def api_validate_urls_route():
        return _call_handler("api_validate_urls")

    @bp.route("/api/url_status", methods=["GET"], endpoint="api_url_status")
    def api_url_status_route():
        return _call_handler("api_url_status")

    @bp.route("/api/auth/certificate_info", methods=["GET"], endpoint="api_auth_certificate_info")
    def api_auth_certificate_info_route():
        return _call_handler("api_auth_certificate_info")

    @bp.route("/api/auth/status", methods=["GET"], endpoint="api_auth_status")
    def api_auth_status_route():
        return _call_handler("api_auth_status")

    @bp.route("/api/route_wrappers/monitor", methods=["GET"], endpoint="api_route_wrapper_monitor_snapshot")
    def api_route_wrapper_monitor_snapshot_route():
        return _call_handler("api_route_wrapper_monitor_snapshot")

    return bp
