from __future__ import annotations

from flask import Blueprint, current_app, jsonify

from .route_monitor import record_route_monitor_event


def _call_handler(handler_name: str, *args, **kwargs):
    handlers = current_app.config.get("_FILE_IO_ROUTE_HANDLERS")
    if not isinstance(handlers, dict):
        record_route_monitor_event("file_io", handler_name, "failure")
        return jsonify({"error": "File I/O routes are not configured."}), 500
    handler = handlers.get(handler_name)
    if not callable(handler):
        record_route_monitor_event("file_io", handler_name, "failure")
        return jsonify({"error": f"Missing file I/O handler: {handler_name}"}), 500
    try:
        response = handler(*args, **kwargs)
        record_route_monitor_event("file_io", handler_name, "success")
        return response
    except Exception:
        record_route_monitor_event("file_io", handler_name, "failure")
        raise


def create_file_io_blueprint() -> Blueprint:
    bp = Blueprint("file_io_routes", __name__)

    @bp.route("/download_fs", methods=["GET"], endpoint="download_fs")
    def download_fs_route():
        return _call_handler("download_fs")

    @bp.route("/view_csv", methods=["GET"], endpoint="view_csv")
    def view_csv_route():
        return _call_handler("view_csv")

    @bp.route("/csv_locate", methods=["GET"], endpoint="csv_locate")
    def csv_locate_route():
        return _call_handler("csv_locate")

    @bp.route("/delete/input/<filename>", methods=["POST"], endpoint="delete_input_file")
    def delete_input_file_route(filename: str):
        return _call_handler("delete_input_file", filename)

    @bp.route("/delete/output/<filename>", methods=["POST"], endpoint="delete_output_file")
    def delete_output_file_route(filename: str):
        return _call_handler("delete_output_file", filename)

    @bp.route("/delete/uploads/<filename>", methods=["POST"], endpoint="delete_upload_file")
    def delete_upload_file_route(filename: str):
        return _call_handler("delete_upload_file", filename)

    @bp.route("/download/input/<filename>", methods=["GET"], endpoint="download_input_file")
    def download_input_file_route(filename: str):
        return _call_handler("download_input_file", filename)

    @bp.route("/download/output/<filename>", methods=["GET"], endpoint="download_output_file")
    def download_output_file_route(filename: str):
        return _call_handler("download_output_file", filename)

    @bp.route("/download/uploads/<filename>", methods=["GET"], endpoint="download_upload_file")
    def download_upload_file_route(filename: str):
        return _call_handler("download_upload_file", filename)

    @bp.route("/upload/input", methods=["POST"], endpoint="upload_to_input")
    def upload_to_input_route():
        return _call_handler("upload_to_input")

    @bp.route("/upload/output", methods=["POST"], endpoint="upload_to_output")
    def upload_to_output_route():
        return _call_handler("upload_to_output")

    @bp.route("/upload/uploads", methods=["POST"], endpoint="upload_to_uploads")
    def upload_to_uploads_route():
        return _call_handler("upload_to_uploads")

    @bp.route("/heartbeat", methods=["GET"], endpoint="heartbeat")
    def heartbeat_route():
        return _call_handler("heartbeat")

    @bp.route("/Heartbeat", methods=["GET"], endpoint="heartbeat_legacy")
    def heartbeat_legacy_route():
        return _call_handler("heartbeat")

    @bp.route("/clear_history", methods=["POST"], endpoint="clear_history")
    def clear_history_route():
        return _call_handler("clear_history")

    @bp.route("/history", methods=["GET"], endpoint="history")
    def history_route():
        return _call_handler("history")

    @bp.route("/rerun/<run_id>", methods=["POST"], endpoint="rerun_prior")
    def rerun_prior_route(run_id: str):
        return _call_handler("rerun_prior", run_id)

    return bp