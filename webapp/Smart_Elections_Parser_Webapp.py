from __future__ import annotations
# Smart_Elections_Parser_Webapp.py
# -----------------------------------------------------------
# Web Application for Smart Elections Parser
# -----------------------------------------------------------
# Structure:
#   1. Imports & Environment Setup
#   2. Flask App & SocketIO Initialization
#   3. Session & State Management
#   4. Utility Functions
#   5. Routes (Flask)
#   6. SocketIO Event Handlers
#   7. Main Entrypoint
# -----------------------------------------------------------
# 1. Imports & Environment Setup
import csv
from datetime import datetime, timezone
from difflib import get_close_matches
from flask import (
    Flask, render_template, request, redirect, session, 
    url_for, flash, send_file, send_from_directory,
    jsonify
)   
from flask_socketio import emit, SocketIO, join_room
import importlib
from io import StringIO
import orjson
import os
import time
from threading import Thread
from queue import Queue

# Project-specific imports
from webapp.parser import data_manager
from webapp.parser.utils.shared_logic import safe_get, safe_split, safe_lower
from webapp.parser.web_pipeline import (
    process_urls_for_web, cancel_processing, safe_sid, safe_rsplit, cancellation_manager
)
from webapp.parser.config import BASE_DIR, PROJECT_ROOT, URL_LIST_FILE
from webapp.parser.utils.logger_singleton import logger, console, prompt

# 2. Flask App & SocketIO Initialization
app = Flask(__name__)
socketio = SocketIO(app)

# 3. Session & State Management
session_prompt_queues = {}
session_threads = {}
active_sessions_backend = set()
session_last_active = {}
session_metadata = {}
session_logs = {}
SESSION_TIMEOUT = 3600

# 4. Utility Functions
def is_owner(sid, username):
    return session_metadata.get(sid, {}).get('username') == username

def create_session_metadata(sid, username=None):
    session_metadata[sid] = {
        "session_id": sid,
        "username": username or "anonymous",
        "created": datetime.now(timezone.utc).isoformat(),
        "last_active": time.time(),
        "parser_status": "idle",
        "locked": False,
    }

def cleanup_sessions():
    now = time.time()
    expired = []
    for sid, last_active in list(session_last_active.items()):
        if now - last_active > SESSION_TIMEOUT:
            expired.append(sid)
            active_sessions_backend.discard(sid)
            del session_last_active[sid]
            session_metadata.pop(sid, None)
    if expired:
        emit('session_expired', {'expired_sessions': expired}, broadcast=True)

def get_prompt_queue(session_id):
    if session_id not in session_prompt_queues:
        session_prompt_queues[session_id] = Queue()
    return session_prompt_queues[session_id]

def broadcast_sessions():
    sessions = [session_metadata[sid] for sid in active_sessions_backend if sid in session_metadata]
    emit('session_list', {'sessions': sessions}, broadcast=True)

def lock_session(sid):
    session_metadata[sid]['locked'] = True
    session_metadata[sid]['parser_status'] = 'running'
    broadcast_sessions()

def unlock_session(sid):
    session_metadata[sid]['locked'] = False
    session_metadata[sid]['parser_status'] = 'idle'
    broadcast_sessions()

# File & Folder Configurations
ALLOWED_EXTENSIONS = {"csv", "json", "pdf", "txt"}
INPUT_FOLDER = os.path.join(PROJECT_ROOT, "input")
OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "output")
PARSER_DIR = os.path.join(BASE_DIR, "parser")
HINT_FILE = os.path.join(PARSER_DIR, "url_hint_overrides.txt")
HISTORY_FILE = os.path.join(PARSER_DIR, "url_hint_history.jsonl")
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "uploads")
URLS_FILE = os.path.join(PARSER_DIR, "urls.txt")
URL_LIST = []

os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.secret_key = os.environ.get("FLASK_SECRET_KEY")
if not app.secret_key:
    raise RuntimeError("FLASK_SECRET_KEY not set in environment variables!")

app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SECURE"] = os.environ.get("FLASK_COOKIE_SECURE", "False").lower() == "true"

# --- Add security and cache headers ---
@app.after_request
def add_headers(response):
    response.headers['Cache-Control'] = 'no-store'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response

# --- Data Management Utilities ---
def add_url() -> None:
    url = input("Enter new URL to add: ").strip()
    if url:
        with open(URLS_FILE, "a", encoding="utf-8") as f:
            f.write(url + "\n")
        log_parser_status(f"[ADDED] {url}")

def allowed_file(filename) -> bool:
    parts = safe_rsplit(filename, '.', 1)
    ext = safe_lower(parts[1]) if len(parts) > 1 else ''
    return filename and ext in ALLOWED_EXTENSIONS and len(filename) < 128

def append_history(data) -> None:
    # Only write if data is not empty
    if not data:
        return
    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": data
    }
    # Write compact JSON (no indent) for JSONL
    with open(HISTORY_FILE, "ab") as f:
        f.write(orjson.dumps(snapshot) + b"\n")

def edit_hint() -> str:
    frag = request.form.get("fragment", "").strip()
    path = request.form.get("module_path", "").strip()
    overrides = load_overrides()
    if frag in overrides and path:
        overrides[frag] = path
        append_history(overrides)
        save_overrides(overrides)
        flash("Hint updated.", "success")
    else:
        flash("Invalid fragment or path.", "danger")
    return redirect(url_for("url_hints"))

def get_url_list() -> list[str]:
    if not os.path.exists(URLS_FILE):
        return []
    with open(URLS_FILE, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    return urls

def list_urls() -> list[str]:
    if not os.path.exists(URLS_FILE):
        log_parser_status("[INFO] No urls.txt found.")
        return []
    with open(URLS_FILE, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    log_parser_status("\n[URLS.TXT ENTRIES]")
    for i, url in enumerate(urls, 1):
        log_parser_status(f"{i}. {url}")
    return urls

def load_overrides() -> dict:
    if os.path.exists(HINT_FILE):
        with open(HINT_FILE, "rb") as f:
            return orjson.loads(f.read())
    return {}

def save_overrides(data) -> None:
    # Always write a valid JSON object (pretty for hints file is fine)
    with open(HINT_FILE, "wb") as f:
        f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))

def validate_module_path(path) -> tuple[bool, str | None]:
    try:
        importlib.import_module(path)
        return True, None
    except ModuleNotFoundError:
        parts = safe_split(os.path, ".")
        base = parts[-1] if parts else ""
        parent = ".".join(parts[:-1]) if len(parts) > 1 else ""
        try:
            pkg = importlib.import_module(parent)
            suggestion = get_close_matches(base, dir(pkg), n=1, cutoff=0.6)
            if suggestion:
                return False, f"Suggested: {parent}.{suggestion[0]}"
        except Exception:
            pass
        return False, "Module not found"

def log_parser_status(msg, session_id=None, rich=False) -> None:
    if logger.mode == "webapp":
        status_msg = f"{msg} (session_id={session_id})" if session_id else msg
        logger.info({
            "level": "INFO",
            "message": status_msg,
            "session_id": session_id
        })
    elif rich:
        console.panel(f"{msg}\nSession: {session_id}", title="Parser Status")
    else:
        log_msg = f"{msg} (session_id={session_id})" if session_id else msg
        logger.info(log_msg)

# --- Utility: Validate all override module paths ---
def get_all_override_validations() -> dict:
    """
    Returns a dict mapping each override fragment to (is_valid, suggestion/message)
    Example: { "electionreturns.pa.gov": (True, None), "badsite.com": (False, "Module not found") }
    """
    overrides = load_overrides()
    return {k: validate_module_path(v) for k, v in overrides.items()}

# --- Utility: List all files in input, output, and uploads folders ---
def get_all_file_lists() -> dict:
    """
    Returns a dict with lists of files in each managed folder.
    Example: { "input_files": [...], "output_files": [...], "uploaded_files": [...] }
    """
    return {
        "input_files": os.listdir(INPUT_FOLDER),
        "output_files": os.listdir(OUTPUT_FOLDER),
        "uploaded_files": os.listdir(UPLOAD_FOLDER),
    }

# 5. Routes (Flask)
@app.route("/")
def index() -> str:
    return render_template("index.html")

@app.route("/api/url_hint_overrides", methods=["GET", "POST", "DELETE"])
def api_url_hint_overrides():
    if request.method == "GET":
        overrides = data_manager.load_overrides()
        return jsonify({"overrides": overrides})
    elif request.method == "POST":
        data = request.get_json()
        frag = data.get("fragment", "").strip()
        path = data.get("module_path", "").strip()
        if not frag or not path:
            return jsonify({"success": False, "error": "Both fields required."}), 400
        overrides = data_manager.load_overrides()
        overrides[frag] = path
        data_manager.save_overrides(overrides)
        return jsonify({"success": True})
    elif request.method == "DELETE":
        data = request.get_json()
        frag = data.get("fragment", "").strip()
        overrides = data_manager.load_overrides()
        if frag in overrides:
            del overrides[frag]
            data_manager.save_overrides(overrides)
            return jsonify({"success": True})
        return jsonify({"success": False, "error": "Not found."}), 404

@app.route("/api/urls", methods=["GET", "POST"])
def api_urls():
    urls_file = str(URL_LIST_FILE)
    if request.method == "GET":
        if not os.path.exists(urls_file):
            return jsonify({"urls": []})
        with open(urls_file, "r", encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
        return jsonify({"urls": urls})
    elif request.method == "POST":
        data = request.get_json()
        url = data.get("url", "").strip()
        if not url:
            return jsonify({"success": False, "error": "URL required."}), 400
        with open(urls_file, "a", encoding="utf-8") as f:
            f.write(url + "\n")
        return jsonify({"success": True})

@app.route("/delete-hint/<frag>", methods=["POST"])
def delete_hint_route(frag) -> None:
    overrides = load_overrides()
    if frag in overrides:
        overrides.pop(frag)
        append_history(overrides)
        save_overrides(overrides)
        flash("Hint deleted.", "info")
    else:
        flash("Hint not found.", "warning")
    return redirect(url_for("url_hints"))

@app.route("/edit-hint", methods=["POST"])
def edit_hint_route() -> None:
    frag = request.form.get("fragment", "").strip()
    path = request.form.get("module_path", "").strip()
    overrides = load_overrides()
    if frag and path and frag in overrides:
        overrides[frag] = path
        append_history(overrides)
        save_overrides(overrides)
        flash("Hint updated.", "success")
    else:
        flash("Invalid fragment or path.", "danger")
    return redirect(url_for("url_hints"))

@app.route("/data_framework", methods=["GET", "POST"])
def data_framework() -> str:
    return render_template("data_framework.html")

@app.route("/delete/input/<filename>", methods=["POST"])
def delete_input_file(filename) -> str:
    file_path = os.path.join(INPUT_FOLDER, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        flash(f"Deleted '{filename}' from input folder.", "success")
    else:
        flash(f"File '{filename}' not found in input folder.", "danger")
    return redirect(request.referrer or url_for("run_parser"))

@app.route("/delete/output/<filename>", methods=["POST"])
def delete_output_file(filename) -> str:
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        flash(f"Deleted '{filename}' from output folder.", "success")
    else:
        flash(f"File '{filename}' not found in output folder.", "danger")
    return redirect(request.referrer or url_for("run_parser"))

@app.route("/delete/uploads/<filename>", methods=["POST"])
def delete_upload_file(filename) -> str:
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        flash(f"Deleted '{filename}' from uploads folder.", "success")
    else:
        flash(f"File '{filename}' not found in uploads folder.", "danger")
    return redirect(request.referrer or url_for("run_parser"))

@app.route("/download/input/<filename>")
def download_input_file(filename) -> str:
    return send_from_directory(INPUT_FOLDER, filename, as_attachment=True)

@app.route("/download/output/<filename>")
def download_output_file(filename) -> str:
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)

@app.route("/download/uploads/<filename>")
def download_upload_file(filename) -> str:
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)

@app.route("/export-hints")
def export_hints() -> str:
    overrides = load_overrides()
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["URL Fragment", "Module Path"])
    for k, v in overrides.items():
        writer.writerow([k, v])
    output.seek(0)
    return send_file(
        StringIO(output.read()),
        mimetype='text/csv',
        as_attachment=True,
        download_name="url_hint_overrides.csv"
    )

@app.route("/history")
def history() -> str:
    snapshots = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "rb") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue  # Skip empty lines
                try:
                    snap = orjson.loads(line)
                    timestamp = safe_get(snap, "timestamp")
                    data = safe_get(snap, "data", snap)
                    snapshots.append({"timestamp": timestamp, "data": data})
                except Exception:
                    continue  # Skip invalid JSON lines
    indexed_snapshots = list(enumerate(snapshots))
    return render_template("history.html", snapshots=indexed_snapshots)

@app.route("/rollback/<int:index>", methods=["POST"])
def rollback(index) -> str:
    if not os.path.exists(HISTORY_FILE):
        flash("No history file found.", "danger")
        return redirect(url_for("history"))
    with open(HISTORY_FILE, "rb") as f:
        lines = [line for line in f if line.strip()]
    if index < 0 or index >= len(lines):
        flash("Invalid snapshot index.", "danger")
        return redirect(url_for("history"))
    selected = orjson.loads(lines[index])
    with open(HISTORY_FILE, "wb") as f:
        f.writelines(lines[:index+1])
    save_overrides(selected)
    flash("Snapshot restored successfully.", "success")
    return redirect(url_for("history", restored=1))

@app.route("/import-hints", methods=["POST"])
def import_hints() -> str:
    file = request.files.get("csv_file")
    if not file:
        flash("No file uploaded.", "danger")
        return redirect(url_for("url_hints"))
    overrides = load_overrides()
    content = file.stream.read().decode("utf-8")
    reader = csv.reader(StringIO(content))
    next(reader, None)
    for row in reader:
        if len(row) == 2:
            frag, path = row[0].strip(), row[1].strip()
            overrides[frag] = path
    append_history(overrides)
    save_overrides(overrides)
    flash("Hints imported.", "success")
    return redirect(url_for("url_hints"))

@app.route("/run_parser", methods=["GET", "POST"])
def run_parser():
    try:
        if request.method == "POST" and "data_file" in request.files:
            file = request.files.get("data_file")
            if file and allowed_file(file.filename):
                filename = file.filename
                file.save(os.path.join(UPLOAD_FOLDER, filename))
                flash(f"File '{filename}' uploaded successfully.", "success")
            else:
                flash("Invalid file type or no file selected.", "danger")

        file_lists = get_all_file_lists()
        validations = get_all_override_validations()
        overrides = load_overrides()
        return render_template(
            "run_parser.html",
            input_files=file_lists["input_files"],
            output_files=file_lists["output_files"],
            uploaded_files=file_lists["uploaded_files"],
            validations=validations,
            overrides=overrides,
        )
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return "Internal Server Error", 500
    
@app.route("/undo-hints", methods=["POST"])
def undo_hints() -> str:
    if not os.path.exists(HISTORY_FILE):
        flash("No history to undo.", "warning")
        return redirect(url_for("url_hints"))
    with open(HISTORY_FILE, "rb") as f:
        lines = [line for line in f if line.strip()]
    if len(lines) < 2:
        flash("Nothing to undo.", "warning")
        return redirect(url_for("url_hints"))
    with open(HISTORY_FILE, "wb") as f:
        f.writelines(lines[:-1])
    last_good = orjson.loads(lines[-2])
    save_overrides(last_good)
    flash("Undo successful.", "success")
    return redirect(url_for("url_hints"))

@app.route("/upload/input", methods=["POST"])
def upload_to_input() -> str:
    file = request.files.get("file")
    log_parser_status(f"Upload to input: {file.filename if file else 'No file'}")
    if file and allowed_file(file.filename):
        filename = file.filename
        file.save(os.path.join(INPUT_FOLDER, filename))
        flash(f"File '{filename}' uploaded to input folder.", "success")
    else:
        flash("Invalid file type or no file selected.", "danger")
    return redirect(request.referrer or url_for("run_parser"))

@app.route("/upload/output", methods=["POST"])
def upload_to_output() -> str:
    file = request.files.get("file")
    if file and allowed_file(file.filename):
        filename = file.filename
        file.save(os.path.join(OUTPUT_FOLDER, filename))
        flash(f"File '{filename}' uploaded to output folder.", "success")
    else:
        flash("Invalid file type or no file selected.", "danger")
    return redirect(request.referrer or url_for("run_parser"))

@app.route("/upload/uploads", methods=["POST"])
def upload_to_uploads() -> str:
    file = request.files.get("file")
    if file and allowed_file(file.filename):
        filename = file.filename
        file.save(os.path.join(UPLOAD_FOLDER, filename))
        flash(f"File '{filename}' uploaded to uploads folder.", "success")
    else:
        flash("Invalid file type or no file selected.", "danger")
    return redirect(request.referrer or url_for("run_parser"))

@app.route("/health")
def health() -> str:
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}

# 6. SocketIO Event Handlers

@socketio.on('get_session_history')
def handle_get_session_history(data) -> None:
    sid = str(data['session_id'])
    if not isinstance(sid, str):
        # Only log backend/server errors
        logger.warning(f"Invalid session_id type: {type(sid)} value: {sid}")
        return
    logs = session_logs.get(sid, [])
    if not isinstance(logs, list):
        logs = []
    emit('session_history', {'session_id': sid, 'logs': logs}, room=request.sid)

@socketio.on('clone_session')
def handle_clone_session(data) -> None:
    old_sid = str(data['session_id'])
    if not isinstance(old_sid, str):
        logger.warning(f"Invalid session_id type: {type(old_sid)} value: {old_sid}")
        return
    new_sid = 'sess_' + os.urandom(6).hex()
    session_metadata[new_sid] = dict(session_metadata[old_sid])
    session_metadata[new_sid]['session_id'] = new_sid
    session_metadata[new_sid]['created'] = datetime.now(timezone.utc).isoformat()
    session_metadata[new_sid]['last_active'] = time.time()
    session_logs[new_sid] = list(session_logs.get(old_sid, []))
    active_sessions_backend.add(new_sid)
    session_last_active[new_sid] = time.time()
    broadcast_sessions()
    emit('session_cloned', {'old_session': old_sid, 'new_session': new_sid}, room=request.sid)

@socketio.on('delete_session')
def handle_delete_session(data) -> None:
    sid = str(data['session_id'])
    if not isinstance(sid, str):
        logger.warning(f"Invalid session_id type: {type(sid)} value: {sid}")
        return
    # Remove session unconditionally (no username check)
    active_sessions_backend.discard(sid)
    session_last_active.pop(sid, None)
    session_metadata.pop(sid, None)
    session_prompt_queues.pop(sid, None)
    session_threads.pop(sid, None)
    session_logs.pop(sid, None)
    emit('session_deleted', {'session_id': sid}, broadcast=True)

@socketio.on('join')
def on_join(data):
    sid = str(data['session_id'])
    if not isinstance(sid, str):
        logger.warning(f"Invalid session_id type: {type(sid)} value: {sid}")
        return
    join_room(sid)
    active_sessions_backend.add(sid)
    session_last_active[sid] = time.time()
    if sid not in session_metadata:
        create_session_metadata(sid, data.get('username'))

@socketio.on('get_sessions')
def handle_get_sessions():
    cleanup_sessions()  # Clean up expired sessions before listing
    sessions = [session_metadata[sid] for sid in active_sessions_backend if sid in session_metadata]
    emit('session_list', {'sessions': sessions}, broadcast=True)

@socketio.on('connect')
def handle_connect():
    cleanup_sessions()
    # Do NOT set logger mode or emit_func here!
    session['log_format'] = "json"
    prev_session_id = request.args.get('prev_session_id')
    if prev_session_id and prev_session_id in session_metadata:
        session_last_active[prev_session_id] = time.time()
    if prev_session_id:
        cancellation_manager.remove(prev_session_id)
        prompt.clear_prompt_session(prev_session_id)
        # Only log backend/server events
        log_parser_status(f"Cleaned up previous session {prev_session_id}", prev_session_id)
    if prev_session_id:
        emit('session_id', {'session_id': prev_session_id})

@socketio.on('disconnect')
def handle_disconnect(sid) -> None:
    cancel_processing(sid)
    log_parser_status(f"Client disconnected (sid={sid})", sid)
    emit('parser_output', {
        "level": "INFO",
        "message": "🚪 Disconnected from server.",
        "color": "#eb4f43"
    }, room=sid)
    prompt.clear_prompt_session(sid)
    cancellation_manager.remove(sid)
    active_sessions_backend.discard(sid)
    session_last_active.pop(sid, None)

@socketio.on('set_output_mode')
def handle_set_output_mode(data) -> None:
    mode = safe_lower(safe_get(data, "mode", "live"))
    valid_modes = {"live", "batch"}
    session_id = safe_sid()
    if mode in valid_modes:
        session['output_mode'] = mode
        emit('parser_output', f'{{"level":"INFO","message":"Output mode set to {mode}.","color":"#00ffe7"}}', room=session_id)
    else:
        emit('parser_output', '{"level":"ERROR","message":"Invalid output mode.","color":"#eb4f43"}', room=session_id)

@socketio.on('parser_prompt')
def handle_parser_prompt(data) -> None:
    # Only act as a mediator: deliver the prompt value to the waiting prompt session
    session_id = None
    value = data
    if isinstance(data, dict):
        value = data.get("value", "")
        session_id = data.get("session_id")
    if not session_id or session_id not in session_metadata:
        # Optionally emit an error to the frontend
        emit('parser_output', {
            "level": "ERROR",
            "message": "Invalid or unknown session_id for prompt.",
            "color": "#eb4f43"
        }, room=session_id)
        return
    # Deliver the value to the waiting prompt session (handled in user_prompt.py)
    prompt_session = prompt.prompt_sessions.get(session_id)
    if prompt_session:
        prompt_session.set_response(value)
    # No logging, validation, or business logic here; handled downstream

@socketio.on('cancel_parser')
def handle_cancel_parser() -> None:
    session_id = safe_sid()
    cancel_processing(session_id)
    if session_id in session_threads:
        del session_threads[session_id]
    if session_id in session_prompt_queues:
        del session_prompt_queues[session_id]
    cleanup_sessions()  # Clean up after cancel

@socketio.on('run_parser')
def handle_run_parser() -> None:
    cleanup_sessions()  # Clean up expired sessions before running
    session_id = safe_sid()
    if session_id not in session_metadata:
        create_session_metadata(session_id)
    if session_metadata[session_id]['locked']:
        emit('parser_output', {
            "level": "ERROR",
            "message": "Session is locked. Wait for current job to finish.",
            "color": "#eb4f43"
        }, room=session_id)
        return
    if session_id in session_threads and session_threads[session_id].is_alive():
        emit('parser_output', {
            "level": "WARNING",
            "message": "Parser already running for this session.",
            "color": "#ffd166"
        }, room=session_id)
        return
    lock_session(session_id)
    log_parser_status("Parser connected. Starting parser run...", session_id, rich=True)
    cancel_flag = cancellation_manager.get_flag(session_id)
    prompt_queue = get_prompt_queue(session_id)

    # --- Robust, session-aware emit function ---
    def emit_to_socketio(line):
        sid = session_id
        # Store logs for session history display only
        if sid:
            if sid not in session_logs:
                session_logs[sid] = []
            # Store as dict or string, do not interpret or format
            if isinstance(line, str) and line.strip().startswith("{"):
                try:
                    obj = orjson.loads(line)
                    session_logs[sid].append(obj)
                except Exception:
                    session_logs[sid].append(line)
            else:
                session_logs[sid].append(line)
        # Forward to SocketIO, do not reformat or filter
        try:
            if isinstance(line, dict) and line.get("type") == "heartbeat":
                socketio.emit('session_heartbeat', line, room=sid)
            elif isinstance(line, str) and line.strip().startswith("{"):
                try:
                    obj = orjson.loads(line)
                    socketio.emit('parser_output', obj, room=sid)
                    return
                except Exception:
                    pass
            if isinstance(line, dict):
                socketio.emit('parser_output', line, room=sid)
            else:
                socketio.emit('parser_output', {
                    "level": "INFO",
                    "message": str(line),
                    "session_id": sid,
                    "source": "backend"
                }, room=sid)
        except Exception:
            pass  # Only log backend errors if needed

    # Set logger and prompt to webapp mode for this session
    logger.set_mode("webapp")
    logger.set_format("json")
    logger.set_socketio_emit_func(emit_to_socketio)
    prompt.set_mode("webapp")
    prompt.set_socketio_emit_func(
        lambda msg: socketio.emit(
            'parser_output',
            msg if isinstance(msg, dict) else {"level": "PROMPT", "message": str(msg), "session_id": session_id, "source": "prompt"},
            room=session_id
        )
    )

    thread = Thread(
        target=process_urls_for_web,
        args=(prompt_queue, session_id, cancel_flag),
        kwargs={"emit_func": emit_to_socketio}
    )
    thread.daemon = True
    thread.start()
    session_threads[session_id] = thread
    unlock_session(session_id)

# 7. Main Entrypoint
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host="0.0.0.0", port=port)