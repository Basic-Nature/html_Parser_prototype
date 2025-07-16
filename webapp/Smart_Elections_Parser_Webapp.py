import atexit
from webapp.parser.postgres_service_control import start_postgres_service, stop_postgres_service

_service_started = start_postgres_service()

def stop_if_started():
    if _service_started:
        stop_postgres_service()

atexit.register(stop_if_started)

import csv
from datetime import datetime, timezone
from difflib import get_close_matches
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, session, url_for, flash, send_file, send_from_directory
from flask_socketio import emit, SocketIO
import importlib
from io import StringIO
import orjson
import os
import subprocess
from threading import Thread
from webapp.parser.utils.shared_logger import SharedLogger, RichConsoleProxy
from webapp.parser.web_pipeline import process_urls_for_web, cancel_processing
from webapp.parser.config import BASE_DIR, POSTGRES_URL, PROJECT_ROOT, POSTGRES_SERVICE_NAME 
from webapp.parser.utils.user_prompt import UserPrompt
# Load environment variables from .env

load_dotenv()

prompt = UserPrompt()
logger = SharedLogger()
console = RichConsoleProxy(logger)
app = Flask(__name__)
socketio = SocketIO(app)


ALLOWED_EXTENSIONS = {"csv", "json", "pdf", "txt"}
INPUT_FOLDER = os.path.join(PROJECT_ROOT, "input")
OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "output")
# Data folders
PARSER_DIR = os.path.join(BASE_DIR, "parser")
HINT_FILE = os.path.join(PARSER_DIR, "url_hint_overrides.txt")
HISTORY_FILE = os.path.join(PARSER_DIR, "url_hint_history.jsonl")
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "uploads")
URLS_FILE = os.path.join(PARSER_DIR, "urls.txt")

# Store URLs in memory for the session (for demo; in production, use session or DB)
URL_LIST = []

# Ensure input/output folders exist
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Secure secret key from environment
app.secret_key = os.environ.get("FLASK_SECRET_KEY")
if not app.secret_key:
    raise RuntimeError("FLASK_SECRET_KEY not set in environment variables!")

# Optional: Set secure cookie flags for production
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SECURE"] = os.environ.get("FLASK_COOKIE_SECURE", "False").lower() == "true"

# SocketIO event for real-time updates

# --- Utility functions for Data management ---
def add_url():
    url = input("Enter new URL to add: ").strip()
    if url:
        with open(URLS_FILE, "a", encoding="utf-8") as f:
            f.write(url + "\n")
        logger.info(f"[ADDED] {url}")
        
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def append_history(data):
    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": data
    }
    with open(HISTORY_FILE, "ab") as f:
        f.write(orjson.dumps(snapshot, option=orjson.OPT_INDENT_2) + b"\n")

def edit_hint():
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

def get_url_list():
    # Load URLs from file
    if not os.path.exists(URLS_FILE):
        return []
    with open(URLS_FILE, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    return urls

def list_urls():
    if not os.path.exists(URLS_FILE):
        logger.info("[INFO] No urls.txt found.")
        return []
    with open(URLS_FILE, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    logger.info("\n[URLS.TXT ENTRIES]")
    for i, url in enumerate(urls, 1):
        logger.info(f"{i}. {url}")
    return urls

def load_overrides():
    if os.path.exists(HINT_FILE):
        with open(HINT_FILE, "rb") as f:
            return orjson.loads(f.read())
    return {}

def postgres_service_status(service_name=None):
    if service_name is None:
        service_name = POSTGRES_SERVICE_NAME
    try:
        result = subprocess.run(["sc", "query", service_name], capture_output=True, text=True)
        if "RUNNING" in result.stdout:
            return "running"
        elif "STOPPED" in result.stdout:
            return "stopped"
        else:
            return "unknown"
    except Exception as e:
        logger.error(f"[ERROR] Could not check service status: {e}")
        return "error"

def save_overrides(data):
    with open(HINT_FILE, "wb") as f:
        f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))

def validate_module_path(path):
    try:
        importlib.import_module(path)
        return True, None
    except ModuleNotFoundError:
        base = path.split(".")[-1]
        parent = ".".join(path.split(".")[:-1])
        try:
            pkg = importlib.import_module(parent)
            suggestion = get_close_matches(base, dir(pkg), n=1, cutoff=0.6)
            if suggestion:
                return False, f"Suggested: {parent}.{suggestion[0]}"
        except Exception:
            pass
        return False, "Module not found"
    
def log_parser_status(msg, session_id=None, rich=False):
    """
    Log parser status to both logger and console as appropriate.
    - msg: The message to log.
    - session_id: The session ID (optional).
    - rich: If True, use rich panel for CLI; else, plain or json log depending on logger.format.
    """
    # For webapp, always output a simple status message (no box drawing)
    if logger.mode == "webapp":
        # In webapp, always send as JSON for frontend rendering
        status_msg = f"{msg} (session_id={session_id})" if session_id else msg
        logger.info({"level": "INFO", "message": status_msg})
    elif rich:
        # In CLI, use rich panel for status
        console.panel(f"{msg}\nSession: {session_id}", title="Parser Status")
    else:
        # Fallback: plain log
        log_msg = f"{msg} (session_id={session_id})" if session_id else msg
        logger.info(log_msg)
            
# --- Routes ---
@app.route("/")
def index():
    return render_template("index.html")

@socketio.on('connect')
def handle_connect():
    # Set logging mode to webapp for this session
    logger.set_mode("webapp")
    # Set logger format to JSON for this session
    logger.set_format("json")
    session['log_format'] = "json"

    # Get the session ID for this client
    session_id = session.get('sid') if 'sid' in session else request.sid

    # Set the global logger's emit function to route logs to this client's SocketIO room
    def emit_to_socketio(line):
        socketio.emit('parser_output', line, room=session_id)
    logger.set_socketio_emit_func(emit_to_socketio)

    # Set the prompt system to webapp mode and route prompts to this client's SocketIO room
    prompt.set_mode("webapp")
    prompt.set_socketio_emit_func(lambda msg: socketio.emit('parser_output', msg, room=session_id))

    # Log connection event
    logger.info("Client connected")

@app.route("/delete-hint/<frag>", methods=["POST"])
def delete_hint_route(frag):
    overrides = load_overrides()
    if frag in overrides:
        overrides.pop(frag)
        append_history(overrides)
        save_overrides(overrides)
        flash("Hint deleted.", "info")
    else:
        flash("Hint not found.", "warning")
    return redirect(url_for("url_hints"))

@socketio.on('disconnect')
def handle_disconnect(sid):
    # Use the sid provided by Socket.IO
    cancel_processing(sid)
    logger.info(f"Client disconnected (sid={sid})")
    # Optionally, emit a styled disconnect message
    emit('parser_output', '{"level":"INFO","message":"🚪 Disconnected from server.","color":"#eb4f43"}', room=sid)
    
@app.route("/edit-hint", methods=["POST"])
def edit_hint_route():
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
def data_framework():
    # add logic here currently returning a placeholder
    return render_template("data_framework.html")

@app.route("/delete/input/<filename>", methods=["POST"])
def delete_input_file(filename):
    file_path = os.path.join(INPUT_FOLDER, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        flash(f"Deleted '{filename}' from input folder.", "success")
    else:
        flash(f"File '{filename}' not found in input folder.", "danger")
    return redirect(request.referrer or url_for("manage_data"))

@app.route("/delete/output/<filename>", methods=["POST"])
def delete_output_file(filename):
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        flash(f"Deleted '{filename}' from output folder.", "success")
    else:
        flash(f"File '{filename}' not found in output folder.", "danger")
    return redirect(request.referrer or url_for("manage_data"))

@app.route("/delete/uploads/<filename>", methods=["POST"])
def delete_upload_file(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        flash(f"Deleted '{filename}' from uploads folder.", "success")
    else:
        flash(f"File '{filename}' not found in uploads folder.", "danger")
    return redirect(request.referrer or url_for("manage_data"))

@app.route("/download/input/<filename>")
def download_input_file(filename):
    return send_from_directory(INPUT_FOLDER, filename, as_attachment=True)

@app.route("/download/output/<filename>")
def download_output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)    

@app.route("/export-hints")
def export_hints():
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
def history():
    # Read all snapshots from the history file
    snapshots = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "rb") as f:
            for line in f:
                try:
                    snap = orjson.loads(line)
                    timestamp = snap.get("timestamp")
                    data = snap.get("data", snap)  # fallback for old entries
                    snapshots.append({"timestamp": timestamp, "data": data})
                    snapshots.append(snap)
                except Exception:
                    continue
    # Pass index and snapshot for the accordion
    indexed_snapshots = list(enumerate(snapshots))
    return render_template("history.html", snapshots=indexed_snapshots)

@app.route("/rollback/<int:index>", methods=["POST"])
def rollback(index):
    # Read all snapshots
    if not os.path.exists(HISTORY_FILE):
        flash("No history file found.", "danger")
        return redirect(url_for("history"))
    with open(HISTORY_FILE, "rb") as f:
        lines = [line for line in f if line.strip()]
    if index < 0 or index >= len(lines):
        flash("Invalid snapshot index.", "danger")
        return redirect(url_for("history"))
    # Restore the selected snapshot
    selected = orjson.loads(lines[index])
    # Truncate history to this point
    with open(HISTORY_FILE, "wb") as f:
        f.writelines(lines[:index+1])
    # Save as current overrides
    save_overrides(selected)
    flash("Snapshot restored successfully.", "success")
    # Add ?restored=1 for toast
    return redirect(url_for("history", restored=1))

@app.route("/import-hints", methods=["POST"])
def import_hints():
    file = request.files.get("csv_file")
    if not file:
        flash("No file uploaded.", "danger")
        return redirect(url_for("url_hints"))
    overrides = load_overrides()
    content = file.stream.read().decode("utf-8")
    reader = csv.reader(StringIO(content))
    next(reader, None)  # Skip header
    for row in reader:
        if len(row) == 2:
            frag, path = row[0].strip(), row[1].strip()
            overrides[frag] = path
    append_history(overrides)
    save_overrides(overrides)
    flash("Hints imported.", "success")
    return redirect(url_for("url_hints"))

@app.route("/input-files")
def input_files():
    files = os.listdir(INPUT_FOLDER)
    return render_template("file_list.html", files=files, folder="Input", download_url="download_input_file")

@app.route("/manage-data", methods=["GET", "POST"])
def manage_data():
    overrides = load_overrides()
    validations = {k: validate_module_path(v) for k, v in overrides.items()}
    uploaded_files = os.listdir(UPLOAD_FOLDER)
    input_files = os.listdir(INPUT_FOLDER)
    output_files = os.listdir(OUTPUT_FOLDER)
    if request.method == "POST":
        # Handle file upload
        file = request.files.get("data_file")
        if file and allowed_file(file.filename):
            filename = file.filename
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            flash(f"File '{filename}' uploaded successfully.", "success")
        else:
            flash("Invalid file type or no file selected.", "danger")
        pass
    return render_template(
        "manage_data.html",
        overrides=overrides,
        validations=validations,
        uploaded_files=uploaded_files,
        input_files=input_files,
        output_files=output_files
    )
        
@socketio.on('set_output_mode')
def handle_set_output_mode(data):
    """
    Allows the frontend to set the output mode at runtime.
    Example payload: {"mode": "live"} or {"mode": "batch"}
    """
    mode = data.get("mode", "live").lower()
    valid_modes = {"live", "batch"}
    session_id = session.get('sid') if 'sid' in session else request.sid
    if mode in valid_modes:
        # Store the mode in the session or a global/session dict as needed
        session['output_mode'] = mode
        emit('parser_output', f'{{"level":"INFO","message":"Output mode set to {mode}.","color":"#00ffe7"}}', room=session_id)
    else:
        emit('parser_output', '{"level":"ERROR","message":"Invalid output mode.","color":"#eb4f43"}', room=session_id)
        
@socketio.on('parser_prompt_response')
def handle_parser_prompt_response(data):
    session_id = session.get('sid') if 'sid' in session else request.sid
    response = data.get('response')
    prompt_session = prompt.get_prompt_session(session_id)
    prompt_session.set_response(response)
    # Optionally clear after use
    prompt.clear_prompt_session(session_id)

@app.route("/output-files")
def output_files():
    files = os.listdir(OUTPUT_FOLDER)
    return render_template("file_list.html", files=files, folder="Output", download_url="download_output_file")

@socketio.on('cancel_parser')
def handle_cancel_parser():
    session_id = session.get('sid') or request.sid
    cancel_processing(session_id)

@socketio.on('parser_prompt')
def handle_parser_prompt(data):
    logger.info(f"Received prompt: {data}")
    session_id = session.get('sid') if 'sid' in session else request.sid
    # Start the parser pipeline in a thread, passing session_id for correct routing
    thread = Thread(target=process_urls_for_web, args=(data, session_id))
    thread.start()

@app.route("/run-parser")
def run_parser_page():
    return render_template("run_parser.html")

@socketio.on('data_framework')
def handle_data_framework(data):
    logger.info(f"Received data_framework event: {data}")
    session_id = session.get('sid') if 'sid' in session else request.sid
    output = postgres_service_status(POSTGRES_SERVICE_NAME)
    emit('parser_output', output, room=session_id)

@socketio.on('run_parser')
def handle_run_parser():
    session_id = session.get('sid') if 'sid' in session else request.sid
    log_parser_status("Starting parser run...", session_id, rich=True)
    # Always pass None for urls to trigger interactive main() pipeline in webapp
    thread = Thread(target=process_urls_for_web, args=(None, session_id))
    thread.start()
    
@app.route("/undo-hints", methods=["POST"])
def undo_hints():
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
def upload_to_input():
    file = request.files.get("file")
    if file and allowed_file(file.filename):
        filename = file.filename
        file.save(os.path.join(INPUT_FOLDER, filename))
        flash(f"File '{filename}' uploaded to input folder.", "success")
    else:
        flash("Invalid file type or no file selected.", "danger")
    return redirect(request.referrer or url_for("manage_data"))

@app.route("/upload/output", methods=["POST"])
def upload_to_output():
    file = request.files.get("file")
    if file and allowed_file(file.filename):
        filename = file.filename
        file.save(os.path.join(OUTPUT_FOLDER, filename))
        flash(f"File '{filename}' uploaded to output folder.", "success")
    else:
        flash("Invalid file type or no file selected.", "danger")
    return redirect(request.referrer or url_for("manage_data"))

@app.route("/upload/uploads", methods=["POST"])
def upload_to_uploads():
    file = request.files.get("file")
    if file and allowed_file(file.filename):
        filename = file.filename
        file.save(os.path.join(UPLOAD_FOLDER, filename))
        flash(f"File '{filename}' uploaded to uploads folder.", "success")
    else:
        flash("Invalid file type or no file selected.", "danger")
    return redirect(request.referrer or url_for("manage_data"))

if __name__ == "__main__":
    socketio.run(app, debug=True) # to stop loop (..., use_reloader=True)
