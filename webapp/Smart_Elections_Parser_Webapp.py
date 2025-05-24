
import csv
from datetime import datetime
from difflib import get_close_matches
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, send_from_directory
from flask_socketio import emit, SocketIO
import importlib
from io import StringIO
import json
import os
import sys
from threading import Thread
from webapp.parser.html_election_parser import main as run_html_parser

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)
socketio = SocketIO(app)

ALLOWED_EXTENSIONS = {"csv", "json", "pdf", "txt"}
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_FOLDER = os.path.join(BASE_DIR, "input")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")
# Data folders
PARSER_DIR = os.path.join(os.path.dirname(__file__), "parser")
HINT_FILE = os.path.join(PARSER_DIR, "url_hint_overrides.txt")
HISTORY_FILE = os.path.join(PARSER_DIR, "url_hint_history.jsonl")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
URLS_FILE = os.path.join(PARSER_DIR, "urls.txt")

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

def run_parser_background():
    # Example: Replace with your real parser logic
    import time
    for i in range(10):
        socketio.emit('parser_output', f"Line {i+1}: Processing...\n")
        time.sleep(1)
    socketio.emit('parser_output', "Parser finished!\n")

def process_user_prompt(data):
    raise NotImplementedError("process_user_prompt function not implemented.")

# --- Utility functions for Data management ---
def add_url():
    url = input("Enter new URL to add: ").strip()
    if url:
        with open(URLS_FILE, "a", encoding="utf-8") as f:
            f.write(url + "\n")
        print(f"[ADDED] {url}")
        
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def append_history(data):
    snapshot = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "data": data
    }
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(snapshot) + "\n")
        
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

def list_urls():
    if not os.path.exists(URLS_FILE):
        print("[INFO] No urls.txt found.")
        return []
    with open(URLS_FILE, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    print("\n[URLS.TXT ENTRIES]")
    for i, url in enumerate(urls, 1):
        print(f"{i}. {url}")
    return urls

def load_overrides():
    if os.path.exists(HINT_FILE):
        with open(HINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_overrides(data):
    with open(HINT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

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

# --- Routes ---
@app.route("/")
def index():
    return render_template("index.html")

@socketio.on('connect')
def handle_connect():
    print("Client connected")
    emit('parser_output', "Connected to server.\n")

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
def handle_disconnect():
    print("Client disconnected")
    emit('parser_output', "Disconnected from server.\n")
@app.route("/edit-hint", methods=["POST"])

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
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    snap = json.loads(line)
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
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if index < 0 or index >= len(lines):
        flash("Invalid snapshot index.", "danger")
        return redirect(url_for("history"))
    # Restore the selected snapshot
    selected = json.loads(lines[index])
    # Truncate history to this point
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
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

@app.route("/output-files")
def output_files():
    files = os.listdir(OUTPUT_FOLDER)
    return render_template("file_list.html", files=files, folder="Output", download_url="download_output_file")

@socketio.on('parser_prompt')
def handle_parser_prompt(data):
    # Process the prompt, send output back
    output = process_user_prompt(data)  # Your function
    emit('parser_output', output)

@app.route("/run-parser", methods=["GET", "POST"])
def run_parser_page():
    # Gather file lists for display
    uploaded_files = os.listdir(UPLOAD_FOLDER)
    input_files = os.listdir(INPUT_FOLDER)
    output_files = os.listdir(OUTPUT_FOLDER)
    parser_output = None

    # Handle file upload to uploads folder
    if request.method == "POST":
        file = request.files.get("data_file")
        if file and allowed_file(file.filename):
            filename = file.filename
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            flash(f"File '{filename}' uploaded successfully to uploads.", "success")
        else:
            flash("Invalid file type or no file selected.", "danger")
        # Optionally, you could trigger the parser here if desired

    # Optionally, you could run the parser and capture output here
    # For example:
    # if request.method == "POST" and 'run_parser' in request.form:
    #     parser_output = run_html_parser()  # Or however your parser returns output

    return render_template(
        "run_parser.html",
        uploaded_files=uploaded_files,
        input_files=input_files,
        output_files=output_files,
        parser_output=parser_output
    )

@app.route("/undo-hints", methods=["POST"])
def undo_hints():
    if not os.path.exists(HISTORY_FILE):
        flash("No history to undo.", "warning")
        return redirect(url_for("url_hints"))
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if len(lines) < 2:
        flash("Nothing to undo.", "warning")
        return redirect(url_for("url_hints"))
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        f.writelines(lines[:-1])
    last_good = json.loads(lines[-2])
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
    socketio.run(app, debug=True)