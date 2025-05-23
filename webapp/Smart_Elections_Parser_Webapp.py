from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from flask_socketio import emit, SocketIO
from threading import Thread
from webapp.parser.html_election_parser import main as run_html_parser
import os
import json
from dotenv import load_dotenv
import sys
from io import StringIO
import csv
import importlib
from difflib import get_close_matches

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)
socketio = SocketIO(app)

# Secure secret key from environment
app.secret_key = os.environ.get("FLASK_SECRET_KEY")
if not app.secret_key:
    raise RuntimeError("FLASK_SECRET_KEY not set in environment variables!")

# Optional: Set secure cookie flags for production
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SECURE"] = os.environ.get("FLASK_COOKIE_SECURE", "False").lower() == "true"

# File paths for hint management
HISTORY_FILE = os.path.join(os.path.dirname(__file__), "url_hint_history.jsonl")
HINT_FILE = os.path.join(os.path.dirname(__file__), "url_hint_overrides.txt")

# SocketIO event for real-time updates

def run_parser_background():
    # Example: Replace with your real parser logic
    import time
    for i in range(10):
        socketio.emit('parser_output', f"Line {i+1}: Processing...\n")
        time.sleep(1)
    socketio.emit('parser_output', "Parser finished!\n")

# --- Utility functions for hint management ---

def load_overrides():
    if os.path.exists(HINT_FILE):
        with open(HINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_overrides(data):
    with open(HINT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def append_history(data):
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")

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
@socketio.on('connect')
def handle_connect():
    print("Client connected")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/history")
def history():
    if not os.path.exists(HISTORY_FILE):
        return render_template("history.html", snapshots=[])
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        snapshots = [json.loads(line) for line in f]
    return render_template("history.html", snapshots=enumerate(snapshots))

@app.route("/url-hints", methods=["GET", "POST"])
def url_hints():
    overrides = load_overrides()
    validations = {k: validate_module_path(v) for k, v in overrides.items()}
    if request.method == "POST":
        frag = request.form.get("fragment", "").strip()
        path = request.form.get("module_path", "").strip()
        if frag and path:
            overrides[frag] = path
            append_history(overrides)
            save_overrides(overrides)
            flash("Hint added or updated!", "success")
        else:
            flash("Both URL fragment and module path are required.", "danger")
        return redirect(url_for("url_hints"))
    return render_template("url_hints.html", overrides=overrides, validations=validations)

@app.route("/run-parser", methods=["GET", "POST"])
def run_parser_page():
    parser_output = None
    if request.method == "POST":
        # Capture output from parser (example)
        import io, sys
        old_stdout = sys.stdout
        sys.stdout = mystdout = io.StringIO()
        try:
            run_html_parser()
        except Exception as e:
            print(f"Error: {e}")
        sys.stdout = old_stdout
        parser_output = mystdout.getvalue()
        flash("Parser run completed.", "success")
    return render_template("run_parser.html", parser_output=parser_output)

@app.route("/edit-hint", methods=["POST"])
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

@app.route("/run-parser", methods=["POST"])
def run_parser():
    # Placeholder: Integrate your parser logic here
    flash("Parser run triggered (not yet implemented).", "info")
    return redirect(url_for("index"))

if __name__ == "__main__":
    socketio.run(app, debug=True)