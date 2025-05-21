# webapp/url_hint_webapp.py
# ==============================================================
# Browser-based Flask UI for managing URL_HINT_OVERRIDES.txt.
# Allows adding, editing, deleting, and validating override entries.
# ==============================================================

from flask import Flask, request, render_template, redirect, url_for, jsonify
import json
import importlib
from difflib import get_close_matches
import os

app = Flask(__name__)
HISTORY_FILE = "url_hint_history.jsonl"
HINT_FILE = "url_hint_overrides.txt"


def load_overrides():
    if os.path.exists(HINT_FILE):
        with open(HINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def append_history(data):
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(data) + "")  # fixed line ending


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


@app.route("/")
def index():
    # Handles pagination and per-page limits via query params
    overrides = load_overrides()
    validations = {
        k: validate_module_path(v) for k, v in overrides.items()
    }
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 10))
    items = list(overrides.items())
    total_pages = (len(items) + per_page - 1) // per_page
    start = (page - 1) * per_page
    end = start + per_page
    paginated = dict(items[start:end])

    return render_template("index.html",
                           overrides=paginated,
                           validations=validations,
                           page=page,
                           per_page=per_page,
                           total_pages=total_pages)


@app.route("/import", methods=["POST"])
def import_csv():
    from io import StringIO
    import csv
    file = request.files.get("csv_file")
    if not file:
        return redirect(url_for("history", restored=1))

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
    return redirect(url_for("index"))


@app.route("/edit", methods=["POST"])
def edit():
    frag = request.form.get("fragment", "").strip()
    path = request.form.get("module_path", "").strip()
    overrides = load_overrides()
    if frag in overrides and path:
        overrides[frag] = path
        save_overrides(overrides)
    return redirect(url_for("index"))


@app.route("/add", methods=["POST"])
def add():
    frag = request.form.get("fragment", "").strip()
    path = request.form.get("module_path", "").strip()
    overrides = load_overrides()
    if frag and path:
        overrides[frag] = path
        save_overrides(overrides)
    return redirect(url_for("index"))


@app.route("/export")
def export():
    import csv
    overrides = load_overrides()
    csv_data = "URL Fragment,Module Path" + "".join(f"{k},{v}" for k, v in overrides.items())
    return app.response_class(
        csv_data,
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=url_hint_overrides.csv'}
    )


@app.route("/undo", methods=["POST"])
def undo():
    if not os.path.exists(HISTORY_FILE):
        return redirect(url_for("index"))
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if len(lines) < 2:
        return redirect(url_for("index"))
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        f.writelines(lines[:-1])
    last_good = json.loads(lines[-2])
    save_overrides(last_good)
    return redirect(url_for("index"))


@app.route("/history")
def history():
    if not os.path.exists(HISTORY_FILE):
        return render_template("history.html", snapshots=[])
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        snapshots = [json.loads(line) for line in f]
    return render_template("history.html", snapshots=enumerate(snapshots))


@app.route("/delete/<frag>", methods=["POST"])
def delete(frag):
    overrides = load_overrides()
    overrides.pop(frag, None)
    save_overrides(overrides)
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True, port=5050)
