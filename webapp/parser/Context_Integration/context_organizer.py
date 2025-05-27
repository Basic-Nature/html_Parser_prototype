"""
context_organizer.py

Advanced context organizer for election HTML parsing and data integrity.
Handles data formatting, ML anomaly detection, cache-aware learning, clustering, and robust DB.
Delegates NLP/semantic logic to the context_coordinator and spacy_utils modules.
"""

import re
import os
import json
import logging
import matplotlib.pyplot as plt
import sqlite3
import sys
import threading
import time
from collections import defaultdict
from rich import print as rprint
# if you want to suppress tqdm globally
# sys.stderr = open(os.devnull, 'w')
from sentence_transformers import SentenceTransformer
# sys.stderr = sys.__stderr__

from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import numpy as np

# If you want to suppress the progress bar from SentenceTransformer
# os.environ["TRANSFORMERS_NO_PROGRESS_BAR"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format="\n[%(levelname)s] %(message)s\n"
)

# Paths
CONTEXT_LIBRARY_PATH = os.path.join(
    os.path.dirname(__file__), "Context_Library", "context_library.json"
)
DB_PATH = os.path.join(os.path.dirname(__file__), "context_elections.db")
PROCESSED_URLS_CACHE = os.path.join(
    os.path.dirname(__file__), "..", "..", ".processed_urls"
)
OUTPUT_CACHE = os.path.join(
    os.path.dirname(__file__), "..", "utils", ".output_cache.jsonl"
)
INPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "input")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "output")

# --- DB Schema Setup ---
def ensure_db_schema():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.executescript("""
    CREATE TABLE IF NOT EXISTS contests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        year INTEGER,
        type TEXT,
        state TEXT,
        county TEXT,
        metadata JSON,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        level TEXT,
        msg TEXT,
        context JSON,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    conn.commit()
    conn.close()

ensure_db_schema()



def fetch_contests_by_filter(filters=None, limit=100):
    """
    Fetch contests from the DB with optional filters (dict).
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    query = "SELECT id, title, year, type, state, county, metadata FROM contests"
    params = []
    if filters:
        clauses = []
        for k, v in filters.items():
            clauses.append(f"{k}=?")
            params.append(v)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    cursor.execute(query, params)
    rows = cursor.fetchall()
    contests = []
    for row in rows:
        try:
            meta = json.loads(row[6]) if row[6] else {}
        except Exception:
            meta = {}
        contest = {
            "id": row[0],
            "title": row[1],
            "year": row[2],
            "type": row[3],
            "state": row[4],
            "county": row[5],
            **meta
        }
        contests.append(contest)
    conn.close()
    return contests

def update_contest_in_db(contest):
    """
    Update a contest record in the DB by id.
    Expects contest dict to have at least 'id' and fields to update.
    """
    if "id" not in contest:
        raise ValueError("Contest must have an 'id' field to update.")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE contests
        SET title = ?, year = ?, type = ?, state = ?, county = ?, metadata = ?
        WHERE id = ?
    """, (
        contest.get("title"),
        contest.get("year"),
        contest.get("type"),
        contest.get("state"),
        contest.get("county"),
        json.dumps(contest),
        contest["id"]
    ))
    conn.commit()
    conn.close()

def normalize_label(label):
    return re.sub(r"\s+", " ", str(label).strip().lower())

def sanitize_context_library(library):
    """
    Ensures all expected keys are present and have the correct type.
    Converts legacy or malformed data to the expected structure.
    """
    # List-of-dict keys: always a list of dicts with at least a 'label' field
    list_of_dict_keys = [
        "buttons", "tables", "panels", "alerts", "contests",
        "panel_tags", "table_tags", "section_keywords", "output_file_patterns",
        "active_domains", "inactive_domains", "captcha_patterns",
        "common_output_headers", "common_error_patterns", "button_keywords",
        "contest_type_patterns", "reporting_status_patterns", "vote_method_patterns",
        "precinct_patterns", "anomaly_log", "user_feedback", "download_link_patterns",
        "known_states", "known_counties", "known_districts", "known_cities",
        "precinct_header_tags", "default_noisy_labels", "election", "title",
        "known_contests"
    ]
    for key in list_of_dict_keys:
        val = library.get(key)
        if val is None:
            library[key] = []
        elif isinstance(val, dict):
            # Convert dict to list of dicts
            library[key] = [val]
        elif isinstance(val, list):
            # Convert list of strings to list of dicts with 'label'
            if key in ["buttons", "tables", "panels"]:
                library[key] = [
                    {"label": v} if isinstance(v, str) else v for v in val
                ]
        else:
            # Fallback: wrap in a list
            library[key] = [{"label": str(val)}]

    # Dict keys: always dicts
    dict_keys = [
        "selectors", "domain_selectors", "domain_scrolls", "captcha_solutions",
        "Known_state_to_county_map", "Known_county_to_district_map", "labels", "regex", "metadata"
    ]
    for key in dict_keys:
        val = library.get(key)
        if val is None or not isinstance(val, dict):
            library[key] = {}

    # Fill in version and last_updated if missing
    if "version" not in library:
        library["version"] = "1.0.0"
    if "last_updated" not in library:
        from datetime import datetime
        library["last_updated"] = datetime.utcnow().isoformat()

    return library

def load_context_library(path=CONTEXT_LIBRARY_PATH):
    if not os.path.exists(path):
        library = {"contests": [], "buttons": [], "panels": [], "tables": [], "alerts": []}
    else:
        with open(path, "r", encoding="utf-8") as f:
            library = json.load(f)
    library = sanitize_context_library(library)
    return library

def save_context_library(library, path=CONTEXT_LIBRARY_PATH):
    library = sanitize_context_library(library)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(library, f, indent=2, ensure_ascii=False)

def append_to_context_library(new_data, path=CONTEXT_LIBRARY_PATH):
    """
    Advanced merge/deduplication for context library.
    - Merges lists by key (title/label), updating existing entries with new fields.
    - Merges dicts recursively.
    - Handles new dynamic sections.
    """
    def merge_dicts(existing, new):
        """Recursively merge two dicts."""
        for k, v in new.items():
            if k in existing and isinstance(existing[k], dict) and isinstance(v, dict):
                merge_dicts(existing[k], v)
            else:
                existing[k] = v

    def find_existing(lst, item, keys=("title", "label")):
        """Find index of existing item in list by normalized key."""
        for i, existing in enumerate(lst):
            for key in keys:
                if key in item and key in existing:
                    if normalize_label(item[key]) == normalize_label(existing[key]):
                        return i
        return -1

    library = load_context_library(path)
    dynamic_keys = [
        "contests", "buttons", "panels", "tables", "alerts",
        "common_output_headers", "common_error_patterns", "domain_scrolls",
        "button_keywords", "contest_type_patterns", "reporting_status_patterns",
        "vote_method_patterns", "precinct_patterns",
        "anomaly_log", "user_feedback", "download_link_patterns",
        "panel_tags", "table_tags", "section_keywords", "output_file_patterns",
        "active_domains", "inactive_domains", "captcha_patterns", "captcha_solutions",
        "Known_state_to_county_map", "Known_county_to_district_map", "state_module_map",
        "selectors", "known_states", "known_counties", "known_districts", "known_cities", "regex",
        "precinct_header_tags", "common_output_headers", "default_noisy_labels", "domain_selectors",
        "labels", "election", "version", "last_updated"
    ]
    for key in dynamic_keys:
        if key in new_data:
            # Handle lists of dicts (merge by key)
            if isinstance(new_data[key], list):
                if not isinstance(library.get(key), list):
                    library[key] = []
                for item in new_data[key]:
                    if isinstance(item, dict):
                        idx = find_existing(library[key], item)
                        if idx >= 0:
                            merge_dicts(library[key][idx], item)
                        else:
                            library[key].append(item)
                    else:
                        if item not in library[key]:
                            library[key].append(item)
            # Handle dicts (merge/update)
            elif isinstance(new_data[key], dict):
                if not isinstance(library.get(key), dict):
                    library[key] = {}
                for subk, subv in new_data[key].items():
                    if (
                        subk in library[key]
                        and isinstance(library[key][subk], dict)
                        and isinstance(subv, dict)
                    ):
                        merge_dicts(library[key][subk], subv)
                    else:
                        library[key][subk] = subv
    save_context_library(library, path)
    rprint(f"[CONTEXT ORGANIZER] Appended/merged new data to context library at {path}")

def load_processed_urls(path=PROCESSED_URLS_CACHE):
    if not os.path.exists(path):
        return {}
    processed = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                meta = json.loads(line)
                url = meta.get("url")
                if url:
                    processed[url] = meta
            except Exception:
                continue
    return processed

def load_output_cache(path=OUTPUT_CACHE):
    if not os.path.exists(path):
        return []
    cache = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                cache.append(json.loads(line))
            except Exception:
                continue
    return cache

def scan_environment():
    input_files = os.listdir(INPUT_DIR) if os.path.exists(INPUT_DIR) else []
    output_files = os.listdir(OUTPUT_DIR) if os.path.exists(OUTPUT_DIR) else []
    env_info = {
        "input_files": input_files,
        "output_files": output_files,
        "os": os.name,
        "cwd": os.getcwd(),
        "python_version": os.sys.version,
    }
    return env_info

# --- ML/AI: Anomaly Detection, Clustering ---
def detect_anomalies_with_ml(contexts, contamination=0.05, n_estimators=100, random_state=42):
    if not contexts:
        return [], []
    features = []
    le_state = LabelEncoder()
    le_county = LabelEncoder()
    states = [c.get("state", "unknown") for c in contexts]
    counties = [c.get("county", "unknown") for c in contexts]
    le_state.fit(states)
    le_county.fit(counties)
    for c in contexts:
        features.append([
            le_state.transform([c.get("state", "unknown")])[0],
            le_county.transform([c.get("county", "unknown")])[0],
            int(c.get("year", 0)) if str(c.get("year", "0")).isdigit() else 0,
            len(c.get("title", "")),
        ])
    X = np.array(features)
    clf = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state
    )
    preds = clf.fit_predict(X)
    anomalies = [i for i, p in enumerate(preds) if p == -1]
    clustering = DBSCAN(eps=3, min_samples=2).fit(X)
    clusters = clustering.labels_
    return anomalies, clusters

def plot_anomaly_scores(scores, cutoff=None):
    def _plot():
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        plt.plot(sorted(scores), marker='o', linestyle='-', label='Anomaly Score')
        if cutoff is not None:
            plt.axhline(cutoff, color='red', linestyle='--', label=f'Cutoff ({cutoff:.2f})')
        plt.title('IsolationForest Anomaly Scores')
        plt.xlabel('Sample Index (sorted)')
        plt.ylabel('Anomaly Score')
        plt.legend()
        plt.tight_layout()
        # Minimize the plot window if possible
        try:
            mng = plt.get_current_fig_manager()
            mng.window.state('iconic')  # TkAgg backend
        except Exception:
            pass
        plt.show()
    threading.Thread(target=_plot, daemon=True).start()

def plot_clusters(X, clusters, anomalies=None, title="PCA Cluster Visualization"):
    def _plot():
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        plt.figure(figsize=(8, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='tab10', alpha=0.7, label='Cluster')
        if anomalies is not None and len(anomalies) > 0:
            plt.scatter(X_pca[anomalies, 0], X_pca[anomalies, 1], color='red', marker='x', s=80, label='Anomaly')
        plt.title(title)
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.legend()
        plt.tight_layout()
        # Minimize the plot window if possible
        try:
            mng = plt.get_current_fig_manager()
            mng.window.state('iconic')
        except Exception:
            pass
        plt.show()
    threading.Thread(target=_plot, daemon=True).start()
    
def auto_tune_contamination(X, initial_contamination=0.2, min_contamination=0.01, max_contamination=0.2, plot=False):
    clf = IsolationForest(contamination=initial_contamination, random_state=42)
    clf.fit(X)
    scores = -clf.decision_function(X)
    cutoff = np.percentile(scores, 90)
    n_anomalies = np.sum(scores >= cutoff)
    contamination = n_anomalies / len(scores)
    contamination = max(min_contamination, min(max_contamination, contamination))
    if plot:
        plot_anomaly_scores(scores, cutoff=cutoff)
    return contamination

def get_title_embedding_features(contests, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    titles = [c.get("title", "") for c in contests]
    embeddings = model.encode(titles)
    return embeddings

def election_integrity_checks(contests):
    seen = set()
    issues = []
    for c in contests:
        key = (normalize_label(c.get("title")), c.get("year"), c.get("state"), c.get("county"))
        if key in seen:
            issues.append(("duplicate", c))
        else:
            seen.add(key)
        if not c.get("county") or not c.get("state"):
            issues.append(("missing_location", c))
        if not c.get("year") or not str(c.get("year")).isdigit():
            issues.append(("missing_year", c))
    return issues

# --- Real-Time Monitoring Thread ---
def monitor_db_for_alerts(poll_interval=10):
    def monitor():
        last_alert_id = 0
        while True:
            try:
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute("SELECT id, level, msg, context, created_at FROM alerts WHERE id > ? ORDER BY id ASC", (last_alert_id,))
                rows = cursor.fetchall()
                for row in rows:
                    last_alert_id = row[0]
                    rprint(f"[REAL-TIME ALERT][{row[1]}] {row[2]} | Context: {row[3]} | ALERT_TYPE: {row[1]}")
                conn.close()
            except Exception as e:
                logging.error(f"[MONITOR] Error in real-time alert monitor: {e}")
            time.sleep(poll_interval)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()

# --- Main Organizer ---
def organize_context(
    raw_context,
    button_features=None,
    panel_features=None,
    use_library=True,
    cache=None,
    enable_ml=True,
    contamination=None,
    n_estimators=100,
    random_state=42,
    embedding_model="all-MiniLM-L6-v2",
    plot_anomalies=True,
    plot_clusters_flag=True
):
    ensure_db_schema()
    library = load_context_library() if use_library else {
        "contests": [],
        "buttons": [],
        "panels": [],
        "tables": [],
        "alerts": [],
        "labels": [],
        "election": [],
        "regex": [],
        "common_output_headers": [],
        "common_output_headers": [],
        "common_error_patterns": [],
        "domain_selectors": {},
        "domain_scrolls": {},
        "button_keywords": [],
        "contest_type_patterns": [],
        "vote_method_patterns": [],
        "precinct_patterns": [],
        "reporting_status_patterns": [],
        "anomaly_log": [],
        "user_feedback": [],
        "download_link_patterns": [],
        "panel_tags": [],
        "table_tags": [],
        "section_keywords": [],
        "output_file_patterns": [],
        "active_domains": [],
        "inactive_domains": [],
        "captcha_patterns": [],
        "captcha_solutions": {},
        "last_updated": None,
        "version": "1.2.0",
        "Known_state_to_county_map": {},
        "Known_county_to_district_map": {},
        "state_module_map": {},
        "selectors": {},
        "known_states": [],
        "known_counties": [],
        "known_districts": [],
        "known_cities": [],
        "precinct_header_tags": [],
        "default_noisy_labels": []
    }
    processed_urls = load_processed_urls()
    output_cache = load_output_cache()

    # Defensive fix: ensure panels is a dict
    if "panels" in raw_context and isinstance(raw_context["panels"], list):
        raw_context["panels"] = {}

    contests = []
    contest_titles = set()
    for c in raw_context.get("contests", []):
        title = c.get("title") or c.get("label") or c
        norm_title = normalize_label(title)
        if norm_title not in contest_titles:
            contest_titles.add(norm_title)
            contests.append({
                "title": title,
                "year": c.get("year"),
                "type": c.get("type"),
                "state": raw_context.get("state"),
                "county": raw_context.get("county"),
                "raw": c
            })
    for c in library.get("contests", []):
        norm_title = normalize_label(c.get("title", c.get("label", str(c))))
        if norm_title not in contest_titles:
            contests.append(c)
            contest_titles.add(norm_title)

    # --- ML Feature Engineering ---
    features = []
    le_state = LabelEncoder()
    le_county = LabelEncoder()
    states = [c.get("state", "unknown") for c in contests]
    counties = [c.get("county", "unknown") for c in contests]
    le_state.fit(states)
    le_county.fit(counties)
    for c in contests:
        features.append([
            le_state.transform([c.get("state", "unknown")])[0],
            le_county.transform([c.get("county", "unknown")])[0],
            int(c.get("year", 0)) if str(c.get("year", "0")).isdigit() else 0,
            len(c.get("title", "")),
        ])
    features = np.array(features)
    embedding_features = get_title_embedding_features(contests, model_name=embedding_model)
    X = np.hstack([features, embedding_features])

    # --- Adaptive ML parameter tuning ---
    if contamination is None:
        if len(X) > 10:
            contamination = auto_tune_contamination(X, plot=plot_anomalies)
        else:
            contamination = 0.2 if len(X) < 10 else 0.1

    panels = {}
    for c in contests:
        panel = None
        if panel_features:
            panel = next((p for p in panel_features if normalize_label(p.get("label", "")) == normalize_label(c["title"])), None)
        if not panel:
            panel = raw_context.get("panels", {}).get(c["title"])
        panels[c["title"]] = panel

    buttons_by_contest = defaultdict(list)
    raw_buttons = button_features or raw_context.get("buttons", [])
    lib_buttons = library.get("buttons", [])
    if not isinstance(raw_buttons, list):
        raw_buttons = []
    if not isinstance(lib_buttons, list):
        lib_buttons = []
    all_buttons = raw_buttons + lib_buttons
    for btn in all_buttons:
        for c in contests:
            if c["title"].lower() in btn.get("label", "").lower():
                buttons_by_contest[c["title"]].append(btn)
            elif "election" in btn.get("label", "").lower() and "election" in c["title"].lower():
                buttons_by_contest[c["title"]].append(btn)
        if not any(btn in v for v in buttons_by_contest.values()):
            buttons_by_contest["__unmatched__"].append(btn)

    tables_by_contest = defaultdict(list)
    raw_tables = raw_context.get("tables", [])
    lib_tables = library.get("tables", [])
    if not isinstance(raw_tables, list):
        raw_tables = []
    if not isinstance(lib_tables, list):
        lib_tables = []
    all_tables = raw_tables + lib_tables
    for tbl in all_tables:
        for c in contests:
            if c["title"].lower() in tbl.get("label", "").lower():
                tables_by_contest[c["title"]].append(tbl)
        if not any(tbl in v for v in tables_by_contest.values()):
            tables_by_contest["__unmatched__"].append(tbl)
            
    metadata = {
        "state": raw_context.get("state"),
        "county": raw_context.get("county"),
        "source_url": raw_context.get("url"),
        "election_type": raw_context.get("election_type"),
        "scrape_time": raw_context.get("scrape_time"),
        "year": None,
        "race": raw_context.get("race"),
        "environment": scan_environment(),
    }

    # --- ML anomaly detection & clustering ---
    anomalies, clusters = [], []
    if enable_ml and len(contests) > 0:
        try:
            anomalies, clusters = detect_anomalies_with_ml(
                contests,
                contamination=contamination,
                n_estimators=n_estimators,
                random_state=random_state
            )
            if anomalies:
                for idx in anomalies:
                    contest = contests[idx]
                    title = contest.get('title', str(contest))
                    rprint(f"[bold magenta][ML][/bold magenta] Context anomaly detected: [bold yellow]{title}[/bold yellow]\n  [dim]Context:[/dim] {contest}")
            if plot_clusters_flag:
                plot_clusters(X, clusters, anomalies=anomalies)
        except Exception as e:
            rprint(f"[bold red][ML] Anomaly detection failed:[/bold red] {e}")

    # --- Election integrity checks ---
    integrity_issues = election_integrity_checks(contests)
    for issue, contest in integrity_issues:
        if issue == "duplicate":
            rprint(f"[bold yellow][INTEGRITY][/bold yellow] Duplicate contest detected.\n  [dim]Context:[/dim] {contest}")
        elif issue == "missing_location":
            rprint(f"[bold yellow][INTEGRITY][/bold yellow] Contest missing location info.\n  [dim]Context:[/dim] {contest}")
        elif issue == "missing_year":
            rprint(f"[bold yellow][INTEGRITY][/bold yellow] Contest missing year.\n  [dim]Context:[/dim] {contest}")

    if len(contests) > 50:
        rprint(f"[bold red][CONTEXT ORGANIZER][/bold red] High contest count detected — possible congestion.\n  [dim]Context:[/dim] contest_count={len(contests)}")

    organized = {
        "contests": contests,
        "buttons": dict(buttons_by_contest),
        "panels": panels,
        "tables": dict(tables_by_contest),
        "metadata": metadata,
        "anomalies": [contests[i] for i in anomalies] if anomalies else [],
        "clusters": clusters.tolist() if hasattr(clusters, "tolist") else clusters,
        "integrity_issues": integrity_issues,
    }
    valid_years = [
        c.get("year")
        for c in contests
        if c.get("year") and c.get("type") and str(c.get("year")).isdigit()
    ]
    if valid_years:
        metadata["year"] = valid_years[0]
    else:
        metadata["year"] = "Unknown"
    append_to_context_library(organized, path=CONTEXT_LIBRARY_PATH)
    rprint(
        f"[bold green][CONTEXT ORGANIZER][/bold green] Organized context for [bold]{len(contests)}[/bold] contests.\n"
        f"  [magenta]Anomalies:[/magenta] {len(anomalies)}  [yellow]Integrity issues:[/yellow] {len(integrity_issues)}"
    )

    # Insert contests into DB for persistent storage and future ML
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    for c in contests:
        cursor.execute("""
            INSERT OR IGNORE INTO contests (title, year, type, state, county, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            c.get("title"),
            c.get("year"),
            c.get("type"),
            c.get("state"),
            c.get("county"),
            json.dumps(c)
        ))
    conn.commit()
    conn.close()

    return organized

def get_best_button(organized_context, contest_title, keywords=None, class_hint=None):
    buttons = organized_context["buttons"].get(contest_title, [])
    if not buttons:
        buttons = organized_context["buttons"].get("__unmatched__", [])
    if keywords:
        for btn in buttons:
            if any(kw.lower() in btn.get("label", "").lower() for kw in keywords):
                if not class_hint or class_hint in btn.get("class", ""):
                    return btn
    if class_hint:
        for btn in buttons:
            if class_hint in btn.get("class", ""):
                return btn
    return buttons[0] if buttons else None

def get_panel(organized_context, contest_title):
    return organized_context["panels"].get(contest_title)

def get_tables(organized_context, contest_title):
    return organized_context["tables"].get(contest_title, [])

monitor_db_for_alerts(poll_interval=10)