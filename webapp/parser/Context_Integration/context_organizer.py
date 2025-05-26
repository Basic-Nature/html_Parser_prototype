"""
context_organizer.py

Advanced context organizer for election HTML parsing and data integrity.
Integrates dynamic logging, ML anomaly detection, cache-aware learning, and robust DB.
"""

import re
import os
import json
import logging
import sqlite3
import threading
import time
from collections import defaultdict

from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

logging.basicConfig(level=logging.DEBUG)

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

def normalize_label(label):
    return re.sub(r"\s+", " ", str(label).strip().lower())

def load_context_library(path=CONTEXT_LIBRARY_PATH):
    if not os.path.exists(path):
        return {"contests": [], "buttons": [], "panels": [], "tables": [], "alerts": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_context_library(library, path=CONTEXT_LIBRARY_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(library, f, indent=2, ensure_ascii=False)

def append_to_context_library(new_data, path=CONTEXT_LIBRARY_PATH):
    library = load_context_library(path)
    for key in ["contests", "buttons", "panels", "tables", "alerts"]:
        if key in new_data:
            for item in new_data[key]:
                norm_label = normalize_label(item.get("title", item.get("label", str(item))))
                if not any(
                    normalize_label(existing.get("title", existing.get("label", str(existing)))) == norm_label
                    for existing in library.get(key, [])
                ):
                    library.setdefault(key, []).append(item)
    save_context_library(library, path)
    logging.info(f"[CONTEXT ORGANIZER] Appended new data to context library at {path}")

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

# --- Advanced ML/AI: Anomaly Detection, Clustering, NLP ---
def detect_anomalies_with_ml(contexts, cache):
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
    clf = IsolationForest(contamination=0.05, random_state=42)
    preds = clf.fit_predict(X)
    anomalies = [i for i, p in enumerate(preds) if p == -1]
    clustering = DBSCAN(eps=3, min_samples=2).fit(X)
    clusters = clustering.labels_
    return anomalies, clusters

def nlp_title_outlier_detection(contests):
    """Detect outlier contest titles using TF-IDF and cosine similarity."""
    titles = [c.get("title", "") for c in contests]
    if len(titles) < 2:
        return []
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(titles)
    similarities = (X * X.T).A
    avg_sim = np.mean(similarities, axis=1)
    threshold = np.percentile(avg_sim, 10)
    outliers = [i for i, sim in enumerate(avg_sim) if sim < threshold]
    return outliers

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
                    logging.alert(f"[REAL-TIME ALERT][{row[1]}] {row[2]}", context=row[3], alert_type=row[1])
                conn.close()
            except Exception as e:
                logging.error(f"[MONITOR] Error in real-time alert monitor: {e}")
            time.sleep(poll_interval)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()

# --- Main Organizer ---
def organize_context(raw_context, button_features=None, panel_features=None, use_library=True, cache=None, enable_ml=True):
    ensure_db_schema()
    library = load_context_library() if use_library else {"contests": [], "buttons": [], "panels": [], "tables": [], "alerts": []}
    processed_urls = load_processed_urls()
    output_cache = load_output_cache()

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

    panels = {}
    for c in contests:
        panel = None
        if panel_features:
            panel = next((p for p in panel_features if normalize_label(p.get("label", "")) == normalize_label(c["title"])), None)
        if not panel:
            panel = raw_context.get("panels", {}).get(c["title"])
        panels[c["title"]] = panel

    buttons_by_contest = defaultdict(list)
    all_buttons = (button_features or raw_context.get("buttons", [])) + library.get("buttons", [])
    for btn in all_buttons:
        for c in contests:
            if c["title"].lower() in btn.get("label", "").lower():
                buttons_by_contest[c["title"]].append(btn)
            elif "election" in btn.get("label", "").lower() and "election" in c["title"].lower():
                buttons_by_contest[c["title"]].append(btn)
        if not any(btn in v for v in buttons_by_contest.values()):
            buttons_by_contest["__unmatched__"].append(btn)

    tables_by_contest = defaultdict(list)
    all_tables = raw_context.get("tables", []) + library.get("tables", [])
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
        "environment": scan_environment(),
    }

    # --- ML anomaly detection & clustering ---
    anomalies, clusters = [], []
    if enable_ml and contests:
        try:
            anomalies, clusters = detect_anomalies_with_ml(contests, output_cache)
            if anomalies:
                for idx in anomalies:
                    logging.alert(
                        f"Anomaly detected in contest: {contests[idx]['title']}",
                        context=contests[idx],
                        alert_type="warning"
                    )
            # NLP outlier detection
            nlp_outliers = nlp_title_outlier_detection(contests)
            for idx in nlp_outliers:
                logging.alert(
                    f"NLP Outlier detected in contest title: {contests[idx]['title']}",
                    context=contests[idx],
                    alert_type="warning"
                )
        except Exception as e:
            logging.error(f"[ML] Anomaly/NLP detection failed: {e}")

    # --- Election integrity checks ---
    integrity_issues = election_integrity_checks(contests)
    for issue, contest in integrity_issues:
        if issue == "duplicate":
            logging.warning("[INTEGRITY] Duplicate contest detected.", context=contest)
        elif issue == "missing_location":
            logging.warning("[INTEGRITY] Contest missing location info.", context=contest)
        elif issue == "missing_year":
            logging.warning("[INTEGRITY] Contest missing year.", context=contest)

    # --- Congestion/flow detection ---
    if len(contests) > 50:
        logging.warning("[CONTEXT ORGANIZER] High contest count detected — possible congestion.", context={"contest_count": len(contests)})

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
    append_to_context_library(organized, path=CONTEXT_LIBRARY_PATH)
    logging.info(f"[CONTEXT ORGANIZER] Organized context for {len(contests)} contests. Anomalies: {len(anomalies)}. Integrity issues: {len(integrity_issues)}")

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

# --- Helper: Get best button for a contest ---
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

# --- Start real-time monitoring thread on import ---
monitor_db_for_alerts(poll_interval=10)