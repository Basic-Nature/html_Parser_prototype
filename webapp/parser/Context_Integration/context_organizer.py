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
import platform
import sqlite3
from collections import defaultdict
from ..utils.shared_logger import rprint
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
from ..utils.db_utils import append_to_context_library, load_processed_urls, load_output_cache, normalize_label
from ..utils.shared_logic import get_title_embedding_features, load_context_library, scan_environment
from .Integrity_check import (
    detect_anomalies_with_ml, plot_clusters,
    auto_tune_contamination, election_integrity_checks, monitor_db_for_alerts
)

# If you want to suppress the progress bar from SentenceTransformer
# os.environ["TRANSFORMERS_NO_PROGRESS_BAR"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format="\n[%(levelname)s] %(message)s\n"
)

# Paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
CONTEXT_LIBRARY_PATH = os.path.join(
    BASE_DIR, "Context_Library", "context_library.json"
)
DB_PATH = os.path.join(BASE_DIR, "context_elections.db")
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





def build_dom_tree(segments):
    """
    Build a DOM tree from a flat list of segments.
    Each node contains: tag, attrs, classes, id, html, children, parent_idx.
    Returns the root nodes (usually head/body) and a flat list with parent/child relationships.
    """
    # Sort segments by their position in the original HTML for hierarchy
    # We'll use the index of the segment's start in the HTML as a proxy for order
    # (Assumes segments are non-overlapping and in order)
    nodes = []
    for idx, seg in enumerate(segments):
        node = dict(seg)
        node["children"] = []
        node["parent_idx"] = None
        node["_idx"] = idx
        nodes.append(node)

    # Build parent-child relationships by containment (simple, not perfect for malformed HTML)
    for i, node in enumerate(nodes):
        node_html = node["html"]
        node_start = node_html
        for j, possible_parent in enumerate(nodes):
            if i == j:
                continue
            parent_html = possible_parent["html"]
            if node_html in parent_html and len(parent_html) > len(node_html):
                # Prefer the smallest parent (closest ancestor)
                if node["parent_idx"] is None or len(nodes[node["parent_idx"]]["html"]) > len(parent_html):
                    node["parent_idx"] = j

    # Assign children
    for node in nodes:
        if node["parent_idx"] is not None:
            nodes[node["parent_idx"]]["children"].append(node["_idx"])

    # Find root nodes (no parent)
    roots = [node for node in nodes if node["parent_idx"] is None]

    # Optionally, group by major wrappers
    dom_tree = {
        "roots": [node["_idx"] for node in roots],
        "nodes": nodes
    }
    return dom_tree

def expose_dom_parts(dom_tree):
    """
    Expose organized DOM parts for downstream use.
    Returns dicts for head, body, wrappers, tables, buttons, etc.
    """
    nodes = dom_tree["nodes"]
    head_nodes = [n for n in nodes if n["tag"].lower() == "head"]
    body_nodes = [n for n in nodes if n["tag"].lower() == "body"]
    wrappers = [n for n in nodes if n["tag"].lower() in ("div", "section", "form")]
    tables = [n for n in nodes if n["tag"].lower() == "table"]
    buttons = [n for n in nodes if n.get("is_button")]
    clickable = [n for n in nodes if n.get("is_clickable")]
    # Expose direct parent for each node
    for n in nodes:
        n["direct_parent"] = nodes[n["parent_idx"]]["tag"] if n["parent_idx"] is not None else None
    return {
        "head_nodes": head_nodes,
        "body_nodes": body_nodes,
        "wrappers": wrappers,
        "tables": tables,
        "buttons": buttons,
        "clickable": clickable,
        "all_nodes": nodes,
        "roots": dom_tree["roots"]
    }

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

    # --- DOM Tree Construction ---
    tagged_segments = raw_context.get("tagged_segments_with_attrs", [])
    dom_tree = build_dom_tree(tagged_segments)
    dom_parts = expose_dom_parts(dom_tree)

    # --- Contest Extraction (legacy, keep for now) ---
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
        "dom_tree": dom_tree,
        "dom_parts": dom_parts
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

monitor_db_for_alerts(poll_interval=10)