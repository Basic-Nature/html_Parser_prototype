"""
context_organizer.py

Advanced context organizer for election HTML parsing and data integrity.
Handles data formatting, ML anomaly detection, cache-aware learning, clustering, and robust DB.
Delegates NLP/semantic logic to the context_coordinator and spacy_utils modules.
"""

import re
import argparse
import os
import json
import logging
import matplotlib.pyplot as plt
import platform
from collections import defaultdict
from ..utils.shared_logger import rprint
from sentence_transformers import SentenceTransformer

from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import numpy as np
from ..config import CONTEXT_DB_PATH
from ..utils.db_utils import (
    append_to_context_library,
    load_processed_urls,
    load_output_cache,
    normalize_label,
    _safe_db_path,
    update_contest_in_db,
)
from ..utils.shared_logic import get_title_embedding_features, load_context_library, scan_environment
from .Integrity_check import (
    detect_anomalies_with_ml, print_ml_anomalies,
    auto_tune_contamination, election_integrity_checks, monitor_db_for_alerts
)
import sqlite3

from rich.console import Console

console = Console()

# If you want to suppress the progress bar from SentenceTransformer
# os.environ["TRANSFORMERS_NO_PROGRESS_BAR"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format="\n[%(levelname)s] %(message)s\n"
)
from ..config import BASE_DIR, CONTEXT_LIBRARY_PATH
# Paths

PROCESSED_URLS_CACHE = os.path.join(BASE_DIR, ".processed_urls")
OUTPUT_CACHE = os.path.join(BASE_DIR, ".output_cache.jsonl")
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# --- DB Schema Setup ---
def ensure_db_schema():
    path = _safe_db_path(CONTEXT_DB_PATH)
    conn = sqlite3.connect(path)
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

processed_urls = load_processed_urls()
output_cache = load_output_cache()

def build_dom_tree(segments):
    """
    Build a DOM tree from a list of segments that already include parent/child relationships and indices.
    Each node contains: tag, attrs, classes, id, html, children, parent_idx, start, end, _idx.
    Returns the root nodes (usually head/body/url_root) and a flat list with parent/child relationships.
    Ensures parent/child consistency and logs any inconsistencies.
    """
    nodes = []
    idx_map = {}
    for i, seg in enumerate(segments):
        node = dict(seg)
        node.setdefault("children", [])
        node.setdefault("parent_idx", None)
        node.setdefault("_idx", i)
        node.setdefault("start", None)
        node.setdefault("end", None)
        nodes.append(node)
        idx_map[node["_idx"]] = node

    # Fix up children and parent_idx in case they're not integers
    for node in nodes:
        node["children"] = [c if isinstance(c, int) else getattr(c, "_idx", None) for c in node.get("children", [])]
        node["children"] = [c for c in node["children"] if c is not None]
        if isinstance(node.get("parent_idx"), dict):
            node["parent_idx"] = node["parent_idx"].get("_idx")
        elif not (isinstance(node.get("parent_idx"), int) or node.get("parent_idx") is None):
            node["parent_idx"] = None

    # Parent/child consistency check
    for node in nodes:
        for child_idx in node["children"]:
            if child_idx >= len(nodes) or nodes[child_idx].get("parent_idx") != node["_idx"]:
                logging.warning(f"Inconsistent parent/child: node {node['_idx']} child {child_idx}")

    roots = [node for node in nodes if node["parent_idx"] is None]
    dom_tree = {
        "roots": [node["_idx"] for node in roots],
        "nodes": nodes
    }
    return dom_tree

def extract_html_by_idx(nodes, idx, full_html):
    """
    Extract the exact HTML for a node using its start/end indices.
    """
    node = nodes[idx]
    if node.get("start") is not None and node.get("end") is not None:
        return full_html[node["start"]:node["end"]]
    return node.get("html", "")

def extract_subtree_html(nodes, idx, full_html):
    """
    Recursively extract the HTML for a node and all its descendants.
    """
    node = nodes[idx]
    if not node["children"]:
        return extract_html_by_idx(nodes, idx, full_html)
    # Get the minimal start and maximal end among all descendants
    indices = [idx]
    stack = list(node["children"])
    while stack:
        child_idx = stack.pop()
        indices.append(child_idx)
        stack.extend(nodes[child_idx]["children"])
    starts = [nodes[i]["start"] for i in indices if nodes[i].get("start") is not None]
    ends = [nodes[i]["end"] for i in indices if nodes[i].get("end") is not None]
    if starts and ends:
        return full_html[min(starts):max(ends)]
    return node.get("html", "")

def group_nodes_by_label(nodes, label_field="ml_label"):
    """
    Group nodes by a given label field (e.g., ml_label or semantic_tags).
    Returns a dict: label -> list of nodes.
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for node in nodes:
        label = node.get(label_field)
        if isinstance(label, list):
            for l in label:
                groups[l].append(node)
        elif label:
            groups[label].append(node)
    return dict(groups)

def expose_dom_parts(dom_tree):
    """
    Expose organized DOM parts for downstream use.
    Returns dicts for head, body, wrappers, tables, buttons, etc.
    Adds 'direct_parent' field for each node.
    """
    nodes = dom_tree["nodes"]
    head_nodes = [n for n in nodes if n["tag"].lower() == "head"]
    body_nodes = [n for n in nodes if n["tag"].lower() == "body"]
    wrappers = [n for n in nodes if n["tag"].lower() in ("div", "section", "form")]
    tables = [n for n in nodes if n["tag"].lower() == "table"]
    buttons = [n for n in nodes if n.get("is_button")]
    clickable = [n for n in nodes if n.get("is_clickable")]
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
    plot_clusters_flag=True,
):
    """
    Organizes the context for a parsed HTML page, including DOM structure, contests, panels, buttons, tables, and ML features.
    Leverages accurate DOM indices and relationships for downstream ML and semantic analysis.
    """
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
        "HTML_TAGS": [],
        "common_output_headers": [],
        "common_error_patterns": [],
        "domain_selectors": {},
        "domain_scrolls": {},
        "button_keywords": [],
        "contest_type_patterns": [],
        "vote_method_patterns": [],
        "location_patterns": [],
        "percent_patterns": [],
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
        "default_noisy_labels": [],
        "download_links": []
    }

    if "panels" in raw_context and isinstance(raw_context["panels"], list):
        raw_context["panels"] = {}

    tagged_segments = raw_context.get("tagged_segments_with_attrs", [])
    url_value = raw_context.get("url", "")

    # --- Virtual root handling: _idx=0, all real roots point to it as parent ---
    virtual_root = {
        "tag": "url_root",
        "attrs": {},
        "classes": [],
        "id": "url_root",
        "html": url_value,
        "is_button": False,
        "is_clickable": False,
        "children": [],
        "parent_idx": None,
        "start": None,
        "end": None,
        "_idx": 0
    }
    # Shift all indices by 1 to accommodate virtual root at _idx=0
    for seg in tagged_segments:
        seg["_idx"] = seg.get("_idx", 0) + 1
        if seg.get("parent_idx") is None:
            seg["parent_idx"] = 0  # Point to virtual root
        elif isinstance(seg["parent_idx"], int):
            seg["parent_idx"] += 1
        seg["children"] = [c + 1 if isinstance(c, int) else c for c in seg.get("children", [])]
    tagged_segments = [virtual_root] + tagged_segments

    dom_tree = build_dom_tree(tagged_segments)
    dom_tree["source_url"] = url_value
    dom_parts = expose_dom_parts(dom_tree)

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
                plot_clusters_flag =print_ml_anomalies(anomalies, contests)
        except Exception as e:
            rprint(f"[bold red][ML] Anomaly detection failed:[/bold red] {e}")

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
        "dom_parts": dom_parts,
        "extract_html_by_idx": lambda idx, html=raw_context.get("raw_html", ""): extract_html_by_idx(dom_tree["nodes"], idx, html),
        "extract_subtree_html": lambda idx, html=raw_context.get("raw_html", ""): extract_subtree_html(dom_tree["nodes"], idx, html),
        "group_nodes_by_label": lambda label_field="ml_label": group_nodes_by_label(dom_tree["nodes"], label_field),
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
    db_path = _safe_db_path(CONTEXT_DB_PATH)
    conn = sqlite3.connect(db_path)
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

def correct_and_update_contest(self, contest_id, correction_data):
    """
    Update a contest in the DB and context library, then re-organize context.
    """
    from ..utils.db_utils import update_contest_in_db
    # 1. Update DB
    update_contest_in_db({"id": contest_id, **correction_data})
    # 2. Update context library if needed
    for key, value in correction_data.items():
        # Example: add new county/state mapping if not present
        if key == "county" and value not in self.library.get("known_counties", []):
            self.library.setdefault("known_counties", []).append(value)
        # Add similar logic for other fields as needed
    # 3. Save updated context library (if you persist it)
    # save_context_library(self.library)
    # 4. Re-organize context
    self.organized = None
    # 5. Log correction
    self.log_field_selection(
        field_type="contest",
        field_name="correction",
        extracted_value=correction_data,
        method="manual",
        score=1.0,
        result="manual_pass",
        context={"contest_id": contest_id},
        user_feedback=None
    )

def safe_filename(s):
    """Sanitize a string to be safe for use as a filename or path component."""
    s = str(s)
    s = s.replace("..", "").replace("/", "").replace("\\", "")
    return "".join(c if c.isalnum() or c in " _-" else "_" for c in s).strip() or "Unknown"

def append_to_context_library(new_data, path=CONTEXT_LIBRARY_PATH):
    # Normalize the path to remove any double slashes/backslashes
    path = os.path.normpath(path)
    # Default structure with all expected keys
    default_library = {
        "contests": [],
        "buttons": [],
        "panels": [],
        "tables": [],
        "alerts": [],
        "labels": [],
        "election": [],
        "regex": [],
        "HTML_TAGS": [],
        "common_output_headers": [],
        "common_error_patterns": [],
        "domain_selectors": {},
        "domain_scrolls": {},
        "button_keywords": [],
        "contest_type_patterns": [],
        "vote_method_patterns": [],
        "location_patterns": [],
        "percent_patterns": [],
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
        "default_noisy_labels": [],
        "download_links": []
    }

    # Load or initialize the library
    if os.path.exists(path):
        with open(path, "r+", encoding="utf-8") as f:
            try:
                library = json.load(f)
            except Exception:
                library = default_library.copy()
    else:
        library = default_library.copy()

    # Ensure all keys exist
    for key, default_val in default_library.items():
        if key not in library:
            library[key] = default_val

    # Merge download_links (deduplicate by (href, format))
    existing_links = { (safe_filename(l.get("href", "")), l.get("format", "")): l for l in library.get("download_links", []) }
    new_links = { (safe_filename(l.get("href", "")), l.get("format", "")): l for l in new_data.get("metadata", {}).get("download_links", []) }
    merged_links = list({**existing_links, **new_links}.values())
    library["download_links"] = merged_links

    # Optionally sanitize other filename/path fields in new_data if needed
    # (e.g., for contests, tables, etc.)

    # Write back to file
    with open(path, "w", encoding="utf-8") as f:
        json.dump(library, f, indent=2)