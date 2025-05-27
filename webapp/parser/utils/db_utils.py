import os
import json
import re
import sqlite3
from pathlib import Path
from typing import Dict, Any

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "Context_Integration", "context_elections.db")

def update_contest_in_db(contest, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE contests
        SET title=?, year=?, type=?, state=?, county=?, metadata=?
        WHERE id=?
    """, (
        contest.get("title"),
        contest.get("year"),
        contest.get("type"),
        contest.get("state"),
        contest.get("county"),
        json.dumps(contest),
        contest.get("id")
    ))
    conn.commit()
    conn.close()
    
def fetch_contests_by_filter(filters=None, limit=100, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
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

def append_to_context_library(data, path=None):
    from ..utils.shared_logic import load_context_library
    if path is None:
        from ..Context_Integration.context_organizer import CONTEXT_LIBRARY_PATH
        path = CONTEXT_LIBRARY_PATH
    library = load_context_library(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(library, f, indent=2, ensure_ascii=False)

def normalize_label(label):
    if not label:
        return ""
    return re.sub(r"\W+", "", str(label).strip().lower())

# --- Utility: Processed URL cache ---
def load_processed_urls() -> Dict[str, Any]:
    """Load the processed URL cache as a dict: url -> metadata dict."""
    from ..utils.output_utils import CACHE_FILE
    cache_path = Path(CACHE_FILE)  # Ensure it's a Path object
    if not cache_path.exists() or os.path.getsize(cache_path) == 0:
        return {}
    with cache_path.open('r', encoding="utf-8") as f:
        try:
            entries = json.load(f)
            if not isinstance(entries, list):
                entries = []
        except Exception:
            entries = []
    processed = {}
    for entry in entries:
        url = entry.get("url")
        if url:
            processed[url] = entry
    return processed

def load_output_cache(path=None):
    if path is None:
        from ..Context_Integration.context_organizer import OUTPUT_CACHE
        path = OUTPUT_CACHE
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]