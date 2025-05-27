"""
context_coordinator.py

Coordinates advanced context analysis, NLP, and orchestration between spaCy (NLP) and the context organizer (data/ML).
Acts as the "brains" for semantic, entity, and text-driven logic, while delegating data formatting and analytics to the organizer.
"""

import os
import json
import sqlite3
from collections import defaultdict
from rich import print as rprint

import importlib.util
import glob


from spacy_utils import (
    extract_entities,
    get_sentences,
    clean_text,
    extract_entities_from_list,
    extract_entity_labels,
    extract_locations,
    extract_dates,
)
from context_organizer import DB_PATH, load_context_library

# --- Database Access Utilities ---

def fetch_contests_from_db(limit=100, filters=None):
    """
    Fetch contests from the database for NLP analysis or reporting.
    Optionally filter by year, state, etc.
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
        # Try to load metadata as dict, fallback to DB columns
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

# --- Context Coordinator Functions (NLP, Reporting, Feedback) ---

def analyze_contest_titles(contests):
    """
    Use spaCy to extract entities, locations, and dates from contest titles.
    Returns a dict mapping contest title to extracted info.
    """
    analysis = {}
    for c in contests:
        title = c.get("title", "")
        entities = extract_entities(title)
        locations = extract_locations(title)
        dates = extract_dates(title)
        analysis[title] = {
            "entities": entities,
            "locations": locations,
            "dates": dates,
        }
    return analysis

def summarize_context_entities(contests):
    """
    Summarize all unique entity labels and values across all contest titles.
    """
    all_entities = []
    all_labels = set()
    for c in contests:
        ents = extract_entities(c.get("title", ""))
        all_entities.extend(ents)
        all_labels.update(label for _, label in ents)
    return {
        "unique_entities": list(set(all_entities)),
        "entity_labels": list(all_labels),
    }

def get_available_states(states_dir=None):
    """
    Dynamically list available state modules in the handlers/states directory.
    Returns a set of state names (title case, e.g., 'Texas').
    """
    if states_dir is None:
        states_dir = os.path.join(os.path.dirname(__file__), "..", "handlers", "states")
    state_files = glob.glob(os.path.join(states_dir, "*.py"))
    states = set()
    for f in state_files:
        name = os.path.splitext(os.path.basename(f))[0]
        if name == "__init__":
            continue
        # Convert snake_case or lowercase to Title Case (e.g., new_york -> New York)
        state_name = " ".join([w.capitalize() for w in name.split("_")])
        states.add(state_name)
    return states

def detect_location_coverage(contests, required_states=None, available_states=None):
    """
    Check if all required or available states are covered in the contest titles.
    Returns missing and extra states.
    """
    found_states = set()
    for c in contests:
        found_states.update(extract_locations(c.get("title", "")))
    # Use available_states if not explicitly provided
    if available_states is None:
        available_states = get_available_states()
    missing = [state for state in available_states if state not in found_states]
    extra = [state for state in found_states if state not in available_states]
    return {
        "found_states": list(found_states),
        "missing_states": missing,
        "extra_states": extra,
        "available_states": list(available_states),
    }

def find_date_anomalies(contests, expected_year=None):
    """
    Find contests whose extracted dates do not match the expected year.
    """
    anomalies = []
    for c in contests:
        dates = extract_dates(c.get("title", ""))
        if expected_year and not any(str(expected_year) in d for d in dates):
            anomalies.append(c)
    return anomalies

def clean_and_segment_titles(contests):
    """
    Clean and segment contest titles into sentences.
    Returns a dict mapping contest title to cleaned sentences.
    """
    result = {}
    for c in contests:
        title = c.get("title", "")
        cleaned = clean_text(title)
        sentences = get_sentences(cleaned)
        result[title] = sentences
    return result

def enrich_contests_with_nlp(contests):
    """
    Add NLP-derived fields (entities, locations, dates) to each contest dict.
    """
    for c in contests:
        title = c.get("title", "")
        c["entities"] = extract_entities(title)
        c["locations"] = extract_locations(title)
        c["dates"] = extract_dates(title)
    return contests

def coordinator_report(contests, required_states=None, expected_year=None, available_states=None):
    """
    Generate a rich report of NLP findings and coverage for the given contests.
    """
    rprint("[bold cyan][COORDINATOR] NLP Entity Summary[/bold cyan]")
    entity_summary = summarize_context_entities(contests)
    rprint(f"Unique entity labels: {entity_summary['entity_labels']}")
    rprint(f"Unique entities: {entity_summary['unique_entities']}")

    # Use available_states for coverage if not explicitly provided
    loc_cov = detect_location_coverage(contests, required_states, available_states)
    rprint(f"Found states: {loc_cov['found_states']}")
    rprint(f"Available states (from handlers): {loc_cov['available_states']}")
    if loc_cov["missing_states"]:
        rprint(f"[yellow]Missing states:[/yellow] {loc_cov['missing_states']}")
    if loc_cov["extra_states"]:
        rprint(f"[red]Unrecognized/extra states in contests:[/red] {loc_cov['extra_states']}")

    if expected_year:
        date_anoms = find_date_anomalies(contests, expected_year)
        if date_anoms:
            rprint(f"[red]Contests with date anomalies (not matching {expected_year}):[/red]")
            for c in date_anoms:
                rprint(f"  - {c.get('title')}")

# --- Active Learning / User Feedback ---

def prompt_user_for_label(contest, prompt_func=input):
    """
    Prompt the user (or UI) to label a contest as correct/incorrect, or annotate.
    Returns the label or annotation.
    """
    rprint(f"[ACTIVE LEARNING] Please review contest: [bold]{contest['title']}[/bold]")
    label = prompt_func("Label as (1) correct, (0) incorrect, or type annotation: ")
    return label

def active_learning_feedback(contests, uncertain_idxs, prompt_func=input):
    """
    For a list of uncertain contest indices, prompt the user for feedback.
    Returns a dict of idx:label/annotation.
    """
    feedback = {}
    for idx in uncertain_idxs:
        contest = contests[idx]
        label = prompt_user_for_label(contest, prompt_func=prompt_func)
        feedback[idx] = label
    return feedback

# --- Coordinator Pipeline ---

def coordinator_pipeline(
    raw_context=None,
    required_states=None,
    expected_year=None,
    from_db=False,
    db_limit=100,
    db_filters=None,
    enable_active_learning=False,
    uncertain_idxs=None,
    prompt_func=input
):
    """
    Main entry: coordinates NLP enrichment and reporting for a raw context dict or DB.
    Returns enriched contests and a summary report.
    """
    if from_db:
        contests = fetch_contests_from_db(limit=db_limit, filters=db_filters)
    else:
        contests = raw_context.get("contests", []) if raw_context else []
    contests = enrich_contests_with_nlp(contests)
    coordinator_report(contests, required_states=required_states, expected_year=expected_year)

    # Optionally run active learning/user feedback
    if enable_active_learning and uncertain_idxs:
        feedback = active_learning_feedback(contests, uncertain_idxs, prompt_func=prompt_func)
        rprint(f"[ACTIVE LEARNING] Feedback received: {feedback}")

    return contests

# Example usage (for testing)
if __name__ == "__main__":
    # Example: from context library
    context_lib = load_context_library()
    sample_context = {
        "contests": [
            {"title": "2024 Presidential Election - New York"},
            {"title": "2022 Senate Race - California"},
            {"title": "2024 Mayoral Election - Houston, TX"},
            {"title": "2023 School Board - Miami"},
        ]
    }
    # Run on sample context
    enriched = coordinator_pipeline(
        raw_context=sample_context,
        required_states=["New York", "California", "Texas"],
        expected_year=2024,
        enable_active_learning=True,
        uncertain_idxs=[0, 2],  # Simulate feedback for first and third contest
        prompt_func=lambda prompt: "1"  # Simulate always labeling as correct
    )
    print(json.dumps(enriched, indent=2))

    # Run on DB (uncomment to use)
    enriched_db = coordinator_pipeline(
         from_db=True,
         db_limit=10,
         required_states=["New York", "California", "Texas"],
         expected_year=2024 
        )