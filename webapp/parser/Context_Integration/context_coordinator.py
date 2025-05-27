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
from spacy_utils import demo_analysis
import importlib.util
import glob
from utils.shared_logger import log_info, log_warning, log_error, log_critical, log_alert

from spacy_utils import (
    extract_entities,
    get_sentences,
    clean_text,
    flag_suspicious_contests,
    extract_entities_from_list,
    extract_entity_labels,
    extract_locations,
    extract_dates,
)
from context_organizer import (
    DB_PATH, load_context_library, election_integrity_checks, organize_context,
    update_contest_in_db, fetch_contests_by_filter
)

# --- Database Access Utilities ---

SAMPLE_JSON_PATH = os.path.join(os.path.dirname(__file__), "sample.json")

def analyze_filtered_contests(filters=None, limit=20):
    """
    Fetch contests from the DB using filters, enrich with NLP, and print a report.
    """
    contests = fetch_contests_by_filter(filters=filters, limit=limit)
    if not contests:
        rprint("[yellow]No contests found for the given filters.[/yellow]")
        return

    # Enrich with NLP
    enriched = enrich_contests_with_nlp(contests)

    # Print a summary report
    rprint(f"[bold cyan]NLP Analysis for {len(enriched)} filtered contests[/bold cyan]")
    for c in enriched:
        rprint({
            "id": c.get("id"),
            "title": c.get("title"),
            "entities": c.get("entities"),
            "locations": c.get("locations"),
            "dates": c.get("dates"),
        })

# Example usage:
if __name__ == "__main__":
    # Example: Find all 2024 contests in New York
    filters = {"year": 2024, "state": "New York"}
    analyze_filtered_contests(filters=filters, limit=10)

def load_sample_json(path=SAMPLE_JSON_PATH):
    if not os.path.exists(path):
        return {"samples": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_sample_json(data, path=SAMPLE_JSON_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def add_sample_entry(text, meta=None, path=SAMPLE_JSON_PATH):
    data = load_sample_json(path)
    entry = {"text": text}
    if meta:
        entry["meta"] = meta
    data.setdefault("samples", []).append(entry)
    save_sample_json(data, path)

def update_sample_entry(index, text=None, meta=None, path=SAMPLE_JSON_PATH):
    data = load_sample_json(path)
    if 0 <= index < len(data.get("samples", [])):
        if text is not None:
            data["samples"][index]["text"] = text
        if meta is not None:
            data["samples"][index]["meta"] = meta
        save_sample_json(data, path)
        return True
    return False

def get_sample_text(index=0, path=SAMPLE_JSON_PATH):
    data = load_sample_json(path)
    samples = data.get("samples", [])
    if samples and 0 <= index < len(samples):
        return samples[index]["text"]
    return None

# --- Database Access Utilities ---

def fetch_contests_from_db(limit=100, filters=None):
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

def coordinator_integrity_pipeline(raw_html_context):
    # Step 1: Organize context (parse, deduplicate, cluster, etc.)
    organized = organize_context(raw_html_context)
    
    # Step 2: NLP & semantic validation
    contests = organized["contests"]
    nlp_enriched = enrich_contests_with_nlp(contests)
    
    # Step 3: Integrity checks (duplicates, missing info, suspicious entities)
    integrity_issues = election_integrity_checks(nlp_enriched)
    flagged = flag_suspicious_contests(nlp_enriched)
    
    # Step 4: Transparency report
    report = {
        "integrity_issues": integrity_issues,
        "flagged_suspicious": flagged,
        "anomalies": organized.get("anomalies", []),
        "clusters": organized.get("clusters", []),
    }
    rprint("[bold green]Election Integrity Report[/bold green]")
    rprint(report)
    return report

def correct_and_update_contest(contest_id, correction_data):
    # Fetch contest from organizer/DB
    contests = fetch_contests_from_db(filters={"id": contest_id})
    if contests:
        contest = contests[0]
        # Apply correction
        contest.update(correction_data)
        # Save back to organizer/DB (implement a save/update function in organizer)
        update_contest_in_db(contest)
        rprint(f"[green]Contest {contest_id} updated with correction.[/green]")

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
    # Define environment context for logging and traceability
    env_context = {
        "ENV": os.getenv("ENV"),
        "LOG_LEVEL": os.getenv("LOG_LEVEL"),
        "DB_PATH": DB_PATH,
        "from_db": from_db,
        "db_limit": db_limit,
        "filters": db_filters,
        "active_learning": enable_active_learning
    }
    log_info("Coordinator pipeline started", context=env_context)
    if from_db:
        contests = fetch_contests_from_db(limit=db_limit, filters=db_filters)
        log_info("Fetched contests from DB", context={"count": len(contests), **env_context})
        
    else:
        contests = raw_context.get("contests", []) if raw_context else []
        log_info("Fetched contests from raw context", context={"count": len(contests), **env_context})
    contests = enrich_contests_with_nlp(contests)
    log_info("NLP enrichment complete", context={"count": len(contests), **env_context})
    coordinator_report(contests, required_states=required_states, expected_year=expected_year)

    # Optionally run active learning/user feedback
    if enable_active_learning and uncertain_idxs:
        feedback = active_learning_feedback(contests, uncertain_idxs, prompt_func=prompt_func)
        rprint(f"[ACTIVE LEARNING] Feedback received: {feedback}")
        log_info("Active learning feedback received", context={"feedback": feedback, **env_context})

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
def robust_db_usage_example():
    env_context = {
    "ENV": os.getenv("ENV"),
    "LOG_LEVEL": os.getenv("LOG_LEVEL"),
    "DB_PATH": DB_PATH,
    "from_db": True,
    "db_limit": 20,
    "filters": {"year": 2024, "state": "New York"},
    "active_learning": False
    }
    # 1. Fetch contests from DB with flexible filters
    filters = {"year": 2024, "state": "New York"}  # Example: all 2024 NY contests
    contests = fetch_contests_by_filter(filters=filters, limit=20)
    if not contests:
        rprint("[yellow]No contests found for the given filters.[/yellow]")
        return

    # 2. Enrich with NLP
    enriched = enrich_contests_with_nlp(contests)

    # 3. Flag suspicious contests
    flagged = flag_suspicious_contests(enriched)
    if flagged:
        rprint(f"[red]Flagged suspicious contests:[/red]")
        for entry in flagged:
            rprint(entry)

    # 4. Print a transparency/integrity report
    rprint("[bold cyan]Transparency/Integrity Report[/bold cyan]")
    for c in enriched:
        rprint({
            "id": c.get("id"),
            "title": c.get("title"),
            "entities": c.get("entities"),
            "locations": c.get("locations"),
            "dates": c.get("dates"),
        })

    # 5. Example: Correct and update a contest (simulate correction)
    if enriched:
        contest_to_update = enriched[0]
        correction = {"title": contest_to_update["title"] + " (Corrected)"}
        contest_to_update.update(correction)
    try:
        update_contest_in_db(contest_to_update)
        log_info("Contest updated in DB", context={"contest_id": contest_to_update["id"], **env_context})       
        rprint(f"[green]Contest {contest_to_update['id']} updated with correction.[/green]")
    except Exception as e:
        log_warning("High number of suspicious contests", context={"flagged_count": len(flagged), **env_context})
def orchestrator_cli():
    rprint("[bold cyan]Election Context Orchestrator CLI[/bold cyan]")
    # 1. Get filters from user
    year = input("Enter year (or leave blank): ").strip()
    state = input("Enter state (or leave blank): ").strip()
    filters = {}
    if year:
        filters["year"] = int(year)
    if state:
        filters["state"] = state

    # 2. Fetch contests
    contests = fetch_contests_by_filter(filters=filters, limit=20)
    if not contests:
        rprint("[yellow]No contests found for the given filters.[/yellow]")
        return

    # 3. NLP enrichment
    enriched = enrich_contests_with_nlp(contests)

    # 4. Flag suspicious
    flagged = flag_suspicious_contests(enriched)
    if flagged:
        rprint(f"[red]Flagged suspicious contests:[/red]")
        for entry in flagged:
            rprint(entry)

    # 5. Print transparency/integrity report
    rprint("[bold cyan]Transparency/Integrity Report[/bold cyan]")
    for c in enriched:
        rprint({
            "id": c.get("id"),
            "title": c.get("title"),
            "entities": c.get("entities"),
            "locations": c.get("locations"),
            "dates": c.get("dates"),
        })

    # 6. Optionally update a contest
    update = input("Update a contest? Enter ID or leave blank: ").strip()
    if update:
        for c in enriched:
            if str(c.get("id")) == update:
                new_title = input(f"New title for contest {update} (leave blank to skip): ").strip()
                if new_title:
                    c["title"] = new_title
                    update_contest_in_db(c)
                    rprint(f"[green]Contest {update} updated.[/green]")
                break
            
if __name__ == "__main__":
    # Ensure at least one sample exists
    if not os.path.exists(SAMPLE_JSON_PATH):
        add_sample_entry(
            "The 2024 election in Rockland, New York was held on November 5th, 2024. Contact info:",
            meta={"source": "default"}
        )

    # Get the first sample from sample.json
    sample_text = get_sample_text(0)
    if sample_text:
        demo_analysis(sample_text)  # This calls the function from spacy_utils.py
    else:
        rprint("[red]No sample text found in sample.json[/red]")
    robust_db_usage_example()
    orchestrator_cli()