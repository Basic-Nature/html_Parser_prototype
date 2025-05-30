"""
manual_correction_bot.py

Deep ML/LLM-enhanced batch review and correction bot for all context fields.
- Reads all *_selection_log.jsonl logs produced by ContextCoordinator.
- Allows user to review, accept, edit, or remove corrections for any field.
- Updates the context library and (optionally) the DB.
- Integrates with integrity_check for anomaly/suspicion highlighting.
- Uses spaCy, ML, and external LLMs for advanced feedback, context awareness, and self-improvement.
- Can connect to ContextCoordinator and context_organizer for deeper learning and automation.
- Supports advanced debate/decision logic, including LLM-powered suggestions and process improvement.
"""
"""For Base ML without LLM integration, you can run:"""

# Only run the feedback loop for manual corrections:
# python -m webapp.parser.bots.manual_correction_bot --feedback --integrity --update-db --enhanced

#Auto-accept all new entries without feedback:
# python -m webapp.parser.bots.manual_correction_bot --auto --update-db --enhanced

"""For LLM integration, you can use OpenAI or Anthropic API keys."""
## python -m webapp.parser.bots.manual_correction_bot --feedback --llm-api-key sk-... --llm-provider openai --llm-model gpt-4-turbo
## python -m webapp.parser.bots.manual_correction_bot --feedback --llm-api-key <anthropic-key> --llm-provider anthropic --llm-model claude-3-opus-20240229

import json
import os
import sys
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import importlib

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None
# --- Optional: External LLM Integration (OpenAI, Anthropic, etc.) ---
def llm_suggest_action(
    entry,
    context=None,
    api_key=None,
    model="gpt-4-turbo",
    provider="openai",
    system_prompt=None,
    temperature=0.2,
    max_tokens=200,
    extra_instructions=None,
):
    """
    Use an external LLM (OpenAI, Anthropic, etc.) to suggest a field or correction for the entry.
    Supports multi-model and advanced prompt engineering.
    """
    prompt = (
        "You are an expert election data context classifier and corrector.\n"
        "Given the following extracted value from an election context, and the context dictionary, "
        "suggest the most appropriate field (e.g., year, state, candidate, contest, etc.), a confidence score (0-1), "
        "and, if possible, a correction or improvement. "
        "If the value is ambiguous, explain why and suggest a process improvement or flag for review.\n"
        f"Extracted value: '{entry.get('extracted_value', '')}'\n"
        f"Context: {json.dumps(context or {}, ensure_ascii=False)}\n"
    )
    if extra_instructions:
        prompt += f"\nAdditional instructions: {extra_instructions}\n"

    # System prompt for OpenAI/Anthropic
    system_prompt = system_prompt or (
        "You are a highly reliable, context-aware election data assistant. "
        "Always provide clear, actionable suggestions and flag ambiguous cases."
    )

    try:
        if provider == "openai":
            import openai # type: ignore
            openai.api_key = api_key or os.environ.get("OPENAI_API_KEY")
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = response.choices[0].message.content.strip()
            return content
        elif provider == "anthropic":
 
            import anthropic # type: ignore
            client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        else:
            print(f"[LLM] Unknown provider: {provider}")
            return None
    except Exception as e:
        print(f"[LLM] LLM suggestion failed ({provider}): {e}")
        return None

        print(f"[LLM] Unknown provider: {provider}")
        return None     
# --- ML/AI Integration ---
def ml_score_entry(entry, coordinator=None):
    """
    Use ML/NER or coordinator's ML model to score the entry for likely correctness.
    Returns a float score between 0 and 1.
    """
    text = entry.get("extracted_value", "")
    score = 0.0
    if coordinator and hasattr(coordinator, "score_entry"):
        try:
            score = coordinator.score_entry(text)
        except Exception:
            pass
    # Fallback: spaCy NER confidence
    if nlp and text:
        doc = nlp(str(text))
        if doc.ents:
            score += 0.2
        if any(ent.label_ in {"DATE", "GPE", "ORG"} for ent in doc.ents):
            score += 0.2
        if len(text.split()) > 2:
            score += 0.1
    return min(score, 1.0)

def ml_suggest_field(entry, coordinator=None):
    """
    Use ML/NER or coordinator to suggest a better field for the entry.
    """
    text = entry.get("extracted_value", "")
    if coordinator and hasattr(coordinator, "suggest_field"):
        try:
            return coordinator.suggest_field(text)
        except Exception:
            pass
    # Fallback: spaCy entity type
    if nlp and text:
        doc = nlp(str(text))
        if doc.ents:
            label_counts = defaultdict(int)
            for ent in doc.ents:
                label_counts[ent.label_] += 1
            if label_counts.get("DATE"):
                return "years"
            if label_counts.get("GPE") or label_counts.get("LOC"):
                return "states"
            if label_counts.get("PERSON"):
                return "candidate"
    return None

# --- Config ---
from ..config import CONTEXT_LIBRARY_PATH, BASE_DIR

SAFE_DIR = Path(BASE_DIR).resolve()
DEFAULT_CONTEXT_LIBRARY_FILE = Path(CONTEXT_LIBRARY_PATH)
LOG_DIR = Path(BASE_DIR) / "log"
FIELD_LOG_SUFFIX = "_selection_log.jsonl"
SUCCESS_RESULTS = {"pass", "fuzzy_pass", "manual_correction", "user_corrected"}
ALL_FIELDS = [
    "buttons", "panels", "tables", "contests", "districts", "states", "election_types", "years", "party", "candidate"
]

# --- Utility Functions ---

def safe_path(path, base_dir=SAFE_DIR):
    """Ensure the path is within the allowed base directory."""
    full_path = (base_dir / Path(path)).resolve()
    if not str(full_path).startswith(str(base_dir)):
        raise ValueError(f"Unsafe path detected: {full_path}")
    return full_path

def load_context_library(path):
    """Load the context library from a JSON file."""
    path = safe_path(path)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_context_library(library, path):
    """Save the context library to a JSON file."""
    path = safe_path(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(library, f, indent=2, ensure_ascii=False)


def find_log_files(log_dir=LOG_DIR):
    """Find all *_selection_log.jsonl files in the log directory."""
    log_dir = Path(log_dir)
    if not log_dir.exists():
        return []
    return list(log_dir.glob(f"*{FIELD_LOG_SUFFIX}"))

def aggregate_successful_field_entries(log_file: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Parse a field log file and aggregate successful entries.
    Returns: dict of {context_key: [entry_dict, ...]}
    """
    field_entries = defaultdict(list)
    if not os.path.exists(log_file):
        print(f"[WARN] Log file not found: {log_file}")
        return field_entries
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except Exception:
                continue
            result = str(entry.get("result", ""))
            if any(result.startswith(s) for s in SUCCESS_RESULTS):
                context_key = entry.get("context", {}).get("contest_title") \
                    or entry.get("context", {}).get("state") \
                    or entry.get("context", {}).get("county") \
                    or entry.get("context", {}).get("field_name") \
                    or entry.get("field_name") \
                    or "unknown"
                field_entries[context_key].append(entry)
    return field_entries

def spacy_feedback_on_entry(entry):
    """
    Use spaCy to analyze and provide feedback on an entry for context awareness/self-improvement.
    """
    if not nlp:
        return {}
    text = entry.get("extracted_value", "")
    doc = nlp(str(text))
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    suggestions = []
    if not entities and isinstance(text, str) and len(text.split()) > 1:
        suggestions.append("No entities detected; consider splitting or clarifying.")
    if any(label in {"GPE", "LOC"} for _, label in entities):
        suggestions.append("Location detected; verify if this is the intended context.")
    if any(label == "DATE" for _, label in entities):
        suggestions.append("Date detected; check if this is a year or a timestamp.")
    return {"entities": entities, "suggestions": suggestions}

def update_context_library_with_field(library, field_type, field_entries, coordinator=None, context_organizer=None):
    """
    Updates the library in-place with new field entries.
    Returns a dict of {context_key: [new_entries]} for feedback.
    Uses coordinator/context_organizer for enhanced learning if available.
    """
    new_entries = defaultdict(list)
    if field_type not in library:
        library[field_type] = {} if field_type in {"buttons", "panels", "tables"} else []
    for context_key, entries in field_entries.items():
        for entry in entries:
            value = entry.get("extracted_value")
            # Enhanced learning: use coordinator/context_organizer for validation or enrichment
            if coordinator and hasattr(coordinator, "validate_and_check_integrity"):
                # Example: Use integrity check for contests
                if field_type == "contests":
                    issues = coordinator.validate_and_check_integrity()
                    if issues.get("integrity_issues"):
                        print(f"[INTEGRITY] Issues found for {context_key}: {issues['integrity_issues']}")
            if context_organizer and field_type == "contests":
                # Example: Use context_organizer to enrich contest info
                enriched = context_organizer({"contests": [entry]})
                if enriched and "contests" in enriched:
                    value = enriched["contests"][0].get("title", value)
            if field_type in {"buttons", "panels", "tables"}:
                if context_key not in library[field_type]:
                    library[field_type][context_key] = []
                if value and value not in library[field_type][context_key]:
                    library[field_type][context_key].append(value)
                    new_entries[context_key].append(value)
            else:
                if value and value not in library[field_type]:
                    library[field_type].append(value)
                    new_entries[context_key].append(value)
    return new_entries

def feedback_loop(
    new_entries,
    field_type,
    context_library_path,
    enhanced=True,
    coordinator=None,
    llm_api_key=None,
    llm_provider="openai",
    llm_model="gpt-4-turbo",
    llm_system_prompt=None,
    llm_extra_instructions=None,
):
    """
    Interactive feedback loop: prompt user to confirm or correct new entries.
    Uses spaCy, ML, and LLM for context awareness and suggestions.
    Supports advanced debate/decision logic and multi-model LLM support.
    """
    if not new_entries:
        print(f"[INFO] No new entries to review for {field_type}.")
        return

    print(f"\n[FEEDBACK] Review new context library entries for [bold]{field_type}[/bold]:")
    context_library = load_context_library(context_library_path)
    changed = False

    for context_key, values in new_entries.items():
        print(f"\nContext: {context_key}")
        for idx, val in enumerate(values):
            print(f"  {idx}: {val!r}")
            ml_score = ml_score_entry({"extracted_value": val}, coordinator=coordinator)
            if enhanced and nlp:
                analysis = spacy_feedback_on_entry({"extracted_value": val})
                if analysis.get("entities"):
                    print(f"    [spaCy entities]: {analysis['entities']}")
                if analysis.get("suggestions"):
                    print(f"    [spaCy suggestions]: {analysis['suggestions']}")
            print(f"    [ML score]: {ml_score:.2f}")
            resp = input(
                "    [A]ccept / [E]dit / [R]emove / [D]ebate / [S]uggest / [L]LM / [C]ancel? (A/E/R/D/S/L/C): "
            ).strip().lower()
            if resp == "c":
                print("[INFO] Cancelled by user. Exiting feedback loop.")
                return
            if resp == "d":
                print("[DEBATE] Should this process be handled differently?")
                print(
                    "  1. Keep as is\n"
                    "  2. Move to another field\n"
                    "  3. Mark as ambiguous\n"
                    "  4. ML Suggestion\n"
                    "  5. LLM Suggestion\n"
                    "  6. LLM (Anthropic)\n"
                    "  7. Custom LLM prompt\n"
                    "  8. Cancel debate"
                )
                debate_choice = input("  Enter choice (1-8): ").strip()
                if debate_choice == "2":
                    new_field = input("    Enter new field type: ").strip()
                    print(f"    [INFO] Would move to field: {new_field} (not implemented in this demo)")
                elif debate_choice == "3":
                    print("    [INFO] Marked as ambiguous for further review.")
                elif debate_choice == "4":
                    suggested = ml_suggest_field({"extracted_value": val}, coordinator=coordinator)
                    print(f"    [ML SUGGESTION] This entry may belong in: {suggested or 'unknown'}")
                elif debate_choice == "5":
                    llm_suggestion = llm_suggest_action(
                        {"extracted_value": val},
                        context={"context_key": context_key},
                        api_key=llm_api_key,
                        model=llm_model,
                        provider="openai",
                        system_prompt=llm_system_prompt,
                        extra_instructions=llm_extra_instructions,
                    )
                    print(f"    [LLM SUGGESTION - OpenAI]: {llm_suggestion or 'No suggestion'}")
                elif debate_choice == "6":
                    llm_suggestion = llm_suggest_action(
                        {"extracted_value": val},
                        context={"context_key": context_key},
                        api_key=llm_api_key,
                        model="claude-3-opus-20240229",
                        provider="anthropic",
                        system_prompt=llm_system_prompt,
                        extra_instructions=llm_extra_instructions,
                    )
                    print(f"    [LLM SUGGESTION - Anthropic]: {llm_suggestion or 'No suggestion'}")
                elif debate_choice == "7":
                    custom_prompt = input("    Enter custom instructions for the LLM: ").strip()
                    llm_suggestion = llm_suggest_action(
                        {"extracted_value": val},
                        context={"context_key": context_key},
                        api_key=llm_api_key,
                        model=llm_model,
                        provider=llm_provider,
                        system_prompt=llm_system_prompt,
                        extra_instructions=custom_prompt,
                    )
                    print(f"    [LLM SUGGESTION - Custom]: {llm_suggestion or 'No suggestion'}")
                elif debate_choice == "8":
                    print("    [INFO] Debate cancelled.")
                continue
            if resp == "l":
                llm_suggestion = llm_suggest_action(
                    {"extracted_value": val},
                    context={"context_key": context_key},
                    api_key=llm_api_key,
                    model=llm_model,
                    provider=llm_provider,
                    system_prompt=llm_system_prompt,
                    extra_instructions=llm_extra_instructions,
                )
                print(f"    [LLM SUGGESTION]: {llm_suggestion or 'No suggestion'}")
                continue
            if resp == "s":
                suggested = ml_suggest_field({"extracted_value": val}, coordinator=coordinator)
                print(f"    [ML SUGGESTION] This entry may belong in: {suggested or 'unknown'}")
                continue
            if resp == "e":
                new_val = input("      New value: ").strip()
                if new_val:
                    if field_type in {"buttons", "panels", "tables"}:
                        context_library[field_type][context_key][idx] = new_val
                    else:
                        context_library[field_type][idx] = new_val
                    print("    [UPDATED]")
                    changed = True
            elif resp == "r":
                if field_type in {"buttons", "panels", "tables"}:
                    context_library[field_type][context_key].remove(val)
                else:
                    context_library[field_type].remove(val)
                print("    [REMOVED]")
                changed = True
            else:
                print("    [ACCEPTED]")
    if changed:
        save_context_library(context_library, context_library_path)
        print(f"[INFO] Context library updated with feedback for {field_type}.")

def highlight_anomalies(context_library, field_type):
    """
    Use integrity_check to highlight suspicious or anomalous entries for review.
    """
    try:
        from ..Context_Integration.Integrity_check import analyze_contest_titles, summarize_context_entities
    except ImportError:
        print("[WARN] Could not import integrity_check for anomaly highlighting.")
        return

    if field_type == "contests" and "contests" in context_library:
        contests = context_library["contests"]
        results = analyze_contest_titles(contests)
        if results.get("integrity_issues"):
            print("\n[INTEGRITY] Issues detected in contests:")
            for issue, contest in results["integrity_issues"]:
                print(f"  - {issue}: {contest}")
        if results.get("flagged_suspicious"):
            print("\n[SUSPICIOUS] Contests flagged as suspicious:")
            for c in results["flagged_suspicious"]:
                print(f"  - {c}")
        entity_summary = summarize_context_entities(contests)
        print("\n[ENTITY SUMMARY]:")
        for label, count in entity_summary.items():
            print(f"  {label}: {count}")

def update_database_with_context(library, db_path=None, enhanced=True, coordinator=None):
    """
    Update a database with the latest context library.
    Uses coordinator for advanced DB handling if available.
    """
    if not db_path:
        db_path = SAFE_DIR / "parser" / "Context_Integration" / "Context_Library" / "context_library_db.json"
    try:
        # If coordinator has a DB update method, use it
        if enhanced and coordinator and hasattr(coordinator, "update_db_with_context"):
            coordinator.update_db_with_context(library, db_path)
            print(f"[DB] Context library written to DB via coordinator at {db_path}")
        else:
            with open(db_path, "w", encoding="utf-8") as f:
                json.dump(library, f, indent=2, ensure_ascii=False)
            print(f"[DB] Context library written to DB at {db_path}")
    except Exception as e:
        print(f"[DB ERROR] Failed to update DB: {e}")

def auto_accept_all_new_entries(new_entries, field_type, context_library_path):
    """
    Automatically accept all new entries (for automation/batch mode).
    """
    if not new_entries:
        return
    context_library = load_context_library(context_library_path)
    changed = False
    for context_key, values in new_entries.items():
        for idx, val in enumerate(values):
            continue
    if changed:
        save_context_library(context_library, context_library_path)
        print(f"[INFO] Context library auto-accepted for {field_type}.")

def connect_to_coordinator():
    """
    Dynamically import and instantiate the ContextCoordinator if available.
    """
    try:
        module = importlib.import_module("webapp.parser.Context_Integration.context_coordinator")
        return getattr(module, "ContextCoordinator")()
    except Exception as e:
        print(f"[WARN] Could not connect to ContextCoordinator: {e}")
        return None

def connect_to_context_organizer():
    """
    Dynamically import and return the context_organizer function if available.
    """
    try:
        module = importlib.import_module("webapp.parser.Context_Integration.context_organizer")
        return getattr(module, "organize_context")
    except Exception as e:
        print(f"[WARN] Could not connect to context_organizer: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Deep ML/LLM-enhanced batch review and correction bot for all context fields.")
    parser.add_argument("--context", type=str, default=str(DEFAULT_CONTEXT_LIBRARY_FILE), help="Path to context_library.json")
    parser.add_argument("--log-dir", type=str, default=str(LOG_DIR), help="Directory containing *_selection_log.jsonl files")
    parser.add_argument("--dry-run", action="store_true", help="Show changes but do not write to context library")
    parser.add_argument("--feedback", action="store_true", help="Enable interactive feedback loop for new entries")
    parser.add_argument("--fields", type=str, nargs="*", default=ALL_FIELDS, help="Fields to process (default: all)")
    parser.add_argument("--integrity", action="store_true", help="Highlight anomalies using integrity_check")
    parser.add_argument("--update-db", action="store_true", help="Update the DB with the new context library after processing")
    parser.add_argument("--db-path", type=str, default=None, help="Path to DB file (if --update-db is set)")
    parser.add_argument("--auto", action="store_true", help="Automatically accept all new entries (no prompt)")
    parser.add_argument("--enhanced", action="store_true", help="Enable enhanced learning and automation (spaCy, coordinator, context_organizer)")
    parser.add_argument("--no-coordinator", action="store_true", help="Do not connect to ContextCoordinator")
    parser.add_argument("--no-organizer", action="store_true", help="Do not connect to context_organizer")
    parser.add_argument("--llm-api-key", type=str, default=None, help="API key for external LLM (e.g., OpenAI/Anthropic)")
    parser.add_argument("--llm-provider", type=str, default="openai", help="LLM provider: openai or anthropic")
    parser.add_argument("--llm-model", type=str, default="gpt-4-turbo", help="LLM model name")
    parser.add_argument("--llm-system-prompt", type=str, default=None, help="Custom system prompt for LLM")
    parser.add_argument("--llm-extra-instructions", type=str, default=None, help="Extra instructions for LLM prompt")
    args = parser.parse_args()

    coordinator = None
    context_organizer = None
    if args.enhanced and not args.no_coordinator:
        coordinator = connect_to_coordinator()
    if args.enhanced and not args.no_organizer:
        context_organizer = connect_to_context_organizer()

    try:
        print("Loading context library...")
        library = load_context_library(args.context)
        log_files = find_log_files(args.log_dir)
        print(f"Found {len(log_files)} field log files in {args.log_dir}.")

        all_new_entries = {}
        for log_file in log_files:
            field_type = log_file.name.replace(FIELD_LOG_SUFFIX, "")
            if field_type not in args.fields:
                continue
            print(f"\n[PROCESSING] {field_type} from {log_file.name}")
            field_entries = aggregate_successful_field_entries(log_file)
            new_entries = update_context_library_with_field(
                library, field_type, field_entries,
                coordinator=coordinator, context_organizer=context_organizer
            )
            all_new_entries[field_type] = new_entries

        if args.dry_run:
            print("[DRY RUN] The following new entries would be added:")
            for field_type, new_entries in all_new_entries.items():
                print(f"  {field_type}:")
                for context_key, values in new_entries.items():
                    print(f"    {context_key}:")
                    for val in values:
                        print(f"      - {val!r}")
            return

        save_context_library(library, args.context)
        print("Context library updated!")

        # Integrity check integration
        if args.integrity:
            for field_type in args.fields:
                highlight_anomalies(library, field_type)

        # Feedback loop or automation
        if args.feedback and not args.auto:
            for field_type, new_entries in all_new_entries.items():
                feedback_loop(
                    new_entries, field_type, args.context,
                    enhanced=args.enhanced,
                    coordinator=coordinator,
                    llm_api_key=args.llm_api_key,
                    llm_provider=args.llm_provider,
                    llm_model=args.llm_model,
                    llm_system_prompt=args.llm_system_prompt,
                    llm_extra_instructions=args.llm_extra_instructions,
                )
        elif args.auto:
            for field_type, new_entries in all_new_entries.items():
                auto_accept_all_new_entries(new_entries, field_type, args.context)

        # DB update hook
        if args.update_db:
            update_database_with_context(library, db_path=args.db_path, enhanced=args.enhanced, coordinator=coordinator)

        # Safe exit/cleanup
        if coordinator and hasattr(coordinator, "close"):
            coordinator.close()
        print("[INFO] Manual correction bot finished successfully.")

    except KeyboardInterrupt:
        print("[INFO] Manual correction bot cancelled by user.")
        if coordinator and hasattr(coordinator, "close"):
            coordinator.close()
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] {e}")
        if coordinator and hasattr(coordinator, "close"):
            coordinator.close()
        sys.exit(1)

if __name__ == "__main__":
    main()