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
from ..config import CONTEXT_LIBRARY_PATH, BASE_DIR
import importlib
import shutil
import difflib
import datetime
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("manual_correction_bot")

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

SEGMENT_FEEDBACK_LOG = Path(BASE_DIR).parent / "log" / "segment_feedback_log.jsonl"
PATTERN_KB_FILE = Path(BASE_DIR).parent / "log" / "dom_pattern_kb.jsonl"
DOWNLOAD_LINKS_LOG = Path(BASE_DIR).parent / "log" / "download_links_log.jsonl"
ANOMALY_LOG = Path(BASE_DIR).parent / "log" / "anomaly_log.jsonl"
EXPORT_DIR = Path(BASE_DIR).parent / "log" / "correction_exports"
SAFE_DIR = Path(BASE_DIR).resolve()
DEFAULT_CONTEXT_LIBRARY_FILE = Path(CONTEXT_LIBRARY_PATH)
LOG_DIR = Path(BASE_DIR).parent / "log"
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
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load context library: {e}")
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
                context = entry.get("context", {})
                context_key = (
                    context.get("contest_title")
                    or context.get("state")
                    or context.get("county")
                    or context.get("field_name")
                    or entry.get("field_name")
                    or context.get("url")
                    or "unknown"
                )
                # Fallback: if still empty, use a hash of the entry
                if not context_key or not str(context_key).strip():
                    context_key = f"entry_{hash(str(entry))[:8]}"
                field_entries[context_key].append(entry)
    return field_entries

def spacy_feedback_on_entry(entry, args=None):
    """
    Use spaCy to analyze and provide feedback on an entry for context awareness/self-improvement.
    """
    if not nlp:
        logger.warning("spaCy not available. Some features will be disabled.")
        return {}
    if args is not None and getattr(args, "llm_api_key", None) is None:
        logger.info("No LLM API key provided. LLM suggestions will be skipped.")
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
            if isinstance(value, str):
                value = value.strip()
            if not value or (isinstance(value, str) and not value.strip()):
                continue 
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
                if value not in library[field_type][context_key]:
                    library[field_type][context_key].append(value)
                    new_entries[context_key].append(value)
            else:
                if value not in library[field_type]:
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
    llm_provider="openai", # Or set for Anthropic specific information
    llm_model="gpt-4-turbo",
    llm_system_prompt=None,
    llm_extra_instructions=None,
    args=None,
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

    print(f"\n[FEEDBACK] Review new context library entries for [bold]{field_type}[/bold]:")
    context_library = load_context_library(context_library_path)
    changed = False
    accepted, edited, removed = 0, 0, 0
    undo_stack = []

    skip_all = False
    accept_all = False

    for context_key, values in new_entries.items():
        if skip_all or accept_all:
            break
        print(f"\nContext: {context_key}")
        for idx, val in enumerate(values):
            if skip_all:
                break
            if accept_all:
                print(f"    [ACCEPTED] {val!r}")
                accepted += 1
                continue
            print(f"  {idx}: {val!r}")
            ml_score = ml_score_entry({"extracted_value": val}, coordinator=coordinator)
            if enhanced and nlp:
                analysis = spacy_feedback_on_entry({"extracted_value": val}, args=args)
                if analysis.get("entities"):
                    print(f"    [spaCy entities]: {analysis['entities']}")
                if analysis.get("suggestions"):
                    print(f"    [spaCy suggestions]: {analysis['suggestions']}")
            print(f"    [ML score]: {ml_score:.2f}")
            resp = input(
                "    [A]ccept / [E]dit / [R]emove / [B]atch Edit / [D]ebate / [S]uggest / [L]LM / [C]ancel / [S]kip all / [Y] Accept all / [U]ndo? (A/E/R/B/D/S/L/C/S/Y/U): "
            ).strip().lower()
            if resp == "c":
                print("[INFO] Cancelled by user. Exiting feedback loop.")
                return
            if resp == "s":
                skip_all = True
                break
            if resp == "y":
                accept_all = True
                accepted += 1
                continue
            if resp == "u" and undo_stack:
                last_action = undo_stack.pop()
                # Implement undo logic here
                print("[INFO] Undo last action (not fully implemented).")
                continue
            if resp == "b":
                batch_val = input("    Enter new value for all similar entries: ").strip()
                for i, v in enumerate(values):
                    if v == val:
                        values[i] = batch_val
                        edited += 1
                changed = True
                continue
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
                    edited += 1
                    changed = True
            elif resp == "r":
                if field_type in {"buttons", "panels", "tables"}:
                    context_library[field_type][context_key].remove(val)
                else:
                    context_library[field_type].remove(val)
                print("    [REMOVED]")
                removed += 1
                changed = True
            else:
                print("    [ACCEPTED]")
                accepted += 1
    if changed:
        save_context_library(context_library, context_library_path)
        print(f"[INFO] Context library updated with feedback for {field_type}.")
    print(f"[SUMMARY] Accepted: {accepted}, Edited: {edited}, Removed: {removed}")

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

def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def save_jsonl(path, entries):
    path = safe_path(path)
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# 1. Segment Feedback Log Review
def review_segment_feedback(log_path=SEGMENT_FEEDBACK_LOG, pattern_kb_path=PATTERN_KB_FILE):
    entries = load_jsonl(log_path)
    if not entries:
        print("[INFO] No segment feedback log entries found.")
        return
    pattern_kb = load_jsonl(pattern_kb_path) if os.path.exists(pattern_kb_path) else []
    changed = False
    for idx, entry in enumerate(entries):
        print(f"\n[{idx}] Pattern ID: {entry.get('pattern_id')}")
        print(f"    Label: {entry.get('label')}")
        html_preview = entry.get('html', '')[:120].replace('\n', ' ')
        print(f"    HTML: {html_preview}{'...' if len(entry.get('html', '')) > 120 else ''}")
        resp = input("    [A]ccept / [E]dit / [R]emove / [L]LM Suggest / [B]atch / [S]kip? (A/E/R/L/B/S): ").strip().lower()
        if resp == "e":
            new_label = input("      New label: ").strip()
            if new_label:
                entry["label"] = new_label
                changed = True
                # Auto-apply to pattern KB if pattern_id matches
                for pat in pattern_kb:
                    if pat.get("pattern_id") == entry.get("pattern_id"):
                        pat["label"] = new_label
                # Batch correction for similar HTML
                batch = input("      Apply to all similar HTML segments? (y/N): ").strip().lower()
                if batch == "y":
                    for e in entries:
                        if e is not entry and difflib.SequenceMatcher(None, e.get("html", ""), entry.get("html", "")).ratio() > 0.9:
                            e["label"] = new_label
        elif resp == "r":
            entries[idx] = None
            changed = True
        elif resp == "l":
            suggestion = llm_suggest_action(entry)
            print(f"      [LLM Suggestion]: {suggestion}")
        elif resp == "b":
            # Batch correction for similar pattern_id
            new_label = input("      New label for all with this pattern_id: ").strip()
            if new_label:
                for e in entries:
                    if e.get("pattern_id") == entry.get("pattern_id"):
                        e["label"] = new_label
                for pat in pattern_kb:
                    if pat.get("pattern_id") == entry.get("pattern_id"):
                        pat["label"] = new_label
                changed = True
        elif resp == "s":
            continue
        # Accept does nothing
    # Remove deleted
    entries = [e for e in entries if e]
    if changed:
        save_jsonl(log_path, entries)
        save_jsonl(pattern_kb_path, pattern_kb)
        print("[INFO] Segment feedback log and pattern KB updated.")

# 2. Pattern KB Editing
def review_pattern_kb(pattern_kb_path=PATTERN_KB_FILE):
    kb = load_jsonl(pattern_kb_path)
    if not kb:
        print("[INFO] No pattern KB entries found.")
        return
    for idx, pat in enumerate(kb):
        print(f"\n[{idx}] Pattern ID: {pat.get('pattern_id')}")
        print(f"    Label: {pat.get('label')}")
        print(f"    Features: {pat.get('features', {})}")
        resp = input("    [E]dit / [R]emove / [M]erge / [L]LM Suggest / [S]kip? (E/R/M/L/S): ").strip().lower()
        if resp == "e":
            new_label = input("      New label: ").strip()
            if new_label:
                pat["label"] = new_label
        elif resp == "r":
            kb[idx] = None
        elif resp == "m":
            merge_idx = input("      Merge with pattern index: ").strip()
            if merge_idx.isdigit() and 0 <= int(merge_idx) < len(kb):
                merge_pat = kb[int(merge_idx)]
                if merge_pat:
                    # Merge features and labels
                    merge_pat["features"].update(pat.get("features", {}))
                    merge_pat["label"] = pat.get("label", merge_pat.get("label"))
                    kb[idx] = None
        elif resp == "l":
            suggestion = llm_suggest_action(pat)
            print(f"      [LLM Suggestion]: {suggestion}")
        elif resp == "s":
            continue
    kb = [p for p in kb if p]
    save_jsonl(pattern_kb_path, kb)
    print("[INFO] Pattern KB updated.")

# 3. Download Links Feedback Loop
def review_download_links(log_path=DOWNLOAD_LINKS_LOG, context_library_path=CONTEXT_LIBRARY_PATH):
    entries = load_jsonl(log_path)
    if not entries:
        print("[INFO] No download links log entries found.")
        return
    context_library = load_context_library(context_library_path)
    changed = False
    for idx, entry in enumerate(entries):
        print(f"\n[{idx}] URL: {entry.get('url')}")
        print(f"    Label: {entry.get('label')}")
        print(f"    Format: {entry.get('format')}")
        resp = input("    [A]ccept / [E]dit / [R]emove / [L]LM Suggest / [S]kip? (A/E/R/L/S): ").strip().lower()
        if resp == "e":
            new_label = input("      New label: ").strip()
            if new_label:
                entry["label"] = new_label
                changed = True
        elif resp == "r":
            entries[idx] = None
            changed = True
        elif resp == "l":
            suggestion = llm_suggest_action(entry)
            print(f"      [LLM Suggestion]: {suggestion}")
        elif resp == "s":
            continue
        # Accept does nothing
        # Optionally, auto-apply to context library
        if resp in ("a", "e"):
            if "download_links" not in context_library:
                context_library["download_links"] = []
            if entry not in context_library["download_links"]:
                context_library["download_links"].append(entry)
                changed = True
    entries = [e for e in entries if e]
    save_jsonl(log_path, entries)
    if changed:
        save_context_library(context_library, context_library_path)
        print("[INFO] Download links log and context library updated.")

# 4. Anomaly Review Integration
def review_anomalies(anomaly_log_path=ANOMALY_LOG, context_library_path=CONTEXT_LIBRARY_PATH):
    entries = load_jsonl(anomaly_log_path)
    if not entries:
        print("[INFO] No anomaly log entries found.")
        return
    context_library = load_context_library(context_library_path)
    for idx, entry in enumerate(entries):
        print(f"\n[{idx}] Anomaly: {entry.get('anomaly')}")
        print(f"    Context: {entry.get('context')}")
        print(f"    Details: {entry.get('details')}")
        resp = input("    [R]esolve / [E]dit / [L]LM Suggest / [S]kip? (R/E/L/S): ").strip().lower()
        if resp == "r":
            entry["resolved"] = True
        elif resp == "e":
            new_details = input("      New details: ").strip()
            if new_details:
                entry["details"] = new_details
        elif resp == "l":
            suggestion = llm_suggest_action(entry)
            print(f"      [LLM Suggestion]: {suggestion}")
        elif resp == "s":
            continue
        # Optionally, update context library with resolution
        if resp == "r":
            if "anomalies" not in context_library:
                context_library["anomalies"] = []
            context_library["anomalies"].append(entry)
    save_jsonl(anomaly_log_path, entries)
    save_context_library(context_library, context_library_path)
    print("[INFO] Anomaly log and context library updated.")

# 8. Export/Import Correction Sessions
def export_correction_session(log_paths, export_dir=EXPORT_DIR):
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_files = []
    for path in log_paths:
        if os.path.exists(path):
            dest = export_dir / (Path(path).stem + f"_{timestamp}.jsonl")
            shutil.copy2(path, dest)
            export_files.append(str(dest))
    print(f"[INFO] Exported correction session logs to: {export_files}")

def import_correction_session(import_file, dest_path):
    if not os.path.exists(import_file):
        print(f"[ERROR] Import file not found: {import_file}")
        return
    shutil.copy2(import_file, dest_path)
    print(f"[INFO] Imported correction session from {import_file} to {dest_path}")

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
    parser.add_argument("--filter-context-key", type=str, help="Only process entries with this context key substring")
    parser.add_argument("--filter-value", type=str, help="Only process entries containing this value substring")
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

    
def extended_cli():
    parser = argparse.ArgumentParser(description="Manual Correction Bot - Extended CLI")
    parser.add_argument("--review-segments", action="store_true", help="Review segment feedback logs")
    parser.add_argument("--review-pattern-kb", action="store_true", help="Review/edit DOM pattern KB")
    parser.add_argument("--review-download-links", action="store_true", help="Review download links logs")
    parser.add_argument("--review-anomalies", action="store_true", help="Review anomaly logs")
    parser.add_argument("--export-session", nargs="+", help="Export correction session logs (provide paths)")
    parser.add_argument("--import-session", nargs=2, metavar=("IMPORT_FILE", "DEST_PATH"), help="Import correction session log")
    args, unknown = parser.parse_known_args()

    if args.review_segments:
        review_segment_feedback()
        return
    if args.review_pattern_kb:
        review_pattern_kb()
        return
    if args.review_download_links:
        review_download_links()
        return
    if args.review_anomalies:
        review_anomalies()
        return
    if args.export_session:
        export_correction_session(args.export_session)
        return
    if args.import_session:
        import_correction_session(args.import_session[0], args.import_session[1])
        return
    # If none of the above, fall back to main
    main()
# --- PATCH: Replace main entry point ---
if __name__ == "__main__":
    extended_cli()