## This script assumes your context library is a JSON file with a structure like:
"""
{
    "buttons": {
        "contest_title": [
            {"label": "Button Label", "selector": "Button Selector"}
        ]
    }
}
"""
## python update_context_library_from_logs.py

import json
import os
import sys
import argparse
from collections import defaultdict
from pathlib import Path

SAFE_DIR = Path(__file__).parent.resolve()
DEFAULT_LOG_FILE = SAFE_DIR / "button_selection_log.jsonl"
DEFAULT_CONTEXT_LIBRARY_FILE = SAFE_DIR / "context_library.json"
SUCCESS_RESULTS = {"pass", "fuzzy_pass", "manual_correction"}

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

def aggregate_successful_buttons(log_file):
    """
    Parse the log file and aggregate successful button selections.
    Returns: dict of {contest_title: [button_dict, ...]}
    """
    log_file = safe_path(log_file)
    contest_buttons = defaultdict(list)
    if not os.path.exists(log_file):
        print(f"[WARN] Log file not found: {log_file}")
        return contest_buttons
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except Exception:
                continue
            result = str(entry.get("result", ""))
            if any(result.startswith(s) for s in SUCCESS_RESULTS):
                contest = entry.get("contest_title")
                label = entry.get("button_label")
                selector = entry.get("selector")
                if contest and label and selector:
                    btn = {"label": label, "selector": selector}
                    if btn not in contest_buttons[contest]:
                        contest_buttons[contest].append(btn)
    return contest_buttons

def update_context_library_with_buttons(library, contest_buttons):
    """
    Updates the library in-place with new button selectors.
    Returns a dict of {contest_title: [new_buttons]} for feedback.
    """
    if "buttons" not in library:
        library["buttons"] = {}
    new_entries = defaultdict(list)
    for contest, buttons in contest_buttons.items():
        if contest not in library["buttons"]:
            library["buttons"][contest] = []
        for btn in buttons:
            if btn not in library["buttons"][contest]:
                library["buttons"][contest].append(btn)
                new_entries[contest].append(btn)
    return new_entries

def feedback_loop(new_entries, context_library_path):
    """
    Interactive feedback loop: prompt user to confirm or correct new entries.
    Updates the context library file if corrections are made.
    """
    if not new_entries:
        print("[INFO] No new entries to review.")
        return

    print("\n[FEEDBACK] Review new context library entries:")
    context_library = load_context_library(context_library_path)
    changed = False

    for contest, buttons in new_entries.items():
        print(f"\nContest: {contest}")
        for idx, btn in enumerate(buttons):
            print(f"  {idx}: label='{btn['label']}', selector='{btn['selector']}'")
            resp = input("    [A]ccept / [E]dit / [R]emove? (A/E/R): ").strip().lower()
            if resp == "e":
                new_label = input("      New label: ").strip()
                new_selector = input("      New selector: ").strip()
                if new_label:
                    btn["label"] = new_label
                if new_selector:
                    btn["selector"] = new_selector
                print("    [UPDATED]")
                changed = True
            elif resp == "r":
                context_library["buttons"][contest].remove(btn)
                print("    [REMOVED]")
                changed = True
            else:
                print("    [ACCEPTED]")
    if changed:
        save_context_library(context_library, context_library_path)
        print("[INFO] Context library updated with feedback.")

def main():
    parser = argparse.ArgumentParser(description="Auto-update context library from button selection logs with feedback learning loop.")
    parser.add_argument("--log", type=str, default=str(DEFAULT_LOG_FILE), help="Path to button_selection_log.jsonl")
    parser.add_argument("--context", type=str, default=str(DEFAULT_CONTEXT_LIBRARY_FILE), help="Path to context_library.json")
    parser.add_argument("--dry-run", action="store_true", help="Show changes but do not write to context library")
    parser.add_argument("--feedback", action="store_true", help="Enable interactive feedback loop for new entries")
    args = parser.parse_args()

    try:
        print("Loading context library...")
        library = load_context_library(args.context)
        print("Aggregating successful button selections from logs...")
        contest_buttons = aggregate_successful_buttons(args.log)
        print(f"Found {sum(len(v) for v in contest_buttons.values())} successful button selectors.")
        print("Updating context library...")
        new_entries = update_context_library_with_buttons(library, contest_buttons)
        if args.dry_run:
            print("[DRY RUN] The following new entries would be added:")
            for contest, buttons in new_entries.items():
                print(f"  {contest}:")
                for btn in buttons:
                    print(f"    - {btn}")
            return
        save_context_library(library, args.context)
        print("Context library updated!")
        if args.feedback:
            feedback_loop(new_entries, args.context)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()