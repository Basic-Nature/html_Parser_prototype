# tools/url_hint_manager.py
# ==============================================================
# UI and CLI tool to manage and validate URL_HINT_OVERRIDES.txt entries
# against live importable handler modules.
# ==============================================================
import importlib
import json
import os
from difflib import get_close_matches

HINT_FILE = os.path.join(os.path.dirname(__file__), ".url_hint_overrides.txt")


# Load the URL_HINT_OVERRIDES from a local .txt file
# If missing, return an empty dict and notify the user
def load_overrides():
    try:
        with open(HINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print("[INFO] No overrides file found. Creating new one...")
        return {}
    except Exception as e:
        print(f"[ERROR] Failed to read {HINT_FILE}: {e}")
        return {}


# Attempt to import the handler module listed for a given URL fragment
# Print out suggestions if the module path appears broken
def validate_entry(url_fragment, module_path):
    try:
        importlib.import_module(module_path)
        return True
    except ModuleNotFoundError:
        print(f"[INVALID] {url_fragment} → {module_path} (module not found)")
        parent = ".".join(module_path.split(".")[:-1])
        base = module_path.split(".")[-1]
        try:
            pkg = importlib.import_module(parent)
            options = dir(pkg)
            suggestion = get_close_matches(base, options, n=1, cutoff=0.6)
            if suggestion:
                print(f"    Suggest: {parent}.{suggestion[0]}")
        except Exception:
            pass
        return False


# Prompt the user to enter a new override entry manually
def interactive_add_entry(overrides):
    frag = input("Enter URL fragment (e.g. electionreturns.pa.gov): ").strip()
    path = input("Enter module path (e.g. handlers.states.pennsylvania): ").strip()
    if frag and path:
        overrides[frag] = path
        print(f"[ADDED] {frag} → {path}")


# Main interface to list, validate, and edit override entries
def run_manager():
    print("=== URL Hint Override Manager ===")
    overrides = load_overrides()

    print("\n[VALIDATION RESULTS]")
    for url, path in overrides.items():
        validate_entry(url, path)

    while True:
        choice = input("\n[A]dd  [S]ave  [Q]uit: ").strip().lower()
        if choice == "a":
            interactive_add_entry(overrides)
        elif choice == "s":
            with open(HINT_FILE, "w", encoding="utf-8") as f:
                json.dump(overrides, f, indent=2)
            print(f"[SAVED] {len(overrides)} entries written to {HINT_FILE}")
        elif choice == "q":
            break


if __name__ == "__main__":
    run_manager()
