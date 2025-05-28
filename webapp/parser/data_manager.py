import importlib
import json
import os
from difflib import get_close_matches
from .config import BASE_DIR

PARSER_DIR = os.path.join(os.path.dirname(__file__))
INPUT_FOLDER = os.path.join(BASE_DIR, "input")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")
HINT_FILE = os.path.join(PARSER_DIR, "url_hint_overrides.txt")
URLS_FILE = os.path.join(PARSER_DIR, "urls.txt")

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

def save_overrides(overrides):
    with open(HINT_FILE, "w", encoding="utf-8") as f:
        json.dump(overrides, f, indent=2)
    print(f"[SAVED] {len(overrides)} entries written to {HINT_FILE}")

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

def interactive_add_override(overrides):
    frag = input("Enter URL fragment (e.g. electionreturns.pa.gov): ").strip()
    path = input("Enter module path (e.g. handlers.states.pennsylvania): ").strip()
    if frag and path:
        overrides[frag] = path
        print(f"[ADDED] {frag} → {path}")

def list_urls():
    if not os.path.exists(URLS_FILE):
        print("[INFO] No urls.txt found.")
        return []
    with open(URLS_FILE, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]
    print("\n[URLS.TXT ENTRIES]")
    for i, url in enumerate(urls, 1):
        print(f"{i}. {url}")
    return urls

def add_url():
    url = input("Enter new URL to add: ").strip()
    if url:
        with open(URLS_FILE, "a", encoding="utf-8") as f:
            f.write(url + "\n")
        print(f"[ADDED] {url}")

def list_files(folder, allow_delete=False):
    print(f"\n[{os.path.basename(folder).upper()} FOLDER FILES]")
    files = os.listdir(folder)
    if not files:
        print("  (empty)")
        return
    for i, fname in enumerate(files, 1):
        print(f"{i}. {fname}")
    if allow_delete and files:
        choice = input("Delete a file? Enter number or leave blank: ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(files):
                os.remove(os.path.join(folder, files[idx]))
                print(f"[DELETED] {files[idx]}")

def copy_file_to_folder(src_path, dest_folder):
    if not os.path.isfile(src_path):
        print("[ERROR] File does not exist.")
        return
    dest_path = os.path.join(dest_folder, os.path.basename(src_path))
    with open(src_path, "rb") as src, open(dest_path, "wb") as dst:
        dst.write(src.read())
    print(f"[COPIED] {src_path} → {dest_path}")

def run_manager():
    print("=== Data Management CLI ===")
    while True:
        print("\nOptions:")
        print(" 1. List/validate URL hint overrides")
        print(" 2. Add URL hint override")
        print(" 3. Save URL hint overrides")
        print(" 4. List urls.txt entries")
        print(" 5. Add URL to urls.txt")
        print(" 6. List input folder files")
        print(" 7. List output folder files")
        print(" 8. Copy file to input folder")
        print(" 9. Copy file to output folder")
        print("10. Delete file from input folder")
        print("11. Delete file from output folder")
        print(" Q. Quit")
        choice = input("Select: ").strip().lower()
        if choice == "1":
            overrides = load_overrides()
            print("\n[VALIDATION RESULTS]")
            for url, path in overrides.items():
                validate_entry(url, path)
        elif choice == "2":
            overrides = load_overrides()
            interactive_add_override(overrides)
        elif choice == "3":
            overrides = load_overrides()
            save_overrides(overrides)
        elif choice == "4":
            list_urls()
        elif choice == "5":
            add_url()
        elif choice == "6":
            list_files(INPUT_FOLDER)
        elif choice == "7":
            list_files(OUTPUT_FOLDER)
        elif choice == "8":
            src = input("Path to file to copy to input/: ").strip()
            copy_file_to_folder(src, INPUT_FOLDER)
        elif choice == "9":
            src = input("Path to file to copy to output/: ").strip()
            copy_file_to_folder(src, OUTPUT_FOLDER)
        elif choice == "10":
            list_files(INPUT_FOLDER, allow_delete=True)
        elif choice == "11":
            list_files(OUTPUT_FOLDER, allow_delete=True)
        elif choice == "q":
            break

if __name__ == "__main__":
    run_manager()