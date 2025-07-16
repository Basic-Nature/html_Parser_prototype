import importlib
import orjson
import os
from difflib import get_close_matches
from .config import BASE_DIR
from .utils.shared_logger import SharedLogger, RichConsoleProxy
from webapp.parser.utils.user_prompt import UserPrompt
PARSER_DIR = os.path.join(os.path.dirname(__file__))
INPUT_FOLDER = os.path.join(BASE_DIR, "input")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")
HINT_FILE = os.path.join(PARSER_DIR, "url_hint_overrides.txt")
URLS_FILE = os.path.join(PARSER_DIR, "urls.txt")
logger = SharedLogger()
console = RichConsoleProxy()
prompt = UserPrompt()
def load_overrides():
    try:
        with open(HINT_FILE, "rb") as f:
            return orjson.loads(f.read())
    except FileNotFoundError:
        logger.info("[INFO] No overrides file found. Creating new one...")
        return {}
    except Exception as e:
        logger.error(f"[ERROR] Failed to read {HINT_FILE}: {e}")
        return {}

def save_overrides(overrides):
    with open(HINT_FILE, "wb") as f:
        f.write(orjson.dumps(overrides))
    logger.info(f"[SAVED] {len(overrides)} entries written to {HINT_FILE}")

def validate_entry(url_fragment, module_path):
    try:
        importlib.import_module(module_path)
        return True
    except ModuleNotFoundError:
        logger.warning(f"[INVALID] {url_fragment} → {module_path} (module not found)")
        parent = ".".join(module_path.split(".")[:-1])
        base = module_path.split(".")[-1]
        try:
            pkg = importlib.import_module(parent)
            options = dir(pkg)
            suggestion = get_close_matches(base, options, n=1, cutoff=0.6)
            if suggestion:
                logger.info(f"    Suggest: {parent}.{suggestion[0]}")
        except Exception:
            pass
        return False

def interactive_add_override(overrides):
    frag = input("Enter URL fragment (e.g. electionreturns.pa.gov): ").strip()
    path = input("Enter module path (e.g. handlers.states.pennsylvania): ").strip()
    if frag and path:
        overrides[frag] = path
        logger.info(f"[ADDED] {frag} → {path}")

def list_urls():
    if not os.path.exists(URLS_FILE):
        logger.info("[INFO] No urls.txt found.")
        return []
    with open(URLS_FILE, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]
    logger.info("\n[URLS.TXT ENTRIES]")
    for i, url in enumerate(urls, 1):
        logger.info(f"{i}. {url}")
    return urls

def add_url():
    url = input("Enter new URL to add: ").strip()
    if url:
        with open(URLS_FILE, "a", encoding="utf-8") as f:
            f.write(url + "\n")
        logger.info(f"[ADDED] {url}")

def list_files(folder, allow_delete=False):
    logger.info(f"\n[{os.path.basename(folder).upper()} FOLDER FILES]")
    files = os.listdir(folder)
    if not files:
        logger.info("  (empty)")
        return
    for i, fname in enumerate(files, 1):
        logger.info(f"{i}. {fname}")
    if allow_delete and files:
        choice = input("Delete a file? Enter number or leave blank: ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(files):
                os.remove(os.path.join(folder, files[idx]))
                logger.warning(f"[DELETED] {files[idx]}")

def copy_file_to_folder(src_path, dest_folder):
    if not os.path.isfile(src_path):
        logger.error("[ERROR] File does not exist.")
        return
    dest_path = os.path.join(dest_folder, os.path.basename(src_path))
    with open(src_path, "rb") as src, open(dest_path, "wb") as dst:
        dst.write(src.read())
    logger.info(f"[COPIED] {src_path} → {dest_path}")

def run_manager():
    console.panel("=== Data Management CLI ===", title="Menu", style="green")
    while True:
        menu = (
            "\nOptions:\n"
            " 1. List/validate URL hint overrides\n"
            " 2. Add URL hint override\n"
            " 3. Save URL hint overrides\n"
            " 4. List urls.txt entries\n"
            " 5. Add URL to urls.txt\n"
            " 6. List input folder files\n"
            " 7. List output folder files\n"
            " 8. Copy file to input folder\n"
            " 9. Copy file to output folder\n"
            "10. Delete file from input folder\n"
            "11. Delete file from output folder\n"
            " Q. Quit"
        )
        console.panel(menu, title="Options", style="cyan")
        choice = prompt.prompt_input("Select: ").strip().lower()
        if choice == "1":
            overrides = load_overrides()
            logger.info("\n[VALIDATION RESULTS]")
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
            src = prompt.prompt_input("Path to file to copy to input/: ").strip()
            copy_file_to_folder(src, INPUT_FOLDER)
        elif choice == "9":
            src = prompt.prompt_input("Path to file to copy to output/: ").strip()
            copy_file_to_folder(src, OUTPUT_FOLDER)
        elif choice == "10":
            list_files(INPUT_FOLDER, allow_delete=True)
        elif choice == "11":
            list_files(OUTPUT_FOLDER, allow_delete=True)
        elif choice == "q":
            break

if __name__ == "__main__":
    run_manager()