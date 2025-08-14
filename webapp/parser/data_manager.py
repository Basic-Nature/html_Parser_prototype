import os
import re
from .config import INPUT_DIR, OUTPUT_DIR, URL_LIST_FILE
from .utils.logger_singleton import logger, console, prompt

URL_LINE_RE = re.compile(r'^\s*(?P<url>(https?://|ftp://|www\.)[^\s#]+)')

def _ensure_parent(path):
    """Ensure parent directory exists (accepts str or Path)."""
    if hasattr(path, "parent"):
        os.makedirs(path.parent, exist_ok=True)
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)

def _atomic_write_lines(path, lines: list[str]):
    path = os.fspath(path)
    _ensure_parent(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for ln in lines:
            f.write(ln.rstrip() + "\n")
    os.replace(tmp, path)

def load_urls() -> list[str]:
    if not os.path.exists(URL_LIST_FILE):
        return []
    urls: list[str] = []
    try:
        with open(URL_LIST_FILE, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                m = URL_LINE_RE.match(s)
                urls.append(m.group("url") if m else s)
    except Exception as e:
        logger.error(f"[ERROR] Reading urls.txt failed: {e}")
    return urls

def save_urls(urls: list[str]) -> None:
    clean = []
    seen = set()
    for u in urls:
        if not isinstance(u, str):
            continue
        s = u.strip()
        if not s:
            continue
        key = s.lower()
        if key not in seen:
            seen.add(key)
            clean.append(s)
    try:
        _atomic_write_lines(URL_LIST_FILE, clean)
        logger.info(f"[SAVED] {len(clean)} URLs to {URL_LIST_FILE.name}")
    except Exception as e:
        logger.error(f"[ERROR] Writing urls.txt failed: {e}")

def add_url(url: str) -> bool:
    if not isinstance(url, str):
        return False
    u = url.strip()
    if not u:
        return False
    urls = load_urls()
    if any(u.lower() == existing.lower() for existing in urls):
        logger.info(f"[SKIP] Duplicate URL: {u}")
        return False
    urls.append(u)
    save_urls(urls)
    logger.info(f"[ADDED] {u}")
    return True

def remove_url(index_or_value) -> bool:
    urls = load_urls()
    if not urls:
        return False
    removed = False
    if isinstance(index_or_value, int):
        if 0 <= index_or_value < len(urls):
            popped = urls.pop(index_or_value)
            logger.warning(f"[REMOVED] {popped}")
            removed = True
    else:
        target = str(index_or_value).strip().lower()
        new_urls = [u for u in urls if u.lower() != target]
        if len(new_urls) != len(urls):
            urls = new_urls
            logger.warning(f"[REMOVED] {index_or_value}")
            removed = True
    if removed:
        save_urls(urls)
    return removed

def replace_urls(new_urls: list[str]) -> None:
    save_urls(new_urls)

def list_urls_cli() -> list[str]:
    urls = load_urls()
    if not urls:
        logger.info("[INFO] No URLs in urls.txt")
        return []
    logger.info(f"\n[{URL_LIST_FILE.name}]")
    for i, u in enumerate(urls, 1):
        logger.info(f"{i}. {u}")
    return urls

def list_files(folder, allow_delete=False):
    folder = os.fspath(folder)
    logger.info(f"\n[{os.path.basename(folder).upper()} FILES]")
    try:
        files = sorted(os.listdir(folder))
    except FileNotFoundError:
        logger.info("  (missing)")
        return
    if not files:
        logger.info("  (empty)")
        return
    for i, f in enumerate(files, 1):
        logger.info(f"{i}. {f}")
    if allow_delete:
        choice = prompt.prompt_input("Delete file # (blank=skip): ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(files):
                try:
                    os.remove(os.path.join(folder, files[idx]))
                    logger.warning(f"[DELETED] {files[idx]}")
                except Exception as e:
                    logger.error(f"[ERROR] Delete failed: {e}")

def copy_file_to_folder(src_path: str, dest_folder):
    dest_folder = os.fspath(dest_folder)
    if not os.path.isfile(src_path):
        logger.error("[ERROR] Source file not found.")
        return
    _ensure_parent(dest_folder)
    dest = os.path.join(dest_folder, os.path.basename(src_path))
    try:
        with open(src_path, "rb") as s, open(dest, "wb") as d:
            d.write(s.read())
        logger.info(f"[COPIED] {src_path} → {dest}")
    except Exception as e:
        logger.error(f"[ERROR] Copy failed: {e}")

def run_manager():
    console.panel("=== Data Management CLI ===", title="Menu", style="green")
    while True:
        menu = (
            "\nOptions:\n"
            " 1. List urls.txt\n"
            " 2. Add URL\n"
            " 3. Remove URL (by number)\n"
            " 4. Replace entire URL list (comma separated)\n"
            " 5. List input folder files\n"
            " 6. List output folder files\n"
            " 7. Copy file to input folder\n"
            " 8. Copy file to output folder\n"
            " 9. Delete file from input folder\n"
            "10. Delete file from output folder\n"
            " Q. Quit"
        )
        console.panel(menu, title="Options", style="cyan")
        choice = prompt.prompt_input("Select: ").strip().lower()
        if choice == "1":
            list_urls_cli()
        elif choice == "2":
            url = prompt.prompt_input("URL: ").strip()
            add_url(url)
        elif choice == "3":
            urls = list_urls_cli()
            if urls:
                sel = prompt.prompt_input("Number to remove: ").strip()
                if sel.isdigit():
                    remove_url(int(sel) - 1)
        elif choice == "4":
            raw = prompt.prompt_input("Enter URLs separated by commas: ")
            new_urls = [s.strip() for s in raw.split(",") if s.strip()]
            replace_urls(new_urls)
        elif choice == "5":
            list_files(INPUT_DIR)
        elif choice == "6":
            list_files(OUTPUT_DIR)
        elif choice == "7":
            src = prompt.prompt_input("Path to file: ").strip()
            copy_file_to_folder(src, INPUT_DIR)
        elif choice == "8":
            src = prompt.prompt_input("Path to file: ").strip()
            copy_file_to_folder(src, OUTPUT_DIR)
        elif choice == "9":
            list_files(INPUT_DIR, allow_delete=True)
        elif choice == "10":
            list_files(OUTPUT_DIR, allow_delete=True)
        elif choice == "q":
            break

if __name__ == "__main__":
    run_manager()