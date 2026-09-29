import os
import re

from .config import INPUT_DIR, OUTPUT_DIR, URL_LIST_FILE
from .utils.logger_singleton import console, logger, prompt

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

def _url_mutation_unavailable(operation: str) -> bool:
    logger.error(
        "[DISABLED] URL "
        f"{operation} is unavailable until the governed Source Registry "
        "persistence mutation plane is activated."
    )
    return False

def load_urls() -> list[str]:
    from .services.source_registry_runtime import (
        load_trusted_url_library_view,
    )

    registry_entries, _registry_diagnostics = (
        load_trusted_url_library_view(URL_LIST_FILE)
    )
    urls: list[str] = []
    for registry_entry in registry_entries:
        registry_url = registry_entry.get("url")
        if not isinstance(registry_url, str) or not registry_url:
            raise RuntimeError(
                "Source Registry returned an invalid URL entry."
            )
        urls.append(registry_url)
    return urls

def save_urls(urls: list[str]) -> None:
    del urls
    _url_mutation_unavailable("save")

def add_url(url: str) -> bool:
    del url
    return _url_mutation_unavailable("add")

def remove_url(index_or_value) -> bool:
    del index_or_value
    return _url_mutation_unavailable("remove")

def replace_urls(new_urls: list[str]) -> None:
    del new_urls
    _url_mutation_unavailable("replace")

def list_urls_cli() -> list[str]:
    try:
        urls = load_urls()
    except Exception:
        logger.error(
            "[ERROR] Reading the Source Registry authority view failed; "
            "no legacy fallback was used."
        )
        return []
    if not urls:
        logger.info("[INFO] No URLs in the Source Registry authority view")
        return []
    logger.info("\n[SOURCE REGISTRY AUTHORITY VIEW]")
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
                    logger.info(f"[DELETED] {files[idx]}")
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
            " 1. List Source Registry URLs\n"
            " 2. Add URL (unavailable)\n"
            " 3. Remove URL (unavailable)\n"
            " 4. Replace URL list (unavailable)\n"
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
            _url_mutation_unavailable("add")
        elif choice == "3":
            _url_mutation_unavailable("remove")
        elif choice == "4":
            _url_mutation_unavailable("replace")
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