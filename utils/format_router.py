# format_router.py
# ==============================================================
# Detects file format links and returns handlers only *after* contest selection.
# Updated to support early pre-handler use in html_handler.py.
# Supports .env-configurable priorities and extended formats.
# ==============================================================
from rich.prompt import Prompt
from utils.shared_logger import log_info, log_debug, log_warning, rprint
from urllib.parse import urljoin
import os

SUPPORTED_FORMATS = [".json", ".csv", ".pdf"]


def detect_format_from_links(page, base_url=None, auto_confirm=False):
    
    """
    Scans a webpage for file links matching supported extensions.
    Returns a flat list in discovery order: [("json", url1), ("csv", url2), ...]
    """
    """
    Scans a webpage for file links matching supported extensions.
    Logs what it finds, but avoids repeated error clutter for missing types.

    Args:
        page (Playwright Page): Browser page object
        base_url (str): Optional fallback URL to resolve relative links

    Returns:
        List[Tuple[str, str]]: Flattened list of (format, url) pairs in discovery order
    """
    links = page.query_selector_all("a")
    found = {ext: [] for ext in SUPPORTED_FORMATS}

    log_info("[INFO] Scanning for available download links...")
    for link in links:
        try:
            href = link.get_attribute("href") or ""
            for ext in SUPPORTED_FORMATS:
                if ext in href:
                    abs_url = urljoin(base_url or page.url, href)
                    found[ext].append(abs_url)
        except Exception as e:
            log_debug(f"[DEBUG] Failed to evaluate a link: {e}")

    for ext in SUPPORTED_FORMATS:
        if found[ext]:
            log_info(f"[FOUND] {ext.upper()} link(s):")
            for i, url in enumerate(found[ext]):
                log_info(f"  [{i}] {url}")
        else:
            log_debug(f"[SCAN] No {ext} files found.")

    # Flatten list in discovery order: [('json', url1), ('csv', url2), ...]
    flat_results = []
    for ext in SUPPORTED_FORMATS:
        for url in found[ext]:
            flat_results.append((ext.strip("."), url))

    # Prompt user whether to parse each file
    from utils.download_utils import download_confirmed_file
    confirmed_links = []
    for fmt, url in flat_results:
        filename = os.path.basename(url)
        if not any(word in filename.lower() for word in ["result", "election", "returns", "vote"]):
            log_debug(f"[SKIP] Skipping unrelated file: {filename}")
            continue

        rprint(f"[bold green]Discovered {fmt.upper()} file:[/bold green] {filename}")
        rprint(f"[dim]Default is 'n'. Skips parsing {fmt.upper()} and moves on to html parsing.[/dim]")
        choice = Prompt.ask("[PROMPT] Parse this file?", choices=["y", "n"], default="n", show_choices=True)
        if choice.lower() == "y":
            file_path = download_confirmed_file(url, page.url)
            if file_path:
                confirmed_links.append((fmt, file_path))
    if auto_confirm:
        return flat_results  # skip Prompt.ask
    return confirmed_links


def route_format_handler(format_str):
    """
    Dynamically import and return a format-specific handler based on string keyword.

    Args:
        format_str (str): One of 'json', 'pdf', 'csv', etc.

    Returns:
        Module or None
    """
    try:
        from handlers.formats import json_handler, pdf_handler, csv_handler
        if "json" in format_str:
            return json_handler
        elif "pdf" in format_str:
            return pdf_handler
        elif "csv" in format_str:
            return csv_handler
        else:
            log_warning(f"[WARN] Unsupported format requested: {format_str}")
            return None
    except ImportError as e:
        log_warning(f"[Router] Failed to load handler for format {format_str}: {e}")
        return None
