# format_router.py
# ==============================================================
# Detects file format links and returns handlers only *after* contest selection.
# Updated to support early pre-handler use in html_handler.py.
# Supports .env-configurable priorities and extended formats.
# ==============================================================

from typing import Optional
from ..utils.shared_logger import log_info, log_debug, log_warning, rprint
from ..utils.user_prompt import prompt_user_input
from urllib.parse import urljoin
from dotenv import load_dotenv
import os

load_dotenv()

SUPPORTED_FORMATS = [
    ext if ext.startswith('.') else f'.{ext}'
    for ext in os.getenv("SUPPORTED_FORMATS", ".json, .csv, .pdf").split(",")
]

def prompt_user_for_format(confirmed, logger=None):
    """
    Prompts the user to select a format from the confirmed list.
    Returns (fmt, local_file) or (None, None) if skipped.
    """
    if not confirmed:
        if logger:
            logger.warning("[WARN] No downloadable formats detected.")
        return None, None
    format_options = [f"{fmt.upper()} ({local_file})" for fmt, local_file in confirmed]
    print("\n[FORMATS] Available formats:")
    for i, opt in enumerate(format_options):
        print(f"  [{i}] {opt}")
    selection = prompt_user_input(
        f"[PROMPT] Select a format to parse (0-{len(format_options)-1}): ",
        default="0",
        validator=lambda x: x.isdigit() and 0 <= int(x) < len(format_options)
    )
    try:
        selected_index = int(selection)
        fmt, local_file = confirmed[selected_index]
        if logger:
            logger.info(f"[INFO] User selected format: {fmt.upper()}")
        return fmt, local_file
    except (IndexError, ValueError):
        if logger:
            logger.warning("[WARN] Invalid selection. Skipping format parsing.")
        return None, None

def detect_format_from_links(page, base_url=None, auto_confirm=False) -> list[tuple[str, str]]:
    """
    Scans a webpage for file links matching supported extensions.
    Returns a flat list in discovery order: [("json", url1), ("csv", url2), ...]
    """
    links = page.query_selector_all("a")
    found = {ext: [] for ext in SUPPORTED_FORMATS}
    log_info("[INFO] Scanning for available download links...")
    for link in links:
        try:
            href = link.get_attribute("href") or ""
            for ext in SUPPORTED_FORMATS:
                if ext.lower() in href.lower():
                    abs_url = urljoin(base_url or page.url, href)
                    found[ext].append(abs_url)
                    log_debug(f"[DEBUG] Found {ext} link: {abs_url}")
        except Exception as e:
            log_debug(f"[DEBUG] Failed to evaluate a link: {e}")

    flat_results = []
    for ext in SUPPORTED_FORMATS:
        for url in found[ext]:
            flat_results.append((ext.strip("."), url))
    if not flat_results:
        log_warning("[WARN] No supported file formats found on the page.")
    # Auto-confirm logic: return only the first found format if enabled
    if auto_confirm and flat_results:
        log_info(f"[INFO] Auto-confirm enabled. Automatically selecting: {flat_results[0]}")
        return [flat_results[0]]
    return flat_results

def route_format_handler(format_str: str) -> Optional[object]:
    """
    Dynamically import and return a format-specific handler based on string keyword.

    Args:
        format_str (str): One of 'json', 'pdf', 'csv', etc.

    Returns:
        Module or None
    """
    try:
        from handlers.formats import json_handler, pdf_handler, csv_handler
        # Add new formats here as needed
        if "json" in format_str:
            return json_handler
        elif "pdf" in format_str:
            return pdf_handler
        elif "csv" in format_str:
            return csv_handler
        # Example for future extensibility:
        # elif "xml" in format_str:
        #     from handlers.formats import xml_handler
        #     return xml_handler
        # elif "xlsx" in format_str:
        #     from handlers.formats import xlsx_handler
        #     return xlsx_handler
        else:
            log_warning(f"[WARN] Unsupported format requested: {format_str}")
            return None
    except ImportError as e:
        log_warning(f"[Router] Failed to load handler for format {format_str}: {e}")
        return None