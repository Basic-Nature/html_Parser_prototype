from dotenv import load_dotenv
from ..handlers.formats import json_handler, pdf_handler, csv_handler
import os
from typing import Optional
from ..utils.shared_logger import logger, rprint
from ..utils.user_prompt import prompt_user_input
from urllib.parse import urljoin

load_dotenv()

# Expandable: add new formats to the .env or here
SUPPORTED_FORMATS = [
    ext if ext.startswith('.') else f'.{ext}'
    for ext in os.getenv("SUPPORTED_FORMATS", ".json, .csv, .pdf").split(",")
]

def detect_format_from_links(page, base_url=None, auto_confirm=False) -> list[tuple[str, str]]:
    """
    Scans a webpage for file links matching supported extensions.
    Returns a flat list in discovery order: [("json", url1), ("csv", url2), ...]
    """
    links = page.query_selector_all("a")
    found = {ext: [] for ext in SUPPORTED_FORMATS}
    logger.info("[INFO] Scanning for available download links...")
    for link in links:
        try:
            href = link.get_attribute("href") or ""
            for ext in SUPPORTED_FORMATS:
                if ext.lower() in href.lower():
                    abs_url = urljoin(base_url or page.url, href)
                    found[ext].append(abs_url)
                    logger.debug(f"[DEBUG] Found {ext} link: {abs_url}")
        except Exception as e:
            logger.debug(f"[DEBUG] Failed to evaluate a link: {e}")

    flat_results = []
    for ext in SUPPORTED_FORMATS:
        for url in found[ext]:
            flat_results.append((ext.strip("."), url))
    if not flat_results:
        logger.warning("[WARN] No supported file formats found on the page.")
    # Auto-confirm logic: return only the first found format if enabled
    if auto_confirm and flat_results:
        logger.info(f"[INFO] Auto-confirm enabled. Automatically selecting: {flat_results[0]}")
        return [flat_results[0]]
    return flat_results

def route_format_handler(format_str: str) -> Optional[object]:
    """
    Dynamically import and return a format-specific handler based on string keyword.
    Expandable: add new formats as needed.
    """
    try:
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
            logger.warning(f"[WARN] Unsupported format requested: {format_str}")
            return None
    except ImportError as e:
        logger.warning(f"[Router] Failed to load handler for format {format_str}: {e}")
        return None

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
    rprint("\n[FORMATS] Available formats:")
    for i, opt in enumerate(format_options):
        rprint(f"  [{i}] {opt}")
    rprint("  [n] Skip format parsing")    
    def validator(x):
        return (x.isdigit() and 0 <= int(x) < len(format_options)) or x.lower() == "n"

    selection = prompt_user_input(
        f"[PROMPT] Select a format to parse (0-{len(format_options)-1}) or 'n' to skip: ",
        default="0",
        validator=validator
    )
    if selection.lower() == "n":
        if logger:
            logger.info("[INFO] User chose to skip format parsing.")
        return None, None
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