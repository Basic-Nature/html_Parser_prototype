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


def detect_format_from_links(page, base_url=None):
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
                if ext in href:
                    abs_url = urljoin(base_url or page.url, href)
                    found[ext].append(abs_url)
        except Exception as e:
            log_debug(f"[DEBUG] Failed to evaluate a link: {e}")

    flat_results = []
    for ext in SUPPORTED_FORMATS:
        for url in found[ext]:
            flat_results.append((ext.strip("."), url))
    return flat_results

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
