import os
import time
import orjson
import re
from typing import Optional, Tuple
from dotenv import load_dotenv
from ..handlers.formats import json_handler, pdf_handler, csv_handler
from ..utils.logger_singleton import logger, prompt
from ..utils.shared_logic import (
    safe_lower, safe_get, safe_isdigit, safe_parse
)
from ..utils.browser_utils import (
    safe_content, safe_query_selector_all, safe_context_library, safe_context_result,
    safe_get_attribute, safe_url
)
from urllib.parse import urljoin
from ..config import CONTEXT_LIBRARY_PATH
load_dotenv()
from .download_utils import download_file
from .html_scanner import load_pattern_kb, append_pattern_kb

# --- Load supported formats from .env or context library ---

if os.path.exists(CONTEXT_LIBRARY_PATH):
    with open(CONTEXT_LIBRARY_PATH, "rb") as f:
        try:
            CONTEXT_LIBRARY = orjson.loads(f.read())
        except Exception as e:
            logger.error(f"[format_router] Failed to load context_library.json: {e}")
            CONTEXT_LIBRARY = {}

    formats_raw = CONTEXT_LIBRARY.get("supported_formats", [".json", ".csv", ".pdf"])
    # Securely handle stringified lists
    if isinstance(formats_raw, list):
        JSON_FORMATS = formats_raw
    elif isinstance(formats_raw, str):
        try:
            import json
            parsed = json.loads(formats_raw)
            JSON_FORMATS = parsed if isinstance(parsed, list) else [".json", ".csv", ".pdf"]
        except Exception as e:
            logger.warning(f"[format_router] Could not parse supported_formats string as JSON: {e}")
            JSON_FORMATS = [".json", ".csv", ".pdf"]
    else:
        JSON_FORMATS = [".json", ".csv", ".pdf"]
else:
    logger.error("[format_router] context_library.json not found. Using default formats.")
    JSON_FORMATS = [".json", ".csv", ".pdf"]

# .env takes priority if set, else use JSON
ENV_FORMATS = os.getenv("SUPPORTED_FORMATS")
if ENV_FORMATS:
    SUPPORTED_FORMATS = [
        ext if ext.startswith('.') else f'.{ext}'
        for ext in ENV_FORMATS.split(",")
    ]
else:
    SUPPORTED_FORMATS = JSON_FORMATS

# Remove HTML if present (HTML is fallback, not a downloadable format)
SUPPORTED_FORMATS = [ext for ext in SUPPORTED_FORMATS if ext.lower() not in [".html", "html"]]

def detect_format_from_links(page, base_url=None, auto_confirm=False) -> list[tuple[str, str]]:
    """
    Scans a webpage for file links matching supported extensions.
    Returns a flat list in discovery order: [("json", url1), ("csv", url2), ...]
    """
    links = safe_query_selector_all(page, "a")
    found = {ext: [] for ext in SUPPORTED_FORMATS}
    logger.info("[INFO] Scanning for available download links...")
    for link in links:
        try:
            href = safe_get_attribute(link, "href", logger) or ""
            for ext in SUPPORTED_FORMATS:
                if safe_lower(ext) in safe_lower(href):
                    abs_url = urljoin(base_url or safe_url(page), href)
                    found[ext].append(abs_url)
                    logger.debug(f"[DEBUG] Found {ext} link: {abs_url}")
        except Exception as e:
            logger.debug(f"[DEBUG] Failed to evaluate a link: {e}")

    flat_results = []
    for ext in SUPPORTED_FORMATS:
        for url in found[ext]:
            flat_results.append((safe_lower(ext).strip("."), url))
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
    """
    try:
        if "json" in format_str:
            return json_handler
        elif "pdf" in format_str:
            return pdf_handler
        elif "csv" in format_str:
            return csv_handler
        else:
            logger.warning(f"[WARN] Unsupported format requested: {format_str}")
            return None
    except ImportError as e:
        logger.warning(f"[Router] Failed to load handler for format {format_str}: {e}")
        return None

def extract_download_links_from_html(html, exts=None) -> list[dict]:
    """
    Extract download links from raw HTML using regex for common file extensions.
    Returns a list of dicts: {"href": ..., "format": ..., "source": "html"}
    """
    if exts is None:
        exts = [".json", ".csv", ".pdf"]
    pattern = re.compile(r'href=[\'"]([^\'"]+\.(?:' + '|'.join(ext[1:] for ext in exts) + r'))[\'"]', re.IGNORECASE)
    matches = pattern.findall(html)
    links = []
    for href in matches:
        for ext in exts:
            if safe_lower(href).endswith(safe_lower(ext)):
                links.append({
                    "href": href,
                    "format": safe_lower(ext).strip("."),
                    "source": "html"
                })
    return links

def prompt_and_handle_download(
    page,
    target_url,
    rejected_downloads=None,
    session_id=None
) -> Tuple[Optional[dict], bool]:
    """
    Extracts download links (from context library, DOM, and HTML), prompts user for format,
    downloads file, and routes to handler.
    Returns (result, handled) where handled=True if a format was selected and processed.
    """
    if rejected_downloads is None:
        rejected_downloads = set()

    html = safe_content(page, session_id=session_id)

    # 1. Extract links from context library
    supported_links = []
    context_lib = safe_context_library(page, session_id=session_id)
    download_links = context_lib.get("download_links", [])
    if isinstance(download_links, list):
        supported_links = [link for link in download_links if isinstance(link, dict)]

    # 2. Extract links from DOM (anchor tags) for supported formats
    dom_links = []
    try:
        anchors = safe_query_selector_all(page, "a", session_id=session_id)
        for a in anchors:
            href = safe_get_attribute(a, "href", logger)
            if not href:
                continue
            for ext in [".json", ".csv", ".pdf"]:
                if safe_lower(ext) in safe_lower(href):
                    dom_links.append({
                        "href": href,
                        "format": safe_lower(ext).strip("."),
                        "source": "dom"
                    })
    except Exception as e:
        logger.warning(f"[format_router] DOM scan failed: {e}")

    # 3. Extract links dynamically from HTML (regex or pattern-based)
    dynamic_links = extract_download_links_from_html(html, exts=[".json", ".csv", ".pdf"])

    # 4. Merge and deduplicate all links by (href, format)
    all_links = {}
    for link in supported_links + dom_links + dynamic_links:
        href = safe_get(link, "href", None)
        fmt = safe_get(link, "format", None)
        if href and fmt:
            all_links[(href, fmt)] = link
    merged_links = list(all_links.values())

    # 5. Remove rejected
    new_links = [
        link for link in merged_links
        if isinstance(link, dict) and safe_get(link, "href") not in rejected_downloads
    ]
    if not new_links:
        logger.info("[format_router] No new downloadable links found.")
        return None, False

    # 6. Update context metadata with discovered links (for downstream use, analytics, or UI)
    context_result = safe_context_result(page, session_id=session_id)
    if isinstance(context_result, dict):
        context_result.setdefault("metadata", {})["download_links"] = merged_links
        logger.debug(f"[format_router][Session:{session_id}] Context metadata updated with download_links.")

    # 7. Add to pattern KB for ML-driven format clustering
    format_kb = load_pattern_kb(session_id=session_id) if 'session_id' in load_pattern_kb.__code__.co_varnames else load_pattern_kb()
    kb_entries = []
    for link in merged_links:
        fmt = safe_get(link, "format", "")
        kb_entry = {
            "pattern_id": f"format_{fmt}_{os.path.basename(safe_get(link, 'href', ''))}",
            "label": "download_format",
            "format": fmt,
            "href": safe_get(link, "href", ""),
            "source_url": getattr(page, "url", target_url),
            "timestamp": time.time(),
            "embedding": [],
            "session_id": session_id
        }
        kb_entries.append(kb_entry)
        append_pattern_kb(kb_entry)
    logger.debug(f"[format_router][Session:{session_id}] Pattern KB entries added: {len(kb_entries)}")
    logger.debug(f"[format_router][Session:{session_id}] KB snapshot: {format_kb}")

    # 8. Prompt user for format
    available_files = [
        f"{os.path.basename(safe_get(link, 'href', ''))} ({safe_lower(safe_get(link, 'format', ''))})"
        for link in new_links if isinstance(link, dict)
    ]
    logger.info(f"[cyan]Downloadable file(s) found: {', '.join(available_files)}.[/cyan]")
    confirmed = [
        (safe_lower(safe_get(link, "format", "")), safe_get(link, "href", ""))
        for link in new_links if isinstance(link, dict)
    ]

    fmt, file_url = prompt_user_for_format(
        confirmed,
        session_id=session_id
    )
    if not fmt or not file_url:
        # User skipped or invalid
        for link in new_links:
            if isinstance(link, dict):
                rejected_downloads.add(safe_get(link, "href", ""))
        return None, False

    # 9. Download and handle
    local_file = download_file(safe_url(page), file_url)
    if not local_file:
        logger.error(f"[red]Failed to download file: {file_url}[/red]")
        return None, False

    format_handler = route_format_handler(fmt)
    if format_handler and hasattr(format_handler, "parse"):
        result = safe_parse(
            format_handler,
            None,
            {"manual_file": local_file, "source_url": target_url},
            logger=logger
        )
        logger.debug(f"[format_router][Session:{session_id}] Format handler completed. KB size: {len(format_kb) if format_kb else 'N/A'}")
        return result, True

    logger.error(f"[red]No handler found for format: {fmt}[/red]")
    return None, False

def prompt_user_for_format(
    confirmed,
    logger=logger,
    session_id=None
) -> tuple[Optional[str], Optional[str]]:
    """
    Prompts the user to select a format from the confirmed list.
    Returns (fmt, file_url) or (None, None) if skipped or denied.
    """
    if not confirmed:
        logger.warning(f"[WARN][Session:{session_id}] No downloadable formats detected.")
        return None, None

    format_options = [
        f"{safe_lower(fmt).upper()} ({os.path.basename(file_url)})"
        for fmt, file_url in confirmed
    ]
    logger.info(f"\n[FORMATS][Session:{session_id}] Available formats:")
    for i, opt in enumerate(format_options):
        logger.info(f"  [{i}] {opt}")
    logger.info("  [n or Enter] Skip download")

    def validator(x) -> bool:
        return (
            safe_lower(x) == "" or safe_lower(x) == "n" or
            (safe_isdigit(x) and 0 <= int(x) < len(format_options))
        )

    selection = prompt.prompt_input(
        f"[PROMPT][Session:{session_id}] Select a format to download (0-{len(format_options)-1}) or 'n' to skip:",
        default="n",
        validator=validator,
        session_id=session_id
    )
    if safe_lower(selection) == "" or safe_lower(selection) == "n":
        logger.info(f"[INFO][Session:{session_id}] User chose to skip format download.")
        return None, None
    try:
        selected_index = int(selection)
        fmt, file_url = confirmed[selected_index]
        logger.info(f"[INFO][Session:{session_id}] User selected format: {safe_lower(fmt).upper()}")
        return fmt, file_url
    except (IndexError, ValueError):
        logger.warning(f"[WARN][Session:{session_id}] Invalid selection. Skipping format download.")
        return None, None