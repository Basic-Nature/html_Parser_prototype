from __future__ import annotations
# webapp/parser/utils/format_router.py
# ---------------------------------------------------------------
# Format routing and download handling for Smart Elections Parser Webapp
# ---------------------------------------------------------------
import os
import time
import re
from typing import Optional, Tuple
from ..handlers.formats import json_handler, pdf_handler, csv_handler
from .logger_singleton import logger, prompt
from .shared_logic import (
    safe_lower, safe_get, safe_isdigit, safe_parse
)
from .browser_utils import (
    safe_content, safe_query_selector_all, safe_context_library, safe_context_result,
    safe_get_attribute, safe_url
)
from urllib.parse import urljoin
from ..config import SUPPORTED_FORMATS
from .download_utils import download_file
from .html_scanner import load_pattern_kb, append_pattern_kb

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
    fmt = format_str.lower().strip().lstrip('.')
    try:
        if fmt == "json":
            return json_handler
        if fmt == "pdf":
            return pdf_handler
        if fmt == "csv":
            return csv_handler
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
        exts = SUPPORTED_FORMATS
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
    session_id=None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Prompts the user to select a format from the confirmed list.
    Returns (fmt, file_url) or (None, None) if skipped or denied.
    """
    if not confirmed or not isinstance(confirmed, list) or not all(isinstance(x, (list, tuple)) and len(x) == 2 for x in confirmed):
        logger.warning(f"[WARN][Session:{session_id}] No downloadable formats detected or invalid input.")
        return None, None

    # Remove duplicates and sanitize
    seen = set()
    unique_confirmed = []
    for fmt, file_url in confirmed:
        key = (str(fmt).lower().strip(), str(file_url).strip())
        if key not in seen:
            seen.add(key)
            unique_confirmed.append((fmt, file_url))

    format_options = [
        f"{str(fmt).upper()} ({os.path.basename(str(file_url))})"
        for fmt, file_url in unique_confirmed
    ]

    # Build the options string for the prompt message
    options_lines = []
    for i, opt in enumerate(format_options):
        options_lines.append(f"  [{i}] {opt}")
    options_lines.append("  [n or Enter] Skip download")
    options_str = "\n".join(options_lines)

    logger.info(f"\n[FORMATS][Session:{session_id}] Available formats:")
    for line in options_lines:
        logger.info(line)

    def validator(x) -> bool:
        x = str(x).strip().lower()
        return (
            x == "" or x == "n" or
            (x.isdigit() and 0 <= int(x) < len(format_options)) or
            (x in [opt.lower() for opt in format_options])
        )

    prompt_message = (
        f"[PROMPT][Session:{session_id}] Select a format to download:\n"
        f"{options_str}\n"
        f"Enter the number of your choice (0-{len(format_options)-1}) or 'n' to skip: [n] (type 'cancel' to abort)"
    )

    try:
        selection = prompt.prompt_input(
            prompt_message,
            default="n",
            validator=validator,
            session_id=session_id,
            context={"options": format_options, "confirmed": unique_confirmed}
        )
    except Exception as e:
        logger.error(f"[Prompt] Exception during prompt: {e}")
        return None, None

    if selection is None or str(selection).strip().lower() in ("", "n"):
        logger.info(f"[INFO][Session:{session_id}] User chose to skip format download.")
        return None, None

    # Robust mapping: handle index or option string
    try:
        sel = str(selection).strip()
        if sel.isdigit() and 0 <= int(sel) < len(unique_confirmed):
            idx = int(sel)
        else:
            idx = next(
                i for i, opt in enumerate(format_options)
                if opt.lower() == sel.lower()
            )
        fmt, file_url = unique_confirmed[idx]
        logger.info(f"[INFO][Session:{session_id}] User selected format: {str(fmt).upper()}")
        return fmt, file_url
    except Exception as e:
        logger.warning(f"[WARN][Session:{session_id}] Invalid selection. Skipping format download. ({e})")
        return None, None