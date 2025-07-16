import os
import time
import orjson
import re
from dotenv import load_dotenv
from ..handlers.formats import json_handler, pdf_handler, csv_handler
from ..utils.shared_logger import SharedLogger
from urllib.parse import urljoin
from ..config import CONTEXT_LIBRARY_PATH
load_dotenv()
from .download_utils import download_file
from ..utils.user_prompt import UserPrompt
from .html_scanner import load_pattern_kb, append_pattern_kb

prompt = UserPrompt()
logger = SharedLogger()
# --- Load supported formats from .env or context library ---

if os.path.exists(CONTEXT_LIBRARY_PATH):
    with open(CONTEXT_LIBRARY_PATH, "rb") as f:
        CONTEXT_LIBRARY = orjson.loads(f.read())
    JSON_FORMATS = CONTEXT_LIBRARY.get("supported_formats", [".json", ".csv", ".pdf"])
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

def route_format_handler(format_str: str):
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

def extract_download_links_from_html(html, exts=None):
    """
    Extract download links from raw HTML using regex for common file extensions.
    Returns a list of dicts: {"href": ..., "format": ..., "source": "html"}
    """
    if exts is None:
        exts = [".json", ".csv", ".pdf"]
    # Regex for hrefs ending with supported extensions
    pattern = re.compile(r'href=[\'"]([^\'"]+\.(?:' + '|'.join(ext[1:] for ext in exts) + r'))[\'"]', re.IGNORECASE)
    matches = pattern.findall(html)
    links = []
    for href in matches:
        for ext in exts:
            if href.lower().endswith(ext):
                links.append({
                    "href": href,
                    "format": ext.strip("."),
                    "source": "html"
                })
    return links

def prompt_and_handle_download(page, target_url, rejected_downloads=None, non_interactive=False):
    """
    Extracts download links (from context library, DOM, and HTML), prompts user for format,
    downloads file, and routes to handler.
    Returns (result, handled) where handled=True if a format was selected and processed.
    """
    if rejected_downloads is None:
        rejected_downloads = set()

    html = page.content()

    # 1. Extract links from context library
    supported_links = []
    if hasattr(page, "context_library") and page.context_library:
        supported_links = [link for link in page.context_library.get("download_links", [])]

    # 2. Extract links from DOM (anchor tags) for supported formats
    dom_links = []
    try:
        anchors = page.query_selector_all("a")
        for a in anchors:
            href = a.get_attribute("href")
            if not href:
                continue
            for ext in [".json", ".csv", ".pdf"]:
                if ext in href.lower():
                    dom_links.append({
                        "href": href,
                        "format": ext.strip("."),
                        "source": "dom"
                    })
    except Exception as e:
        logger.warning(f"[format_router] DOM scan failed: {e}")

    # 3. Extract links dynamically from HTML (regex or pattern-based)
    dynamic_links = extract_download_links_from_html(html, exts=[".json", ".csv", ".pdf"])

    # 4. Merge and deduplicate all links by (href, format)
    all_links = {}
    for link in supported_links + dom_links + dynamic_links:
        href = link.get("href")
        fmt = link.get("format")
        if href and fmt:
            all_links[(href, fmt)] = link
    merged_links = list(all_links.values())

    # 5. Remove rejected
    new_links = [link for link in merged_links if link["href"] not in rejected_downloads]
    if not new_links:
        logger.info("[format_router] No new downloadable links found.")
        return None, False
    # 6. Optionally update context metadata (if available)
    if hasattr(page, "context_result") and isinstance(page.context_result, dict):
        page.context_result.setdefault("metadata", {})["download_links"] = merged_links

    # 7. Add to pattern KB for ML-driven format clustering
    format_kb = load_pattern_kb()
    for link in merged_links:
        fmt = link["format"]
        append_pattern_kb({
            "pattern_id": f"format_{fmt}_{os.path.basename(link['href'])}",
            "label": "download_format",
            "format": fmt,
            "href": link["href"],
            "source_url": getattr(page, "url", target_url),
            "timestamp": time.time(),
            "embedding": [],
        })

    # 8. Prompt user for format
    available_files = [f"{os.path.basename(link['href'])} ({link['format']})" for link in new_links]
    logger.info(f"[cyan]Downloadable file(s) found: {', '.join(available_files)}.[/cyan]")
    confirmed = [(link["format"], link["href"]) for link in new_links]
    fmt, file_url = prompt_user_for_format(confirmed)
    if not fmt or not file_url:
        # User skipped or invalid
        for link in new_links:
            rejected_downloads.add(link["href"])
        return None, False

    # 9. Download and handle
    local_file = download_file(page.url, file_url)
    if not local_file:
        logger.error(f"[red]Failed to download file: {file_url}[/red]")
        return None, False

    format_handler = route_format_handler(fmt)
    if format_handler and hasattr(format_handler, "parse"):
        result = format_handler.parse(None, {"manual_file": local_file, "source_url": target_url})
        return result, True

    logger.error(f"[red]No handler found for format: {fmt}[/red]")
    return None, False

def prompt_user_for_format(confirmed, logger=None):
    """
    Prompts the user to select a format from the confirmed list.
    Returns (fmt, file_url) or (None, None) if skipped or denied.
    """
    if not confirmed:
        logger.warning("[WARN] No downloadable formats detected.")
        return None, None

    format_options = [f"{fmt.upper()} ({os.path.basename(file_url)})" for fmt, file_url in confirmed]
    logger.info("\n[FORMATS] Available formats:")
    for i, opt in enumerate(format_options):
        logger.info(f"  [{i}] {opt}")
    logger.info("  [n or Enter] Skip download")
    def validator(x):
        return (
            x == "" or x.lower() == "n" or
            (x.isdigit() and 0 <= int(x) < len(format_options))
        )

    selection = prompt.prompt_input(
        f"[PROMPT] Select a format to download (0-{len(format_options)-1}) or 'n' to skip:",
        default="n",
        validator=validator
    )
    if selection == "" or selection.lower() == "n":
        logger.info("[INFO] User chose to skip format download.")
        return None, None
    try:
        selected_index = int(selection)
        fmt, file_url = confirmed[selected_index]
        logger.info(f"[INFO] User selected format: {fmt.upper()}")
        return fmt, file_url
    except (IndexError, ValueError):
        logger.warning("[WARN] Invalid selection. Skipping format download.")
        return None, None