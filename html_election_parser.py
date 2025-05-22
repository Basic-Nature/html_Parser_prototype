# ============================================================
# 🗳️ Smart Elections: HTML Election Parser Pipeline
# ============================================================
#
# This script serves as the main pipeline for parsing U.S. election results
# from county and state-level canvass websites. It supports both HTML scraping
# and structured file parsing (e.g., JSON, CSV, PDF).
#
# Key Features:
# - Loads URLs from a list and caches already processed entries
# - Supports rich output logging and CAPTCHA detection
# - Allows manual file parsing via .env overrides
# - Dynamically routes to state/county-specific parsing modules
# - Writes results to structured CSV directories by state, county, and race
# ============================================================

# Standard library
import os
import logging
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool
from typing import cast

# Third-party
from dotenv import load_dotenv
from rich.console import Console
from rich import print as rprint
from playwright.sync_api import sync_playwright, Page

# Local project utils
from handlers.formats import html_handler
from utils.browser_utils import launch_browser_with_stealth
from utils.download_utils import ensure_input_directory, ensure_output_directory
from utils.format_router import detect_format_from_links, route_format_handler
from utils.captcha_tools import handle_cloudflare_captcha
from utils.html_scanner import scan_html_for_context
from state_router import get_handler as get_state_handler
from utils.shared_logger import rprint
# Load settings from .env file
load_dotenv()

# Rich console styling
console = Console()

# Initialize logging based on LOG_LEVEL in .env
log_level_str = os.getenv("LOG_LEVEL", "INFO").split(",")[0].strip().upper()
log_level = getattr(logging, log_level_str, logging.INFO)
logging.basicConfig(level=log_level, format='[%(levelname)s] %(message)s')

# Flags for caching behavior
CACHE_PROCESSED_URLS = os.getenv("CACHE_PROCESSED", "true").lower() == "true"
CACHE_RESET = os.getenv("CACHE_RESET", "false").lower() == "true"
CACHE_FILE = Path(".processed_urls")

# Reset cache if flag is enabled
if CACHE_RESET and CACHE_FILE.exists():
    logging.debug("Deleting .processed_urls cache for fresh start...")
    CACHE_FILE.unlink()

# Project paths
INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")
URL_LIST_FILE = Path("urls.txt")
HEADLESS_DEFAULT = os.getenv("HEADLESS", "true").lower() == "true"
TIMEOUT_SEC = int(os.getenv("CAPTCHA_TIMEOUT", "300"))
INCLUDE_TIMESTAMP_IN_FILENAME = os.getenv("TIMESTAMP_IN_FILENAME", "true").lower() == "true"


# Load a list of target URLs from urls.txt.
# Prompts the user to add one manually if the file is missing or empty.
# If the file is missing or empty, prompt the user to input a fallback URL.
def load_urls():
    if not URL_LIST_FILE.exists():
        logging.error("urls.txt not found. Please provide a valid file.")
        console.print("[bold red]\nNo urls.txt found. Please input a URL to append:")
        url = input("URL: ").strip()
        if url:
            URL_LIST_FILE.write_text(url + "\n")
            logging.info(f"Appended URL to urls.txt: {url}")
        return [url] if url else []

    with open(URL_LIST_FILE, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip() and not line.strip().startswith("#")]
        if not lines:
            logging.warning("urls.txt is empty or contains only comments.")
            console.print("[bold red]\nurls.txt has no usable URLs. Please input a URL to append:")
            url = input("URL: ").strip()
            if url:
                with open(URL_LIST_FILE, 'a') as f_append:
                    f_append.write(url + "\n")
                logging.info(f"Appended URL to urls.txt: {url}")
                return [url]
        return lines

# Load previously processed URLs from .processed_urls cache file.
# Returns a dictionary keyed by URL with timestamp and status.
def load_processed_urls():
    if not CACHE_PROCESSED_URLS or not CACHE_FILE.exists():
        return {}
    processed = {}
    with open(CACHE_FILE, 'r') as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 3:
                url, timestamp, status = parts[:3]
                processed[url] = {"timestamp": timestamp, "status": status}
    return processed

# Append a URL entry to .processed_urls to mark it as processed.
# Includes a timestamp and success/failure status.
def mark_url_processed(url, status="success"):
    if not CACHE_PROCESSED_URLS:
        return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(CACHE_FILE, 'a') as f:
        f.write(f"{url}|{timestamp}|{status}\n")

# Display the list of available URLs and prompt the user to select some.
# Supports entering indices, 'all', or nothing to cancel.
def prompt_url_selection(urls):
    console.print("\n[bold #eb4f43]URLs loaded:[/bold #eb4f43]")
    for i, url in enumerate(urls):
        console.print(f"  [{i+1}] {url}", style="#45818e")
    user_input = input("\n[INPUT] Enter indices (comma-separated), 'all', or leave empty to cancel: ").strip().lower()
    if not user_input:
        return []
    if user_input == 'all':
        return urls
    indices = [int(i) - 1 for i in user_input.split(',') if i.strip().isdigit()]
    return [urls[i] for i in indices if 0 <= i < len(urls)]

# Determine appropriate handler and dispatch to parser

# Dispatch the URL to the appropriate state/county handler using state_router.
# If not available, fallback to the default HTML handler.
# If not available, fallback to the generic HTML parser.
def resolve_and_parse(page, context, url):
    context["source_url"] = url
    state = context.get("state")
    county = context.get("county")
    handler = get_state_handler(state_abbreviation=state, county_name=county)
    if handler and hasattr(handler, 'parse'):
        logging.info(f"[State Router] Matched — routing to {handler.__name__}")
        return handler.parse(page=page, html_context=context)
    if handler and hasattr(handler, 'parse_fallback'):
        logging.info(f"[State Router] Matched — routing to {handler.__name__} fallback")
        return handler.parse_fallback(page=page, html_context=context)
    if hasattr(html_handler, 'parse'):
        logging.info("[State Router] No match — routing to HTML handler")
        return html_handler.parse(page=page, html_context=context)
    if hasattr(html_handler, 'parse_fallback'):
        logging.info("[State Router] No match — routing to HTML handler fallback")
        return html_handler.parse_fallback(page=page, html_context=context)
    return html_handler.parse(page=page, html_context=context)


# Pipeline runner

# If manual parsing override is enabled (via .env), prompt the user to select a file
# and route it through the corresponding handler based on its extension. in the input folder
# and process using the appropriate format handler (JSON, CSV, PDF).
def process_format_override():

    force_parse = os.getenv("FORCE_PARSE_INPUT_FILE", "false").lower() == "true"
    force_format = os.getenv("FORCE_PARSE_FORMAT", "").strip().lower()
    if not force_parse or not force_format:
        return None

    input_folder = "input"
    files = [f for f in os.listdir(input_folder) if f.endswith(f".{force_format}")]
    if not files:
        rprint(f"[red][ERROR] No .{force_format} files found in 'input' folder.[/red]")
        return None

    rprint(f"[yellow]Manual override enabled for format:[/yellow] [bold]{force_format}[/bold]")
    for i, f in enumerate(files):
        rprint(f"  [bold cyan][{i}][/bold cyan] {f}")

    try:
        selection = input("[PROMPT] Select a file index to parse: ").strip()
        index = int(selection)
        target_file = files[index]
    except (IndexError, ValueError, EOFError, KeyboardInterrupt):
        rprint("[red]Invalid selection. Aborting manual parse.[/red]")
        return None
    # Determine the handler for the file format specified in manual override (.env)
    handler = route_format_handler(force_format)
    if not handler:
        rprint(f"[red][ERROR] No format handler found for '{force_format}'[/red]")
        return None

    full_path = os.path.join(input_folder, target_file)
    html_context = {"manual_file": full_path}
    dummy_page = cast(Page, None)  # Temporarily trick the type checker
    result = handler.parse(dummy_page, html_context)

    if result and all(result):
        *_, metadata = result
        if "output_file" in metadata:
            logging.info(f"[OUTPUT] CSV written to: {metadata['output_file']}")
        else:
            logging.warning("[WARN] No output file path returned from parser.")
        mark_url_processed("manual_override", status="success")
        return True
    else:
        rprint("[red][ERROR] Manual parsing failed or returned no data.[/red]")
        return None

# Core handler for processing a single URL: launch browser, detect format,
# scrape or parse election data, route to appropriate handler, and track output.
def process_url(target_url):
    logging.info(f"Navigating to: {target_url}")
    with sync_playwright() as p:
        try:
            # Launch browser using stealth mode to reduce bot detection
            browser, context, page, user_agent = launch_browser_with_stealth(
                p, headless=HEADLESS_DEFAULT, minimized=HEADLESS_DEFAULT
            )
            page.goto(target_url, timeout=60000)
            # Detect and resolve Cloudflare CAPTCHA if triggered
            captcha_result = handle_cloudflare_captcha(p, page, target_url)
            if captcha_result:
                browser, context, page, user_agent = captcha_result
            # Scan the HTML page for embedded context such as state, county, and contest list
            html_context = scan_html_for_context(page)
            html_context["source_url"] = target_url

            FORMAT_DETECTION_ENABLED = os.getenv("FORMAT_DETECTION_ENABLED", "true").lower() == "true"
            format_type = None
            result = None
            if FORMAT_DETECTION_ENABLED:
                # Automatically detect if downloadable formats like JSON/CSV/PDF are available
                confirmed = detect_format_from_links(page, auto_confirm=True)
                for fmt, local_file in confirmed:
                    format_handler = route_format_handler(fmt)
                    if format_handler and hasattr(format_handler, "parse"):
                        file_context = {**html_context, "filename": os.path.basename(local_file), "skip_format": False}
                        dummy_page = cast(Page, None)
                        # Skip parsing if user declined
                        if local_file is None or local_file == "skip":
                            logging.info(f"[INFO] Skipping {fmt.upper()} based on user input.")
                            continue

                        result = format_handler.parse(dummy_page, file_context)
                        if isinstance(result, tuple) and len(result) == 4:
                          *_, metadata = result
                          if metadata.get("skipped"):
                              logging.info(f"[INFO] {fmt.upper()} parsing was intentionally skipped by user.")
                              continue # skip this format and move on 
                        if not isinstance(result, tuple) or len(result) != 4:
                            logging.error(f"[ERROR] Handler returned unexpected structure: expected 4 values, got {len(result) if isinstance(result, tuple) else 'non-tuple'}")
                            mark_url_processed(target_url, status="fail")
                            return                        
                        if result and all(result):
                            headers, data, contest_title, metadata = result
                            if "output_file" in metadata:
                                logging.info(f"[OUTPUT] CSV written to: {metadata['output_file']}")
                            mark_url_processed(target_url, status="success")
                            return
                    else:
                        logging.warning(f"[WARN] No handler found for format: {fmt}")
            # If multiple races are present and batch_mode is enabled, iterate through all
            if html_context.get("batch_mode") and "selected_races" in html_context:
                for race_title in html_context["selected_races"]:
                    if "filter_race_type" in html_context:
                        if html_context["filter_race_type"].lower() not in race_title.lower():
                            logging.info(f"[Batch Mode] Skipped (filter mismatch): {race_title}")
                            continue
                    sub_context = dict(html_context)
                    sub_context["selected_race"] = race_title
                    logging.info(f"[Batch Mode] Parsing: {race_title}")
                    result = resolve_and_parse(page, sub_context, target_url)
                    if not isinstance(result, tuple) or len(result) != 4:
                        logging.warning(f"[Batch Mode] Skipped: {race_title} — Handler error.")
                        continue
                    headers, data, contest_title, metadata = result
                    if all([headers, data, contest_title, metadata]):
                        if "output_file" in metadata:
                            logging.info(f"[OUTPUT] CSV written to: {metadata['output_file']}")
                        else:
                            logging.warning("[WARN] No output file path returned from parser.")
                    else:
                        logging.warning(f"[Batch Mode] Incomplete data for: {race_title}")
                mark_url_processed(target_url, status="success")
                return

            if not result:
                result = resolve_and_parse(page, html_context, target_url)
                if not isinstance(result, tuple) or len(result) != 4:
                    logging.error(f"[ERROR] Handler returned unexpected structure: expected 4 values, got {len(result) if isinstance(result, tuple) else 'non-tuple'}")
                    mark_url_processed(target_url, status="fail")
                    return
            else:
                if isinstance(result, tuple) and len(result) == 4:
                     *_, metadata = result 
                     if metadata.get("skipped"):
                         logging.info(f"[INFO] Handler skipped parsing. Falling back to HTML.")
                         result = resolve_and_parse(page, html_context, target_url)
            # Final result check: if all parts returned, log the outcome and mark URL
            if not result or not isinstance(result, tuple) or len(result) != 4:
                logging.error("[ERROR] Handler returned unexpected structure: expected 4 values.")
                mark_url_processed(target_url, status="fail")
                return             
            headers, data, contest_title, metadata = result
            if all([headers, data, contest_title, metadata]):
                if "output_file" in metadata:
                    logging.info(f"[OUTPUT] CSV written to: {metadata['output_file']}")
                else:
                    logging.warning("[WARN] No output file path returned from parser.")
                mark_url_processed(target_url, status="success")
            else:
                logging.warning("Incomplete result structure — skipping CSV write.")
                mark_url_processed(target_url, status="partial")

        except Exception as e:
            logging.error(f"Failed to process {target_url}: {e}", exc_info=True)
            mark_url_processed(target_url, status="error")
# Main entry point for the HTML election parser pipeline.
# Handles URL queueing, manual override, routing, parsing, and structured output..
# Handles manual overrides, URL queueing, scraping logic, contest selection, and output.
def main():
    if process_format_override():
        return

    ensure_input_directory()
    ensure_output_directory()

    urls = load_urls()
    logging.debug(f"Raw URLs loaded: {urls}")
    logging.debug(f"Loaded {len(urls)} raw URLs from urls.txt")

    max_urls = os.getenv("MAX_URLS_DISPLAYED")
    if max_urls and max_urls.isdigit():
        urls = urls[:int(max_urls)]

    if not urls:
        logging.error("No URLs to process. Exiting.")
        return

    processed_info = load_processed_urls()
    urls = [u for u in urls if u not in processed_info]
    logging.debug(f"{len(urls)} URLs remain after filtering .processed_urls")

    if not urls:
        logging.info("All listed URLs have already been processed.")
        return

    selected_urls = prompt_url_selection(urls)
    if not selected_urls:
        logging.info("No URLs selected. Exiting.")
        return

    # Optionally enable multiprocessing for batch mode
    if os.getenv("ENABLE_PARALLEL", "false").lower() == "true":
        with Pool() as pool:
            pool.map(process_url, selected_urls)
    else:
        for url in selected_urls:
            process_url(url)


if __name__ == "__main__":
    main()
