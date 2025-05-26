# ============================================================
# 🗳️ Smart Elections: HTML Election Parser Pipeline
# ============================================================
#
# Main orchestrator for parsing U.S. election results from county/state canvass sites.
# Supports HTML scraping, structured file parsing (JSON, CSV, PDF), and batch/multiprocessing.
# Delegates all specialized logic to modular handlers/utilities for maintainability.
# Designed for future extensibility: AI anomaly detection, real-time streaming, and distributed collection.
# ============================================================

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import cast, Dict, Any, List
from multiprocessing import Pool

from dotenv import load_dotenv
from rich.console import Console
from rich import print as rprint
from playwright.sync_api import sync_playwright, Page

# --- Local imports (all logic is modularized) ---
from .Context_Integration.context_organizer import append_to_context_library, CONTEXT_LIBRARY_PATH, load_context_library, organize_context
from .handlers.formats import html_handler
from .state_router import get_handler as get_state_handler
from .utils.browser_utils import (
    launch_browser_with_stealth,
    relaunch_browser_fullscreen_if_needed,
    relaunch_browser_stealth,
    detect_environment_for_captcha,
)
from .utils.captcha_tools import (
    is_cloudflare_captcha_present,
    wait_for_user_to_solve_captcha,
)
from .utils.download_utils import ensure_input_directory, ensure_output_directory
from .utils.format_router import detect_format_from_links, prompt_user_for_format, route_format_handler
from .utils.html_scanner import scan_html_for_context
from .utils.shared_logger import logger, rprint
from .utils.user_prompt import prompt_user_input, PromptCancelled


# Optional: Bot integration and future AI/ML hooks
try:
    from .bots.bot_router import run_bot_task
except ImportError:
    run_bot_task = None

# --- Environment & Path Setup ---
load_dotenv()
console = Console()
BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
URL_LIST_FILE = BASE_DIR / "webapp/parser/urls.txt"
CACHE_FILE = BASE_DIR / ".processed_urls"

# --- Config Flags ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").split(",")[0].strip().upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format='[%(levelname)s] %(message)s')
CACHE_PROCESSED_URLS = os.getenv("CACHE_PROCESSED", "true").lower() == "true"
CACHE_RESET = os.getenv("CACHE_RESET", "false").lower() == "true"
HEADLESS_DEFAULT = os.getenv("HEADLESS", "true").lower() == "true"
TIMEOUT_SEC = int(os.getenv("CAPTCHA_TIMEOUT", "300"))
INCLUDE_TIMESTAMP_IN_FILENAME = os.getenv("TIMESTAMP_IN_FILENAME", "true").lower() == "true"
ENABLE_PARALLEL = os.getenv("ENABLE_PARALLEL", "false").lower() == "true"
ENABLE_AI_ANALYSIS = os.getenv("ENABLE_AI_ANALYSIS", "false").lower() == "true"
ENABLE_REALTIME_STREAM = os.getenv("ENABLE_REALTIME_STREAM", "false").lower() == "true"
ENABLE_BOT_TASKS = os.getenv("ENABLE_BOT_TASKS", "false").lower() == "true"

def handle_cloudflare_captcha(playwright, page_or_driver, url, timeout=TIMEOUT_SEC, proxy=None, backend="playwright"):
    """
    Handles Cloudflare CAPTCHA detection and user intervention.
    If persistent CAPTCHA is detected, relaunches browser in GUI mode for manual solve,
    and if still persistent, relaunches in stealth mode.
    Returns (browser, context, page, user_agent) if browser was relaunched, else None.
    """
    logger.info("[CAPTCHA] Checking for Cloudflare CAPTCHA...")
    if not is_cloudflare_captcha_present(page_or_driver):
        logger.info("[CAPTCHA] No CAPTCHA detected.")
        return None

    logger.warning("[CAPTCHA] CAPTCHA detected! Attempting user intervention.")
    solved = wait_for_user_to_solve_captcha(page_or_driver, timeout=timeout)
    if solved:
        logger.info("[CAPTCHA] CAPTCHA solved by user.")
        return None

    # If still not solved, relaunch in GUI mode for manual solve
    logger.warning("[CAPTCHA] Persistent CAPTCHA. Relaunching browser in GUI mode for manual solve.")
    browser, context, page, user_agent = relaunch_browser_fullscreen_if_needed(
        playwright, url, timeout=timeout, proxy=proxy, backend=backend
    )
    if not page:
        logger.error("[CAPTCHA] Failed to relaunch browser for CAPTCHA resolution.")
        return None

    solved = wait_for_user_to_solve_captcha(page, timeout=timeout)
    if solved:
        logger.info("[CAPTCHA] CAPTCHA solved after relaunch.")
        return browser, context, page, user_agent

    # If still persistent, relaunch in stealth mode
    logger.error("[CAPTCHA] CAPTCHA still persistent after GUI intervention. Relaunching in stealth mode.")
    browser, context, page, user_agent = relaunch_browser_stealth(
        playwright, url, proxy=proxy, backend=backend
    )
    return browser, context, page, user_agent

# --- Cache Reset ---
if CACHE_RESET and CACHE_FILE.exists():
    logging.debug("Deleting .processed_urls cache for fresh start...")
    CACHE_FILE.unlink()

# --- Utility: Load URLs from file or prompt user ---
def load_urls() -> List[str]:
    if not URL_LIST_FILE.exists():
        console.print("[bold red]\nNo urls.txt found. Please input a URL to append:")
        url = prompt_user_input("URL: ").strip()
        if url:
            URL_LIST_FILE.write_text(url + "\n")
            logging.info(f"Appended URL to urls.txt: {url}")
        return [url] if url else []
    with open(URL_LIST_FILE, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
        if not lines:
            console.print("[bold red]\nurls.txt has no usable URLs. Please input a URL to append:")
            url = prompt_user_input("URL: ").strip()
            if url:
                with open(URL_LIST_FILE, 'a') as f_append:
                    f_append.write(url + "\n")
                logging.info(f"Appended URL to urls.txt: {url}")
                return [url]
        return lines

# --- Utility: Processed URL cache ---
def load_processed_urls() -> Dict[str, Any]:
    """Load the processed URL cache as a dict: url -> metadata dict."""
    if not CACHE_FILE.exists():
        return {}
    processed = {}
    with open(CACHE_FILE, 'r', encoding="utf-8") as f:
        for line in f:
            # Now support JSON per line for richer metadata
            try:
                if "|" in line:
                    # Legacy: url|timestamp|status|error|handler|state|county|contest
                    parts = line.strip().split("|")
                    url, timestamp, status = parts[:3]
                    extra = parts[3:] if len(parts) > 3 else []
                    processed[url] = {
                        "timestamp": timestamp,
                        "status": status,
                        "extra": extra
                    }
                else:
                    # New: JSON per line
                    meta = json.loads(line)
                    url = meta.get("url")
                    if url:
                        processed[url] = meta
            except Exception as e:
                logger.warning(f"[CACHE] Failed to parse cache line: {line.strip()} ({e})")
    return processed

def mark_url_processed(url, status="success", **metadata):
    """Append or update a processed URL with rich metadata."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = {
        "url": url,
        "timestamp": timestamp,
        "status": status,
        **metadata
    }
    # Append as JSON per line for extensibility
    with open(CACHE_FILE, 'a', encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

def organize_context_with_cache(raw_context, button_features=None, panel_features=None, use_library=True, cache=None):
    """
    Main entry point. Optionally uses the persistent context library to improve mapping.
    Args:
        raw_context: dict from html_scanner or similar
        button_features: Optional pre-extracted button features (list of dicts)
        panel_features: Optional pre-extracted panel features (list of dicts)
        use_library: Whether to use the persistent context library for reference
        cache: Optional processed_info or other cache for deduplication/learning
    Returns:
        dict with keys:
            - contests: list of contest dicts (title, year, type, etc.)
            - buttons: {contest_title: [button_dict, ...], ...}
            - panels: {contest_title: panel_locator, ...}
            - tables: {contest_title: [table_locator, ...], ...}
            - metadata: {state, county, ...}
    """
    if use_library:
        library = load_context_library()
    else:
        library = {"contests": [], "buttons": [], "panels": [], "tables": []}

    # Organize context using the utility function
    organized = organize_context_with_cache(
        raw_context=raw_context,
        button_features=button_features,
        panel_features=panel_features,
        use_library=use_library
    )

    # Optionally append to the context library
    if use_library and organized:
        append_to_context_library(organized, path=CONTEXT_LIBRARY_PATH)

    # If cache is provided, deduplicate contests/buttons against it
    if cache:
        # Deduplicate contests
        contests = organized.get("contests", [])
        if cache.get("contests"):
            existing_titles = {c["title"].lower() for c in cache["contests"]}
            contests = [c for c in contests if c["title"].lower() not in existing_titles]
        # Deduplicate buttons
        buttons = organized.get("buttons", {})
        if cache.get("buttons"):
            for title, btns in buttons.items():
                existing_btns = {b["label"].lower() for b in cache["buttons"].get(title, [])}
                buttons[title] = [b for b in btns if b["label"].lower() not in existing_btns]
        # Deduplicate panels
        panels = organized.get("panels", {})
        if cache.get("panels"):
            for title, panel in panels.items():
                existing_panels = {p["id"].lower() for p in cache["panels"].get(title, [])}
                panels[title] = [p for p in panel if p["id"].lower() not in existing_panels]
        # Deduplicate tables
        tables = organized.get("tables", {})
        if cache.get("tables"):
            for title, tbls in tables.items():
                existing_tbls = {t["id"].lower() for t in cache["tables"].get(title, [])}
                tables[title] = [t for t in tbls if t["id"].lower() not in existing_tbls]
        # Reconstruct organized context with deduplicated data
        organized = {
            "contests": contests,
            "buttons": buttons,
            "panels": panels,
            "tables": tables,
            "metadata": organized.get("metadata", {})
        }

    return organized


        # Deduplicate panels and tables similarly...

    return organized

def get_urls_by_status(processed, status):
    """Return a list of URLs with the given status."""
    return [url for url, meta in processed.items() if meta.get("status") == status]

def get_url_metadata(url):
    """Return metadata for a given URL."""
    processed = load_processed_urls()
    return processed.get(url, {})

# --- Utility: Prompt user to select URLs to process, showing status ---
def prompt_url_selection(urls: List[str], processed: Dict[str, Any]) -> List[str]:
    console.print("\n[bold #eb4f43]URLs loaded:[/bold #eb4f43]")
    for i, url in enumerate(urls):
        status = processed.get(url, {}).get("status", "unprocessed")
        status_color = {
            "success": "green",
            "fail": "red",
            "partial": "yellow",
            "error": "red"
        }.get(status, "white")
        console.print(f"  [{i+1}] {url} [bold {status_color}]({status})[/bold {status_color}]")
    user_input = prompt_user_input("\n[INPUT] Enter indices (comma-separated), 'all', or leave empty to cancel: ").strip().lower()
    if not user_input:
        return []
    if user_input == 'all':
        return urls
    indices = [int(i) - 1 for i in user_input.split(',') if i.strip().isdigit()]
    return [urls[i] for i in indices if 0 <= i < len(urls)]

# --- Handler Resolution: State/County/Format Routing ---
def resolve_and_parse(page, context, url):
    """
    Given a Playwright page, context, and URL, route to the correct handler.
    Tries state/county handler, then HTML handler, then fallback.
    """
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

# --- Manual Format Override (for direct file parsing) ---
def process_format_override():
    force_parse = os.getenv("FORCE_PARSE_INPUT_FILE", "false").lower() == "true"
    force_format = os.getenv("FORCE_PARSE_FORMAT", "").strip().lower()
    if not force_parse or not force_format:
        return None
    input_folder = INPUT_DIR
    files = [f for f in os.listdir(input_folder) if f.endswith(f".{force_format}")]
    if not files:
        rprint(f"[red][ERROR] No .{force_format} files found in 'input' folder.[/red]")
        return None
    rprint(f"[yellow]Manual override enabled for format:[/yellow] [bold]{force_format}[/bold]")
    for i, f in enumerate(files):
        rprint(f"  [bold cyan][{i}][/bold cyan] {f}")
    try:
        selection = prompt_user_input("[PROMPT] Select a file index to parse: ").strip()
        index = int(selection)
        target_file = files[index]
    except (IndexError, ValueError, EOFError, KeyboardInterrupt):
        rprint("[red]Invalid selection. Aborting manual parse.[/red]")
        return None
    handler = route_format_handler(force_format)
    if not handler:
        rprint(f"[red][ERROR] No format handler found for '{force_format}'[/red]")
        return None
    full_path = str(input_folder / target_file)
    html_context = {"manual_file": full_path}
    dummy_page = cast(Page, None)
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

# --- AI/ML Anomaly Detection Stub ---
def ai_analyze_results(headers, data, contest_title, metadata):
    """
    Placeholder for future AI/ML anomaly detection.
    This could call an external service, run a local model, or use AutoGPT.
    """
    # Example: send data to an anomaly detection service or model
    if ENABLE_AI_ANALYSIS:
        try:
            # from .ai_tools import analyze_results
            # anomalies = analyze_results(headers, data, contest_title, metadata)
            anomalies = []  # Placeholder
            if anomalies:
                rprint(f"[bold red][AI ALERT][/bold red] Potential anomalies detected: {anomalies}")
                logger.warning(f"[AI] Anomalies detected: {anomalies}")
            else:
                logger.info("[AI] No anomalies detected.")
        except Exception as e:
            logger.error(f"[AI] Analysis failed: {e}")

# --- Real-time Streaming Stub ---
def stream_results(headers, data, contest_title, metadata):
    """
    Placeholder for future real-time streaming of results.
    Could push to a websocket, message queue, or distributed ledger.
    """
    if ENABLE_REALTIME_STREAM:
        try:
            # from .streaming_tools import stream_to_network
            # stream_to_network(headers, data, contest_title, metadata)
            logger.info("[STREAM] Results streamed in real-time (stub).")
        except Exception as e:
            logger.error(f"[STREAM] Streaming failed: {e}")

# --- Main URL Processing Logic ---
def process_url(target_url, processed_info):
    logging.info(f"Navigating to: {target_url}")
    with sync_playwright() as p:
        try:
            browser, context, page, user_agent = launch_browser_with_stealth(
                p, headless=HEADLESS_DEFAULT, minimized=HEADLESS_DEFAULT
            )
            page.goto(target_url, timeout=60000)
            # --- CAPTCHA handling and environment detection ---
            backend = detect_environment_for_captcha(page)
            captcha_result = handle_cloudflare_captcha(p, page, target_url, timeout=TIMEOUT_SEC, backend=backend)
            if captcha_result:
                browser, context, page, user_agent = captcha_result
            # Scan for context (state, county, races, etc.)
            html_context = scan_html_for_context(page)
            logger.debug(f"html_context after scan: {html_context}")
            html_context["source_url"] = target_url

            # Organize context with cache for learning/deduplication
            organized_context = organize_context_with_cache(html_context, cache=processed_info)

            # --- Format Detection (JSON/CSV/PDF) ---
            FORMAT_DETECTION_ENABLED = os.getenv("FORMAT_DETECTION_ENABLED", "true").lower() == "true"
            result = None
            if FORMAT_DETECTION_ENABLED:
                auto_confirm = os.getenv("FORMAT_AUTO_CONFIRM", "true").lower() == "true"
                confirmed = detect_format_from_links(page, target_url, auto_confirm=auto_confirm)
                try:
                    fmt, local_file = prompt_user_for_format(confirmed, logger=logging)
                except PromptCancelled:
                    rprint("[bold yellow][EXIT][/bold yellow] User cancelled the prompt. Exiting gracefully.")
                    mark_url_processed(target_url, status="cancelled")
                    return  # Clean exit for this URL
                if fmt and local_file:
                    format_handler = route_format_handler(fmt)
                    if format_handler and hasattr(format_handler, "parse"):
                        # add format handler logic here if needed
                        pass
                else:
                    logging.info("[INFO] No format selected or skipped by user. Falling back to HTML parsing.")
            if not result:
                result = resolve_and_parse(page, html_context, target_url)
                if not isinstance(result, tuple) or len(result) != 4:
                    logging.error(f"[ERROR] Handler returned unexpected structure: expected 4 values, got {len(result) if isinstance(result, tuple) else 'non-tuple'}")
                    mark_url_processed(target_url, status="fail")
                    return
            # --- Batch Mode: Multiple Races ---
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
                        ai_analyze_results(headers, data, contest_title, metadata)
                        stream_results(headers, data, contest_title, metadata)
                        if "output_file" in metadata:
                            logging.info(f"[OUTPUT] CSV written to: {metadata['output_file']}")
                        else:
                            logging.warning("[WARN] No output file path returned from parser.")
                    else:
                        logging.warning(f"[Batch Mode] Incomplete data for: {race_title}")
                mark_url_processed(target_url, status="success")
                return

            # --- Default: Single Race/Context ---
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
            # --- Final Output ---
            if not result or not isinstance(result, tuple) or len(result) != 4:
                logging.error("[ERROR] Handler returned unexpected structure: expected 4 values.")
                mark_url_processed(target_url, status="fail")
                return             
            headers, data, contest_title, metadata = result
            if all([headers, data, contest_title, metadata]):
                ai_analyze_results(headers, data, contest_title, metadata)
                stream_results(headers, data, contest_title, metadata)
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

# --- Main Entry Point ---
def main():
    # --- Bot integration: run bot tasks if enabled ---
    if ENABLE_BOT_TASKS and run_bot_task:
        run_bot_task("scan_and_notify", context={})
        return

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
    logging.debug(f"{len(urls)} URLs remain after filtering .processed_urls")

    selected_urls = prompt_url_selection(urls, processed_info)
    if not selected_urls:
        logging.info("No URLs selected. Exiting.")
        return

    # --- Multiprocessing for batch mode ---
    if ENABLE_PARALLEL:
        with Pool() as pool:
            pool.starmap(process_url, [(url, processed_info) for url in selected_urls])
    else:
        for url in selected_urls:
            process_url(url, processed_info)            


if __name__ == "__main__":
    main()