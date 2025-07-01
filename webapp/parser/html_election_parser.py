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
import orjson
import re
import threading
import sys
import psycopg2
from pathlib import Path
from datetime import datetime
from typing import cast, Dict, Any, List
from multiprocessing import Pool

from dotenv import load_dotenv
from rich.console import Console
from .utils.shared_logger import log_info, log_debug, log_warning, log_error
from playwright.sync_api import sync_playwright, Page
from sqlalchemy.exc import OperationalError


# --- Local imports (all logic is modularized) ---
from .Context_Integration.Integrity_check import analyze_contest_titles, summarize_context_entities
from .config import BASE_DIR, CONTEXT_DB_PATH, PROJECT_ROOT
from .handlers.formats.html_handler import parse as html_handler
from .state_router import get_handler
from .utils.browser_utils import browser_pipeline
from .utils.db_utils import load_processed_urls
from .utils.download_utils import ensure_input_directory, ensure_output_directory
from .utils.format_router import prompt_and_handle_download


from .utils.html_scanner import scan_html_for_context
from .utils.shared_logic import infer_state_county_from_url
from .bots.librarian import safe_join
from .utils.user_prompt import prompt_user_input
import hashlib

# --- Environment & Path Setup ---
load_dotenv()
console = Console()
INPUT_DIR = os.path.join(PROJECT_ROOT, "input")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
URL_LIST_FILE = os.path.join(BASE_DIR, "parser", "urls.txt")
PROCESSED_URLS_FILE = os.path.join(os.path.dirname(CONTEXT_DB_PATH), ".processed_urls")


# Convert to Path objects for .exists() and .write_text()
URL_LIST_FILE = Path(URL_LIST_FILE)
PROCESSED_URLS_FILE = Path(PROCESSED_URLS_FILE)

# --- Config Flags ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").split(",")[0].strip().upper()
CACHE_PROCESSED_URLS = os.getenv("CACHE_PROCESSED", "true").lower() == "true"
CACHE_LOCK = threading.Lock()
CACHE_RESET = os.getenv("CACHE_RESET", "false").lower() == "true"
HEADLESS_DEFAULT = os.getenv("HEADLESS", "true").lower() == "true"
TIMEOUT_SEC = int(os.getenv("CAPTCHA_TIMEOUT", "300"))
INCLUDE_TIMESTAMP_IN_FILENAME = os.getenv("TIMESTAMP_IN_FILENAME", "true").lower() == "true"
ENABLE_PARALLEL = os.getenv("ENABLE_PARALLEL", "false").lower() == "true"
ENABLE_AI_ANALYSIS = os.getenv("ENABLE_AI_ANALYSIS", "false").lower() == "true"
ENABLE_REALTIME_STREAM = os.getenv("ENABLE_REALTIME_STREAM", "false").lower() == "true"


context_cache = {}

def safe_filename(name):
    return re.sub(r'[^a-zA-Z0-9_\-\.]', '_', name)


# --- Cache Reset ---
if CACHE_RESET and PROCESSED_URLS_FILE.exists():
    log_debug("Deleting .processed_urls cache for fresh start...")
    PROCESSED_URLS_FILE.unlink()

# --- Utility: Load URLs from file or prompt user ---
def load_urls() -> List[str]:
    if not URL_LIST_FILE.exists():
        console.print("[bold red]\nNo urls.txt found. Please input a URL to append:")
        url = prompt_user_input("URL: ").strip()
        if url:
            URL_LIST_FILE.write_text(url + "\n")
            log_info(f"Appended URL to urls.txt: {url}")
        return [url] if url else []
    with URL_LIST_FILE.open('r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
        if not lines:
            console.print("[bold red]\nurls.txt has no usable URLs. Please input a URL to append:")
            url = prompt_user_input("URL: ").strip()
            if url:
                with URL_LIST_FILE.open('a') as f_append:
                    f_append.write(url + "\n")
                log_info(f"Appended URL to urls.txt: {url}")
                return [url]
        return lines

def get_page_hash(page):
    """Returns a hash of the page's HTML content."""
    html = page.content() if hasattr(page, "content") else page.inner_html("html")
    return hashlib.sha256(html.encode("utf-8")).hexdigest()

def mark_url_processed(url, status="success", **metadata):
    """Append or update a processed URL with rich metadata, storing all entries in a JSON array."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = {
        "url": url,
        "timestamp": timestamp,
        "status": status,
        **metadata
    }
    with CACHE_LOCK:
        # Load existing entries
        if PROCESSED_URLS_FILE.exists() and os.path.getsize(PROCESSED_URLS_FILE) > 0:
            try:
                with open(PROCESSED_URLS_FILE, 'rb') as f:
                    entries = orjson.loads(f.read())
                    if not isinstance(entries, list):
                        entries = []
            except Exception:
                entries = []
        else:
            entries = []
        # Update or append
        updated = False
        for i, e in enumerate(entries):
            if e.get("url") == url:
                entries[i] = entry
                updated = True
                break
        if not updated:
            entries.append(entry)
        # Write back as a JSON array
        with open(PROCESSED_URLS_FILE, 'wb') as f:
            f.write(orjson.dumps(entries, option=orjson.OPT_INDENT_2))

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

# --- Manual Format Override (for direct file parsing) ---
def process_format_override():
    from .utils.format_router import route_format_handler
    force_parse = os.getenv("FORCE_PARSE_INPUT_FILE", "false").lower() == "true"
    force_format = os.getenv("FORCE_PARSE_FORMAT", "").strip().lower()
    if not force_parse or not force_format:
        return None
    input_folder = INPUT_DIR
    files = [f for f in os.listdir(input_folder) if f.endswith(f".{force_format}")]
    if not files:
        log_error(f"[red][ERROR] No .{force_format} files found in 'input' folder.[/red]")
        return None
    log_warning(f"[yellow]Manual override enabled for format:[/yellow] [bold]{force_format}[/bold]")
    for i, f in enumerate(files):
        log_info(f"  [bold cyan][{i}][/bold cyan] {f}")
    try:
        selection = prompt_user_input("[PROMPT] Select a file index to parse: ").strip()
        index = int(selection)
        if not (0 <= index < len(files)):
            raise ValueError("Invalid file index")
        target_file = safe_filename(files[index])
    except (IndexError, ValueError, EOFError, KeyboardInterrupt):
        log_error("[red]Invalid selection. Aborting manual parse.[/red]")
        return None
    handler = route_format_handler(force_format)
    if not handler:
        log_error(f"[red][ERROR] No format handler found for '{force_format}'[/red]")
        return None
    full_path = safe_join(input_folder, target_file)
    html_context = {"manual_file": full_path}
    dummy_page = cast(Page, None)
    result = handler.parse(dummy_page, html_context)
    if result and all(result):
        *_, metadata = result
        if "output_file" in metadata:
            log_info(f"[OUTPUT] CSV written to: {metadata['output_file']}")
        else:
            log_warning("[WARN] No output file path returned from parser.")
        mark_url_processed("manual_override", status="success")
        return True
    else:
        log_error("[red][ERROR] Manual parsing failed or returned no data.[/red]")
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
                log_error(f"[bold red][AI ALERT][/bold red] Potential anomalies detected: {anomalies}")
                log_warning(f"[AI] Anomalies detected: {anomalies}")
            else:
                log_info("[AI] No anomalies detected.")
        except Exception as e:
            log_error(f"[AI] Analysis failed: {e}")

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
            log_info("[STREAM] Results streamed in real-time (stub).")
        except Exception as e:
            log_error(f"[STREAM] Streaming failed: {e}")

def resolve_and_parse(page, context, url):
    """
    Use the full context and URL to resolve the best handler via state_router.
    Falls back to html_handler if no handler is found.
    """
    from .Context_Integration.context_coordinator import ContextCoordinator
    # Use the full context for routing
    handler_result = get_handler(context, url=url)
    handler = handler_result.get("handler")
    summary = handler_result.get("summary")
    coordinator = ContextCoordinator()
    coordinator.organize_and_enrich(context)
    if handler and hasattr(handler, 'parse'):
        return handler.parse(page, coordinator, context)
    # Optionally log the routing summary for diagnostics
    if summary and summary.get("log"):
        for entry in summary["log"]:
            log_info(f"[Router] {entry}")
    # Fallback to generic HTML handler
    if hasattr(html_handler, 'parse'):
        return html_handler(page, coordinator, context)
    return html_handler(page, coordinator, context)  

def process_url(target_url, processed_info, cancel_flag=None):
    from .Context_Integration.context_coordinator import ContextCoordinator
    rejected_downloads = set()
    log_info(f"Navigating to: {target_url}")

    browser = context = page = user_agent = None
    try:
        with sync_playwright() as p:
            browser, context, page, user_agent = browser_pipeline(
                p, target_url, cache_exit_callback=mark_url_processed, non_interactive=False
            )
            if cancel_flag and cancel_flag.is_set():
                return
            if not page:
                return
            # --- 1. Prompt for downloadable format and handle if chosen ---
            result, handled = prompt_and_handle_download(page, target_url, rejected_downloads, non_interactive=False)
            if handled:
                # Already handled by format handler, mark as processed and return
                mark_url_processed(target_url, status="success")
                return

            coordinator = ContextCoordinator()
            # --- Detect page hash and use context cache (already enriched) ---
            html_context = scan_html_for_context(target_url, page, non_interactive=False)
            html_context = coordinator.organize_and_enrich(html_context)
            html_context["source_url"] = target_url
            # --- Robust state/county inference and validation ---
            # Only fill state/county if missing, prefer dynamic_state_county_detection as final authority
            if not html_context.get("state") or not html_context.get("county"):
                state, county = infer_state_county_from_url(target_url)
                if state and not html_context.get("state"):
                    html_context["state"] = state
                if county and not html_context.get("county"):
                    html_context["county"] = county

            # --- NLP/NER Analysis (optional, for logger/diagnostics) ---
            try:
                nlp_report = analyze_contest_titles(html_context.get("contests", []))
                entity_summary = summarize_context_entities(html_context.get("contests", []))
                log_info(f"[NLP] Contest Title Analysis: {nlp_report}")
                log_info(f"[NLP] Entity Summary: {entity_summary}")
            except Exception as e:
                log_warning(f"[NLP] Context coordinator analysis failed: {e}")

            # --- Route to state/county/HTML handler ---
            result = resolve_and_parse(page, html_context, target_url)
            if not isinstance(result, tuple) or len(result) != 4:
                log_error("Handler did not return a valid result tuple.")
                mark_url_processed(target_url, status="fail")
                return

            headers, data, contest_title, metadata = result

            # --- Batch Mode: Hand off to coordinator if needed ---
            if html_context.get("batch_mode") and "selected_races" in html_context:
                try:
                    coordinator.handle_batch(
                        page=page,
                        context=html_context,
                        target_url=target_url,
                        processed_info=processed_info,
                        ai_analyze_results=ai_analyze_results,
                        stream_results=stream_results,
                        mark_url_processed=mark_url_processed,
                        output_dir=OUTPUT_DIR
                    )
                except Exception as e:
                    log_error(f"[Batch Mode] Coordinator batch handling failed: {e}", exc_info=True)
                    mark_url_processed(target_url, status="error")
                return

            # --- Single result (non-batch) ---
            if all([headers, data, contest_title, metadata]):
                ai_analyze_results(headers, data, contest_title, metadata)
                stream_results(headers, data, contest_title, metadata)
                output_file = metadata.get("output_file")
                if output_file:
                    if os.path.exists(output_file):
                        log_info(f"[OUTPUT] CSV written to: {output_file}")
                    else:
                        log_warning(f"[WARN] Output file path returned but file does not exist: {output_file}")
                else:
                    output_dir = metadata.get("output_dir") or OUTPUT_DIR
                    possible_files = []
                    if os.path.isdir(output_dir):
                        for f in os.listdir(output_dir):
                            if f.endswith(".csv") or f.endswith(".json"):
                                possible_files.append(os.path.join(output_dir, f))
                    if possible_files:
                        log_warning(f"[WARN] No output file path returned from parser, but found files: {possible_files[-3:]}")
                    else:
                        log_warning("[WARN] No output file path returned from parser and no output files found.")
                mark_url_processed(target_url, status="success")
            else:
                log_warning("Incomplete result structure — skipping CSV write.")
                mark_url_processed(target_url, status="partial")

    except Exception as e:
        log_error(f"[ERROR] Exception while processing {target_url}: {e}", exc_info=True)
        mark_url_processed(target_url, status="error")
    finally:
        try:
            if browser:
                browser.close()
        except Exception:
            pass    
           
# --- Main Entry Point ---
def main():
    try:
        if process_format_override():
            return

        ensure_input_directory()
        ensure_output_directory()

        urls = load_urls()
        log_debug(f"Raw URLs loaded: {urls}")
        log_debug(f"Loaded {len(urls)} raw URLs from urls.txt")

        max_urls = os.getenv("MAX_URLS_DISPLAYED")
        if max_urls and max_urls.isdigit():
            urls = urls[:int(max_urls)]

        if not urls:
            log_error("No URLs to process. Exiting.")
            return

        processed_info = load_processed_urls()
        log_debug(f"{len(urls)} URLs remain after filtering .processed_urls")

        selected_urls = prompt_url_selection(urls, processed_info)
        if not selected_urls:
            log_info("No URLs selected. Exiting.")
            return

        # --- Multiprocessing for batch mode ---
        if ENABLE_PARALLEL:
            with Pool() as pool:
                pool.starmap(process_url, [(url, processed_info) for url in selected_urls])
        else:
            for url in selected_urls:
                process_url(url, processed_info)
        summary = {"success": 0, "fail": 0, "partial": 0, "error": 0, "flagged": 0}
        processed = load_processed_urls()
        for url in selected_urls:
            status = processed.get(url, {}).get("status", "unprocessed")
            if status in summary:
                summary[status] += 1
            if processed.get(url, {}).get("flagged_for_review"):
                summary["flagged"] += 1

        print("\n[SUMMARY]")
        print(f"  URLs processed: {len(selected_urls)}")
        print(f"  Success: {summary['success']}")
        print(f"  Failures: {summary['fail']}")
        print(f"  Partial: {summary['partial']}")
        print(f"  Errors: {summary['error']}")
        print(f"  Flagged for review: {summary['flagged']}")
    except (OperationalError, psycopg2.OperationalError) as db_err:
        log_error(f"[DB ERROR] Could not connect to the database: {db_err}")
        print("[FATAL] Database connection failed. Exiting pipeline.")
        sys.exit(1)          

if __name__ == "__main__":
    main()