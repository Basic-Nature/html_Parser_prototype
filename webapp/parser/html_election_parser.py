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

from .Context_Integration.Integrity_check import analyze_contest_titles, summarize_context_entities
from .config import BASE_DIR, CONTEXT_DB_PATH, PROJECT_ROOT
from .handlers.formats.html_handler import parse as html_handler
from .state_router import get_handler
from .utils.browser_utils import browser_pipeline
from .utils.db_utils import load_processed_urls
from .utils.download_utils import ensure_input_directory, ensure_output_directory
from .utils.format_router import prompt_and_handle_download
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
URL_LIST_FILE = Path(URL_LIST_FILE)
PROCESSED_URLS_FILE = Path(PROCESSED_URLS_FILE)

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

def safe_filename(name):
    return re.sub(r'[^a-zA-Z0-9_\-\.]', '_', name)

if CACHE_RESET and PROCESSED_URLS_FILE.exists():
    log_debug("Deleting .processed_urls cache for fresh start...")
    PROCESSED_URLS_FILE.unlink()

def load_urls(prompt_func=prompt_user_input) -> List[str]:
    if not URL_LIST_FILE.exists():
        console.log_error("[bold red]\nNo urls.txt found. Please input a URL to append:")
        url = prompt_func("URL: ").strip()
        if url:
            URL_LIST_FILE.write_text(url + "\n")
            log_info(f"Appended URL to urls.txt: {url}")
        return [url] if url else []
    with URL_LIST_FILE.open('r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
        if not lines:
            console.log_error("[bold red]\nurls.txt has no usable URLs. Please input a URL to append:")
            url = prompt_func("URL: ").strip()
            if url:
                with URL_LIST_FILE.open('a') as f_append:
                    f_append.write(url + "\n")
                log_info(f"Appended URL to urls.txt: {url}")
                return [url]
        return lines

def mark_url_processed(url, status="success", **metadata):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = {
        "url": url,
        "timestamp": timestamp,
        "status": status,
        **metadata
    }
    with CACHE_LOCK:
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
        updated = False
        for i, e in enumerate(entries):
            if e.get("url") == url:
                entries[i] = entry
                updated = True
                break
        if not updated:
            entries.append(entry)
        with open(PROCESSED_URLS_FILE, 'wb') as f:
            f.write(orjson.dumps(entries, option=orjson.OPT_INDENT_2))

def prompt_url_selection(urls: List[str], processed: Dict[str, Any], prompt_func=prompt_user_input) -> List[str]:
    log_info("\n[bold #eb4f43]URLs loaded:[/bold #eb4f43]")
    for i, url in enumerate(urls):
        status = processed.get(url, {}).get("status", "unprocessed")
        status_color = {
            "success": "green",
            "fail": "red",
            "partial": "yellow",
            "error": "red"
        }.get(status, "white")
        log_info(f"  [{i+1}] {url} [bold {status_color}]({status})[/bold {status_color}]")
    user_input = prompt_func("\n[INPUT] Enter indices (comma-separated), 'all', or leave empty to cancel: ").strip().lower()
    if not user_input:
        return []
    if user_input == 'all':
        return urls
    indices = [int(i) - 1 for i in user_input.split(',') if i.strip().isdigit()]
    return [urls[i] for i in indices if 0 <= i < len(urls)]

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

def ai_analyze_results(headers, data, contest_title, metadata):
    if ENABLE_AI_ANALYSIS:
        try:
            anomalies = []  # Placeholder
            if anomalies:
                log_error(f"[bold red][AI ALERT][/bold red] Potential anomalies detected: {anomalies}")
                log_warning(f"[AI] Anomalies detected: {anomalies}")
            else:
                log_info("[AI] No anomalies detected.")
        except Exception as e:
            log_error(f"[AI] Analysis failed: {e}")

def stream_results(headers, data, contest_title, metadata):
    if ENABLE_REALTIME_STREAM:
        try:
            log_info("[STREAM] Results streamed in real-time (stub).")
        except Exception as e:
            log_error(f"[STREAM] Streaming failed: {e}")

def orchestrate_url(target_url, processed_info, cancel_flag=None):
    """
    Unified orchestration: infers state/county from URL, routes to handler,
    and delegates all DOM/context scanning to the handler.
    """
    from .Context_Integration.context_coordinator import ContextCoordinator
    rejected_downloads = set()
    log_info(f"Navigating to: {target_url}")

    browser = page = None
    try:
        with sync_playwright() as p:
            browser, _, page, _ = browser_pipeline(
                p, target_url, cache_exit_callback=mark_url_processed, non_interactive=False
            )
            if cancel_flag and cancel_flag.is_set():
                return
            if not page:
                return

            # 1. Prompt for downloadable format and handle if chosen
            result, handled = prompt_and_handle_download(page, target_url, rejected_downloads, non_interactive=False)
            if handled:
                mark_url_processed(target_url, status="success")
                return

            # 2. Infer state/county from URL and build minimal context
            state, county = infer_state_county_from_url(target_url)
            context = {"state": state, "county": county, "url": target_url}
            from .state_router import preload_handler_map
            if state:
                preload_handler_map(restrict_to_states=[state])
            else:
                preload_handler_map()
            # 3. Route to handler using context (no DOM scan here)
            handler_result = get_handler(context, url=target_url)
            handler = handler_result.get("handler")
            summary = handler_result.get("summary")

            # Optionally log the routing summary for diagnostics
            if summary and summary.get("log"):
                for entry in summary["log"]:
                    log_info(f"[Router] {entry}")

            # 4. Prepare coordinator (for handler use)
            coordinator = ContextCoordinator()

            # 5. Call handler (handler is responsible for all DOM/context scanning)
            if handler and hasattr(handler, 'parse'):
                result = handler.parse(page, coordinator, context)
            else:
                log_warning("[Router] No suitable handler found, using generic HTML handler.")
                result = html_handler(page, coordinator, context)

            # 6. Validate result
            if not isinstance(result, tuple) or len(result) != 4:
                log_error("Handler did not return a valid result tuple.")
                mark_url_processed(target_url, status="fail")
                return

            headers, data, contest_title, metadata = result

            # 7. Batch Mode: Hand off to coordinator if needed
            if context.get("batch_mode") and "selected_races" in context:
                try:
                    coordinator.handle_batch(
                        page=page,
                        context=context,
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

            # 8. Single result (non-batch)
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

def main(prompt_func=prompt_user_input, output_func=log_info):
    try:
        if process_format_override():
            return

        ensure_input_directory()
        ensure_output_directory()

        urls = load_urls(prompt_func=prompt_func)
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

        selected_urls = prompt_url_selection(urls, processed_info, prompt_func=prompt_func)
        if not selected_urls:
            log_info("No URLs selected. Exiting.")
            return

        if ENABLE_PARALLEL:
            with Pool() as pool:
                pool.starmap(orchestrate_url, [(url, processed_info) for url in selected_urls])
        else:
            for url in selected_urls:
                orchestrate_url(url, processed_info)
        summary = {"success": 0, "fail": 0, "partial": 0, "error": 0, "flagged": 0}
        processed = load_processed_urls()
        for url in selected_urls:
            status = processed.get(url, {}).get("status", "unprocessed")
            if status in summary:
                summary[status] += 1
            if processed.get(url, {}).get("flagged_for_review"):
                summary["flagged"] += 1

        output_func("\n[SUMMARY]")
        output_func(f"  URLs processed: {len(selected_urls)}")
        output_func(f"  Success: {summary['success']}")
        output_func(f"  Failures: {summary['fail']}")
        output_func(f"  Partial: {summary['partial']}")
        output_func(f"  Errors: {summary['error']}")
        output_func(f"  Flagged for review: {summary['flagged']}")
    except (OperationalError, psycopg2.OperationalError) as db_err:
        log_error(f"[DB ERROR] Could not connect to the database: {db_err}")
        output_func("[FATAL] Database connection failed. Exiting pipeline.")
        sys.exit(1)

if __name__ == "__main__":
    main()