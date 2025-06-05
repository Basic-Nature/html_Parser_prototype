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
import re
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import cast, Dict, Any, List
from multiprocessing import Pool

from dotenv import load_dotenv
from rich.console import Console
from .utils.shared_logger import rprint
from playwright.sync_api import sync_playwright, Page

# --- Local imports (all logic is modularized) ---
from .Context_Integration.Integrity_check import analyze_contest_titles, summarize_context_entities
from .Context_Integration.context_organizer import append_to_context_library, load_context_library, organize_context
from .config import BASE_DIR, CONTEXT_DB_PATH, CONTEXT_LIBRARY_PATH
from .handlers.formats.html_handler import parse as html_handler
from .state_router import get_handler as get_state_handler
from .utils.browser_utils import browser_pipeline
from .utils.db_utils import append_to_context_library, load_processed_urls
from .utils.download_utils import ensure_input_directory, ensure_output_directory
from .utils.format_router import route_format_handler


from .utils.html_scanner import scan_html_for_context
from .utils.logger_instance import logger
from .utils.shared_logic import infer_state_county_from_url, safe_join
from .utils.user_prompt import prompt_user_input
import hashlib

# Optional: Bot integration and future AI/ML hooks
try:
    from .bots.bot_router import run_bot_task
except ImportError:
    run_bot_task = None

# --- Bot Orchestration: Run retrainer/correction bots if needed ---
def run_preprocessing_bots():
    if os.getenv("SKIP_BOT_TASKS", "false").lower() == "true":
        print("[INFO] Skipping bot tasks as requested.")
        return
    if not run_bot_task:
        print("[WARN] Bot router not available; skipping bot tasks.")
        return

    # Only retrain if model is missing, outdated, or not being saved
    model_path = os.path.join(BASE_DIR, "parser", "Context_Integration", "Context_Library", "table_structure_model.pkl")
    retrain_needed = not os.path.exists(model_path) or (time.time() - os.path.getmtime(model_path) > 7 * 86400)
    retrain_lock = os.path.join(os.path.dirname(model_path), ".retrain_lock")
    if retrain_needed and not os.path.exists(retrain_lock):
        try:
            open(retrain_lock, "w").close()
            run_bot_task("retrain_table_structure_models")
        except Exception as e:
            print(f"[ERROR] Retrainer failed: {e}\nIf on Windows, ensure no file explorer or editor is open on the model directory.")
        finally:
            if os.path.exists(retrain_lock):
                os.remove(retrain_lock)
    else:
        print("[BOT] Table structure model is up-to-date or retraining already in progress.")

    # Only run correction bot if new logs exist
    log_dir = os.getenv("LOG_DIR", os.path.join(BASE_DIR, "..", "..", "log"))
    last_run_time = time.time() - 3600  # Example: last hour
    from .bots.bot_router import should_run_correction_bot
    if should_run_correction_bot(log_dir, last_run_time):
        try:
            run_bot_task("manual_correction_bot", args=["--enhanced", "--feedback", "--update-db"])
        except Exception as e:
            print(f"[ERROR] Correction bot failed: {e}")

    # Let the bot router suggest additional bots (Auto-GPT style)
    if hasattr(run_bot_task, "suggest_bots"):
        for bot_name, args in run_bot_task.suggest_bots():
            try:
                run_bot_task(bot_name, args=args)
            except Exception as e:
                print(f"[ERROR] Bot {bot_name} failed: {e}")
# Call this before main logic
run_preprocessing_bots()

# --- Environment & Path Setup ---
load_dotenv()
console = Console()
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
URL_LIST_FILE = os.path.join(BASE_DIR, "parser", "urls.txt")
PROCESSED_URLS_FILE = os.path.join(os.path.dirname(CONTEXT_DB_PATH), ".processed_urls")

# Convert to Path objects for .exists() and .write_text()
URL_LIST_FILE = Path(URL_LIST_FILE)
PROCESSED_URLS_FILE = Path(PROCESSED_URLS_FILE)

# --- Config Flags ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").split(",")[0].strip().upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format='[%(levelname)s] %(message)s')
CACHE_PROCESSED_URLS = os.getenv("CACHE_PROCESSED", "true").lower() == "true"
CACHE_LOCK = threading.Lock()
CACHE_RESET = os.getenv("CACHE_RESET", "false").lower() == "true"
HEADLESS_DEFAULT = os.getenv("HEADLESS", "true").lower() == "true"
TIMEOUT_SEC = int(os.getenv("CAPTCHA_TIMEOUT", "300"))
INCLUDE_TIMESTAMP_IN_FILENAME = os.getenv("TIMESTAMP_IN_FILENAME", "true").lower() == "true"
ENABLE_PARALLEL = os.getenv("ENABLE_PARALLEL", "false").lower() == "true"
ENABLE_AI_ANALYSIS = os.getenv("ENABLE_AI_ANALYSIS", "false").lower() == "true"
ENABLE_REALTIME_STREAM = os.getenv("ENABLE_REALTIME_STREAM", "false").lower() == "true"
ENABLE_BOT_TASKS = os.getenv("ENABLE_BOT_TASKS", "false").lower() == "true"

context_cache = {}

def safe_filename(name):
    return re.sub(r'[^a-zA-Z0-9_\-\.]', '_', name)


# --- Cache Reset ---
if CACHE_RESET and PROCESSED_URLS_FILE.exists():
    logging.debug("Deleting .processed_urls cache for fresh start...")
    PROCESSED_URLS_FILE.unlink()

# --- Utility: Load URLs from file or prompt user ---
def load_urls() -> List[str]:
    if not URL_LIST_FILE.exists():
        console.print("[bold red]\nNo urls.txt found. Please input a URL to append:")
        url = prompt_user_input("URL: ").strip()
        if url:
            URL_LIST_FILE.write_text(url + "\n")
            logging.info(f"Appended URL to urls.txt: {url}")
        return [url] if url else []
    with URL_LIST_FILE.open('r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
        if not lines:
            console.print("[bold red]\nurls.txt has no usable URLs. Please input a URL to append:")
            url = prompt_user_input("URL: ").strip()
            if url:
                with URL_LIST_FILE.open('a') as f_append:
                    f_append.write(url + "\n")
                logging.info(f"Appended URL to urls.txt: {url}")
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
                with open(PROCESSED_URLS_FILE, 'r', encoding="utf-8") as f:
                    entries = json.load(f)
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
        with open(PROCESSED_URLS_FILE, 'w', encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)

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
    organized = organize_context(
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
        if not (0 <= index < len(files)):
            raise ValueError("Invalid file index")
        target_file = safe_filename(files[index])
    except (IndexError, ValueError, EOFError, KeyboardInterrupt):
        rprint("[red]Invalid selection. Aborting manual parse.[/red]")
        return None
    handler = route_format_handler(force_format)
    if not handler:
        rprint(f"[red][ERROR] No format handler found for '{force_format}'[/red]")
        return None
    full_path = safe_join(input_folder, target_file)
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
    from .Context_Integration.context_coordinator import dynamic_state_county_detection, ContextCoordinator
    rejected_downloads = set()
    logging.info(f"Navigating to: {target_url}")

    browser = context = page = user_agent = None
    try:
        with sync_playwright() as p:
            browser, context, page, user_agent = browser_pipeline(
                p, target_url, cache_exit_callback=mark_url_processed
            )
            if not page:
                return

            coordinator = ContextCoordinator()

            # --- Detect page hash and use context cache ---
            html_context = get_or_scan_context(page, coordinator, rejected_downloads=rejected_downloads)
            html_context["source_url"] = target_url

            # --- Robust state/county inference and validation ---
            state, county = infer_state_county_from_url(target_url)
            if state and not html_context.get("state"):
                html_context["state"] = state
            if county and not html_context.get("county"):
                html_context["county"] = county

            context_library = load_context_library()
            validated_county, validated_state, handler_path, issues = dynamic_state_county_detection(
                html_context,
                html_context.get("raw_html", ""),
                context_library
            )
            if validated_state and not html_context.get("state"):
                html_context["state"] = validated_state
            if validated_county and not html_context.get("county"):
                html_context["county"] = validated_county
            if issues:
                logger.warning(f"[STATE/COUNTY VALIDATION] {issues}")

            # --- Organize and enrich context with ML/NER ---
            organized_context = coordinator.organize_and_enrich(html_context)
            try:
                nlp_report = analyze_contest_titles(organized_context.get("contests", []))
                entity_summary = summarize_context_entities(organized_context.get("contests", []))
                logger.info(f"[NLP] Contest Title Analysis: {nlp_report}")
                logger.info(f"[NLP] Entity Summary: {entity_summary}")
            except Exception as e:
                logger.warning(f"[NLP] Context coordinator analysis failed: {e}")

            # --- Route to state/county/HTML handler ---
            result = resolve_and_parse(page, html_context, target_url)
            if not isinstance(result, tuple) or len(result) != 4:
                logging.error("Handler did not return a valid result tuple.")
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
                    logging.error(f"[Batch Mode] Coordinator batch handling failed: {e}", exc_info=True)
                    mark_url_processed(target_url, status="error")
                return

            # --- Single result (non-batch) ---
            if all([headers, data, contest_title, metadata]):
                ai_analyze_results(headers, data, contest_title, metadata)
                stream_results(headers, data, contest_title, metadata)
                output_file = metadata.get("output_file")
                if output_file:
                    if os.path.exists(output_file):
                        logging.info(f"[OUTPUT] CSV written to: {output_file}")
                    else:
                        logging.warning(f"[WARN] Output file path returned but file does not exist: {output_file}")
                else:
                    output_dir = metadata.get("output_dir") or OUTPUT_DIR
                    possible_files = []
                    if os.path.isdir(output_dir):
                        for f in os.listdir(output_dir):
                            if f.endswith(".csv") or f.endswith(".json"):
                                possible_files.append(os.path.join(output_dir, f))
                    if possible_files:
                        logging.warning(f"[WARN] No output file path returned from parser, but found files: {possible_files[-3:]}")
                    else:
                        logging.warning("[WARN] No output file path returned from parser and no output files found.")
                mark_url_processed(target_url, status="success")
            else:
                logging.warning("Incomplete result structure — skipping CSV write.")
                mark_url_processed(target_url, status="partial")

    except Exception as e:
        logging.error(f"[ERROR] Exception while processing {target_url}: {e}", exc_info=True)
        mark_url_processed(target_url, status="error")
    finally:
        try:
            if browser:
                browser.close()
        except Exception:
            pass
    
def resolve_and_parse(page, context, url):
    from .Context_Integration.context_coordinator import ContextCoordinator
    state = context.get("state")
    county = context.get("county")
    handler = get_state_handler({"state": state, "county": county})
    coordinator = ContextCoordinator()
    coordinator.organize_and_enrich(context)
    if handler and hasattr(handler, 'parse'):
        return handler.parse(page, coordinator, context)
    if hasattr(html_handler, 'parse'):
        return html_handler(page, coordinator, context)
    return html_handler(page, coordinator, context)                
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

def get_or_scan_context(page, coordinator, rejected_downloads=None):
    if rejected_downloads is None:
        rejected_downloads = set()
    page_hash = get_page_hash(page)
    if page_hash in context_cache:
        html_context = context_cache[page_hash]
        logger.info(f"[CONTEXT] Using cached context for hash {page_hash}")
    else:
        html_context = scan_html_for_context(page.url, page, rejected_downloads=rejected_downloads)
        html_context = coordinator.organize_and_enrich(html_context)
        context_cache[page_hash] = html_context
        logger.info(f"[CONTEXT] Scanned and cached context for hash {page_hash}")
    return html_context

if __name__ == "__main__":
    main()