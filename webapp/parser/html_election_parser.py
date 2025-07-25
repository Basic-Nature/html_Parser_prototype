import os
import orjson
import re
import threading
import sys
import psycopg2
from pathlib import Path
from datetime import datetime
from typing import cast, Dict, Any, List, Callable
from multiprocessing import Pool
from dotenv import load_dotenv

from playwright.sync_api import sync_playwright, Page
from sqlalchemy.exc import OperationalError
from .config import BASE_DIR, CONTEXT_DB_PATH, PROJECT_ROOT
from .handlers.formats.html_handler import parse as html_handler
from .state_router import get_handler, preload_handler_map
from .utils.browser_utils import browser_pipeline, safe_browser_close
from .utils.db_utils import load_processed_urls
from .utils.download_utils import ensure_input_directory, ensure_output_directory
from .utils.format_router import prompt_and_handle_download
from .utils.shared_logic import infer_state_county_from_url, safe_parse, safe_is_set
from .bots.librarian import safe_join
from .utils.user_prompt import UserPrompt
from .utils.shared_logger import SharedLogger, RichConsoleProxy
prompt = UserPrompt()
logger = SharedLogger()
# --- Environment & Path Setup ---
load_dotenv()
console = RichConsoleProxy()
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
    logger.warning("Deleting .processed_urls cache for fresh start...")
    PROCESSED_URLS_FILE.unlink()

def load_urls(prompt_func=prompt.prompt_input) -> List[str]:
    def safe_strip(val):
        return val.strip() if isinstance(val, str) else ""

    if not URL_LIST_FILE.exists():
        console.print("[bold red]\nNo urls.txt found. Please input a URL to append:")
        url = safe_strip(prompt_func("URL: "))
        if url:
            URL_LIST_FILE.write_text(url + "\n")
            logger.info(f"Appended URL to urls.txt: {url}")
        return [url] if url else []

    with URL_LIST_FILE.open('r') as f:
        lines = []
        for line in f:
            line_stripped = safe_strip(line)
            if line_stripped and not line_stripped.startswith("#"):
                lines.append(line_stripped)

    if not lines:
        console.print("[bold red]\nurls.txt has no usable URLs. Please input a URL to append:")
        url = safe_strip(prompt_func("URL: "))
        if url:
            with URL_LIST_FILE.open('a') as f_append:
                f_append.write(url + "\n")
            logger.info(f"Appended URL to urls.txt: {url}")
            return [url]
    return lines

def mark_url_processed(url, status="success", **metadata) -> None:
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
            url_val = e.get("url") if isinstance(e, dict) else None
            if url_val == url:
                entries[i] = entry
                updated = True
                break
        if not updated:
            entries.append(entry)
        with open(PROCESSED_URLS_FILE, 'wb') as f:
            f.write(orjson.dumps(entries, option=orjson.OPT_INDENT_2))

def prompt_url_selection(
    urls: List[str],
    processed: Dict[str, Any],
    prompt_func: Callable[[str], str],
    output_func: Callable[[str], None],
    cancel_flag: threading.Event = None,
    non_interactive=False
) -> List[str]:
    output_func("\n[bold #eb4f43]URLs loaded:[/bold #eb4f43]")
    for i, url in enumerate(urls):
        proc_entry = processed.get(url)
        status = "unprocessed"
        if isinstance(proc_entry, dict):
            status_val = proc_entry.get("status")
            if isinstance(status_val, str):
                status = status_val
        status_color = {
            "success": "green",
            "fail": "red",
            "partial": "yellow",
            "error": "red"
        }.get(status, "white")
        output_func(f"  [{i+1}] {url} [bold {status_color}]({status})[/bold {status_color}]")

    if non_interactive:
        output_func("[INFO] Non-interactive mode: awaiting selection from frontend or API.")
        return []

    # Check cancel_flag before prompting
    if cancel_flag is not None and hasattr(cancel_flag, "is_set") and callable(cancel_flag.is_set):
        if cancel_flag.is_set():
            output_func("[CANCELLED] Selection cancelled before prompt.")
            return []

    user_input = prompt_func("\n[INPUT] Enter indices (comma-separated), 'all', or leave empty to cancel: ")

    # Check cancel_flag after prompt
    if cancel_flag is not None and hasattr(cancel_flag, "is_set") and callable(cancel_flag.is_set):
        if cancel_flag.is_set():
            output_func("[CANCELLED] Selection cancelled after prompt.")
            return []

    if not isinstance(user_input, str):
        return []
    user_input = user_input.strip().lower()
    if not user_input:
        return []
    if user_input == 'all':
        return urls
    indices = []
    for i in user_input.split(',') if isinstance(user_input, str) else []:
        i_stripped = i.strip() if isinstance(i, str) else ""
        if i_stripped.isdigit():
            idx = int(i_stripped) - 1
            if 0 <= idx < len(urls):
                indices.append(idx)
    return [urls[i] for i in indices]

def process_format_override() -> bool:
    from .utils.format_router import route_format_handler
    force_parse = os.getenv("FORCE_PARSE_INPUT_FILE", "false").lower() == "true"
    force_format = os.getenv("FORCE_PARSE_FORMAT", "").strip().lower()
    if not force_parse or not force_format:
        return False
    input_folder = INPUT_DIR
    files = [f for f in os.listdir(input_folder) if f.endswith(f".{force_format}")]
    if not files:
        logger.error(f"[red][ERROR] No .{force_format} files found in 'input' folder.[/red]")
        return None
    logger.warning(f"[yellow]Manual override enabled for format:[/yellow] [bold]{force_format}[/bold]")
    for i, f in enumerate(files):
        logger.info(f"  [bold cyan][{i}][/bold cyan] {f}")
    try:
        selection = prompt.prompt_input("[PROMPT] Select a file index to parse: ").strip()
        index = int(selection)
        if not (0 <= index < len(files)):
            raise ValueError("Invalid file index")
        target_file = safe_filename(files[index])
    except (IndexError, ValueError, EOFError, KeyboardInterrupt):
        logger.error("[red]Invalid selection. Aborting manual parse.[/red]")
        return None
    handler = route_format_handler(force_format)
    if not handler:
        logger.error(f"[red][ERROR] No format handler found for '{force_format}'[/red]")
        return None
    full_path = safe_join(input_folder, target_file)
    html_context = {"manual_file": full_path}
    dummy_page = cast(Page, None)
    result = safe_parse(handler, dummy_page, html_context, logger=logger)
    if result and all(result):
        *_, metadata = result
        if "output_file" in metadata:
            logger.info(f"[OUTPUT] CSV written to: {metadata['output_file']}")
        else:
            logger.warning("No output file path returned from parser.")
        mark_url_processed("manual_override", status="success")
        return True
    else:
        logger.error("[red][ERROR] Manual parsing failed or returned no data.[/red]")
        return None

def ai_analyze_results(headers, data, contest, metadata):
    """
    Uses advanced NLP and ML utilities to analyze results for anomalies and integrity issues.
    Logs findings and flags suspicious contests.
    """
    if ENABLE_AI_ANALYSIS:
        try:
            from .Context_Integration.Integrity_check import analyze_contests, print_integrity_summary

            # Prepare context for analysis
            contests = []
            if isinstance(contest, list):
                contests = contest
            elif isinstance(contest, dict):
                contests = [contest]

            # Advanced: Attach headers, data, and metadata to each contest for richer analysis
            for c in contests:
                if isinstance(c, dict):
                    c["_headers"] = headers
                    c["_data"] = data
                    c["_metadata"] = metadata

            # Run integrity and anomaly checks
            results = analyze_contests(contests)
            anomalies = results.get("ml_anomalies", [])
            flagged = results.get("flagged_suspicious", [])
            integrity_issues = results.get("integrity_issues", [])
            summary_stats = results.get("summary_stats", {})

            # Log summary with all context
            if anomalies or flagged or integrity_issues:
                logger.error(
                    f"[bold red][AI ALERT][/bold red] Potential anomalies: {anomalies}, "
                    f"Flagged: {flagged}, Integrity issues: {integrity_issues}, "
                    f"Summary: {summary_stats}"
                )
                logger.warning(
                    f"[AI] Anomalies: {anomalies}, Flagged: {flagged}, Integrity Issues: {integrity_issues}, "
                    f"Metadata: {metadata}"
                )
                print_integrity_summary(contests)
            else:
                logger.info(f"[AI] No anomalies or suspicious contests detected. Metadata: {metadata}")
        except Exception as e:
            logger.error(f"[AI] Analysis failed: {e}")

def stream_results(headers, data, contest, metadata):
    """
    Streams results in real-time if enabled, using rich output and context-aware formatting.
    """
    if ENABLE_REALTIME_STREAM:
        try:
            from .Context_Integration.Integrity_check import print_integrity_summary

            # Prepare contests for streaming, attach context
            contests = []
            if isinstance(contest, list):
                contests = contest
            elif isinstance(contest, dict):
                contests = [contest]

            for c in contests:
                if isinstance(c, dict):
                    c["_headers"] = headers
                    c["_data"] = data
                    c["_metadata"] = metadata

            logger.info("[STREAM] Streaming results in real-time with full context...")
            print_integrity_summary(contests)
            # Optionally, stream metadata and summary stats if needed
            logger.info(f"[STREAM] Metadata: {metadata}")
        except Exception as e:
            logger.error(f"[STREAM] Streaming failed: {e}")

def orchestrate_url(
    target_url,
    processed_info,
    prompt_func,
    output_func,
    session_id=None,
    cancel_flag=None,
    non_interactive=False
):
    from .Context_Integration.context_coordinator import ContextCoordinator
    rejected_downloads = set()
    output_func(f"Navigating to: {target_url} (Session: {session_id})")

    browser = page = None
    try:
        with sync_playwright() as p:
            browser, _, page, _ = browser_pipeline(
                p, target_url, cache_exit_callback=mark_url_processed, non_interactive=non_interactive, session_id=session_id
            )
            # Robust cancel_flag check
            if cancel_flag is not None and safe_is_set(cancel_flag):
                try:
                    if safe_is_set(cancel_flag):
                        output_func(f"[CANCELLED] Processing stopped for {target_url} (Session: {session_id})")
                        safe_browser_close(browser, output_func, session_id)
                        return
                except Exception as e:
                    output_func(f"[WARN] Exception during cancel_flag check: {e} (Session: {session_id})")
                    safe_browser_close(browser, output_func, session_id)
                    return

            if not page:
                output_func(f"[ERROR] Could not open page for {target_url} (Session: {session_id})")
                safe_browser_close(browser, output_func, session_id)
                return

            # 1. Prompt for downloadable format and handle if chosen
            result, handled = prompt_and_handle_download(
                page, target_url, rejected_downloads, non_interactive=non_interactive, prompt_func=prompt_func, session_id=session_id
            )
            if handled:
                mark_url_processed(target_url, status="success", session_id=session_id)
                output_func(f"[INFO] Download handled for {target_url} (Session: {session_id})")
                safe_browser_close(browser, output_func, session_id)
                return

            # 2. Infer state/county from URL and build minimal context
            state, county = infer_state_county_from_url(target_url)
            context = {
                "state": state,
                "county": county,
                "url": target_url,
                "session_id": session_id
            }
            if state:
                preload_handler_map(restrict_to_states=[state])
            else:
                preload_handler_map()
            # 3. Route to handler using context (no DOM scan here)
            handler_result = get_handler(
                context,
                url=target_url,
                debug=False,
                fuzzy_cutoff=None,
                non_interactive=non_interactive,
                session_id=session_id
            )
            handler = handler_result.get("handler") if isinstance(handler_result, dict) else None
            summary = handler_result.get("summary") if isinstance(handler_result, dict) else None

            # Optionally log the routing summary for diagnostics
            if summary and isinstance(summary, dict) and summary.get("log"):
                log_entries = summary.get("log")
                if isinstance(log_entries, list):
                    for entry in log_entries:
                        output_func(f"[Router] {entry}")

            # 4. Prepare coordinator (for handler use)
            coordinator = ContextCoordinator()

            # 5. Call handler (handler is responsible for all DOM/context scanning)
            result = None
            if handler and hasattr(handler, 'parse'):
                result = safe_parse(handler, page, coordinator, context, logger=logger)
            else:
                output_func("[Router] No suitable handler found, using generic HTML handler.")
                result = safe_parse(html_handler, page, coordinator, context, logger=logger)

            # 6. Validate result
            if not isinstance(result, tuple) or len(result) != 4:
                output_func(f"[ERROR] Handler did not return a valid result tuple. (Session: {session_id})")
                mark_url_processed(target_url, status="fail", session_id=session_id)
                safe_browser_close(browser, output_func, session_id)
                return

            headers, data, contest, metadata = result

            # 7. Batch Mode: Hand off to coordinator if needed
            batch_mode = context.get("batch_mode") if isinstance(context, dict) else None
            selected_races = context.get("selected_races") if isinstance(context, dict) else None
            if batch_mode and selected_races:
                try:
                    coordinator.handle_batch(
                        page=page,
                        context=context,
                        target_url=target_url,
                        processed_info=processed_info,
                        ai_analyze_results=ai_analyze_results,
                        stream_results=stream_results,
                        mark_url_processed=mark_url_processed,
                        output_dir=OUTPUT_DIR,
                        session_id=session_id
                    )
                except Exception as e:
                    output_func(f"[Batch Mode] Coordinator batch handling failed: {e} (Session: {session_id})")
                    mark_url_processed(target_url, status="error", session_id=session_id)
                safe_browser_close(browser, output_func, session_id)
                return

            # 8. Single result (non-batch)
            if all([headers, data, contest, metadata]):
                ai_analyze_results(headers, data, contest, metadata)
                stream_results(headers, data, contest, metadata)
                output_file = metadata.get("output_file") if isinstance(metadata, dict) else None
                if output_file:
                    if os.path.exists(output_file):
                        output_func(f"[OUTPUT] CSV written to: {output_file} (Session: {session_id})")
                    else:
                        output_func(f"[WARN] Output file path returned but file does not exist: {output_file} (Session: {session_id})")
                else:
                    output_dir = metadata.get("output_dir") if isinstance(metadata, dict) else OUTPUT_DIR
                    possible_files = []
                    if os.path.isdir(output_dir):
                        for f in os.listdir(output_dir):
                            if f.endswith(".csv") or f.endswith(".json"):
                                possible_files.append(os.path.join(output_dir, f))
                    if possible_files:
                        output_func("[WARN] No output file path returned from parser, but found files:\n" + "\n".join(possible_files[-3:]))
                    else:
                        output_func("[WARN] No output file path returned from parser and no output files found.")
                mark_url_processed(target_url, status="success", session_id=session_id)
            else:
                output_func(f"Incomplete result structure — skipping CSV write. (Session: {session_id})")
                mark_url_processed(target_url, status="partial", session_id=session_id)

    except Exception as e:
        output_func(f"[ERROR] Exception while processing {target_url}: {e} (Session: {session_id})")
        mark_url_processed(target_url, status="error", session_id=session_id)
    finally:
        # Robust browser close (only once)
        safe_browser_close(browser, output_func, session_id)

def main(prompt_func=prompt.prompt_input, output_func=logger.info, session_id=None, cancel_flag=None, non_interactive=False):
    try:
        if process_format_override():
            return

        ensure_input_directory()
        ensure_output_directory()

        urls = load_urls(prompt_func=prompt_func)
        # Output URLs as JSON array for webapp, plain text for CLI
        if logger.mode == "webapp" and logger.format == "json":
            output_func({"level": "INFO", "message": urls})
        else:
            output_func("Raw URLs loaded:\n" + "\n".join(urls))
        logger.info(f"Loaded {len(urls)} raw URLs from urls.txt")

        max_urls = os.getenv("MAX_URLS_DISPLAYED")
        if max_urls and max_urls.isdigit():
            urls = urls[:int(max_urls)]

        if not urls:
            output_func("[ERROR] No URLs to process. Exiting.")
            return

        processed_info = load_processed_urls()
        logger.warning(f"{len(urls)} URLs remain after filtering .processed_urls")

        selected_urls = prompt_url_selection(urls, processed_info, prompt_func=prompt_func, output_func=output_func, cancel_flag=cancel_flag, non_interactive=non_interactive)
        if not selected_urls:
            output_func("[INFO] No URLs selected. Exiting.")
            return

        if ENABLE_PARALLEL:
            with Pool() as pool:
                pool.starmap(orchestrate_url, [(url, processed_info, prompt_func, output_func, session_id, cancel_flag, non_interactive) for url in selected_urls])
        else:
            for url in selected_urls:
                orchestrate_url(url, processed_info, prompt_func, output_func, session_id, cancel_flag, non_interactive)
        summary = {"success": 0, "fail": 0, "partial": 0, "error": 0, "flagged": 0}
        processed = load_processed_urls()
        for url in selected_urls:
            proc_entry = processed.get(url, {})
            if not isinstance(proc_entry, dict):
                proc_entry = {}
            status = proc_entry.get("status", "unprocessed")
            if status in summary:
                summary[status] += 1
            if proc_entry.get("flagged_for_review"):
                summary["flagged"] += 1

        output_func("\n[SUMMARY]")
        output_func(f"  URLs processed: {len(selected_urls)}")
        output_func(f"  Success: {summary['success']}")
        output_func(f"  Failures: {summary['fail']}")
        output_func(f"  Partial: {summary['partial']}")
        output_func(f"  Errors: {summary['error']}")
        output_func(f"  Flagged for review: {summary['flagged']}")
    except (OperationalError, psycopg2.OperationalError) as db_err:
        output_func(f"[DB ERROR] Could not connect to the database: {db_err}")
        output_func("[FATAL] Database connection failed. Exiting pipeline.")
        sys.exit(1)

if __name__ == "__main__":
    main()