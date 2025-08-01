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
from .utils.misc_utils import load_processed_urls
from .utils.download_utils import ensure_input_directory, ensure_output_directory
from .utils.format_router import prompt_and_handle_download
from .utils.shared_logic import infer_state_county_from_url, safe_parse, safe_is_set
from .Context_Integration.librarian import safe_join
from .utils.logger_singleton import logger, console, prompt

# --- Environment & Path Setup ---
load_dotenv()

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

def load_urls() -> List[str]:
    def safe_strip(val):
        return val.strip() if isinstance(val, str) else ""

    if not URL_LIST_FILE.exists():
        msg = "No urls.txt found. Please input a URL to append:"
        if logger.mode == "cli":
            console.print(f"[bold red]\n{msg}")
        else:
            payload = {
                "level": "ERROR",
                "type": "input",
                "message": msg,
            }
            logger.error(payload)
        url = safe_strip(prompt.prompt_input("URL: "))
        if url:
            URL_LIST_FILE.write_text(url + "\n")
            msg = f"Appended URL to urls.txt: {url}"
            if logger.mode == "cli":
                console.print(f"[green]{msg}[/green]")
            else:
                payload = {
                    "level": "INFO",
                    "type": "input",
                    "message": msg,
                }
                logger.info(payload)
        return [url] if url else []

    with URL_LIST_FILE.open('r') as f:
        lines = []
        for line in f:
            line_stripped = safe_strip(line)
            if line_stripped and not line_stripped.startswith("#"):
                lines.append(line_stripped)

    if not lines:
        msg = "urls.txt has no usable URLs. Please input a URL to append:"
        if logger.mode == "cli":
            console.print(f"[bold red]\n{msg}")
        else:
            payload = {
                "level": "ERROR",
                "type": "input",
                "message": msg,
            }
            logger.error(payload)
        url = safe_strip(prompt.prompt_input("URL: "))
        if url:
            with URL_LIST_FILE.open('a') as f_append:
                f_append.write(url + "\n")
            msg = f"Appended URL to urls.txt: {url}"
            if logger.mode == "cli":
                console.print(f"[green]{msg}[/green]")
            else:
                payload = {
                    "level": "INFO",
                    "type": "input",
                    "message": msg,
                }
                logger.info(payload)
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
    cancel_flag: threading.Event = None,
    session_id=None,
    non_interactive=False
) -> List[str]:
    # Mode-aware: URLs loaded
    msg = "URLs loaded"
    if logger.mode == "cli":
        console.panel(msg, title="Status")
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
            console.print(f"[{i+1}] {url} ({status})", style=status_color)
    else:
        payload = {
            "level": "INFO",
            "type": "input",
            "message": msg,
            "urls": urls,
            "processed": processed,
            "session_id": session_id
        }
        logger.info(payload)
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
            payload = {
                "level": "INFO",
                "type": "input",
                "message": f"[{i+1}] {url} ({status})",
                "session_id": session_id,
                "status_color": status_color,
                "status": status,
            }
            logger.info(payload)

    if non_interactive:
        msg = "Non-interactive mode: awaiting selection from frontend or API."
        if logger.mode == "cli":
            console.panel(msg, title="Info", style="cyan")
        else:
            payload = {
                "level": "INFO",
                "type": "input",
                "message": msg,
                "session_id": session_id
            }
            logger.info(payload)
        return []

    # Check cancel_flag before prompting
    if cancel_flag is not None and hasattr(cancel_flag, "is_set") and callable(cancel_flag.is_set):
        if cancel_flag.is_set():
            msg = "Selection cancelled before prompt."
            if logger.mode == "cli":
                console.panel(msg, title="Cancelled", style="yellow")
            else:
                payload = {
                    "level": "INFO",
                    "type": "cancel",
                    "message": msg,
                    "session_id": session_id
                }
                logger.info(payload)
            return []

    prompt_text = "\nEnter indices (comma-separated), 'all', or leave empty to cancel: "
    user_input = prompt.prompt_input(
        prompt_text,
        session_id=session_id,
        context={"urls": urls, "processed": processed}
    )

    # Check cancel_flag after prompt
    if cancel_flag is not None and hasattr(cancel_flag, "is_set") and callable(cancel_flag.is_set):
        if cancel_flag.is_set():
            msg = "Selection cancelled after prompt."
            if logger.mode == "cli":
                console.panel(msg, title="Cancelled", style="yellow")
            else:
                payload = {
                    "level": "INFO",
                    "type": "cancel",
                    "message": msg,
                    "session_id": session_id
                }
                logger.info(payload)
            return []

    if not isinstance(user_input, str):
        return []
    user_input = user_input.strip().lower()
    if not user_input:
        return []
    if user_input == 'all':
        return urls
    indices = []
    for part in user_input.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-', 1)
            if start.isdigit() and end.isdigit():
                indices.extend(range(int(start)-1, int(end)))
        elif part.isdigit():
            idx = int(part) - 1
            if 0 <= idx < len(urls):
                indices.append(idx)
    indices = sorted(set(i for i in indices if 0 <= i < len(urls)))
    return [urls[i] for i in indices]

def process_format_override(session_id=None) -> bool:
    from .utils.format_router import route_format_handler
    force_parse = os.getenv("FORCE_PARSE_INPUT_FILE", "false").lower() == "true"
    force_format = os.getenv("FORCE_PARSE_FORMAT", "").strip().lower()
    if not force_parse or not force_format:
        return False
    input_folder = INPUT_DIR
    files = [f for f in os.listdir(input_folder) if f.endswith(f".{force_format}")]
    if not files:
        msg = f"[ERROR] No .{force_format} files found in 'input' folder."
        if logger.mode == "cli":
            console.panel(msg, title="Error", style="red")
        else:
            payload = {
                "level": "ERROR",
                "type": "input",
                "message": msg,
                "session_id": None
            }
            logger.error(payload)
        return None
    msg = f"Found {len(files)} .{force_format} files in 'input' folder. Manual override enabled."
    if logger.mode == "cli":
        console.panel(msg, title="Manual Override", style="yellow")
        for i, f in enumerate(files):
            console.print(f"[{i}] {f}", style="cyan")
    else:
        payload = {
            "level": "INFO",
            "type": "manual_override",
            "message": msg,
            "session_id": session_id
        }
        logger.warning(payload)
        for i, f in enumerate(files):
            payload = {
                "level": "INFO",
                "type": "info",
                "message": f"[{i}] {f}",
                "session_id": session_id
            }
            logger.info(payload)
    try:
        selection = prompt.prompt_input(
            "[PROMPT] Select a file index to parse:",
            session_id=session_id,
            context={"files": files},
        ).strip()
        index = int(selection)
        if not (0 <= index < len(files)):
            raise ValueError("Invalid file index")
        target_file = safe_filename(files[index])
    except (IndexError, ValueError, EOFError, KeyboardInterrupt):
        msg = "[ERROR] Invalid selection. Aborting manual parse."
        if logger.mode == "cli":
            console.panel(msg, title="Error", style="red")
        else:
            payload = {
                "level": "ERROR",
                "type": "error",
                "message": msg,
                "session_id": session_id
            }
            logger.error(payload)
        return None
    handler = route_format_handler(force_format)
    if not handler:
        msg = f"[ERROR] No format handler found for '{force_format}'"
        if logger.mode == "cli":
            console.panel(msg, title="Error", style="red")
        else:
            payload = {
                "level": "ERROR",
                "type": "error",
                "message": msg,
                "session_id": session_id
            }
            logger.error(payload)
        return None
    full_path = safe_join(input_folder, target_file)
    html_context = {"manual_file": full_path}
    dummy_page = cast(Page, None)
    result = safe_parse(handler, dummy_page, html_context, logger=logger)
    if result and all(result):
        *_, metadata = result
        if "output_file" in metadata:
            msg = f"Manual override parsing completed for {target_file}"
            if logger.mode == "cli":
                console.panel(msg, title="Manual Override", style="green")
            else:
                payload = {
                    "level": "INFO",
                    "type": "manual_override",
                    "message": msg,
                    "session_id": session_id,
                    "output_file": metadata["output_file"],
                    "metadata": metadata
                }
                logger.info(payload)
        else:
            msg = f"Manual override parsing completed for {target_file}, but no output file was generated."
            if logger.mode == "cli":
                console.panel(msg, title="Manual Override", style="yellow")
            else:
                payload = {
                    "level": "WARNING",
                    "type": "manual_override",
                    "message": msg,
                    "session_id": session_id,
                    "metadata": metadata
                }
                logger.warning(payload)
        mark_url_processed("manual_override", status="success")
        return True
    else:
        msg = "[ERROR] Manual parsing failed or returned no data."
        if logger.mode == "cli":
            console.panel(msg, title="Error", style="red")
        else:
            logger.error(msg)
        return None

def ai_analyze_results(headers, data, contest, metadata, target_url=None, session_id=None):
    """
    Uses advanced NLP and ML utilities to analyze results for anomalies and integrity issues.
    Logs findings and flags suspicious contests.
    """
    if ENABLE_AI_ANALYSIS:
        try:
            from .Context_Integration.Integrity_check import analyze_contests, print_integrity_summary

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

            results = analyze_contests(contests)
            anomalies = results.get("ml_anomalies", [])
            flagged = results.get("flagged_suspicious", [])
            integrity_issues = results.get("integrity_issues", [])
            summary_stats = results.get("summary_stats", {})

            if anomalies or flagged or integrity_issues:
                msg = "AI analysis results"
                if logger.mode == "cli":
                    console.panel(msg, title="AI Analysis", style="red")
                    console.print(f"Anomalies: {anomalies}")
                    console.print(f"Flagged: {flagged}")
                    console.print(f"Integrity Issues: {integrity_issues}")
                    console.print(f"Summary Stats: {summary_stats}")
                else:
                    payload_1 = {
                        "level": "ERROR",
                        "type": "ai_analysis",
                        "message": msg,
                        "session_id": session_id,
                        "anomalies": anomalies,
                        "flagged": flagged,
                        "integrity_issues": integrity_issues,
                        "summary_stats": summary_stats,
                        "metadata": metadata
                    }
                    logger.error(payload_1)
                    payload_2 = {
                        "level": "INFO",
                        "type": "info",
                        "message": f"AI analysis completed for {target_url}",
                        "session_id": session_id,
                        "anomalies": anomalies,
                        "flagged": flagged,
                        "integrity_issues": integrity_issues,
                        "summary_stats": summary_stats,
                        "metadata": metadata
                    }
                    logger.warning(payload_2)
                print_integrity_summary(contests)
            else:
                msg = f"No anomalies or suspicious contests detected for {target_url}"
                if logger.mode == "cli":
                    console.panel(msg, title="AI Analysis", style="green")
                else:
                    payload = {
                        "level": "INFO",
                        "type": "info",
                        "message": msg,
                        "session_id": session_id,
                        "metadata": metadata
                    }
                    logger.info(payload)
        except Exception as e:
            msg = f"[AI] Analysis failed: {e}"
            if logger.mode == "cli":
                console.panel(msg, title="AI Analysis Error", style="red")
            else:
                logger.error(msg)

def stream_results(headers, data, contest, metadata, target_url=None, session_id=None):
    """
    Streams results in real-time if enabled, using rich output and context-aware formatting.
    """
    if ENABLE_REALTIME_STREAM:
        try:
            from .Context_Integration.Integrity_check import print_integrity_summary

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
            msg = f"Streaming results for {target_url}"
            if logger.mode == "cli":
                console.panel(msg, title="Stream", style="cyan")
            else:
                payload = {
                    "level": "INFO",
                    "type": "stream",
                    "message": msg,
                    "session_id": session_id,
                    "contests": contests,
                    "metadata": metadata
                }
                logger.info(payload)
            print_integrity_summary(contests)
            # Optionally, stream metadata and summary stats if needed
            msg = f"Streaming results for {target_url}"
            if logger.mode == "cli":
                console.print(msg)
            else:
                payload = {
                    "level": "INFO",
                    "type": "info",
                    "message": msg,
                    "session_id": session_id,
                    "metadata": metadata
                }
                logger.info(payload)
        except Exception as e:
            msg = f"[STREAM] Streaming failed: {e}"
            if logger.mode == "cli":
                console.panel(msg, title="Stream Error", style="red")
            else:
                payload = {
                    "level": "ERROR",
                    "type": "error",
                    "message": msg,
                    "session_id": session_id
                }
                logger.error(payload)

def orchestrate_url(
    target_url,
    processed_info,
    session_id=None,
    cancel_flag=None,
    non_interactive=False,
    **kwargs
):
    from .Context_Integration.context_coordinator import ContextCoordinator
    rejected_downloads = set()

    # Mode-aware: Navigating to URL
    msg = f"Navigating to: {target_url}"
    if logger.mode == "cli":
        console.panel(msg, title="Status")
    else:
        payload = {
            "level": "INFO",
            "type": "status",
            "message": msg,
            "session_id": session_id
        }
        logger.info(payload)

    browser = page = None
    try:
        with sync_playwright() as p:
            browser, _, page, _ = browser_pipeline(
                p, target_url, cache_exit_callback=mark_url_processed, non_interactive=non_interactive, session_id=session_id
            )
            # cancel_flag check
            if cancel_flag is not None and safe_is_set(cancel_flag):
                try:
                    if safe_is_set(cancel_flag):
                        msg = f"Processing cancelled for {target_url}"
                        if logger.mode == "cli":
                            console.panel(msg, title="Cancelled", style="yellow")
                        else:
                            payload = {
                                "level": "INFO",
                                "type": "cancel",
                                "message": msg,
                                "session_id": session_id
                            }
                            logger.info(payload)
                        safe_browser_close(browser, session_id)
                        return
                except Exception as e:
                    msg = f"Exception during cancel_flag check: {e}"
                    if logger.mode == "cli":
                        console.panel(msg, title="Cancel Error", style="yellow")
                    else:
                        payload = {
                            "level": "WARNING",
                            "type": "cancel",
                            "message": msg,
                            "session_id": session_id
                        }
                        logger.warning(payload)
                    safe_browser_close(browser, session_id)
                    return

            if not page:
                msg = f"Could not open page for {target_url}"
                if logger.mode == "cli":
                    console.panel(msg, title="Browser Error", style="red")
                else:
                    payload = {
                        "level": "ERROR",
                        "type": "browser",
                        "message": msg,
                        "session_id": session_id
                    }
                    logger.error(payload)
                safe_browser_close(browser, session_id)
                return

            # 1. Prompt for downloadable format and handle if chosen
            result, handled = prompt_and_handle_download(
                page, target_url, rejected_downloads, non_interactive=non_interactive, session_id=session_id
            )
            if handled:
                mark_url_processed(target_url, status="success", session_id=session_id)
                msg = f"Download handled for {target_url}"
                if logger.mode == "cli":
                    console.panel(msg, title="Download", style="green")
                else:
                    payload = {
                        "level": "INFO",
                        "type": "download",
                        "message": msg,
                        "session_id": session_id
                    }
                    logger.info(payload)
                safe_browser_close(browser, session_id)
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
                for entry in log_entries:
                    if logger.mode == "cli":
                        console.panel(entry, title="Router", style="cyan")
                    else:
                        payload = {
                            "level": "INFO",
                            "type": "router",
                            "message": entry,
                            "session_id": session_id
                        }
                        logger.info(payload)

            # 4. Prepare coordinator (for handler use)
            coordinator = ContextCoordinator()

            # 5. Call handler (handler is responsible for all DOM/context scanning)
            result = None
            if handler and hasattr(handler, 'parse'):
                result = safe_parse(handler, page, coordinator, context, session_id=session_id, non_interactive=non_interactive, logger=logger, **kwargs)
            else:
                msg = f"[Router] No suitable handler found for {target_url}, using generic HTML handler."
                if logger.mode == "cli":
                    console.panel(msg, title="Router", style="yellow")
                else:
                    payload = {
                        "level": "WARNING",
                        "type": "router",
                        "message": msg,
                        "session_id": session_id
                    }
                    logger.warning(payload)
                result = safe_parse(html_handler, page, coordinator, context, session_id=session_id, non_interactive=non_interactive, logger=logger, **kwargs)

            # 6. Validate result
            if not isinstance(result, tuple) or len(result) != 4:
                msg = f"Handler did not return a valid result tuple. (Session: {session_id})"
                if logger.mode == "cli":
                    console.panel(msg, title="Handler Error", style="red")
                else:
                    payload = {
                        "level": "ERROR",
                        "type": "handler",
                        "message": msg,
                        "session_id": session_id
                    }
                    logger.error(payload)
                mark_url_processed(target_url, status="fail", session_id=session_id)
                safe_browser_close(browser, session_id)
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
                    msg = f"[Batch Mode] Coordinator batch handling failed: {e} (Session: {session_id})"
                    if logger.mode == "cli":
                        console.panel(msg, title="Batch Error", style="red")
                    else:
                        payload = {
                            "level": "ERROR",
                            "type": "batch",
                            "message": msg,
                            "session_id": session_id
                        }
                        logger.error(payload)
                    mark_url_processed(target_url, status="error", session_id=session_id)
                safe_browser_close(browser, session_id)
                return

            # 8. Single result (non-batch)
            if all([headers, data, contest, metadata]):
                ai_analyze_results(headers, data, contest, metadata, target_url=target_url, session_id=session_id)
                stream_results(headers, data, contest, metadata, target_url=target_url, session_id=session_id)
                output_file = metadata.get("output_file") if isinstance(metadata, dict) else None
                if output_file:
                    if os.path.exists(output_file):
                        msg = f"CSV written to: {output_file} (Session: {session_id})"
                        if logger.mode == "cli":
                            console.panel(msg, title="Output", style="green")
                        else:
                            payload = {
                                "level": "INFO",
                                "type": "output",
                                "message": msg,
                                "session_id": session_id
                            }
                            logger.info(payload)
                    else:
                        msg = f"Output file path returned but file does not exist: {output_file} (Session: {session_id})"
                        if logger.mode == "cli":
                            console.panel(msg, title="Output Warning", style="yellow")
                        else:
                            payload = {
                                "level": "WARNING",
                                "type": "output",
                                "message": msg,
                                "session_id": session_id
                            }
                            logger.warning(payload)
                else:
                    output_dir = metadata.get("output_dir") if isinstance(metadata, dict) else OUTPUT_DIR
                    possible_files = []
                    if os.path.isdir(output_dir):
                        for f in os.listdir(output_dir):
                            if f.endswith(".csv") or f.endswith(".json"):
                                possible_files.append(os.path.join(output_dir, f))
                    if possible_files:
                        msg = f"No output file path returned from parser, but found files: {', '.join(possible_files)} (Session: {session_id})"
                        if logger.mode == "cli":
                            console.panel(msg, title="Output Warning", style="yellow")
                        else:
                            payload = {
                                "level": "WARNING",
                                "type": "output",
                                "message": msg,
                                "session_id": session_id
                            }
                            logger.warning(payload)
                    else:
                        msg = "[WARN] No output file path returned from parser and no output files found."
                        if logger.mode == "cli":
                            console.panel(msg, title="Output Warning", style="yellow")
                        else:
                            payload = {
                                "level": "WARNING",
                                "type": "output",
                                "message": msg,
                                "session_id": session_id
                            }
                            logger.warning(payload)
                mark_url_processed(target_url, status="success", session_id=session_id)
            else:
                msg = f"Incomplete result structure for {target_url} — skipping CSV write. (Session: {session_id})"
                if logger.mode == "cli":
                    console.panel(msg, title="Output Warning", style="yellow")
                else:
                    payload = {
                        "level": "WARNING",
                        "type": "output",
                        "message": msg,
                        "session_id": session_id
                    }
                    logger.warning(payload)
                mark_url_processed(target_url, status="partial", session_id=session_id)

    except Exception as e:
        msg = f"Exception while processing {target_url}: {e}"
        if logger.mode == "cli":
            console.panel(msg, title="Exception", style="red")
        else:
            payload = {
                "level": "ERROR",
                "type": "exception",
                "message": msg,
                "session_id": session_id
            }
            logger.error(payload)
        mark_url_processed(target_url, status="error", session_id=session_id)
    finally:
        # browser close (only once)
        safe_browser_close(browser, session_id)

def main(session_id=None, cancel_flag=None, non_interactive=False, **kwargs):
    try:
        if process_format_override(session_id=session_id):
            return

        ensure_input_directory()
        ensure_output_directory()

        urls = load_urls()

        # Mode-aware output for "Loaded X raw URLs..."
        msg = f"Loaded {len(urls)} raw URLs from urls.txt"
        if logger.mode == "cli":
            console.panel(msg, title="Status")
        else:
            payload = {
                "level": "INFO",
                "type": "input",
                "message": msg,
                "session_id": session_id
            }
            logger.info(payload)

        max_urls = os.getenv("MAX_URLS_DISPLAYED")
        if max_urls and max_urls.isdigit():
            urls = urls[:int(max_urls)]

        if not urls:
            msg = "No URLs to process. Exiting."
            if logger.mode == "cli":
                console.panel(msg, title="Error", style="red")
            else:
                payload = {
                    "level": "ERROR",
                    "type": "input",
                    "message": msg,
                    "session_id": session_id
                }
                logger.error(payload)
            return

        processed_info = load_processed_urls()
        msg = f"{len(urls)} URLs remain after filtering .processed_urls"
        if logger.mode == "cli":
            console.panel(msg, title="Warning", style="yellow")
        else:
            payload = {
                "level": "WARNING",
                "type": "status",
                "message": msg,
                "session_id": session_id
            }
            logger.warning(payload)

        selected_urls = prompt_url_selection(
            urls, processed_info,
            session_id=session_id,
            cancel_flag=cancel_flag,
            non_interactive=non_interactive
        )
        if not selected_urls:
            msg = "No URLs selected. Exiting."
            if logger.mode == "cli":
                console.panel(msg, title="Info", style="cyan")
            else:
                payload = {
                    "level": "INFO",
                    "type": "input",
                    "message": msg,
                    "session_id": session_id
                }
                logger.info(payload)
            return

        if ENABLE_PARALLEL:
            with Pool() as pool:
                pool.starmap(orchestrate_url, [(url, processed_info, session_id, cancel_flag, non_interactive, *kwargs.values()) for url in selected_urls])
        else:
            for url in selected_urls:
                orchestrate_url(url, processed_info, session_id, cancel_flag, non_interactive, **kwargs)

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

        # Mode-aware output for summary
        if logger.mode == "cli":
            console.panel(f"Summary: {summary}", title="Summary", style="green")
        else:
            payload = {
                "level": "INFO",
                "type": "summary",
                "message": summary,
                "session_id": session_id
            }
            logger.info(payload)

    except (OperationalError, psycopg2.OperationalError) as db_err:
        msg = f"DB ERROR: Could not connect to the database: {db_err}"
        if logger.mode == "cli":
            console.panel(msg, title="Database Error", style="red")
            console.panel("Database connection failed. Exiting pipeline.", title="Fatal", style="red")
        else:
            payload = {
                "level": "ERROR",
                "type": "error",
                "message": msg,
                "session_id": session_id
            }
            logger.error(payload)
            payload = {
                "level": "ERROR",
                "type": "fatal",
                "message": "Database connection failed. Exiting pipeline.",
                "session_id": session_id
            }
            logger.error(payload)
        sys.exit(1)

if __name__ == "__main__":
    logger.set_mode("cli")
    main()