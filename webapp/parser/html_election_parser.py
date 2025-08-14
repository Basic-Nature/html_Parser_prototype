from __future__ import annotations
# ==============================================================
# 🗳️ Smart Elections: HTML Election Results Parser
# ==============================================================
import os
import orjson
import threading
import sys
import psycopg2
from datetime import datetime
from typing import cast, Dict, Any, List
from multiprocessing import Pool

from playwright.sync_api import sync_playwright, Page
from sqlalchemy.exc import OperationalError
from .handlers.formats.html_handler import parse as html_handler
from .state_router import get_handler, preload_handler_map
from .utils.browser_utils import browser_pipeline, safe_browser_close
from .utils.misc_utils import load_processed_urls
from .utils.download_utils import ensure_input_directory, ensure_output_directory
from .utils.format_router import prompt_and_handle_download
from .utils.shared_logic import (
    infer_state_county_from_url, safe_parse, safe_is_set, safe_filename,
    safe_strip
)
from .Context_Integration.librarian import safe_join
from .utils.logger_singleton import logger, console, prompt
from .config import (
    UPLOADS_DIR,
    CACHE_LOCK, CACHE_RESET,
    ENABLE_PARALLEL, ENABLE_AI_ANALYSIS, ENABLE_REALTIME_STREAM,
    FORCE_PARSE_INPUT_FILE, FORCE_PARSE_FORMAT, MAX_URLS_DISPLAYED,
    INPUT_DIR, OUTPUT_DIR, URL_LIST_FILE, PROCESSED_URLS_FILE
)

if CACHE_RESET and PROCESSED_URLS_FILE.exists():
    logger.warning("Deleting .processed_urls cache for fresh start...")
    PROCESSED_URLS_FILE.unlink()

def load_urls() -> List[str]:
    if not URL_LIST_FILE.exists():
        msg = "No urls.txt found. Please input a URL to append:"
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
) -> List[str]:
    msg = "URLs loaded"
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

    # Check cancel_flag before prompting
    if cancel_flag is not None and hasattr(cancel_flag, "is_set") and callable(cancel_flag.is_set):
        if cancel_flag.is_set():
            msg = "Selection cancelled before prompt."
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

def process_format_override(session_id=None, source_dir='input', output_bypass=False) -> bool:
    """
    Manual single-file parse override.
    Engaged only when BOTH FORCE_PARSE_INPUT_FILE and FORCE_PARSE_FORMAT are truthy.

    Returns:
      True  -> manual parse succeeded (caller should short-circuit)
      False -> override not engaged (proceed normally)
      None  -> override attempted but failed
    """
    from .utils.format_router import route_format_handler
    from .utils.download_utils import ensure_output_directory

    force_parse = FORCE_PARSE_INPUT_FILE
    force_format = FORCE_PARSE_FORMAT
    if not force_parse or not force_format:
        return False  # Not engaged

    if not output_bypass:
        # Ensure output folder exists so handler can write
        try:
            ensure_output_directory()
        except Exception as e:
            logger.error({
                "level": "ERROR",
                "type": "manual_override",
                "message": f"[ManualOverride] Could not ensure output directory: {e}",
                "session_id": session_id
            })
            return None

    # Normalize format (strip leading dot, lowercase)
    force_format_norm = str(force_format).lower().lstrip('.').strip()
    if not force_format_norm:
        logger.error({
            "level": "ERROR",
            "type": "manual_override",
            "message": "[ManualOverride] FORCE_PARSE_FORMAT empty after normalization.",
            "session_id": session_id
        })
        return None

    # Resolve folder
    input_folder = UPLOADS_DIR if source_dir == 'uploads' else INPUT_DIR
    if not os.path.isdir(input_folder):
        logger.error({
            "level": "ERROR",
            "type": "manual_override",
            "message": f"[ManualOverride] Source directory does not exist: {input_folder}",
            "session_id": session_id,
            "source_dir": source_dir
        })
        return None

    # Discover candidate files (case-insensitive extension match)
    files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith(f".{force_format_norm}")
    ]
    if not files:
        logger.error({
            "level": "ERROR",
            "type": "manual_override",
            "message": f"[ManualOverride] No .{force_format_norm} files found in '{source_dir}' folder.",
            "session_id": session_id,
            "source_dir": source_dir
        })
        return None

    logger.warning({
        "level": "INFO",
        "type": "manual_override",
        "message": f"[ManualOverride] Found {len(files)} .{force_format_norm} file(s) in '{source_dir}' folder. Override engaged.",
        "session_id": session_id,
        "source_dir": source_dir,
        "format": force_format_norm
    })

    # List files (stable order)
    files.sort()
    for i, fname in enumerate(files):
        logger.info({
            "level": "INFO",
            "type": "manual_override",
            "message": f"[{i}] {fname}",
            "index": i,
            "file": fname,
            "session_id": session_id
        })

    # Auto-selection logic:
    # If FORCE_PARSE_INPUT_FILE is:
    #   - True / 'first': pick index 0
    #   - Int / numeric string: treat as index
    #   - Otherwise: prompt user
    auto_index = None
    if isinstance(force_parse, bool) and force_parse:
        auto_index = 0
    elif isinstance(force_parse, str):
        if force_parse.lower() == 'first':
            auto_index = 0
        elif force_parse.isdigit():
            auto_index = int(force_parse)
    elif isinstance(force_parse, (int,)):
        auto_index = int(force_parse)

    selected_index = None
    if auto_index is not None:
        if 0 <= auto_index < len(files):
            selected_index = auto_index
            logger.info({
                "level": "INFO",
                "type": "manual_override",
                "message": f"[ManualOverride] Auto-selected index {auto_index}",
                "session_id": session_id
            })
        else:
            logger.warning({
                "level": "WARNING",
                "type": "manual_override",
                "message": f"[ManualOverride] Auto index {auto_index} out of range; falling back to prompt.",
                "session_id": session_id
            })

    if selected_index is None:
        try:
            selection = prompt.prompt_input(
                "[PROMPT] Select a file index to parse:",
                session_id=session_id,
                context={"files": files},
            )
            if not isinstance(selection, str):
                raise ValueError("Non-string selection")
            selection = selection.strip()
            selected_index = int(selection)
            if not (0 <= selected_index < len(files)):
                raise ValueError("Index out of range")
        except (ValueError, EOFError, KeyboardInterrupt):
            logger.error({
                "level": "ERROR",
                "type": "manual_override",
                "message": "[ManualOverride] Invalid selection. Aborting manual parse.",
                "session_id": session_id
            })
            return None

    target_file = safe_filename(files[selected_index])
    handler = route_format_handler(force_format_norm)
    if not handler:
        logger.error({
            "level": "ERROR",
            "type": "manual_override",
            "message": f"[ManualOverride] No format handler found for '{force_format_norm}'",
            "session_id": session_id
        })
        return None

    full_path = safe_join(input_folder, target_file)
    if not os.path.isfile(full_path):
        logger.error({
            "level": "ERROR",
            "type": "manual_override",
            "message": f"[ManualOverride] Selected file missing: {full_path}",
            "session_id": session_id,
            "file": full_path
        })
        return None

    html_context = {
        "manual_file": full_path,
        "manual_source_dir": source_dir,
        "manual_format": force_format_norm
    }
    dummy_page = cast(Page, None)

    logger.info({
        "level": "INFO",
        "type": "manual_override",
        "message": f"[ManualOverride] Parsing file: {target_file}",
        "session_id": session_id,
        "file_path": full_path
    })

    result = safe_parse(handler, dummy_page, html_context, logger=logger)
    if not result or not all(result):
        logger.error({
            "level": "ERROR",
            "type": "manual_override",
            "message": "[ManualOverride] Parsing failed or returned incomplete result.",
            "session_id": session_id,
            "file_path": full_path
        })
        return None

    *_, metadata = result
    if isinstance(metadata, dict) and "output_file" in metadata:
        output_file_path = metadata.get("output_file")
        if output_bypass and output_file_path and os.path.exists(output_file_path):
            try:
                os.remove(output_file_path)
                logger.info({
                    "level": "INFO",
                    "type": "manual_override",
                    "message": f"[ManualOverride] Output bypass active — removed {output_file_path}",
                    "session_id": session_id
                })
            except Exception as e:
                logger.warning({
                    "level": "WARNING",
                    "type": "manual_override",
                    "message": f"[ManualOverride] Failed to remove bypassed file: {e}",
                    "session_id": session_id
                })
    # Single-file summary
    logger.info({
        "level": "INFO",
        "type": "summary",
        "message": {"success": 1, "fail": 0, "partial": 0, "error": 0, "flagged": 0, "mode": "manual_override"},
        "session_id": session_id
    })
    mark_url_processed("manual_override", status="success", file=target_file, format=force_format_norm)
    return True

def ai_analyze_results(headers, data, contest, metadata, target_url=None, session_id=None):
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
            payload = {
                "level": "ERROR",
                "type": "error",
                "message": msg,
                "session_id": session_id
            }
            logger.error(payload)

def stream_results(headers, data, contest, metadata, target_url=None, session_id=None):
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
            msg = f"Streaming results for {target_url}"
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
    output_bypass=False,
    **kwargs
):
    from .Context_Integration.context_coordinator import ContextCoordinator
    rejected_downloads = set()

    msg = f"Navigating to: {target_url}"
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
                p, target_url, cache_exit_callback=mark_url_processed, session_id=session_id
            )
            if cancel_flag is not None and safe_is_set(cancel_flag):
                try:
                    if safe_is_set(cancel_flag):
                        msg = f"Processing cancelled for {target_url}"
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
                payload = {
                    "level": "ERROR",
                    "type": "browser",
                    "message": msg,
                    "session_id": session_id
                }
                logger.error(payload)
                safe_browser_close(browser, session_id)
                return

            result, handled = prompt_and_handle_download(
                page, target_url, rejected_downloads, session_id=session_id
            )
            if handled:
                mark_url_processed(target_url, status="success", session_id=session_id)
                msg = f"Download handled for {target_url}"
                payload = {
                    "level": "INFO",
                    "type": "download",
                    "message": msg,
                    "session_id": session_id
                }
                logger.info(payload)
                safe_browser_close(browser, session_id)
                return

            state, county = infer_state_county_from_url(target_url)
            context = {
                "state": state,
                "county": county,
                "url": target_url,
                "session_id": session_id,
                "output_bypass": output_bypass
            }
            if state:
                preload_handler_map(restrict_to_states=[state])
            else:
                preload_handler_map()
            handler_result = get_handler(
                context,
                url=target_url,
                debug=False,
                fuzzy_cutoff=None,
                session_id=session_id
            )
            handler = handler_result.get("handler") if isinstance(handler_result, dict) else None
            summary = handler_result.get("summary") if isinstance(handler_result, dict) else None

            if summary and isinstance(summary, dict) and summary.get("log"):
                log_entries = summary.get("log")
                for entry in log_entries:
                    payload = {
                        "level": "INFO",
                        "type": "router",
                        "message": entry,
                        "session_id": session_id
                    }
                    logger.info(payload)

            coordinator = ContextCoordinator()

            result = None
            if handler and hasattr(handler, 'parse'):
                result = safe_parse(handler, page, coordinator, context, session_id=session_id, logger=logger, **kwargs)
            else:
                msg = f"[Router] No suitable handler found for {target_url}, using generic HTML handler."
                payload = {
                    "level": "WARNING",
                    "type": "router",
                    "message": msg,
                    "session_id": session_id
                }
                logger.warning(payload)
                result = safe_parse(html_handler, page, coordinator, context, session_id=session_id, logger=logger, **kwargs)

            if not isinstance(result, tuple) or len(result) != 4:
                msg = f"Handler did not return a valid result tuple. (Session: {session_id})"
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

            if all([headers, data, contest, metadata]):
                ai_analyze_results(headers, data, contest, metadata, target_url=target_url, session_id=session_id)
                stream_results(headers, data, contest, metadata, target_url=target_url, session_id=session_id)
                output_file = metadata.get("output_file") if isinstance(metadata, dict) else None
                if output_file:
                    if os.path.exists(output_file):
                        if output_bypass:
                            msg = f"Output bypass active — suppressing file: {output_file}"
                            payload = {
                                "level": "INFO",
                                "type": "output",
                                "message": msg,
                                "session_id": session_id
                            }
                            logger.info(payload)
                            try:
                                os.remove(output_file)
                            except Exception as e:
                                logger.warning({
                                    "level": "WARNING",
                                    "type": "output",
                                    "message": f"Could not remove suppressed file: {e}",
                                    "session_id": session_id
                                })
                        else:
                            msg = f"CSV written to: {output_file} (Session: {session_id})"
                            payload = {
                                "level": "INFO",
                                "type": "output",
                                "message": msg,
                                "session_id": session_id
                            }
                            logger.info(payload)
                    else:
                        msg = f"Output file path returned but file does not exist: {output_file} (Session: {session_id})"
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
                        payload = {
                            "level": "WARNING",
                            "type": "output",
                            "message": msg,
                            "session_id": session_id
                        }
                        logger.warning(payload)
                    else:
                        msg = "[WARN] No output file path returned from parser and no output files found."
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
        payload = {
            "level": "ERROR",
            "type": "exception",
            "message": msg,
            "session_id": session_id
        }
        logger.error(payload)
        mark_url_processed(target_url, status="error", session_id=session_id)
    finally:
        safe_browser_close(browser, session_id)

def _orchestrate_url_worker(args):
    """
    args tuple:
      (url, processed_info, session_id, output_bypass, extra_kwargs_dict)
    cancel_flag intentionally omitted (not pickle-friendly across processes).
    """
    (url, processed_info, session_id, output_bypass, extra_kwargs) = args
    # cancel_flag cannot be shared reliably across processes; ignore here
    orchestrate_url(
        url,
        processed_info,
        session_id=session_id,
        cancel_flag=None,
        output_bypass=output_bypass,
        **(extra_kwargs or {})
    )

def main(
    urls=None,
    session_id=None,
    cancel_flag=None,
    output_bypass=False,
    manual_source='input',
    continue_on_override_failure=True,
    **kwargs
):
    try:
        payload = {"level": "DEBUG", "message": "Entered main()", "session_id": session_id}
        logger.info(payload)

        override_result = process_format_override(
            session_id=session_id,
            source_dir=manual_source,
            output_bypass=output_bypass
        )
        if override_result is True:
            return
        if override_result is None:
            if continue_on_override_failure:
                logger.warning({
                    "level": "WARNING",
                    "type": "manual_override",
                    "message": "[ManualOverride] Override failed; continuing with normal URL pipeline.",
                    "session_id": session_id
                })
            else:
                logger.error({
                    "level": "ERROR",
                    "type": "manual_override",
                    "message": "[ManualOverride] Override failed; aborting as configured.",
                    "session_id": session_id
                })
                return

        ensure_input_directory()
        if not output_bypass:
            ensure_output_directory()
        else:
            logger.info({
                "level": "INFO",
                "type": "output",
                "message": "Output bypass enabled — not creating output directory.",
                "session_id": session_id
            })

        if urls is None:
            urls = load_urls()

        msg = f"Loaded {len(urls)} raw URLs from urls.txt"
        payload = {
            "level": "INFO",
            "type": "input",
            "message": msg,
            "session_id": session_id
        }
        logger.info(payload)

        max_urls = MAX_URLS_DISPLAYED
        if isinstance(max_urls, (int, str)) and str(max_urls).isdigit():
            urls = urls[:int(max_urls)]

        if not urls:
            msg = "No URLs to process. Exiting."
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
        payload = {
            "level": "WARNING",
            "type": "status",
            "message": msg,
            "session_id": session_id
        }
        logger.warning(payload)

        selected_urls = prompt_url_selection(
            urls,
            processed_info,
            session_id=session_id,
            cancel_flag=cancel_flag
        )
        if not selected_urls:
            msg = "No URLs selected. Exiting."
            payload = {
                "level": "INFO",
                "type": "input",
                "message": msg,
                "session_id": session_id
            }
            logger.info(payload)
            return

        if ENABLE_PARALLEL:
            # Note: cancel_flag not passed to subprocesses (would not sync); single-process mode supports cancellation.
            arg_list = [
                (url, processed_info, session_id, output_bypass, kwargs)
                for url in selected_urls
            ]
            with Pool() as pool:
                pool.map(_orchestrate_url_worker, arg_list)
        else:
            for url in selected_urls:
                if cancel_flag and hasattr(cancel_flag, "is_set") and cancel_flag.is_set():
                    logger.info({
                        "level": "INFO",
                        "type": "cancel",
                        "message": "Cancellation requested; stopping remaining URLs.",
                        "session_id": session_id
                    })
                    break
                orchestrate_url(
                    url,
                    processed_info,
                    session_id,
                    cancel_flag,
                    output_bypass=output_bypass,
                    **kwargs
                )

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

        payload = {
            "level": "INFO",
            "type": "summary",
            "message": summary,
            "session_id": session_id
        }
        logger.info(payload)

    except (OperationalError, psycopg2.OperationalError) as db_err:
        msg = f"DB ERROR: Could not connect to the database: {db_err}"
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