from __future__ import annotations

# ==============================================================
# 🗳️ Smart Elections: HTML Election Results Parser
# ==============================================================
import os
import re
import sys
import threading
from datetime import datetime
from multiprocessing import Pool
from typing import Any, Dict, List

import orjson
import psycopg2
from playwright.sync_api import sync_playwright
from sqlalchemy.exc import OperationalError

from .config import (
    CACHE_LOCK,
    CACHE_RESET,
    ENABLE_AI_ANALYSIS,
    ENABLE_PARALLEL,
    ENABLE_REALTIME_STREAM,
    MAX_URLS_DISPLAYED,
    OUTPUT_DIR,
    PROCESSED_URLS_FILE,
    UPLOADS_DIR,
    URL_LIST_FILE,
)
from .handlers.formats.html_handler import parse as html_handler
from .state_router import get_handler, preload_handler_map
from .utils.browser_utils import (
    autoscroll_until_stable,
    sync_browser_pipeline,
    sync_safe_browser_close,
)
from .utils.download_utils import ensure_input_directory, ensure_output_directory
from .utils.format_router import prompt_and_handle_download
from .utils.logger_singleton import logger, prompt
from .utils.misc_utils import load_processed_urls
from .utils.shared_logic import infer_state_county_from_url, safe_is_set, safe_parse, safe_strip

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
    max_display=20,  # Limit for log display
) -> List[str]:
    msg = "URLs loaded"
    payload = {
        "level": "INFO",
        "type": "input",
        "message": msg,
        "urls": urls[:max_display] + (["... (truncated)"] if len(urls) > max_display else []),
        "processed": processed,
        "session_id": session_id
    }
    logger.info(payload)
    for i, url in enumerate(urls[:max_display]):
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
    if len(urls) > max_display:
        logger.info({
            "level": "INFO",
            "type": "input",
            "message": f"...and {len(urls) - max_display} more URLs not shown.",
            "session_id": session_id
        })

    # Check cancel_flag before prompting
    if cancel_flag is not None and hasattr(cancel_flag, "is_set") and callable(cancel_flag.is_set):
        if cancel_flag.is_set():
            msg = "Selection cancelled before prompt."
            payload = {"level": "INFO","type": "cancel","message": msg,"session_id": session_id}
            logger.info(payload)
            return []

    # Changed prompt text + strip early
    try:
        user_input_stripped = prompt.prompt_input(
            "[PROMPT] Enter URL indices (e.g., 1,3-5) or filter (state:/county: or text): ",
            session_id=session_id,
            context={"urls": urls, "processed": processed}
        ).strip()
    except Exception:
        return []

    # Check cancel_flag after prompt
    if cancel_flag is not None and hasattr(cancel_flag, "is_set") and callable(cancel_flag.is_set):
        if cancel_flag.is_set():
            msg = "Selection cancelled after prompt."
            payload = {"level": "INFO","type": "cancel","message": msg,"session_id": session_id}
            logger.info(payload)
            return []

    if not isinstance(user_input_stripped, str):
        return []
    if not user_input_stripped:
        return []
    if user_input_stripped.lower() == 'all':
        return urls

    # Prioritize pure numeric indices/ranges to avoid "Multiple matches" noise
    # Matches "2", "1,3,5", "1-3", "1,3-5,7"
    if re.fullmatch(r"\d+(?:\s*(?:,\s*\d+|\-\s*\d+))*", user_input_stripped):
        indices = []
        for part in user_input_stripped.split(','):
            part = part.strip()
            if not part:
                continue
            if '-' in part:
                s, e = [p.strip() for p in part.split('-', 1)]
                if s.isdigit() and e.isdigit():
                    indices.extend(range(int(s)-1, int(e)))
            elif part.isdigit():
                idx = int(part) - 1
                if 0 <= idx < len(urls):
                    indices.append(idx)
        indices = sorted(set(i for i in indices if 0 <= i < len(urls)))
        return [urls[i] for i in indices]

    # --- State/county search (regex, fuzzy, or substring) ---
    def search_urls(query, prefix):
        q = query.strip().lower()
        try:
            regex = re.compile(q, re.I)
            matches = [u for u in urls if regex.search(u)]
        except Exception:
            matches = [u for u in urls if q in u.lower()]
        return matches

    if user_input_stripped.lower().startswith("state:"):
        state_query = user_input_stripped[6:]
        matches = search_urls(state_query, "state")
        if matches:
            logger.info({
                "level": "INFO",
                "type": "input",
                "message": f"Matched {len(matches)} URLs for state search: '{state_query}'",
                "session_id": session_id,
                "matches": matches[:max_display] + (["... (truncated)"] if len(matches) > max_display else [])
            })
        return matches

    if user_input_stripped.lower().startswith("county:"):
        county_query = user_input_stripped[7:]
        matches = search_urls(county_query, "county")
        if matches:
            logger.info({
                "level": "INFO",
                "type": "input",
                "message": f"Matched {len(matches)} URLs for county search: '{county_query}'",
                "session_id": session_id,
                "matches": matches[:max_display] + (["... (truncated)"] if len(matches) > max_display else [])
            })
        return matches

    # --- Exact URL match (case-insensitive) ---
    for u in urls:
        if user_input_stripped.lower() == u.lower():
            return [u]

    # --- Partial URL match (case-insensitive substring) ---
    partial_matches = [u for u in urls if user_input_stripped.lower() in u.lower()]
    if len(partial_matches) == 1:
        return partial_matches
    elif len(partial_matches) > 1:
        logger.info({
            "level": "INFO",
            "type": "input",
            "message": f"Multiple matches found for '{user_input_stripped}': {partial_matches}",
            "session_id": session_id
        })
        return partial_matches

    # Fallback indices/ranges (kept for mixed inputs)
    indices = []
    for part in user_input_stripped.split(','):
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

def process_format_override(session_id=None, source_dir='input', output_bypass=False, force_parse_input_file=None, force_parse_format=None) -> bool:
    """
    Manual single-file parse override.
    If source_dir == 'uploads', always prompt for file selection from uploads folder.

    Returns:
      True  -> manual parse succeeded (caller should short-circuit)
      False -> override not engaged (proceed normally)
      None  -> override attempted but failed
    """
    from .utils.format_router import prompt_and_handle_download

    # Engage only when 'uploads' is selected
    if source_dir != 'uploads':
        return False
        # Always prompt for file selection from uploads folder
    input_folder = os.path.abspath(UPLOADS_DIR)
    if not os.path.isdir(input_folder):
        logger.error({
            "level": "ERROR",
            "type": "manual_override",
            "message": f"[ManualOverride] Source directory does not exist: {input_folder}",
            "session_id": session_id,
            "source_dir": source_dir
        })
        return None

    files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    if not files:
        logger.error({
            "level": "ERROR",
            "type": "manual_override",
            "message": "[ManualOverride] No files found in uploads folder.",
            "session_id": session_id
        })
        return None

    logger.info({
        "level": "INFO",
        "type": "manual_override",
        "message": f"[ManualOverride] Found {len(files)} file(s) in 'uploads' folder. Override engaged.",
        "session_id": session_id,
        "source_dir": source_dir
    })

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

    result, handled = prompt_and_handle_download(
        page=None,
        target_url="manual_override",
        rejected_downloads=None,
        session_id=session_id,
        manual_upload_mode=True,
        uploads_dir=input_folder
    )

    if handled:
        output_file_path = None
        if result and isinstance(result, (list, tuple)) and len(result) > 0:
            metadata = result[-1]
            if isinstance(metadata, dict):
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
        logger.info({
            "level": "INFO",
            "type": "manual_override",
            "message": "[ManualOverride] Manual upload parse succeeded via prompt_and_handle_download.",
            "session_id": session_id
        })
        mark_url_processed("manual_override", status="success")
        return True

    logger.error({
        "level": "ERROR",
        "type": "manual_override",
        "message": "[ManualOverride] Manual upload parse failed via prompt_and_handle_download.",
        "session_id": session_id
    })
    return None

def ai_analyze_results(headers, data, contest, metadata, target_url=None, session_id=None):
    if ENABLE_AI_ANALYSIS:
        try:
            from .Context_Integration.Integrity_check import (  # noqa: E402
                analyze_contests,
                print_integrity_summary,
            )

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
            from .Context_Integration.Integrity_check import print_integrity_summary  # noqa: E402

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
            browser, _, page, _ = sync_browser_pipeline(
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
                        sync_safe_browser_close(browser, session_id)
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
                    sync_safe_browser_close(browser, session_id)
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
                sync_safe_browser_close(browser, session_id)
                return

            try:
                autoscroll_until_stable(
                    page,
                    wait_for_selector='table, a[href$=".csv"], a[href$=".json"], [role="table"]',
                    max_total_time=20000,
                    delay_ms=250
                )
            except Exception:
                # Soft-fail: continue; downstream will warn if nothing found
                pass

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
                sync_safe_browser_close(browser, session_id)
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
                sync_safe_browser_close(browser, session_id)
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
                sync_safe_browser_close(browser, session_id)
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
        payload = {"level": "ERROR","type": "exception","message": msg,"session_id": session_id}
        logger.error(payload)
        try:
            choice = prompt.prompt_input(
                "[PROMPT] Error encountered. Retry (r) / Skip (s) ? ",
                validator=lambda x: str(x).lower().strip() in ("r","s"),
                session_id=session_id
            ).strip().lower()
        except Exception:
            choice = "s"
        if choice == "r":
            # simple one-shot retry
            return orchestrate_url(
                target_url,
                processed_info,
                session_id=session_id,
                cancel_flag=cancel_flag,
                output_bypass=output_bypass,
                **kwargs
            )
        mark_url_processed(target_url, status="error", session_id=session_id)
    finally:
        sync_safe_browser_close(browser, session_id)

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
    """
    Main orchestration entrypoint for the HTML Election Parser.

    - If manual_source is 'uploads', always attempt manual override (parse from uploads folder).
      - If override succeeds, short-circuit and exit.
      - If override fails, continue or abort based on continue_on_override_failure.
    - Otherwise, proceed with normal pipeline (input folder or download).
    - Handles parallel and sequential processing.
    - Logs all major steps and errors.
    """
    try:
        logger.info({
            "level": "DEBUG",
            "type": "status",
            "message": f"main() called with manual_source={manual_source}",
            "session_id": session_id
        })

        # --- 1. Manual Upload Override Path ---
        if manual_source == 'uploads':
            override_result = process_format_override(
                session_id=session_id,
                source_dir='uploads',
                output_bypass=output_bypass,
                force_parse_input_file=kwargs.get("force_parse_input_file"),
                force_parse_format=kwargs.get("force_parse_format")
            )
            if override_result is True:
                logger.info({
                    "level": "INFO",
                    "type": "manual_override",
                    "message": "[ManualOverride] Manual upload parse succeeded. Exiting main().",
                    "session_id": session_id
                })
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

        # --- 2. Ensure Directories ---
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

        # --- 3. Load URLs ---
        if urls is None:
            urls = load_urls()
        logger.info({
            "level": "INFO",
            "type": "input",
            "message": f"Loaded {len(urls)} raw URLs from urls.txt",
            "session_id": session_id
        })

        # --- 4. Limit URLs if needed ---
        max_urls = MAX_URLS_DISPLAYED
        if isinstance(max_urls, (int, str)) and str(max_urls).isdigit():
            urls = urls[:int(max_urls)]

        if not urls:
            logger.error({
                "level": "ERROR",
                "type": "input",
                "message": "No URLs to process. Exiting.",
                "session_id": session_id
            })
            return

        # --- 5. Load Processed Info ---
        processed_info = load_processed_urls()
        logger.warning({
            "level": "WARNING",
            "type": "status",
            "message": f"{len(urls)} URLs remain after filtering .processed_urls",
            "session_id": session_id
        })

        # --- 6. Prompt for URL Selection ---
        selected_urls = prompt_url_selection(
            urls,
            processed_info,
            session_id=session_id,
            cancel_flag=cancel_flag
        )
        if not selected_urls:
            logger.info({
                "level": "INFO",
                "type": "input",
                "message": "No URLs selected. Exiting.",
                "session_id": session_id
            })
            return

        # --- 7. Process URLs (Parallel or Sequential) ---
        if ENABLE_PARALLEL:
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

        # --- 8. Summarize Results ---
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

        logger.info({
            "level": "INFO",
            "type": "summary",
            "message": summary,
            "session_id": session_id
        })

    except (OperationalError, psycopg2.OperationalError) as db_err:
        msg = f"DB ERROR: Could not connect to the database: {db_err}"
        logger.error({
            "level": "ERROR",
            "type": "error",
            "message": msg,
            "session_id": session_id
        })
        logger.error({
            "level": "ERROR",
            "type": "fatal",
            "message": "Database connection failed. Exiting pipeline.",
            "session_id": session_id
        })
        sys.exit(1)

if __name__ == "__main__":
    logger.set_mode("cli")
    main()