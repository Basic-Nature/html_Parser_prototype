from __future__ import annotations

# ==============================================================
# 🗳️ Smart Elections: HTML Election Results Parser
# ==============================================================
import os
import re
import threading
from collections import Counter, defaultdict
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
    ENABLE_SELENIUM_FALLBACK,
    INPUT_DIR,
    MAX_URLS_DISPLAYED,
    NAV_MAX_ATTEMPTS,
    NAV_TIMEOUT_PLAYWRIGHT_MS,
    NAV_TIMEOUT_SELENIUM_MS,
    OUTPUT_DIR,
    PROCESSED_URLS_FILE,
    UPLOADS_DIR,
    URL_LIST_FILE,
)
from .navigator import NavigationInstructionRunner, NavigationRecipeStore
from .navigator.dom_snapshot import snapshot_mode_pipeline
from .state_router import get_handler, preload_handler_map
from .Context_Integration.librarian import get_safe_log_path
from .utils.browser_utils import (
    SCROLL_METRIC_KEYS,
    TABLE_DISCOVERY_SELECTOR,
    autoscroll_until_stable,
    safe_content,
    safe_count,
    safe_locator,
    safe_query_selector_all,
    sync_browser_pipeline,
    sync_safe_browser_close,
)
from .utils.captcha_tools import detect_cloudflare_challenge
from .utils.download_utils import ensure_input_directory, ensure_output_directory
from .utils.dynamic_table_extractor import dynamic_table_extractor
from .utils.format_router import prompt_and_handle_download, route_format_handler
from .utils.logger_singleton import logger, prompt
from .utils.misc_utils import extract_url_and_label, load_processed_urls
from .utils.output_utils import finalize_election_output
from .utils.seleniumbase_launcher import SELENIUMBASE_AVAILABLE, close_driver, launch_browser
from .utils.shared_logic import (
    infer_state_county_from_url,
    safe_is_set,
    safe_parse,
    safe_slug,
    safe_strip,
)
from .utils.table_builder import build_table_noninteractive
from .utils.telemetry import emit_telemetry_event
from .utils.telemetry_agg import increment_counter
from .utils.url_trust_scorer import (
    compute_trust_score,
    should_quarantine,
    should_reject,
    should_use_snapshot_mode,
)

if CACHE_RESET and PROCESSED_URLS_FILE.exists():
    logger.warning("Deleting .processed_urls cache for fresh start...")
    PROCESSED_URLS_FILE.unlink()

_navigation_store = NavigationRecipeStore()
NAVIGATION_RUNNER = NavigationInstructionRunner(_navigation_store)


def _close_browser_quietly(browser, session_id=None) -> None:
    """Attempt to close Playwright browser safely; log non-fatal errors.

    Use this helper everywhere instead of calling `sync_safe_browser_close`
    directly so closing errors won't raise and will be logged uniformly.
    """
    try:
        if browser is not None:
            sync_safe_browser_close(browser, session_id=session_id)
    except Exception as exc:
        try:
            logger.warning({
                "level": "WARNING",
                "type": "browser",
                "message": f"Failed to close browser cleanly: {exc}",
                "session_id": session_id,
            })
        except Exception:
            # Best-effort logging; do not raise
            pass


def _sanitize_error_metadata(metadata: dict) -> dict:
    allowed_keys = {
        "error",
        "exception",
        "exception_type",
        "handler",
        "handler_module",
        "handler_qualname",
        "arg_types",
        "kwarg_keys",
        "traceback",
        "session_id",
    }
    safe_meta: dict[str, Any] = {}
    for key in allowed_keys:
        if key in metadata:
            value = metadata.get(key)
            if isinstance(value, str) and len(value) > 4000:
                value = f"{value[:4000]}\n...truncated..."
            safe_meta[key] = value
    return safe_meta


def _log_session_exception_metadata(session_id: str | None, payload: dict) -> None:
    if not session_id:
        return
    try:
        filename = f"{session_id}.ndjson" if session_id.startswith("sess_") else f"sess_{session_id}.ndjson"
        log_path = get_safe_log_path(filename)
        with open(log_path, "ab") as handle:
            handle.write(orjson.dumps(payload) + b"\n")
    except Exception:
        pass


def _count_dom_table_rows(page) -> int:
    if page is None:
        return 0
    try:
        # Prefer Playwright locator/element-handle APIs to avoid injecting JS via evaluate
        total = 0
        try:
            nodes = []
            if hasattr(page, "query_selector_all"):
                nodes = page.query_selector_all(TABLE_DISCOVERY_SELECTOR)
            else:
                nodes = safe_query_selector_all(page, TABLE_DISCOVERY_SELECTOR)
        except Exception:
            nodes = safe_query_selector_all(page, TABLE_DISCOVERY_SELECTOR)

        for tbl in nodes:
            try:
                if hasattr(tbl, "query_selector_all"):
                    trs = tbl.query_selector_all("tr")
                    total += len(trs) if trs is not None else 0
                else:
                    # If ElementHandle lacks query_selector_all, try safe_locator within its context
                    locator = safe_locator(tbl, "tr")
                    total += safe_count(locator)
            except Exception:
                continue
        return int(total)
    except Exception:
        return 0


def load_urls(*, allowlist_bypass: bool = False) -> List[str]:
    if not URL_LIST_FILE.exists():
        msg = "No urls.txt found. Please input a URL to append:"
        payload = {
            "level": "ERROR",
            "type": "input",
            "message": msg,
        }
        logger.error(payload)
        user_input = safe_strip(prompt.prompt_input("URL: "))
        if user_input:
            u, lbl = extract_url_and_label(user_input, allowlist_bypass=allowlist_bypass)
            write_val = u or user_input
            URL_LIST_FILE.write_text(write_val + "\n")
            msg = f"Appended URL to urls.txt: {write_val}"
            payload = {
                "level": "INFO",
                "type": "input",
                "message": msg,
            }
            logger.info(payload)
        return [write_val] if user_input else []

    with URL_LIST_FILE.open('r', encoding='utf-8') as f:
        lines = []
        for raw_line in f:
            line_stripped = safe_strip(raw_line)
            if not line_stripped or line_stripped.startswith("#"):
                continue
            u, lbl = extract_url_and_label(line_stripped, allowlist_bypass=allowlist_bypass)
            if u:
                lines.append(u)
            else:
                # keep the raw line if no URL extracted (back-compat)
                lines.append(line_stripped)

    if not lines:
        msg = "urls.txt has no usable URLs. Please input a URL to append:"
        payload = {
            "level": "ERROR",
            "type": "input",
            "message": msg,
        }
        logger.error(payload)
        user_input = safe_strip(prompt.prompt_input("URL: "))
        if user_input:
            u, lbl = extract_url_and_label(user_input, allowlist_bypass=allowlist_bypass)
            write_val = u or user_input
            with URL_LIST_FILE.open('a', encoding='utf-8') as f_append:
                f_append.write(write_val + "\n")
            msg = f"Appended URL to urls.txt: {write_val}"
            payload = {
                "level": "INFO",
                "type": "input",
                "message": msg,
            }
            logger.info(payload)
            return [write_val]
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
    try:
        # Emit telemetry event for processed URL (best-effort)
        emit_telemetry_event("url_processed", entry)
    except Exception:
        pass
    try:
        # Update lightweight aggregation counters
        increment_counter('processed_total', 1)
        if isinstance(status, str):
            s = status.lower()
            if s in ('success',):
                increment_counter('processed_success', 1)
            elif s in ('fail', 'error'):
                increment_counter('processed_fail', 1)
            elif s in ('partial',):
                increment_counter('processed_partial', 1)
            elif s in ('cancelled', 'cancel'):
                increment_counter('processed_cancelled', 1)
        # flag fallbacks
        if isinstance(metadata, dict) and metadata.get('fallback'):
            increment_counter('fallbacks', 1)
        # track tables_seen if present
        try:
            tbls = int(metadata.get('tables_seen') or 0)
            if tbls:
                increment_counter('tables_seen_total', tbls)
        except Exception:
            pass
    except Exception:
        pass

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

def process_format_override(
    session_id=None,
    source_dir='input',
    output_bypass=False,
    force_parse_input_file=None,
    force_parse_format=None,
    cancel_flag=None,
    **kwargs,
) -> bool:
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

    if force_parse_input_file:
        forced_rel = str(force_parse_input_file).replace("\\", "/").strip("/")
        forced_path = os.path.normpath(os.path.join(input_folder, forced_rel))
        if forced_path.startswith(input_folder) and os.path.isfile(forced_path):
            fmt = (force_parse_format or os.path.splitext(forced_path)[1].lstrip(".")).lower()
            if not fmt:
                logger.error({
                    "level": "ERROR",
                    "type": "manual_override",
                    "message": f"[ManualOverride] Could not infer format for {forced_rel}.",
                    "session_id": session_id
                })
                return None
            handler = route_format_handler(fmt)
            if not handler:
                logger.error({
                    "level": "ERROR",
                    "type": "manual_override",
                    "message": f"[ManualOverride] No handler for format: {fmt}",
                    "session_id": session_id
                })
                return None
            logger.info({
                "level": "INFO",
                "type": "manual_override",
                "message": f"[ManualOverride] Parsing selected upload: {forced_rel}",
                "session_id": session_id
            })
            headers, rows, contest, metadata = safe_parse(
                handler,
                page=None,
                manual_file=forced_path,
                source_url="manual_override",
                logger=logger,
                session_id=session_id,
                cancel_flag=cancel_flag,
                **kwargs,
            )
            if isinstance(metadata, dict) and metadata.get("error"):
                logger.error({
                    "level": "ERROR",
                    "type": "manual_override",
                    "message": f"[ManualOverride] Handler error: {metadata.get('error')}",
                    "session_id": session_id
                })
                return None
            output_file_path = None
            # metadata already set from safe_parse
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
                except Exception as exc:
                    logger.warning({
                        "level": "WARNING",
                        "type": "manual_override",
                        "message": f"[ManualOverride] Failed to remove bypassed file: {exc}",
                        "session_id": session_id
                    })
            mark_url_processed("manual_override", status="success", session_id=session_id)
            logger.info({
                "level": "INFO",
                "type": "manual_override",
                "message": "[ManualOverride] Manual upload parse succeeded via direct selection.",
                "session_id": session_id
            })
            return True
        else:
            logger.warning({
                "level": "WARNING",
                "type": "manual_override",
                "message": f"[ManualOverride] Forced upload file not found: {force_parse_input_file}",
                "session_id": session_id
            })

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
        uploads_dir=input_folder,
        cancel_flag=cancel_flag,
        **kwargs,
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

def ai_analyze_results(headers, data, contest, metadata, target_url=None, session_id=None, trust_factors=None, privilege_tier=None):
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

            results = analyze_contests(contests, trust_factors=trust_factors, privilege_tier=privilege_tier)
            anomalies = results.get("ml_anomalies", [])
            flagged = results.get("flagged_suspicious", [])
            integrity_issues = results.get("integrity_issues", [])
            summary_stats = results.get("summary_stats", {})
            tier_summary = results.get("tier_summary", {})

            contest_count = len(contests) if isinstance(contests, list) else 0
            anomalies_count = len(anomalies) if isinstance(anomalies, list) else 0
            semantic_mismatch_count = len(flagged) if isinstance(flagged, list) else 0
            denom = max(1, contest_count)
            anomaly_rate = anomalies_count / denom
            semantic_rate = semantic_mismatch_count / denom
            weighted_score = (2.0 / 3.0) * anomaly_rate + (1.0 / 3.0) * semantic_rate

            if isinstance(metadata, dict):
                metadata["audit_signals"] = {
                    "contest_count": contest_count,
                    "anomaly_count": anomalies_count,
                    "semantic_mismatch_count": semantic_mismatch_count,
                    "anomaly_rate": anomaly_rate,
                    "semantic_mismatch_rate": semantic_rate,
                    "audit_weighted_score": weighted_score,
                    "weights": {
                        "anomaly": 2.0 / 3.0,
                        "semantic_mismatch": 1.0 / 3.0,
                    },
                }

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
                    "tier_summary": tier_summary,
                    "audit_signals": metadata.get("audit_signals") if isinstance(metadata, dict) else None,
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
                    "tier_summary": tier_summary,
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
                    "tier_summary": tier_summary,
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


def _read_text_file_with_fallback(path: str) -> str | None:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with open(path, "r", encoding=encoding) as handle:
                return handle.read()
        except UnicodeDecodeError:
            continue
        except Exception:
            return None
    try:
        with open(path, "rb") as handle:
            return handle.read().decode("utf-8", "ignore")
    except Exception:
        return None


def _extract_text_blocks(html_text: str, max_rows: int = 200) -> tuple[list[str], list[Dict[str, Any]]]:
    def _is_numeric_like(value: str) -> bool:
        if not value:
            return False
        token = value.strip().replace(",", "")
        if token.endswith("%"):
            token = token[:-1]
        token = token.strip()
        return bool(re.fullmatch(r"-?\d+(\.\d+)?", token))

    def _guess_column_name(index: int, values: List[str], delimiter: str) -> str:
        clean_vals = [v for v in (val.strip() for val in values) if v]
        if not clean_vals:
            return f"Column{index+1}"
        percent_hits = sum(1 for v in clean_vals if "%" in v)
        numeric_hits = sum(1 for v in clean_vals if _is_numeric_like(v))
        party_tokens = {"DEM", "DEMOCRAT", "REPUBLICAN", "REP", "IND", "NP", "LIB", "GREEN"}
        party_hits = sum(1 for v in clean_vals if v.strip().upper() in party_tokens)

        majority_threshold = max(1, int(len(clean_vals) * 0.5))

        if delimiter == "colon" and index == 0:
            return "Label"
        if delimiter == "colon" and index == 1:
            if percent_hits >= majority_threshold:
                return "Percent"
            if numeric_hits >= majority_threshold:
                return "Value"
            return "Value"
        if percent_hits >= majority_threshold:
            return "Percent"
        if numeric_hits >= majority_threshold:
            return "Value"
        if party_hits >= majority_threshold:
            return "Party"
        if index == 0:
            return "Label"
        return f"Column{index+1}"

    def _detect_structured_lines(section_lines: List[str]) -> tuple[str, List[List[str]]] | None:
        cleaned = [line.strip() for line in section_lines if line and line.strip()]
        if len(cleaned) < 2:
            return None
        delimiter_specs = [
            ("colon", r"\s*:\s*"),
            ("pipe", r"\s*\|\s*"),
            ("dash", r"\s*[-\u2013\u2014]+\s*"),
            ("arrow", r"\s*->\s*"),
            ("semicolon", r"\s*;\s*"),
            ("comma", r"\s*,\s*"),
            ("whitespace", r"\s{2,}"),
        ]
        for name, pattern in delimiter_specs:
            split_rows: List[List[str]] = []
            lengths: List[int] = []
            for line in cleaned:
                parts = [part.strip() for part in re.split(pattern, line) if part.strip()]
                if len(parts) < 2:
                    continue
                split_rows.append(parts)
                lengths.append(len(parts))
            if not split_rows:
                continue
            length_counts = Counter(lengths)
            most_common = length_counts.most_common(1)[0]
            common_len, occurrences = most_common
            if occurrences < 2 or common_len > 8:
                continue
            filtered = [parts for parts in split_rows if len(parts) == common_len]
            if len(filtered) >= 2:
                return name, filtered
        whitespace_rows: List[List[str]] = []
        for line in cleaned:
            tokens = [tok.strip() for tok in line.split() if tok.strip()]
            if len(tokens) < 2:
                continue
            numeric_tail = sum(1 for tok in tokens if _is_numeric_like(tok) or tok.endswith("%"))
            if numeric_tail >= 1:
                whitespace_rows.append(tokens)
        if whitespace_rows:
            length_counts = Counter(len(row) for row in whitespace_rows)
            most_common = length_counts.most_common(1)[0]
            common_len, occurrences = most_common
            if occurrences >= 2 and common_len <= 8:
                filtered = [row for row in whitespace_rows if len(row) == common_len]
                if len(filtered) >= 2:
                    return "whitespace", filtered
        return None

    def _build_structured_rows(sections: Dict[str, List[str]], limit: int) -> tuple[list[str], list[Dict[str, Any]]]:
        structured: list[Dict[str, Any]] = []
        column_order: list[str] = []
        for section, lines in sections.items():
            detection = _detect_structured_lines(lines)
            if not detection:
                continue
            delimiter_name, parsed_rows = detection
            if not parsed_rows:
                continue
            column_names: list[str] = []
            width = len(parsed_rows[0])
            for idx in range(width):
                values_at_idx = [parts[idx] for parts in parsed_rows if idx < len(parts)]
                candidate_name = _guess_column_name(idx, values_at_idx, delimiter_name)
                dedup_index = 1
                adjusted_name = candidate_name
                while adjusted_name in column_names:
                    dedup_index += 1
                    adjusted_name = f"{candidate_name}_{dedup_index}"
                column_names.append(adjusted_name)
                if adjusted_name not in column_order:
                    column_order.append(adjusted_name)
            for parts in parsed_rows:
                if len(structured) >= limit:
                    break
                row_payload = {"Section": section}
                for idx, col_name in enumerate(column_names):
                    row_payload[col_name] = parts[idx] if idx < len(parts) else ""
                structured.append(row_payload)
            if len(structured) >= limit:
                break
        if not structured:
            return [], []
        headers = ["Section"] + column_order
        normalized_rows = [
            {header: row.get(header, "") for header in headers}
            for row in structured
        ]
        return headers, normalized_rows

    try:
        from selectolax.parser import HTMLParser
    except ImportError:
        return [], []
    if not html_text:
        return [], []
    try:
        parser = HTMLParser(html_text)
    except Exception:
        return [], []
    root = getattr(parser, "body", None) or getattr(parser, "root", None)
    if root is None:
        return [], []

    heading_tags = {"h1", "h2", "h3", "h4", "h5", "h6", "dt"}
    content_tags = {"p", "li", "dd", "div", "span"}
    skipped_tags = {"script", "style", "noscript", "template", "meta", "link"}

    rows: list[Dict[str, Any]] = []
    sections: Dict[str, List[str]] = defaultdict(list)
    seen: set[str] = set()
    current_section = "Body"

    for node in root.iter():
        tag = getattr(node, "tag", "") or ""
        tag = tag.lower()
        if not tag or tag in skipped_tags:
            continue
        try:
            text = node.text(strip=True)
        except Exception:
            text = ""
        if not text:
            continue
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        if tag in heading_tags:
            current_section = text
            continue
        if tag in content_tags:
            if text in seen:
                continue
            seen.add(text)
            sections[current_section].append(text)
            rows.append({"Section": current_section, "Content": text})
        if len(rows) >= max_rows:
            break

    structured_headers, structured_rows = _build_structured_rows(sections, max_rows)
    if structured_headers and structured_rows:
        return structured_headers, structured_rows

    if not rows:
        return [], []
    return ["Section", "Content"], rows


def generate_generic_html_result(
    *,
    page=None,
    coordinator=None,
    context: Dict[str, Any] | None = None,
    session_id: str | None = None,
    html_text: str | None = None,
    log_type: str = "fallback"
) -> tuple[list[str], list[Dict[str, Any]], str, Dict[str, Any]] | None:
    """Extract tabular data using the dynamic table extractor and emit a CSV result.

    Returns a standard (headers, data, contest, metadata) tuple when successful,
    otherwise ``None``.
    """
    from .Context_Integration.context_coordinator import ContextCoordinator
    from .utils.header_utils import normalize_table_headers

    ctx: Dict[str, Any] = dict(context or {})
    if session_id:
        ctx.setdefault("session_id", session_id)
    ctx.setdefault("url", ctx.get("url") or (getattr(page, "url", None) if page else None))
    ctx.setdefault("fallback_reason", ctx.get("fallback_reason") or "generic_html")

    fallback_strategy = "table_candidates"
    resolved_source_path = None

    if not html_text:
        source_file = ctx.get("source_file")
        source_file = safe_strip(source_file) if isinstance(source_file, str) else None
        if source_file:
            candidate_paths = []
            if os.path.isabs(source_file):
                candidate_paths.append(source_file)
            else:
                candidate_paths.append(os.path.join(str(UPLOADS_DIR), source_file))
                candidate_paths.append(os.path.join(str(INPUT_DIR), source_file))
            for candidate in candidate_paths:
                if not candidate or not os.path.exists(candidate):
                    continue
                ext = os.path.splitext(candidate)[1].lower()
                if ext not in {".html", ".htm", ".xhtml", ".txt"}:
                    continue
                file_html = _read_text_file_with_fallback(candidate)
                if file_html:
                    html_text = file_html
                    resolved_source_path = candidate
                    logger.info({
                        "level": "INFO",
                        "type": log_type,
                        "message": f"[GenericHTML] Loaded fallback HTML from {candidate}",
                        "session_id": session_id
                    })
                    break
                else:
                    logger.warning({
                        "level": "WARNING",
                        "type": log_type,
                        "message": f"[GenericHTML] Failed to read fallback source file: {candidate}",
                        "session_id": session_id
                    })
                    continue
            if resolved_source_path:
                ctx.setdefault("source_file_path", resolved_source_path)

    if not html_text and page is not None:
        page_html = safe_content(page, session_id=session_id)
        if page_html:
            html_text = page_html

    active_coordinator = coordinator or ContextCoordinator()

    try:
        headers, rows = dynamic_table_extractor(
            page,
            ctx,
            active_coordinator,
            table_html=html_text
        )
    except Exception as exc:
        logger.error({
            "level": "ERROR",
            "type": log_type,
            "message": f"[GenericHTML] Extraction failed: {exc}",
            "session_id": session_id
        })
        return None

    if (not headers or not rows) and html_text:
        text_headers, text_rows = _extract_text_blocks(html_text)
        if text_headers and text_rows:
            headers, rows = text_headers, text_rows
            fallback_strategy = "text_blocks"
            logger.info({
                "level": "INFO",
                "type": log_type,
                "message": "[GenericHTML] No tables detected; using text block fallback.",
                "session_id": session_id,
                "row_count": len(text_rows)
            })

    if not headers or not rows:
        logger.warning({
            "level": "WARNING",
            "type": log_type,
            "message": "[GenericHTML] No tabular data detected during fallback extraction.",
            "session_id": session_id
        })
        return None

    fieldnames = list(headers)
    normalized_rows: list[Dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
            normalized_rows.append({key: row.get(key, "") for key in fieldnames})
        elif isinstance(row, (list, tuple)):
            while len(fieldnames) < len(row):
                fieldnames.append(f"Column{len(fieldnames) + 1}")
            normalized_rows.append({fieldnames[i]: row[i] if i < len(row) else "" for i in range(len(fieldnames))})
        else:
            if not fieldnames:
                fieldnames.append("Value")
            normalized_rows.append({fieldnames[0]: row})

    fieldnames, normalized_rows = normalize_table_headers(fieldnames, normalized_rows)

    base_label = ctx.get("file_name") or ctx.get("county") or ctx.get("state") or ctx.get("url") or "generic_html"
    base_label = safe_strip(str(base_label)) if base_label else "generic_html"
    base_label = re.sub(r"[^A-Za-z0-9]+", "_", base_label or "generic_html")
    base_label = base_label.strip("_") or "generic_html"
    contest_label = ctx.get("contest") or ctx.get("state") or "Generic HTML Extraction"
    ctx.setdefault("contest", contest_label)

    builder_context = dict(ctx)
    builder_context.update({
        "handler": "generic_html_fallback",
        "fallback_strategy": fallback_strategy,
        "session_id": session_id,
        "source_url": ctx.get("url"),
    })

    domain = safe_slug(base_label)
    try:
        headers_final, rows_final, entity_info = build_table_noninteractive(
            domain=domain,
            headers=fieldnames,
            data=normalized_rows,
            coordinator=active_coordinator,
            context=builder_context,
            pivot_to_wide=True,
            debug=False,
        )
    except Exception as exc:
        logger.warning({
            "level": "WARNING",
            "type": log_type,
            "message": f"[GenericHTML] Table builder failed; using raw fallback data. ({exc})",
            "session_id": session_id
        })
        headers_final, rows_final, entity_info = fieldnames, normalized_rows, {}

    export_context = dict(builder_context)
    export_context.setdefault("entity_info", entity_info)

    try:
        export_result = finalize_election_output(
            headers=headers_final,
            data=rows_final,
            coordinator=active_coordinator,
            contest=contest_label,
            state=ctx.get("state"),
            county=ctx.get("county"),
            context=export_context,
            enable_user_feedback=False,
            session_id=session_id,
        )
    except Exception as exc:
        logger.error({
            "level": "ERROR",
            "type": log_type,
            "message": f"[GenericHTML] Failed to finalize output: {exc}",
            "session_id": session_id
        })
        return None

    metadata = {
        "output_file": export_result.get("csv_path"),
        "metadata_path": export_result.get("metadata_path"),
        "output_dir": os.path.dirname(export_result.get("csv_path", OUTPUT_DIR)),
        "handler": "generic_html_fallback",
        "fallback": True,
        "fallback_reason": ctx.get("fallback_reason"),
        "fallback_strategy": fallback_strategy,
        "source_url": ctx.get("url"),
        "session_id": session_id,
        "row_count": len(rows_final),
        "column_count": len(headers_final)
    }
    if ctx.get("source_file"):
        metadata["source_file"] = ctx["source_file"]
    if resolved_source_path:
        metadata["source_file_path"] = resolved_source_path
    if entity_info:
        metadata["entity_info"] = entity_info

    logger.info({
        "level": "INFO",
        "type": log_type,
        "message": "[GenericHTML] Fallback extraction succeeded.",
        "session_id": session_id,
        "output_file": export_result.get("csv_path"),
        "row_count": len(rows_final),
        "column_count": len(headers_final)
    })
    
    # Add ML quality metrics
    from .config import log_extraction_quality  # type: ignore[attr-defined]
    quality = log_extraction_quality(
        headers_final, rows_final, metadata, "html_handler", logger, session_id
    )
    metadata["quality_metrics"] = quality
    
    return headers_final, rows_final, contest_label, metadata

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
    coordinator = ContextCoordinator()

    msg = f"Navigating to: {target_url}"
    payload = {
        "level": "INFO",
        "type": "status",
        "message": msg,
        "session_id": session_id
    }
    logger.info(payload)
    try:
        emit_telemetry_event("navigation_start", {"url": target_url, "session_id": session_id})
    except Exception:
        pass

    browser = page = None
    playwright_instance = None
    result = None
    nav_meta: Dict[str, Any] = {}
    agent_used = None
    state, county = infer_state_county_from_url(target_url)
    
    # Extract principal/tier info from kwargs
    principal = kwargs.get("principal")
    principal_source = kwargs.get("principal_source")
    privilege_tier = None
    if principal and principal_source:
        try:
            from .utils.privilege_tiers import get_principal_tier
            privilege_tier = get_principal_tier(principal, principal_source)
        except Exception:
            privilege_tier = None
    
    trust_bypass = bool(kwargs.get("trust_bypass"))
    trust_score = 0
    trust_factors: Dict[str, Any] = {}
    force_snapshot_mode = False

    if trust_bypass:
        trust_score = 100
        trust_factors = {"bypass": True}
        logger.info({
            "level": "INFO",
            "type": "trust_scorer",
            "message": "Trust checks bypassed for dev run; proceeding with direct navigation.",
            "session_id": session_id,
            "url": target_url,
        })
    else:
        # --- URL Trust Scoring (Step 1: Intelligent Verification) ---
        trust_context = {
            "state": state,
            "county": county,
            "source_url": target_url,
        }
        trust_score, trust_factors = compute_trust_score(
            target_url,
            trust_context,
            session_id,
            principal=principal,
            principal_source=principal_source,
        )

        # Check for prior quarantine review decisions (approval/rejection)
        quarantine_status = None
        quarantine_approved = False
        quarantine_rejected = False
        hard_blockers = bool(trust_factors.get("phishing_indicators")) or bool(trust_factors.get("domain_mimicry"))
        try:
            from .health.quarantine_queue import get_quarantine_queue, ReviewStatus
            queue = get_quarantine_queue()
            quarantine_status = queue.get_latest_review_status_for_url(target_url)
            if quarantine_status == ReviewStatus.APPROVED.value:
                quarantine_approved = True
            elif quarantine_status == ReviewStatus.REJECTED.value:
                quarantine_rejected = True
        except Exception:
            quarantine_status = None

        # Check if URL should be rejected outright (tier-aware)
        if should_reject(trust_score, target_url, privilege_tier=privilege_tier):
            if quarantine_rejected:
                logger.error({
                    "level": "ERROR",
                    "type": "trust_scorer",
                    "message": f"URL rejected (prior review rejected; score {trust_score}/100).",
                    "session_id": session_id,
                    "url": target_url,
                    "trust_score": trust_score,
                    "trust_factors": trust_factors,
                    "privilege_tier": privilege_tier.value if privilege_tier else None
                })
                mark_url_processed(target_url, status="rejected", session_id=session_id, trust_score=trust_score)
                return
            if quarantine_approved and not hard_blockers:
                logger.warning({
                    "level": "WARNING",
                    "type": "trust_scorer",
                    "message": f"URL approved by quarantine review despite low trust score ({trust_score}/100).",
                    "session_id": session_id,
                    "url": target_url,
                    "trust_score": trust_score,
                    "trust_factors": trust_factors,
                    "review_status": quarantine_status,
                    "privilege_tier": privilege_tier.value if privilege_tier else None
                })
                force_snapshot_mode = True
            else:
                logger.error({
                    "level": "ERROR",
                    "type": "trust_scorer",
                    "message": f"URL rejected due to low trust score ({trust_score}/100).",
                    "session_id": session_id,
                    "url": target_url,
                    "trust_score": trust_score,
                    "trust_factors": trust_factors,
                    "privilege_tier": privilege_tier.value if privilege_tier else None
                })
                mark_url_processed(target_url, status="rejected", session_id=session_id, trust_score=trust_score)
                return

        # Check if URL should be quarantined for manual review (tier-aware)
        if should_quarantine(trust_score, target_url, privilege_tier=privilege_tier):
            if quarantine_rejected:
                logger.error({
                    "level": "ERROR",
                    "type": "trust_scorer",
                    "message": f"URL rejected (prior review rejected; score {trust_score}/100).",
                    "session_id": session_id,
                    "url": target_url,
                    "trust_score": trust_score,
                    "trust_factors": trust_factors,
                    "review_status": quarantine_status,
                    "privilege_tier": privilege_tier.value if privilege_tier else None
                })
                mark_url_processed(target_url, status="rejected", session_id=session_id, trust_score=trust_score)
                return
            if quarantine_approved and not hard_blockers:
                logger.info({
                    "level": "INFO",
                    "type": "trust_scorer",
                    "message": f"URL approved by quarantine review (score {trust_score}/100); proceeding in snapshot mode.",
                    "session_id": session_id,
                    "url": target_url,
                    "trust_score": trust_score,
                    "trust_factors": trust_factors,
                    "review_status": quarantine_status,
                    "privilege_tier": privilege_tier.value if privilege_tier else None
                })
                force_snapshot_mode = True
            else:
                logger.warning({
                    "level": "WARNING",
                    "type": "trust_scorer",
                    "message": f"URL quarantined for manual review (trust score: {trust_score}/100).",
                    "session_id": session_id,
                    "url": target_url,
                    "trust_score": trust_score,
                    "trust_factors": trust_factors,
                    "privilege_tier": privilege_tier.value if privilege_tier else None
                })

                # --- Enqueue for transparent review with audit trail ---
                try:
                    from .health.quarantine_queue import (
                        get_quarantine_queue,
                        QuarantineReason,
                        DataCollectionNotice,
                    )

                    queue = get_quarantine_queue()
                    if not queue.has_pending_url(target_url):
                        data_notices = [
                            DataCollectionNotice(
                                data_type="trust_score",
                                description=f"Computed trust score: {trust_score}/100. Indicates URL reliability confidence.",
                                usage="Security filtering to prevent extraction from untrusted/malicious sources",
                                retention_days=30,
                            ),
                            DataCollectionNotice(
                                data_type="trust_factors",
                                description=f"Breakdown of trust assessment: {orjson.dumps(trust_factors).decode('utf-8') if trust_factors else 'none'}",
                                usage="Forensic analysis; helps identify why URL was flagged",
                                retention_days=30,
                            ),
                        ]

                        queue.enqueue(
                            url=target_url,
                            reason=QuarantineReason.LOW_TRUST_SCORE,
                            session_id=session_id,
                            principal=principal,
                            trust_score=trust_score,
                            trust_factors=trust_factors,
                            data_notices=data_notices,
                        )

                        logger.info({
                            "level": "INFO",
                            "type": "quarantine",
                            "message": "[Quarantine] URL enqueued for transparent review",
                            "session_id": session_id,
                            "url": target_url,
                            "reason": "LOW_TRUST_SCORE",
                        })
                    else:
                        logger.info({
                            "level": "INFO",
                            "type": "quarantine",
                            "message": "[Quarantine] URL already pending review",
                            "session_id": session_id,
                            "url": target_url,
                        })
                except Exception as e:
                    logger.error({
                        "level": "ERROR",
                        "type": "quarantine",
                        "message": f"[Quarantine] Failed to enqueue: {e}",
                        "session_id": session_id,
                        "url": target_url,
                    })

                mark_url_processed(target_url, status="quarantined", session_id=session_id, trust_score=trust_score)
                return

        # Log trust decision for allowed URLs
        use_snapshot = force_snapshot_mode or should_use_snapshot_mode(trust_score, target_url)
        if use_snapshot:
            logger.info({
                "level": "INFO",
                "type": "trust_scorer",
                "message": f"Using DOM snapshot mode for medium-trust URL (score: {trust_score}/100).",
                "session_id": session_id,
                "url": target_url,
                "trust_score": trust_score,
                "privilege_tier": privilege_tier.value if privilege_tier else None
            })
            # Execute DOM snapshot mode pipeline (Step 2)
            # This captures static HTML without JS execution for safety
            snapshot_context = {
                "state": state,
                "county": county,
                "url": target_url,
                "trust_score": trust_score,
                "trust_factors": trust_factors,
                "principal": principal,
                "principal_source": principal_source,
                "privilege_tier": privilege_tier.value if privilege_tier else None,
            }

            # Use Playwright to navigate but capture snapshot instead of full interaction
            snapshot_browser = snapshot_page = None
            try:
                with sync_playwright() as p:
                    snapshot_browser, _, snapshot_page, _, _ = sync_browser_pipeline(
                        p,
                        target_url,
                        cache_exit_callback=mark_url_processed,
                        session_id=session_id,
                        nav_timeout_ms=NAV_TIMEOUT_PLAYWRIGHT_MS,
                        cancel_flag=cancel_flag,
                    )

                    if cancel_flag is not None and safe_is_set(cancel_flag):
                        logger.info({
                            "level": "INFO",
                            "type": "cancel",
                            "message": f"Processing cancelled for {target_url}",
                            "session_id": session_id,
                        })
                        _close_browser_quietly(snapshot_browser, session_id)
                        return

                    if not snapshot_page:
                        logger.error({
                            "level": "ERROR",
                            "type": "dom_snapshot",
                            "message": f"[DOMSnapshot] Could not open page for {target_url}",
                            "session_id": session_id,
                        })
                        _close_browser_quietly(snapshot_browser, session_id)
                        mark_url_processed(target_url, status="error", session_id=session_id)
                        return

                    # Execute snapshot mode pipeline
                    headers, data, contest, metadata = snapshot_mode_pipeline(
                        snapshot_page,
                        snapshot_context,
                        session_id
                    )

                    # Finalize output (same as normal pipeline)
                    from .Context_Integration.context_coordinator import ContextCoordinator
                    from .utils.output_utils import finalize_election_output

                    coordinator = ContextCoordinator()
                    export_context = dict(snapshot_context)
                    export_context.update(metadata)

                    try:
                        export_result = finalize_election_output(
                            headers=headers,
                            data=data,
                            coordinator=coordinator,
                            contest=contest,
                            state=state,
                            county=county,
                            context=export_context,
                            enable_user_feedback=False,
                            session_id=session_id,
                        )

                        metadata["output_file"] = export_result.get("csv_path")
                        metadata["metadata_path"] = export_result.get("metadata_path")

                        logger.info({
                            "level": "INFO",
                            "type": "dom_snapshot",
                            "message": f"[DOMSnapshot] Output written to: {export_result.get('csv_path')}",
                            "session_id": session_id,
                            "output_file": export_result.get("csv_path")
                        })

                        ai_analyze_results(headers, data, contest, metadata, target_url=target_url, session_id=session_id, trust_factors=trust_factors, privilege_tier=privilege_tier)
                        stream_results(headers, data, contest, metadata, target_url=target_url, session_id=session_id)
                        processed_meta = {}
                        if isinstance(metadata, dict):
                            for key in (
                                "output_file",
                                "metadata_path",
                                "output_dir",
                                "contest",
                                "state",
                                "county",
                                "handler",
                                "source_url",
                            ):
                                if key in metadata and metadata.get(key):
                                    processed_meta[key] = metadata.get(key)
                        mark_url_processed(
                            target_url,
                            status="success",
                            session_id=session_id,
                            snapshot_mode=True,
                            **processed_meta,
                        )
                    except Exception as exc:
                        logger.error({
                            "level": "ERROR",
                            "type": "dom_snapshot",
                            "message": f"[DOMSnapshot] Output finalization failed: {exc}",
                            "session_id": session_id
                        })
                        mark_url_processed(target_url, status="error", session_id=session_id)
            except Exception as exc:
                logger.error({
                    "level": "ERROR",
                    "type": "dom_snapshot",
                    "message": f"[DOMSnapshot] Snapshot mode pipeline failed: {exc}",
                    "session_id": session_id
                })
                mark_url_processed(target_url, status="error", session_id=session_id)
            finally:
                _close_browser_quietly(snapshot_browser, session_id)

            # Early return after snapshot mode completes
            return
        else:
            logger.info({
                "level": "INFO",
                "type": "trust_scorer",
                "message": f"High-trust URL (score: {trust_score}/100) - proceeding with direct navigation.",
                "session_id": session_id,
                "url": target_url,
                "trust_score": trust_score
            })
    
    strategies = [
        {"agent": "playwright", "timeout_ms": NAV_TIMEOUT_PLAYWRIGHT_MS},
    ]
    if ENABLE_SELENIUM_FALLBACK:
        strategies.append({"agent": "selenium", "timeout_ms": NAV_TIMEOUT_SELENIUM_MS})
    max_attempts = min(len(strategies), NAV_MAX_ATTEMPTS)

    try:
        for attempt_idx, strat in enumerate(strategies[:max_attempts], start=1):
            agent = strat.get("agent")
            try:
                increment_counter("nav_agent_attempt_total", 1)
            except Exception:
                pass

            if agent == "playwright":
                try:
                    playwright_instance = sync_playwright().start()
                    browser, _, page, _, nav_meta = sync_browser_pipeline(
                        playwright_instance,
                        target_url,
                        cache_exit_callback=mark_url_processed,
                        session_id=session_id,
                        nav_timeout_ms=strat.get("timeout_ms") or NAV_TIMEOUT_PLAYWRIGHT_MS,
                        cancel_flag=cancel_flag,
                    )
                    if cancel_flag is not None and safe_is_set(cancel_flag):
                        payload = {
                            "level": "INFO",
                            "type": "cancel",
                            "message": f"Processing cancelled for {target_url}",
                            "session_id": session_id,
                        }
                        logger.info(payload)
                        _close_browser_quietly(browser, session_id)
                        return
                    if not page:
                        logger.error({
                            "level": "ERROR",
                            "type": "browser",
                            "message": f"Could not open page for {target_url}",
                            "session_id": session_id,
                        })
                        _close_browser_quietly(browser, session_id)
                        if playwright_instance is not None:
                            try:
                                playwright_instance.stop()
                            except Exception:
                                pass
                            playwright_instance = None
                        continue
                    if nav_meta.get("cloudflare_detected") and ENABLE_SELENIUM_FALLBACK:
                        logger.warning({
                            "level": "WARNING",
                            "type": "browser",
                            "message": "Cloudflare challenge detected — will attempt Selenium fallback if enabled.",
                            "session_id": session_id,
                        })
                        try:
                            increment_counter("nav_agent_playwright_cloudflare", 1)
                        except Exception:
                            pass
                        _close_browser_quietly(browser, session_id)
                        browser = page = None
                        if playwright_instance is not None:
                            try:
                                playwright_instance.stop()
                            except Exception:
                                pass
                            playwright_instance = None
                        continue
                    agent_used = "playwright"
                    try:
                        increment_counter("nav_agent_playwright_success", 1)
                    except Exception:
                        pass
                    break
                except Exception as exc:
                    logger.error({
                        "level": "ERROR",
                        "type": "browser",
                        "message": f"Playwright navigation failed: {exc}",
                        "session_id": session_id,
                    })
                    try:
                        increment_counter("nav_agent_playwright_fail", 1)
                    except Exception:
                        pass
                    _close_browser_quietly(browser, session_id)
                    browser = page = None
                    if playwright_instance is not None:
                        try:
                            playwright_instance.stop()
                        except Exception:
                            pass
                        playwright_instance = None
                    continue

            if agent == "selenium":
                if not (ENABLE_SELENIUM_FALLBACK and SELENIUMBASE_AVAILABLE):
                    logger.warning({
                        "level": "WARNING",
                        "type": "browser",
                        "message": "Selenium fallback requested but dependency is unavailable.",
                        "session_id": session_id,
                    })
                    continue
                driver = None
                try:
                    _, _, driver = launch_browser()
                    try:
                        driver.set_page_load_timeout((strat.get("timeout_ms") or NAV_TIMEOUT_SELENIUM_MS) / 1000)
                    except Exception:
                        pass
                    driver.get(target_url)
                    html_text = getattr(driver, "page_source", "") or ""
                    blocked = detect_cloudflare_challenge(driver)
                    nav_meta = {"agent": "selenium", "cloudflare_detected": blocked}
                    if not html_text:
                        raise RuntimeError("Selenium fallback returned empty page_source")
                    fallback_ctx = {
                        "state": state,
                        "county": county,
                        "url": target_url,
                        "fallback_reason": "selenium_fallback",
                    }
                    result = generate_generic_html_result(
                        page=None,
                        coordinator=coordinator,
                        context=fallback_ctx,
                        session_id=session_id,
                        html_text=html_text,
                        log_type="selenium_fallback",
                    )
                    if result:
                        agent_used = "selenium"
                        try:
                            increment_counter("nav_agent_selenium_success", 1)
                        except Exception:
                            pass
                        break
                    try:
                        increment_counter("nav_agent_selenium_fail", 1)
                    except Exception:
                        pass
                except Exception as exc:
                    logger.error({
                        "level": "ERROR",
                        "type": "browser",
                        "message": f"Selenium fallback failed: {exc}",
                        "session_id": session_id,
                    })
                    try:
                        increment_counter("nav_agent_selenium_fail", 1)
                    except Exception:
                        pass
                finally:
                    if driver:
                        close_driver(driver)

        # If Selenium fallback produced a direct result, finish early
        if agent_used == "selenium" and result is not None:
            headers, data, contest, metadata = result
            if all([headers, data, contest, metadata]):
                ai_analyze_results(headers, data, contest, metadata, target_url=target_url, session_id=session_id, trust_factors=trust_factors, privilege_tier=privilege_tier)
                stream_results(headers, data, contest, metadata, target_url=target_url, session_id=session_id)
                mark_url_processed(target_url, status="success", session_id=session_id)
            else:
                mark_url_processed(target_url, status="partial", session_id=session_id)
            return

        if page is None:
            logger.error({
                "level": "ERROR",
                "type": "browser",
                "message": f"Navigation failed for {target_url} (no agent succeeded).",
                "session_id": session_id,
            })
            mark_url_processed(target_url, status="error", session_id=session_id)
            return
        nav_context = {
            "state": state,
            "county": county,
            "url": target_url,
        }
        nav_context_before = dict(nav_context)
        nav_output = None
        if NAVIGATION_RUNNER:
            try:
                nav_output = NAVIGATION_RUNNER.run(
                    page,
                    context=nav_context,
                    coordinator=coordinator,
                    session_id=session_id,
                )
            except Exception as exc:
                nav_output = None
                logger.warning({
                    "level": "WARNING",
                    "type": "navigation",
                    "message": f"Navigation runner failed: {exc}",
                    "session_id": session_id,
                })
            else:
                if nav_output.executed:
                    logger.info({
                        "level": "INFO",
                        "type": "navigation",
                        "message": f"Navigation script executed: {nav_output.script_id}",
                        "session_id": session_id,
                    })
                if nav_output.context_updates:
                    nav_context.update(nav_output.context_updates)
                coordinator.record_navigation_feedback(
                    script_id=nav_output.script_id,
                    success=nav_output.executed,
                    context_before=nav_context_before,
                    context_after=dict(nav_context),
                    telemetry=nav_output.telemetry,
                    metadata=nav_output.metadata,
                )
            try:
                emit_telemetry_event("navigation_complete", {
                    "url": target_url,
                    "session_id": session_id,
                    "nav_executed": bool(getattr(nav_output, 'executed', False)) if nav_output is not None else False,
                    "nav_script_id": getattr(nav_output, 'script_id', None) if nav_output is not None else None,
                    "nav_telemetry": getattr(nav_output, 'telemetry', None) if nav_output is not None else None,
                })
            except Exception:
                pass
        scroll_metrics: Dict[str, Any] = {}
        try:
            autoscroll_until_stable(
                page,
                wait_for_selector='table, a[href$=".csv"], a[href$=".json"], [role="table"]',
                max_total_time=20000,
                delay_ms=250,
                session_id=session_id,
                metrics=scroll_metrics,
            )
        except Exception:
            pass
        if scroll_metrics:
            logger.info(
                {
                    "level": "INFO",
                    "type": "telemetry",
                    "message": "[Telemetry] Autoscroll metrics collected.",
                    "session_id": session_id,
                    **{k: v for k, v in scroll_metrics.items() if k in SCROLL_METRIC_KEYS},
                }
            )
            try:
                emit_telemetry_event("page_scrolled", {"url": target_url, "session_id": session_id, "scroll_metrics": scroll_metrics})
            except Exception:
                pass

        dom_table_rows = _count_dom_table_rows(page)
        download_parse_tuple = None
        handled = False
        if dom_table_rows <= 0:
            download_parse_tuple, handled = prompt_and_handle_download(
                page,
                target_url,
                rejected_downloads,
                session_id=session_id,
                cancel_flag=cancel_flag,
                **kwargs,
            )
        else:
            logger.info(
                {
                    "level": "INFO",
                    "type": "download",
                    "message": "[format_router] DOM tables detected; deferring downloads to keep HTML parsing priority.",
                    "session_id": session_id,
                    "dom_table_rows": dom_table_rows,
                }
            )
        if handled:
            if isinstance(download_parse_tuple, tuple) and len(download_parse_tuple) == 4:
                result = download_parse_tuple
                msg = f"Download handled for {target_url}; continuing pipeline."
                payload = {
                    "level": "INFO",
                    "type": "download",
                    "message": msg,
                    "session_id": session_id
                }
                logger.info(payload)
            else:
                logger.warning({
                    "level": "WARNING",
                    "type": "download",
                    "message": "Download handler returned invalid result tuple; falling back to HTML pipeline.",
                    "session_id": session_id
                })

        handler = None
        context = {
            **nav_context,
            "url": target_url,
            "session_id": session_id,
            "output_bypass": output_bypass,
            "principal": kwargs.get("principal"),
            "principal_source": kwargs.get("principal_source"),
        }
        if result is None:
            active_state = context.get("state") or state
            if active_state:
                preload_handler_map(restrict_to_states=[active_state])
            else:
                preload_handler_map()
            handler_result = get_handler(
                context,
                url=target_url,
                debug=False,
                fuzzy_cutoff=None,
                session_id=session_id
            )
            try:
                handler_name = None
                summary = None
                if isinstance(handler_result, dict):
                    handler_name = handler_result.get('handler').__class__.__name__ if handler_result.get('handler') is not None else None
                    summary = handler_result.get('summary')
                emit_telemetry_event("handler_selected", {"url": target_url, "session_id": session_id, "handler": handler_name, "summary": summary})
            except Exception:
                pass
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

            if handler and hasattr(handler, 'parse'):
                result = safe_parse(
                    handler,
                    page,
                    coordinator,
                    context,
                    session_id=session_id,
                    logger=logger,
                    cancel_flag=cancel_flag,
                    **kwargs,
                )
            else:
                msg = f"[Router] No suitable handler found for {target_url}, using generic HTML fallback."
                payload = {
                    "level": "WARNING",
                    "type": "router",
                    "message": msg,
                    "session_id": session_id
                }
                logger.warning(payload)
                html_content = None
                try:
                    html_content = page.inner_html("body") if page else None
                except Exception:
                    html_content = None
                result = generate_generic_html_result(
                    page=page,
                    coordinator=coordinator,
                    context=context,
                    session_id=session_id,
                    html_text=html_content,
                    log_type="router"
                )

            if not (isinstance(result, tuple) and len(result) == 4):
                msg = f"Handler did not return a valid result tuple. (Session: {session_id})"
                payload = {
                    "level": "ERROR",
                    "type": "handler",
                    "message": msg,
                    "session_id": session_id
                }
                logger.error(payload)
                mark_url_processed(target_url, status="fail", session_id=session_id)
                _close_browser_quietly(browser, session_id)
                return

            # Emit parse result telemetry (best-effort)
            try:
                if isinstance(result, tuple) and len(result) == 4:
                    _h, _d, _c, _m = result
                    row_count = len(_d) if isinstance(_d, (list, tuple)) else 0
                    col_count = len(_h) if isinstance(_h, (list, tuple)) else (len(_d[0]) if row_count and isinstance(_d[0], (list, tuple)) else 0)
                    error_metadata = None
                    if isinstance(_m, dict) and _m.get("error"):
                        error_metadata = _sanitize_error_metadata(_m)
                    emit_telemetry_event("parse_result", {
                        "url": target_url,
                        "session_id": session_id,
                        "handler": getattr(handler, '__class__', None).__name__ if handler else None,
                        "row_count": row_count,
                        "column_count": col_count,
                        "metadata_keys": list(_m.keys()) if isinstance(_m, dict) else None,
                        "error_metadata": error_metadata,
                    })
                    if error_metadata:
                        _log_session_exception_metadata(session_id, {
                            "level": "ERROR",
                            "type": "handler_exception",
                            "message": "Handler exception metadata captured.",
                            "session_id": session_id,
                            "url": target_url,
                            "handler": getattr(handler, '__class__', None).__name__ if handler else None,
                            "error_metadata": error_metadata,
                            "timestamp": datetime.utcnow().isoformat() + "Z",
                        })
            except Exception:
                pass

            headers, data, contest, metadata = result

            if isinstance(metadata, dict) and metadata.get("error"):
                msg = f"Handler reported error: {metadata.get('error')} (Session: {session_id})"
                logger.error({"level": "ERROR", "type": "handler", "message": msg, "session_id": session_id})
                mark_url_processed(target_url, status="fail", session_id=session_id)
                _close_browser_quietly(browser, session_id)
                return

            if isinstance(metadata, dict) and metadata.get("cancelled"):
                cancel_msg = metadata.get("cancel_reason") or "Parsing cancelled by user request."
                logger.info({
                    "level": "CANCELLED",
                    "type": "cancel",
                    "message": cancel_msg,
                    "session_id": session_id,
                })
                mark_url_processed(target_url, status="cancelled", session_id=session_id)
                _close_browser_quietly(browser, session_id)
                return

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
                        session_id=session_id,
                        handler=handler,
                        initial_result=(headers, data, contest, metadata)
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
                _close_browser_quietly(browser, session_id)
                return

            if all([headers, data, contest, metadata]):
                ai_analyze_results(headers, data, contest, metadata, target_url=target_url, session_id=session_id, trust_factors=trust_factors, privilege_tier=privilege_tier)
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
                processed_meta = {}
                if isinstance(metadata, dict):
                    for key in (
                        "output_file",
                        "metadata_path",
                        "output_dir",
                        "contest",
                        "state",
                        "county",
                        "handler",
                        "source_url",
                    ):
                        if key in metadata and metadata.get(key):
                            processed_meta[key] = metadata.get(key)
                mark_url_processed(target_url, status="success", session_id=session_id, **processed_meta)
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
            emit_telemetry_event("processing_exception", {"url": target_url, "session_id": session_id, "error": str(e)})
        except Exception:
            pass
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
        _close_browser_quietly(browser, session_id)
        if playwright_instance is not None:
            try:
                playwright_instance.stop()
            except Exception:
                pass
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

        skip_url_prompt = bool(kwargs.pop("skip_url_prompt", False))
        url_source_label = kwargs.pop("url_source_label", None)

        # --- 1. Manual Upload Override Path ---
        if manual_source == 'uploads':
            override_result = process_format_override(
                session_id=session_id,
                source_dir='uploads',
                output_bypass=output_bypass,
                force_parse_input_file=kwargs.get("force_parse_input_file"),
                force_parse_format=kwargs.get("force_parse_format"),
                cancel_flag=cancel_flag,
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
                context = {
                    "state": kwargs.get("state"),
                    "county": kwargs.get("county"),
                    "source_file": kwargs.get("force_parse_input_file"),
                    "fallback_reason": "manual_override_failed",
                    "principal": kwargs.get("principal"),
                    "principal_source": kwargs.get("principal_source"),
                }
                fallback = generate_generic_html_result(
                    context=context,
                    session_id=session_id,
                    html_text=kwargs.get("force_parse_raw_html"),
                    log_type="manual_override"
                )
                if fallback:
                    headers, data, contest, metadata = fallback
                    source_key = context.get("source_file") or "manual_override"
                    stream_results(headers, data, contest, metadata, target_url=source_key, session_id=session_id)
                    mark_url_processed(source_key, status="success", session_id=session_id)
                    logger.info({
                        "level": "INFO",
                        "type": "manual_override",
                        "message": "[ManualOverride] Fallback generic HTML extraction succeeded.",
                        "session_id": session_id
                    })
                    return
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
            urls = load_urls(allowlist_bypass=bool(kwargs.get("allowlist_bypass")))
            if not url_source_label:
                url_source_label = "urls.txt"
        else:
            if not isinstance(urls, list):
                urls = list(urls)
            if not url_source_label:
                url_source_label = "direct override"

        logger.info({
            "level": "INFO",
            "type": "input",
            "message": f"Loaded {len(urls)} raw URLs from {url_source_label}",
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
        if skip_url_prompt:
            selected_urls = urls
            logger.info({
                "level": "INFO",
                "type": "input",
                "message": f"Using pre-selected URL list ({len(selected_urls)} item(s)).",
                "session_id": session_id
            })
        else:
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
        # Raise instead of exiting so callers (e.g., web pipeline) can handle failures
        raise

if __name__ == "__main__":
    logger.set_mode("cli")
    main()
