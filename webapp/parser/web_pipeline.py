from .html_election_parser import main
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import os
import orjson
import traceback
from .utils.shared_logic import (
    safe_set,
    safe_clear,
    safe_is_set
)
from .utils.logger_singleton import logger, prompt
from flask import session, request
from typing import Any
from .config import PIPELINE_MAX_WORKERS, PIPELINE_MAX_ERRORS, PIPELINE_HEARTBEAT_INTERVAL

class CancellationManager(threading.Thread):
    """
    Manages cancellation flags per session/user.
    """
    def __init__(self) -> None:
        super().__init__()
        self._flags = {}
        self._lock = threading.Lock()

    def get_flag(self, session_id) -> threading.Event:
        with self._lock:
            if session_id not in self._flags:
                self._flags[session_id] = threading.Event()
                logger.info({
                    "level": "DEBUG",
                    "type": "cancellation",
                    "message": f"New cancellation flag created for session_id={session_id}",
                    "session_id": session_id
                })
            else:
                logger.info({
                    "level": "DEBUG",
                    "type": "cancellation",
                    "message": f"get_flag called for session_id={session_id}",
                    "session_id": session_id
                })
            return self._flags[session_id]

    def cancel(self, session_id) -> None:
        with self._lock:
            if session_id in self._flags:
                safe_set(self._flags[session_id])
                logger.info({
                    "level": "CANCELLED",
                    "type": "cancel",
                    "message": f"Cancellation requested for session_id={session_id}",
                    "session_id": session_id
                })
            else:
                logger.warning({
                    "level": "WARNING",
                    "type": "cancel",
                    "message": f"Cancellation requested for unknown session_id={session_id}",
                    "session_id": session_id
                })

    def reset(self, session_id) -> None:
        with self._lock:
            if session_id in self._flags:
                safe_clear(self._flags[session_id])
                logger.info({
                    "level": "DEBUG",
                    "type": "cancellation",
                    "message": f"Cancellation reset for session_id={session_id}",
                    "session_id": session_id
                })
            else:
                logger.warning({
                    "level": "WARNING",
                    "type": "cancellation",
                    "message": f"Reset requested for unknown session_id={session_id}",
                    "session_id": session_id
                })

    def remove(self, session_id) -> None:
        with self._lock:
            if session_id in self._flags:
                del self._flags[session_id]
                logger.info({
                    "level": "DEBUG",
                    "type": "cancellation",
                    "message": f"Cancellation flag removed for session_id={session_id}",
                    "session_id": session_id
                })
            else:
                logger.warning({
                    "level": "WARNING",
                    "type": "cancellation",
                    "message": f"Remove requested for unknown session_id={session_id}",
                    "session_id": session_id
                })

# Instantiate globally
cancellation_manager = CancellationManager()

def heartbeat(session_id, cancel_flag, interval=10, emit_func=None):
    while True:
        time.sleep(interval)
        # Only emit heartbeat to frontend, don't log to terminal
        if emit_func:
            emit_func({
                "type": "heartbeat",
                "session_id": session_id,
                "status": "alive",
                "timestamp": time.time()
            })
        if safe_is_set(cancel_flag):
            break

def save_pipeline_report(session_id, results, errors):
    report_dir = os.path.join("output", "reports")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, f"report_{session_id}.json")
    with open(report_path, "wb") as f:
        f.write(orjson.dumps({
            "results": results,
            "errors": errors
        }, option=orjson.OPT_INDENT_2))
    return report_path

def process_urls_for_web(
    prompt_queue,
    session_id,
    cancel_flag,
    max_workers=PIPELINE_MAX_WORKERS,
    emit_func=None,
    **kwargs
) -> None:
    """
    Advanced pipeline: per-URL timing, global timing, live progress, error threshold, tracebacks,
    output file saving, env-configurable workers, heartbeat, and prompt queue handling.
    """
    MAX_ERRORS = PIPELINE_MAX_ERRORS
    HEARTBEAT_INTERVAL = PIPELINE_HEARTBEAT_INTERVAL

    # Safeguard: Only reset if the flag is not already set (i.e., not cancelled)
    if not safe_is_set(cancel_flag):
        cancellation_manager.reset(session_id)
    else:
        logger.warning({
            "level": "WARNING",
            "type": "cancellation",
            "message": f"Session {session_id} is already cancelled. Not resetting.",
            "session_id": session_id
        })

    # Always set logger and prompt to webapp mode
    logger.set_mode("webapp")
    logger.set_format("json")
    logger.set_socketio_emit_func(emit_func)
    prompt.set_mode("webapp")
    prompt.set_socketio_emit_func(emit_func)

    # Start heartbeat thread with emit_func
    threading.Thread(
        target=heartbeat,
        args=(session_id, cancel_flag, HEARTBEAT_INTERVAL, emit_func),
        daemon=True
    ).start()
    
    pipeline_start = time.time()
    try:
        logger.info({
            "level": "INFO",
            "type": "status",
            "message": f"Session started for {session_id}",
            "session_id": session_id
        })

        # --- Main pipeline logic (initial run) ---
        urls = kwargs.get("urls")
        if urls is None:
            main(
                session_id=session_id,
                cancel_flag=cancel_flag,
                **kwargs
            )
        else:
            if isinstance(urls, str):
                urls = [urls]
            elif not isinstance(urls, list) or not all(isinstance(u, str) for u in urls):
                logger.error({
                    "level": "ERROR",
                    "type": "input",
                    "message": "[ERROR] Invalid URLs input. Must be list of strings.",
                    "session_id": session_id
                })
                cancellation_manager.remove(session_id)
                return

            total = len(urls)
            if total == 0:
                logger.warning({
                    "level": "WARNING",
                    "type": "input",
                    "message": "No URLs provided for processing.",
                    "session_id": session_id
                })
                cancellation_manager.remove(session_id)
                return

            logger.info({
                "level": "INFO",
                "type": "status",
                "message": f"Starting pipeline for {total} URL(s)...",
                "total": total,
                "session_id": session_id
            })
            completed = 0
            errors = []
            results = []
            url_timings = []

            if max_workers and max_workers > 1 and total > 1:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_url = {
                        executor.submit(
                            main,
                            url=url,
                            session_id=session_id,
                            cancel_flag=cancel_flag,
                            **kwargs
                        ): (idx, url) for idx, url in enumerate(urls)
                    }
                    for i, future in enumerate(as_completed(future_to_url)):
                        idx, url = future_to_url[future]
                        url_start = time.time()
                        try:
                            result = future.result()
                            url_duration = time.time() - url_start
                            results.append((url, result))
                            url_timings.append({"url": url, "index": idx, "duration": url_duration})
                            logger.info({
                                "level": "SUCCESS",
                                "type": "status",
                                "message": f"{url} processed: {result}" if result is not None else f"{url} processed.",
                                "url": url,
                                "result": result,
                                "index": idx,
                                "duration": url_duration,
                                "session_id": session_id
                            })
                        except Exception as exc:
                            url_duration = time.time() - url_start
                            errors.append((url, str(exc)))
                            logger.error({
                                "level": "ERROR",
                                "type": "exception",
                                "message": f"Exception for {url}: {exc}",
                                "url": url,
                                "error": str(exc),
                                "traceback": traceback.format_exc(),
                                "index": idx,
                                "duration": url_duration,
                                "session_id": session_id
                            })
                        completed += 1
                        logger.info({
                            "level": "PROGRESS",
                            "type": "status",
                            "message": f"Progress update: {completed}/{total}",
                            "progress": completed / total,
                            "current_url": url,
                            "index": idx,
                            "session_id": session_id
                        })
                        if safe_is_set(cancel_flag):
                            logger.info({
                                "level": "CANCELLED",
                                "type": "cancel",
                                "message": "[CANCELLED] Processing stopped by user.",
                                "session_id": session_id
                            })
                            break
                        if len(errors) >= MAX_ERRORS:
                            logger.error({
                                "level": "ERROR",
                                "type": "exception",
                                "message": f"Too many errors ({MAX_ERRORS}), aborting pipeline.",
                                "errors": errors,
                                "session_id": session_id
                            })
                            break
                if errors:
                    logger.warning({
                        "level": "SUMMARY",
                        "type": "summary",
                        "message": f"{len(errors)} URLs failed.",
                        "errors": errors,
                        "session_id": session_id
                    })
                if results:
                    logger.info({
                        "level": "SUMMARY",
                        "type": "summary",
                        "message": f"{len(results)} URLs processed successfully.",
                        "results": results,
                        "session_id": session_id
                    })
            else:
                for i, url in enumerate(urls):
                    if safe_is_set(cancel_flag):
                        logger.info({
                            "level": "CANCELLED",
                            "type": "cancel",
                            "message": "[CANCELLED] Processing stopped by user.",
                            "session_id": session_id
                        })
                        break
                    url_start = time.time()
                    try:
                        main(
                            url=url,
                            session_id=session_id,
                            cancel_flag=cancel_flag,
                            emit_func=emit_func,
                            **kwargs
                        )
                        url_duration = time.time() - url_start
                        results.append((url, "success"))
                        url_timings.append({"url": url, "index": i, "duration": url_duration})
                        logger.info({
                            "level": "SUCCESS",
                            "type": "status",
                            "message": f"{url} processed.",
                            "url": url,
                            "index": i,
                            "duration": url_duration,
                            "session_id": session_id
                        })
                    except Exception as exc:
                        url_duration = time.time() - url_start
                        errors.append((url, str(exc)))
                        logger.error({
                            "level": "ERROR",
                            "type": "exception",
                            "message": f"Exception for {url}: {exc}",
                            "url": url,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                            "index": i,
                            "duration": url_duration,
                            "session_id": session_id
                        })
                    completed += 1
                    logger.info({
                        "level": "PROGRESS",
                        "type": "status",
                        "message": f"Progress update: {completed}/{total}",
                        "progress": completed / total,
                        "current_url": url,
                        "index": i,
                        "session_id": session_id
                    })
                    if len(errors) >= MAX_ERRORS:
                        logger.error({
                            "level": "ERROR",
                            "type": "exception",
                            "message": f"Too many errors ({MAX_ERRORS}), aborting pipeline.",
                            "errors": errors,
                            "session_id": session_id
                        })
                        break
                if errors:
                    logger.warning({
                        "level": "SUMMARY",
                        "type": "summary",
                        "message": f"{len(errors)} URLs failed.",
                        "errors": errors,
                        "session_id": session_id
                    })
        # Global Pipeline Timing and Summary
        pipeline_duration = time.time() - pipeline_start
        logger.info({
            "level": "SUMMARY",
            "type": "summary",
            "message": "Pipeline finished.",
            "total_urls": total if urls else 0,
            "errors": len(errors),
            "duration": pipeline_duration,
            "url_timings": url_timings,
            "session_id": session_id
        })
        # Output File Saving
        report_path = save_pipeline_report(session_id, results, errors)
        logger.info({
            "level": "INFO",
            "type": "output",
            "message": f"Pipeline report saved.",
            "report_path": report_path,
            "session_id": session_id
        })

        # --- Prompt queue handling loop ---
        while not safe_is_set(cancel_flag):
            try:
                prompt_data = prompt_queue.get(timeout=1)
                if safe_is_set(cancel_flag):
                    break
                logger.info({
                    "level": "PROMPT",
                    "type": "prompt",
                    "message": f"Processing prompt: {prompt_data}",
                    "session_id": session_id
                })
                # Handle prompt (custom logic, e.g. call main, emit output, etc.)
                main(
                    prompt=prompt_data,
                    session_id=session_id,
                    cancel_flag=cancel_flag,
                    **kwargs
                )
            except Exception:
                continue  # Timeout or empty queue, just check cancel_flag again

    except Exception as e:
        logger.error({
            "level": "ERROR",
            "type": "exception",
            "message": f"Exception in pipeline: {e}",
            "traceback": traceback.format_exc(),
            "session_id": session_id
        })
    finally:
        if safe_is_set(cancel_flag):
            logger.info({
                "level": "INFO",
                "type": "cancel",
                "message": "Processing cancelled by user.",
                "session_id": session_id
            })
        else:
            logger.info({
                "level": "INFO",
                "type": "status",
                "message": "All URLs processed.",
                "session_id": session_id
            })
        cancellation_manager.remove(session_id)

def cancel_processing(session_id) -> None:
    cancellation_manager.cancel(session_id)
    logger.info({
        "level": "CANCELLED",
        "type": "cancel",
        "message": f"Cancellation requested for session_id={session_id}",
        "session_id": session_id
    })

def safe_sid() -> str:
    """
    Returns a valid SocketIO session ID for the current request/session.
    """
    sid: Any = session.get('sid')
    if isinstance(sid, str) and sid:
        return sid
    sid = getattr(request, 'sid', None)
    if isinstance(sid, str) and sid:
        return sid
    # Fallback: try flask_socketio's request.sid
    try:
        sid = getattr(request, 'sid', None)
        if isinstance(sid, str) and sid:
            return sid
    except Exception:
        pass
    raise RuntimeError("No valid session ID found for SocketIO connection.")

def safe_rsplit(val, sep=None, maxsplit=-1):
    """
    Safely call .rsplit on a string-like object.
    Returns a list, or [str(val)] if not a string or error occurs.
    """
    try:
        if isinstance(val, str):
            return val.rsplit(sep, maxsplit)
        if isinstance(val, bytes):
            return val.decode(errors="replace").rsplit(sep, maxsplit)
        return [str(val)]
    except Exception:
        return [str(val)]