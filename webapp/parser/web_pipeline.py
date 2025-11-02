import os
import threading
import time
import traceback

import orjson

from .config import (
    PIPELINE_HEARTBEAT_INTERVAL,
    PIPELINE_MAX_WORKERS,
    URL_LIST_FILE,
)
from .html_election_parser import main
from .utils.logger_singleton import logger, prompt
from .utils.shared_logic import safe_clear, safe_is_set, safe_set


class CancellationManager(threading.Thread):
    """
    Manages cancellation flags per session/user.
    """
    def __init__(self) -> None:
        super().__init__()
        self._flags = {}
        self._lock = threading.Lock()
        self._unknown_warned = set()

    def get_flag(self, session_id) -> threading.Event:
        with self._lock:
            if session_id not in self._flags:
                self._flags[session_id] = threading.Event()
                # Only log to backend, not frontend
                print(f"[DEBUG] Created cancellation flag for session_id={session_id}")
            else:
                print(f"[DEBUG] Reusing cancellation flag for session_id={session_id}")
            return self._flags[session_id]

    def cancel(self, session_id) -> None:
        with self._lock:
            if session_id in self._flags:
                safe_set(self._flags[session_id])
                logger.info({
                    "level": "CANCELLED",
                    "type": "cancel",
                    "message": f"Cancellation requested (session_id={session_id})",
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
                ev = self._flags[session_id]
                if ev.is_set():
                    safe_clear(ev)
                    print(f"[DEBUG] Cancellation flag reset for session_id={session_id}")
            else:
                if session_id not in self._unknown_warned:
                    self._unknown_warned.add(session_id)
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
                # Only log to backend, not frontend
                print(f"[DEBUG] Cancellation flag removed for session_id={session_id}")
            else:
                # Only emit to frontend if something is wrong
                if session_id not in self._unknown_warned:
                    self._unknown_warned.add(session_id)
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
    output_bypass=False,
    manual_source='input',
    disable_internal_heartbeat=False,
    force_parse_format=None,
    force_parse_input_file=None,
    **kwargs
) -> None:
    """
    Single-run (interactive) pipeline:
      - Sets up logging / heartbeat
      - Invokes main() exactly once (interactive or with provided urls)
      - Passes prompt_queue and max_workers to main()
      - No per-URL threading, batching, summary aggregation, or prompt queue loop here.
      Batch / parallel logic is delegated to main() internally.
    """
    cancellation_manager.reset(session_id)

    logger.set_mode("webapp")
    logger.set_format("json")
    if emit_func:
        prompt.set_mode("webapp")
        prompt.set_socketio_emit_func(emit_func)

    if not disable_internal_heartbeat:
        threading.Thread(
            target=heartbeat,
            args=(session_id, cancel_flag, PIPELINE_HEARTBEAT_INTERVAL, emit_func),
            daemon=True
        ).start()

    logger.info({
        "level": "INFO",
        "type": "status",
        "message": f"Session started for {session_id}",
        "session_id": session_id
    })

    try:
        urls = kwargs.get("urls")

        # Always pass prompt_queue and max_workers to main()
        main_kwargs = dict(
            session_id=session_id,
            cancel_flag=cancel_flag,
            output_bypass=output_bypass,
            manual_source=manual_source,
            force_parse_input_file=force_parse_input_file,
            force_parse_format=force_parse_format,
            # If uploads is selected, do not fall back to URL list on failure
            continue_on_override_failure=False if manual_source == 'uploads' else True,
            prompt_queue=prompt_queue,
            max_workers=max_workers,
            **kwargs
        )

        if urls is None:
            # Interactive / internal URL selection path (main() handles listing & prompts)
            try:
                if os.path.exists(URL_LIST_FILE):
                    with open(URL_LIST_FILE, "r", encoding="utf-8") as f:
                        raw_urls = [
                            ln.strip() for ln in f
                            if ln.strip() and not ln.strip().startswith("#")
                        ]
                else:
                    raw_urls = []
                if not raw_urls:
                    logger.error({
                        "level": "ERROR",
                        "type": "input",
                        "message": "urls.txt has no usable URLs (aborting before interactive main()).",
                        "session_id": session_id
                    })
                    logger.info({
                        "level": "INFO",
                        "type": "input",
                        "message": f"Edit file at: {URL_LIST_FILE}",
                        "session_id": session_id
                    })
                    cancellation_manager.remove(session_id)
                    return
            except Exception as e:
                logger.error({
                    "level": "ERROR",
                    "type": "exception",
                    "message": f"Failed preparing URL list: {e}",
                    "session_id": session_id,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                })
                cancellation_manager.remove(session_id)
                return

            main(**main_kwargs)
        else:
            # Explicit URLs provided (pass through to main; let it batch internally)
            if isinstance(urls, str):
                urls = [urls]
            if not isinstance(urls, list) or not all(isinstance(u, str) for u in urls):
                logger.error({
                    "level": "ERROR",
                    "type": "input",
                    "message": "Invalid 'urls' argument (must be list[str] or str).",
                    "session_id": session_id
                })
                cancellation_manager.remove(session_id)
                return

            logger.info({
                "level": "INFO",
                "type": "status",
                "message": f"Dispatching main() with {len(urls)} provided URL(s).",
                "count": len(urls),
                "session_id": session_id
            })

            main(urls=urls, emit_func=emit_func, **main_kwargs)

        # Completion (single-run)
        if safe_is_set(cancel_flag):
            logger.info({
                "level": "CANCELLED",
                "type": "cancel",
                "message": "Run cancelled.",
                "session_id": session_id
            })
        else:
            logger.info({
                "level": "SUMMARY",
                "type": "summary",
                "message": "Single-run main() completed.",
                "session_id": session_id
            })

    except Exception as e:
        logger.error({
            "level": "ERROR",
            "type": "exception",
            "message": f"Unhandled exception in process_urls_for_web: {e}",
            "session_id": session_id,
            "error": str(e),
            "traceback": traceback.format_exc()
        })
    finally:
        cancellation_manager.remove(session_id)

def cancel_processing(session_id) -> None:
    cancellation_manager.cancel(session_id)
    logger.info({
        "level": "CANCELLED",
        "type": "cancel",
        "message": f"Cancellation requested for session_id={session_id}",
        "session_id": session_id
    })

