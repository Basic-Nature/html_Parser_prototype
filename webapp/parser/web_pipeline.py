from .html_election_parser import main
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import os
import orjson
import traceback
from .utils.shared_logger import SharedLogger, RichConsoleProxy
from .utils.user_prompt import UserPrompt

prompt = UserPrompt()
console = RichConsoleProxy()
logger = SharedLogger()

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
                "message": f"get_flag called for session_id={session_id}"
            })
            return self._flags[session_id]

    def cancel(self, session_id) -> None:
        with self._lock:
            if session_id in self._flags:
                self._flags[session_id].set()
                logger.info({
                    "level": "INFO",
                    "message": f"Cancellation requested for session_id={session_id}"
                })

    def reset(self, session_id) -> None:
        with self._lock:
            if session_id in self._flags:
                self._flags[session_id].clear()
                logger.info({
                    "level": "DEBUG",
                    "message": f"Cancellation reset for session_id={session_id}"
                })

    def remove(self, session_id) -> None:
        with self._lock:
            if session_id in self._flags:
                del self._flags[session_id]
                logger.info({
                    "level": "DEBUG",
                    "message": f"Cancellation flag removed for session_id={session_id}"
                })

# Instantiate globally
cancellation_manager = CancellationManager()

def heartbeat(session_id, cancel_flag, interval=10):
    while True:
        time.sleep(interval)
        logger.info({
            "level": "HEARTBEAT",
            "message": "Session is alive.",
            "session_id": session_id
        })
        if cancel_flag.is_set():
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
    urls,
    session_id,
    max_workers=2,
    mode="webapp"
) -> None:
    """
    Advanced pipeline: per-URL timing, global timing, live progress, error threshold, tracebacks,
    output file saving, env-configurable workers, heartbeat.
    """
    # 7. Customizable Worker Count via Environment Variable
    max_workers = int(os.environ.get("PIPELINE_MAX_WORKERS", max_workers))
    MAX_ERRORS = int(os.environ.get("PIPELINE_MAX_ERRORS", 5))
    HEARTBEAT_INTERVAL = int(os.environ.get("PIPELINE_HEARTBEAT_INTERVAL", 10))

    cancel_flag = cancellation_manager.get_flag(session_id)
    cancellation_manager.reset(session_id)

    # Start heartbeat thread
    threading.Thread(target=heartbeat, args=(session_id, cancel_flag, HEARTBEAT_INTERVAL), daemon=True).start()

    # --- Mode-aware prompt and output functions ---
    if mode == "webapp":
        logger.set_mode("webapp")
        logger.set_format("json")
        prompt.set_mode("webapp")
        def prompt_func(message) -> str:
            return prompt.prompt_user(message, session_id=session_id, timeout=300)
        def output_func(msg) -> None:
            if isinstance(msg, (dict, list)):
                logger.info(msg, context={"session_id": session_id})
            else:
                logger.info(str(msg), context={"session_id": session_id})
    else:
        logger.set_mode("cli")
        logger.set_format("plain")
        prompt.set_mode("cli")
        def prompt_func(message) -> str:
            return console.input(message)
        def output_func(msg) -> None:
            if isinstance(msg, dict) and "message" in msg:
                console.print(msg["message"])
            elif isinstance(msg, (list, tuple)):
                for item in msg:
                    console.print(str(item))
            else:
                console.print(str(msg))

    pipeline_start = time.time()
    try:
        logger.info({
            "level": "INFO",
            "message": f"Session started for {session_id}",
            "session_id": session_id
        })

        if urls is None:
            main(
                prompt_func=prompt_func,
                output_func=output_func,
                session_id=session_id,
                cancel_flag=cancel_flag
            )
        else:
            if isinstance(urls, str):
                urls = [urls]
            elif not isinstance(urls, list) or not all(isinstance(u, str) for u in urls):
                output_func({"level": "ERROR", "message": "[ERROR] Invalid URLs input. Must be list of strings."})
                cancellation_manager.remove(session_id)
                return

            total = len(urls)
            if total == 0:
                output_func({"level": "WARNING", "message": "No URLs provided for processing."})
                cancellation_manager.remove(session_id)
                return

            output_func({"level": "INFO", "message": f"Starting pipeline for {total} URL(s)...", "total": total})
            completed = 0
            errors = []
            results = []
            url_timings = []

            if max_workers and max_workers > 1 and total > 1:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_url = {
                        executor.submit(
                            main,
                            prompt_func=prompt_func,
                            output_func=output_func,
                            url=url,
                            session_id=session_id,
                            cancel_flag=cancel_flag
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
                            output_func({
                                "level": "SUCCESS",
                                "message": f"{url} processed: {result}" if result is not None else f"{url} processed.",
                                "url": url,
                                "result": result,
                                "index": idx,
                                "duration": url_duration
                            })
                        except Exception as exc:
                            url_duration = time.time() - url_start
                            errors.append((url, str(exc)))
                            output_func({
                                "level": "ERROR",
                                "message": f"Exception for {url}: {exc}",
                                "url": url,
                                "error": str(exc),
                                "traceback": traceback.format_exc(),
                                "index": idx,
                                "duration": url_duration
                            })
                        completed += 1
                        # 3. Live Progress Events
                        output_func({
                            "level": "PROGRESS",
                            "message": f"Progress update: {completed}/{total}",
                            "progress": completed / total,
                            "current_url": url,
                            "index": idx
                        })
                        logger.info({
                            "level": "INFO",
                            "message": f"Processing URL {completed}/{total}: {url}",
                            "progress": completed / total,
                            "url": url,
                            "index": idx,
                            "session_id": session_id
                        })
                        if cancel_flag.is_set():
                            output_func({"level": "CANCELLED", "message": "[CANCELLED] Processing stopped by user."})
                            break
                        # 4. Early Exit on Too Many Errors
                        if len(errors) >= MAX_ERRORS:
                            output_func({
                                "level": "ERROR",
                                "message": f"Too many errors ({MAX_ERRORS}), aborting pipeline.",
                                "errors": errors
                            })
                            break
                if errors:
                    output_func({
                        "level": "SUMMARY",
                        "message": f"{len(errors)} URLs failed.",
                        "errors": errors
                    })
                if results:
                    output_func({
                        "level": "SUMMARY",
                        "message": f"{len(results)} URLs processed successfully.",
                        "results": results
                    })
            else:
                for i, url in enumerate(urls):
                    if cancel_flag.is_set():
                        output_func({"level": "CANCELLED", "message": "[CANCELLED] Processing stopped by user."})
                        break
                    url_start = time.time()
                    try:
                        main(
                            prompt_func=prompt_func,
                            output_func=output_func,
                            url=url,
                            session_id=session_id,
                            cancel_flag=cancel_flag
                        )
                        url_duration = time.time() - url_start
                        results.append((url, "success"))
                        url_timings.append({"url": url, "index": i, "duration": url_duration})
                        output_func({
                            "level": "SUCCESS",
                            "message": f"{url} processed.",
                            "url": url,
                            "index": i,
                            "duration": url_duration
                        })
                    except Exception as exc:
                        url_duration = time.time() - url_start
                        errors.append((url, str(exc)))
                        output_func({
                            "level": "ERROR",
                            "message": f"Exception for {url}: {exc}",
                            "url": url,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                            "index": i,
                            "duration": url_duration
                        })
                    completed += 1
                    output_func({
                        "level": "PROGRESS",
                        "message": f"Progress update: {completed}/{total}",
                        "progress": completed / total,
                        "current_url": url,
                        "index": i
                    })
                    logger.info({
                        "level": "INFO",
                        "message": f"Processing URL {completed}/{total}: {url}",
                        "progress": completed / total,
                        "url": url,
                        "index": i,
                        "session_id": session_id
                    })
                    if len(errors) >= MAX_ERRORS:
                        output_func({
                            "level": "ERROR",
                            "message": f"Too many errors ({MAX_ERRORS}), aborting pipeline.",
                            "errors": errors
                        })
                        break
                if errors:
                    output_func({
                        "level": "SUMMARY",
                        "message": f"{len(errors)} URLs failed.",
                        "errors": errors
                    })
        # 2. Global Pipeline Timing and Summary
        pipeline_duration = time.time() - pipeline_start
        output_func({
            "level": "SUMMARY",
            "message": "Pipeline finished.",
            "total_urls": total if urls else 0,
            "errors": len(errors),
            "duration": pipeline_duration,
            "url_timings": url_timings
        })
        # 6. Support for Output File Saving
        report_path = save_pipeline_report(session_id, results, errors)
        output_func({
            "level": "INFO",
            "message": f"Pipeline report saved.",
            "report_path": report_path
        })
    except Exception as e:
        output_func({
            "level": "ERROR",
            "message": f"Exception in pipeline: {e}",
            "traceback": traceback.format_exc()
        })
        logger.info({
            "level": "ERROR",
            "message": f"Exception in pipeline: {e}",
            "traceback": traceback.format_exc(),
            "session_id": session_id
        })
    finally:
        if cancel_flag.is_set():
            output_func({"level": "INFO", "message": "Processing cancelled by user."})
            logger.info({
                "level": "INFO",
                "message": "Processing cancelled by user.",
                "session_id": session_id
            })
        else:
            output_func({"level": "INFO", "message": "All URLs processed."})
            logger.info({
                "level": "INFO",
                "message": "All URLs processed.",
                "session_id": session_id
            })
        cancellation_manager.remove(session_id)

def cancel_processing(session_id) -> None:
    cancellation_manager.cancel(session_id)
    logger.info({
        "level": "INFO",
        "message": f"cancel_processing called for session_id={session_id}"
    })