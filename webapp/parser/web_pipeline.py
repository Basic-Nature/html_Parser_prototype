from .html_election_parser import main
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import orjson
from .utils.shared_logger import SharedLogger, RichConsoleProxy
from .utils.user_prompt import UserPrompt
prompt = UserPrompt()
console = RichConsoleProxy()
logger = SharedLogger()
# Global cancellation flag (could be improved for multi-user)

class CancellationManager(threading.Thread):
    """
    Manages cancellation flags per session/user.
    """
    def __init__(self) -> None:
        self._flags = {}
        self._lock = threading.Lock()

    def get_flag(self, session_id) -> threading.Event:
        with self._lock:
            if session_id not in self._flags:
                self._flags[session_id] = threading.Event()
            return self._flags[session_id]

    def cancel(self, session_id) -> None:
        with self._lock:
            if session_id in self._flags:
                self._flags[session_id].set()

    def reset(self, session_id) -> None:
        with self._lock:
            if session_id in self._flags:
                self._flags[session_id].clear()

    def remove(self, session_id) -> None:
        with self._lock:
            if session_id in self._flags:
                del self._flags[session_id]

# Instantiate globally
cancellation_manager = CancellationManager()

def process_urls_for_web(
    urls,
    session_id,
    max_workers=2,
    mode="webapp"
) -> None:
    """
    Orchestrates parsing for webapp or CLI, handling cancellation and session-aware logging/prompting.
    Delegates all parsing logic to html_election_parser.main.
    Args:
        urls: List of URLs to process (or None to prompt).
        session_id: Unique session/user ID.
        max_workers: Number of parallel workers.
        mode: "webapp" or "cli"
    """
    cancel_flag = cancellation_manager.get_flag(session_id)
    cancellation_manager.reset(session_id)

    # --- Mode-aware prompt and output functions ---
    if mode == "webapp":
        logger.set_mode("webapp")
        logger.set_format("json")
        prompt.set_mode("webapp")
        def prompt_func(message) -> str:
            return prompt.prompt_user(message, session_id=session_id, timeout=300)
        def output_func(msg) -> None:
            # Defensive: always stringify non-string messages
            if not isinstance(msg, (str, bytes)):
                try:
                    msg = orjson.dumps(msg, option=orjson.OPT_INDENT_2).decode("utf-8")
                except Exception:
                    msg = str(msg)
            logger.info(msg, context={"session_id": session_id})
    else:
        logger.set_mode("cli")
        logger.set_format("plain")
        prompt.set_mode("cli")
        def prompt_func(message) -> str:
            return console.input(message)
        def output_func(msg) -> None:
            if not isinstance(msg, (str, bytes)):
                try:
                    msg = orjson.dumps(msg, option=orjson.OPT_INDENT_2).decode("utf-8")
                except Exception:
                    msg = str(msg)
            console.print(msg)

    try:
        if urls is None:
            # Interactive: let main() handle all logic, including URL loading and selection
            main(
                prompt_func=prompt_func,
                output_func=output_func,
                session_id=session_id,
                cancel_flag=cancel_flag
            )
        else:
            # Batch: pass urls directly to main (main must support this)
            if isinstance(urls, str):
                urls = [urls]
            elif not isinstance(urls, list):
                output_func("[ERROR] Invalid URLs input.")
                cancellation_manager.remove(session_id)
                return

            total = len(urls)
            output_func(f"[INFO] Starting pipeline for {total} URL(s)...")
            completed = 0
            errors = []
            results = []  # Collect successful results
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
                        ): url for url in urls
                    }
                    for future in as_completed(future_to_url):
                        url = future_to_url[future]
                        try:
                            result = future.result()
                            results.append((url, result))
                            # Output a success message with result summary if available
                            if result is not None:
                                output_func(f"[SUCCESS] {url} processed: {result}")
                            else:
                                output_func(f"[SUCCESS] {url} processed.")
                        except Exception as exc:
                            errors.append((url, str(exc)))
                            output_func(f"[ERROR] Exception for {url}: {exc}")
                        completed += 1
                        output_func(f"[PROGRESS] {completed}/{total} URLs complete.")
                        if cancel_flag.is_set():
                            output_func("[CANCELLED] Processing stopped by user.")
                            break
                # Optionally, report all errors at the end
                if errors:
                    output_func(f"[SUMMARY] {len(errors)} URLs failed:")
                    for url, err in errors:
                        output_func(f"  - {url}: {err}")
                # Optionally, summarize all results at the end
                if results:
                    output_func(f"[SUMMARY] {len(results)} URLs processed successfully:")
                    for url, res in results:
                        output_func(f"  - {url}: {res}")
            else:
                for url in urls:
                    if cancel_flag.is_set():
                        output_func("[CANCELLED] Processing stopped by user.")
                        break
                    try:
                        main(
                            prompt_func=prompt_func,
                            output_func=output_func,
                            url=url,
                            session_id=session_id,
                            cancel_flag=cancel_flag
                        )
                    except Exception as exc:
                        errors.append((url, str(exc)))
                        output_func(f"[ERROR] Exception for {url}: {exc}")
                    completed += 1
                    output_func(f"[PROGRESS] {completed}/{total} URLs complete.")
                if errors:
                    output_func(f"[SUMMARY] {len(errors)} URLs failed:")
                    for url, err in errors:
                        output_func(f"  - {url}: {err}")
    except Exception as e:
        import traceback
        output_func(f"[ERROR] Exception in pipeline: {e}\n{traceback.format_exc()}")
    finally:
        if cancel_flag.is_set():
            output_func("\n[INFO] Processing cancelled by user.")
        else:
            output_func("\n[INFO] All URLs processed.")
        cancellation_manager.remove(session_id)
    
def cancel_processing(session_id) -> None:
    cancellation_manager.cancel(session_id)