from .html_election_parser import main
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from .utils.shared_logger import SharedLogger
from .utils.user_prompt import UserPrompt
prompt = UserPrompt()
logger = SharedLogger()
# Global cancellation flag (could be improved for multi-user)

class CancellationManager:
    """
    Manages cancellation flags per session/user.
    """
    def __init__(self):
        self._flags = {}
        self._lock = threading.Lock()

    def get_flag(self, session_id):
        with self._lock:
            if session_id not in self._flags:
                self._flags[session_id] = threading.Event()
            return self._flags[session_id]

    def cancel(self, session_id):
        with self._lock:
            if session_id in self._flags:
                self._flags[session_id].set()

    def reset(self, session_id):
        with self._lock:
            if session_id in self._flags:
                self._flags[session_id].clear()

    def remove(self, session_id):
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
):
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

    # Mode-aware prompt and output functions
    if mode == "webapp":
        def prompt_func(message):
            return prompt.prompt_user(message, session_id=session_id, timeout=300)
        def output_func(msg):
            if isinstance(msg, list):
                msg = "\n".join(str(item) for item in msg)
            logger.info(msg, context={"session_id": session_id})
    else:  # CLI mode
        def prompt_func(message):
            return input(message)
        def output_func(msg):
            print(msg)

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
    
def cancel_processing(session_id):
    cancellation_manager.cancel(session_id)