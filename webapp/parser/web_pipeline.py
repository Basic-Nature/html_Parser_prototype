from .html_election_parser import main, load_urls
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from .Context_Integration.context_organizer import ContextOrganizer
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

def process_single_url(url, idx, total, session_id, cancel_flag):
    if cancel_flag.is_set():
        logger.info(f"[CANCELLED] Skipping {url}", context={"session_id": session_id})
        return
    logger.info(f"\n[Parsing {idx}/{total}] {url}", context={"session_id": session_id})
    try:
        # Step 1: Parse the URL and get raw_context
        raw_context = main(url, session_id=session_id)  # Pass session_id if supported

        # Step 2: Organize context using ContextOrganizer
        organizer = ContextOrganizer()
        result = organizer.organize_context(raw_context)

        # Step 3: Output summary/log
        logger.info(f"[DONE] Finished: {url}", context={"session_id": session_id})
        if "log" in result:
            for line in result["log"]:
                logger.info(f"[LOG] {line}", context={"session_id": session_id})
        if "error" in result and result["error"]:
            logger.error(f"[ERROR] {result['error']}", context={"session_id": session_id})
    except Exception as e:
        logger.error(f"[ERROR] Exception while processing {url}: {e}", context={"session_id": session_id})

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
            logger.info(msg, context={"session_id": session_id})
    else:  # CLI mode
        def prompt_func(message):
            return input(message)
        def output_func(msg):
            print(msg)

    # Load URLs if not provided
    if not urls:
        urls = load_urls(prompt_func=prompt_func)

    if not urls:
        output_func("[ERROR] No URLs provided.")
        cancellation_manager.remove(session_id)
        return
    try:
        max_workers = int(max_workers)
    except Exception:
        max_workers = 2

    if isinstance(urls, str):
        urls = [urls]
        
    total = len(urls)
    output_func(f"[INFO] Starting pipeline for {total} URL(s)...")

    try:
        if max_workers > 1 and total > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for url in urls:
                    futures.append(executor.submit(
                        main,
                        prompt_func=prompt_func,
                        output_func=output_func,
                        url=url,
                        session_id=session_id,
                        cancel_flag=cancel_flag
                    ))
                completed = 0
                for future in as_completed(futures):
                    completed += 1
                    output_func(f"[PROGRESS] {completed}/{total} URLs complete.")
                    if cancel_flag.is_set():
                        output_func("[CANCELLED] Processing stopped by user.")
                        break
        else:
            for url in urls:
                if cancel_flag.is_set():
                    output_func("[CANCELLED] Processing stopped by user.")
                    break
                main(
                    prompt_func=prompt_func,
                    output_func=output_func,
                    url=url,
                    session_id=session_id,
                    cancel_flag=cancel_flag
                )
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