from webapp.parser.html_election_parser import process_url_stream
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

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

def process_single_url(url, output_callback, idx, total, cancel_flag):
    if cancel_flag.is_set():
        output_callback(f"[CANCELLED] Skipping {url}\n")
        return
    output_callback(f"\n[Parsing {idx}/{total}] {url}\n")
    try:
        for line in process_url_stream(url):
            if cancel_flag.is_set():
                output_callback(f"[CANCELLED] Stopping {url}\n")
                return
            output_callback(line)
        output_callback(f"[DONE] Finished: {url}\n")
    except Exception as e:
        output_callback(f"[ERROR] Exception while processing {url}: {e}\n")

def process_urls_for_web(urls, output_callback, session_id, max_workers=2):
    """
    Processes a list of URLs for the webapp, streaming output via output_callback.
    Features:
      - Progress tracking
      - Per-session/user cancellation (call cancel_processing(session_id) from webapp)
      - Parallel processing (set max_workers > 1)
    """
    cancel_flag = cancellation_manager.get_flag(session_id)
    cancellation_manager.reset(session_id)
    if not urls:
        output_callback("[ERROR] No URLs provided.\n")
        return

    total = len(urls)
    output_callback(f"[INFO] Starting web pipeline for {total} URL(s)...\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx, url in enumerate(urls, 1):
            futures.append(executor.submit(process_single_url, url, output_callback, idx, total, cancel_flag))

        completed = 0
        for future in as_completed(futures):
            completed += 1
            output_callback(f"[PROGRESS] {completed}/{total} URLs complete.\n")
            if cancel_flag.is_set():
                output_callback("[CANCELLED] Processing stopped by user.\n")
                break

    if cancel_flag.is_set():
        output_callback("\n[INFO] Processing cancelled by user.\n")
    else:
        output_callback("\n[INFO] All URLs processed.\n")
    # Optionally clean up flag
    cancellation_manager.remove(session_id)

def cancel_processing(session_id):
    """Call this from the webapp to request cancellation for a session/user."""
    cancellation_manager.cancel(session_id)