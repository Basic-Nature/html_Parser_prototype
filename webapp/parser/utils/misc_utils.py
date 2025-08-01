from __future__ import annotations
import hashlib
import os
import orjson
from ..utils.logger_singleton import logger
from ..utils.shared_logic import safe_get
from typing import Dict, Any, List
from pathlib import Path
from ..config import CONTEXT_LIBRARY_PATH

# --- Utility: Processed URL cache (unchanged, not DB) ---
def load_processed_urls() -> Dict[str, Any]:
    from ..utils.output_utils import CACHE_FILE
    cache_path = Path(CACHE_FILE).resolve()
    if not cache_path.exists() or os.path.getsize(cache_path) == 0:
        return {}
    with cache_path.open('rb') as f:
        try:
            entries = orjson.loads(f.read())
            if not isinstance(entries, list):
                entries = []
        except Exception:
            entries = []
    processed = {}
    for entry in entries:
        url = safe_get(entry, "url")
        if url:
            processed[url] = entry
    return processed

# --- DB Path Safety (for legacy compatibility, not used for SQLAlchemy) ---
def _safe_db_path(path) -> str:
    return str(Path(path or CONTEXT_LIBRARY_PATH).resolve())

def load_output_cache(path=None) -> List[dict]:
    if path is None:
        from ..Context_Integration.context_organizer import OUTPUT_CACHE
        path = OUTPUT_CACHE
    safe_path = Path(_safe_db_path(path)).resolve()
    if not safe_path.exists():
        return []
    with open(safe_path, "rb") as f:
        return [orjson.loads(line) for line in f if line.strip()]
    
def file_hash(filepath, algo="sha256", blocksize=65536):
    """Compute the hash of a file for deduplication/integrity, with error handling."""
    if not isinstance(filepath, str) or not filepath or not os.path.exists(filepath):
        logger.error(f"[file_hash] Invalid or missing file: {filepath}")
        return None
    try:
        h = hashlib.new(algo)
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(blocksize), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        logger.error(f"[file_hash] Error hashing file '{filepath}': {e}")
        return None

