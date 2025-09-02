from __future__ import annotations
# webapp/parser/utils/misc_utils.py
# ---------------------------------------------------------------
# Miscellaneous utility functions for Smart Elections Parser Webapp
# ---------------------------------------------------------------
import hashlib
import os
import orjson
from .logger_singleton import logger
from .shared_logic import safe_get
from typing import Dict, Any, List
from pathlib import Path
from ..config import CONTEXT_LIBRARY_PATH, PROCESSED_URLS_FILE, OUTPUT_CACHE

# --- Utility: Processed URL cache (unchanged, not DB) ---
def load_processed_urls() -> Dict[str, Any]:
    cache_path = Path(PROCESSED_URLS_FILE).resolve()
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

def is_safe_path(basedir: str, path: str) -> bool:
    """
    Returns True if 'path' is within 'basedir' (prevents path traversal).
    Accepts either a string or Path for basedir and path.
    """
    basedir = str(basedir)
    path = str(path)
    try:
        # Python 3.9+: use is_relative_to
        return Path(os.path.abspath(path)).resolve().is_relative_to(Path(basedir).resolve())
    except AttributeError:
        # For Python <3.9, fallback:
        basedir = os.path.abspath(basedir)
        path = os.path.abspath(path)
        return os.path.commonpath([basedir]) == os.path.commonpath([basedir, path])