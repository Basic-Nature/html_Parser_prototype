import orjson
import threading
import logging
from pathlib import Path

logger = logging.getLogger("context_schema")
_CONTEXT_LOCK = threading.Lock()
SCHEMA_VERSION = "1.0"

DEFAULT_STRUCTURE = {
    "schema_version": SCHEMA_VERSION,
    "contests": [],
    "panels": [],
    "tables": [],
    "buttons": [],
    "metadata": {},
}

def load_context_library(path):
    path = Path(path)
    if not path.exists():
        logger.warning(f"Context library not found at {path}, creating new.")
        save_context_library(DEFAULT_STRUCTURE, path)
        return DEFAULT_STRUCTURE.copy()
    try:
        with _CONTEXT_LOCK, open(path, "rb") as f:
            data = orjson.loads(f.read())
            if data.get("schema_version") != SCHEMA_VERSION:
                logger.warning("Schema version mismatch! Consider migrating context library.")
            return data
    except Exception as e:
        logger.error(f"Failed to load context library: {e}")
        return DEFAULT_STRUCTURE.copy()

def save_context_library(data, path):
    path = Path(path)
    with _CONTEXT_LOCK, open(path, "wb") as f:
        f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))

def update_context_library(path, update_fn):
    """Atomic update: load, modify with update_fn, save."""
    data = load_context_library(path)
    update_fn(data)
    save_context_library(data, path)