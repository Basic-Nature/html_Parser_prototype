import re
import sqlite3
import numpy as np
import os
from functools import lru_cache
import threading

# --- In-memory LRU cache for single-segment embedding retrieval ---
@lru_cache(maxsize=2048)
def get_embedding_from_memory(segment_hash):
    return load_embedding(segment_hash)

# --- In-memory process-level cache for batch operations ---
_batch_cache = {}
_batch_cache_lock = threading.Lock()

def is_trivial_segment(seg):
    html = seg.get("html", "")
    tag = seg.get("tag", "")
    if not html or not html.strip():
        return True
    if tag in {"br", "hr", "wbr"} and not html.strip():
        return True
    if html.strip() in {"&nbsp;", "&#160;"}:
        return True
    classes = [c.lower() for c in seg.get("classes", [])]
    if tag == "span" and len(classes) > 0 and all("icon" in cls for cls in classes) and not re.sub(r"<[^>]+>", "", html).strip():
        return True
    return False

def get_embedding_cache_db_path():
    from ..config import BASE_DIR
    log_folder = os.path.join(os.path.dirname(BASE_DIR), "log")
    os.makedirs(log_folder, exist_ok=True)
    return os.path.join(log_folder, "embedding_cache.sqlite3")

_db_lock = threading.Lock()

def ensure_embedding_cache_table():
    db_path = get_embedding_cache_db_path()
    with _db_lock:
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                segment_hash TEXT PRIMARY KEY,
                embedding BLOB
            )
        """)
        conn.commit()
        conn.close()

def save_embedding(segment_hash, embedding):
    """Save a single embedding to the cache."""
    ensure_embedding_cache_table()
    db_path = get_embedding_cache_db_path()
    with _db_lock:
        conn = sqlite3.connect(db_path)
        emb_bytes = np.array(embedding).astype(np.float32).tobytes()
        conn.execute("REPLACE INTO embeddings (segment_hash, embedding) VALUES (?, ?)", (segment_hash, emb_bytes))
        conn.commit()
        conn.close()
    # Update in-memory cache
    with _batch_cache_lock:
        _batch_cache[segment_hash] = np.array(embedding, dtype=np.float32)

def load_embedding(segment_hash):
    """Load a single embedding from the cache, using in-memory cache if available."""
    with _batch_cache_lock:
        if segment_hash in _batch_cache:
            return _batch_cache[segment_hash]
    ensure_embedding_cache_table()
    db_path = get_embedding_cache_db_path()
    with _db_lock:
        conn = sqlite3.connect(db_path)
        cur = conn.execute("SELECT embedding FROM embeddings WHERE segment_hash = ?", (segment_hash,))
        row = cur.fetchone()
        conn.close()
    if row:
        emb = np.frombuffer(row[0], dtype=np.float32)
        with _batch_cache_lock:
            _batch_cache[segment_hash] = emb
        return emb
    return None

def save_embeddings_batch(hash_emb_list):
    """
    Save a batch of (segment_hash, embedding) tuples.
    """
    if not hash_emb_list:
        return
    ensure_embedding_cache_table()
    db_path = get_embedding_cache_db_path()
    with _db_lock:
        conn = sqlite3.connect(db_path)
        data = [(h, np.array(e).astype(np.float32).tobytes()) for h, e in hash_emb_list]
        conn.executemany("REPLACE INTO embeddings (segment_hash, embedding) VALUES (?, ?)", data)
        conn.commit()
        conn.close()
    # Update in-memory cache
    with _batch_cache_lock:
        for h, e in hash_emb_list:
            _batch_cache[h] = np.array(e, dtype=np.float32)

def load_embeddings_batch(segment_hashes):
    """
    Load a batch of embeddings for the given segment_hashes.
    Returns a dict: {segment_hash: embedding or None}
    """
    ensure_embedding_cache_table()
    db_path = get_embedding_cache_db_path()
    result = {h: None for h in segment_hashes}
    # First, try in-memory cache
    with _batch_cache_lock:
        for h in segment_hashes:
            if h in _batch_cache:
                result[h] = _batch_cache[h]
    # Only query DB for missing
    missing = [h for h in segment_hashes if result[h] is None]
    if missing:
        with _db_lock:
            conn = sqlite3.connect(db_path)
            placeholders = ",".join("?" for _ in missing)
            cur = conn.execute(f"SELECT segment_hash, embedding FROM embeddings WHERE segment_hash IN ({placeholders})", missing)
            for h, emb_bytes in cur.fetchall():
                emb = np.frombuffer(emb_bytes, dtype=np.float32)
                result[h] = emb
                with _batch_cache_lock:
                    _batch_cache[h] = emb
            conn.close()
    return result