import sqlite3
import numpy as np
import os

def get_embedding_cache_db_path():
    from ..config import BASE_DIR
    log_folder = os.path.join(os.path.dirname(BASE_DIR), "log")
    os.makedirs(log_folder, exist_ok=True)
    return os.path.join(log_folder, "embedding_cache.sqlite3")

def ensure_embedding_cache_table():
    db_path = get_embedding_cache_db_path()
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
    ensure_embedding_cache_table()
    db_path = get_embedding_cache_db_path()
    conn = sqlite3.connect(db_path)
    # Store as bytes
    emb_bytes = np.array(embedding).astype(np.float32).tobytes()
    conn.execute("REPLACE INTO embeddings (segment_hash, embedding) VALUES (?, ?)", (segment_hash, emb_bytes))
    conn.commit()
    conn.close()

def load_embedding(segment_hash):
    ensure_embedding_cache_table()
    db_path = get_embedding_cache_db_path()
    conn = sqlite3.connect(db_path)
    cur = conn.execute("SELECT embedding FROM embeddings WHERE segment_hash = ?", (segment_hash,))
    row = cur.fetchone()
    conn.close()
    if row:
        emb = np.frombuffer(row[0], dtype=np.float32)
        return emb
    return None
