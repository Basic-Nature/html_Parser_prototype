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

def save_embeddings_batch(hash_emb_list):
    """
    Save a batch of (segment_hash, embedding) tuples.
    """
    if not hash_emb_list:
        return
    ensure_embedding_cache_table()
    db_path = get_embedding_cache_db_path()
    conn = sqlite3.connect(db_path)
    data = [(h, np.array(e).astype(np.float32).tobytes()) for h, e in hash_emb_list]
    conn.executemany("REPLACE INTO embeddings (segment_hash, embedding) VALUES (?, ?)", data)
    conn.commit()
    conn.close()

def load_embeddings_batch(segment_hashes):
    """
    Load a batch of embeddings for the given segment_hashes.
    Returns a dict: {segment_hash: embedding or None}
    """
    ensure_embedding_cache_table()
    db_path = get_embedding_cache_db_path()
    conn = sqlite3.connect(db_path)
    placeholders = ",".join("?" for _ in segment_hashes)
    cur = conn.execute(f"SELECT segment_hash, embedding FROM embeddings WHERE segment_hash IN ({placeholders})", segment_hashes)
    result = {h: None for h in segment_hashes}
    for h, emb_bytes in cur.fetchall():
        result[h] = np.frombuffer(emb_bytes, dtype=np.float32)
    conn.close()
    return result