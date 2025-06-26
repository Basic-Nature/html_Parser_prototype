import re
import numpy as np
import os
from rich.console import Console
from functools import lru_cache
import threading
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from webapp.parser.utils.db_utils import get_session
from webapp.parser.utils.models import EmbeddingCache
console = Console()
# --- In-memory LRU cache for single-segment embedding retrieval ---
@lru_cache(maxsize=2048)
def get_embedding_from_memory(segment_hash):
    return load_embedding(segment_hash)

# --- In-memory process-level cache for batch operations ---
_batch_cache = {}
_batch_cache_lock = threading.Lock()

_db_lock = threading.Lock()

def ensure_embedding_cache_table():
    # Table is managed by SQLAlchemy migrations; nothing to do here
    pass

def save_embedding(segment_hash, embedding):
    """Save a single embedding to the cache (PostgreSQL via SQLAlchemy)."""
    ensure_embedding_cache_table()
    emb_bytes = np.array(embedding).astype(np.float32).tobytes()
    with _db_lock:
        with get_session() as session:
            try:
                obj = session.get(EmbeddingCache, segment_hash)
                if obj:
                    obj.embedding = emb_bytes
                else:
                    obj = EmbeddingCache(segment_hash=segment_hash, embedding=emb_bytes)
                    session.add(obj)
                session.commit()
            except SQLAlchemyError as e:
                session.rollback()
                raise e
    # Update in-memory cache
    with _batch_cache_lock:
        _batch_cache[segment_hash] = np.array(embedding, dtype=np.float32)

def load_embedding(segment_hash):
    """Load a single embedding from the cache, using in-memory cache if available."""
    with _batch_cache_lock:
        if segment_hash in _batch_cache:
            return _batch_cache[segment_hash]
    ensure_embedding_cache_table()
    with _db_lock:
        with get_session() as session:
            obj = session.get(EmbeddingCache, segment_hash)
    if obj and obj.embedding:
        emb = np.frombuffer(obj.embedding, dtype=np.float32)
        with _batch_cache_lock:
            _batch_cache[segment_hash] = emb
        return emb
    return None

def save_embeddings_batch(hash_emb_list):
    """
    Save a batch of (segment_hash, embedding) tuples using PostgreSQL upsert (ON CONFLICT DO UPDATE).
    This prevents unique constraint errors and ensures robust batch saving.
    """
    if not hash_emb_list:
        return
    ensure_embedding_cache_table()
    # Prepare data for bulk upsert
    records = []
    for h, e in hash_emb_list:
        emb_bytes = np.array(e).astype(np.float32).tobytes()
        records.append({"segment_hash": h, "embedding": emb_bytes})
    with _db_lock:
        with get_session() as session:
            try:
                stmt = insert(EmbeddingCache).values(records)
                stmt = stmt.on_conflict_do_update(
                    index_elements=['segment_hash'],
                    set_={'embedding': stmt.excluded.embedding}
                )
                session.execute(stmt)
                session.commit()
            except SQLAlchemyError as e:
                session.rollback()
                # Print only the error message in a static line
                console.print(f"[red][BATCH EMBEDDING ERROR][/red] {str(e)}", highlight=False, end="\r")
                raise
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
            with get_session() as session:
                stmt = select(EmbeddingCache).where(EmbeddingCache.segment_hash.in_(missing))
                for obj in session.execute(stmt).scalars():
                    emb = np.frombuffer(obj.embedding, dtype=np.float32)
                    result[obj.segment_hash] = emb
                    with _batch_cache_lock:
                        _batch_cache[obj.segment_hash] = emb
    return result

# --- SQLAlchemy ORM model for embedding cache (if not already present) ---
# class EmbeddingCache(Base):
#     __tablename__ = 'embeddings'
#     segment_hash = Column(String, primary_key=True)
#     embedding = Column(LargeBinary)
#
# Ensure this model is present in models.py and included in Alembic migrations.