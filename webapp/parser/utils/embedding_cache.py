import re
import numpy as np
import os
import logging
from rich.console import Console
from functools import lru_cache
import threading
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm.exc import DetachedInstanceError
from sqlalchemy import select, inspect
from sqlalchemy.dialects.postgresql import insert
from ..utils.db_utils import get_session, engine
from ..utils.models import EmbeddingCache
from ..config import LOG_DIR, CACHE_DIR
console = Console()
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy.dialects").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
import itertools

_spinner = itertools.cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])
def get_loading_indicator():
    return next(_spinner)

# --- In-memory LRU cache for single-segment embedding retrieval ---
@lru_cache(maxsize=2048)
def get_embedding_from_memory(segment_hash):
    try:
        emb = load_embedding(segment_hash)
        if emb is None:
            indicator = get_loading_indicator()
            console.print(
                f"{indicator} [yellow][EMBEDDING CACHE] No embedding found for hash: {segment_hash}[/yellow]",
                highlight=False,
                end="\r"
            )
        return emb
    except DetachedInstanceError as e:
        console.print(f"[red][EMBEDDING CACHE ERROR][/red] DetachedInstanceError for hash {segment_hash}: {str(e)}", highlight=False)
        return None
    except Exception as e:
        console.print(f"[red][EMBEDDING CACHE ERROR][/red] Unexpected error for hash {segment_hash}: {str(e)}", highlight=False)
        return None

# --- In-memory process-level cache for batch operations ---
_batch_cache = {}
_batch_cache_lock = threading.Lock()

_db_lock = threading.Lock()

def ensure_embedding_cache_table():
    inspector = inspect(engine)
    table_name = EmbeddingCache.__tablename__
    if not inspector.has_table(table_name):
        console.print(f"[yellow][EMBEDDING CACHE] Table '{table_name}' does not exist. Creating...[/yellow]", highlight=False)
        try:
            EmbeddingCache.metadata.create_all(engine, tables=[EmbeddingCache.__table__])
            console.print(f"[green][EMBEDDING CACHE] Table '{table_name}' created.[/green]", highlight=False)
        except Exception as e:
            console.print(f"[red][EMBEDDING CACHE ERROR] Failed to create table '{table_name}': {e}[/red]", highlight=False)
            raise

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
                console.print(f"[red][EMBEDDING ERROR][/red] {str(e)}", highlight=False)
                return
    # Update in-memory cache
    with _batch_cache_lock:
        _batch_cache[segment_hash] = np.array(embedding, dtype=np.float32)

def load_embedding(segment_hash):
    """Load a single embedding from the cache, using in-memory cache if available."""
    with _batch_cache_lock:
        if segment_hash in _batch_cache:
            return _batch_cache[segment_hash]
    ensure_embedding_cache_table()
    obj = None
    with _db_lock:
        with get_session() as session:
            obj = session.get(EmbeddingCache, segment_hash)
            # Access .embedding inside the session!
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
        # Defensive: always convert to float32 numpy array
        arr = np.array(e, dtype=np.float32)
        emb_bytes = arr.tobytes()
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
                # Condensed log: show count and a few hashes
                hashes = [r["segment_hash"] for r in records]
                preview = ", ".join(str(h) for h in hashes[:3])
                if len(hashes) > 3:
                    preview += ", ..."
                console.log(
                    f"[green][EMBEDDING CACHE] Saved/updated {len(records)} embeddings in batch: [{preview}][/green]",
                    highlight=False,
                    end="\r"
                )
            except SQLAlchemyError as e:
                session.rollback()
                # Condensed error log: show only error type and first line, truncate if too long
                err_line = str(e).splitlines()[0]
                if len(err_line) > 120:
                    err_line = err_line[:117] + "..."
                console.print(
                    f"[red][BATCH EMBEDDING ERROR][/red] {type(e).__name__}: {err_line} (batch size: {len(records)})",
                    highlight=False,
                    end="\r"
                )
                return
    # Update in-memory cache (always latest)
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
    cache_hits = 0
    db_hits = 0
    # First, try in-memory cache
    with _batch_cache_lock:
        for h in segment_hashes:
            if h in _batch_cache:
                result[h] = _batch_cache[h]
                cache_hits += 1
    # Only query DB for missing
    missing = [h for h in segment_hashes if result[h] is None]
    if missing:
        try:
            with _db_lock:
                with get_session() as session:
                    stmt = select(EmbeddingCache).where(EmbeddingCache.segment_hash.in_(missing))
                    for obj in session.execute(stmt).scalars():
                        try:
                            emb = np.frombuffer(obj.embedding, dtype=np.float32)
                            result[obj.segment_hash] = emb
                            db_hits += 1
                            with _batch_cache_lock:
                                _batch_cache[obj.segment_hash] = emb
                        except Exception as e:
                            console.print(f"[red][EMBEDDING CACHE ERROR][/red] Failed to load embedding for hash {obj.segment_hash}: {e}", highlight=False)
        except SQLAlchemyError as e:
            console.print(f"[red][EMBEDDING CACHE DB ERROR][/red] {str(e)}", highlight=False)
    total = len(segment_hashes)
    console.log(f"[cyan][EMBEDDING CACHE] Batch load: {cache_hits} from cache, {db_hits} from DB, {total - cache_hits - db_hits} missing.[/cyan]", highlight=False)
    return result