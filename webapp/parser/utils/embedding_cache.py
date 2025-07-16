import os
import logging
import threading
import atexit
import numpy as np
from functools import lru_cache
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm.exc import DetachedInstanceError
from sqlalchemy import select, inspect
from sqlalchemy.dialects.postgresql import insert
from ..utils.db_utils import get_session, engine
from ..utils.models import EmbeddingCache
from ..config import LOG_DIR, CACHE_DIR
from ..utils.shared_logger import RichConsoleProxy, SharedLogger, SQLAlchemyToSharedLoggerHandler

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    import pickle
    JOBLIB_AVAILABLE = False

console = RichConsoleProxy()
logger = SharedLogger()

for name in [
    "sqlalchemy",
    "sqlalchemy.engine",
    "sqlalchemy.dialects",
    "sqlalchemy.pool"
]:
    logger_obj = logging.getLogger(name)
    logger_obj.addHandler(SQLAlchemyToSharedLoggerHandler(logger))


DISK_CACHE_PATH = os.path.join(CACHE_DIR, "embedding_disk_cache.pkl")
MISSING_LOG_PATH = os.path.join(LOG_DIR, "missing_embeddings.log")

with logger.progress_bar("Loading...", total=100) as update_progress:
    for i in range(100):
        # ... do work ...
        update_progress(i + 1)

# --- In-memory process-level cache for batch operations ---
_batch_cache = {}
_batch_cache_lock = threading.Lock()
_db_lock = threading.Lock()

# --- Disk cache using joblib if available, else pickle ---
if JOBLIB_AVAILABLE:
    def load_disk_cache():
        if os.path.exists(DISK_CACHE_PATH):
            try:
                cache = joblib.load(DISK_CACHE_PATH)
                console.print(f"[cyan][EMBEDDING CACHE] Loaded disk cache with {len(cache)} embeddings.[/cyan]")
                return cache
            except Exception as e:
                console.print(f"[red][EMBEDDING CACHE] Failed to load disk cache: {e}[/red]")
        return {}
    def save_disk_cache():
        try:
            joblib.dump(_disk_cache, DISK_CACHE_PATH)
            console.print(f"[cyan][EMBEDDING CACHE] Saved disk cache with {len(_disk_cache)} embeddings.[/cyan]")
        except Exception as e:
            console.print(f"[red][EMBEDDING CACHE] Failed to save disk cache: {e}[/red]")
else:
    def load_disk_cache():
        if os.path.exists(DISK_CACHE_PATH):
            try:
                with open(DISK_CACHE_PATH, "rb") as f:
                    cache = pickle.load(f)
                console.print(f"[cyan][EMBEDDING CACHE] Loaded disk cache with {len(cache)} embeddings.[/cyan]")
                return cache
            except Exception as e:
                console.print(f"[red][EMBEDDING CACHE] Failed to load disk cache: {e}[/red]")
        return {}
    def save_disk_cache():
        try:
            with open(DISK_CACHE_PATH, "wb") as f:
                pickle.dump(_disk_cache, f)
            console.print(f"[cyan][EMBEDDING CACHE] Saved disk cache with {len(_disk_cache)} embeddings.[/cyan]")
        except Exception as e:
            console.print(f"[red][EMBEDDING CACHE] Failed to save disk cache: {e}[/red]")

_disk_cache = load_disk_cache()
atexit.register(save_disk_cache)

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

def compute_embedding_for_hash(segment_hash):
    """
    Compute or fetch the embedding for a given hash.
    Replace this logic with your actual embedding computation or retrieval.
    """
    # Example: If you have a deterministic way to get the original text from the hash,
    # and a model to compute the embedding, use that here.
    # For demonstration, we'll just return None.
    # Example:
    # text = reverse_lookup_text(segment_hash)
    # if text:
    #     return my_embedding_model.encode(text)
    return None

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
    # Update in-memory and disk cache
    with _batch_cache_lock:
        _batch_cache[segment_hash] = np.array(embedding, dtype=np.float32)
    _disk_cache[segment_hash] = np.array(embedding, dtype=np.float32)

def load_embedding(segment_hash):
    """Load a single embedding from the cache, using in-memory and disk cache if available."""
    with _batch_cache_lock:
        if segment_hash in _batch_cache:
            return _batch_cache[segment_hash]
    if segment_hash in _disk_cache:
        emb = _disk_cache[segment_hash]
        with _batch_cache_lock:
            _batch_cache[segment_hash] = emb
        return emb
    ensure_embedding_cache_table()
    obj = None
    with _db_lock:
        with get_session() as session:
            obj = session.get(EmbeddingCache, segment_hash)
            if obj and obj.embedding:
                emb = np.frombuffer(obj.embedding, dtype=np.float32)
                with _batch_cache_lock:
                    _batch_cache[segment_hash] = emb
                _disk_cache[segment_hash] = emb
                return emb
    # Log missing hash for diagnostics
    with open(MISSING_LOG_PATH, "a") as f:
        f.write(f"{segment_hash}\n")
    return None

@lru_cache(maxsize=2048)
def get_embedding_from_memory(segment_hash):
    try:
        emb = load_embedding(segment_hash)
        if emb is None:
            msg = f"[EMBEDDING CACHE] No embedding found for hash: {segment_hash}"
            if logger.mode == "cli":
                # In CLI, overwrite the line in place
                print(msg.ljust(80), end="\r", flush=True)
            else:
                # In webapp mode, emit as a normal log message
                logger.warning(msg)
        return emb
    except DetachedInstanceError as e:
        logger.error(f"[EMBEDDING CACHE ERROR] DetachedInstanceError for hash {segment_hash}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"[EMBEDDING CACHE ERROR] Unexpected error for hash {segment_hash}: {str(e)}")
        return None

def save_embeddings_batch(hash_emb_list):
    """
    Save a batch of (segment_hash, embedding) tuples using PostgreSQL upsert (ON CONFLICT DO UPDATE).
    Deduplicates by segment_hash to avoid ON CONFLICT cardinality errors.
    """
    if not hash_emb_list:
        return
    ensure_embedding_cache_table()
    # Deduplicate by segment_hash (keep last occurrence)
    deduped = {}
    for h, e in hash_emb_list:
        deduped[h] = e
    records = []
    for h, e in deduped.items():
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
                err_line = str(e).splitlines()[0]
                if len(err_line) > 120:
                    err_line = err_line[:117] + "..."
                console.print(
                    f"[red][BATCH EMBEDDING ERROR][/red] {type(e).__name__}: {err_line} (batch size: {len(records)})",
                    highlight=False,
                    end="\r"
                )
                return
    # Update in-memory and disk cache (always latest)
    with _batch_cache_lock:
        for h, e in deduped.items():
            _batch_cache[h] = np.array(e, dtype=np.float32)
            _disk_cache[h] = np.array(e, dtype=np.float32)

def load_embeddings_batch(segment_hashes):
    """
    Load a batch of embeddings for the given segment_hashes.
    Returns a dict: {segment_hash: embedding or None}
    """
    ensure_embedding_cache_table()
    result = {h: None for h in segment_hashes}
    cache_hits = 0
    disk_hits = 0
    db_hits = 0
    # First, try in-memory cache
    with _batch_cache_lock:
        for h in segment_hashes:
            if h in _batch_cache:
                result[h] = _batch_cache[h]
                cache_hits += 1
    # Next, try disk cache
    for h in segment_hashes:
        if result[h] is None and h in _disk_cache:
            result[h] = _disk_cache[h]
            disk_hits += 1
            with _batch_cache_lock:
                _batch_cache[h] = _disk_cache[h]
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
                            _disk_cache[obj.segment_hash] = emb
                        except Exception as e:
                            console.print(f"[red][EMBEDDING CACHE ERROR][/red] Failed to load embedding for hash {obj.segment_hash}: {e}", highlight=False)
        except SQLAlchemyError as e:
            console.print(f"[red][EMBEDDING CACHE DB ERROR][/red] {str(e)}", highlight=False)
    # Log missing hashes
    still_missing = [h for h in segment_hashes if result[h] is None]
    if still_missing:
        with open(MISSING_LOG_PATH, "a") as f:
            for h in still_missing:
                f.write(f"{h}\n")
    total = len(segment_hashes)
    console.log(
        f"[cyan][EMBEDDING CACHE] Batch load: {cache_hits} from mem, {disk_hits} from disk, {db_hits} from DB, {total - cache_hits - disk_hits - db_hits} missing.[/cyan]",
        highlight=False
    )
    return result

def fix_missing_embeddings():
    """
    Scan missing_embeddings.log, try to compute/fetch missing embeddings,
    and save them to the cache if possible.
    """
    if not os.path.exists(MISSING_LOG_PATH):
        return
    with open(MISSING_LOG_PATH, "r") as f:
        missing_hashes = [line.strip() for line in f if line.strip()]
    if not missing_hashes:
        return
    fixed = []
    total = len(missing_hashes)
    with logger.progress_bar("Fixing missing embeddings...", total=total) as update_progress:
        for idx, h in enumerate(missing_hashes, 1):
            emb = load_embedding(h)
            if emb is not None:
                fixed.append(h)
                update_progress(idx)
                continue
            emb = compute_embedding_for_hash(h)
            if emb is not None:
                save_embedding(h, emb)
                fixed.append(h)
            update_progress(idx)
    # Remove fixed hashes from log
    if fixed:
        remaining = set(missing_hashes) - set(fixed)
        with open(MISSING_LOG_PATH, "w") as f:
            for h in remaining:
                f.write(f"{h}\n")
        console.print(f"[green][EMBEDDING CACHE] Fixed {len(fixed)} missing embeddings automatically.[/green]")

# --- Ensure table and fix missing embeddings at startup ---
ensure_embedding_cache_table()
fix_missing_embeddings()