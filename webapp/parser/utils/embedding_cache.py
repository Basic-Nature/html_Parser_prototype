from __future__ import annotations

import atexit
import logging

# webapp/parser/utils/embedding_cache.py
# ---------------------------------------------------------------
# Embedding cache management for Smart Elections Parser Webapp
# ---------------------------------------------------------------
import os
import threading
from functools import lru_cache

import numpy as np
import orjson
from sqlalchemy import inspect, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm.exc import DetachedInstanceError

from ..config import DISK_CACHE_PATH, MISSING_LOG_PATH
from .db_utils import TEST_SQLITE_URL, engine, get_session
from .logger_singleton import console, logger
from .models import EmbeddingCache
from .shared_logger import SQLAlchemyToSharedLoggerHandler

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    import pickle
    JOBLIB_AVAILABLE = False

for name in [
    "sqlalchemy",
    "sqlalchemy.engine",
    "sqlalchemy.dialects",
    "sqlalchemy.pool"
]:
    logger_obj = logging.getLogger(name)
    logger_obj.addHandler(SQLAlchemyToSharedLoggerHandler(logger))

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

# --- DB readiness / disablement flags ---
EMBEDDING_CACHE_DISABLE_DB = os.environ.get("EMBEDDING_CACHE_DISABLE_DB", "").lower() in ("1", "true", "yes")
EMBEDDING_CACHE_DB_MODE = os.environ.get("EMBEDDING_CACHE_DB_MODE", "rw").lower()
if EMBEDDING_CACHE_DB_MODE not in ("rw", "ro", "off"):
    EMBEDDING_CACHE_DB_MODE = "rw"
try:
    EMBEDDING_CACHE_MAX_BATCH = max(1, int(os.environ.get("EMBEDDING_CACHE_MAX_BATCH", "500")))
except Exception:
    EMBEDDING_CACHE_MAX_BATCH = 500
_db_disabled_reason = None
if EMBEDDING_CACHE_DB_MODE == "off":
    _db_disabled_reason = "EMBEDDING_CACHE_DB_MODE=off"
elif EMBEDDING_CACHE_DISABLE_DB:
    _db_disabled_reason = "EMBEDDING_CACHE_DISABLE_DB"
elif TEST_SQLITE_URL:
    # Tests explicitly avoid external DB; treat as DB-disabled for embeddings
    _db_disabled_reason = "TEST_SQLITE_URL"
_db_readonly = EMBEDDING_CACHE_DB_MODE == "ro"
_db_ready = None
_db_warning_logged = False
_db_readonly_logged = False
EMBEDDING_CACHE_AUTO_WARMUP = os.environ.get("EMBEDDING_CACHE_AUTO_WARMUP", "false").lower() in ("1", "true", "yes")


def _log_cache_status():
    """Emit a concise startup summary without touching the DB."""
    mode = EMBEDDING_CACHE_DB_MODE
    disabled = _db_disabled_reason or "enabled"
    joblib_status = "joblib" if JOBLIB_AVAILABLE else "pickle"
    disk_exists = os.path.exists(DISK_CACHE_PATH)
    disk_label = f"disk_cache={'present' if disk_exists else 'missing'}"
    sqlite_flag = "TEST_SQLITE_URL" if TEST_SQLITE_URL else None
    helpers = "table_builder, table_core, dynamic_table_extractor, context_coordinator, health_router"
    console.print(
        f"[cyan][EMBEDDING CACHE] init: db_mode={mode} ({disabled}), writes={'ro' if _db_readonly else 'rw'}, {disk_label}, serializer={joblib_status}, env={sqlite_flag or 'postgres/default'}, helpers={helpers}[/cyan]",
        highlight=False,
    )


_log_cache_status()


def ensure_embedding_cache_table():
    global _db_ready, _db_warning_logged

    if _db_disabled_reason:
        if not _db_warning_logged:
            console.print(
                f"[yellow][EMBEDDING CACHE] DB disabled ({_db_disabled_reason}); using in-memory/disk cache only.[/yellow]",
                highlight=False,
            )
            _db_warning_logged = True
        _db_ready = False
        return False

    if _db_ready is False:
        return False

    try:
        inspector = inspect(engine)
        table_name = EmbeddingCache.__tablename__
        if not inspector.has_table(table_name):
            if _db_readonly:
                if not _db_warning_logged:
                    console.print(
                        f"[yellow][EMBEDDING CACHE] Table '{table_name}' missing but DB is read-only; skipping creation.[/yellow]",
                        highlight=False,
                    )
                    _db_warning_logged = True
                _db_ready = False
                return False
            console.print(f"[yellow][EMBEDDING CACHE] Table '{table_name}' does not exist. Creating...[/yellow]", highlight=False)
            EmbeddingCache.metadata.create_all(engine, tables=[EmbeddingCache.__table__])
            console.print(f"[green][EMBEDDING CACHE] Table '{table_name}' created.[/green]", highlight=False)
        _db_ready = True
        return True
    except Exception as e:
        _db_ready = False
        if not _db_warning_logged:
            console.print(
                f"[yellow][EMBEDDING CACHE] DB unavailable; using in-memory/disk cache only: {e}[/yellow]",
                highlight=False,
            )
            _db_warning_logged = True
        return False


def _db_write_allowed() -> bool:
    """Returns True if DB writes are permitted and table is ready."""
    global _db_readonly_logged
    if _db_disabled_reason:
        return False
    if _db_readonly:
        if not _db_readonly_logged:
            console.print(
                f"[yellow][EMBEDDING CACHE] DB is read-only (mode=ro); write operations will be skipped.[/yellow]",
                highlight=False,
            )
            _db_readonly_logged = True
        return False
    return ensure_embedding_cache_table()


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
    db_ok = _db_write_allowed()
    emb_bytes = np.array(embedding).astype(np.float32).tobytes()
    if db_ok:
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
    if not ensure_embedding_cache_table():
        return None
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
    with open(MISSING_LOG_PATH, "ab") as f:
        f.write(orjson.dumps({"segment_hash": segment_hash}) + b"\n")
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
                logger.info(msg)
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
    db_ok = _db_write_allowed()
    # Deduplicate by segment_hash (keep last occurrence)
    deduped = {}
    for h, e in hash_emb_list:
        deduped[h] = e
    records = []
    for h, e in deduped.items():
        arr = np.array(e, dtype=np.float32)
        emb_bytes = arr.tobytes()
        records.append({"segment_hash": h, "embedding": emb_bytes})
    if len(records) > EMBEDDING_CACHE_MAX_BATCH:
        console.print(
            f"[yellow][EMBEDDING CACHE] Truncating batch from {len(records)} to {EMBEDDING_CACHE_MAX_BATCH} for safety.[/yellow]",
            highlight=False,
            end="\r"
        )
        records = records[:EMBEDDING_CACHE_MAX_BATCH]
    if db_ok:
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
                    console.print(
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
    db_ok = ensure_embedding_cache_table()
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
    if missing and db_ok:
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
        with open(MISSING_LOG_PATH, "ab") as f:
            for h in still_missing:
                f.write(orjson.dumps({"segment_hash": h}) + b"\n")
    total = len(segment_hashes)
    console.log(
        f"[cyan][EMBEDDING CACHE] Batch load: {cache_hits} from mem, {disk_hits} from disk, {db_hits} from DB, {total - cache_hits - disk_hits - db_hits} missing.[/cyan]",
        highlight=False
    )
    return result

def fix_missing_embeddings():
    """
    Scan missing_embeddings_log.jsonl, try to compute/fetch missing embeddings,
    and save them to the cache if possible.
    """
    if not os.path.exists(MISSING_LOG_PATH):
        return
    with open(MISSING_LOG_PATH, "rb") as f:
        missing_hashes = []
        for line in f:
            try:
                obj = orjson.loads(line)
                missing_hashes.append(obj["segment_hash"])
            except Exception:
                continue
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
        with open(MISSING_LOG_PATH, "wb") as f:
            for h in remaining:
                f.write(orjson.dumps({"segment_hash": h}) + b"\n")
        console.print(f"[green][EMBEDDING CACHE] Fixed {len(fixed)} missing embeddings automatically.[/green]")

# --- Optional: best-effort warmup without failing startup (guarded) ---
if EMBEDDING_CACHE_AUTO_WARMUP:
    try:
        if ensure_embedding_cache_table():
            fix_missing_embeddings()
    except Exception:
        # Already logged; continue with in-memory/disk caches only
        pass