from __future__ import annotations

import concurrent.futures
import datetime

# webapp/parser/utils/html_scanner.py
# ---------------------------------------------------------------
# HTML scanning utilities for Smart Elections Parser Webapp
# ---------------------------------------------------------------
import hashlib
import os
import re
import tempfile
import threading
import time
import traceback
from collections import Counter
from difflib import get_close_matches
from typing import Any, Dict, List, Optional, Pattern, Set

import numpy as np
import orjson
from selectolax.parser import HTMLParser

from ..Context_Integration.context_write_policy import (
    ContextWriteKind,
)

from ..config import (
    CACHE_DIR,
    CONTEXT_CACHE_PATH,
    CONTEXT_LIBRARY_PATH,
    ENABLE_SEGMENT_LABEL_PROMPT,
    LOG_DIR,
    SEGMENT_ML_LABEL_THRESHOLD,
    SEGMENT_ML_LABEL_THRESHOLD_STRICT,
)
from ..Context_Integration.Context_Library.constants import (
    ALLOWED_LABELS,
    ALWAYS_IGNORE_CLASSES,
    ALWAYS_IGNORE_IDS,
    ALWAYS_IGNORE_TAGS,
    BALLOT_TYPES,
    BALLOT_TYPES_SORT_ORDER,
    BUTTON_CLASSES,
    BUTTON_TAGS,
    CANDIDATE_KEYWORDS,
    CANONICAL_SEGMENT_LABELS,
    CONTEST_KEYWORDS,
    CONTEST_PANEL_TAGS,
    CUSTOM_ATTR_PATTERNS,
    DISTRICT_REGEX,
    ELECTION_TYPES,
    EXTRA_HEADING_TAGS,
    HEADING_CLASSES,
    HEADING_TAGS,
    HTML_TAGS,
    ICON_CLASSES,
    ICON_TAGS,
    KNOWN_COUNTY_TO_PRECINCTS_MAP,
    KNOWN_STATE_TO_COUNTY_MAP,
    LOCATION_ABBREVIATIONS,
    LOCATION_KEYWORDS,
    MISC_FOOTER_KEYWORDS,
    NOISY_LABEL_PATTERNS,
    OFFICE_KEYWORDS,
    PANEL_CLASSES,
    PANEL_TAGS,
    PARTY_KEYWORDS,
    PERCENT_KEYWORDS,
    PRECINCT_HEADER_PATTERNS,
    ROOT_CONTAINER_TAGS,
    SELECTORS,
    STATE_ABBR,
    STATE_TAGS,
    STRUCTURAL_TAGS,
    TABLE_TAGS,
    TIMESTAMP_ATTRS,
    TIMESTAMP_CLASSES,
    TIMESTAMP_ID_PATTERNS,
    TOTAL_KEYWORDS,
    UPDATE_PANEL_KEYWORDS,
    VIEW_BY_PHRASES,
)
from ..Context_Integration.librarian import (
    get_canonical_segment_label,
    load_context_library,
)
from .embedding_cache import (
    get_embedding_from_memory,
    load_embeddings_batch,
    save_embedding,
    save_embeddings_batch,
)
from .logger_singleton import console, logger, prompt
from .model_registry import ModelRegistry
from .shared_logic import (
    clean_cache_inplace,
    convert_ndarrays,
    keyword_in_text,
    log_rejection_reason,
    normalize_html_for_hash,
    safe_add,
    safe_append,
    safe_encode,
    safe_filename,
    safe_get,
    safe_get_first,
    safe_items,
    safe_keys,
    safe_lower,
    safe_model_encode,
    safe_setdefault,
    safe_startswith,
    safe_strip,
    safe_update,
    sync_type_and_election_types,
)

# --- Caching and threading ---
_LABEL_CACHE_FILENAME = "segment_label_cache.json"
_LABEL_CACHE_LOCK = threading.Lock()
_LABEL_CACHE = None
_context_cache = None
_pattern_kb_cache = None
_TEMP_FILES_TRACKER = set()
embedding_cache_hits = set()
embedding_cache_misses = set()

def robust_orjson_loads(val) -> Any:
    """Safely decodes JSON using orjson, handling both bytes and str inputs."""
    if val is None:
        return None
    if not isinstance(val, (bytes, str)):
        raise TypeError(f"Expected bytes or str, got {type(val)}")
    if not val:
        return None
    if isinstance(val, str) and not val.isascii():
        # If the string is not ASCII, encode it to bytes first
        val = val.encode("utf-8")
    if not isinstance(val, bytes):
        raise TypeError(f"Expected bytes, got {type(val)}") 
    if isinstance(val, bytes):
        return orjson.loads(val)
    elif isinstance(val, str):
        return orjson.loads(safe_encode(val, "utf-8"))
    else:
        raise TypeError(f"Cannot decode type {type(val)} with orjson")

def _get_label_cache_path() -> str:
    """Returns the path to the label cache file, ensuring it is safe and does not exceed OS limits.
    Cleans up previous temp files if switching to a new temp path."""
    global _TEMP_FILES_TRACKER
    if not _LABEL_CACHE_FILENAME:
        raise ValueError("Label cache filename cannot be empty")
    if not isinstance(_LABEL_CACHE_FILENAME, str):
        raise TypeError("Label cache filename must be a string")
    path = safe_cache_path(_LABEL_CACHE_FILENAME)
    abs_path = os.path.abspath(path)
    if os.name == "nt" and len(abs_path) >= 260:
        short_path = os.path.join(tempfile.gettempdir(), _LABEL_CACHE_FILENAME)
        msg = f"Path too long for Windows, using temp path: {short_path}"
        if logger.mode == "cli":
            console.print(f"[CACHE] {msg}")
        else:
            payload = {
                "level": "WARNING",
                "type": "cache",
                "message": msg
            }
            logger.warning(payload)
        # Clean up previous temp files
        for temp_file in list(_TEMP_FILES_TRACKER):
            if temp_file != short_path and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    msg = f"[CACHE] Removed old temp file: {temp_file}"
                    if logger.mode == "cli":
                        console.print(msg)
                    else:
                        payload = {
                            "level": "INFO",
                            "type": "cache",
                            "message": msg
                        }
                        logger.info(payload)
                except Exception as e:
                    msg = f"[CACHE] Failed to remove temp file {temp_file}: {e}"
                    if logger.mode == "cli":
                        console.print(msg)
                    else:
                        payload = {
                            "level": "WARNING",
                            "type": "cache",
                            "message": msg
                        }
                        logger.warning(payload)
                _TEMP_FILES_TRACKER.discard(temp_file)
        _TEMP_FILES_TRACKER.add(short_path)
        return short_path
    return path

def _load_label_cache() -> Dict[str, Any]:
    """Loads the label cache from disk, or initializes it if it doesn't exist."""
    global _LABEL_CACHE
    if _LABEL_CACHE is not None:
        return _LABEL_CACHE
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR, exist_ok=True)
    if not os.path.isdir(CACHE_DIR):
        raise ValueError(f"Cache directory {CACHE_DIR} is not a directory")
    path = _get_label_cache_path()
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                _LABEL_CACHE = robust_orjson_loads(f.read())
        except Exception:
            _LABEL_CACHE = {}
    else:
        _LABEL_CACHE = {}
    return _LABEL_CACHE

def _save_label_cache() -> None:
    """Saves the label cache to disk."""
    global _LABEL_CACHE
    if _LABEL_CACHE is None:
        _LABEL_CACHE = _load_label_cache()
    if not isinstance(_LABEL_CACHE, Dict):
        raise ValueError("Label cache must be a dictionary")
    path = _get_label_cache_path()
    with open(path, "wb") as f:
        f.write(orjson.dumps(_LABEL_CACHE, option=orjson.OPT_INDENT_2))

def cache_segment_label(seg_hash, label) -> None:
    """Caches a segment label by its hash."""
    global _LABEL_CACHE
    if _LABEL_CACHE is None:
        _LABEL_CACHE = _load_label_cache()
    with _LABEL_CACHE_LOCK:
        _LABEL_CACHE[seg_hash] = {"label": label, "timestamp": int(time.time())}
        _save_label_cache()

def get_cached_segment_label(seg_hash) -> Optional[List[str]]:
    """Retrieves a cached segment label by its hash."""
    global _LABEL_CACHE
    if _LABEL_CACHE is None:
        _LABEL_CACHE = _load_label_cache()
    if not seg_hash:
        raise ValueError("Segment hash cannot be empty")
    if not isinstance(seg_hash, str):
        raise TypeError("Segment hash must be a string")
    if not re.match(r"^[a-f0-9]{64}$", seg_hash):
        raise ValueError("Segment hash must be a valid SHA-256 hash")
    if seg_hash in _LABEL_CACHE:
        entry = _LABEL_CACHE[seg_hash]
        if isinstance(entry, dict) and "label" in entry:
            return safe_get(entry, "label", [])
    # If not found in cache, check the file
    path = _get_label_cache_path()
    if not os.path.exists(path):
        return None
    if not os.path.isfile(path):
        raise ValueError(f"Cache path {path} is not a file")
    with _LABEL_CACHE_LOCK:
        cache = _load_label_cache()
        entry = cache.get(seg_hash, {})
        if entry:
            return safe_get(entry, "label", [])
        return None

def safe_cache_path(filename: str) -> str:
    """Generates a safe cache file path, ensuring it does not escape the cache directory.
    Cleans up previous temp files if switching to a new temp path."""
    global _TEMP_FILES_TRACKER
    if not filename:
        raise ValueError("Filename cannot be empty")
    if not isinstance(filename, str):
        raise TypeError("Filename must be a string")
    if not re.match(r"^[\w\-. ]+$", filename):
        raise ValueError("Filename contains unsafe characters")
    filename = safe_filename(filename)
    cache_folder = CACHE_DIR
    full_path = os.path.join(cache_folder, filename)
    abs_path = os.path.abspath(full_path)
    if os.name == "nt" and len(abs_path) >= 240:
        temp_path = os.path.join(tempfile.gettempdir(), filename)
        msg = f"Path too long for Windows, using temp path: {temp_path}"
        if logger.mode == "cli":
            console.print(f"[CACHE] {msg}")
        else:
            payload = {
                "level": "WARNING",
                "type": "cache",
                "message": msg
            }
            logger.warning(payload)
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        # Clean up previous temp files
        for temp_file in list(_TEMP_FILES_TRACKER):
            if temp_file != temp_path and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    msg = f"[CACHE] Removed old temp file: {temp_file}"
                    if logger.mode == "cli":
                        console.print(msg)
                    else:
                        payload = {
                            "level": "INFO",
                            "type": "cache",
                            "message": msg
                        }
                        logger.info(payload)
                except Exception as e:
                    msg = f"[CACHE] Failed to remove temp file {temp_file}: {e}"
                    if logger.mode == "cli":
                        console.print(msg)
                    else:
                        payload = {
                            "level": "WARNING",
                            "type": "cache",
                            "message": msg
                        }
                        logger.warning(payload)
                _TEMP_FILES_TRACKER.discard(temp_file)
        _TEMP_FILES_TRACKER.add(temp_path)
        return temp_path
    os.makedirs(cache_folder, exist_ok=True)
    if not abs_path.startswith(os.path.abspath(cache_folder)):
        raise ValueError("Unsafe cache path detected!")
    return full_path

def safe_log_path(filename: str, default_ext: str = ".jsonl") -> str:
    """
    Generates a safe log file path for JSONL logs, ensuring it does not escape the log directory.
    Cleans up previous temp files if switching to a new temp path.
    """
    global _TEMP_FILES_TRACKER
    if not filename:
        raise ValueError("Filename cannot be empty")
    if not isinstance(filename, str):
        raise TypeError("Filename must be a string")
    if not re.match(r"^[\w\-. ]+$", filename):
        raise ValueError("Filename contains unsafe characters")
    filename = safe_filename(filename)
    if not filename.endswith(default_ext):
        filename = re.sub(r"\.[^.]+$", "", filename) + default_ext
    log_folder = LOG_DIR
    full_path = os.path.join(log_folder, filename)
    abs_path = os.path.abspath(full_path)
    if os.name == "nt" and len(abs_path) >= 240:
        temp_path = os.path.join(tempfile.gettempdir(), filename)
        msg = f"Path too long for Windows, using temp path: {temp_path}"
        if logger.mode == "cli":
            console.print(f"[LOG] {msg}")
        else:
            payload = {
                "level": "WARNING",
                "type": "log",
                "message": msg
            }
            logger.warning(payload)
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        # Clean up previous temp files
        for temp_file in list(_TEMP_FILES_TRACKER):
            if temp_file != temp_path and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    msg = f"[LOG] Removed old temp file: {temp_file}"
                    if logger.mode == "cli":
                        console.print(msg)
                    else:
                        payload = {
                            "level": "INFO",
                            "type": "log",
                            "message": msg
                        }
                        logger.info(payload)
                except Exception as e:
                    msg = f"[LOG] Failed to remove temp file {temp_file}: {e}"
                    if logger.mode == "cli":
                        console.print(msg)
                    else:
                        payload = {
                            "level": "WARNING",
                            "type": "log",
                            "message": msg
                        }
                        logger.warning(payload)
                _TEMP_FILES_TRACKER.discard(temp_file)
        _TEMP_FILES_TRACKER.add(temp_path)
        return temp_path
    os.makedirs(log_folder, exist_ok=True)
    if not abs_path.startswith(os.path.abspath(log_folder)):
        raise ValueError("Unsafe log path detected!")
    return full_path

def is_trivial_segment(seg, diagnostics=False) -> bool:
    """
    Determines if a segment is trivial (should be ignored for semantic processing).
    Checks for empty, whitespace, HTML entities, tags, icons, comments, scripts, styles,
    numeric-only, special-char-only, and other non-informative content.
    Optionally logs diagnostics for audit/debug.
    """
    html = safe_get(seg, "html", "")
    tag = safe_lower(safe_get(seg, "tag", ""))
    classes = [safe_lower(c) for c in safe_get(seg, "classes", [])]
    attrs = safe_get(seg, "attrs", {}) or {}

    def log_trivial(msg):
        if diagnostics:
            if logger.mode == "cli":
                console.print(msg)
            else:
                payload = {
                    "level": "INFO",
                    "type": "trivial_segment",
                    "message": msg
                }
                logger.info(payload)

    # Empty or whitespace-only HTML
    if not html or not safe_strip(html):
        log_trivial(f"[TRIVIAL] Empty or whitespace HTML for tag={tag}")
        return True

    # HTML entities or just tags
    html_stripped = safe_strip(html)
    if html_stripped in {"&nbsp;", "&#160;"}:
        log_trivial(f"[TRIVIAL] HTML entity only for tag={tag}")
        return True
    if re.fullmatch(r"<[^>]+>", html_stripped):
        log_trivial(f"[TRIVIAL] Just a tag for tag={tag}")
        return True

    # Known trivial tags
    if tag in {"br", "hr", "wbr"} and not html_stripped:
        log_trivial(f"[TRIVIAL] Known trivial tag: {tag}")
        return True

    # Icon-only spans
    if tag == "span" and classes and all("icon" in cls for cls in classes) and not safe_strip(re.sub(r"<[^>]+>", "", html)):
        log_trivial(f"[TRIVIAL] Icon-only span for tag={tag}, classes={classes}")
        return True

    # Comments, scripts, styles
    if re.search(r"<!--.*?-->", html, re.DOTALL):
        log_trivial(f"[TRIVIAL] HTML comment detected for tag={tag}")
        return True
    if re.search(r"<script.*?>.*?</script>", html, re.DOTALL) or re.search(r"<style.*?>.*?</style>", html, re.DOTALL):
        log_trivial(f"[TRIVIAL] Script or style block detected for tag={tag}")
        return True

    # Numeric-only or special-char-only segments
    text_content = re.sub(r"<[^>]+>", "", html_stripped)
    if safe_strip(text_content).isdigit():
        log_trivial(f"[TRIVIAL] Numeric-only content for tag={tag}")
        return True
    if bool(re.fullmatch(r'[\W_]+', safe_strip(text_content))):
        log_trivial(f"[TRIVIAL] Special-char-only content for tag={tag}")
        return True

    # Trivial by attribute (e.g., aria-hidden, display:none)
    if safe_lower(attrs.get("aria-hidden", "")) == "true" or "display:none" in safe_lower(attrs.get("style", "")):
        log_trivial(f"[TRIVIAL] aria-hidden or display:none for tag={tag}")
        return True

    # Defensive: very short text (1 char or less)
    if len(safe_strip(text_content)) <= 1:
        log_trivial(f"[TRIVIAL] Very short content for tag={tag}")
        return True

    return False

def segment_identity_hash(segment) -> str:
    tag = safe_lower(safe_get(segment, "tag", ""))
    classes = " ".join(sorted([safe_lower(c) for c in safe_get(segment, "classes", [])]))
    attrs = safe_get(segment, "attrs", {}) or {}
    attrs_filtered = {
        k: v for k, v in attrs.items()
        if not (
            safe_startswith(k, '_ngcontent-') or
            safe_startswith(k, '_nghost-') or
            safe_startswith(k, 'ng-') or
            safe_startswith(k, 'data-') or
            k in {'style', 'id', 'class', 'tabindex', 'aria-checked'}
        )
    }
    html = safe_lower(safe_get(segment, "html", ""))
    html_norm = re.sub(
        r'\s+', ' ',
        re.sub(r'\s*([=;:,])\s*', r'\1', re.sub(r'\s+', ' ', safe_strip(html)))
    )[:256]
    try:
        attrs_json = orjson.dumps(attrs_filtered, option=orjson.OPT_SORT_KEYS).decode()
    except Exception:
        attrs_json = "{}"
    base = tag + "|" + classes + "|" + attrs_json + "|" + html_norm
    return hashlib.sha256(safe_encode(base)).hexdigest()

def embedding_cache_hash(segment, model_id) -> str:
    tag = safe_lower(safe_get(segment, "tag", ""))
    attrs = safe_get(segment, "attrs", {})
    attrs_filtered = {
        k: v for k, v in safe_items(attrs or {})
        if not (
            safe_startswith(k, '_ngcontent-') or
            safe_startswith(k, '_nghost-') or
            safe_startswith(k, 'ng-') or
            safe_startswith(k, 'data-') or
            k in {'style', 'id', 'class', 'tabindex', 'aria-checked'}
        )
    }
    html = safe_get(segment, "html", "")
    attrs_sorted = {k: attrs_filtered[k] for k in sorted(attrs_filtered)}
    html_norm = normalize_html_for_hash(html)
    base = tag + orjson.dumps(attrs_sorted, option=orjson.OPT_SORT_KEYS).decode() + html_norm + str(model_id)
    return hashlib.sha256(safe_encode(base, "utf-8")).hexdigest()

def get_segment_embedding(
    model,
    segment,
    cache=None,
    cache_hits=None,
    cache_misses=None,
    diagnostics=False,
    logger=logger
) -> Optional[np.ndarray]:
    """
    Robustly computes or retrieves the embedding for a segment.
    - Uses cache if available.
    - Logs cache hits/misses and errors.
    - Handles trivial/empty segments.
    - Optionally logs timing and diagnostics.
    """
    model_id = getattr(model, 'name_or_path', str(model))
    identity = embedding_cache_hash(segment, model_id)
    if cache is not None:
        clean_cache_inplace(cache)
    if is_trivial_segment(segment):
        if diagnostics:
            msg = f"[EMBED] Skipping trivial segment: {identity}"
            if logger.mode == "cli":
                console.print(msg)
            else:
                payload = {
                    "level": "INFO",
                    "type": "embedding",
                    "message": msg
                }
                logger.info(payload)
        return None
    emb = get_embedding_from_memory(identity)
    if emb is not None:
        safe_add(cache_hits, str(identity))
        if diagnostics:
            msg = f"[EMBED] Cache hit for {identity}"
            if logger.mode == "cli":
                console.print(msg)
            else:
                payload = {
                    "level": "INFO",
                    "type": "embedding",
                    "message": msg
                }
                logger.info(payload)
        return emb
    tag = safe_get(segment, "tag", "")
    attrs = " ".join([f"{k}={v}" for k, v in safe_items(safe_get(segment, "attrs", {}))])
    html = safe_get(segment, "html", "")
    try:
        tree = HTMLParser(html)
        text = tree.body.text(separator=" ", strip=True) if tree.body else tree.text(separator=" ", strip=True)
    except Exception:
        text = ""
    full_text = f"{tag} {attrs} {text}"
    if not safe_strip(full_text):
        if diagnostics:
            msg = f"[EMBED] Empty text for segment: {identity}"
            if logger.mode == "cli":
                console.print(msg)
            else:
                payload = {
                    "level": "WARNING",
                    "type": "embedding",
                    "message": msg
                }
                logger.warning(payload)
        return None
    try:
        t0 = time.time()
        emb = safe_model_encode(model, full_text, convert_to_numpy=True, show_progress_bar=False)
        save_embedding(identity, emb)
        safe_add(cache_misses, str(identity))
        if diagnostics:
            msg = f"[EMBED] Computed embedding for {identity} in {time.time()-t0:.3f}s"
            if logger.mode == "cli":
                console.print(msg)
            else:
                payload = {
                    "level": "INFO",
                    "type": "embedding",
                    "message": msg
                }
                logger.info(payload)
        return emb
    except Exception as e:
        segment["embedding_error"] = str(e)
        if diagnostics:
            msg = f"[EMBED] Error for {identity}: {e}"
            if logger.mode == "cli":
                console.print(msg)
            else:
                payload = {
                    "level": "ERROR",
                    "type": "embedding",
                    "message": msg
                }
                logger.error(payload)
        return None

def batch_get_segment_embeddings(
    model,
    segments,
    diagnostics=False,
    logger=logger
) -> List[Optional[np.ndarray]]:
    """
    Robust batch embedding retrieval/computation for segments.
    - Uses cache where possible.
    - Parallelizes computation for large batches.
    - Logs progress and errors.
    """
    model_id = getattr(model, 'name_or_path', str(model))
    identities = [embedding_cache_hash(seg, model_id) if not is_trivial_segment(seg) else None for seg in segments]
    cached = [get_embedding_from_memory(identity) if identity else None for identity in identities]
    to_compute = [i for i, emb in enumerate(cached) if emb is None and identities[i] is not None]
    if isinstance(segments, list):
        segments[:] = [s for s in segments if isinstance(s, dict)]
    texts = []
    idx_map = []
    for idx in to_compute:
        seg = segments[idx]
        tag = safe_get(seg, "tag", "")
        attrs = " ".join([f"{k}={v}" for k, v in safe_items(safe_get(seg, "attrs", {}))])
        html = safe_get(seg, "html", "")
        try:
            tree = HTMLParser(html)
            text = tree.body.text(separator=" ", strip=True) if tree.body else tree.text(separator=" ", strip=True)
        except Exception:
            text = ""
        if not safe_strip(text):
            continue
        texts.append(f"{tag} {attrs} {text}")
        idx_map.append(idx)
    if texts:
        try:
            t0 = time.time()
            if len(texts) > 128:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    chunks = [texts[i:i+32] for i in range(0, len(texts), 32)]
                    results = list(executor.map(lambda chunk: safe_model_encode(chunk, convert_to_numpy=True, show_progress_bar=False, batch_size=16), chunks))
                new_embs = np.concatenate(results)
            else:
                new_embs = safe_model_encode(texts, convert_to_numpy=True, show_progress_bar=False, batch_size=16)
            for i, idx in enumerate(idx_map):
                save_embedding(identities[idx], new_embs[i])
                cached[idx] = new_embs[i]
            if diagnostics:
                msg = f"[EMBED] Batch computed {len(texts)} embeddings in {time.time()-t0:.2f}s"
                if logger.mode == "cli":
                    console.print(msg)
                else:
                    payload = {
                        "level": "INFO",
                        "type": "embedding",
                        "message": msg
                    }
                    logger.info(payload)
        except Exception as e:
            if diagnostics:
                msg = f"[EMBED] Batch embedding error: {e}"
                if logger.mode == "cli":
                    console.print(msg)
                else:
                    payload = {
                        "level": "ERROR",
                        "type": "embedding",
                        "message": msg
                    }
                    logger.error(payload)
    return [emb if identity else None for emb, identity in zip(cached, identities)]

def deduplicate_pattern_kb(pattern_kb) -> List[Dict[str, Any]]:
    """Deduplicate pattern KB entries by segment_hash, keeping the latest timestamp."""
    dedup = {}
    for entry in pattern_kb:
        seg_hash = safe_get(entry, "segment_hash", None)
        ts = safe_get(entry, "timestamp", 0)
        if seg_hash not in dedup or ts > safe_get(dedup.get(seg_hash, {}), "timestamp", 0):
            dedup[seg_hash] = entry
    return list(dedup.values())

def prune_embedding_cache(valid_hashes) -> None:
    """Remove embeddings not in valid_hashes from the cache directory."""
    cache_dir = CACHE_DIR
    for fname in os.listdir(cache_dir):
        if fname.endswith(".npy"):
            h = fname.replace(".npy", "")
            if h not in valid_hashes:
                try:
                    os.remove(os.path.join(cache_dir, fname))
                except Exception:
                    pass

def submit_segment_correction(segment_hash, new_label, context_library=None) -> None:
    """Allow downstream modules to submit corrections for a segment label."""
    cache_segment_label(segment_hash, new_label)
    if context_library is not None:
        for seg in safe_get(context_library, "cached_segments", []):
            if safe_get(seg, "segment_hash", None) == segment_hash:
                seg["ml_label"] = new_label
                break

def auto_label_segment(
    segment,
    context_library=None,
    context_cache=None,
    pattern_kb=None,
    model=None,
    ml_threshold=SEGMENT_ML_LABEL_THRESHOLD,
    coordinator=None,
) -> Optional[tuple]:
    from ..Context_Integration.context_coordinator import ContextCoordinator
    coordinator = ContextCoordinator()
    seg_hash = segment_identity_hash(segment)
    # 1. Persistent label cache
    cached_label = get_cached_segment_label(seg_hash)
    if cached_label:
        return cached_label
    # 2. Context cache
    if context_cache and seg_hash in context_cache:
        cache_entry = safe_get(context_cache, seg_hash, {})
        label = safe_get(cache_entry, "ml_label", None)
        if label:
            return label
    # 3. Context library
    if context_library and "cached_segments" in context_library:
        for seg in safe_get(context_library, "cached_segments", []):
            if safe_get(seg, "segment_hash", None) == seg_hash and safe_get(seg, "ml_label", None):
                return safe_get(seg, "ml_label", None)
    # 4. Pattern KB
    if pattern_kb:
        for entry in pattern_kb:
            if safe_get(entry, "segment_hash", None) == seg_hash and safe_get(entry, "label", None):
                return safe_get(entry, "label", None)
    # 5. Coordinator as oracle
    if coordinator and hasattr(coordinator, "auto_label_segment"):
        try:
            label = coordinator.auto_label_segment(segment)
            if label:
                return label, "coordinator"
        except Exception:
            pass
    # 6. Fallback: ML similarity
    if model and pattern_kb:
        try:
            emb = get_segment_embedding(model, segment, diagnostics=True, logger_instance=logger)
            if emb is not None:
                best_label = "unknown"
                best_conf = 0.0
                best_entry = None
                for entry in pattern_kb:
                    kb_emb = np.array(safe_get(entry, "embedding", []))
                    if kb_emb.shape != emb.shape or kb_emb.size == 0:
                        continue
                    sim = float(np.dot(emb, kb_emb) / (np.linalg.norm(emb) * np.linalg.norm(kb_emb) + 1e-8))
                    if sim > best_conf:
                        best_conf = sim
                        best_label = safe_get(entry, "label", "unknown")
                        best_entry = entry
                if best_conf >= ml_threshold and best_label != "unknown":
                    logger.info(f"[ML SIMILARITY] Segment matched with label '{best_label}' (sim={best_conf:.3f})")
                    segment["ml_similarity_confidence"] = best_conf
                    segment["ml_similarity_label"] = best_label
                    segment["ml_similarity_entry"] = best_entry
                    return best_label, "ml"
                elif best_conf > 0.0 and best_label != "unknown":
                    # Log that this segment did not meet ML threshold (constructive criticism)
                    log_rejection_reason(
                        decision_context="segment_labeling",
                        confidence_score=best_conf,
                        rejection_reason="below ML threshold despite match",
                        candidate_info={"segment_hash": safe_get(segment, "segment_hash", None), "candidate_label": best_label},
                        threshold_name="SEGMENT_ML_LABEL_THRESHOLD",
                        threshold_value=ml_threshold,
                        function_name="auto_label_segment",
                    )
            else:
                logger.warning(f"[ML SIMILARITY] No embedding computed for segment: {safe_get(segment, 'segment_hash', None)}")
        except Exception as e:
            segment["embedding_error"] = str(e)
            logger.error(f"[ML SIMILARITY] Error during embedding similarity: {e}")
    # 6.5. Fallback: ML similarity with context library
    if model and context_library and "cached_segments" in context_library:
        try:
            emb = get_segment_embedding(model, segment, diagnostics=True, logger_instance=logger)
            if emb is not None:
                best_label = "unknown"
                best_conf = 0.0
                for seg in safe_get(context_library, "cached_segments", []):
                    kb_emb = np.array(safe_get(seg, "embedding", []))
                    if kb_emb.shape != emb.shape or kb_emb.size == 0:
                        continue
                    sim = float(np.dot(emb, kb_emb) / (np.linalg.norm(emb) * np.linalg.norm(kb_emb) + 1e-8))
                    if sim > best_conf:
                        best_conf = sim
                        best_label = safe_get(seg, "ml_label", "unknown")
                if best_conf >= ml_threshold and best_label != "unknown":
                    logger.info(f"[ML SIMILARITY] Segment matched with label '{best_label}' (sim={best_conf:.3f})")
                    return best_label, "ml_context_lib"
                elif best_conf > 0.0 and best_label != "unknown":
                    # Log that this segment did not meet ML threshold (constructive criticism)
                    log_rejection_reason(
                        decision_context="segment_labeling",
                        confidence_score=best_conf,
                        rejection_reason="below ML threshold with context library",
                        candidate_info={"segment_hash": safe_get(segment, "segment_hash", None), "candidate_label": best_label},
                        threshold_name="SEGMENT_ML_LABEL_THRESHOLD",
                        threshold_value=ml_threshold,
                        function_name="auto_label_segment",
                    )
            else:
                logger.warning(f"[ML SIMILARITY] No embedding computed for segment: {safe_get(segment, 'segment_hash', None)}")
        except Exception as e:
            segment["embedding_error"] = str(e)
            logger.error(f"[ML SIMILARITY] Error during embedding similarity with context library: {e}")
    # 7. Heuristic fallback
    tag = safe_lower(safe_get(segment, "tag", ""))
    classes = [safe_lower(c) for c in safe_get(segment, "classes", [])]
    attrs = safe_get(segment, "attrs", {})
    attrs_copy = attrs.copy() if isinstance(attrs, dict) else {}
    safe_attr_keys = [k for k, _ in safe_items(attrs_copy)]
    html = safe_lower(safe_get(segment, "html", ""))
    id_ = safe_lower(safe_get(segment, "id", ""))
    text = safe_lower(safe_strip(safe_get(segment, "text", ""))) if safe_get(segment, "text", None) else safe_lower(_extract_clean_text(html))
    # --- Use librarian keywords for robust labeling ---
    if keyword_in_text(text, CONTEST_KEYWORDS) or keyword_in_text(html, CONTEST_KEYWORDS):
        return "contest"
    if keyword_in_text(text, CANDIDATE_KEYWORDS) or keyword_in_text(html, CANDIDATE_KEYWORDS):
        return "candidate_panel"
    if keyword_in_text(text, PARTY_KEYWORDS) or keyword_in_text(html, PARTY_KEYWORDS):
        return "party_label"
    if keyword_in_text(text, LOCATION_KEYWORDS) or keyword_in_text(html, LOCATION_KEYWORDS):
        return "location_panel"
    if keyword_in_text(text, BALLOT_TYPES) or keyword_in_text(html, BALLOT_TYPES):
        return "ballot_types"
    if tag == "table" or keyword_in_text(text, TOTAL_KEYWORDS | PERCENT_KEYWORDS | MISC_FOOTER_KEYWORDS):
        return "results_table"
    if tag in HEADING_TAGS or HEADING_CLASSES & set(classes):
        return "heading"
    if tag in PANEL_TAGS or PANEL_CLASSES & set(classes):
        return "panel"
    if (
        tag in {"span", "time", "div", "p", "small", "label"}
        and (
            any(cls in TIMESTAMP_CLASSES for cls in classes)
            or any(re.search(pat, id_) for pat in TIMESTAMP_ID_PATTERNS if id_)
            or any(attr in attrs for attr in TIMESTAMP_ATTRS)
            or any(re.search(pat, " ".join(safe_attr_keys)) for pat in TIMESTAMP_ID_PATTERNS)
            or re.search(r"\bago\b|\bupdated\b|\blast\b|\bposted\b|\bas of\b|\breported\b", html)
            or re.search(r"\b\d{1,2}:\d{2}\s*(am|pm)?\b", html)
            or re.search(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", html)
            or re.search(r"\b\d{4}-\d{2}-\d{2}\b", html)
        )
    ):
        return "results_timestamp"
    if tag in ROOT_CONTAINER_TAGS:
        return "ignore"
    if tag == "div" and ("container" in classes or "main" in classes) and not html.strip():
        return "ignore"
    if tag in ALWAYS_IGNORE_TAGS:
        return "ignore"
    if set(classes) & ALWAYS_IGNORE_CLASSES:
        return "ignore"
    if id_ in ALWAYS_IGNORE_IDS:
        return "ignore"
    if tag in ICON_TAGS and (ICON_CLASSES & set(classes)):
        if tag != "span" or (set(classes) <= ICON_CLASSES and not html.strip()):
            return "ignore"
        if tag == "span" and set(classes) <= ICON_CLASSES and not re.sub(r"<[^>]+>", "", html).strip():
            return "ignore"
    if tag in {"i", "span"} and not html.strip():
        return "ignore"
    if tag == "a" and "href" in attrs:
        href = safe_lower(str(attrs["href"]))
        if any(href.endswith(ext) for ext in [".csv", ".json", ".pdf", ".xlsx", ".zip", ".xls", ".doc", ".docx"]):
            return "download_link"
    if safe_get(segment, "is_button", []) or BUTTON_CLASSES & set(classes) or "toggle" in id_:
        return "ballot_toggle"
    if tag in HEADING_TAGS or HEADING_CLASSES & set(classes):
        return "heading"
    if tag in PANEL_TAGS or PANEL_CLASSES & set(classes):
        return "panel"
    if tag == "table":
        return "results_table"
    if context_library:
        if 'party' in context_library:
            known_parties = [safe_lower(p) for p in safe_get(context_library, "party", [])]
            if text in known_parties or html in known_parties:
                return "party_label"
            close = get_close_matches(text, known_parties, n=1, cutoff=0.85)
            if close:
                return "party_label"
        if 'vote_methods' in context_library:
            known_vote_methods = [safe_lower(v) for v in safe_get(context_library, "vote_methods", [])]
            if text in known_vote_methods or html in known_vote_methods:
                return "vote_method"
            close = get_close_matches(text, known_vote_methods, n=1, cutoff=0.85)
            if close:
                return "vote_method"
        if 'contests' in context_library:
            known_contests = [safe_lower(safe_strip(safe_get(c, "title", ""))) for c in safe_get(context_library, "contests", []) if "title" in c]
            if text in known_contests or html in known_contests:
                return "contest"
    if any(bt in html for bt in BALLOT_TYPES):
        return "ballot_types"
    if safe_get(segment, "is_clickable", []):
        return "clickable"
    if (
        tag in {"span", "time", "div", "p", "small", "label"}
        and (
            any(cls in TIMESTAMP_CLASSES for cls in classes)
            or any(re.search(pat, id_) for pat in TIMESTAMP_ID_PATTERNS if id_)
            or any(attr in attrs for attr in TIMESTAMP_ATTRS)
            or any(re.search(pat, " ".join(safe_attr_keys)) for pat in TIMESTAMP_ID_PATTERNS)
            or re.search(r"\bago\b|\bupdated\b|\blast\b|\bposted\b|\bas of\b|\breported\b", html)
            or re.search(r"\b\d{1,2}:\d{2}\s*(am|pm)?\b", html)
            or re.search(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", html)
            or re.search(r"\b\d{4}-\d{2}-\d{2}\b", html)
        )
    ):
        return "results_timestamp"
    if tag in STRUCTURAL_TAGS and not html.strip():
        return "ignore"
    if not html.strip() or html.strip() in {"&nbsp;", "&#160;"}:
        return "ignore"
    if tag == "span" and len(classes) > 0 and all(cls in ICON_CLASSES for cls in classes):
        return "ignore"
    canonical = get_canonical_segment_label(text)
    if canonical:
        return canonical
    return "unknown", "heuristic"

def _extract_clean_text(html) -> str:
    """Extracts clean text from HTML using selectolax, fallback to raw HTML if parsing fails.
    Returns '' if the result is only whitespace or just tags."""
    try:
        tree = HTMLParser(html)
        text = tree.body.text(strip=True) if tree.body else tree.text(strip=True)
        if not text or not text.strip():
            return ""
        # If text is just a tag (e.g., "<br>") or only HTML entities
        if re.fullmatch(r"<[^>]+>", text.strip()) or text.strip() in {"&nbsp;", "&#160;"}:
            return ""
        return text.strip()
    except Exception:
        return ""

def _label_in(label, target) -> bool:
    """Robustly checks if label is or contains the target label."""
    if isinstance(label, str):
        return label == target
    if isinstance(label, list):
        return target in label
    return False

def _extract_segments_by_label(
    segments,
    label_name,
    extra_fields=None,
    dedupe_on: str = "segment_hash",
    min_text_len: int = 2,
    max_text_len: int = 500,
    allow_numeric_only: bool = False,
    allow_special_only: bool = False,
    custom_validator=None,
    diagnostics: bool = False,
    logger=logger,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Advanced extraction and cleaning of segments by label.
    - Skips empty, whitespace-only, trivial, numeric-only, or special-char-only text.
    - Supports deduplication, custom validators, and diagnostics.
    - Logs filtered-out segments if logger provided.
    """
    results = []
    filtered_out = []
    seen = set()

    def is_numeric_only(val):
        return isinstance(val, str) and val.strip().isdigit()

    def is_special_only(val):
        return isinstance(val, str) and bool(re.fullmatch(r'[\W_]+', val.strip()))

    for seg in segments:
        label = safe_get(seg, "ml_label", None)
        if not _label_in(label, label_name):
            continue
        text = _extract_clean_text(safe_get(seg, "html", ""))
        raw_html = safe_get(seg, "html", "")
        segment_hash = safe_get(seg, "segment_hash", None)
        skip_reason = None

        # Filtering logic
        if not text or not text.strip():
            skip_reason = "empty"
        elif text.strip() in {"&nbsp;", "&#160;"}:
            skip_reason = "html_entity"
        elif re.fullmatch(r"<[^>]+>", text.strip()):
            skip_reason = "just_tag"
        elif len(text.strip()) < min_text_len:
            skip_reason = "too_short"
        elif len(text.strip()) > max_text_len:
            skip_reason = "too_long"
        elif is_numeric_only(text) and not allow_numeric_only:
            skip_reason = "numeric_only"
        elif is_special_only(text) and not allow_special_only:
            skip_reason = "special_only"
        elif custom_validator and not custom_validator(text, seg):
            skip_reason = "custom_validator_failed"
        elif dedupe_on and segment_hash in seen:
            skip_reason = "duplicate"
        else:
            seen.add(segment_hash)

        if skip_reason:
            filtered_out.append({"segment": seg, "reason": skip_reason})
            continue

        entry = {
            "text": text,
            "raw_html": raw_html,
            "segment_hash": segment_hash,
        }
        if extra_fields:
            for field in extra_fields:
                entry[field] = safe_get(seg, field, None)
        results.append(entry)

    # Optional diagnostics logging
    if diagnostics and filtered_out:
        for item in filtered_out[:10]:
            msg = f"[SEGMENT FILTER] Skipped segment ({item['reason']}): {str(item['segment'])[:120]}..."
            if logger.mode == "cli":
                console.print(msg)
            else:
                payload = {
                    "level": "WARNING",
                    "type": "segment_filter",
                    "message": msg
                }
                logger.warning(payload)
        if len(filtered_out) > 10:
            msg = f"[SEGMENT FILTER] {len(filtered_out)} segments filtered out (showing first 10)."
            if logger.mode == "cli":
                console.print(msg)
            else:
                payload = {
                    "level": "WARNING",
                    "type": "segment_filter",
                    "message": msg
                }
                logger.warning(payload)

    return results

def extract_year_and_type(text, url=None) -> tuple:
    """
    Extracts the most likely year and election type from anywhere in the string or url.
    Also extracts a 'last updated' date if present.
    Returns (year, election_type, cleaned_text, last_updated)
    """
    
    # Helper: Remove "last updated" and similar phrases
    def remove_last_updated(s):
        s = re.sub(r'last updated.*', '', s, flags=re.IGNORECASE)
        s = re.sub(r'this page auto-refreshes.*', '', s, flags=re.IGNORECASE)
        s = re.sub(r'updated\s*(on)?\s*\w+day,?.*', '', s, flags=re.IGNORECASE)
        return s

    # Helper: Extract year/type from a string
    def extract_from_string(s):
        years = re.findall(r'(20\d{2})', s)
        type_matches = []
        for t in ELECTION_TYPES:
            for m in re.finditer(rf'\b{re.escape(t)}\b', s, re.IGNORECASE):
                type_matches.append((m.start(), t))
        return years, type_matches

    # Helper: Extract last updated date
    def extract_last_updated(s) -> Optional[str]:
        # Match patterns like "Last Updated Wednesday, February 5, 2025, 11:57:22 AM"
        m = re.search(
            r'(last updated|updated)\s*[:,\-]?\s*([A-Za-z]+day,?\s+[A-Za-z]+\s+\d{1,2},\s+20\d{2}.*?\d{1,2}:\d{2}(:\d{2})?\s*(AM|PM)?)',
            s, re.IGNORECASE)
        if m:
            return m.group(2).strip()
        # Fallback: just a date/time after "updated"
        m = re.search(
            r'updated\s*[:,\-]?\s*([A-Za-z]+day,?\s+[A-Za-z]+\s+\d{1,2},\s+20\d{2}.*?\d{1,2}:\d{2}(:\d{2})?\s*(AM|PM)?)',
            s, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return None

    last_updated = extract_last_updated(text)

    cleaned_text = remove_last_updated(text)
    years, type_matches = extract_from_string(cleaned_text)

    url_years, url_type_matches = [], []
    if url:
        url_years = re.findall(r'(20\d{2})', url)
        url_type_matches = []
        for t in ELECTION_TYPES:
            if re.search(rf'{t}20\d{{2}}|20\d{{2}}{t}', url, re.IGNORECASE):
                url_type_matches.append((0, t))
            for m in re.finditer(rf'\b{re.escape(t)}\b', url, re.IGNORECASE):
                url_type_matches.append((m.start(), t))

    all_years = years if years else url_years
    all_type_matches = type_matches if type_matches else url_type_matches

    year = all_years[-1] if all_years else None
    type_found = None
    if all_type_matches:
        types_only = [t for _, t in all_type_matches]
        type_found = safe_get_first([t for t, _ in Counter(types_only).most_common(1)], "type_found", None, logger)
        if type_found is None:
            type_found = safe_get_first([x[1] for x in sorted(all_type_matches, key=lambda x: x[0])], "type_found_sorted", None, logger)

    cleaned = cleaned_text
    if all_years:
        for y in all_years:
            cleaned = re.sub(rf'\b{y}\b', '', cleaned)
    if all_type_matches:
        for _, t in all_type_matches:
            cleaned = re.sub(rf'\b{re.escape(t)}\b', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+20\d{2}\b', '', cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip(" -:|,")
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return year, type_found, cleaned, last_updated

def is_update_panel(text) -> bool:
    """
    Detects if a panel/heading is a last-updated, status, or reporting info panel.
    Uses robust keyword and phrase matching.
    """
    t = safe_lower(text)
    # Direct keyword match
    if any(safe_lower(kw) in t for kw in UPDATE_PANEL_KEYWORDS):
        return True
    # Dynamic "view by ..." phrase match
    if any(f"view by {safe_lower(phrase)}" in t for phrase in VIEW_BY_PHRASES):
        return True
    # Common "as of" or "updated" patterns
    if "as of" in t or "updated" in t:
        return True
    return False

def split_possible_contests(text) -> List[str]:
    """
    Split a long contest block into individual contest-like lines.
    Uses contest keywords, newlines, and vote/candidate patterns.
    """
    # Split on double newlines, or lines starting with known contest/office keywords
    lines = re.split(r'\n{2,}|(?:\r?\n)+', text)
    blocks = []
    buffer = []
    for line in lines:
        line = line.strip()
        # Heuristic: start a new block if line contains a contest keyword and is not a status/update line
        if any(kw in line.lower() for kw in CONTEST_KEYWORDS) and not is_update_panel(line):
            if buffer:
                blocks.append(" ".join(buffer))
                buffer = []
            buffer.append(line)
        else:
            buffer.append(line)
    if buffer:
        blocks.append(" ".join(buffer))
    # Remove blocks that are just status/update lines
    return [b for b in blocks if not is_update_panel(b) and len(b) > 20]

def extract_tagged_segments_with_attrs(
    html: str,
    context_library: dict = None,
    context_cache: Optional[Dict[str, Any]] = None,
    include_data_attrs: bool = True,
    fallback_on_error: bool = True,
    model_name: Optional[str] = None,
    use_finetuned: bool = True,
    pattern_kb: list = None,
    ml_threshold: float = SEGMENT_ML_LABEL_THRESHOLD_STRICT,
    model=None,
    coordinator=None,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Ultra-advanced DOM segment extraction with ML and dynamic context enrichment.
    - Uses selectolax for DOM parsing.
    - Leverages ContextCoordinator for all NLP/entity enrichment.
    - Multi-level filtering, robust parent/child relationships, unique indices, and auditability.
    - Uses coordinator.extract_field for dynamic context enrichment.
    """
    from ..Context_Integration.context_coordinator import ContextCoordinator
    from ..Context_Integration.context_organizer import ContextOrganizer

    coordinator = coordinator or ContextCoordinator()
    if context_cache is not None:
        clean_cache_inplace(context_cache)
    if model is None:
        model = ModelRegistry.get_sentence_transformer(model_name=model_name, use_finetuned=use_finetuned)
    if pattern_kb is None:
        pattern_kb = load_pattern_kb()
    if context_library is None:
        context_library = {}
    if context_cache is None:
        context_cache = load_context_cache_from_disk()

    # --- Dynamic context enrichment ---
    all_panel_tags: Set[str] = PANEL_TAGS | CONTEST_PANEL_TAGS | {"main", "aside", "article"}  # Set[str]
    all_heading_tags: Set[str] = HEADING_TAGS | EXTRA_HEADING_TAGS  # Set[str]
    all_table_tags: Set[str] = TABLE_TAGS  # Set[str]
    all_structural_tags: Set[str] = STRUCTURAL_TAGS | ROOT_CONTAINER_TAGS | ALWAYS_IGNORE_TAGS  # Set[str]
    all_icon_tags: Set[str] = ICON_TAGS  # Set[str]
    all_button_tags: Set[str] = BUTTON_TAGS  # Set[str]
    all_classes_ignore: Set[str] = ALWAYS_IGNORE_CLASSES | ICON_CLASSES | BUTTON_CLASSES | PANEL_CLASSES | HEADING_CLASSES  # Set[str]
    all_ids_ignore: Set[str] = ALWAYS_IGNORE_IDS  # Set[str]
    all_location_keywords: Set[str] = LOCATION_KEYWORDS | set(LOCATION_ABBREVIATIONS.keys())  # Set[str]
    all_candidate_keywords: Set[str] = CANDIDATE_KEYWORDS | PARTY_KEYWORDS  # Set[str]
    all_ballot_types: Set[str] = set(BALLOT_TYPES) | set(BALLOT_TYPES_SORT_ORDER)  # Set[str]
    all_contest_keywords: Set[str] = set(CONTEST_KEYWORDS)  # Set[str]
    all_misc_keywords: Set[str] = set(TOTAL_KEYWORDS) | set(PERCENT_KEYWORDS) | set(MISC_FOOTER_KEYWORDS) | set(UPDATE_PANEL_KEYWORDS)  # Set[str]
    all_state_tags: Set[str] = set(STATE_TAGS) | set(STATE_ABBR.keys())  # Set[str]
    all_office_keywords: Set[str] = set([k for k, _ in OFFICE_KEYWORDS])  # Set[str]
    all_precinct_patterns: List[Pattern] = [re.compile(pat) for pat in PRECINCT_HEADER_PATTERNS]  # List[Pattern]
    all_noisy_label_patterns: List[Pattern] = NOISY_LABEL_PATTERNS  # List[Pattern]
    all_selectors: Dict[str, Dict[str, str]] = SELECTORS  # Dict[str, Dict[str, str]]
    all_canonical_labels: Dict[str, str] = CANONICAL_SEGMENT_LABELS  # Dict[str, str]
    all_district_regex: Pattern = DISTRICT_REGEX  # Pattern
    all_election_types: Set[str] = set(ELECTION_TYPES)  # Set[str]
    all_party_keywords: Set[str] = set(PARTY_KEYWORDS)  # Set[str]
    
    # --- Coordinator-driven context using extract_field ---
    context_contests = set()
    context_parties = set()
    context_vote_methods = set()
    if coordinator:
        try:
            context_contests = set(
                safe_lower(c.get("title", "")) for c in coordinator.get_contests() if isinstance(c, dict)
            )
        except Exception:
            context_contests = set()

        try:
            all_state_tags |= set(coordinator.extract_field("states") or [])
        except Exception:
            all_state_tags |= set()

        try:
            for state in all_state_tags:
                context_counties = coordinator.extract_field("precincts", context={"state": state}) or []
                all_location_keywords |= set(context_counties)
        except Exception:
            all_location_keywords |= set()

        try:
            all_election_types |= set(coordinator.extract_field("election_types") or [])
        except Exception:
            all_election_types |= set()

        try:
            for county in coordinator.get_known_county_to_PRECINCTS_map():
                precincts = coordinator.extract_field("precincts", context={"county": county}) or []
                all_location_keywords |= set(precincts)
        except Exception:
            all_location_keywords |= set()

        try:
            context_parties = set(safe_lower(p) for p in (coordinator.extract_field("party") or []))
        except Exception:
            context_parties = set()

        try:
            context_vote_methods = set(
                safe_lower(vm) for vm in (coordinator.extract_field("vote_methods") or [])
            )
        except Exception:
            context_vote_methods = set()

    segments: List[Dict[str, Any]] = []

    try:
        tree = HTMLParser(html)
        idx_counter = [0]

        def safe_split(val, sep=None):
            try:
                return val.split(sep) if isinstance(val, str) else []
            except Exception:
                return []

        def is_semantic_tag(tag, classes, id_, attrs, text):
            tag = safe_lower(tag)
            classes = set(safe_lower(c) for c in classes)
            id_ = safe_lower(id_)
            text = safe_lower(text)
            # --- Heuristic rules ---
            if tag in all_panel_tags or classes & PANEL_CLASSES:
                return "panel"
            if tag in all_heading_tags or classes & HEADING_CLASSES:
                return "heading"
            if tag in all_table_tags or "table" in classes or "results" in classes:
                return "results_table"
            if tag in all_button_tags or classes & BUTTON_CLASSES or "button" in id_:
                return "ballot_toggle"
            if tag in all_icon_tags or classes & ICON_CLASSES:
                return "ignore"
            if classes & TIMESTAMP_CLASSES or any(re.search(pat, id_) for pat in TIMESTAMP_ID_PATTERNS):
                return "results_timestamp"
            if tag in all_state_tags or any(st in text for st in all_state_tags):
                return "state_panel"
            if any(ok in text for ok in all_office_keywords) or text in context_contests:
                return "contest"
            if any(pat.search(text) for pat in all_precinct_patterns) or all_district_regex.search(text):
                return "location_panel"
            for sel_type, sel_dict in all_selectors.items():
                for k, v in sel_dict.items():
                    if safe_lower(safe_get(attrs, k, "")) == safe_lower(v):
                        return sel_type
            if any(et in text for et in all_election_types):
                return "ballot_types"
            # Use coordinator.extract_field for party and vote_method detection
            party_label = None
            vote_method_label = None
            try:
                party_label = coordinator.extract_field("party", text=text)
            except Exception:
                party_label = None
            try:
                vote_method_label = coordinator.extract_field("vote_methods", text=text)
            except Exception:
                vote_method_label = None
            if party_label:
                return "party_label"
            if vote_method_label:
                return "vote_method"
            if any(pk in text for pk in all_party_keywords) or text in context_parties:
                return "party_label"
            if text in context_vote_methods:
                return "vote_method"
            if any(mk in text for mk in all_misc_keywords):
                return "misc_info"
            if tag in all_structural_tags or classes & all_classes_ignore or id_ in all_ids_ignore:
                return "ignore"
            canonical = all_canonical_labels.get(text)
            if canonical:
                return canonical
            if any(pat.search(text) for pat in all_noisy_label_patterns):
                return "ignore"
            # --- spaCy entity-based rules using spacy_utils ---
            entities = coordinator.extract_entities(text) if coordinator else []
            locations = coordinator.extract_locations(text) if coordinator else []
            dates = coordinator.extract_dates(text) if coordinator else []
            for ent, ent_label in entities:
                ent_text_lower = safe_lower(ent)
                if ent_label == "PERSON" and any(ent_text_lower in safe_lower(kw) for kw in all_candidate_keywords):
                    return "candidate_panel"
                if ent_label in {"GPE", "LOC"} and any(ent_text_lower in safe_lower(kw) for kw in all_location_keywords):
                    return "location_panel"
                if ent_label == "ORG" and any(ent_text_lower in safe_lower(kw) for kw in all_party_keywords):
                    return "party_label"
            if locations:
                return "location_panel"
            if dates:
                return "results_timestamp"
            return None

        def walk(node, parent_idx=None, heading_idx=None, panel_idx=None, **kwargs):
            tag = getattr(node, "tag", None)
            tag_lower = safe_lower(tag or "")
            if not tag or tag_lower not in HTML_TAGS:
                msg = f"[UNKNOWN_TAG] {tag}"
                if logger.mode == "cli":
                    console.print(msg)
                else:
                    payload = {
                        "level": "WARNING",
                        "type": "unknown_tag",
                        "message": msg
                    }
                    logger.warning(payload)
                for child in getattr(node, "iter", lambda **kw: [])(include_text=True, **kwargs):
                    walk(child, parent_idx, heading_idx, panel_idx, **kwargs)
                return None
            attrs = dict(getattr(node, "attributes", {}))
            if include_data_attrs:
                attrs.update({k: v for k, v in getattr(node, "attributes", {}).items() if safe_lower(k or "").startswith("data-")})
            classes = safe_split(attrs.get("class", "") or "")
            id_ = attrs.get("id", "")
            is_button = tag_lower == "button" or (tag_lower == "input" and safe_lower(attrs.get("type", "") or "") in ["button", "submit"])
            button_text = (
                attrs.get("aria-label")
                or attrs.get("value")
                or (getattr(node, "text", lambda **kw: "")(strip=True, **kwargs) if hasattr(node, "text") else "")
                or ""
            ) if is_button else ""
            is_clickable = (
                is_button
                or tag_lower == "a"
                or "onclick" in attrs
                or "btn" in classes
                or "button" in classes
            )

            this_heading_idx = heading_idx
            if tag_lower in all_heading_tags:
                this_heading_idx = safe_get_first(idx_counter, "heading_idx", None, logger)
            this_panel_idx = panel_idx
            if tag_lower in all_panel_tags:
                this_panel_idx = safe_get_first(idx_counter, "panel_idx", None, logger)

            seg = {
                "tag": tag_lower,
                "attrs": attrs,
                "classes": classes,
                "id": id_,
                "html": "",
                "is_button": is_button,
                "is_clickable": is_clickable,
                "button_text": button_text,
                "parent_idx": parent_idx,
                "children": [],
                "start": getattr(node, "start", None),
                "end": getattr(node, "end", None),
                "_idx": safe_get_first(idx_counter, "_idx", None, logger),
                "context_heading_idx": this_heading_idx,
                "panel_ancestor_idx": this_panel_idx,
                "panel_ancestor_heading": None,
            }
            idx_counter[0] = safe_get_first([idx_counter[0] + 1], "idx_counter_increment", idx_counter[0] + 1, logger)
            for k in attrs:
                if any(pat.match(k) for pat in CUSTOM_ATTR_PATTERNS):
                    seg["has_custom_attr"] = True
                msg = f"[UNKNOWN_ATTR] {k}"
                if logger.mode == "cli":
                    console.print(msg)
                else:
                    payload = {
                        "level": "WARNING",
                        "type": "unknown_attr",
                        "message": msg
                    }
                    logger.warning(payload)
            node_start = getattr(node, "start", None)
            node_end = getattr(node, "end", None)
            if node_start is not None and node_end is not None:
                html_bytes = html.encode("utf-8")
                try:
                    seg["html"] = html_bytes[node_start:node_end].decode("utf-8", errors="replace")
                except Exception:
                    seg["html"] = html[node_start:node_end] if isinstance(html, str) else ""
            else:
                seg["html"] = getattr(node, "html", "") if hasattr(node, "html") else ""

            seg["segment_hash"] = segment_identity_hash(seg)

            # --- Multi-level filtering ---
            clean_text = _extract_clean_text(seg["html"])
            if not clean_text or not clean_text.strip():
                return None
            if clean_text.strip() in {"&nbsp;", "&#160;"}:
                return None
            if re.fullmatch(r"<[^>]+>", clean_text.strip()):
                return None
            if tag_lower in ROOT_CONTAINER_TAGS or tag_lower in ALWAYS_IGNORE_TAGS or tag_lower in STRUCTURAL_TAGS:
                return None
            if set(classes) & ALWAYS_IGNORE_CLASSES or id_ in ALWAYS_IGNORE_IDS:
                return None
            if tag_lower in ICON_TAGS and (set(classes) & ICON_CLASSES or not clean_text.strip()):
                return None
            if len(clean_text.strip()) < 2 or clean_text.strip().isdigit() or bool(re.fullmatch(r'[\W_]+', clean_text.strip())):
                return None
            if tag_lower in {"script", "style", "meta", "link", "base"}:
                return None

            # --- Smart semantic categorization ---
            semantic_label = is_semantic_tag(tag_lower, classes, id_, attrs, clean_text)
            text_lower = safe_lower(clean_text)
            html_lower = safe_lower(seg["html"])

            # --- Context-driven overrides ---
            if any(kw in text_lower for kw in all_location_keywords) or any(kw in html_lower for kw in all_location_keywords):
                semantic_label = semantic_label or "location_panel"
            if any(kw in text_lower for kw in all_candidate_keywords) or any(kw in html_lower for kw in all_candidate_keywords):
                semantic_label = semantic_label or "candidate_panel"
            if any(bt in text_lower for bt in all_ballot_types) or any(bt in html_lower for bt in all_ballot_types):
                semantic_label = semantic_label or "ballot_types"
            if any(kw in text_lower for kw in all_contest_keywords) or any(kw in html_lower for kw in all_contest_keywords) or text_lower in context_contests:
                semantic_label = semantic_label or "contest"
            if any(kw in text_lower for kw in all_misc_keywords):
                semantic_label = semantic_label or "misc_info"
            if any(kw in text_lower for kw in TOTAL_KEYWORDS | PERCENT_KEYWORDS | MISC_FOOTER_KEYWORDS):
                semantic_label = semantic_label or "results_table"
            if any(kw in text_lower for kw in UPDATE_PANEL_KEYWORDS) or re.search(r"\b\d{1,2}:\d{2}\s*(am|pm)?\b", text_lower) or re.search(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", text_lower):
                semantic_label = semantic_label or "results_timestamp"
            if tag_lower == "a" and "href" in attrs:
                href = safe_lower(str(attrs["href"]))
                if any(href.endswith(ext) for ext in [".csv", ".json", ".pdf", ".xlsx", ".zip", ".xls", ".doc", ".docx"]):
                    semantic_label = "download_link"
            if is_clickable:
                semantic_label = semantic_label or "clickable"
            # Use coordinator.extract_field for party and vote_method detection (again for override)
            try:
                if coordinator.extract_field("party", text=text_lower):
                    semantic_label = semantic_label or "party_label"
            except Exception:
                pass
            try:
                if coordinator.extract_field("vote_methods", text=text_lower):
                    semantic_label = semantic_label or "vote_method"
            except Exception:
                pass
            if any(pk in text_lower for pk in all_party_keywords) or text_lower in context_parties:
                semantic_label = semantic_label or "party_label"
            if text_lower in context_vote_methods:
                semantic_label = semantic_label or "vote_method"
            if any(et in text_lower for et in all_election_types):
                semantic_label = semantic_label or "ballot_types"

            # --- ML/heuristic fallback ---
            if not semantic_label and coordinator and hasattr(coordinator, "auto_label_segment"):
                try:
                    ml_label = coordinator.auto_label_segment(seg, context_library=context_library, context_cache=context_cache, pattern_kb=pattern_kb, model=model, ml_threshold=ml_threshold)
                    if ml_label and isinstance(ml_label, str):
                        semantic_label = ml_label
                except Exception:
                    semantic_label = None
            if not semantic_label:
                semantic_label = "unknown"

            # --- Use coordinator for NLP enrichment ---
            seg["ml_label"] = semantic_label
            seg["nlp_entities"] = coordinator.extract_entities(clean_text) if coordinator else []
            seg["nlp_locations"] = coordinator.extract_locations(clean_text) if coordinator else []
            seg["nlp_dates"] = coordinator.extract_dates(clean_text) if coordinator else []
            segments.append(seg)
            this_idx = seg["_idx"]
            for child in getattr(node, "iter", lambda **kw: [])(include_text=True, **kwargs):
                child_idx = walk(child, this_idx, this_heading_idx, this_panel_idx, **kwargs)
                if child_idx is not None:
                    if not isinstance(seg.get("children"), list):
                        seg["children"] = []
                    seg["children"].append(child_idx)
            return this_idx

        root = tree.body or tree.html or tree.root
        walk(root, **kwargs)

        # --- Ensure unique _idx and parent/child consistency ---
        idx_map = {seg["_idx"]: seg for seg in segments}
        for seg in segments:
            seg["children"] = [c for c in seg.get("children", []) if c in idx_map]
            if seg.get("parent_idx") is not None and seg["parent_idx"] not in idx_map:
                seg["parent_idx"] = None

        # --- Advanced DOM enrichment, organization, and audit ---
        organizer = ContextOrganizer()
        dom_tree = organizer.build_dom_tree(segments)
        for seg in segments:
            seg["dom_node"] = dom_tree["nodes"][seg["_idx"]] if seg["_idx"] < len(dom_tree["nodes"]) else None

        label_groups = organizer.group_nodes_by_label(dom_tree["nodes"], label_field="ml_label")
        panels_and_tables = organizer.get_panels_and_tables(dom_tree)
        for seg in segments:
            seg["panel_group"] = None
            for panel in panels_and_tables:
                if seg["_idx"] in safe_get(panel, "panel_indices", []) or seg["_idx"] in safe_get(panel, "table_indices", []):
                    seg["panel_group"] = panel
                    break

        N = min(5, len(dom_tree["nodes"]))
        for i in range(N):
            node_html = organizer.extract_html_by_idx(dom_tree["nodes"], i, html)
            subtree_html = organizer.extract_subtree_html(dom_tree["nodes"], i, html)
            msg_node = f"[DOM NODE HTML] Node {i}: {node_html[:120]}..."
            msg_subtree = f"[DOM SUBTREE HTML] Node {i}: {subtree_html[:120]}..."
            if logger.mode == "cli":
                console.print(msg_node)
                console.print(msg_subtree)
            else:
                logger.info({"level": "INFO", "type": "dom_node_html", "message": msg_node})
                logger.info({"level": "INFO", "type": "dom_subtree_html", "message": msg_subtree})

        seg_hashes = [segment_hash(seg.get("html", "")) for seg in segments]
        seg_htmls = [seg.get("html", "") for seg in segments]
        total_segments = len(seg_hashes)
        msg_embed = f"[EMBED] Total segments: {total_segments}"
        if logger.mode == "cli":
            console.print(msg_embed)
        else:
            logger.info({"level": "INFO", "type": "embedding", "message": msg_embed})

        hash_to_embedding = {}
        CHUNK_SIZE = 1024
        for i in range(0, total_segments, CHUNK_SIZE):
            chunk_hashes = seg_hashes[i:i+CHUNK_SIZE]
            chunk_result = load_embeddings_batch(chunk_hashes)
            hash_to_embedding.update(chunk_result)
            hits = sum(1 for v in chunk_result.values() if v is not None)
            msg_batch = f"[EMBED] Batch {i//CHUNK_SIZE+1}: {hits} hits, {len(chunk_hashes)-hits} misses"
            if logger.mode == "cli":
                console.print(msg_batch)
            else:
                logger.debug({"level": "DEBUG", "type": "embedding_batch", "message": msg_batch})
        missing = [(h, html) for h, html in zip(seg_hashes, seg_htmls) if hash_to_embedding.get(h) is None]
        if missing:
            msg_missing = f"[EMBED] Computing {len(missing)} missing embeddings in chunks of {CHUNK_SIZE}"
            if logger.mode == "cli":
                console.print(msg_missing)
            else:
                logger.info({"level": "INFO", "type": "embedding_missing", "message": msg_missing})
            for i in range(0, len(missing), CHUNK_SIZE):
                chunk = missing[i:i+CHUNK_SIZE]
                missing_hashes, missing_htmls = zip(*chunk)
                try:
                    new_embs = model.encode(list(missing_htmls), convert_to_numpy=True, show_progress_bar=False)
                except Exception as e:
                    msg = f"[EMBED] Batch embedding computation failed: {e}"
                    if logger.mode == "cli":
                        console.print(msg)
                    else:
                        logger.error({"level": "ERROR", "type": "embedding", "message": msg})
                    continue
                save_embeddings_batch(list(zip(missing_hashes, new_embs)))
                msg_saved = f"[EMBED] Saved {len(chunk)} new embeddings to cache."
                for h, emb in zip(missing_hashes, new_embs):
                    hash_to_embedding[h] = emb
                if logger.mode == "cli":
                    console.print(msg_saved)
                else:
                    logger.debug({"level": "DEBUG", "type": "embedding", "message": msg_saved})
        for seg, h in zip(segments, seg_hashes):
            seg["_embedding"] = hash_to_embedding[h]
        msg_complete = f"[EMBED] Embedding assignment complete for {len(segments)} segments."
        if logger.mode == "cli":
            console.print(msg_complete)
        else:
            logger.info({"level": "INFO", "type": "embedding_complete", "message": msg_complete})

        # --- Final enrichment and audit ---
        for seg in segments:
            text = safe_lower(seg.get("html") or "")
            seg["contains_election_keyword"] = any(
                safe_lower(kw) in text for kw in (list(all_location_keywords) + list(all_candidate_keywords) + list(all_ballot_types))
            )
            seg["contains_candidate"] = any(
                safe_lower(cand) in text for cand in all_candidate_keywords
            )
            seg["contains_misc_info"] = any(
                safe_lower(mk) in text for mk in all_misc_keywords
            )
            seg["contains_nlp_person"] = any(safe_get(ent, "label", "") == "PERSON" for ent in safe_get(seg, "nlp_entities", []))
            seg["contains_nlp_location"] = bool(seg.get("nlp_locations", []))
            seg["contains_nlp_date"] = bool(seg.get("nlp_dates", []))
            emb = seg.get("_embedding")
            label = seg["ml_label"]
            seg["ml_confidence"] = 1.0 if label != "unknown" else 0.0
            html_val = seg.get('html') or ''
            if not isinstance(html_val, str):
                html_val = str(html_val)
            seg["pattern_id"] = f"pattern_{hashlib.sha256(html_val.encode('utf-8')).hexdigest()[:10]}"
            seg["is_actionable"] = label in (
                "results_table", "contest", "candidate_panel", "location_panel", "state_panel",
                "party_label", "ballot_types", "vote_method", "misc_info"
            )
            seg["is_election_result"] = label == "results_table"
            seg["is_contest"] = label == "contest"
            seg["label_group"] = label_groups.get(label, [])
            seg["panel_table_context"] = seg.get("panel_group", {})

        if segments:
            seg = segments[0]
            msg_debug = (
                f"[DOM SEGMENTS] Extracted {len(segments)} segments. Example: "
                f"tag={seg.get('tag','')}, label={seg.get('ml_label','')}, text={_extract_clean_text(seg.get('html',''))[:80]}..."
            )
            if logger.mode == "cli":
                console.print(msg_debug)
            else:
                logger.debug({"level": "DEBUG", "type": "dom_segments", "message": msg_debug})
        else:
            msg_debug = "[DOM SEGMENTS] Extracted 0 segments."
            if logger.mode == "cli":
                console.print(msg_debug)
            else:
                logger.debug({"level": "DEBUG", "type": "dom_segments", "message": msg_debug})
        if not segments:
            msg_warn = "[DOM SEGMENTS] No DOM segments extracted. Check HTML input and parser logic."
            if logger.mode == "cli":
                console.print(msg_warn)
            else:
                logger.warning({"level": "WARNING", "type": "dom_segments", "message": msg_warn})

        dom_enrichment = {
            "dom_tree": dom_tree,
            "label_groups": label_groups,
            "panels_and_tables": panels_and_tables,
            "node_html_samples": [organizer.extract_html_by_idx(dom_tree["nodes"], i, html) for i in range(N)],
            "subtree_html_samples": [organizer.extract_subtree_html(dom_tree["nodes"], i, html) for i in range(N)],
        }
        for k, v in dom_enrichment.items():
            for seg in segments:
                seg[f"dom_{k}"] = v

        return segments

    except Exception as e:
        error_details = {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "html_snippet": (html or "")[:200],
            "segments_extracted": len(segments),
        }
        msg_error = f"[FALLBACK] selectolax failed: {e}\nDetails: {error_details}"
        if logger.mode == "cli":
            console.print(msg_error)
        else:
            logger.error({"level": "ERROR", "type": "selectolax_fallback", "message": msg_error})
        if not fallback_on_error:
            raise
        return [{
            "error_info": {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "html_snippet": (html or "")[:200],
                "segments_extracted": len(segments),
            },
            "segments": [],
        }]

def get_page_hash(page) -> str:
    """
    Robustly compute a hash for the page content.
    Handles None, bytes, and normalizes whitespace for stability.
    Mode-aware logging.
    """
    try:
        if page is None:
            content = ""
        else:
            try:
                content = getattr(page, "content", lambda: "")()
            except Exception:
                msg = "[PAGE_HASH] Exception when calling page.content(), using empty string."
                if logger.mode == "cli":
                    console.print(msg)
                else:
                    logger.warning({"level": "WARNING", "type": "page_hash", "message": msg})
                content = ""
        if content is None:
            msg = "[PAGE_HASH] Page content is None, using empty string for hash."
            if logger.mode == "cli":
                console.print(msg)
            else:
                logger.warning({"level": "WARNING", "type": "page_hash", "message": msg})
            content = ""
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")
        elif not isinstance(content, str):
            content = str(content)
        content = content.replace('\r\n', '\n').replace('\r', '\n').strip()
        if not content:
            msg = "[PAGE_HASH] Page content is empty after normalization."
            if logger.mode == "cli":
                console.print(msg)
            else:
                logger.warning({"level": "WARNING", "type": "page_hash", "message": msg})
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
    except Exception as e:
        msg = f"[PAGE_HASH] Failed to compute hash: {e}"
        if logger.mode == "cli":
            console.print(msg)
        else:
            logger.error({"level": "ERROR", "type": "page_hash", "message": msg})
        return hashlib.sha256(b"").hexdigest()


def load_context_cache_from_disk(filename=None) -> Dict[str, Any]:
    """
    Loads the context cache from disk as a dict of dicts.
    Mode-aware logging.
    """
    global _context_cache
    path = CONTEXT_CACHE_PATH
    if filename is not None and os.path.basename(filename) != os.path.basename(CONTEXT_CACHE_PATH):
        msg = f"[CACHE] Ignoring filename '{filename}', using CONTEXT_CACHE_PATH."
        if logger.mode == "cli":
            console.print(msg)
        else:
            logger.warning({"level": "WARNING", "type": "cache", "message": msg})
    msg_debug = f"[DEBUG] Loading context cache from: {path}"
    if logger.mode == "cli":
        console.print(msg_debug)
    else:
        logger.debug({"level": "DEBUG", "type": "cache", "message": msg_debug})
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                raw_cache = robust_orjson_loads(f.read())
                _context_cache = {k: v for k, v in safe_items(raw_cache or {}) if isinstance(v, dict)}
                return _context_cache
        except Exception as e:
            msg_error = f"[ERROR] Failed to load context cache: {e}. Resetting context cache."
            if logger.mode == "cli":
                console.print(msg_error)
            else:
                logger.error({"level": "ERROR", "type": "cache", "message": msg_error})
            _context_cache = {}
            save_context_cache_to_disk(_context_cache)
            return {}
    _context_cache = {}
    return {}

def save_context_cache_to_disk(context_cache, path=None) -> None:
    """
    Saves the entire context cache as a single JSON object (dict of dicts).
    Mode-aware logging.
    """
    cache_path = CONTEXT_CACHE_PATH
    if path is not None and os.path.basename(path) != os.path.basename(CONTEXT_CACHE_PATH):
        msg = f"[CACHE] Ignoring path '{path}', using CONTEXT_CACHE_PATH."
        if logger.mode == "cli":
            console.print(msg)
        else:
            logger.warning({"level": "WARNING", "type": "cache", "message": msg})
    msg_debug = f"[DEBUG] Saving context cache to: {cache_path}"
    if logger.mode == "cli":
        console.print(msg_debug)
    else:
        logger.debug({"level": "DEBUG", "type": "cache", "message": msg_debug})
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        context_cache = convert_ndarrays(context_cache)
        with open(cache_path, "wb") as f:
            try:
                f.write(orjson.dumps(context_cache, option=orjson.OPT_INDENT_2))
            except Exception as e:
                msg_error = f"[ERROR] Failed to serialize context cache: {e}"
                if logger.mode == "cli":
                    console.print(msg_error)
                else:
                    logger.error({"level": "ERROR", "type": "cache", "message": msg_error})
    except Exception as e:
        msg_error = f"[ERROR] Failed to save context cache to disk at {cache_path}: {e}"
        if logger.mode == "cli":
            console.print(msg_error)
        else:
            logger.error({"level": "ERROR", "type": "cache", "message": msg_error})

def add_context_entry(page_hash: str, context: dict, path=None) -> None:
    """
    Adds or updates a context entry for a page hash and saves to disk.
    Always uses CONTEXT_CACHE_PATH from config.py.
    """
    cache = load_context_cache_from_disk()
    # Always ensure required metadata
    context.setdefault("page_hash", page_hash)
    context.setdefault("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
    cache[page_hash] = context
    save_context_cache_to_disk(cache)

def get_context_entry(page_hash: str, path=CONTEXT_CACHE_PATH) -> Optional[dict]:
    """
    Retrieves a context entry by page hash.
    """
    cache = load_context_cache_from_disk(path)
    return cache.get(page_hash)

def export_context_cache_for_db(path=CONTEXT_CACHE_PATH) -> List[dict]:
    """
    Flattens the context cache into a list of dicts for DB insertion.
    Each dict contains url, page_hash, timestamp, and all context fields.
    """
    cache = load_context_cache_from_disk(path)
    export = []
    for page_hash, context in cache.items():
        entry = dict(context)
        entry["page_hash"] = page_hash
        export.append(entry)
    return export

def load_pattern_kb() -> List[Dict[str, Any]]:
    """
    Loads the pattern KB from dom_pattern_kb.jsonl as a list of dicts.
    Deduplicates by pattern_id and timestamp, and ignores corrupt lines.
    Caches the result for future calls.
    """
    global _pattern_kb_cache
    if _pattern_kb_cache is not None:
        return _pattern_kb_cache
    kb = []
    path = safe_log_path("dom_pattern_kb.jsonl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            for line in f:
                try:
                    entry = robust_orjson_loads(line)
                    # Only accept dicts with required keys
                    if isinstance(entry, dict) and "pattern_id" in entry and "label" in entry:
                        kb.append(entry)
                except Exception:
                    continue
    # Deduplicate by pattern_id, keep latest timestamp
    dedup = {}
    for entry in kb:
        pid = safe_get(entry, "pattern_id", None)
        ts = safe_get(entry, "timestamp", 0)
        if pid not in dedup or ts > safe_get(dedup.get(pid, {}), "timestamp", 0):
            dedup[pid] = entry
    _pattern_kb_cache = list(dedup.values())
    return _pattern_kb_cache

def append_pattern_kb(entry) -> None:
    """
    Appends a pattern KB entry to dom_pattern_kb.jsonl as a single-line JSON object.
    Converts numpy embeddings to lists, and ensures valid structure.
    """
    if not isinstance(entry, dict):
        raise ValueError("Only dict entries can be written to dom_pattern_kb.jsonl")
    entry = convert_ndarrays(entry)
    # Ensure embedding is a list (even if empty)
    if "embedding" in entry:
        emb = entry["embedding"]
        if isinstance(emb, np.ndarray):
            entry["embedding"] = emb.tolist()
        elif not isinstance(emb, list):
            entry["embedding"] = list(emb) if emb else []
    else:
        entry["embedding"] = []
    # Defensive: ensure required keys
    for key in ["pattern_id", "label", "timestamp"]:
        if key not in entry:
            raise ValueError(f"Missing required key '{key}' in pattern KB entry")
    path = safe_log_path("dom_pattern_kb.jsonl")
    with open(path, "ab") as f:
        f.write(orjson.dumps(entry) + b"\n")

def append_feedback_log(entry) -> None:
    if not isinstance(entry, dict):
        raise ValueError("Only dict entries can be written to segment_feedback_log.jsonl")
    entry = convert_ndarrays(entry)
    # Use safe_get for all dict access
    embedding = safe_get(entry, "embedding", [])
    if isinstance(embedding, np.ndarray):
        entry["embedding"] = (embedding or np.array([])).tolist()
    path = safe_log_path("segment_feedback_log.jsonl")
    with open(path, "ab") as f:
        f.write(orjson.dumps(entry, option=orjson.OPT_INDENT_2) + b"\n")
    global _pattern_kb_cache
    if safe_get(entry, "pattern_id", None) and safe_get(entry, "label", None) and safe_get(entry, "html", None):
        seg_hash = segment_identity_hash({
            "tag": safe_get(entry, "tag", ""),
            "attrs": safe_get(entry, "attrs", {}),
            "html": safe_get(entry, "html", "")
        })
        kb_entry = {
            "pattern_id": safe_get(entry, "pattern_id", None),
            "label": safe_get(entry, "label", None),
            "embedding": (safe_get(entry, "embedding", []) or np.array([])).tolist(),
            "example_html": safe_get(entry, "html", "")[:500],
            "segment_hash": seg_hash,
            "timestamp": safe_get(entry, "timestamp", 0),
        }
        if _pattern_kb_cache is not None and isinstance(_pattern_kb_cache, list):
            _pattern_kb_cache.append(kb_entry)

def label_validator(val: str) -> bool:
    return safe_lower(safe_strip(val)) in ALLOWED_LABELS

def prompt_for_segment_label(
    segment,
    context_library=None,
    session_id=None,
) -> str:
    seg_hash = segment_identity_hash(segment)
    cached_label = get_cached_segment_label(seg_hash)
    if cached_label:
        return cached_label
    html_preview = safe_get(segment, "html", "")
    canonical_label = get_canonical_segment_label(html_preview)
    if canonical_label:
        cache_segment_label(seg_hash, canonical_label)
        return canonical_label
    auto = auto_label_segment(segment, context_library=context_library)
    if auto != "ignore" and auto != "unknown":
        cache_segment_label(seg_hash, auto)
        return auto
    if not ENABLE_SEGMENT_LABEL_PROMPT:
        return "unknown"
    if not html_preview:
        html_preview = f"[No HTML] tag={safe_get(segment, 'tag', [])} attrs={safe_get(segment, 'attrs', [])}"
    msg = (
        f"\n[bold yellow]Segment needs review:[/bold yellow]\n"
        f"{html_preview[:200]}{'...' if len(html_preview) > 200 else ''}"
    )
    info_msg = (
        "[cyan]What is the semantic role of this segment? Allowed labels: "
        f"{', '.join(sorted(ALLOWED_LABELS))}[/cyan]"
    )
    if logger.mode == "cli":
        console.print(msg)
        console.print(info_msg)
    else:
        logger.warning({"level": "WARNING", "type": "segment_review", "message": msg})
        logger.info({"level": "INFO", "type": "segment_prompt", "message": info_msg})
    label = prompt.prompt_input(
        "> ",
        session_id=session_id,
        validator=label_validator,
        on_error=lambda msg: (
            console.print(f"[PROMPT] {msg} Allowed: {', '.join(sorted(ALLOWED_LABELS))}")
            if logger.mode == "cli"
            else logger.warning({
                "level": "WARNING",
                "type": "prompt",
                "message": f"[PROMPT] {msg} Allowed: {', '.join(sorted(ALLOWED_LABELS))}"
            })
        )
    ).strip()
    cache_segment_label(seg_hash, label)
    return label

def segment_hash(html) -> str:
    canon = canonicalize_segment(html)
    return hashlib.sha256(canon.encode('utf-8')).hexdigest()

def canonicalize_segment(html: str) -> str:
    """
    Canonicalize HTML for stable hashing:
    - Lowercase, strip whitespace
    - Remove volatile attributes (id, class, ng*, data-*, aria-*, style, tabindex, etc.)
    - Remove empty tags and comments
    - Normalize whitespace and attribute order
    - Remove script/style content
    """
    html = html.strip().lower()
    if html in ('<br>', '<br/>'):
        return '<br>'

    # Remove HTML comments
    html = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)

    # Remove script/style blocks
    html = re.sub(r'<script.*?>.*?</script>', '', html, flags=re.DOTALL)
    html = re.sub(r'<style.*?>.*?</style>', '', html, flags=re.DOTALL)

    # Remove volatile attributes (id, class, ng*, data-*, aria-*, style, tabindex, etc.)
    html = re.sub(r'\s(id|class|style|tabindex|title|role|aria-[\w-]+|data-[\w-]+|_ngcontent-[^=]+|_nghost-[^=]+|ng-\w+)="[^"]*"', '', html)
    html = re.sub(r"\s(id|class|style|tabindex|title|role|aria-[\w-]+|data-[\w-]+|_ngcontent-[^=]+|_nghost-[^=]+|ng-\w+)='[^']*'", '', html)

    # Remove empty tags (e.g., <span></span>)
    html = re.sub(r'<(\w+)[^>]*>\s*</\1>', '', html)

    # Remove extra whitespace between tags and in attributes
    html = re.sub(r'\s+', ' ', html)
    html = re.sub(r'>\s+<', '><', html)

    # Remove leading/trailing whitespace again
    html = html.strip()

    # Optionally: sort attributes within tags for stability
    def sort_attrs(match: re.Match) -> str:
        # Safety net for match and group extraction
        if match is None:
            return ""
        try:
            # Check if match has at least 2 groups
            if match.lastindex is None or match.lastindex < 2:
                return match.group(0)
            if match.lastindex is None or match.lastindex < 2:
                return match.group(0)
            tag = match.group(1)
            attrs = match.group(2)
            if not isinstance(attrs, str):
                return match.group(0)
        except (IndexError, AttributeError, TypeError):
            return match.group(0) if match else ""
        # Split attributes, sort, and rejoin
        attrs_list = re.findall(r'(\S+="[^"]*"|\S+=\'[^\']*\')', attrs)
        attrs_sorted = ' '.join(sorted(attrs_list))
        return f"<{tag} {attrs_sorted}>"

    html = re.sub(r'<(\w+)\s+([^>]+)>', sort_attrs, html)

    return html

def validate_dom_parts(dom_parts: dict, verbose: bool = True, context_expected=None) -> bool:
    """
    Robust validation for dom_parts structure.
    - Checks for expected keys, types, required fields, value formats, allowed values, cross-field consistency.
    - Uses STATE_ABBR for state normalization.
    - Suppresses redundant warnings, adapts to context.
    - Returns True if valid, False otherwise.
    """
    MAX_WARNINGS = 20
    warning_count = 0
    valid = True

    expected_keys = [
        "contests", "panels", "tables", "candidate_panels", "location_panels",
        "headings", "ballot_types", "results_timestamps", "party_labels", "vote_methods",
        "pattern_kb_matches", "segments_needing_review", "selector_log", "metadata",
        "tagged_segments", "tagged_segments_with_attrs", "raw_html", "error", "url"
    ]
    required_keys = ["contests", "panels", "tables", "candidate_panels", "location_panels"]
    if context_expected is not None:
        required_keys = [k for k in required_keys if k in context_expected]

    section_fields = {
        "contests": ["title", "year", "type_", "state", "county", "segment_hash"],
        "panels": ["panel_text", "panel_html", "segment_hash"],
        "tables": ["table_text", "table_html", "year", "type_", "segment_hash"],
        "candidate_panels": ["candidate_panel_text", "candidate_panel_html", "year", "type_", "segment_hash"],
        "location_panels": ["location_panel_text", "location_panel_html", "year", "type_", "segment_hash", "county"],
        "headings": ["heading_text", "heading_html", "segment_hash", "heading_type"],
        "ballot_types": ["ballot_types_text", "ballot_types_html", "year", "type_", "segment_hash"],
        "results_timestamps": ["timestamp_text", "timestamp_html", "segment_hash"],
        "party_labels": ["party_label_text", "party_label_html", "segment_hash"],
        "vote_methods": ["vote_method_text", "vote_method_html", "segment_hash"],
    }

    for key in expected_keys:
        if key not in dom_parts:
            if verbose and (context_expected is None or key in context_expected):
                msg = f"[DOM_PARTS] Missing key: {key}"
                if logger.mode == "cli":
                    console.print(msg)
                else:
                    payload = {
                        "level": "WARNING",
                        "type": "dom_parts",
                        "message": msg
                    }
                    logger.warning(payload)
            valid = False

    for key in required_keys:
        val = dom_parts.get(key)
        if not isinstance(val, list):
            if verbose:
                msg = f"[DOM_PARTS] Key '{key}' is not a list."
                if logger.mode == "cli":
                    console.print(msg)
                else:
                    payload = {
                        "level": "WARNING",
                        "type": "dom_parts",
                        "message": msg
                    }
                    logger.warning(payload)
            valid = False
        elif len(val) == 0:
            if verbose:
                msg = f"[DOM_PARTS] No items found in '{key}'."
                if logger.mode == "cli":
                    console.print(msg)
                else:
                    payload = {
                        "level": "WARNING",
                        "type": "dom_parts",
                        "message": msg
                    }
                    logger.warning(payload)
            valid = False

    for section, fields in section_fields.items():
        items = dom_parts.get(section, [])
        if not isinstance(items, list):
            continue
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                if verbose and warning_count < MAX_WARNINGS:
                    msg = f"[DOM_PARTS] Item {i} in '{section}' is not a dict."
                    if logger.mode == "cli":
                        console.print(msg)
                    else:
                        payload = {
                            "level": "WARNING",
                            "type": "dom_parts",
                            "message": msg
                        }
                        logger.warning(payload)
                warning_count += 1
                continue
            for field in fields:
                value = item.get(field)
                if value is None or (isinstance(value, str) and not value.strip()):
                    if verbose and warning_count < MAX_WARNINGS and (context_expected is None or section in context_expected):
                        msg = f"[DOM_PARTS] Item {i} in '{section}' missing or empty field '{field}'."
                        if logger.mode == "cli":
                            console.print(msg)
                        else:
                            payload = {
                                "level": "WARNING",
                                "type": "dom_parts",
                                "message": msg
                            }
                            logger.warning(payload)
                    warning_count += 1
                if field.endswith("_html") and value and not isinstance(value, str):
                    if verbose:
                        msg = f"[DOM_PARTS] Item {i} in '{section}' field '{field}' should be str (HTML)."
                        if logger.mode == "cli":
                            console.print(msg)
                        else:
                            payload = {
                                "level": "WARNING",
                                "type": "dom_parts",
                                "message": msg
                            }
                            logger.warning(payload)
                    valid = False
                if field.endswith("_text") and value and not isinstance(value, str):
                    if verbose:
                        msg = f"[DOM_PARTS] Item {i} in '{section}' field '{field}' should be str (text)."
                        if logger.mode == "cli":
                            console.print(msg)
                        else:
                            payload = {
                                "level": "WARNING",
                                "type": "dom_parts",
                                "message": msg
                            }
                            logger.warning(payload)
                    valid = False
                if field == "year" and value:
                    if not re.fullmatch(r"20\d{2}", str(value)):
                        if verbose:
                            msg = f"[DOM_PARTS] Item {i} in '{section}' has invalid year format: {value}"
                            if logger.mode == "cli":
                                console.print(msg)
                            else:
                                payload = {
                                    "level": "WARNING",
                                    "type": "dom_parts",
                                    "message": msg
                                }
                                logger.warning(payload)
                        valid = False
                    else:
                        year_int = int(value)
                        if year_int < 2000 or year_int > datetime.datetime.now().year + 1:
                            if verbose:
                                msg = f"[DOM_PARTS] Item {i} in '{section}' has out-of-range year: {value}"
                                if logger.mode == "cli":
                                    console.print(msg)
                                else:
                                    payload = {
                                        "level": "WARNING",
                                        "type": "dom_parts",
                                        "message": msg
                                    }
                                    logger.warning(payload)
                            valid = False
                if field == "type_" and value:
                    if safe_lower(value) not in {safe_lower(t or "") for t in ELECTION_TYPES}:
                        if verbose:
                            msg = f"[DOM_PARTS] Item {i} in '{section}' has unknown election type: {value}"
                            if logger.mode == "cli":
                                console.print(msg)
                            else:
                                payload = {
                                    "level": "WARNING",
                                    "type": "dom_parts",
                                    "message": msg
                                }
                                logger.warning(payload)
                        valid = False
                if field == "county" and value and "state" in item:
                    state_val = safe_lower(item.get("state") or "")
                    state_val = STATE_ABBR.get(state_val, state_val)
                    if state_val and safe_lower(value) not in {safe_lower(c) for c in KNOWN_STATE_TO_COUNTY_MAP.get(state_val, [])}:
                        if verbose:
                            msg = f"[DOM_PARTS] Item {i} in '{section}' has unknown county '{value}' for state '{state_val}'"
                            if logger.mode == "cli":
                                console.print(msg)
                            else:
                                payload = {
                                    "level": "WARNING",
                                    "type": "dom_parts",
                                    "message": msg
                                }
                                logger.warning(payload)
                        valid = False
                if field == "state" and value:
                    state_norm = STATE_ABBR.get(safe_lower(value), safe_lower(value))
                    if state_norm not in KNOWN_STATE_TO_COUNTY_MAP:
                        if verbose:
                            msg = f"[DOM_PARTS] Item {i} in '{section}' has unknown state: {value}"
                            if logger.mode == "cli":
                                console.print(msg)
                            else:
                                payload = {
                                    "level": "WARNING",
                                    "type": "dom_parts",
                                    "message": msg
                                }
                                logger.warning(payload)
                        valid = False
                if field == "timestamp_text" and value:
                    if not re.search(r"\d{4}.*\d{1,2}:\d{2}", value):
                        if verbose:
                            msg = f"[DOM_PARTS] Item {i} in '{section}' field '{field}' does not look like a timestamp: {value}"
                            if logger.mode == "cli":
                                console.print(msg)
                            else:
                                payload = {
                                    "level": "WARNING",
                                    "type": "dom_parts",
                                    "message": msg
                                }
                                logger.warning(payload)
                        valid = False
                if section == "ballot_types" and field == "ballot_types_text" and value:
                    if safe_lower(value) not in {safe_lower(bt) for bt in BALLOT_TYPES}:
                        if verbose:
                            msg = f"[DOM_PARTS] Item {i} in '{section}' has unknown ballot type: {value}"
                            if logger.mode == "cli":
                                console.print(msg)
                            else:
                                payload = {
                                    "level": "WARNING",
                                    "type": "dom_parts",
                                    "message": msg
                                }
                                logger.warning(payload)
                        valid = False
                if section == "party_labels" and field == "party_label_text" and value:
                    if safe_lower(value) not in {safe_lower(k) for k in PARTY_KEYWORDS}:
                        if verbose:
                            msg = f"[DOM_PARTS] Item {i} in '{section}' has unknown party label: {value}"
                            if logger.mode == "cli":
                                console.print(msg)
                            else:
                                payload = {
                                    "level": "WARNING",
                                    "type": "dom_parts",
                                    "message": msg
                                }
                                logger.warning(payload)
                        valid = False
                if section == "location_panels" and field == "location_panel_text" and value:
                    if not any(safe_lower(kw) in safe_lower(value) for kw in LOCATION_KEYWORDS):
                        if verbose:
                            msg = f"[DOM_PARTS] Item {i} in '{section}' has location text missing known keywords: {value}"
                            if logger.mode == "cli":
                                console.print(msg)
                            else:
                                payload = {
                                    "level": "WARNING",
                                    "type": "dom_parts",
                                    "message": msg
                                }
                                logger.warning(payload)
                        valid = False
                    county_val = safe_lower(item.get("county", "") or "")
                    for abbrev, full_names in LOCATION_ABBREVIATIONS.items():
                        if safe_lower(abbrev) in safe_lower(value):
                            for full_name in full_names:
                                if safe_lower(full_name) in safe_lower(value) and county_val in KNOWN_COUNTY_TO_PRECINCTS_MAP:
                                    precincts = KNOWN_COUNTY_TO_PRECINCTS_MAP[county_val]
                                    found = any(safe_lower(p) in safe_lower(value) for p in precincts)
                                    if not found:
                                        if verbose:
                                            msg = f"[DOM_PARTS] Location panel {i}: '{value}' does not match any known precinct/district for county '{county_val}'."
                                            if logger.mode == "cli":
                                                console.print(msg)
                                            else:
                                                payload = {
                                                    "level": "WARNING",
                                                    "type": "dom_parts",
                                                    "message": msg
                                                }
                                                logger.warning(payload)
                                        valid = False
                if section == "headings" and field == "heading_text" and value:
                    canonical = CANONICAL_SEGMENT_LABELS.get(safe_lower(value))
                    if canonical and canonical != "heading":
                        if verbose:
                            msg = f"[DOM_PARTS] Heading {i}: text '{value}' has canonical label '{canonical}' not 'heading'."
                            if logger.mode == "cli":
                                console.print(msg)
                            else:
                                payload = {
                                    "level": "WARNING",
                                    "type": "dom_parts",
                                    "message": msg
                                }
                                logger.warning(payload)
                        valid = False
                if section == "panels" and field == "panel_text" and value:
                    canonical = CANONICAL_SEGMENT_LABELS.get(safe_lower(value))
                    if canonical and canonical != "panel":
                        if verbose:
                            msg = f"[DOM_PARTS] Panel {i}: text '{value}' has canonical label '{canonical}' not 'panel'."
                            if logger.mode == "cli":
                                console.print(msg)
                            else:
                                payload = {
                                    "level": "WARNING",
                                    "type": "dom_parts",
                                    "message": msg
                                }
                                logger.warning(payload)
                        valid = False
                if section == "headings" and "heading_html" in item:
                    tag_match = any(safe_lower(tag) in safe_lower(item["heading_html"] or "") for tag in HEADING_TAGS | EXTRA_HEADING_TAGS)
                    if not tag_match:
                        if verbose:
                            msg = f"[DOM_PARTS] Heading {i}: html '{item['heading_html']}' does not contain a valid heading tag."
                            if logger.mode == "cli":
                                console.print(msg)
                            else:
                                payload = {
                                    "level": "WARNING",
                                    "type": "dom_parts",
                                    "message": msg
                                }
                                logger.warning(payload)
                        valid = False
                if section == "panels" and "panel_html" in item:
                    tag_match = any(safe_lower(tag) in safe_lower(item["panel_html"] or "") for tag in PANEL_TAGS)
                    if not tag_match:
                        if verbose:
                            msg = f"[DOM_PARTS] Panel {i}: html '{item['panel_html']}' does not contain a valid panel tag."
                            if logger.mode == "cli":
                                console.print(msg)
                            else:
                                payload = {
                                    "level": "WARNING",
                                    "type": "dom_parts",
                                    "message": msg
                                }
                                logger.warning(payload)
                        valid = False
    if warning_count > MAX_WARNINGS:
        msg = f"[DOM_PARTS] {warning_count} items missing required fields (warnings suppressed after {MAX_WARNINGS})."
        if logger.mode == "cli":
            console.print(msg)
        else:
            payload = {
                "level": "WARNING",
                "type": "dom_parts",
                "message": msg
            }
            logger.warning(payload)
    meta = dom_parts.get("metadata", {})
    if not isinstance(meta, dict):
        if verbose:
            msg = "[DOM_PARTS] 'metadata' is not a dict."
            if logger.mode == "cli":
                console.print(msg)
            else:
                payload = {
                    "level": "WARNING",
                    "type": "dom_parts",
                    "message": msg
                }
                logger.warning(payload)
        valid = False
    else:
        scrape_time = meta.get("scrape_time")
        if scrape_time:
            try:
                datetime.datetime.strptime(scrape_time, "%Y-%m-%d %H:%M:%S")
            except Exception:
                if verbose:
                    msg = f"[DOM_PARTS] metadata.scrape_time has invalid format: {scrape_time}"
                    if logger.mode == "cli":
                        console.print(msg)
                    else:
                        payload = {
                            "level": "WARNING",
                            "type": "dom_parts",
                            "message": msg
                        }
                        logger.warning(payload)
                valid = False

    if "selector_log" in dom_parts and not isinstance(dom_parts["selector_log"], list):
        if verbose:
            msg = "[DOM_PARTS] 'selector_log' is not a list."
            if logger.mode == "cli":
                console.print(msg)
            else:
                payload = {
                    "level": "WARNING",
                    "type": "dom_parts",
                    "message": msg
                }
                logger.warning(payload)
        valid = False

    for key in ["tagged_segments", "tagged_segments_with_attrs"]:
        if key in dom_parts and not isinstance(dom_parts[key], list):
            if verbose:
                msg = f"[DOM_PARTS] '{key}' is not a list."
                if logger.mode == "cli":
                    console.print(msg)
                else:
                    payload = {
                        "level": "WARNING",
                        "type": "dom_parts",
                        "message": msg
                    }
                    logger.warning(payload)
            valid = False

    if "url" in dom_parts and dom_parts["url"] is not None and not isinstance(dom_parts["url"], str):
        if verbose:
            msg = "[DOM_PARTS] 'url' is not a string."
            if logger.mode == "cli":
                console.print(msg)
            else:
                payload = {
                    "level": "WARNING",
                    "type": "dom_parts",
                    "message": msg
                }
                logger.warning(payload)
        valid = False

    if "raw_html" in dom_parts and dom_parts["raw_html"] is not None and not isinstance(dom_parts["raw_html"], str):
        if verbose:
            msg = "[DOM_PARTS] 'raw_html' is not a string."
            if logger.mode == "cli":
                console.print(msg)
            else:
                payload = {
                    "level": "WARNING",
                    "type": "dom_parts",
                    "message": msg
                }
                logger.warning(payload)
        valid = False

    if "error" in dom_parts and dom_parts["error"] is not None and not isinstance(dom_parts["error"], str):
        if verbose:
            msg = "[DOM_PARTS] 'error' is not a string or None."
            if logger.mode == "cli":
                console.print(msg)
            else:
                payload = {
                    "level": "WARNING",
                    "type": "dom_parts",
                    "message": msg
                }
                logger.warning(payload)
        valid = False

    if not valid and verbose:
        msg = "[DOM_PARTS] Validation failed. Downstream consumers may not function correctly."
        if logger.mode == "cli":
            console.print(msg)
        else:
            payload = {
                "level": "ERROR",
                "type": "dom_parts",
                "message": msg
            }
            logger.error(payload)

    return valid

def scan_html_for_context(
    target_url,
    page,
    coordinator=None,
    session_id=None,
    allow_duplicates=False,
    context_cache=None,
    debug=False,
    model_name: Optional[str] = None,
    use_finetuned: bool = True,
    ml_threshold: float = SEGMENT_ML_LABEL_THRESHOLD_STRICT,
    **kwargs
) -> Dict[str, Any]:
    """
    Main pipeline entry: Efficient, dynamic, and feedback-driven HTML scanner.
    Organizes all logic into clear pipeline stages for maintainability and extensibility.
    Implements robust feedback, pattern KB, context library update, election type extraction,
    semantic tags, selector log, debug logging, and list field safety.
    Mode-aware logging for CLI and non-CLI environments.
    """
    from ..Context_Integration.context_coordinator import ContextCoordinator
    coordinator = coordinator or ContextCoordinator()

    # --- 1. Load context library, pattern KB, and ML model ---
    context_library, pattern_kb, model = _load_context_resources(coordinator, model_name, use_finetuned)

    # --- 2. Get HTML and check cache ---
    html, page_hash, page_url, context_cache = _prepare_html_and_cache(page, target_url, context_cache)
    if _fast_path_cache_hit(html, page_hash, page_url, context_cache, coordinator):
        return context_cache[page_hash]

    # --- 3. Segment extraction and labeling ---
    segments_with_attrs = extract_tagged_segments_with_attrs(
        html,
        context_library=context_library,
        context_cache=context_cache,
        include_data_attrs=True,
        fallback_on_error=True,
        model_name=model_name,
        use_finetuned=use_finetuned,
        pattern_kb=pattern_kb,
        ml_threshold=ml_threshold,
        model=model,
        coordinator=coordinator,
        **kwargs 
    )

    # --- 4. Organize and filter segments by type ---
    context_result = _organize_segments_and_sections(
        segments_with_attrs, target_url, context_library, coordinator, allow_duplicates, session_id, **kwargs
    )

    # --- 4b. Optional OCR ingestion (text-only) ---
    ocr_text = kwargs.get("ocr_text")
    ocr_path = kwargs.get("ocr_path") or kwargs.get("ocr_text_path")
    if not ocr_text and isinstance(ocr_path, str) and ocr_path:
        try:
            if os.path.exists(ocr_path):
                with open(ocr_path, "r", encoding="utf-8", errors="replace") as f:
                    ocr_text = f.read()
        except Exception:
            ocr_text = None
    if ocr_text:
        try:
            ocr_context = coordinator.ingest_ocr_text(ocr_text, source=ocr_path or "ocr_text")
            if ocr_context:
                context_result["ocr_context"] = ocr_context
        except Exception:
            pass

    # --- 4a. Election Types Extraction from ballot_types ---
    election_types = []
    for seg in _extract_segments_by_label(segments_with_attrs, "ballot_types"):
        ballot_types_text = _extract_clean_text(safe_get(seg, "raw_html", safe_get(seg, "html", "")))
        if ballot_types_text:
            etype = None
            if coordinator and hasattr(coordinator, "extract_field"):
                etype = coordinator.extract_field("election_types", text=ballot_types_text)
            if etype:
                election_types.append(etype)
    context_result["election_types"] = election_types if election_types else []

    # --- 5. Pattern KB and Feedback Log Integration, Semantic Tags, Selector Log ---
    pattern_kb_matches = []
    segments_needing_review = []
    selector_log = set()
    for seg in segments_with_attrs:
        # Selector log
        if safe_get(seg, "id", None):
            selector_log.add(f'#{safe_get(seg, "id", "")}')
        for cls in safe_get(seg, "classes", []):
            selector_log.add(f'.{cls}')
        selector_log.add(safe_lower(safe_get(seg, "tag", "")))
        # Semantic tags
        if "semantic_tags" not in seg:
            seg["semantic_tags"] = []
        if safe_get(seg, "ml_label", "") not in ("unknown", "ignore"):
            safe_append(seg["semantic_tags"], safe_get(seg, "ml_label", ""), logger)
        # Pattern KB/feedback logic
        if safe_get(seg, "ml_confidence", 0.0) < 0.7 or safe_get(seg, "ml_label", "unknown") == "unknown":
            html_val = safe_get(seg, "html", "")
            if not isinstance(html_val, str):
                html_val = str(html_val)
            seg["pattern_id"] = f"pattern_{hashlib.sha256(html_val.encode('utf-8')).hexdigest()[:10]}"
            emb = get_segment_embedding(model, seg)
            if emb is not None:
                emb = emb.tolist()
            kb_entry = {
                "pattern_id": seg["pattern_id"],
                "label": safe_get(seg, "ml_label", "unknown"),
                "embedding": emb,
                "example_html": html_val[:500],
                "source_url": page_url,
                "timestamp": time.time(),
            }
            append_pattern_kb(kb_entry)
            append_feedback_log({
                "pattern_id": seg["pattern_id"],
                "label": safe_get(seg, "ml_label", "unknown"),
                "html": html_val[:500],
                "source_url": page_url,
                "timestamp": time.time(),
            })
            # Raw parser segments are evidence, not canonical context.
            # Keep them in the review result and evidence logs only.
            if safe_get(seg, "segment_hash", None):
                seg.setdefault("review_metadata", {})
                seg["review_metadata"].update({
                    "status": "pending_review",
                    "source": "html_scanner",
                    "source_url": page_url,
                    "segment_hash": safe_get(seg, "segment_hash", None),
                })
            segments_needing_review.append(seg)
        else:
            pattern_kb_matches.append({
                "pattern_id": safe_get(seg, "pattern_id", None),
                "label": safe_get(seg, "ml_label", None),
                "confidence": safe_get(seg, "ml_confidence", None),
                "segment_html": safe_get(seg, "html", "")[:200],
            })
    context_result["pattern_kb_matches"] = pattern_kb_matches
    context_result["segments_needing_review"] = segments_needing_review
    context_result["selector_log"] = sorted(selector_log)

    # --- 6. Debug Logging for Extraction ---
    if debug:
        msg_debug = "\n[orange][DEBUG] Extracted HTML segments with ML labels:[/orange]"
        if logger.mode == "cli":
            console.print(msg_debug)
        else:
            logger.debug({"level": "DEBUG", "type": "scan_html", "message": msg_debug})
        for seg in segments_with_attrs:
            info_msg = (
                f"{safe_get(seg, 'tag', '')} {safe_get(seg, 'attrs', {})} "
                f"[label={safe_get(seg, 'ml_label', '')}, conf={safe_get(seg, 'ml_confidence', 0.0):.2f}] "
                f"{safe_get(seg, 'html', '')[:80]}{'...' if len(safe_get(seg, 'html', '')) > 80 else ''}"
            )
            if logger.mode == "cli":
                console.print(info_msg)
            else:
                logger.info({"level": "INFO", "type": "scan_html", "message": info_msg})
        if segments_needing_review:
            msg_review = f"\n[red][DEBUG] {len(segments_needing_review)} segments flagged for review.[/red]"
            if logger.mode == "cli":
                console.print(msg_review)
            else:
                logger.debug({"level": "DEBUG", "type": "scan_html", "message": msg_review})

    # --- 7. Ensure All List Fields Are Lists ---
    for key in [
        "contests",
        "panels",
        "tables",
        "candidate_panels",
        "location_panels",
        "headings",
        "ballot_types",
        "results_timestamps",
        "party_labels",
        "vote_methods",
        "pattern_kb_matches",
        "segments_needing_review",
        "segment_evidence",
        "selector_log",
        "tagged_segments",
        "tagged_segments_with_attrs",
    ]:
        if key not in context_result or not isinstance(context_result[key], list):
            context_result[key] = []

    # --- 8. Build Reviewable Segment Evidence ---
    segment_evidence = []

    for seg in segments_with_attrs:
        segment_hash = safe_get(seg, "segment_hash", None)
        if not segment_hash:
            continue

        segment_evidence.append({
            "type": "segment_observation",
            "status": "pending_review",
            "source": "html_scanner",
            "value": safe_get(seg, "html", "")[:500],
            "canonical_value": None,
            "confidence": safe_get(seg, "ml_confidence", None),
            "label": safe_get(seg, "ml_label", None),
            "pattern_id": safe_get(seg, "pattern_id", None),
            "jurisdiction": {
                "state": context_result.get("state"),
                "county": context_result.get("county"),
            },
            "provenance": {
                "session_id": session_id,
                "source_url": page_url,
                "segment_hash": segment_hash,
            },
        })

    context_result["segment_evidence"] = segment_evidence

    # --- 9. Enrich, propagate, and validate context ---
    context_result = _enrich_and_validate_context(
        context_result, page_hash, html, context_cache, coordinator, debug
    )

    # --- 10. Context digest (for diagnostics + UI) ---
    digest = _build_context_digest(context_result, page_url, selector_log)
    context_result["context_digest"] = digest
    emit_func = kwargs.get("emit_func")
    _write_context_digest(digest, session_id)
    if callable(emit_func) and session_id:
        try:
            emit_func({
                "type": "context_digest",
                "session_id": session_id,
                "digest": digest,
                "timestamp": time.time(),
            })
        except Exception:
            pass
    
    # --- 11. Integrity signal (trend deltas for ML/NLP drift alerts) ---
    if callable(emit_func) and session_id:
        try:
            # Lazily import analyzer to avoid circular imports and heavy startup cost
            from tools.analyze_context_digest_trends import compute_integrity_signal
            
            signal = compute_integrity_signal(
                trend_file="tools/debug_headless_output/context_digest_trends.json",
                window=30,
                recent=5,
                conf_drop_threshold=0.08,
                unknown_spike_threshold=0.10,
                review_spike_threshold=5.0,
            )
            emit_func({
                "type": "integrity_signal",
                "session_id": session_id,
                "signal": signal,
                "timestamp": time.time(),
            })
        except Exception:
            # Don't fail the pipeline if integrity signal computation fails
            pass

    return context_result

# --- Helper functions for each pipeline stage ---

def _load_context_resources(coordinator, model_name, use_finetuned):
    """Load context library, pattern KB, and ML model using coordinator methods/properties."""
    from ..Context_Integration.context_coordinator import ContextCoordinator
    coordinator = coordinator or ContextCoordinator()
    context_library = None
    pattern_kb = None
    model = None

    if coordinator:
        # Use properties and methods directly
        try:
            context_library = coordinator.library if hasattr(coordinator, "library") else None
        except Exception as e:
            msg = f"[CONTEXT] Failed to load coordinator.library: {e}"
            if logger.mode == "cli":
                console.print(msg)
            else:
                logger.warning({"level": "WARNING", "type": "context", "message": msg})
            context_library = None
        try:
            # Prefer method if available, else property
            if hasattr(coordinator, "get_feedback_pattern_kb"):
                feedback_kb = coordinator.get_feedback_pattern_kb()
                pattern_kb = feedback_kb if feedback_kb else []
            elif hasattr(coordinator, "pattern_kb"):
                pattern_kb = coordinator.pattern_kb
            else:
                pattern_kb = []
        except Exception as e:
            msg = f"[CONTEXT] Failed to load coordinator.pattern_kb: {e}"
            if logger.mode == "cli":
                console.print(msg)
            else:
                logger.warning({"level": "WARNING", "type": "context", "message": msg})
            pattern_kb = []
        try:
            model = coordinator._semantic_model if hasattr(coordinator, "_semantic_model") else None
        except Exception as e:
            msg = f"[CONTEXT] Failed to load coordinator._semantic_model: {e}"
            if logger.mode == "cli":
                console.print(msg)
            else:
                logger.warning({"level": "WARNING", "type": "context", "message": msg})
            model = None

        # Defensive: fallback to loading if any are still None
        if context_library is None:
            try:
                context_library = load_context_library(CONTEXT_LIBRARY_PATH)
            except Exception as e:
                msg = f"[CONTEXT] Fallback load_context_library failed: {e}"
                if logger.mode == "cli":
                    console.print(msg)
                else:
                    logger.warning({"level": "WARNING", "type": "context", "message": msg})
                context_library = {}
        if not pattern_kb:
            try:
                pattern_kb = load_pattern_kb()
            except Exception as e:
                msg = f"[CONTEXT] Fallback load_pattern_kb failed: {e}"
                if logger.mode == "cli":
                    console.print(msg)
                else:
                    logger.warning({"level": "WARNING", "type": "context", "message": msg})
                pattern_kb = []
        if model is None:
            try:
                model = ModelRegistry.get_sentence_transformer(model_name=model_name, use_finetuned=use_finetuned)
            except Exception as e:
                msg = f"[CONTEXT] Fallback ModelRegistry failed: {e}"
                if logger.mode == "cli":
                    console.print(msg)
                else:
                    logger.warning({"level": "WARNING", "type": "context", "message": msg})
                model = None

        # Deduplicate pattern_kb if needed
        if pattern_kb:
            pattern_kb = deduplicate_pattern_kb(pattern_kb)
    else:
        context_library = load_context_library(CONTEXT_LIBRARY_PATH)
        pattern_kb = load_pattern_kb()
        model = ModelRegistry.get_sentence_transformer(model_name=model_name, use_finetuned=use_finetuned)

    return context_library, pattern_kb, model

def _prepare_html_and_cache(page, target_url, context_cache):
    """Extract HTML, compute hash, and check cache. Mode-aware logging."""
    try:
        html = getattr(page, "content", lambda: "")()
    except Exception as e:
        msg = f"[SCAN_HTML] Exception when calling page.content(): {e}. Using empty string."
        if logger.mode == "cli":
            console.print(msg)
        else:
            logger.warning({"level": "WARNING", "type": "scan_html", "message": msg})
        html = ""
    if html is None:
        html = ""
    page_hash = get_page_hash(page)
    page_url = safe_get(page, "url", None) or target_url
    if context_cache is None:
        context_cache = load_context_cache_from_disk()
    return html, page_hash, page_url, context_cache

def _fast_path_cache_hit(html, page_hash, page_url, context_cache, coordinator):
    """Check if all segments are already cached with high confidence. Mode-aware logging."""
    from ..Context_Integration.context_coordinator import ContextCoordinator
    coordinator = coordinator or ContextCoordinator()
    segment_htmls = [n.html for n in HTMLParser(html).root.traverse() if hasattr(n, "html")]
    segment_hashes = [segment_hash(h) for h in segment_htmls]
    fast_path_hits = [
        h for h in segment_hashes
        if h in context_cache and safe_get(context_cache[h], "ml_confidence", 0) > 0.95
    ]
    if len(fast_path_hits) == len(segment_hashes) and segment_hashes:
        msg = "[FAST-PATH] All segments covered by cache. Skipping full scan."
        if logger.mode == "cli":
            console.print(msg)
        else:
            logger.info({"level": "INFO", "type": "scan_html", "message": msg})
        fast_path_result = {h: context_cache[h] for h in segment_hashes}
        if coordinator is not None:
            coordinator.organize_and_enrich(
                fast_path_result,
                write_kind=ContextWriteKind.NONE,
            )
        return True
    if page_hash in context_cache:
        msg1 = f"[SCAN] Using cached context for {page_url}"
        msg2 = "[bold green][CACHE] Entire context loaded from cache. Skipping scan.[/bold green]"
        if logger.mode == "cli":
            console.print(msg1)
            console.print(msg2)
        else:
            logger.info({"level": "INFO", "type": "scan_html", "message": msg1})
            logger.info({"level": "INFO", "type": "scan_html", "message": msg2})
        cached_result = context_cache[page_hash]
        if coordinator is not None:
            coordinator.organize_and_enrich(
                cached_result,
                write_kind=ContextWriteKind.NONE,
            )
        return True
    return False

def _organize_segments_and_sections(
    segments_with_attrs,
    target_url,
    context_library,
    coordinator,
    allow_duplicates,
    session_id,
    **kwargs
) -> Dict[str, Any]:
    """
    Organize segments into sections (contests, panels, tables, etc.) and filter.
    Uses robust filtering, deduplication, context enrichment, and context_library-aware logic.
    Passes **kwargs to all helpers for future extensibility.
    Mode-aware logging for diagnostics.
    """
    # Helper for diagnostics and filtering
    def diagnostics_and_filter(
        data, field, **local_kwargs
    ) -> List[Dict[str, Any]]:
        # Merge local_kwargs with outer kwargs and always pass context_library
        merged_kwargs = {**kwargs, **local_kwargs, "context_library": context_library}
        if "diagnostics_and_filter" in globals():
            return globals()["diagnostics_and_filter"](
                data,
                field,
                allow_duplicates=allow_duplicates,
                session_id=session_id,
                coordinator=coordinator,
                **merged_kwargs
            )
        return data

    # --- Contests ---
    contests = []
    for seg in _extract_segments_by_label(segments_with_attrs, "contest", context_library=context_library, **kwargs):
        for possible in split_possible_contests(safe_get(seg, "text", "")):
            seg_year, seg_type, cleaned_title, _ = extract_year_and_type(possible, url=target_url)
            if cleaned_title and not any(
                safe_get(c, "title", "") == cleaned_title and safe_get(c, "year", None) == seg_year and safe_get(c, "type_", None) == seg_type
                for c in contests
            ):
                contests.append({
                    "title": cleaned_title,
                    "year": seg_year,
                    "type_": seg_type,
                    "segment_hash": safe_get(seg, "segment_hash", None),
                })
    contests = [c for c in contests if safe_get(c, "title", None)]
    contests = diagnostics_and_filter(contests, ["title", "year", "type_"])

    # --- Panels ---
    panels = []
    for seg in _extract_segments_by_label(segments_with_attrs, "panel", context_library=context_library, **kwargs):
        panel_text = _extract_clean_text(safe_get(seg, "raw_html", safe_get(seg, "html", "")))
        if panel_text:
            panels.append({
                "panel_text": panel_text,
                "panel_html": safe_get(seg, "raw_html", safe_get(seg, "html", "")),
                "segment_hash": safe_get(seg, "segment_hash", None),
            })
    panels = diagnostics_and_filter(panels, "panel_text")

    # --- Tables ---
    tables = []
    for seg in _extract_segments_by_label(segments_with_attrs, "results_table", context_library=context_library, **kwargs):
        table_text = _extract_clean_text(safe_get(seg, "raw_html", safe_get(seg, "html", "")))
        if table_text:
            tables.append({
                "table_text": table_text,
                "table_html": safe_get(seg, "raw_html", safe_get(seg, "html", "")),
                "year": None,
                "type_": None,
                "segment_hash": safe_get(seg, "segment_hash", None),
            })
    tables = diagnostics_and_filter(tables, "table_text")

    # --- Candidate Panels ---
    candidate_panels = []
    for seg in _extract_segments_by_label(segments_with_attrs, "candidate_panel", context_library=context_library, **kwargs):
        candidate_panel_text = _extract_clean_text(safe_get(seg, "raw_html", safe_get(seg, "html", "")))
        if candidate_panel_text:
            candidate_panels.append({
                "candidate_panel_text": candidate_panel_text,
                "candidate_panel_html": safe_get(seg, "raw_html", safe_get(seg, "html", "")),
                "year": None,
                "type_": None,
                "segment_hash": safe_get(seg, "segment_hash", None),
            })
    candidate_panels = diagnostics_and_filter(candidate_panels, "candidate_panel_text")

    # --- Location Panels ---
    location_panels = []
    for seg in _extract_segments_by_label(segments_with_attrs, "location_panel", context_library=context_library, **kwargs):
        location_panel_text = _extract_clean_text(safe_get(seg, "raw_html", safe_get(seg, "html", "")))
        if location_panel_text:
            location_panels.append({
                "location_panel_text": location_panel_text,
                "location_panel_html": safe_get(seg, "raw_html", safe_get(seg, "html", "")),
                "year": None,
                "type_": None,
                "segment_hash": safe_get(seg, "segment_hash", None),
                "county": None,
            })
    location_panels = diagnostics_and_filter(location_panels, "location_panel_text")

    # --- Headings ---
    headings = []
    for seg in _extract_segments_by_label(segments_with_attrs, "heading", context_library=context_library, **kwargs):
        heading_text = _extract_clean_text(safe_get(seg, "raw_html", safe_get(seg, "html", "")))
        if heading_text:
            headings.append({
                "heading_text": heading_text,
                "heading_html": safe_get(seg, "raw_html", safe_get(seg, "html", "")),
                "segment_hash": safe_get(seg, "segment_hash", None),
                "heading_type": None,
            })
    headings = diagnostics_and_filter(headings, "heading_text")

    # --- Ballot Types ---
    ballot_types = []
    for seg in _extract_segments_by_label(segments_with_attrs, "ballot_types", context_library=context_library, **kwargs):
        ballot_types_text = _extract_clean_text(safe_get(seg, "raw_html", safe_get(seg, "html", "")))
        if ballot_types_text:
            ballot_types.append({
                "ballot_types_text": ballot_types_text,
                "ballot_types_html": safe_get(seg, "raw_html", safe_get(seg, "html", "")),
                "year": None,
                "type_": None,
                "segment_hash": safe_get(seg, "segment_hash", None),
            })
    ballot_types = diagnostics_and_filter(ballot_types, "ballot_types_text")

    # --- Results Timestamps ---
    results_timestamps = []
    for seg in _extract_segments_by_label(segments_with_attrs, "results_timestamp", context_library=context_library, **kwargs):
        timestamp_text = _extract_clean_text(safe_get(seg, "raw_html", safe_get(seg, "html", "")))
        if timestamp_text:
            results_timestamps.append({
                "timestamp_text": timestamp_text,
                "timestamp_html": safe_get(seg, "raw_html", safe_get(seg, "html", "")),
                "segment_hash": safe_get(seg, "segment_hash", None),
            })
    results_timestamps = diagnostics_and_filter(results_timestamps, "timestamp_text")

    # --- Party Labels ---
    party_labels = []
    for seg in _extract_segments_by_label(segments_with_attrs, "party_label", context_library=context_library, **kwargs):
        party_label_text = _extract_clean_text(safe_get(seg, "raw_html", safe_get(seg, "html", "")))
        if party_label_text:
            party_labels.append({
                "party_label_text": party_label_text,
                "party_label_html": safe_get(seg, "raw_html", safe_get(seg, "html", "")),
                "segment_hash": safe_get(seg, "segment_hash", None),
            })
    party_labels = diagnostics_and_filter(party_labels, "party_label_text")

    # --- Vote Methods ---
    vote_methods = []
    for seg in _extract_segments_by_label(segments_with_attrs, "vote_method", context_library=context_library, **kwargs):
        vote_method_text = _extract_clean_text(safe_get(seg, "raw_html", safe_get(seg, "html", "")))
        if vote_method_text:
            vote_methods.append({
                "vote_method_text": vote_method_text,
                "vote_method_html": safe_get(seg, "raw_html", safe_get(seg, "html", "")),
                "segment_hash": safe_get(seg, "segment_hash", None),
            })
    vote_methods = diagnostics_and_filter(vote_methods, "vote_method_text")

    # --- Pattern KB Matches and Segments Needing Review ---
    pattern_kb_matches = []
    segments_needing_review = []
    for seg in segments_with_attrs:
        # Use context_library for additional review logic if needed
        ml_conf = safe_get(seg, "ml_confidence", 0.0)
        ml_label = safe_get(seg, "ml_label", "unknown")
        if ml_conf < 0.7 or ml_label == "unknown":
            # Optionally, context_library can be used here for more advanced review logic
            segments_needing_review.append(seg)
        else:
            pattern_kb_matches.append({
                "pattern_id": safe_get(seg, "pattern_id", None),
                "label": ml_label,
                "confidence": ml_conf,
                "segment_html": safe_get(seg, "html", "")[:200],
            })

    # --- Selector Log and Semantic Tags ---
    selector_log = set()
    for seg in segments_with_attrs:
        if safe_get(seg, "id", None):
            selector_log.add(f'#{safe_get(seg, "id", "")}')
        for cls in safe_get(seg, "classes", []):
            selector_log.add(f'.{cls}')
        selector_log.add(safe_lower(safe_get(seg, "tag", "")))
        # Add semantic tags using context_library if available
        if "semantic_tags" not in seg:
            seg["semantic_tags"] = []
        ml_label = safe_get(seg, "ml_label", "")
        if ml_label not in ("unknown", "ignore"):
            safe_append(seg["semantic_tags"], ml_label, logger)
        # Optionally, add context_library-driven tags
        if context_library and "extra_tags" in context_library:
            for tag in context_library["extra_tags"]:
                if tag not in seg["semantic_tags"]:
                    seg["semantic_tags"].append(tag)

    # --- Metadata ---
    metadata = {
        "source_url": target_url,
        "scrape_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # --- Compose context_result ---
    context_result = {
        "contests": contests,
        "panels": panels,
        "tables": tables,
        "candidate_panels": candidate_panels,
        "location_panels": location_panels,
        "headings": headings,
        "ballot_types": ballot_types,
        "results_timestamps": results_timestamps,
        "party_labels": party_labels,
        "vote_methods": vote_methods,
        "pattern_kb_matches": pattern_kb_matches,
        "segments_needing_review": segments_needing_review,
        "selector_log": sorted(selector_log),
        "metadata": metadata,
        "tagged_segments_with_attrs": segments_with_attrs,
        "tagged_segments": [safe_get(seg, "html", "") for seg in segments_with_attrs],
    }
    return context_result

def _enrich_and_validate_context(
    context_result, page_hash, html, context_cache, coordinator, debug
) -> Dict[str, Any]:
    """
    Propagate year/type, validate, enrich, and save to cache.
    - Propagates year/type to all relevant sections.
    - Validates dom_parts structure and logs issues.
    - Enriches with dom_tree, label_groups, panels_and_tables, and HTML samples.
    - Updates context cache and context library if available.
    - Handles downstream enrichment via coordinator if present.
    - Returns the enriched context_result.
    """
    from ..Context_Integration.context_coordinator import ContextCoordinator
    from ..Context_Integration.context_organizer import ContextOrganizer
    coordinator = coordinator or ContextCoordinator()
    # --- 1. Propagate year/type to all relevant sections ---
    contests = safe_get(context_result, "contests", [])
    best_year = safe_get_first([safe_get(c, "year", None) for c in contests if safe_get(c, "year", None)], "best_year", None, logger)
    best_type = safe_get_first([safe_get(c, "type_", None) for c in contests if safe_get(c, "type_", None)], "best_type", None, logger)
    best_election_types = []
    if contests:
        best_election_types = safe_get(contests[0], "election_types", []) or []
    def propagate_year_type(items, year, type_, election_types=None) -> None:
        for item in items:
            if isinstance(item, dict):
                if "year" not in item or item["year"] is None:
                    item["year"] = year
                if "type_" not in item or item["type_"] is None:
                    item["type_"] = type_
                sync_type_and_election_types(item, fallback_types=election_types or [type_] if type_ else None, fallback_type=type_)
    for section in ["tables", "candidate_panels", "location_panels", "ballot_types"]:
        propagate_year_type(safe_get(context_result, section, []), best_year, best_type, best_election_types)

    # --- 2. Validate dom_parts structure ---
    dom_parts = {
        "contests": safe_get(context_result, "contests", []),
        "panels": safe_get(context_result, "panels", []),
        "tables": safe_get(context_result, "tables", []),
        "candidate_panels": safe_get(context_result, "candidate_panels", []),
        "location_panels": safe_get(context_result, "location_panels", []),
        "headings": safe_get(context_result, "headings", []),
        "ballot_types": safe_get(context_result, "ballot_types", []),
        "results_timestamps": safe_get(context_result, "results_timestamps", []),
        "party_labels": safe_get(context_result, "party_labels", []),
        "vote_methods": safe_get(context_result, "vote_methods", []),
        "pattern_kb_matches": safe_get(context_result, "pattern_kb_matches", []),
        "segments_needing_review": safe_get(context_result, "segments_needing_review", []),
        "selector_log": safe_get(context_result, "selector_log", []),
        "metadata": safe_get(context_result, "metadata", {}),
        "tagged_segments": safe_get(context_result, "tagged_segments", []),
        "tagged_segments_with_attrs": safe_get(context_result, "tagged_segments_with_attrs", []),
        "raw_html": safe_get(context_result, "raw_html", html),
        "error": safe_get(context_result, "error", None),
        "url": safe_get(context_result, "url", None),
    }
    # Ensure all list fields are lists
    for key in [
        "contests",
        "panels",
        "tables",
        "candidate_panels",
        "location_panels",
        "headings",
        "ballot_types",
        "results_timestamps",
        "party_labels",
        "vote_methods",
        "pattern_kb_matches",
        "segments_needing_review",
        "segment_evidence",
        "selector_log",
        "tagged_segments",
        "tagged_segments_with_attrs",
    ]:
        if key not in dom_parts or not isinstance(dom_parts[key], list):
            dom_parts[key] = []
    valid = validate_dom_parts(dom_parts, verbose=debug)
    if not valid:
        msg = "[DOM_PARTS] Validation failed. Downstream consumers may not function correctly."
        if logger.mode == "cli":
            console.print(msg)
        else:
            logger.error({"level": "ERROR", "type": "dom_parts", "message": msg})

    context_result["dom_parts"] = dom_parts

    # --- 3. Enrich with dom_tree, label_groups, panels_and_tables, HTML samples ---
    organizer = ContextOrganizer()
    segments = dom_parts.get("tagged_segments_with_attrs", [])
    if segments:
        dom_tree = organizer.build_dom_tree(segments)
        context_result["dom_tree"] = dom_tree
        label_groups = organizer.group_nodes_by_label(dom_tree["nodes"], label_field="ml_label")
        context_result["dom_label_groups"] = label_groups
        panels_and_tables = organizer.get_panels_and_tables(dom_tree)
        context_result["dom_panels_and_tables"] = panels_and_tables
        for seg in segments:
            seg["dom_node"] = dom_tree["nodes"][seg["_idx"]] if seg["_idx"] < len(dom_tree["nodes"]) else None
            seg["label_group"] = label_groups.get(safe_get(seg, "ml_label", ""), [])
            seg["panel_group"] = None
            for panel in panels_and_tables:
                if seg["_idx"] in safe_get(panel, "panel_indices", []) or seg["_idx"] in safe_get(panel, "table_indices", []):
                    seg["panel_group"] = panel
                    break
        N = min(5, len(dom_tree["nodes"]))
        context_result["dom_node_html_samples"] = [
            organizer.extract_html_by_idx(dom_tree["nodes"], i, safe_get(context_result, "raw_html", html))
            for i in range(N)
        ]
        context_result["dom_subtree_html_samples"] = [
            organizer.extract_subtree_html(dom_tree["nodes"], i, safe_get(context_result, "raw_html", html))
            for i in range(N)
        ]
        msg_enrich = "[DOM ENRICHMENT] Added dom_tree, label_groups, panels_and_tables, and HTML samples to context_result."
        if logger.mode == "cli":
            console.print(msg_enrich)
        else:
            logger.info({"level": "INFO", "type": "dom_enrichment", "message": msg_enrich})

    # --- 4. Save to context cache ---
    if context_cache is not None:
        safe_setdefault(context_result, "page_hash", page_hash)
        safe_setdefault(context_result, "timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
        context_cache[page_hash] = context_result
        msg_cache = f"[CACHE] Saving context cache for page_hash={page_hash} with {len(context_result.get('tagged_segments_with_attrs', []))} segments."
        if logger.mode == "cli":
            console.print(msg_cache)
        else:
            logger.debug({"level": "DEBUG", "type": "cache", "message": msg_cache})
        try:
            save_context_cache_to_disk(context_cache)
        except Exception as e:
            msg_error = f"[ERROR] Exception during save_context_cache_to_disk: {e}"
            if logger.mode == "cli":
                console.print(msg_error)
            else:
                logger.error({"level": "ERROR", "type": "cache", "message": msg_error})

    # --- 5. Downstream enrichment via coordinator if present ---
    if coordinator is not None and hasattr(coordinator, "organize_and_enrich"):
        organized = coordinator.organize_and_enrich(
            context_result,
            write_kind=ContextWriteKind.EVIDENCE,
        )
        if organized and isinstance(organized, dict):
            safe_update(context_result, organized, logger)
            if "dom_parts" in organized:
                dom_parts_keys = list(safe_keys(organized["dom_parts"]))
                msg_dom_parts = f"[DOM_PARTS] dom_parts successfully organized with keys: {dom_parts_keys}"
                if logger.mode == "cli":
                    console.print(msg_dom_parts)
                else:
                    logger.debug({"level": "DEBUG", "type": "dom_parts", "message": msg_dom_parts})

    # --- 6. Debug logging ---
    if debug and "dom_tree" in context_result:
        dom_tree = context_result["dom_tree"]
        nodes = safe_get(dom_tree, "nodes", [])
        for idx in range(min(5, len(nodes))):
            node = nodes[idx] if idx < len(nodes) else None
            if node is None:
                msg_warn = f"[DOM DEBUG] Node {idx} is None."
                if logger.mode == "cli":
                    console.print(msg_warn)
                else:
                    logger.warning({"level": "WARNING", "type": "dom_debug", "message": msg_warn})
                continue
            html_snippet = organizer.extract_html_by_idx(nodes, idx, safe_get(context_result, "raw_html", html))
            msg_html = f"[DOM DEBUG] Node {idx} HTML: {html_snippet[:100]}"
            if logger.mode == "cli":
                console.print(msg_html)
            else:
                logger.info({"level": "INFO", "type": "dom_debug", "message": msg_html})
            subtree_html = organizer.extract_subtree_html(nodes, idx, safe_get(context_result, "raw_html", html))
            msg_subtree = f"[DOM DEBUG] Subtree HTML for node {idx}: {subtree_html[:200]}"
            if logger.mode == "cli":
                console.print(msg_subtree)
            else:
                logger.info({"level": "INFO", "type": "dom_debug", "message": msg_subtree})
                
    # --- 7. Final type/election type propagation for all sections ---
    for contest in safe_get(context_result, "contests", []):
        sync_type_and_election_types(contest)
    best_contest = safe_get_first(
        safe_get(context_result, "contests", []),
        "contests",
        safe_get(context_result, "url", None),
        logger,
        default={}
    )
    best_type = safe_get(best_contest, "type_", None)
    best_election_types = safe_get(best_contest, "election_types", [])
    for section in ["tables", "candidate_panels", "location_panels", "ballot_types"]:
        for item in safe_get(context_result, section, []):
            sync_type_and_election_types(item, fallback_types=best_election_types, fallback_type=best_type)
    sync_type_and_election_types(context_result, fallback_types=best_election_types, fallback_type=best_type)

    return context_result

def _extract_heading_text(heading: Any) -> str:
    if isinstance(heading, str):
        return heading.strip()
    if isinstance(heading, dict):
        for key in ("text", "label", "heading", "raw_html", "html"):
            val = safe_get(heading, key, None)
            if isinstance(val, str) and val.strip():
                return _extract_clean_text(val)
    return ""

def _build_model_signals(context_result: Dict[str, Any]) -> Dict[str, Any]:
    segments = safe_get(context_result, "tagged_segments_with_attrs", []) or []
    if not isinstance(segments, list):
        segments = []

    label_counter: Counter = Counter()
    confidences: List[float] = []
    confidence_buckets = {"low": 0, "medium": 0, "high": 0}

    for seg in segments:
        if not isinstance(seg, dict):
            continue
        label = str(safe_get(seg, "ml_label", "unknown") or "unknown").strip().lower()
        label_counter[label] += 1

        conf_raw = safe_get(seg, "ml_confidence", None)
        try:
            if conf_raw is not None:
                conf = float(conf_raw)
                if conf < 0:
                    conf = 0.0
                elif conf > 1:
                    conf = 1.0
                confidences.append(conf)
                if conf < 0.4:
                    confidence_buckets["low"] += 1
                elif conf < 0.7:
                    confidence_buckets["medium"] += 1
                else:
                    confidence_buckets["high"] += 1
        except Exception:
            continue

    total_segments = len(segments)
    total_labeled = sum(v for k, v in label_counter.items() if k not in {"unknown", "ignore"})
    unknown_count = label_counter.get("unknown", 0)
    ignore_count = label_counter.get("ignore", 0)

    if confidences:
        conf_min = float(min(confidences))
        conf_max = float(max(confidences))
        conf_avg = float(sum(confidences) / len(confidences))
        conf_median = float(np.median(np.array(confidences, dtype=float)))
    else:
        conf_min = conf_max = conf_avg = conf_median = 0.0

    labels_top = [
        {"label": label, "count": count}
        for label, count in label_counter.most_common(15)
    ]

    return {
        "segment_count": total_segments,
        "labeled_segment_count": total_labeled,
        "unknown_segment_count": unknown_count,
        "ignore_segment_count": ignore_count,
        "label_distribution": labels_top,
        "confidence": {
            "count": len(confidences),
            "min": conf_min,
            "max": conf_max,
            "avg": conf_avg,
            "median": conf_median,
            "buckets": confidence_buckets,
        },
        "review_signals": {
            "segments_needing_review": len(safe_get(context_result, "segments_needing_review", []) or []),
            "pattern_kb_matches": len(safe_get(context_result, "pattern_kb_matches", []) or []),
        },
    }

DIGEST_SCHEMA_VERSION = "1.1"
DIGEST_TRENDS_FILE = "context_digest_trends.json"
DIGEST_TRENDS_MAX_ITEMS = 120

def _build_context_digest(context_result: Dict[str, Any], page_url: str, selector_log: Set[str]) -> Dict[str, Any]:
    contests = safe_get(context_result, "contests", []) or []
    headings = safe_get(context_result, "headings", []) or []
    panels = safe_get(context_result, "panels", []) or []
    tables = safe_get(context_result, "tables", []) or []
    ocr_context = safe_get(context_result, "ocr_context", {}) or {}
    model_signals = _build_model_signals(context_result)

    panel_coverage = []
    for panel in panels:
        heading = ""
        if isinstance(panel, dict):
            heading = str(panel.get("panel_heading") or panel.get("heading") or "").strip()
            scores = panel.get("keyword_scores") if isinstance(panel.get("keyword_scores"), dict) else {}
            table_count = len(panel.get("tables") or []) if isinstance(panel.get("tables"), list) else 0
        else:
            scores = {}
            table_count = 0
        if heading or scores:
            panel_coverage.append({
                "heading": heading or "(unlabeled panel)",
                "keyword_scores": scores,
                "table_count": table_count,
            })
        if len(panel_coverage) >= 12:
            break

    top_headings = []
    for heading in headings:
        text = _extract_heading_text(heading)
        if text:
            top_headings.append(text)
        if len(top_headings) >= 10:
            break

    contest_samples = []
    for contest in contests:
        if isinstance(contest, dict):
            title = safe_get(contest, "title", "") or safe_get(contest, "contest", "")
        else:
            title = str(contest)
        title = str(title).strip()
        if title:
            contest_samples.append(title)
        if len(contest_samples) >= 5:
            break

    return {
        "schema_version": DIGEST_SCHEMA_VERSION,
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "page_url": page_url,
        "counts": {
            "contests": len(contests),
            "panels": len(panels),
            "tables": len(tables),
            "headings": len(headings),
        },
        "top_headings": top_headings,
        "contest_samples": contest_samples,
        "selector_sample": sorted(list(selector_log))[:20],
        "ocr_present": bool(ocr_context),
        "ocr_fields": safe_get(ocr_context, "fields", {}) if ocr_context else {},
        "panel_coverage": panel_coverage,
        "model_signals": model_signals,
    }

def _update_digest_trends(out_dir: str, digest: Dict[str, Any], session_id: Optional[str]) -> None:
    trends_path = os.path.join(out_dir, DIGEST_TRENDS_FILE)
    existing: List[Dict[str, Any]] = []
    try:
        if os.path.exists(trends_path):
            with open(trends_path, "rb") as f:
                raw = f.read()
            loaded = orjson.loads(raw) if raw else []
            if isinstance(loaded, list):
                existing = [x for x in loaded if isinstance(x, dict)]
    except Exception:
        existing = []

    model_signals = safe_get(digest, "model_signals", {}) or {}
    confidence = safe_get(model_signals, "confidence", {}) or {}
    review_signals = safe_get(model_signals, "review_signals", {}) or {}
    segment_count = int(safe_get(model_signals, "segment_count", 0) or 0)
    unknown_count = int(safe_get(model_signals, "unknown_segment_count", 0) or 0)
    unknown_ratio = float(unknown_count / segment_count) if segment_count > 0 else 0.0

    trend_entry = {
        "schema_version": DIGEST_SCHEMA_VERSION,
        "session_id": session_id,
        "generated_at": safe_get(digest, "generated_at", datetime.datetime.utcnow().isoformat() + "Z"),
        "page_url": safe_get(digest, "page_url", ""),
        "counts": safe_get(digest, "counts", {}),
        "panel_count": len(safe_get(digest, "panel_coverage", []) or []),
        "segment_count": segment_count,
        "unknown_segment_count": unknown_count,
        "unknown_ratio": unknown_ratio,
        "labeled_segment_count": int(safe_get(model_signals, "labeled_segment_count", 0) or 0),
        "confidence": {
            "count": safe_get(confidence, "count", 0),
            "avg": safe_get(confidence, "avg", 0.0),
            "median": safe_get(confidence, "median", 0.0),
            "buckets": safe_get(confidence, "buckets", {}),
        },
        "review_signals": {
            "segments_needing_review": safe_get(review_signals, "segments_needing_review", 0),
            "pattern_kb_matches": safe_get(review_signals, "pattern_kb_matches", 0),
        },
    }

    existing.append(trend_entry)
    if len(existing) > DIGEST_TRENDS_MAX_ITEMS:
        existing = existing[-DIGEST_TRENDS_MAX_ITEMS:]

    with open(trends_path, "wb") as f:
        f.write(orjson.dumps(existing, option=orjson.OPT_INDENT_2))

def _write_context_digest(digest: Dict[str, Any], session_id: Optional[str]) -> None:
    try:
        out_dir = os.path.join("tools", "debug_headless_output")
        os.makedirs(out_dir, exist_ok=True)
        suffix = session_id or hashlib.sha256(safe_encode(str(digest.get("page_url", "")))).hexdigest()[:10]
        filename = os.path.join(out_dir, f"context_digest_{suffix}.json")
        with open(filename, "wb") as f:
            f.write(orjson.dumps(digest, option=orjson.OPT_INDENT_2))
        _update_digest_trends(out_dir, digest, session_id)
    except Exception:
        pass