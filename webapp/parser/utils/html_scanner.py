import hashlib
import orjson
import os
import re
import time
import threading
import traceback
import numpy as np
from typing import Dict, Any, List, Optional, TYPE_CHECKING
import concurrent.futures
from ..config import CONTEXT_LIBRARY_PATH, CACHE_DIR, LOG_DIR, CONTEXT_CACHE_PATH
from ..utils.shared_logger import SharedLogger
from ..utils.shared_logic import (
    safe_append_cached_segment, safe_append, safe_update, safe_extend,
    convert_ndarrays, _sanitize_log_filename, _normalize_html_for_hash, clean_cache_inplace,
    _keyword_in_text, safe_lower, safe_encode, safe_startswith, safe_add, safe_items, safe_model_encode,
    safe_get_first, _sync_type_and_election_types, safe_get, safe_strip
)
from ..Context_Integration.Context_Library.constants import (
    STATE_ABBR, KNOWN_STATE_TO_COUNTY_MAP, KNOWN_COUNTY_TO_PRECINCTS_MAP,
    ELECTION_TYPES, BALLOT_TYPES, PARTY_KEYWORDS, CONTEST_KEYWORDS,
    CANDIDATE_KEYWORDS, BALLOT_TYPES, ELECTION_TYPES,
    HTML_TAGS, PANEL_TAGS, HEADING_TAGS, CUSTOM_ATTR_PATTERNS, LOCATION_KEYWORDS, EXTRA_HEADING_TAGS,
    ALWAYS_IGNORE_TAGS, ALWAYS_IGNORE_CLASSES, ALWAYS_IGNORE_IDS, ICON_CLASSES, ICON_TAGS, BUTTON_CLASSES,
    HEADING_CLASSES, PANEL_CLASSES, TIMESTAMP_CLASSES, STRUCTURAL_TAGS, TIMESTAMP_ID_PATTERNS, TIMESTAMP_ATTRS,
    MISC_FOOTER_KEYWORDS, UPDATE_PANEL_KEYWORDS, VIEW_BY_PHRASES, CANONICAL_SEGMENT_LABELS,
    TOTAL_KEYWORDS, PERCENT_KEYWORDS, ROOT_CONTAINER_TAGS, LOCATION_ABBREVIATIONS
)
from ..bots.librarian import (
    
    update_context_library, load_context_library, log_unknown_tag, log_unknown_attr, 
    get_canonical_segment_label, cache_segment_label, get_cached_segment_label,    
)
from ..utils.embedding_cache import (
    save_embedding, get_embedding_from_memory, load_embeddings_batch, save_embeddings_batch
)
from ..utils.user_prompt import UserPrompt
from selectolax.parser import HTMLParser
from ..utils.model_registry import ModelRegistry
from difflib import get_close_matches

if TYPE_CHECKING:
    from ..Context_Integration.context_coordinator import ContextCoordinator

prompt = UserPrompt()
logger = SharedLogger()
ENABLE_SEGMENT_LABEL_PROMPT = os.getenv("ENABLE_SEGMENT_LABEL_PROMPT", "true").lower() == "true"
console = None  # Only import rich.console.Console if needed for interactive output

# --- Caching and threading ---
_LABEL_CACHE_FILENAME = "segment_label_cache.json"
_LABEL_CACHE_LOCK = threading.Lock()
_LABEL_CACHE = None
_context_cache = None
_pattern_kb_cache = None

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
    """Returns the path to the label cache file, ensuring it is safe and does not exceed OS limits."""
    if not _LABEL_CACHE_FILENAME:
        raise ValueError("Label cache filename cannot be empty")
    if not isinstance(_LABEL_CACHE_FILENAME, str):
        raise TypeError("Label cache filename must be a string")
    path = safe_cache_path(_LABEL_CACHE_FILENAME)
    if os.name == "nt" and len(os.path.abspath(path)) >= 260:
        import tempfile
        short_path = os.path.join(tempfile.gettempdir(), _LABEL_CACHE_FILENAME)
        logger.warning(f"[CACHE] Path too long for Windows, using temp path: {short_path}")
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
    if not isinstance(_LABEL_CACHE, dict):
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
    """Generates a safe cache file path, ensuring it does not escape the cache directory."""
    if not filename:
        raise ValueError("Filename cannot be empty")
    if not isinstance(filename, str):
        raise TypeError("Filename must be a string")
    if not re.match(r"^[\w\-. ]+$", filename):
        raise ValueError("Filename contains unsafe characters")
    # Sanitize filename to prevent directory traversal or unsafe characters
    filename = _sanitize_log_filename(filename)
    cache_folder = CACHE_DIR
    # Defensive: fallback to temp if path too long
    full_path = os.path.join(cache_folder, filename)
    if os.name == "nt" and len(os.path.abspath(full_path)) >= 240:
        import tempfile
        temp_path = os.path.join(tempfile.gettempdir(), filename)
        logger.warning(f"[CACHE] Path too long for Windows, using temp path: {temp_path}")
        # Ensure temp dir exists
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        return temp_path
    # Ensure cache dir exists
    os.makedirs(cache_folder, exist_ok=True)
    if not os.path.abspath(full_path).startswith(os.path.abspath(cache_folder)):
        raise ValueError("Unsafe cache path detected!")
    return full_path

def safe_log_path(filename: str) -> str:
    """Generates a safe log file path, ensuring it does not escape the log directory."""
    if not filename:
        raise ValueError("Filename cannot be empty")
    if not isinstance(filename, str):
        raise TypeError("Filename must be a string")
    if not re.match(r"^[\w\-. ]+$", filename):
        raise ValueError("Filename contains unsafe characters")
    # Sanitize filename to prevent directory traversal or unsafe characters
    filename = _sanitize_log_filename(filename)
    if not filename.endswith(".log"):
        filename += ".log"
    log_folder = LOG_DIR
    # Defensive: fallback to temp if path too long
    if os.name == "nt" and len(os.path.abspath(os.path.join(log_folder, filename))) >= 240:
        import tempfile
        temp_path = os.path.join(tempfile.gettempdir(), filename)
        logger.warning(f"[LOG] Path too long for Windows, using temp path: {temp_path}")
        # Ensure temp dir exists
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        return temp_path
    os.makedirs(log_folder, exist_ok=True)
    # Ensure log dir exists
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    full_path = os.path.join(log_folder, filename)
    # Ensure the log path does not escape the log directory
    if not os.path.abspath(full_path).startswith(os.path.abspath(log_folder)):
        raise ValueError("Unsafe log path detected!")
    return full_path

def is_trivial_segment(seg, diagnostics=False, logger_instance=None) -> bool:
    """
    Determines if a segment is trivial (should be ignored for semantic processing).
    Checks for empty, whitespace, HTML entities, tags, icons, comments, scripts, styles,
    numeric-only, special-char-only, and other non-informative content.
    Optionally logs diagnostics for audit/debug.
    """
    logger_obj = logger_instance or logger
    html = safe_get(seg, "html", "")
    tag = safe_lower(safe_get(seg, "tag", ""))
    classes = [safe_lower(c) for c in safe_get(seg, "classes", [])]
    attrs = safe_get(seg, "attrs", {}) or {}

    # Empty or whitespace-only HTML
    if not html or not safe_strip(html):
        if diagnostics:
            logger_obj.info(f"[TRIVIAL] Empty or whitespace HTML for tag={tag}")
        return True

    # HTML entities or just tags
    html_stripped = safe_strip(html)
    if html_stripped in {"&nbsp;", "&#160;"}:
        if diagnostics:
            logger_obj.info(f"[TRIVIAL] HTML entity only for tag={tag}")
        return True
    if re.fullmatch(r"<[^>]+>", html_stripped):
        if diagnostics:
            logger_obj.info(f"[TRIVIAL] Just a tag for tag={tag}")
        return True

    # Known trivial tags
    if tag in {"br", "hr", "wbr"} and not html_stripped:
        if diagnostics:
            logger_obj.info(f"[TRIVIAL] Known trivial tag: {tag}")
        return True

    # Icon-only spans
    if tag == "span" and classes and all("icon" in cls for cls in classes) and not safe_strip(re.sub(r"<[^>]+>", "", html)):
        if diagnostics:
            logger_obj.info(f"[TRIVIAL] Icon-only span for tag={tag}, classes={classes}")
        return True

    # Comments, scripts, styles
    if re.search(r"<!--.*?-->", html, re.DOTALL):
        if diagnostics:
            logger_obj.info(f"[TRIVIAL] HTML comment detected for tag={tag}")
        return True
    if re.search(r"<script.*?>.*?</script>", html, re.DOTALL) or re.search(r"<style.*?>.*?</style>", html, re.DOTALL):
        if diagnostics:
            logger_obj.info(f"[TRIVIAL] Script or style block detected for tag={tag}")
        return True

    # Numeric-only or special-char-only segments
    text_content = re.sub(r"<[^>]+>", "", html_stripped)
    if safe_strip(text_content).isdigit():
        if diagnostics:
            logger_obj.info(f"[TRIVIAL] Numeric-only content for tag={tag}")
        return True
    if bool(re.fullmatch(r'[\W_]+', safe_strip(text_content))):
        if diagnostics:
            logger_obj.info(f"[TRIVIAL] Special-char-only content for tag={tag}")
        return True

    # Trivial by attribute (e.g., aria-hidden, display:none)
    if safe_lower(attrs.get("aria-hidden", "")) == "true" or "display:none" in safe_lower(attrs.get("style", "")):
        if diagnostics:
            logger_obj.info(f"[TRIVIAL] aria-hidden or display:none for tag={tag}")
        return True

    # Defensive: very short text (1 char or less)
    if len(safe_strip(text_content)) <= 1:
        if diagnostics:
            logger_obj.info(f"[TRIVIAL] Very short content for tag={tag}")
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
    html_norm = _normalize_html_for_hash(html)
    base = tag + orjson.dumps(attrs_sorted, option=orjson.OPT_SORT_KEYS).decode() + html_norm + str(model_id)
    return hashlib.sha256(safe_encode(base, "utf-8")).hexdigest()

def get_segment_embedding(
    model,
    segment,
    cache=None,
    cache_hits=None,
    cache_misses=None,
    diagnostics=False,
    logger_instance=None
) -> Optional[np.ndarray]:
    """
    Robustly computes or retrieves the embedding for a segment.
    - Uses cache if available.
    - Logs cache hits/misses and errors.
    - Handles trivial/empty segments.
    - Optionally logs timing and diagnostics.
    """
    logger_obj = logger_instance or logger
    model_id = getattr(model, 'name_or_path', str(model))
    identity = embedding_cache_hash(segment, model_id)
    if cache is not None:
        clean_cache_inplace(cache)
    if is_trivial_segment(segment):
        if diagnostics:
            logger_obj.info(f"[EMBED] Skipping trivial segment: {identity}")
        return None
    emb = get_embedding_from_memory(identity)
    if emb is not None:
        safe_add(cache_hits, str(identity))
        if diagnostics:
            logger_obj.info(f"[EMBED] Cache hit for {identity}")
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
            logger_obj.warning(f"[EMBED] Empty text for segment: {identity}")
        return None
    try:
        import time
        t0 = time.time()
        emb = safe_model_encode(model, full_text, convert_to_numpy=True, show_progress_bar=False)
        save_embedding(identity, emb)
        safe_add(cache_misses, str(identity))
        if diagnostics:
            logger_obj.info(f"[EMBED] Computed embedding for {identity} in {time.time()-t0:.3f}s")
        return emb
    except Exception as e:
        segment["embedding_error"] = str(e)
        if diagnostics:
            logger_obj.error(f"[EMBED] Error for {identity}: {e}")
        return None

def batch_get_segment_embeddings(
    model,
    segments,
    diagnostics=False,
    logger_instance=None
) -> List[Optional[np.ndarray]]:
    """
    Robust batch embedding retrieval/computation for segments.
    - Uses cache where possible.
    - Parallelizes computation for large batches.
    - Logs progress and errors.
    """
    logger_obj = logger_instance or logger
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
            import time
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
                logger_obj.info(f"[EMBED] Batch computed {len(texts)} embeddings in {time.time()-t0:.2f}s")
        except Exception as e:
            if diagnostics:
                logger_obj.error(f"[EMBED] Batch embedding error: {e}")
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
    ml_threshold=0.7,
    coordinator=None
) -> Optional[tuple]:
    if coordinator is None:
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
    if _keyword_in_text(text, CONTEST_KEYWORDS) or _keyword_in_text(html, CONTEST_KEYWORDS):
        return "contest"
    if _keyword_in_text(text, CANDIDATE_KEYWORDS) or _keyword_in_text(html, CANDIDATE_KEYWORDS):
        return "candidate_panel"
    if _keyword_in_text(text, PARTY_KEYWORDS) or _keyword_in_text(html, PARTY_KEYWORDS):
        return "party_label"
    if _keyword_in_text(text, LOCATION_KEYWORDS) or _keyword_in_text(html, LOCATION_KEYWORDS):
        return "location_panel"
    if _keyword_in_text(text, BALLOT_TYPES) or _keyword_in_text(html, BALLOT_TYPES):
        return "ballot_types"
    if tag == "table" or _keyword_in_text(text, TOTAL_KEYWORDS | PERCENT_KEYWORDS | MISC_FOOTER_KEYWORDS):
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
    logger_instance=None,
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
    logger_obj = logger_instance or logger

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
            logger_obj.warning(
                f"[SEGMENT FILTER] Skipped segment ({item['reason']}): {str(item['segment'])[:120]}..."
            )
        if len(filtered_out) > 10:
            logger_obj.warning(f"[SEGMENT FILTER] {len(filtered_out)} segments filtered out (showing first 10).")

    return results

def extract_year_and_type(text, url=None) -> tuple:
    """
    Extracts the most likely year and election type from anywhere in the string or url.
    Also extracts a 'last updated' date if present.
    Returns (year, election_type, cleaned_text, last_updated)
    """
    import re
    from collections import Counter

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
    ml_threshold: float = 0.85,   
    model=None,
    coordinator=None,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Extract DOM segments with attributes and ML-driven semantic labels.
    Uses selectolax for DOM, leverages context, pattern KB, and coordinator for optimal labeling.
    Ensures robust parent/child relationships, unique indices, and auditability for downstream use.
    """
    from ..Context_Integration.context_organizer import ContextOrganizer
    if coordinator is None:
        coordinator = ContextCoordinator()
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
    panel_tags = PANEL_TAGS
    heading_tags = HEADING_TAGS
    custom_attr_patterns = CUSTOM_ATTR_PATTERNS
    location_keywords = LOCATION_KEYWORDS
    candidate_keywords = CANDIDATE_KEYWORDS
    ballot_types = BALLOT_TYPES
    segments: List[Dict[str, Any]] = []

    try:
        tree = HTMLParser(html)
        # --- Robust index assignment ---
        idx_counter = [0]

        def safe_split(val, sep=None):
            try:
                return val.split(sep) if isinstance(val, str) else []
            except Exception:
                return []

        def walk(node, parent_idx=None, heading_idx=None, panel_idx=None, **kwargs):
            tag = getattr(node, "tag", None)
            tag_lower = safe_lower(tag or "")
            if not tag or tag_lower not in HTML_TAGS:
                log_unknown_tag(tag, context_library)
                for child in getattr(node, "iter", lambda **kw: [] if not kw else [] )(include_text=True, **kwargs):
                    walk(child, parent_idx, heading_idx, panel_idx, **kwargs)
                return None
            attrs = dict(getattr(node, "attributes", {}))
            if include_data_attrs:
                attrs.update({k: v for k, v in getattr(node, "attributes", {}).items() if safe_lower(k or "").startswith("data-")})
            classes = safe_split(attrs.get("class", "") or "")
            id_ = attrs.get("id", "")
            is_button = tag_lower == "button" or (tag_lower == "input" and safe_lower(attrs.get("type", "") or "") in ["button", "submit"])
            button_text = ""
            if is_button:
                button_text = (
                    attrs.get("aria-label")
                    or attrs.get("value")
                    or (
                        getattr(
                            node,
                            "text",
                            lambda **kw: (logger.debug(f"text lambda received kw: {kw}") or "")
                        )(strip=True, **kwargs)
                        if hasattr(node, "text")
                        else ""
                    )
                    or ""
                )
            is_clickable = (
                is_button
                or tag_lower == "a"
                or "onclick" in attrs
                or "btn" in classes
                or "button" in classes
            )

            this_heading_idx = heading_idx
            if tag_lower in heading_tags:
                this_heading_idx = safe_get_first(idx_counter, "heading_idx", None, logger)

            this_panel_idx = panel_idx
            if tag_lower in panel_tags:
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
            # Attribute audit
            for k in attrs:
                if any(pat.match(k) for pat in custom_attr_patterns):
                    seg["has_custom_attr"] = True
                log_unknown_attr(k, context_library)
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
            # --- Filter out trivial segments here ---
            clean_text = _extract_clean_text(seg["html"])
            if not clean_text or not clean_text.strip():
                return None
            if clean_text.strip() in {"&nbsp;", "&#160;"}:
                return None
            if re.fullmatch(r"<[^>]+>", clean_text.strip()):
                return None
            segments.append(seg)
            this_idx = seg["_idx"]
            for child in getattr(node, "iter", lambda **kw: [] if not kw else [] )(include_text=True, **kwargs):
                if kwargs:
                    logger.debug(f"walk: kwargs passed to iter: {kwargs}")
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
        for label, group in label_groups.items():
            logger.info(f"[DOM LABEL GROUP] '{label}': {len(group)} nodes")

        panels_and_tables = organizer.get_panels_and_tables(dom_tree)
        logger.info(f"[DOM PANELS/TABLES] {len(panels_and_tables)} panel/table groups extracted.")

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
            logger.info(f"[DOM NODE HTML] Node {i}: {node_html[:120]}...")
            logger.info(f"[DOM SUBTREE HTML] Node {i}: {subtree_html[:120]}...")

        seg_hashes = [segment_hash(seg.get("html", "")) for seg in segments]
        seg_htmls = [seg.get("html", "") for seg in segments]
        total_segments = len(seg_hashes)
        logger.info(f"[EMBED] Total segments: {total_segments}")

        hash_to_embedding = {}
        CHUNK_SIZE = 1024
        for i in range(0, total_segments, CHUNK_SIZE):
            chunk_hashes = seg_hashes[i:i+CHUNK_SIZE]
            chunk_result = load_embeddings_batch(chunk_hashes)
            hash_to_embedding.update(chunk_result)
            hits = sum(1 for v in chunk_result.values() if v is not None)
            logger.debug(f"[EMBED] Batch {i//CHUNK_SIZE+1}: {hits} hits, {len(chunk_hashes)-hits} misses")
        missing = [(h, html) for h, html in zip(seg_hashes, seg_htmls) if hash_to_embedding.get(h) is None]
        if missing:
            logger.info(f"[EMBED] Computing {len(missing)} missing embeddings in chunks of {CHUNK_SIZE}")
            for i in range(0, len(missing), CHUNK_SIZE):
                chunk = missing[i:i+CHUNK_SIZE]
                missing_hashes, missing_htmls = zip(*chunk)
                try:
                    new_embs = model.encode(list(missing_htmls), convert_to_numpy=True, show_progress_bar=False)
                except Exception as e:
                    logger.error(f"[EMBED] Batch embedding computation failed: {e}")
                    continue
                save_embeddings_batch(list(zip(missing_hashes, new_embs)))
                for h, emb in zip(missing_hashes, new_embs):
                    hash_to_embedding[h] = emb
                logger.debug(f"[EMBED] Saved {len(chunk)} new embeddings to cache.")
        for seg, h in zip(segments, seg_hashes):
            seg["_embedding"] = hash_to_embedding[h]
        logger.info(f"[EMBED] Embedding assignment complete for {len(segments)} segments.")

        for seg in segments:
            text = safe_lower(seg.get("html") or "")
            seg["contains_election_keyword"] = any(
                safe_lower(kw) in text for kw in (list(location_keywords) + list(candidate_keywords) + list(ballot_types))
            )
            seg["contains_candidate"] = any(
                safe_lower(cand) in text for cand in candidate_keywords
            )
            emb = seg.get("_embedding")
            label = auto_label_segment(
                seg,
                context_library=context_library,
                context_cache=context_cache,
                pattern_kb=pattern_kb,
                model=model,
                ml_threshold=ml_threshold,
                coordinator=coordinator
            )
            seg["ml_label"] = label
            seg["ml_confidence"] = 1.0 if label != "unknown" else 0.0
            html_val = seg.get('html') or ''
            if not isinstance(html_val, str):
                html_val = str(html_val)
            seg["pattern_id"] = f"pattern_{hashlib.sha256(html_val.encode('utf-8')).hexdigest()[:10]}"
            seg["is_actionable"] = label in ("results_table", "contest", "candidate_panel", "location_panel")
            seg["is_election_result"] = label == "results_table"
            seg["is_contest"] = label == "contest"
            seg["label_group"] = label_groups.get(label, [])
            seg["panel_table_context"] = seg.get("panel_group", {})

        logger.debug(f"[DOM SEGMENTS] Extracted {len(segments)} segments. Example: {safe_get_first(segments, 'segments', None, logger, default='None')}")
        if not segments:
            logger.warning("[DOM SEGMENTS] No DOM segments extracted. Check HTML input and parser logic.")

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
        logger.error(f"[FALLBACK] selectolax failed: {e}\nDetails: {error_details}")
        if not fallback_on_error:
            raise
        return [{
            "error_info": error_details,
            "segments": [],
        }]

def get_page_hash(page) -> str:
    """
    Robustly compute a hash for the page content.
    Handles None, bytes, and normalizes whitespace for stability.
    """
    try:
        if page is None:
            content = ""
        else:
            # Safety net for .content
            try:
                content = getattr(page, "content", lambda: "")()
            except Exception:
                logger.warning("[PAGE_HASH] Exception when calling page.content(), using empty string.")
                content = ""
        if content is None:
            logger.warning("[PAGE_HASH] Page content is None, using empty string for hash.")
            content = ""
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")
        elif not isinstance(content, str):
            content = str(content)
        # Normalize line endings and strip leading/trailing whitespace
        content = content.replace('\r\n', '\n').replace('\r', '\n').strip()
        if not content:
            logger.warning("[PAGE_HASH] Page content is empty after normalization.")
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
    except Exception as e:
        logger.error(f"[PAGE_HASH] Failed to compute hash: {e}")
        return hashlib.sha256(b"").hexdigest()


def load_context_cache_from_disk(filename=None) -> Dict[str, Any]:
    """
    Loads the context cache from disk as a dict of dicts.
    If the file is corrupted, logs and resets the cache.
    """
    global _context_cache
    if filename is None:
        filename = os.path.basename(CONTEXT_CACHE_PATH)
    path = safe_cache_path(filename)
    logger.debug(f"[DEBUG] Loading context cache from: {path}")
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                raw_cache = robust_orjson_loads(f.read())
                # Defensive: Only keep dict values
                _context_cache = {k: v for k, v in safe_items(raw_cache or {}) if isinstance(v, dict)}
                return _context_cache
        except Exception as e:
            logger.error(f"[ERROR] Failed to load {filename}: {e}. Resetting context cache.")
            _context_cache = {}
            save_context_cache_to_disk(_context_cache, path)
            return {}
    _context_cache = {}
    return {}

def save_context_cache_to_disk(context_cache, path=CONTEXT_CACHE_PATH) -> None:
    """
    Saves the entire context cache as a single JSON object (dict of dicts).
    Always overwrites the file.
    """
    logger.debug(f"[DEBUG] Saving context cache to: {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    context_cache = convert_ndarrays(context_cache)
    with open(path, "wb") as f:
        f.write(orjson.dumps(context_cache, option=orjson.OPT_INDENT_2))

def add_context_entry(page_hash: str, context: dict, path=CONTEXT_CACHE_PATH) -> None:
    """
    Adds or updates a context entry for a page hash and saves to disk.
    """
    cache = load_context_cache_from_disk(path)
    # Always ensure required metadata
    context.setdefault("page_hash", page_hash)
    context.setdefault("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
    cache[page_hash] = context
    save_context_cache_to_disk(cache, path)

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
    global _pattern_kb_cache
    if _pattern_kb_cache is not None:
        return _pattern_kb_cache
    kb = []
    path = safe_log_path("dom_pattern_kb.jsonl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            for line in f:
                try:
                    kb.append(robust_orjson_loads(line))
                except Exception:
                    continue
    _pattern_kb_cache = kb
    return kb

def append_pattern_kb(entry) -> None:
    if not isinstance(entry, dict):
        raise ValueError("Only dict entries can be written to dom_pattern_kb.jsonl")
    entry = convert_ndarrays(entry)
    if "embedding" in entry and isinstance(entry["embedding"], np.ndarray):
        entry["embedding"] = (entry["embedding"] or np.array([])).tolist()
    path = safe_log_path("dom_pattern_kb.jsonl")
    with open(path, "ab") as f:
        f.write(orjson.dumps(entry, option=orjson.OPT_INDENT_2) + b"\n")

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

def prompt_for_segment_label(
    segment,
    context_library=None,
    session_id=None,
    non_interactive=False
) -> str:
    """
    Prompt for a semantic label for a segment, with robust support for session_id and non_interactive mode.
    If non_interactive is True, returns 'unknown' without prompting.
    """
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
    # Robust non-interactive toggle for webapp UI/CLI
    if non_interactive or not ENABLE_SEGMENT_LABEL_PROMPT:
        return "unknown"
    if not html_preview:
        html_preview = f"[No HTML] tag={safe_get(segment, 'tag', [])} attrs={safe_get(segment, 'attrs', [])}"
    logger.warning(
        f"\n[bold yellow]Segment needs review:[/bold yellow]\n{html_preview[:200]}{'...' if len(html_preview) > 200 else ''}"
    )
    logger.info(
        "[cyan]What is the semantic role of this segment? (e.g., results_table, ballot_toggle, heading, panel, candidate_panel, location_panel, ballot_types, results_timestamp, download_link, clickable, footer, legend, contest, party_label, vote_method, reporting_status, summary, error_message, warning, info_box, navigation, pagination, tab, modal, tooltip, ignore, unknown, etc.)[/cyan]"
    )
    label = prompt.prompt_input("> ", session_id=session_id).strip()
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
    import datetime

    MAX_WARNINGS = 20
    warning_count = 0
    valid = True

    expected_keys = [
        "contests", "panels", "tables", "candidate_panels", "location_panels",
        "headings", "ballot_types", "results_timestamps", "party_labels", "vote_methods",
        "pattern_kb_matches", "segments_needing_review", "selector_log", "metadata",
        "tagged_segments", "tagged_segments_with_attrs", "raw_html", "error", "url"
    ]
    # Only warn about missing required keys if context expects them
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

    # Check all expected keys exist, but only warn if context expects them
    for key in expected_keys:
        if key not in dom_parts:
            if verbose and (context_expected is None or key in context_expected):
                logger.warning(f"[DOM_PARTS] Missing key: {key}")
            valid = False

    # Check required keys are lists and not empty, but only warn if context expects them
    for key in required_keys:
        val = dom_parts.get(key)
        if not isinstance(val, list):
            if verbose:
                logger.warning(f"[DOM_PARTS] Key '{key}' is not a list.")
            valid = False
        elif len(val) == 0:
            if verbose:
                logger.warning(f"[DOM_PARTS] No items found in '{key}'.")
            valid = False

    # Deep schema, regex, allowed values, and cross-field checks
    for section, fields in section_fields.items():
        items = dom_parts.get(section, [])
        if not isinstance(items, list):
            continue
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                if verbose and warning_count < MAX_WARNINGS:
                    logger.warning(f"[DOM_PARTS] Item {i} in '{section}' is not a dict.")
                warning_count += 1
                continue
            for field in fields:
                value = item.get(field)
                # Only warn for missing/empty fields if context expects this section
                if value is None or (isinstance(value, str) and not value.strip()):
                    if verbose and warning_count < MAX_WARNINGS and (context_expected is None or section in context_expected):
                        logger.warning(f"[DOM_PARTS] Item {i} in '{section}' missing or empty field '{field}'.")
                    warning_count += 1
                # Type checks
                if field.endswith("_html") and value and not isinstance(value, str):
                    if verbose:
                        logger.warning(f"[DOM_PARTS] Item {i} in '{section}' field '{field}' should be str (HTML).")
                    valid = False
                if field.endswith("_text") and value and not isinstance(value, str):
                    if verbose:
                        logger.warning(f"[DOM_PARTS] Item {i} in '{section}' field '{field}' should be str (text).")
                    valid = False
                if field == "year" and value:
                    if not re.fullmatch(r"20\d{2}", str(value)):
                        if verbose:
                            logger.warning(f"[DOM_PARTS] Item {i} in '{section}' has invalid year format: {value}")
                        valid = False
                    else:
                        year_int = int(value)
                        if year_int < 2000 or year_int > datetime.datetime.now().year + 1:
                            if verbose:
                                logger.warning(f"[DOM_PARTS] Item {i} in '{section}' has out-of-range year: {value}")
                            valid = False
                if field == "type_" and value:
                    if safe_lower(value) not in {safe_lower(t or "") for t in ELECTION_TYPES}:
                        if verbose:
                            logger.warning(f"[DOM_PARTS] Item {i} in '{section}' has unknown election type: {value}")
                        valid = False
                if field == "county" and value and "state" in item:
                    state_val = safe_lower(item.get("state") or "")
                    # Normalize state using STATE_ABBR
                    state_val = STATE_ABBR.get(state_val, state_val)
                    if state_val and safe_lower(value) not in {safe_lower(c) for c in KNOWN_STATE_TO_COUNTY_MAP.get(state_val, [])}:
                        if verbose:
                            logger.warning(f"[DOM_PARTS] Item {i} in '{section}' has unknown county '{value}' for state '{state_val}'")
                        valid = False
                if field == "state" and value:
                    state_norm = STATE_ABBR.get(safe_lower(value), safe_lower(value))
                    if state_norm not in KNOWN_STATE_TO_COUNTY_MAP:
                        if verbose:
                            logger.warning(f"[DOM_PARTS] Item {i} in '{section}' has unknown state: {value}")
                        valid = False
                if field == "timestamp_text" and value:
                    if not re.search(r"\d{4}.*\d{1,2}:\d{2}", value):
                        if verbose:
                            logger.warning(f"[DOM_PARTS] Item {i} in '{section}' field '{field}' does not look like a timestamp: {value}")
                        valid = False
                if section == "ballot_types" and field == "ballot_types_text" and value:
                    if safe_lower(value) not in {safe_lower(bt) for bt in BALLOT_TYPES}:
                        if verbose:
                            logger.warning(f"[DOM_PARTS] Item {i} in '{section}' has unknown ballot type: {value}")
                        valid = False
                if section == "party_labels" and field == "party_label_text" and value:
                    if safe_lower(value) not in {safe_lower(k) for k in PARTY_KEYWORDS}:
                        if verbose:
                            logger.warning(f"[DOM_PARTS] Item {i} in '{section}' has unknown party label: {value}")
                        valid = False
                if section == "location_panels" and field == "location_panel_text" and value:
                    if not any(safe_lower(kw) in safe_lower(value) for kw in LOCATION_KEYWORDS):
                        if verbose:
                            logger.warning(f"[DOM_PARTS] Item {i} in '{section}' has location text missing known keywords: {value}")
                        valid = False
                    # Precinct/district validation
                    county_val = safe_lower(item.get("county", "") or "")
                    for abbrev, full_names in LOCATION_ABBREVIATIONS.items():
                        if safe_lower(abbrev) in safe_lower(value):
                            for full_name in full_names:
                                if safe_lower(full_name) in safe_lower(value) and county_val in KNOWN_COUNTY_TO_PRECINCTS_MAP:
                                    precincts = KNOWN_COUNTY_TO_PRECINCTS_MAP[county_val]
                                    found = any(safe_lower(p) in safe_lower(value) for p in precincts)
                                    if not found:
                                        if verbose:
                                            logger.warning(f"[DOM_PARTS] Location panel {i}: '{value}' does not match any known precinct/district for county '{county_val}'.")
                                        valid = False
                # Canonical label checks for headings/panels
                if section == "headings" and field == "heading_text" and value:
                    canonical = CANONICAL_SEGMENT_LABELS.get(safe_lower(value))
                    if canonical and canonical != "heading":
                        if verbose:
                            logger.warning(f"[DOM_PARTS] Heading {i}: text '{value}' has canonical label '{canonical}' not 'heading'.")
                        valid = False
                if section == "panels" and field == "panel_text" and value:
                    canonical = CANONICAL_SEGMENT_LABELS.get(safe_lower(value))
                    if canonical and canonical != "panel":
                        if verbose:
                            logger.warning(f"[DOM_PARTS] Panel {i}: text '{value}' has canonical label '{canonical}' not 'panel'.")
                        valid = False
                # Tag checks for headings/panels
                if section == "headings" and "heading_html" in item:
                    tag_match = any(safe_lower(tag) in safe_lower(item["heading_html"] or "") for tag in HEADING_TAGS | EXTRA_HEADING_TAGS)
                    if not tag_match:
                        if verbose:
                            logger.warning(f"[DOM_PARTS] Heading {i}: html '{item['heading_html']}' does not contain a valid heading tag.")
                        valid = False
                if section == "panels" and "panel_html" in item:
                    tag_match = any(safe_lower(tag) in safe_lower(item["panel_html"] or "") for tag in PANEL_TAGS)
                    if not tag_match:
                        if verbose:
                            logger.warning(f"[DOM_PARTS] Panel {i}: html '{item['panel_html']}' does not contain a valid panel tag.")
                        valid = False
    if warning_count > MAX_WARNINGS:
        logger.warning(f"[DOM_PARTS] {warning_count} items missing required fields (warnings suppressed after {MAX_WARNINGS}).")
    # Metadata checks
    meta = dom_parts.get("metadata", {})
    if not isinstance(meta, dict):
        if verbose:
            logger.warning("[DOM_PARTS] 'metadata' is not a dict.")
        valid = False
    else:
        scrape_time = meta.get("scrape_time")
        if scrape_time:
            try:
                datetime.datetime.strptime(scrape_time, "%Y-%m-%d %H:%M:%S")
            except Exception:
                if verbose:
                    logger.warning(f"[DOM_PARTS] metadata.scrape_time has invalid format: {scrape_time}")
                valid = False

    # Check selector_log is a list
    if "selector_log" in dom_parts and not isinstance(dom_parts["selector_log"], list):
        if verbose:
            logger.warning("[DOM_PARTS] 'selector_log' is not a list.")
        valid = False

    # Check tagged_segments and tagged_segments_with_attrs are lists
    for key in ["tagged_segments", "tagged_segments_with_attrs"]:
        if key in dom_parts and not isinstance(dom_parts[key], list):
            if verbose:
                logger.warning(f"[DOM_PARTS] '{key}' is not a list.")
            valid = False

    # Check url is a string
    if "url" in dom_parts and dom_parts["url"] is not None and not isinstance(dom_parts["url"], str):
        if verbose:
            logger.warning("[DOM_PARTS] 'url' is not a string.")
        valid = False

    # Check raw_html is a string
    if "raw_html" in dom_parts and dom_parts["raw_html"] is not None and not isinstance(dom_parts["raw_html"], str):
        if verbose:
            logger.warning("[DOM_PARTS] 'raw_html' is not a string.")
        valid = False

    # Check error is None or str
    if "error" in dom_parts and dom_parts["error"] is not None and not isinstance(dom_parts["error"], str):
        if verbose:
            logger.warning("[DOM_PARTS] 'error' is not a string or None.")
        valid = False

    if not valid and verbose:
        logger.error("[DOM_PARTS] Validation failed. Downstream consumers may not function correctly.")

    return valid

def scan_html_for_context(
    target_url,
    page,
    coordinator=None,
    debug=False,
    context_cache=None,
    model_name: Optional[str] = None,
    use_finetuned: bool = True,
    non_interactive=False,
    session_id=None,
    allow_duplicates=False,
    ml_threshold: float = 0.85  # Add default value for ml_threshold
) -> Dict[str, Any]:
    """
    Main pipeline entry: Efficient, dynamic, and feedback-driven HTML scanner.
    Leverages ContextCoordinator for context, ML model, and feedback logs.
    Robustly utilizes session_id, coordinator, allow_duplicates, and non_interactive for webapp GUI use.
    """
    from ..Context_Integration.context_organizer import ContextOrganizer
    if coordinator is None:
        from ..Context_Integration.context_coordinator import ContextCoordinator
        coordinator = ContextCoordinator()
    def extract_all_segment_html(html: str) -> List[str]:
        try:
            tree = HTMLParser(html)
            return [n.html for n in tree.root.traverse() if hasattr(n, "html")]
        except Exception:
            return []

    def diagnostics_and_filter(
        data: List[dict],
        field,
        max_title_len: int = 500,
        min_title_len: int = 2,
        allow_duplicates: bool = False,
        allow_empty: bool = False,
        allow_numeric_only: bool = False,
        allow_special_only: bool = False,
        log_sample_count: int = 5,
        dedupe_on: str = None,
        custom_validator=None,
        parallel: bool = False,
        session_id=None,
        coordinator=None,
        non_interactive=False
    ) -> List[Dict[str, Any]]:
        """
        Advanced diagnostics and filtering for extracted data.
        - Handles single or multiple fields.
        - Filters out empty, too short, too long, numeric-only, or special-char-only fields.
        - Optionally deduplicates on a field.
        - Allows custom validation logic.
        - Logs detailed diagnostics and samples of filtered items.
        - Optionally runs in parallel for large datasets.
        - Robustly uses session_id, coordinator, and non_interactive for segment prompting.
        """
        if not isinstance(data, list):
            logger.warning(f"[diagnostics_and_filter] Input data is not a list: {type(data)}")
            return []

        if not data:
            logger.warning(f"[{field}] No valid items extracted after validation.")
            return []

        def is_numeric_only(val):
            return isinstance(val, str) and val.strip().isdigit()

        def is_special_only(val):
            return isinstance(val, str) and bool(re.fullmatch(r'[\W_]+', val.strip()))

        def is_empty(val):
            return val is None or (isinstance(val, str) and not val.strip())

        def get_fields(d):
            return field if isinstance(field, list) else [field]

        seen = set()
        filtered = []
        filtered_out = []

        def filter_item(d):
            skip_reason = None
            for f in get_fields(d):
                val = safe_get(d, f, "")
                if is_empty(val):
                    if not allow_empty:
                        skip_reason = f"empty {f}"
                        break
                if isinstance(val, str) and len(val.strip()) < min_title_len:
                    skip_reason = f"too short {f}"
                    break
                if isinstance(val, str) and len(val.strip()) > max_title_len:
                    skip_reason = f"too long {f}"
                    break
                if is_numeric_only(val) and not allow_numeric_only:
                    skip_reason = f"numeric only {f}"
                    break
                if is_special_only(val) and not allow_special_only:
                    skip_reason = f"special chars only {f}"
                    break
                if custom_validator and not custom_validator(val, d):
                    skip_reason = f"custom validator failed for {f}"
                    break
            # Dedupe logic
            if not skip_reason and dedupe_on and not allow_duplicates:
                dedupe_val = safe_get(d, dedupe_on, None)
                if dedupe_val in seen:
                    skip_reason = f"duplicate {dedupe_on}"
                else:
                    seen.add(dedupe_val)
            return (d, skip_reason)

        # Parallel filtering for large datasets
        if parallel and len(data) > 1000:
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                results = list(executor.map(filter_item, data))
        else:
            results = [filter_item(d) for d in data]

        for d, reason in results:
            if reason is not None:
                filtered_out.append((d, reason))
            else:
                filtered.append(d)

        # Logging
        if filtered:
            try:
                avg_len = sum(
                    len(str(safe_get(d, safe_get_first(get_fields(d), "get_fields", None, logger, default=""), ""))) for d in filtered
                ) / len(filtered)
            except Exception:
                avg_len = 0
            logger.info(f"[{field}] Extracted {len(filtered)} items, avg field length: {avg_len:.1f}")
        else:
            logger.warning(f"[{field}] No items passed all filters.")

        if filtered_out:
            logger.warning(f"[{field}] Filtered out {len(filtered_out)} items due to validation.")
            for d, reason in filtered_out[:log_sample_count]:
                logger.warning(f"  [Filtered] {reason}: {str(d)[:100]}...")

        # Segment prompting (if coordinator and session_id are provided and not non_interactive)
        if coordinator and hasattr(coordinator, "segment_prompt") and session_id and not non_interactive:
            for d, reason in filtered_out:
                if coordinator is None:
                    from ..Context_Integration.context_coordinator import ContextCoordinator
                    coordinator = ContextCoordinator()
                if reason and safe_startswith(reason, "custom validator failed"):
                    coordinator.segment_prompt(d, session_id=session_id)

        return filtered

    # --- Main logic ---
    start_time = time.time()
    page_hash = get_page_hash(page)
    if context_cache is None:
        context_cache = load_context_cache_from_disk()

    try:
        html = getattr(page, "content", lambda: "")()
    except Exception:
        logger.warning("[SCAN_HTML] Exception when calling page.content(), using empty string.")
        html = ""
    if html is None:
        logger.warning("[SCAN_HTML] Page content is None, using empty string.")
        html = ""
    segment_htmls = extract_all_segment_html(html)
    segment_hashes = [segment_hash(h) for h in segment_htmls]
    fast_path_hits = [
        h for h in segment_hashes
        if h in context_cache and safe_get(context_cache[h], "ml_confidence", 0) > 0.95
    ]
    if len(fast_path_hits) == len(segment_hashes) and segment_hashes:
        logger.info("[FAST-PATH] All segments covered by cache. Skipping full scan.")
        fast_path_result = {h: context_cache[h] for h in segment_hashes}
        if coordinator is not None:
            coordinator.organize_and_enrich(fast_path_result)
        return fast_path_result
    if page_hash in context_cache:
        logger.info(f"[SCAN] Using cached context for {target_url}")
        logger.info("[bold green][CACHE] Entire context loaded from cache. Skipping scan.[/bold green]")
        cached_result = context_cache[page_hash]
        if coordinator is not None:
            coordinator.organize_and_enrich(cached_result)
        return cached_result
    try:
        page_url = safe_get(page, "url", None)
    except Exception:
        logger.warning("[SCAN_HTML] Exception when accessing page.url, using None.")
        page_url = None
    if not page_url:
        page_url = target_url

    context_result = {
        "raw_html": "",
        "tagged_segments": [],
        "tagged_segments_with_attrs": [],
        "metadata": {},
        "selector_log": [],
        "error": None,
        "url": page_url,
        "pattern_kb_matches": [],
        "segments_needing_review": [],
        "session_id": session_id,
        "coordinator": str(type(coordinator).__name__) if coordinator else None,
        "non_interactive": non_interactive,
    }
    if context_cache is not None:
        page_hash = get_page_hash(page)
        context_result.setdefault("page_hash", page_hash)
        context_result.setdefault("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
        context_cache[page_hash] = context_result
        save_context_cache_to_disk(context_cache)

    try:
        # --- 1. Get context library, pattern KB, and ML model from coordinator if available ---
        if coordinator:
            context_library = getattr(coordinator, "library", None)
            pattern_kb = getattr(coordinator, "pattern_kb", None)
            model = getattr(coordinator, "_semantic_model", None)
            if hasattr(coordinator, "get_feedback_pattern_kb"):
                feedback_kb = coordinator.get_feedback_pattern_kb()
                if feedback_kb:
                    if pattern_kb is None:
                        pattern_kb = []
                    pattern_kb.extend(feedback_kb)
                    pattern_kb = deduplicate_pattern_kb(pattern_kb)
        else:
            try:
                context_library = load_context_library(CONTEXT_LIBRARY_PATH)
                logger.debug("DEBUG: Loaded context library:", type(context_library))
                if not isinstance(context_library, dict):
                    logger.error("ERROR: Context library is not a dictionary. Check your context library loading logic.")
                    raise ValueError("Context library must be a dictionary. Check your context library loading logic.")
            except Exception:
                context_library = {}
            pattern_kb = load_pattern_kb()
            model = ModelRegistry.get_sentence_transformer(model_name=model_name, use_finetuned=use_finetuned)

        # --- 2. Extract segments with attributes and ML labels ---
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
            session_id=session_id,
            non_interactive=non_interactive,
            allow_duplicates=allow_duplicates,
            debug=debug
        )

        if (
            not segments_with_attrs
            or (isinstance(segments_with_attrs, list) and "error_info" in segments_with_attrs[0])
        ):
            error_info = safe_get(segments_with_attrs[0], "error_info", {}) if segments_with_attrs else {}
            logger.error(f"[SEGMENT EXTRACTION ERROR] {safe_get(error_info, 'error', 'Unknown error')}")
            context_result["tagged_segments_with_attrs"] = []
            context_result["tagged_segments"] = []
            context_result["error"] = safe_get(error_info, "error", "Unknown error")
        else:
            context_result["tagged_segments_with_attrs"] = segments_with_attrs
            context_result["tagged_segments"] = [safe_get(seg, "html", "") for seg in segments_with_attrs]

        # --- 3. Robust Contest Extraction ---
        contests = []
        for seg in _extract_segments_by_label(segments_with_attrs, "contest"):
            for possible in split_possible_contests(safe_get(seg, "text", "")):
                seg_year, seg_type, cleaned_title, _ = extract_year_and_type(possible, url=target_url)
                if cleaned_title and not any(
                    safe_get(c, "title", "") == cleaned_title and safe_get(c, "year", None) == seg_year and safe_get(c, "type_", None) == seg_type
                    for c in contests
                ):
                    contests.append({
                        "title": cleaned_title,
                        "state": safe_get(context_result, "state", None),
                        "county": safe_get(context_result, "county", None),
                        "year": seg_year,
                        "type_": seg_type,
                        "segment_hash": safe_get(seg, "segment_hash", None),
                    })
        contests = [c for c in contests if safe_get(c, "title", None)]
        context_result["contests"] = diagnostics_and_filter(
            contests, ["title", "year", "type_"], allow_duplicates=allow_duplicates, session_id=session_id, coordinator=coordinator, non_interactive=non_interactive
        )

        # --- Panels ---
        panels = []
        for seg in _extract_segments_by_label(segments_with_attrs, "panel"):
            panel_text = _extract_clean_text(safe_get(seg, "raw_html", safe_get(seg, "html", "")))
            if panel_text:
                panels.append({
                    "panel_text": panel_text,
                    "panel_html": safe_get(seg, "raw_html", safe_get(seg, "html", "")),
                    "segment_hash": safe_get(seg, "segment_hash", None),
                })
        context_result["panels"] = diagnostics_and_filter(
            panels, "panel_text", allow_duplicates=allow_duplicates, session_id=session_id, coordinator=coordinator, non_interactive=non_interactive
        )

        # --- Tables ---
        tables = []
        for seg in _extract_segments_by_label(segments_with_attrs, "results_table"):
            table_text = _extract_clean_text(safe_get(seg, "raw_html", safe_get(seg, "html", "")))
            if table_text:
                tables.append({
                    "table_text": table_text,
                    "table_html": safe_get(seg, "raw_html", safe_get(seg, "html", "")),
                    "year": None,
                    "type_": None,
                    "segment_hash": safe_get(seg, "segment_hash", None),
                })
        context_result["tables"] = diagnostics_and_filter(
            tables, "table_text", allow_duplicates=allow_duplicates, session_id=session_id, coordinator=coordinator, non_interactive=non_interactive
        )

        # --- Candidate Panels ---
        candidate_panels = []
        for seg in _extract_segments_by_label(segments_with_attrs, "candidate_panel"):
            candidate_panel_text = _extract_clean_text(safe_get(seg, "raw_html", safe_get(seg, "html", "")))
            if candidate_panel_text:
                candidate_panels.append({
                    "candidate_panel_text": candidate_panel_text,
                    "candidate_panel_html": safe_get(seg, "raw_html", safe_get(seg, "html", "")),
                    "year": None,
                    "type_": None,
                    "segment_hash": safe_get(seg, "segment_hash", None),
                })
        context_result["candidate_panels"] = diagnostics_and_filter(
            candidate_panels, "candidate_panel_text", allow_duplicates=allow_duplicates, session_id=session_id, coordinator=coordinator, non_interactive=non_interactive
        )

        # --- Location Panels ---
        location_panels = []
        for seg in _extract_segments_by_label(segments_with_attrs, "location_panel"):
            location_panel_text = _extract_clean_text(safe_get(seg, "raw_html", safe_get(seg, "html", "")))
            if location_panel_text:
                location_panels.append({
                    "location_panel_text": location_panel_text,
                    "location_panel_html": safe_get(seg, "raw_html", safe_get(seg, "html", "")),
                    "year": None,
                    "type_": None,
                    "segment_hash": safe_get(seg, "segment_hash", None),
                    "county": safe_get(context_result, "county", None),
                })
        context_result["location_panels"] = diagnostics_and_filter(
            location_panels, "location_panel_text", allow_duplicates=allow_duplicates, session_id=session_id, coordinator=coordinator, non_interactive=non_interactive
        )

        # --- Headings ---
        headings = []
        for seg in _extract_segments_by_label(segments_with_attrs, "heading"):
            heading_text = _extract_clean_text(safe_get(seg, "raw_html", safe_get(seg, "html", "")))
            if heading_text:
                headings.append({
                    "heading_text": heading_text,
                    "heading_html": safe_get(seg, "raw_html", safe_get(seg, "html", "")),
                    "segment_hash": safe_get(seg, "segment_hash", None),
                    "heading_type": None,
                })
        context_result["headings"] = diagnostics_and_filter(
            headings, "heading_text", allow_duplicates=allow_duplicates, session_id=session_id, coordinator=coordinator, non_interactive=non_interactive
        )

        # --- Ballot Types ---
        ballot_types = []
        for seg in _extract_segments_by_label(segments_with_attrs, "ballot_types"):
            ballot_types_text = _extract_clean_text(safe_get(seg, "raw_html", safe_get(seg, "html", "")))
            if ballot_types_text:
                ballot_types.append({
                    "ballot_types_text": ballot_types_text,
                    "ballot_types_html": safe_get(seg, "raw_html", safe_get(seg, "html", "")),
                    "year": None,
                    "type_": None,
                    "segment_hash": safe_get(seg, "segment_hash", None),
                })
        context_result["ballot_types"] = diagnostics_and_filter(
            ballot_types, "ballot_types_text", allow_duplicates=allow_duplicates, session_id=session_id, coordinator=coordinator, non_interactive=non_interactive
        )

        election_types = []
        for seg in _extract_segments_by_label(segments_with_attrs, "ballot_types"):
            ballot_types_text = _extract_clean_text(safe_get(seg, "raw_html", safe_get(seg, "html", "")))
            if ballot_types_text:
                etype = None
                if coordinator and hasattr(coordinator, "extract_field"):
                    etype = coordinator.extract_field("election_types", text=ballot_types_text)
                if etype:
                    election_types.append(etype)
        context_result["election_types"] = election_types

        # Defensive: Ensure election_types is always a list
        if "election_types" not in context_result or not isinstance(context_result["election_types"], list):
            context_result["election_types"] = []

        # --- Results Timestamps ---
        results_timestamps = []
        for seg in _extract_segments_by_label(segments_with_attrs, "results_timestamp"):
            timestamp_text = _extract_clean_text(safe_get(seg, "raw_html", safe_get(seg, "html", "")))
            if timestamp_text:
                results_timestamps.append({
                    "timestamp_text": timestamp_text,
                    "timestamp_html": safe_get(seg, "raw_html", safe_get(seg, "html", "")),
                    "segment_hash": safe_get(seg, "segment_hash", None),
                })
        context_result["results_timestamps"] = diagnostics_and_filter(
            results_timestamps, "timestamp_text", allow_duplicates=allow_duplicates, session_id=session_id, coordinator=coordinator, non_interactive=non_interactive
        )

        # --- Party Labels ---
        party_labels = []
        for seg in _extract_segments_by_label(segments_with_attrs, "party_label"):
            party_label_text = _extract_clean_text(safe_get(seg, "raw_html", safe_get(seg, "html", "")))
            if party_label_text:
                party_labels.append({
                    "party_label_text": party_label_text,
                    "party_label_html": safe_get(seg, "raw_html", safe_get(seg, "html", "")),
                    "segment_hash": safe_get(seg, "segment_hash", None),
                })
        context_result["party_labels"] = diagnostics_and_filter(
            party_labels, "party_label_text", allow_duplicates=allow_duplicates, session_id=session_id, coordinator=coordinator, non_interactive=non_interactive
        )

        # --- Vote Methods ---
        vote_methods = []
        for seg in _extract_segments_by_label(segments_with_attrs, "vote_method"):
            vote_method_text = _extract_clean_text(safe_get(seg, "raw_html", safe_get(seg, "html", "")))
            if vote_method_text:
                vote_methods.append({
                    "vote_method_text": vote_method_text,
                    "vote_method_html": safe_get(seg, "raw_html", safe_get(seg, "html", "")),
                    "segment_hash": safe_get(seg, "segment_hash", None),
                })
        context_result["vote_methods"] = diagnostics_and_filter(
            vote_methods, "vote_method_text", allow_duplicates=allow_duplicates, session_id=session_id, coordinator=coordinator, non_interactive=non_interactive
        )

        # --- Propagate best year/type to all sections ---
        def propagate_year_type(items, year, type_):
            for item in items:
                if isinstance(item, dict):
                    if "year" not in item or item["year"] is None:
                        item["year"] = year
                    if "type_" not in item or item["type_"] is None:
                        item["type_"] = type_
                    # Ensure type_ and election_types are synced
                    _sync_type_and_election_types(item, fallback_types=[type_] if type_ else None, fallback_type=type_)

        best_year = safe_get_first([safe_get(c, "year", None) for c in contests if safe_get(c, "year", None)], "best_year", None, logger)
        best_type = safe_get_first([safe_get(c, "type_", None) for c in contests if safe_get(c, "type_", None)], "best_type", None, logger)
        for section in ["tables", "candidate_panels", "location_panels", "ballot_types"]:
            propagate_year_type(context_result.get(section, []), best_year, best_type)

        # Defensive: Ensure all lists are present and are lists, even if empty
        for key in [
            "contests", "panels", "tables", "candidate_panels", "location_panels",
            "headings", "ballot_types", "results_timestamps", "party_labels", "vote_methods"
        ]:
            if key not in context_result or not isinstance(context_result[key], list):
                context_result[key] = []

        # Defensive: If any required section is empty, log a warning (for downstream [0] access)
        required_sections = ["contests", "panels", "tables", "candidate_panels", "location_panels"]
        for section in required_sections:
            if not context_result.get(section):
                logger.warning(f"[DOM_PARTS] No items found in '{section}'. Downstream code should check for empty lists before accessing [0].")

        # --- 4. ML-driven DOM pattern clustering and tagging ---
        pattern_matches = []
        segments_needing_review = []
        seen = set()
        unique_segments = []
        for seg in segments_with_attrs:
            html_norm = _normalize_html_for_hash(safe_get(seg, "html", ""))
            if html_norm not in seen:
                seen.add(html_norm)
                unique_segments.append(seg)
        segments_with_attrs = unique_segments
        for seg in segments_with_attrs:
            if safe_get(seg, "ml_confidence", 0.0) < 0.7 or safe_get(seg, "ml_label", "unknown") == "unknown":
                user_label = None
                if coordinator and hasattr(coordinator, "auto_label_segment"):
                    try:
                        user_label = coordinator.auto_label_segment(seg)
                    except Exception:
                        user_label = None
                if not user_label:
                    user_label = prompt_for_segment_label(
                        seg,
                        context_library=context_library,
                        session_id=session_id,
                        non_interactive=non_interactive
                    )
                seg["ml_label"] = user_label
                seg["ml_confidence"] = 1.0
                html_val = safe_get(seg, "html", "")
                if not isinstance(html_val, str):
                    html_val = str(html_val)
                seg["pattern_id"] = f"pattern_{hashlib.sha256(html_val.encode('utf-8')).hexdigest()[:10]}"
                emb = get_segment_embedding(model, seg, cache_hits=embedding_cache_hits, cache_misses=embedding_cache_misses)
                if emb is not None:
                    emb = emb.tolist()
                kb_entry = {
                    "pattern_id": seg["pattern_id"],
                    "label": user_label,
                    "embedding": emb,
                    "example_html": safe_get(seg, "html", "")[:500],
                    "source_url": page_url,
                    "timestamp": time.time(),
                    "session_id": session_id,
                    "coordinator": str(type(coordinator).__name__) if coordinator else None,
                    "non_interactive": non_interactive,
                }
                append_pattern_kb(kb_entry)
                append_feedback_log({
                    "pattern_id": seg["pattern_id"],
                    "label": user_label,
                    "html": safe_get(seg, "html", "")[:500],
                    "source_url": page_url,
                    "timestamp": time.time(),
                    "session_id": session_id,
                    "coordinator": str(type(coordinator).__name__) if coordinator else None,
                    "non_interactive": non_interactive,
                })
                segments_needing_review.append(seg)

                if context_library is not None and safe_get(seg, "segment_hash", None):
                    update_context_library(
                        CONTEXT_LIBRARY_PATH,
                        lambda lib: safe_append_cached_segment(
                            lib,
                            safe_get(seg, "segment_hash", None),
                            user_label
                        )
                    )
                    valid_hashes = set(safe_get(s, "segment_hash", None) for s in context_library.get("cached_segments", []))
                    prune_embedding_cache(valid_hashes)
            else:
                pattern_matches.append({
                    "pattern_id": seg["pattern_id"],
                    "label": seg["ml_label"],
                    "confidence": seg["ml_confidence"],
                    "segment_html": safe_get(seg, "html", "")[:200],
                    "session_id": session_id,
                    "coordinator": str(type(coordinator).__name__) if coordinator else None,
                    "non_interactive": non_interactive,
                })

        context_result["pattern_kb_matches"] = pattern_matches
        context_result["segments_needing_review"] = segments_needing_review

        # --- 5. Dynamic tagging and context enrichment ---
        selector_log = set()
        for seg in segments_with_attrs:
            if safe_get(seg, "id", None):
                selector_log.add(f'#{safe_get(seg, "id", "")}')
            for cls in safe_get(seg, "classes", []):
                selector_log.add(f'.{cls}')
            selector_log.add(safe_lower(safe_get(seg, "tag", "")))
            if "semantic_tags" not in seg:
                seg["semantic_tags"] = []
            if safe_get(seg, "ml_label", "") not in ("unknown", "ignore"):
                safe_append(seg["semantic_tags"], safe_get(seg, "ml_label", ""), logger)
        context_result["selector_log"] = sorted(selector_log)

        safe_update(context_result.get("metadata", {}), {
            "source_url": page_url,
            "scrape_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pattern_kb_size": len(pattern_kb) if pattern_kb else 0,
            "session_id": session_id,
            "coordinator": str(type(coordinator).__name__) if coordinator else None,
            "non_interactive": non_interactive,
        }, logger)

        if debug:
            logger.debug("\n[orange][DEBUG] Extracted HTML segments with ML labels:[/orange]")
            for seg in segments_with_attrs:
                logger.info(f"{safe_get(seg, 'tag', '')} {safe_get(seg, 'attrs', {})} [label={safe_get(seg, 'ml_label', '')}, conf={safe_get(seg, 'ml_confidence', 0.0):.2f}] {safe_get(seg, 'html', '')[:80]}{'...' if len(safe_get(seg, 'html', '')) > 80 else ''}")
            if segments_needing_review:
                logger.debug(f"\n[red][DEBUG] {len(segments_needing_review)} segments flagged for review.[/red]")

        # --- 6. Update context library with new segments for future runs ---
        if context_library is not None:
            if "cached_segments" not in context_library:
                context_library["cached_segments"] = []
            known_hashes = set(safe_get(seg, "segment_hash", None) for seg in context_library["cached_segments"])
            for seg in segments_with_attrs:
                if safe_get(seg, "segment_hash", None) and safe_get(seg, "segment_hash", None) not in known_hashes:
                    safe_append(
                        context_library.get("cached_segments"),
                        {
                            "segment_hash": safe_get(seg, "segment_hash", None),
                            "ml_label": safe_get(seg, "ml_label", None),
                            "ml_confidence": safe_get(seg, "ml_confidence", None),
                            "pattern_id": safe_get(seg, "pattern_id", None),
                            "session_id": session_id,
                            "coordinator": str(type(coordinator).__name__) if coordinator else None,
                            "non_interactive": non_interactive,
                        },
                        logger
                    )
            update_context_library(
                CONTEXT_LIBRARY_PATH,
                lambda lib: safe_extend(
                    lib,
                    "cached_segments",
                    [
                        {
                            "segment_hash": safe_get(seg, "segment_hash", None),
                            "ml_label": safe_get(seg, "ml_label", None),
                            "ml_confidence": safe_get(seg, "ml_confidence", None),
                            "pattern_id": safe_get(seg, "pattern_id", None),
                            "session_id": session_id,
                            "coordinator": str(type(coordinator).__name__) if coordinator else None,
                            "non_interactive": non_interactive,
                        }
                        for seg in segments_with_attrs
                        if safe_get(seg, "segment_hash", None) and safe_get(seg, "segment_hash", None) not in known_hashes
                    ]
                )
            )
            valid_hashes = set(safe_get(seg, "segment_hash", None) for seg in context_library.get("cached_segments", []))
            prune_embedding_cache(valid_hashes)
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"[SCAN ERROR] HTML parsing failed: {e}\n{tb}")
        context_result["error"] = f"[SCAN ERROR] HTML parsing failed: {e}\n{tb}"

    if context_cache is not None:
        context_cache[page_hash] = context_result
        save_context_cache_to_disk(context_cache)
    if embedding_cache_hits and not embedding_cache_misses:
        logger.info(f"[bold green][CACHE] All segment embeddings loaded from cache.[/bold green]")
    elif embedding_cache_hits:
        logger.warning(f"[yellow][CACHE] {len(embedding_cache_hits)} embeddings loaded from cache, {len(embedding_cache_misses)} computed.[/yellow]")
    logger.info(f"[PROFILE] scan_html_for_context completed in {time.time() - start_time:.2f} seconds.")

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
        "raw_html": safe_get(context_result, "raw_html", ""),
        "error": safe_get(context_result, "error", None),
        "url": safe_get(context_result, "url", None),
        "session_id": session_id,
        "coordinator": str(type(coordinator).__name__) if coordinator else None,
        "non_interactive": non_interactive,
    }
    # Defensive: Ensure all dom_parts lists are lists, even if empty
    for key in [
        "contests", "panels", "tables", "candidate_panels", "location_panels",
        "headings", "ballot_types", "results_timestamps", "party_labels", "vote_methods",
        "pattern_kb_matches", "segments_needing_review", "selector_log",
        "tagged_segments", "tagged_segments_with_attrs"
    ]:
        if key not in dom_parts or not isinstance(dom_parts[key], list):
            dom_parts[key] = []

    # --- Advanced DOM validation and enrichment ---
    valid = validate_dom_parts(dom_parts)
    if not valid:
        logger.error("[DOM_PARTS] Validation failed. Downstream consumers may not function correctly.")

    context_result["dom_parts"] = dom_parts

    # --- Advanced DOM enrichment and organization for downstream consumers ---
    organizer = ContextOrganizer()
    segments = dom_parts.get("tagged_segments_with_attrs", [])
    if segments:
        dom_tree = organizer.build_dom_tree(segments)
        context_result["dom_tree"] = dom_tree

        # Group nodes by label for fast lookup and context enrichment
        label_groups = organizer.group_nodes_by_label(dom_tree["nodes"], label_field="ml_label")
        context_result["dom_label_groups"] = label_groups

        # Panels and tables association for context-aware extraction
        panels_and_tables = organizer.get_panels_and_tables(dom_tree)
        context_result["dom_panels_and_tables"] = panels_and_tables

        # Attach enrichment to each segment for downstream context
        for seg in segments:
            seg["dom_node"] = dom_tree["nodes"][seg["_idx"]] if seg["_idx"] < len(dom_tree["nodes"]) else None
            seg["label_group"] = label_groups.get(safe_get(seg, "ml_label", ""), [])
            seg["panel_group"] = None
            for panel in panels_and_tables:
                if seg["_idx"] in safe_get(panel, "panel_indices", []) or seg["_idx"] in safe_get(panel, "table_indices", []):
                    seg["panel_group"] = panel
                    break

        # Add HTML samples for review/debug
        N = min(5, len(dom_tree["nodes"]))
        context_result["dom_node_html_samples"] = [
            organizer.extract_html_by_idx(dom_tree["nodes"], i, safe_get(context_result, "raw_html", ""))
            for i in range(N)
        ]
        context_result["dom_subtree_html_samples"] = [
            organizer.extract_subtree_html(dom_tree["nodes"], i, safe_get(context_result, "raw_html", ""))
            for i in range(N)
        ]

        logger.info(f"[DOM ENRICHMENT] Added dom_tree, label_groups, panels_and_tables, and HTML samples to context_result.")

    if not dom_parts or not dom_parts.get("tagged_segments_with_attrs"):
        logger.error("[DOM_PARTS] dom_parts is empty or missing tagged_segments_with_attrs. Downstream consumers will not function.")
    else:
        logger.debug(f"[DOM_PARTS] dom_parts keys: {list(dom_parts.keys())}, tagged_segments_with_attrs count: {len(dom_parts['tagged_segments_with_attrs'])}")

    # --- Coordinator-driven organization and enrichment ---
    if coordinator is not None:
        organized = coordinator.organize_and_enrich(context_result)
        if not organized or "dom_parts" not in organized or not organized["dom_parts"]:
            logger.error("[DOM_PARTS] dom_parts missing after organize_and_enrich.")
        else:
            dom_parts_keys = []
            if isinstance(organized, dict) and "dom_parts" in organized and isinstance(organized["dom_parts"], dict):
                dom_parts_keys = list(organized["dom_parts"].keys())
            logger.debug(f"[DOM_PARTS] dom_parts successfully organized with keys: {dom_parts_keys}")

    # --- Advanced debug logging for DOM review ---
    if debug and "dom_tree" in context_result:
        dom_tree = context_result["dom_tree"]
        nodes = safe_get(dom_tree, "nodes", [])
        for idx in range(min(5, len(nodes))):
            node = nodes[idx] if idx < len(nodes) else None
            if node is None:
                logger.warning(f"[DOM DEBUG] Node {idx} is None.")
                continue
            html_snippet = organizer.extract_html_by_idx(nodes, idx, safe_get(context_result, "raw_html", ""))
            logger.info(f"[DOM DEBUG] Node {idx} HTML: {html_snippet[:100]}")
            subtree_html = organizer.extract_subtree_html(nodes, idx, safe_get(context_result, "raw_html", ""))
            logger.info(f"[DOM DEBUG] Subtree HTML for node {idx}: {subtree_html[:200]}")

    # Sync contests
    for contest in safe_get(context_result, "contests", []):
        _sync_type_and_election_types(contest)

    # Get best contest type/election_types for fallback
    best_contest = safe_get(context_result, "contests", [{}])[0] if safe_get(context_result, "contests", []) else {}
    best_type = safe_get(best_contest, "type_", None)
    best_election_types = safe_get(best_contest, "election_types", [])

    # Sync other sections
    for section in ["tables", "candidate_panels", "location_panels", "ballot_types"]:
        for item in safe_get(context_result, section, []):
            _sync_type_and_election_types(item, fallback_types=best_election_types, fallback_type=best_type)

    # Sync top-level context_result
    _sync_type_and_election_types(context_result, fallback_types=best_election_types, fallback_type=best_type)

    return context_result