import hashlib
import orjson
import os
import re
import time
from typing import Dict, Any, List, Optional
from ..config import CONTEXT_LIBRARY_PATH, CACHE_DIR, LOG_DIR
from ..utils.download_utils import download_file
from ..utils.format_router import route_format_handler 
from ..utils.shared_logic import infer_state_county_from_url, update_context_library
from ..utils.shared_logger import logger
from rich import print as rprint
from rich.console import Console
from ..utils.user_prompt import prompt_user_input
from selectolax.parser import HTMLParser
from ..utils.model_registry import ModelRegistry
from ..bots.librarian import (
    HTML_TAGS, PANEL_TAGS, HEADING_TAGS, CUSTOM_ATTR_PATTERNS, DISTRICT_REGEX, LOCATION_KEYWORDS, CANDIDATE_KEYWORDS, BALLOT_TYPES,
    extend_panel_tags, extend_heading_tags, extend_html_tags, extend_custom_attr_patterns,
    log_unknown_tag, log_unknown_attr, get_canonical_segment_label, cache_segment_label, get_cached_segment_label, ROOT_CONTAINER_TAGS,
    ALWAYS_IGNORE_TAGS, ALWAYS_IGNORE_CLASSES, ALWAYS_IGNORE_IDS, ICON_CLASSES, ICON_TAGS, BUTTON_CLASSES,
    HEADING_CLASSES, PANEL_CLASSES, TIMESTAMP_CLASSES, STRUCTURAL_TAGS, TIMESTAMP_ID_PATTERNS, TIMESTAMP_ATTRS,
    STRUCTURAL_TAGS
)
ENABLE_SEGMENT_LABEL_PROMPT = os.getenv("ENABLE_SEGMENT_LABEL_PROMPT", "true").lower() == "true"
import numpy as np
console = Console()
from bs4 import BeautifulSoup, Tag
from ..utils.embedding_cache import (
    save_embedding, load_embedding, get_embedding_from_memory
)
import traceback
from difflib import get_close_matches
embedding_cache_hits = set()
embedding_cache_misses = set()

import threading

_LABEL_CACHE_FILENAME = "segment_label_cache.json"
_LABEL_CACHE_LOCK = threading.Lock()
_LABEL_CACHE = None
_context_cache = None
_pattern_kb_cache = None


def robust_orjson_loads(val):
    """Load JSON robustly from either bytes or str."""
    if isinstance(val, bytes):
        return orjson.loads(val)
    elif isinstance(val, str):
        return orjson.loads(val.encode("utf-8"))
    else:
        raise TypeError(f"Cannot decode type {type(val)} with orjson")

def _get_label_cache_path():
    path = safe_cache_path(_LABEL_CACHE_FILENAME)
    return path

def _load_label_cache():
    global _LABEL_CACHE
    if _LABEL_CACHE is not None:
        return _LABEL_CACHE
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

def _save_label_cache():
    global _LABEL_CACHE
    path = _get_label_cache_path()
    with open(path, "wb") as f:
        f.write(orjson.dumps(_LABEL_CACHE, option=orjson.OPT_INDENT_2))

def cache_segment_label(seg_hash, label):
    """Persistently cache the label for a segment (by robust segment hash)."""
    with _LABEL_CACHE_LOCK:
        cache = _load_label_cache()
        cache[seg_hash] = {"label": label, "timestamp": int(time.time())}
        _save_label_cache()

def get_cached_segment_label(seg_hash):
    """Retrieve a cached label for a segment, or None if not found."""
    with _LABEL_CACHE_LOCK:
        cache = _load_label_cache()
        entry = cache.get(seg_hash, {})
        if entry:
            return entry.get("label", [])
        return None

# Example: dynamically extend from learning/feedback
extend_panel_tags(["custom-panel"])
extend_custom_attr_patterns([r"^x-data-"])
extend_heading_tags(["custom-heading", "special-h2"])
extend_html_tags(["custom-element", "widget"])

def save_context_library(context_library, context_library_path=CONTEXT_LIBRARY_PATH):
    with open(context_library_path, "wb") as f:
        f.write(orjson.dumps(context_library, option=orjson.OPT_INDENT_2))
        
def load_additional_tags_from_context_library():
    tags = set()
    if os.path.exists(CONTEXT_LIBRARY_PATH):
        with open(CONTEXT_LIBRARY_PATH, "rb") as f:
            context_lib = robust_orjson_loads(f.read())
            for key in ["panel_tags", "table_tags", "section_keywords"]:
                if key in context_lib and isinstance(context_lib[key], list):
                    tags.update([t.lower() for t in context_lib[key] if isinstance(t, str)])
    return tags
HTML_TAGS |= load_additional_tags_from_context_library()

def safe_cache_path(filename: str) -> str:
    filename = _sanitize_log_filename(filename)
    cache_folder = CACHE_DIR
    os.makedirs(cache_folder, exist_ok=True)
    full_path = os.path.join(cache_folder, filename)
    if not os.path.abspath(full_path).startswith(os.path.abspath(cache_folder)):
        raise ValueError("Unsafe cache path detected!")
    return full_path

def safe_log_path(filename: str, log_dir: str = "log") -> str:
    filename = _sanitize_log_filename(filename)
    log_folder = LOG_DIR
    os.makedirs(log_folder, exist_ok=True)
    full_path = os.path.join(log_folder, filename)
    if not os.path.abspath(full_path).startswith(os.path.abspath(log_folder)):
        raise ValueError("Unsafe log path detected!")
    return full_path

def _sanitize_log_filename(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_\-\.]', '_', name)

def _normalize_html_for_hash(html: str, maxlen: int = 256) -> str:
    """
    Normalize and truncate HTML for hashing: collapse whitespace, strip, and limit length.
    Remove dynamic attributes (e.g., ng-*, _ngcontent-*, timestamps, random ids) for robustness.
    """
    # Remove Angular and similar dynamic attributes
    html = re.sub(r'\s(_ngcontent-[^=]+|ng-version|ng-star-inserted|_nghost-[^=]+|_ngcontent-[^=]+|aria-checked|tabindex|style|data-[^=]+|id|class)="[^"]*"', '', html)
    # Remove timestamps and numbers that look like datetimes
    html = re.sub(r'\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}', '', html)
    html = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}', '', html)
    html = re.sub(r'\d{1,2}:\d{2}(:\d{2})? ?(am|pm|AM|PM)?', '', html)
    # Collapse whitespace
    html = re.sub(r'\s+', ' ', html.strip())
    return html[:maxlen]

def extract_attrs_bs4(bs4_tag: Tag) -> Dict[str, Any]:
    """
    Extract attributes from a BeautifulSoup Tag object, including data-* attributes.
    Returns a dictionary of attribute names to values.
    """
    attrs = {}
    for k, v in bs4_tag.attrs.items():
        if isinstance(v, list):
            attrs[k] = " ".join(v)
        elif v is None:
            attrs[k] = True
        else:
            attrs[k] = v
        log_unknown_attr(k)
    # Include data-* attributes
    for k, v in bs4_tag.attrs.items():
        if k.startswith("data-"):
            attrs[k] = v
    return attrs

def extract_custom_attrs(attrs: Dict[str, Any], include_data: bool = True) -> Dict[str, Any]:
    """Extract custom attributes (data-*, aria-*, role, etc.) based on dynamic patterns."""
    custom = {}
    for k, v in attrs.items():
        for pat in CUSTOM_ATTR_PATTERNS:
            if pat.match(k):
                custom[k] = v
                break
        else:
            log_unknown_attr(k)
    return custom

def is_trivial_segment(seg):
    html = seg.get("html", "")
    tag = seg.get("tag", "")
    if not html or not html.strip():
        return True
    if tag in {"br", "hr", "wbr"} and not html.strip():
        return True
    if html.strip() in {"&nbsp;", "&#160;"}:
        return True
    # Decorative/icon-only spans
    classes = [c.lower() for c in seg.get("classes", [])]
    if tag == "span" and len(classes) > 0 and all("icon" in cls for cls in classes) and not re.sub(r"<[^>]+>", "", html).strip():
        return True
    return False

def extract_tagged_segments_with_attrs(
    html: str,
    context_cache: Optional[Dict[str, Any]] = None,
    include_data_attrs: bool = True,
    fallback_on_error: bool = True,
    model_name: Optional[str] = None,
    use_finetuned: bool = True,
    pattern_kb: list = None,
    ml_threshold: float = 0.85,
    context_library: dict = None
) -> List[Dict[str, Any]]:
    """
    Extracts DOM segments with attributes and ML-driven semantic labels.
    Recursively walks the DOM tree (using selectolax or BeautifulSoup fallback),
    collecting all relevant segments, their attributes, and context relationships.
    Each segment gets: ml_label, ml_confidence, pattern_id.
    Trivial/structural segments are always labeled "ignore" and never prompt the user.
    Batch embedding is used for non-trivial segments for speed.
    All file/directory paths are constructed using config.py constants only.
    """
    if context_cache is not None:
        clean_cache_inplace(context_cache)
    def get_cached_segment(tag, attrs, html_snippet):
        cache = load_context_cache_from_disk()
        cache = [e for e in cache if isinstance(e, dict)]
        for entry in cache:
            if entry.get("tag", []) != tag:
                continue
            if entry.get("attrs", []) != attrs:
                continue
            if "html_snippet" in entry and html_snippet.startswith(entry["html_snippet"]):
                return entry
        attrs_sorted = {k: attrs[k] for k in sorted(attrs)}
        key = hashlib.sha256((tag + orjson.dumps(attrs_sorted, option=orjson.OPT_SORT_KEYS).decode() + html_snippet[:200]).encode("utf-8")).hexdigest()
        if context_cache and key in context_cache:
            return context_cache[key]
        if pattern_kb:
            for entry in pattern_kb:
                if not isinstance(entry, dict):
                    logger.warning(f"Non-dict entry in cache: {entry!r}")
                    continue
                if entry.get("segment_hash", []) == key:
                    return entry
        if context_library:
            for seg in context_library.get("cached_segments", []):
                if seg.get("segment_hash", []) == key:
                    return seg
        return None

    def label_segment(seg, emb=None):
        # Trivial segments: always ignore, cache label
        if is_trivial_segment(seg):
            seg["ml_label"] = "ignore"
            seg["ml_confidence"] = 1.0
            seg["pattern_id"] = None
            if seg.get("html", []):
                cache_segment_label(seg["html"], "ignore")
            return seg
        # Try cache/context KB
        cached = get_cached_segment(seg["tag"], seg["attrs"], seg["html"])
        if isinstance(cached, list):
            cached = [e for e in cached if isinstance(e, dict)]
            if cached:
                cached = cached[0]
            else:
                cached = None
        if cached:
            seg["ml_label"] = cached.get("ml_label", "unknown")
            seg["ml_confidence"] = cached.get("ml_confidence", 1.0)
            seg["pattern_id"] = cached.get("pattern_id", [])
            seg["segment_hash"] = cached.get("segment_hash", [])
            return seg
        # ML-driven labeling (optionally use precomputed embedding)
        if emb is not None:
            best_label = "unknown"
            best_conf = 0.0
            best_pattern_id = None
            for entry in pattern_kb:
                kb_emb = np.array(entry.get("embedding", []))
                if kb_emb.shape != emb.shape:
                    continue
                sim = float(np.dot(emb, kb_emb) / (np.linalg.norm(emb) * np.linalg.norm(kb_emb) + 1e-8))
                if sim > best_conf:
                    best_conf = sim
                    best_label = entry.get("label", "unknown")
                    best_pattern_id = entry.get("pattern_id", [])
            if best_conf < ml_threshold:
                seg["ml_label"] = "unknown"
                seg["ml_confidence"] = best_conf
                seg["pattern_id"] = None
            else:
                seg["ml_label"] = best_label
                seg["ml_confidence"] = best_conf
                seg["pattern_id"] = best_pattern_id
            return seg
        # Fallback: single-segment ML
        label, confidence, pattern_id = ml_classify_segment(seg, model, pattern_kb, threshold=ml_threshold)
        seg["ml_label"] = label
        seg["ml_confidence"] = confidence
        seg["pattern_id"] = pattern_id
        return seg

    start_time = time.time()
    segments: List[Dict[str, Any]] = []
    heading_tags = HEADING_TAGS
    panel_tags = PANEL_TAGS
    if pattern_kb is not None and isinstance(pattern_kb, list):
        pattern_kb[:] = [e for e in pattern_kb if isinstance(e, dict)]
    if context_library is not None and isinstance(context_library, dict):
        if "cached_segments" in context_library and isinstance(context_library["cached_segments"], list):
            context_library["cached_segments"] = [e for e in context_library["cached_segments"] if isinstance(e, dict)]
    # Load ML model and pattern KB/context only once
    model = ModelRegistry.get_sentence_transformer(model_name=model_name, use_finetuned=use_finetuned)
    if model is None:
        logger.error("[ERROR] SentenceTransformer model could not be loaded. Check model path and files.")
        raise RuntimeError("SentenceTransformer model could not be loaded. Aborting segment extraction.")
    if pattern_kb is None:
        pattern_kb = load_pattern_kb()
    if context_library is None:
        context_library = {}
    if context_cache is None:
        context_cache = load_context_cache_from_disk()

    try:
        tree = HTMLParser(html)
        def walk(node, parent_idx=None, heading_idx=None, panel_idx=None):
            tag = node.tag
            if not tag or tag.lower() not in HTML_TAGS:
                log_unknown_tag(tag)
                for child in node.iter(include_text=True):
                    walk(child, parent_idx, heading_idx, panel_idx)
                return
            attrs = dict(node.attributes)
            if include_data_attrs:
                attrs.update({k: v for k, v in node.attributes.items() if k.startswith("data-")})
            for k in attrs:
                log_unknown_attr(k)
            classes = attrs.get("class", "").split() if "class" in attrs else []
            id_ = attrs.get("id", "")
            is_button = tag == "button" or (tag == "input" and attrs.get("type", "").lower() in ["button", "submit"])
            is_clickable = is_button or tag == "a" or "onclick" in attrs or "btn" in classes or "button" in classes

            this_heading_idx = heading_idx
            if tag.lower() in heading_tags:
                this_heading_idx = len(segments)

            this_panel_idx = panel_idx
            if tag.lower() in panel_tags:
                this_panel_idx = len(segments)

            seg = {
                "tag": tag.lower(),
                "attrs": attrs,
                "classes": classes,
                "id": id_,
                "html": "",
                "is_button": is_button,
                "is_clickable": is_clickable,
                "parent_idx": parent_idx,
                "children": [],
                "start": getattr(node, "start", None),
                "end": getattr(node, "end", None),
                "_idx": len(segments),
                "context_heading_idx": this_heading_idx,  # <-- store nearest heading ancestor index
                "panel_ancestor_idx": this_panel_idx,
                "panel_ancestor_heading": None,
            }
            if hasattr(node, "start") and hasattr(node, "end") and node.start is not None and node.end is not None:
                html_bytes = html.encode("utf-8")
                try:
                    seg["html"] = html_bytes[node.start:node.end].decode("utf-8", errors="replace")
                except Exception:
                    seg["html"] = html[node.start:node.end]
            else:
                try:
                    seg["html"] = node.html if hasattr(node, "html") else ""
                except Exception:
                    seg["html"] = ""
            segments.append(seg)
            this_idx = seg["_idx"]
            for child in node.iter(include_text=True):
                child_idx = walk(child, this_idx, this_heading_idx, this_panel_idx)
                if child_idx is not None:
                    seg["children"].append(child_idx)
            return this_idx

        root = tree.body or tree.html or tree.root
        walk(root)

        # --- Batch embedding for all segments (robust for large files, with diagnostics and chunking) ---
        import math
        CHUNK_SIZE = 1024  # Tune as needed for memory/performance
        seg_hashes = []
        seg_htmls = []
        for seg in segments:
            seg_html = seg.get("html", "")
            seg_hash_val = segment_hash(seg_html)
            seg_hashes.append(seg_hash_val)
            seg_htmls.append(seg_html)
        total_segments = len(seg_hashes)
        logger.info(f"[EMBED] Total segments: {total_segments}")
        from ..utils.embedding_cache import load_embeddings_batch, save_embeddings_batch
        hash_to_embedding = {}
        cache_hits = 0
        cache_misses = 0
        # Chunked batch loading
        for i in range(0, total_segments, CHUNK_SIZE):
            chunk_hashes = seg_hashes[i:i+CHUNK_SIZE]
            chunk_result = load_embeddings_batch(chunk_hashes)
            hash_to_embedding.update(chunk_result)
            hits = sum(1 for v in chunk_result.values() if v is not None)
            cache_hits += hits
            cache_misses += len(chunk_hashes) - hits
            logger.debug(f"[EMBED] Batch {i//CHUNK_SIZE+1}: {hits} hits, {len(chunk_hashes)-hits} misses")
        logger.info(f"[EMBED] Total cache hits: {cache_hits}, misses: {cache_misses}")
        # Identify missing hashes
        missing = [(h, html) for h, html in zip(seg_hashes, seg_htmls) if hash_to_embedding.get(h) is None]
        # Chunked batch computation and saving
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
        # Assign embeddings to segments
        for seg, h in zip(segments, seg_hashes):
            seg["_embedding"] = hash_to_embedding[h]
        logger.info(f"[EMBED] Embedding assignment complete for {len(segments)} segments.")
        # Second pass: assign context_heading and panel_ancestor_heading
        for seg in segments:
            seg_html = seg.get("html", "")
            seg_hash_val = segment_hash(seg_html)
            embedding = load_embedding(seg_hash_val)
            if embedding is not None:
                seg["_embedding"] = embedding
            else:
                # Compute and save embedding if needed
                emb = model.encode(seg_html, convert_to_numpy=True, show_progress_bar=False)
                save_embedding(seg_hash_val, emb)
                seg["_embedding"] = emb
            if seg["tag"] in panel_tags or seg["tag"] == "table":
                parent_idx = seg["parent_idx"]
                heading_html = None
                while parent_idx is not None:
                    parent = segments[parent_idx]
                    if parent["tag"] in heading_tags:
                        heading_html = parent["html"]
                        break
                    parent_idx = parent["parent_idx"]
                seg["context_heading"] = heading_html
            if seg["tag"] == "table" and seg["panel_ancestor_idx"] is not None:
                panel_node = segments[seg["panel_ancestor_idx"]]
                seg["panel_ancestor_heading"] = panel_node.get("context_heading", [])

        logger.info(f"[PERF] DOM extraction (selectolax+ML+batch) took {time.time() - start_time:.2f} seconds, {len(segments)} segments.")
        return segments

    except Exception as e:
        logger.error(f"[FALLBACK] selectolax failed: {e}", extra={"traceback": traceback.format_exc(), "html_snippet": html[:200]})
        if not fallback_on_error:
            raise
        # Fallback: BeautifulSoup, but still add ML labels
        try:
            soup = BeautifulSoup(html, "html.parser")
            def walk_bs4(node, parent_idx=None, heading_idx=None, start_search=0):
                if not isinstance(node, Tag):
                    return start_search
                tag = node.name.lower()
                if tag not in HTML_TAGS:
                    log_unknown_tag(tag)
                    for child in node.children:
                        start_search = walk_bs4(child, parent_idx, heading_idx, start_search)
                    return start_search
                tag_html = str(node)
                start, end = html.find(tag_html, start_search), -1
                if start != -1:
                    end = start + len(tag_html)
                attrs = extract_attrs_bs4(node)
                for k in attrs:
                    log_unknown_attr(k)
                classes = attrs.get("class", "").split() if "class" in attrs else []
                id_ = attrs.get("id", "")
                is_button = tag == "button" or (tag == "input" and attrs.get("type", "").lower() in ["button", "submit"])
                is_clickable = is_button or tag == "a" or "onclick" in attrs or "btn" in classes or "button" in classes

                this_heading_idx = heading_idx
                if tag in heading_tags:
                    this_heading_idx = len(segments)

                seg = {
                    "tag": tag,
                    "attrs": attrs,
                    "classes": classes,
                    "id": id_,
                    "html": tag_html,
                    "is_button": is_button,
                    "is_clickable": is_clickable,
                    "parent_idx": parent_idx,
                    "children": [],
                    "start": start,
                    "end": end,
                    "_idx": len(segments),
                    "context_heading_idx": this_heading_idx,
                    "context_heading": None
                }
                segments.append(seg)
                return seg["_idx"]

            root = soup.find("html") or soup.find("body") or soup
            walk_bs4(root)

            # Batch embedding for all segments (BeautifulSoup fallback, with diagnostics and chunking)
            seg_hashes = []
            seg_htmls = []
            for seg in segments:
                seg_html = seg.get("html", "")
                seg_hash_val = segment_hash(seg_html)
                seg_hashes.append(seg_hash_val)
                seg_htmls.append(seg_html)
            total_segments = len(seg_hashes)
            logger.info(f"[EMBED] (BS4) Total segments: {total_segments}")
            hash_to_embedding = {}
            cache_hits = 0
            cache_misses = 0
            for i in range(0, total_segments, CHUNK_SIZE):
                chunk_hashes = seg_hashes[i:i+CHUNK_SIZE]
                chunk_result = load_embeddings_batch(chunk_hashes)
                hash_to_embedding.update(chunk_result)
                hits = sum(1 for v in chunk_result.values() if v is not None)
                cache_hits += hits
                cache_misses += len(chunk_hashes) - hits
                logger.debug(f"[EMBED] (BS4) Batch {i//CHUNK_SIZE+1}: {hits} hits, {len(chunk_hashes)-hits} misses")
            logger.info(f"[EMBED] (BS4) Total cache hits: {cache_hits}, misses: {cache_misses}")
            missing = [(h, html) for h, html in zip(seg_hashes, seg_htmls) if hash_to_embedding.get(h) is None]
            if missing:
                logger.info(f"[EMBED] (BS4) Computing {len(missing)} missing embeddings in chunks of {CHUNK_SIZE}")
                for i in range(0, len(missing), CHUNK_SIZE):
                    chunk = missing[i:i+CHUNK_SIZE]
                    missing_hashes, missing_htmls = zip(*chunk)
                    try:
                        new_embs = model.encode(list(missing_htmls), convert_to_numpy=True, show_progress_bar=False)
                    except Exception as e:
                        logger.error(f"[EMBED] (BS4) Batch embedding computation failed: {e}")
                        continue
                    save_embeddings_batch(list(zip(missing_hashes, new_embs)))
                    for h, emb in zip(missing_hashes, new_embs):
                        hash_to_embedding[h] = emb
                    logger.debug(f"[EMBED] (BS4) Saved {len(chunk)} new embeddings to cache.")
            for seg, h in zip(segments, seg_hashes):
                seg["_embedding"] = hash_to_embedding[h]
            logger.info(f"[EMBED] (BS4) Embedding assignment complete for {len(segments)} segments.")
            for seg in segments:
                if seg["tag"] in panel_tags or seg["tag"] == "table":
                    parent_idx = seg["parent_idx"]
                    heading_html = None
                    while parent_idx is not None:
                        parent = segments[parent_idx]
                        if parent["tag"] in heading_tags:
                            heading_html = parent["html"]
                            break
                        parent_idx = parent["parent_idx"]
                    seg["context_heading"] = heading_html
            logger.info(f"[PERF] DOM extraction (BeautifulSoup fallback+ML+batch) took {time.time() - start_time:.2f} seconds, {len(segments)} segments.")
            return segments
        except Exception as bs4e:
            logger.error(f"[ERROR] BeautifulSoup fallback also failed: {bs4e}", extra={"traceback": traceback.format_exc(), "html_snippet": html[:200]})
            raise

def canonicalize_segment(html):
    # Remove whitespace, lowercase, sort attributes, remove dynamic IDs/classes, collapse text for hashing
    # For <br>, just return '<br>'
    html = html.strip().lower()
    if html == '<br>' or html == '<br/>':
        return '<br>'
    # Remove ng-*, data-*, id, class attributes (except for semantic classes)
    import re
    html = re.sub(r'\s_ngcontent-[^=]+="[^"]*"', '', html)
    html = re.sub(r'\sclass="[^"]*"', '', html)
    html = re.sub(r'\sid="[^"]*"', '', html)
    html = re.sub(r'\sdata-[^=]+="[^"]*"', '', html)
    html = re.sub(r'\sng-\w+="[^"]*"', '', html)
    html = re.sub(r'\s+', ' ', html)
    return html

def segment_hash(html):
    canon = canonicalize_segment(html)
    return hashlib.sha256(canon.encode('utf-8')).hexdigest()

def load_segment_label_cache(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return orjson.loads(f.read())
    return {}

def save_segment_label_cache(cache, path):
    with open(path, 'wb') as f:
        f.write(orjson.dumps(cache))

# In your segment review loop, before prompting:
# cache_path = 'segment_label_cache.jsonl'
# segment_label_cache = load_segment_label_cache(cache_path)
# h = segment_hash(segment_html)
# if h in segment_label_cache:
#     label = segment_label_cache[h]
#     # Use label, skip prompt
# else:
#     # Prompt user, then:
#     segment_label_cache[h] = user_label
#     save_segment_label_cache(segment_label_cache, cache_path)

# --- not being used yet, but useful for future ---
def extract_panel_table_hierarchy(segments, model_name: Optional[str] = None, use_finetuned: bool = True, min_panel_score=0.65):
    """
    Advanced: Extract panels and their associated tables from DOM segments.
    Uses ML embeddings, clustering, DOM proximity, and semantic heuristics for robust extraction.
    Returns a list of panel dicts, each with ML confidence and association logs.
    """

    panel_tags = PANEL_TAGS

    idx_to_seg = {seg["_idx"]: seg for seg in segments if "_idx" in seg}
    table_segs = [seg for seg in segments if seg.get("tag", []) == "table"]
    if isinstance(segments, list):
        segments[:] = [s for s in segments if isinstance(s, dict)]
    # --- ML Model for Embeddings ---
    model = ModelRegistry.get_sentence_transformer(model_name=model_name, use_finetuned=use_finetuned)

    # --- Helper: Find all panel-like segments ---
    panel_segs = [
        seg for seg in segments
        if (
            (seg.get("tag", []) == "div" and "p-panel" in seg.get("classes", []))
            or (seg.get("tag", []) in panel_tags)
            or any(
                kw in (seg.get("classes", []) + [seg.get("id", "")])
                for kw in [
                    "panel", "card", "container", "box", "section-panel", "results", "content", "main", "section", "p-panel-content"
                ]
            )
            or seg.get("ml_label", []) in ("panel", "location_panel", "candidate_panel")
        )
    ]

    # --- Compute embeddings for all panels and tables ---
    all_segs = panel_segs + table_segs
    embeddings = batch_get_segment_embeddings(model, all_segs)
    for seg, emb in zip(all_segs, embeddings):
        seg["_embedding"] = emb
    # --- Score panel-table associations using DOM and ML similarity ---
    panel_table_scores = []
    for panel in panel_segs:
        for table in table_segs:
            # DOM proximity: walk up from table to see if panel is ancestor
            dom_score = 0
            parent_idx = table.get("parent_idx", [])
            hops = 0
            while parent_idx is not None and hops < 10:
                if parent_idx == panel["_idx"]:
                    dom_score = 1.0 - 0.1 * hops  # closer is better
                    break
                parent_idx = idx_to_seg.get(parent_idx, {}).get("parent_idx", [])
                hops += 1
            # ML similarity
            ml_score = float(np.dot(panel["_embedding"], table["_embedding"]) /
                             (np.linalg.norm(panel["_embedding"]) * np.linalg.norm(table["_embedding"]) + 1e-8))
            # Final score: weighted sum
            score = 0.6 * dom_score + 0.4 * ml_score
            panel_table_scores.append({
                "panel_idx": panel["_idx"],
                "table_idx": table["_idx"],
                "dom_score": dom_score,
                "ml_score": ml_score,
                "score": score,
            })

    # --- Assign tables to panels based on best score ---
    table_to_panel = {}
    for table in table_segs:
        best = max(
            (s for s in panel_table_scores if s["table_idx"] == table["_idx"]),
            key=lambda s: s["score"],
            default=None
        )
        if best and best["score"] >= min_panel_score:
            table_to_panel.setdefault(best["panel_idx"], []).append((table, best))
        else:
            table_to_panel.setdefault(None, []).append((table, {"score": 0.0}))

    # --- Build panel objects with ML confidence and association logs ---
    panels = []
    for panel in panel_segs:
        tables_and_scores = table_to_panel.get(panel["_idx"], [])
        if not tables_and_scores:
            continue

        # Improved heading extraction
        heading = extract_heading_text_from_panel_or_ancestors(panel, segments)
        if not heading:
            heading = panel.get("context_heading", [])
        if not heading:
            heading = panel.get("panel_ancestor_heading", [])
        if not heading:
            heading = f"Panel {panel['_idx']}"
        # Debug output if heading is still generic
        if not heading or heading.startswith("Panel"):
            print(f"[DEBUG] Panel idx={panel['_idx']} has no heading. context_heading={panel.get('context_heading')}, panel_ancestor_heading={panel.get('panel_ancestor_heading')}")
            print(f"[DEBUG] Panel HTML snippet: {panel.get('html', '')[:200]}")

        panels.append({
            "panel_idx": panel["_idx"],
            "panel_tag": panel.get("tag", []),
            "panel_heading": heading,
            "panel_html": panel.get("html", []),
            "fully_reported": "",  # Could add ML extraction for reporting status
            "ml_confidence": float(np.mean([s["score"] for _, s in tables_and_scores])),
            "tables": [
                {
                    "table_idx": t["_idx"],
                    "table_html": t.get("html", ""),
                    "context_heading": heading,
                    "panel_ancestor_heading": heading,
                    "ml_panel_score": s["score"],
                    "ml_panel_dom_score": s["dom_score"],
                    "ml_panel_semantic_score": s["ml_score"],
                }
                for t, s in tables_and_scores
            ],
            "association_log": [
                {
                    "table_idx": t["_idx"],
                    "score": s["score"],
                    "dom_score": s["dom_score"],
                    "ml_score": s["ml_score"]
                }
                for t, s in tables_and_scores
            ]
        })

    # --- Fallback: treat orphan tables as their own panels ---
    orphan_tables = table_to_panel.get(None, [])
    for t, s in orphan_tables:
        heading = extract_heading_text_from_panel_or_ancestors(t, segments)
        if not heading:
            heading = t.get("context_heading", [])
        if not heading:
            heading = t.get("panel_ancestor_heading", [])
        if not heading:
            heading = f"Panel {t['_idx']}"
        if not heading or heading.startswith("Panel"):
            print(f"[DEBUG] Orphan table idx={t['_idx']} has no heading. context_heading={t.get('context_heading', [])}, panel_ancestor_heading={t.get('panel_ancestor_heading', [])}")
            print(f"[DEBUG] Table HTML snippet: {t.get('html', '')[:200]}")

        panels.append({
            "panel_idx": t["_idx"],
            "panel_tag": "table",
            "panel_heading": heading,
            "panel_html": t.get("html", ""),
            "fully_reported": "",
            "ml_confidence": s["score"],
            "tables": [{
                "table_idx": t["_idx"],
                "table_html": t.get("html", ""),
                "context_heading": heading,
                "panel_ancestor_heading": heading,
                "ml_panel_score": s["score"],
                "ml_panel_dom_score": s.get("dom_score", 0.0),
                "ml_panel_semantic_score": s.get("ml_score", 0.0),
            }],
            "association_log": [{
                "table_idx": t["_idx"],
                "score": s["score"],
                "dom_score": s.get("dom_score", 0.0),
                "ml_score": s.get("ml_score", 0.0)
            }]
        })
    return panels

def extract_heading_text_from_panel_or_ancestors(panel_seg, segments, max_depth=6):

    # --- Cache BeautifulSoup objects for unique HTML strings ---
    soup_cache = {}
    if isinstance(segments, list):
        segments[:] = [s for s in segments if isinstance(s, dict)]
    def get_soup(html):
        if html in soup_cache:
            return soup_cache[html]
        soup = BeautifulSoup(html, "html.parser")
        soup_cache[html] = soup
        return soup
    # --- 1. Get HTML for this panel, fallback to concatenating children if empty ---
    def get_full_html(seg):
        html = seg.get("html", "")
        if html and html.strip():
            return html
        # Recursively concatenate all descendants' HTML
        child_htmls = []
        for idx in seg.get("children", []):
            child_html = get_full_html(segments[idx])
            if child_html:
                child_htmls.append(child_html)
        return "\n".join(child_htmls)

    html = get_full_html(panel_seg)
    soup = get_soup(html)
    found_texts = []

    # 2. Try heading tags first (as before)
    for tag in HEADING_TAGS:
        for el in soup.find_all(tag):
            txt = el.get_text(strip=True)
            if not txt:
                continue
            found_texts.append(txt)
            match = DISTRICT_REGEX.search(txt)
            if match:
                return match.group(0)
            if len(txt) < 40 and any(word in txt.lower() for word in LOCATION_KEYWORDS):
                return txt

    # 3. Try common heading classes/ids (PrimeNG/Enhanced Voting)
    for el in soup.find_all(True, class_=lambda c: c and any(h in c for h in ["panel-header", "contest-header", "ng-star-inserted", "section-title"])):
        txt = el.get_text(strip=True)
        if txt:
            found_texts.append(txt)
            match = DISTRICT_REGEX.search(txt)
            if match:
                return match.group(0)
            if len(txt) < 40 and any(word in txt.lower() for word in LOCATION_KEYWORDS):
                return txt
    for el in soup.find_all(True, id=lambda i: i and any(h in i for h in ["panel-header", "contest-header", "ng-star-inserted", "section-title"])):
        txt = el.get_text(strip=True)
        if txt:
            found_texts.append(txt)
            match = DISTRICT_REGEX.search(txt)
            if match:
                return match.group(0)
            if len(txt) < 40 and any(word in txt.lower() for word in LOCATION_KEYWORDS):
                return txt

    # 4. Fallback: any short, non-empty text node in the panel
    for el in soup.find_all(text=True):
        txt = el.strip()
        if not txt or len(txt) > 60:
            continue
        found_texts.append(txt)
        match = DISTRICT_REGEX.search(txt)
        if match:
            return match.group(0)
        if any(word in txt.lower() for word in LOCATION_KEYWORDS):
            return txt

    # 5. Fuzzy match if nothing found
    if found_texts:
        close = get_close_matches(
            found_texts[0], LOCATION_KEYWORDS, n=1, cutoff=0.7
        )
        if close:
            return close[0]

    # 6. Fallback: walk up ancestors
    parent_idx = panel_seg.get("parent_idx", [])
    depth = 0
    while parent_idx is not None and depth < max_depth:
        parent = segments[parent_idx]
        parent_html = parent.get("html", "")
        if not parent_html or parent_html.strip() == "":
            # Try to concatenate children
            child_htmls = []
            for idx in parent.get("children", []):
                child_html = segments[idx].get("html", "")
                if child_html:
                    child_htmls.append(child_html)
            parent_html = "\n".join(child_htmls)
        parent_soup = get_soup(parent_html)
        for tag in HEADING_TAGS:
            el = parent_soup.find(tag)
            if el and el.get_text(strip=True):
                txt = el.get_text(strip=True)
                found_texts.append(txt)
                match = DISTRICT_REGEX.search(txt)
                if match:
                    return match.group(0)
                if len(txt) < 40 and any(word in txt.lower() for word in LOCATION_KEYWORDS):
                    return txt
        parent_idx = parent.get("parent_idx", [])
        depth += 1

    # 7. Fuzzy match on ancestors if nothing found
    if found_texts:
        close = get_close_matches(
            found_texts[0], LOCATION_KEYWORDS, n=1, cutoff=0.7
        )
        if close:
            return close[0]

    # 8. Debug: print panel HTML if heading is still None
    rprint(f"[red][DEBUG] No heading found for panel idx={panel_seg.get('_idx', [])}. Panel HTML snippet:\n{html[:300]}...[/red]")

    return None

def extract_fully_reported_from_panel(panel_html):
    soup = BeautifulSoup(panel_html, "html.parser")
    for span in soup.find_all("span", class_="fw-bold"):
        txt = span.get_text(strip=True)
        if "Reported" in txt:
            return txt
    txt = soup.get_text(" ", strip=True)
    for part in txt.splitlines():
        if "Reported" in part:
            return part.strip()
    return ""

TAG_PATTERN = re.compile(
    r"<({tags})(\s[^>]*)?>.*?</\1\s*>|<({tags})(\s[^>]*)?/?>".format(
        tags="|".join(HTML_TAGS)
    ),
    re.IGNORECASE | re.DOTALL
)
ATTR_PATTERN = re.compile(
    r'([a-zA-Z_:][a-zA-Z0-9_\-.:]*)'
    r'(?:\s*=\s*'
    r'(?:'
    r'"([^"]*)"'
    r"|"
    r"'([^']*)'"
    r"|"
    r'([^\s"\'=<>`]+)'
    r'))?',
    re.UNICODE
)

def extract_attrs(attr_str):
    attrs = {}
    for match in ATTR_PATTERN.finditer(attr_str):
        name = match.group(1)
        value = match.group(2) if match.group(2) is not None else (
            match.group(3) if match.group(3) is not None else (
                match.group(4) if match.group(4) is not None else None
            )
        )
        if value is None:
            attrs[name] = True
        else:
            attrs[name] = value
        log_unknown_attr(name)
    return attrs
# Load or initialize the DOM pattern knowledge base
def load_pattern_kb():
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

def save_pattern_kb(kb):
    path = safe_log_path("dom_pattern_kb.jsonl")
    with open(path, "wb") as f:
        for entry in kb:
            if "embedding" in entry and isinstance(entry["embedding"], np.ndarray):
                entry["embedding"] = entry["embedding"].tolist()
            f.write(orjson.dumps(entry, option=orjson.OPT_INDENT_2) + b"\n")
            
def append_pattern_kb(entry):
    # Convert any ndarray to list before saving
    if "embedding" in entry and isinstance(entry["embedding"], np.ndarray):
        entry["embedding"] = entry["embedding"].tolist()
    path = safe_log_path("dom_pattern_kb.jsonl")
    with open(path, "ab") as f:
        f.write(orjson.dumps(entry, option=orjson.OPT_INDENT_2) + b"\n")
        
def append_feedback_log(entry):
    # Convert any ndarray to list before saving
    if "embedding" in entry and isinstance(entry["embedding"], np.ndarray):
        entry["embedding"] = entry["embedding"].tolist()    
    path = safe_log_path("segment_feedback_log.jsonl")
    with open(path, "ab") as f:
        f.write(orjson.dumps(entry, option=orjson.OPT_INDENT_2) + b"\n")
    # --- Ensure feedback is also loaded into pattern KB cache for immediate effect ---
    if "pattern_id" in entry and "label" in entry and "html" in entry:
        # Use the same hash logic as segment_identity_hash
        seg_hash = segment_identity_hash({"tag": entry.get("tag", ""), "attrs": entry.get("attrs", {}), "html": entry["html"]})
        kb_entry = {
            "pattern_id": entry["pattern_id"],
            "label": entry["label"],
            "embedding": entry.get("embedding", []),
            "example_html": entry["html"][:500],
            "segment_hash": seg_hash,
            "timestamp": entry.get("timestamp", 0),
        }
        # Add to in-memory pattern KB cache if available
        global _pattern_kb_cache
        if _pattern_kb_cache is not None:
            _pattern_kb_cache.append(kb_entry)

def get_page_hash(page):
    content = page.content()
    return hashlib.sha256(content.encode("utf-8")).hexdigest()

def extract_download_links_from_html(html, exts=(".csv", ".json", ".pdf")):
    pattern = re.compile(r'<a[^>]+href=["\']([^"\']+\.(?:csv|json|pdf))["\']', re.IGNORECASE)
    links = []
    for match in pattern.finditer(html):
        href = match.group(1)
        for ext in exts:
            if href.lower().endswith(ext):
                links.append({"href": href, "format": ext})
    return links

# --- ML/Embedding/Clustering helpers ---

def auto_label_segment(
    segment,
    context_library=None,
    context_cache=None,
    pattern_kb=None,
    model_name=None,
    use_finetuned=True,
    ml_threshold=0.7
):
    """
    Improved auto_label_segment:
    - Checks persistent cache, context_cache, context_library, and pattern_kb for prior labels.
    - Explicitly ignores root/container tags.
    - Uses all available context for robust labeling.
    - Falls back to ML/heuristics only if no prior label is found.
    """
    # --- 0. Robust segment hash for deduplication ---
    seg_hash = segment_identity_hash(segment)

    # --- 1. Check persistent label cache ---
    cached_label = get_cached_segment_label(seg_hash)
    if cached_label:
        return cached_label

    # --- 2. Check context_cache ---
    if context_cache and seg_hash in context_cache:
        label = context_cache[seg_hash].get("ml_label")
        if label:
            return label

    # --- 3. Check context_library cached_segments ---
    if context_library and "cached_segments" in context_library:
        for seg in context_library["cached_segments"]:
            if seg.get("segment_hash") == seg_hash and seg.get("ml_label"):
                return seg["ml_label"]

    # --- 4. Check pattern_kb ---
    if pattern_kb:
        for entry in pattern_kb:
            if entry.get("segment_hash") == seg_hash and entry.get("label"):
                return entry["label"]

    tag = segment.get("tag", "").lower()
    classes = [c.lower() for c in segment.get("classes", [])]
    attrs = segment.get("attrs", {})
    html = segment.get("html", "").lower()
    id_ = segment.get("id", "").lower()
    text = segment.get("text", "").strip().lower() if segment.get("text", []) else ""

    # --- 5. Explicitly ignore root/container tags ---
    
    if tag in ROOT_CONTAINER_TAGS:
        return "ignore"
    if tag == "div" and ("container" in classes or "main" in classes) and not html.strip():
        return "ignore"

    # --- 6. Always-ignored tags/classes/ids ---

    if tag in ALWAYS_IGNORE_TAGS:
        return "ignore"
    if set(classes) & ALWAYS_IGNORE_CLASSES:
        return "ignore"
    if id_ in ALWAYS_IGNORE_IDS:
        return "ignore"

    # --- 7. Decorative/icon detection ---

    if tag in ICON_TAGS and (ICON_CLASSES & set(classes)):
        if tag != "span" or (set(classes) <= ICON_CLASSES and not html.strip()):
            return "ignore"
        if tag == "span" and set(classes) <= ICON_CLASSES and not re.sub(r"<[^>]+>", "", html).strip():
            return "ignore"
    if tag in {"i", "span"} and not html.strip():
        return "ignore"

    # --- 8. Download links ---
    if tag == "a" and "href" in attrs:
        href = str(attrs["href"]).lower()
        if any(href.endswith(ext) for ext in [".csv", ".json", ".pdf", ".xlsx", ".zip", ".xls", ".doc", ".docx"]):
            return "download_link"

    # --- 9. Ballot toggle/button ---
    
    if segment.get("is_button", []) or BUTTON_CLASSES & set(classes) or "toggle" in id_:
        return "ballot_toggle"

    # --- 10. Heading ---
    
    if tag in HEADING_TAGS or HEADING_CLASSES & set(classes):
        return "heading"

    # --- 11. Panel/section/card/box ---
    
    if tag in PANEL_TAGS or PANEL_CLASSES & set(classes):
        return "panel"

    # --- 12. Table ---
    if tag == "table":
        return "results_table"

    # --- 13. Context-driven: party, vote method, contest, etc. ---
    from difflib import get_close_matches
    if context_library:
        if 'party' in context_library:
            known_parties = [p.lower() for p in context_library['party']]
            if text in known_parties or html in known_parties:
                return "party_label"
            close = get_close_matches(text, known_parties, n=1, cutoff=0.85)
            if close:
                return "party_label"
        if 'vote_methods' in context_library:
            known_vote_methods = [v.lower() for v in context_library['vote_methods']]
            if text in known_vote_methods or html in known_vote_methods:
                return "vote_method"
            close = get_close_matches(text, known_vote_methods, n=1, cutoff=0.85)
            if close:
                return "vote_method"
        if 'contests' in context_library:
            known_contests = [c["title"].lower() for c in context_library['contests'] if "title" in c]
            if text in known_contests or html in known_contests:
                return "contest_title"

    # --- 14. Ballot type ---
    if any(bt in html for bt in BALLOT_TYPES):
        return "ballot_type"

    # --- 15. Clickable (fallback for links/buttons) ---
    if segment.get("is_clickable", []):
        return "clickable"

    # --- 16. Results timestamp ---

    if (
        tag in {"span", "time", "div", "p", "small", "label"}
        and (
            any(cls in TIMESTAMP_CLASSES for cls in classes)
            or any(re.search(pat, id_) for pat in TIMESTAMP_ID_PATTERNS if id_)
            or any(attr in attrs for attr in TIMESTAMP_ATTRS)
            or any(re.search(pat, " ".join(attrs.keys())) for pat in TIMESTAMP_ID_PATTERNS)
            or re.search(r"\bago\b|\bupdated\b|\blast\b|\bposted\b|\bas of\b|\breported\b", html)
            or re.search(r"\b\d{1,2}:\d{2}\s*(am|pm)?\b", html)
            or re.search(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", html)
            or re.search(r"\b\d{4}-\d{2}-\d{2}\b", html)
        )
    ):
        return "results_timestamp"

    # --- 17. Fallback: ignore common empty/structural tags ---
    
    if tag in STRUCTURAL_TAGS and not html.strip():
        return "ignore"

    # --- 18. Fallback: ignore if only whitespace or non-breaking space ---
    if not html.strip() or html.strip() in {"&nbsp;", "&#160;"}:
        return "ignore"

    # --- 19. Fallback: ignore if only contains a single icon or decorative element ---
    if tag == "span" and len(classes) > 0 and all(cls in ICON_CLASSES for cls in classes):
        return "ignore"

    # --- 20. Canonical label mapping ---
    canonical = get_canonical_segment_label(text)
    if canonical:
        return canonical

    # --- 21. Fallback: unknown/ambiguous, needs review ---
    return "unknown"

def embedding_cache_hash(segment, model_id):
    """
    Construct a robust hash for embedding cache that includes segment content and model identifier.
    """
    tag = segment.get("tag", "")
    attrs = segment.get("attrs", {})
    # Remove dynamic attributes from attrs for hashing
    attrs_filtered = {k: v for k, v in attrs.items() if not (k.startswith('_ngcontent-') or k.startswith('_nghost-') or k.startswith('ng-') or k.startswith('data-') or k in {'style', 'id', 'class', 'tabindex', 'aria-checked'})}
    html = segment.get("html", "")
    attrs_sorted = {k: attrs_filtered[k] for k in sorted(attrs_filtered)}
    html_norm = _normalize_html_for_hash(html)
    base = tag + orjson.dumps(attrs_sorted, option=orjson.OPT_SORT_KEYS).decode() + html_norm + str(model_id)
    return hashlib.sha256(base.encode("utf-8")).hexdigest()

def get_segment_embedding(model, segment, cache=None, cache_hits=None, cache_misses=None):
    model_id = getattr(model, 'name_or_path', str(model))
    identity = embedding_cache_hash(segment, model_id)
    emb = get_embedding_from_memory(identity)
    if cache is not None:
        clean_cache_inplace(cache)    
    if emb is not None:
        if cache_hits is not None:
            cache_hits.add(identity)
        return emb
    # In-memory cache miss, compute embedding
    text = segment.get("html", "")
    tag = segment.get("tag", "")
    attrs = " ".join([f"{k}={v}" for k, v in segment.get("attrs", {}).items()])
    full_text = f"{tag} {attrs} {text}"
    try:
        emb = model.encode(full_text, convert_to_numpy=True, show_progress_bar=False)
        save_embedding(identity, emb)
        if cache_misses is not None:
            cache_misses.add(identity)
        return emb
    except Exception:
        # Fallback: return None if embedding fails
        return None

def cosine_sim(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def ml_classify_segment(segment, model, pattern_kb, threshold=0.85):
    emb = get_segment_embedding(model, segment, cache_hits=embedding_cache_hits, cache_misses=embedding_cache_misses)
    if isinstance(emb, list):
        emb = np.array(emb)
    best_label = "unknown"
    best_conf = 0.0
    best_pattern_id = None
    for entry in pattern_kb:
        kb_emb = np.array(entry.get("embedding", []))
        if kb_emb.shape != emb.shape:
            continue
        sim = cosine_sim(emb, kb_emb)
        if sim > best_conf:
            best_conf = sim
            best_label = entry.get("label", "unknown")
            best_pattern_id = entry.get("pattern_id", [])
    if best_conf < threshold:
        return "unknown", best_conf, None
    return best_label, best_conf, best_pattern_id

def prompt_for_segment_label(segment, context_library=None):
    # Use robust segment identity hash for deduplication
    seg_hash = segment_identity_hash(segment)
    # Check persistent cache first
    cached_label = get_cached_segment_label(seg_hash)
    if cached_label:
        return cached_label
    # Try canonical and cache-based auto-labeling first
    html_preview = segment.get("html", "")
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
    # Fallback to user prompt if ambiguous
    if not html_preview:
        html_preview = f"[No HTML] tag={segment.get('tag', [])} attrs={segment.get('attrs', [])}"

    rprint(f"\n[bold yellow]Segment needs review:[/bold yellow]\n{html_preview[:200]}{'...' if len(html_preview) > 200 else ''}")
    rprint(
        "[cyan]What is the semantic role of this segment? (e.g., results_table, ballot_toggle, heading, panel, candidate_panel, location_panel, ballot_type, results_timestamp, download_link, clickable, footer, legend, contest_title, party_label, vote_method, reporting_status, summary, error_message, warning, info_box, navigation, pagination, tab, modal, tooltip, ignore, unknown, etc.)[/cyan]"
    )
    label = prompt_user_input("> ").strip()
    cache_segment_label(seg_hash, label)
    return label

def scan_html_for_context(
    target_url,
    page,
    debug=False,
    context_cache=None,
    rejected_downloads: Optional[set] = None,
    model_name: Optional[str] = None,
    use_finetuned: bool = True,
    non_interactive=False,
) -> Dict[str, Any]:
    """
    Advanced HTML scanner with ML-driven DOM pattern clustering, active learning, dynamic tagging,
    confidence-driven processing, and persistent knowledge base.
    Args:
        target_url: The URL being scanned.
        page: Page object with .content() and .url.
        debug: If True, print debug output.
        context_cache: Optional context cache dict.
        rejected_downloads: Set of download links to skip.
        model_name: Name of ML model for segment labeling.
        non_interactive: If True, disables user prompts.
    Returns:
        context_result: Dict with scan results, segments, metadata, and errors if any.
    """
    
    context_library = None
    if os.path.exists(CONTEXT_LIBRARY_PATH):
        with open(CONTEXT_LIBRARY_PATH, "rb") as f:
            CONTEXT_LIBRARY = robust_orjson_loads(f.read())
            context_library = CONTEXT_LIBRARY
        supported_formats = CONTEXT_LIBRARY.get("supported_formats", {})
        supported_links = [link for link in CONTEXT_LIBRARY.get("download_links", []) if link["format"] in supported_formats]
    else:
        supported_formats = {}
        supported_links = []    
    if rejected_downloads is None:
        rejected_downloads = set()
    page_hash = get_page_hash(page)
    # Ensure context_cache is initialized
    if context_cache is None:
        context_cache = load_context_cache_from_disk()
    # Try to load full context_result from disk cache if available
    if page_hash in context_cache:
        logger.info(f"[SCAN] Using cached context for {target_url}")
        rprint("[bold green][CACHE] Entire context loaded from cache. Skipping scan.[/bold green]")
        return context_cache[page_hash]

    # Always define context_result before any return
    context_result = {
        "raw_html": "",
        "tagged_segments": [],
        "tagged_segments_with_attrs": [],
        "available_formats": list(supported_formats) if isinstance(supported_formats, list) else list(supported_formats.keys()),
        "metadata": {},
        "selector_log": [],
        "error": None,
        "url": page.url,
        "pattern_kb_matches": [],
        "segments_needing_review": [],
    }
    try:
        page_url = target_url or page.url
        SCAN_WAIT_SECONDS = 3
        logger.info(f"[SCAN] Waiting {SCAN_WAIT_SECONDS} seconds to scan page content...")
        time.sleep(SCAN_WAIT_SECONDS)
        html = page.content()
        context_result["raw_html"] = html
        state, county = infer_state_county_from_url(page_url)
        if state:
            context_result["state"] = state
        if county:
            context_result["county"] = county
        # --- 3. Download link extraction and merging ---
        dynamic_links = extract_download_links_from_html(html)
        all_links = { (l["href"], l["format"]): l for l in (supported_links + dynamic_links) }
        supported_links = list(all_links.values())
        context_result["metadata"]["download_links"] = supported_links

        # --- 4. Downloadable file prompt logic (with ML-driven format clustering) ---
        format_kb = load_pattern_kb()
        for link in supported_links:
            fmt = link["format"]
            append_pattern_kb({
                "pattern_id": f"format_{fmt}_{os.path.basename(link['href'])}",
                "label": "download_format",
                "format": fmt,
                "href": link["href"],
                "source_url": page.url,
                "timestamp": time.time(),
                "embedding": [],
            })

        new_links = [link for link in supported_links if link["href"] not in rejected_downloads]
        if new_links:
            available_files = [f"{os.path.basename(link['href'])} ({link['format']})" for link in new_links]
            rprint(f"[cyan]Downloadable file(s) found: {', '.join(available_files)}.[/cyan]")
            rprint("[magenta]Would you like to download one now? (y/n) (type 'cancel' to abort)[/magenta]")
            user_input = prompt_user_input("> ")
            if user_input and user_input.strip().lower().startswith("y"):
                if len(new_links) > 1:
                    rprint("[bold cyan]Which format do you want to download?[/bold cyan] " + ", ".join(available_files))
                    chosen_fmt = prompt_user_input("> ").strip().lower()
                    chosen_link = next((l for l in new_links if l["format"].lower() == chosen_fmt.lower()), None)
                else:
                    chosen_link = new_links[0]
                if chosen_link:
                    from ..html_election_parser import mark_url_processed
                    local_file = download_file(page.url, chosen_link["href"])
                    if local_file:
                        fmt = chosen_link["format"]
                        format_handler = route_format_handler(fmt)
                        if format_handler and hasattr(format_handler, "parse"):
                            result = format_handler.parse(None, {"manual_file": local_file, "source_url": target_url})
                            if result and all(result):
                                *_, metadata = result
                                mark_url_processed(target_url, status="success", **metadata)
                            else:
                                mark_url_processed(target_url, status="fail")
                            return context_result
                if not chosen_link:
                    rprint(f"[red]No download link found for format: {chosen_fmt}[/red]")
            else:
                for link in new_links:
                    rejected_downloads.add(link["href"])
                context_result["metadata"]["download_links"] = [
                    {"format": link["format"], "url": link["href"]} for link in supported_links
                ]

        # --- 5. HTML tag extraction for context organization (with ML) ---
        pattern_kb = load_pattern_kb()
        segments_with_attrs = extract_tagged_segments_with_attrs(
            html,
            context_cache=context_cache,
            include_data_attrs=True,
            fallback_on_error=True,
            pattern_kb=pattern_kb,
            ml_threshold=0.85,
            context_library=context_library
        )
        context_result["tagged_segments_with_attrs"] = segments_with_attrs
        context_result["tagged_segments"] = [seg["html"] for seg in segments_with_attrs]

        # --- 6. ML-driven DOM pattern clustering and tagging ---
        model = ModelRegistry.get_sentence_transformer(model_name=model_name, use_finetuned=use_finetuned)
        pattern_matches = []
        segments_needing_review = []
        seen = set()
        unique_segments = []
        for seg in segments_with_attrs:
            html_norm = _normalize_html_for_hash(seg['html'])
            if html_norm not in seen:
                seen.add(html_norm)
                unique_segments.append(seg)
        segments_with_attrs = unique_segments    
        for seg in segments_with_attrs:
            # Already labeled in extract_tagged_segments_with_attrs, but check for low confidence
            if seg.get("ml_confidence", 0.0) < 0.7 or seg.get("ml_label", "unknown") == "unknown":
                user_label = prompt_for_segment_label(seg, context_library=context_library)
                seg["ml_label"] = user_label
                seg["ml_confidence"] = 1.0
                seg["pattern_id"] = f"pattern_{hashlib.sha256(seg['html'].encode('utf-8')).hexdigest()[:10]}"
                emb = get_segment_embedding(model, seg, cache_hits=embedding_cache_hits, cache_misses=embedding_cache_misses).tolist()
                kb_entry = {
                    "pattern_id": seg["pattern_id"],
                    "label": user_label,
                    "embedding": emb,
                    "example_html": seg["html"][:500],
                    "source_url": page.url,
                    "timestamp": time.time(),
                }
                append_pattern_kb(kb_entry)
                append_feedback_log({
                    "pattern_id": seg["pattern_id"],
                    "label": user_label,
                    "html": seg["html"][:500],
                    "source_url": page.url,
                    "timestamp": time.time(),
                })
                segments_needing_review.append(seg)
                # --- update context library with the correction ---
                if context_library is not None and seg.get("segment_hash", []):
                    update_context_library(context_library, seg["segment_hash"], user_label)
                    # Optionally save immediately:
                    save_context_library(context_library)
            else:
                pattern_matches.append({
                    "pattern_id": seg["pattern_id"],
                    "label": seg["ml_label"],
                    "confidence": seg["ml_confidence"],
                    "segment_html": seg["html"][:200],
                })

        context_result["pattern_kb_matches"] = pattern_matches
        context_result["segments_needing_review"] = segments_needing_review

        # --- 7. Dynamic tagging and context enrichment ---
        selector_log = set()
        for seg in segments_with_attrs:
            if seg["id"]:
                selector_log.add(f'#{seg["id"]}')
            for cls in seg["classes"]:
                selector_log.add(f'.{cls}')
            selector_log.add(seg["tag"].lower())
            if "semantic_tags" not in seg:
                seg["semantic_tags"] = []
            if seg["ml_label"] not in ("unknown", "ignore"):
                seg["semantic_tags"].append(seg["ml_label"])
        context_result["selector_log"] = sorted(selector_log)

        context_result["metadata"].update({
            "source_url": page.url,
            "scrape_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pattern_kb_size": len(pattern_kb),
        })

        if debug:
            rprint("\n[orange][DEBUG] Extracted HTML segments with ML labels:[/orange]")
            for seg in segments_with_attrs:
                rprint(f"{seg['tag']} {seg['attrs']} [label={seg['ml_label']}, conf={seg['ml_confidence']:.2f}] {seg['html'][:80]}{'...' if len(seg['html']) > 80 else ''}")
            if supported_links:
                rprint("\n[orange][DEBUG] Detected download links:[/orange]")
                for link in supported_links:
                    file_name = os.path.basename(link["href"])
                    rprint(f"[green]  - {file_name} ({link['format']})[/green]")
            if segments_needing_review:
                rprint(f"\n[red][DEBUG] {len(segments_needing_review)} segments flagged for review.[/red]")
        if context_library is not None:
            if "cached_segments" not in context_library:
                context_library["cached_segments"] = []
            known_hashes = {seg.get("segment_hash", []) for seg in context_library["cached_segments"]}
            for seg in segments_with_attrs:
                if seg.get("segment_hash", []) and seg["segment_hash"] not in known_hashes:
                    # Only store minimal info needed for reuse
                    context_library["cached_segments"].append({
                        "segment_hash": seg["segment_hash"],
                        "ml_label": seg["ml_label"],
                        "ml_confidence": seg["ml_confidence"],
                        "pattern_id": seg["pattern_id"],
                        # Optionally add more fields if needed
                    })
            # Save back to disk
            save_context_library(context_library)
   
    except Exception as e:
        tb = traceback.format_exc()
        rprint(f"[SCAN ERROR] HTML parsing failed: {e}\n{tb}")
        logger.error(f"[SCAN ERROR] HTML parsing failed: {e}", extra={"traceback": tb, "url": getattr(page, 'url', None)})
        context_result["error"] = f"[SCAN ERROR] HTML parsing failed: {e}\n{tb}"

    logger.debug(f"Available formats detected: {context_result['available_formats']}")
    if context_cache is not None:
        context_cache[page_hash] = context_result
        save_context_cache_to_disk(context_cache)
    if embedding_cache_hits and not embedding_cache_misses:
        rprint(f"[bold green][CACHE] All segment embeddings loaded from cache.[/bold green]")
    elif embedding_cache_hits:
        rprint(f"[yellow][CACHE] {len(embedding_cache_hits)} embeddings loaded from cache, {len(embedding_cache_misses)} computed.[/yellow]")        
    return context_result

def get_log_folder():
    log_folder = LOG_DIR
    os.makedirs(log_folder, exist_ok=True)
    return log_folder

def load_context_cache_from_disk(filename="context_cache.json"):
    global _context_cache
    if _context_cache is not None:
        _context_cache = {k: v for k, v in _context_cache.items() if isinstance(v, dict)}
        return _context_cache
    path = safe_cache_path(filename)
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                raw_cache = robust_orjson_loads(f.read())
                _context_cache = {k: v for k, v in raw_cache.items() if isinstance(v, dict)}
                return _context_cache
        except Exception as e:
            logger.error(f"[ERROR] Failed to load {filename}: {e}")
            return {}
    _context_cache = {}
    return {}

def save_context_cache_to_disk(context_cache, filename="context_cache.json"):
    path = safe_cache_path(filename)
    with open(path, "wb") as f:
        f.write(orjson.dumps(_to_json_safe(context_cache), option=orjson.OPT_INDENT_2))

def clean_cache_inplace(cache):
    """
    Remove all non-dict entries from the cache in-place. Returns number of removed entries.
    """
    if isinstance(cache, dict):
        keys_to_remove = [k for k, v in cache.items() if not isinstance(v, dict)]
        for k in keys_to_remove:
            del cache[k]
        return len(keys_to_remove)
    elif isinstance(cache, list):
        original_len = len(cache)
        cache[:] = [v for v in cache if isinstance(v, dict)]
        return original_len - len(cache)
    return 0

def _to_json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_safe(v) for v in obj]
    return obj
        
def segment_identity_hash(segment):
    """
    Returns a SHA-256 hash for segment identity (NOT for passwords).
    Uses normalized/truncated HTML, tag, and classes for performance and deduplication.
    Ignores dynamic attributes and normalizes whitespace for robustness.
    """
    tag = segment.get("tag", "").lower()

    classes = " ".join(sorted([c.lower() for c in segment.get("classes", [])]))
    attrs = segment.get("attrs", {})
    # Remove dynamic attributes from attrs for hashing
    attrs_filtered = {k: v for k, v in attrs.items() if not (k.startswith('_ngcontent-') or k.startswith('_nghost-') or k.startswith('ng-') or k.startswith('data-') or k in {'style', 'id', 'class', 'tabindex', 'aria-checked'})}
    html = segment.get("html", "").lower()
    # Aggressive normalization: strip all whitespace, remove dynamic attrs, collapse spaces
    html_norm = re.sub(r'\s+', ' ', re.sub(r'\s*([=;:,])\s*', r'\1', re.sub(r'\s+', ' ', html.strip())))[:256]
    base = tag + "|" + classes + "|" + orjson.dumps(attrs_filtered, option=orjson.OPT_SORT_KEYS).decode() + "|" + html_norm
    return hashlib.sha256(base.encode("utf-8")).hexdigest()

def batch_get_segment_embeddings(model, segments):
    """
    Efficiently get embeddings for a list of segments using batch encoding and cache.
    Skips segments with empty/trivial HTML (whitespace, only icons), returns None for those.
    Returns a list of embeddings in the same order as segments.
    """
    model_id = getattr(model, 'name_or_path', str(model))
    identities = [embedding_cache_hash(seg, model_id) if not is_trivial_segment(seg) else None for seg in segments]
    cached = [get_embedding_from_memory(identity) if identity else None for identity in identities]
    to_compute = [i for i, emb in enumerate(cached) if emb is None and identities[i] is not None]
    if isinstance(segments, list):
        segments[:] = [s for s in segments if isinstance(s, dict)]    
    if to_compute:
        texts = []
        idx_map = []
        for idx in to_compute:
            seg = segments[idx]
            tag = seg.get("tag", "")
            attrs = " ".join([f"{k}={v}" for k, v in seg.get("attrs", {}).items()])
            text = BeautifulSoup(seg.get("html", ""), "html.parser").get_text(" ", strip=True)
            if not text.strip():
                continue
            texts.append(f"{tag} {attrs} {text}")
            idx_map.append(idx)
        if texts:
            new_embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False, batch_size=16)
            for i, idx in enumerate(idx_map):
                save_embedding(identities[idx], new_embs[i])  # Save to disk cache
                cached[idx] = new_embs[i]
    return [emb if identity else None for emb, identity in zip(cached, identities)]