import hashlib
import orjson
import os
import re
import time
import threading
import traceback
import numpy as np
from typing import Dict, Any, List, Optional
import concurrent.futures
from ..config import CONTEXT_LIBRARY_PATH, CACHE_DIR, LOG_DIR, CONTEXT_CACHE_PATH
from ..utils.shared_logger import log_info, log_debug, log_warning, log_error
from ..bots.librarian import (
    HTML_TAGS, PANEL_TAGS, HEADING_TAGS, CUSTOM_ATTR_PATTERNS, LOCATION_KEYWORDS, 
    CANDIDATE_KEYWORDS, BALLOT_TYPES, update_context_library, load_context_library,
    log_unknown_tag, log_unknown_attr, get_canonical_segment_label, cache_segment_label, get_cached_segment_label, 
    ALWAYS_IGNORE_TAGS, ALWAYS_IGNORE_CLASSES, ALWAYS_IGNORE_IDS, ICON_CLASSES, ICON_TAGS, BUTTON_CLASSES,
    HEADING_CLASSES, PANEL_CLASSES, TIMESTAMP_CLASSES, STRUCTURAL_TAGS, TIMESTAMP_ID_PATTERNS, TIMESTAMP_ATTRS,
    CONTEST_KEYWORDS, PARTY_KEYWORDS, MISC_FOOTER_KEYWORDS, ELECTION_TYPES, UPDATE_PANEL_KEYWORDS, VIEW_BY_PHRASES,
    TOTAL_KEYWORDS, PERCENT_KEYWORDS, ROOT_CONTAINER_TAGS,
)
from ..utils.embedding_cache import (
    save_embedding, get_embedding_from_memory, load_embeddings_batch, save_embeddings_batch
)
from ..utils.user_prompt import prompt_user_input
from selectolax.parser import HTMLParser
from ..utils.model_registry import ModelRegistry
from difflib import get_close_matches

ENABLE_SEGMENT_LABEL_PROMPT = os.getenv("ENABLE_SEGMENT_LABEL_PROMPT", "true").lower() == "true"
console = None  # Only import rich.console.Console if needed for interactive output

def convert_ndarrays(obj):
    if isinstance(obj, dict):
        return {k: convert_ndarrays(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarrays(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# --- Caching and threading ---
_LABEL_CACHE_FILENAME = "segment_label_cache.json"
_LABEL_CACHE_LOCK = threading.Lock()
_LABEL_CACHE = None
_context_cache = None
_pattern_kb_cache = None

embedding_cache_hits = set()
embedding_cache_misses = set()

def robust_orjson_loads(val):
    if isinstance(val, bytes):
        return orjson.loads(val)
    elif isinstance(val, str):
        return orjson.loads(val.encode("utf-8"))
    else:
        raise TypeError(f"Cannot decode type {type(val)} with orjson")

def _get_label_cache_path():
    path = safe_cache_path(_LABEL_CACHE_FILENAME)
    if os.name == "nt" and len(os.path.abspath(path)) >= 260:
        import tempfile
        short_path = os.path.join(tempfile.gettempdir(), _LABEL_CACHE_FILENAME)
        log_warning(f"[CACHE] Path too long for Windows, using temp path: {short_path}")
        return short_path
    return path

def _load_label_cache():
    global _LABEL_CACHE
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
    with _LABEL_CACHE_LOCK:
        cache = _load_label_cache()
        cache[seg_hash] = {"label": label, "timestamp": int(time.time())}
        _save_label_cache()

def get_cached_segment_label(seg_hash):
    with _LABEL_CACHE_LOCK:
        cache = _load_label_cache()
        entry = cache.get(seg_hash, {})
        if entry:
            return entry.get("label", [])
        return None

def safe_cache_path(filename: str) -> str:
    filename = _sanitize_log_filename(filename)
    cache_folder = CACHE_DIR
    # Defensive: fallback to temp if path too long
    full_path = os.path.join(cache_folder, filename)
    if os.name == "nt" and len(os.path.abspath(full_path)) >= 240:
        import tempfile
        temp_path = os.path.join(tempfile.gettempdir(), filename)
        log_warning(f"[CACHE] Path too long for Windows, using temp path: {temp_path}")
        # Ensure temp dir exists
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        return temp_path
    # Ensure cache dir exists
    os.makedirs(cache_folder, exist_ok=True)
    if not os.path.abspath(full_path).startswith(os.path.abspath(cache_folder)):
        raise ValueError("Unsafe cache path detected!")
    return full_path

def safe_log_path(filename: str) -> str:
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
    html = re.sub(r'\s(_ngcontent-[^=]+|ng-version|ng-star-inserted|_nghost-[^=]+|_ngcontent-[^=]+|aria-checked|tabindex|style|data-[^=]+|id|class)="[^"]*"', '', html)
    html = re.sub(r'\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}', '', html)
    html = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}', '', html)
    html = re.sub(r'\d{1,2}:\d{2}(:\d{2})? ?(am|pm|AM|PM)?', '', html)
    html = re.sub(r'\s+', ' ', html.strip())
    return html[:maxlen]

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

def segment_identity_hash(segment):
    tag = segment.get("tag", "").lower()
    classes = " ".join(sorted([c.lower() for c in segment.get("classes", [])]))
    attrs = segment.get("attrs", {})
    attrs_filtered = {k: v for k, v in attrs.items() if not (k.startswith('_ngcontent-') or k.startswith('_nghost-') or k.startswith('ng-') or k.startswith('data-') or k in {'style', 'id', 'class', 'tabindex', 'aria-checked'})}
    html = segment.get("html", "").lower()
    html_norm = re.sub(r'\s+', ' ', re.sub(r'\s*([=;:,])\s*', r'\1', re.sub(r'\s+', ' ', html.strip())))[:256]
    base = tag + "|" + classes + "|" + orjson.dumps(attrs_filtered, option=orjson.OPT_SORT_KEYS).decode() + "|" + html_norm
    return hashlib.sha256(base.encode("utf-8")).hexdigest()

def embedding_cache_hash(segment, model_id):
    tag = segment.get("tag", "")
    attrs = segment.get("attrs", {})
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
    except Exception as e:
        segment["embedding_error"] = str(e)
        return None

def batch_get_segment_embeddings(model, segments):
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
            html = seg.get("html", "")
            try:
                tree = HTMLParser(html)
                text = tree.body.text(separator=" ", strip=True) if tree.body else tree.text(separator=" ", strip=True)
            except Exception:
                text = ""
            if not text.strip():
                continue
            texts.append(f"{tag} {attrs} {text}")
            idx_map.append(idx)
        if texts:
            # Parallelize encoding if large batch
            if len(texts) > 128:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    chunks = [texts[i:i+32] for i in range(0, len(texts), 32)]
                    results = list(executor.map(lambda chunk: model.encode(chunk, convert_to_numpy=True, show_progress_bar=False, batch_size=16), chunks))
                new_embs = np.concatenate(results)
            else:
                new_embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False, batch_size=16)
            for i, idx in enumerate(idx_map):
                save_embedding(identities[idx], new_embs[i])
                cached[idx] = new_embs[i]
    return [emb if identity else None for emb, identity in zip(cached, identities)]

def deduplicate_pattern_kb(pattern_kb):
    """Deduplicate pattern KB entries by segment_hash, keeping the latest timestamp."""
    dedup = {}
    for entry in pattern_kb:
        seg_hash = entry.get("segment_hash")
        ts = entry.get("timestamp", 0)
        if seg_hash not in dedup or ts > dedup[seg_hash].get("timestamp", 0):
            dedup[seg_hash] = entry
    return list(dedup.values())

def prune_embedding_cache(valid_hashes):
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

def submit_segment_correction(segment_hash, new_label, context_library=None):
    """Allow downstream modules to submit corrections for a segment label."""
    cache_segment_label(segment_hash, new_label)
    if context_library is not None:
        for seg in context_library.get("cached_segments", []):
            if seg.get("segment_hash") == segment_hash:
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
):
    seg_hash = segment_identity_hash(segment)
    # 1. Persistent label cache
    cached_label = get_cached_segment_label(seg_hash)
    if cached_label:
        return cached_label
    # 2. Context cache
    if context_cache and seg_hash in context_cache:
        label = context_cache[seg_hash].get("ml_label")
        if label:
            return label
    # 3. Context library
    if context_library and "cached_segments" in context_library:
        for seg in context_library["cached_segments"]:
            if seg.get("segment_hash") == seg_hash and seg.get("ml_label"):
                return seg["ml_label"]
    # 4. Pattern KB
    if pattern_kb:
        for entry in pattern_kb:
            if entry.get("segment_hash") == seg_hash and entry.get("label"):
                return entry["label"]
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
            emb = get_segment_embedding(model, segment)
            if emb is not None:
                best_label = "unknown"
                best_conf = 0.0
                for entry in pattern_kb:
                    kb_emb = np.array(entry.get("embedding", []))
                    if kb_emb.shape != emb.shape:
                        continue
                    sim = float(np.dot(emb, kb_emb) / (np.linalg.norm(emb) * np.linalg.norm(kb_emb) + 1e-8))
                    if sim > best_conf:
                        best_conf = sim
                        best_label = entry.get("label", "unknown")
                if best_conf >= ml_threshold and best_label != "unknown":
                    return best_label, "ml"
        except Exception:
            pass
    # 7. Heuristic fallback
    tag = segment.get("tag", "").lower()
    classes = [c.lower() for c in segment.get("classes", [])]
    attrs = segment.get("attrs", {})
    html = segment.get("html", "").lower()
    id_ = segment.get("id", "").lower()
    text = segment.get("text", "").strip().lower() if segment.get("text", []) else _extract_clean_text(html).lower()
    # --- Use librarian keywords for robust labeling ---
    # Contest title detection
    if _keyword_in_text(text, CONTEST_KEYWORDS) or _keyword_in_text(html, CONTEST_KEYWORDS):
        return "contest_title"
    # Candidate panel detection
    if _keyword_in_text(text, CANDIDATE_KEYWORDS) or _keyword_in_text(html, CANDIDATE_KEYWORDS):
        return "candidate_panel"
    # Party label detection
    if _keyword_in_text(text, PARTY_KEYWORDS) or _keyword_in_text(html, PARTY_KEYWORDS):
        return "party_label"
    # Location panel detection
    if _keyword_in_text(text, LOCATION_KEYWORDS) or _keyword_in_text(html, LOCATION_KEYWORDS):
        return "location_panel"
    # Ballot type detection
    if _keyword_in_text(text, BALLOT_TYPES) or _keyword_in_text(html, BALLOT_TYPES):
        return "ballot_types"
    # Table detection (results table)
    if tag == "table" or _keyword_in_text(text, TOTAL_KEYWORDS | PERCENT_KEYWORDS | MISC_FOOTER_KEYWORDS):
        return "results_table"
    # Heading detection
    if tag in HEADING_TAGS or HEADING_CLASSES & set(classes):
        return "heading"
    # Panel detection
    if tag in PANEL_TAGS or PANEL_CLASSES & set(classes):
        return "panel"
    # Timestamp detection
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
        href = str(attrs["href"]).lower()
        if any(href.endswith(ext) for ext in [".csv", ".json", ".pdf", ".xlsx", ".zip", ".xls", ".doc", ".docx"]):
            return "download_link"
    if segment.get("is_button", []) or BUTTON_CLASSES & set(classes) or "toggle" in id_:
        return "ballot_toggle"
    if tag in HEADING_TAGS or HEADING_CLASSES & set(classes):
        return "heading"
    if tag in PANEL_TAGS or PANEL_CLASSES & set(classes):
        return "panel"
    if tag == "table":
        return "results_table"
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
    if any(bt in html for bt in BALLOT_TYPES):
        return "ballot_types"
    if segment.get("is_clickable", []):
        return "clickable"
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

def _extract_clean_text(html):
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

def _label_in(label, target):
    """Robustly checks if label is or contains the target label."""
    if isinstance(label, str):
        return label == target
    if isinstance(label, list):
        return target in label
    return False

def _extract_segments_by_label(segments, label_name, extra_fields=None):
    """
    Extracts and cleans segments by label, returns list of dicts with clean text and extra fields.
    Skips segments with empty, whitespace-only, or trivial text (e.g., only tags or &nbsp;).
    """
    results = []
    for seg in segments:
        label = seg.get("ml_label")
        if _label_in(label, label_name):
            text = _extract_clean_text(seg.get("html", ""))
            # Skip if text is empty, whitespace, or just HTML tags/entities
            if not text or not text.strip():
                continue
            # Skip if text is just &nbsp; or similar
            if text.strip() in {"&nbsp;", "&#160;"}:
                continue
            # Skip if text is just a tag (e.g., "<br>")
            if re.fullmatch(r"<[^>]+>", text.strip()):
                continue
            entry = {
                "text": text,
                "raw_html": seg.get("html", ""),
                "segment_hash": seg.get("segment_hash"),
            }
            if extra_fields:
                for field in extra_fields:
                    entry[field] = seg.get(field)
            results.append(entry)
    return results

def _keyword_in_text(text, keywords):
    """Check if any keyword is present in the text (case-insensitive, word-boundary)."""
    text = text.lower()
    for kw in keywords:
        if re.search(rf'\b{re.escape(kw.lower())}\b', text):
            return True
    return False
def extract_year_and_type(text):
    """
    Extracts the most likely year and election type from anywhere in the string.
    Picks the last year and the most frequent or last type found.
    Returns (year, election_types, cleaned_text)
    """
    # Find all years (4 digits, 2020-2099)
    years = re.findall(r'(20\d{2})', text)
    year = years[-1] if years else None

    # Find all types (case-insensitive, from ELECTION_TYPE)
    type_matches = []
    for t in ELECTION_TYPES:
        for m in re.finditer(rf'\b{re.escape(t)}\b', text, re.IGNORECASE):
            type_matches.append((m.start(), t))
    # Pick the last type found, or the most frequent if multiple
    type_found = None
    if type_matches:
        # Option 1: last found
        type_found = sorted(type_matches, key=lambda x: x[0])[-1][1]
        # Option 2: most frequent (uncomment if this is preferred)
        # from collections import Counter
        # type_found = Counter([t for _, t in type_matches]).most_common(1)[0][0]

    # Remove all years/types from text for cleaner title
    cleaned = text
    if years:
        for y in years:
            cleaned = re.sub(rf'\b{y}\b', '', cleaned)
    if type_matches:
        for _, t in type_matches:
            cleaned = re.sub(rf'\b{re.escape(t)}\b', '', cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip(" -:|,")
    return year, type_found, cleaned

def is_update_panel(text):
    """
    Detects if a panel/heading is a last-updated, status, or reporting info panel.
    Uses robust keyword and phrase matching.
    """
    t = text.lower()
    # Direct keyword match
    if any(kw in t for kw in UPDATE_PANEL_KEYWORDS):
        return True
    # Dynamic "view by ..." phrase match
    if any(f"view by {phrase}" in t for phrase in VIEW_BY_PHRASES):
        return True
    # Common "as of" or "updated" patterns
    if "as of" in t or "updated" in t:
        return True
    return False

def split_possible_contests(text):
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

def scan_html_for_context(
    target_url,
    page,
    coordinator=None,
    debug=False,
    context_cache=None,
    model_name: Optional[str] = None,
    use_finetuned: bool = True,
    non_interactive=False,    
) -> Dict[str, Any]:
    """
    Main pipeline entry: Efficient, dynamic, and feedback-driven HTML scanner.
    Leverages ContextCoordinator for context, ML model, and feedback logs.
    """
    start_time = time.time()
    page_hash = get_page_hash(page)
    if context_cache is None:
        context_cache = load_context_cache_from_disk()
    # --- FAST-PATH: If all segment hashes are in cache with high confidence, skip full scan ---
    html = page.content()
    def extract_all_segment_html(html):
        try:
            tree = HTMLParser(html)
            return [n.html for n in tree.root.traverse() if hasattr(n, "html")]
        except Exception:
            return []
    segment_htmls = extract_all_segment_html(html)
    segment_hashes = [segment_hash(h) for h in segment_htmls]
    fast_path_hits = [
        h for h in segment_hashes
        if h in context_cache and context_cache[h].get("ml_confidence", 0) > 0.95
    ]
    if len(fast_path_hits) == len(segment_hashes) and segment_hashes:
        log_info("[FAST-PATH] All segments covered by cache. Skipping full scan.")
        return {h: context_cache[h] for h in segment_hashes}
    if page_hash in context_cache:
        log_info(f"[SCAN] Using cached context for {target_url}")
        log_info("[bold green][CACHE] Entire context loaded from cache. Skipping scan.[/bold green]")
        return context_cache[page_hash]

    context_result = {
        "raw_html": "",
        "tagged_segments": [],
        "tagged_segments_with_attrs": [],
        "metadata": {},
        "selector_log": [],
        "error": None,
        "url": page.url,
        "pattern_kb_matches": [],
        "segments_needing_review": [],
    }

    try:
        # --- 1. Get context library, pattern KB, and ML model from coordinator if available ---
        if coordinator:
            context_library = getattr(coordinator, "library", None)
            pattern_kb = getattr(coordinator, "pattern_kb", None)
            model = getattr(coordinator, "_semantic_model", None)
            # Optionally merge in feedback logs
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
            except Exception:
                context_library = {}
            pattern_kb = load_pattern_kb()
            model = ModelRegistry.get_sentence_transformer(model_name=model_name, use_finetuned=use_finetuned)
        # --- 2. Extract segments with attributes and ML labels ---
        segments_with_attrs = extract_tagged_segments_with_attrs(
            html,
            context_cache=context_cache,
            include_data_attrs=True,
            fallback_on_error=True,
            pattern_kb=pattern_kb,
            ml_threshold=0.85,
            context_library=context_library,
            model=model,
            coordinator=coordinator
        )
        context_result["tagged_segments_with_attrs"] = segments_with_attrs
        context_result["tagged_segments"] = [seg["html"] for seg in segments_with_attrs]

        # --- Helper for diagnostics and filtering ---
        def diagnostics_and_filter(data, name, required_fields=None, max_title_len=500):
            # Diagnostics
            if data:
                avg_len = sum(len(str(d.get("title", d.get("text", "")))) for d in data) / len(data)
                log_info(f"[{name.upper()}] Extracted {len(data)} items, avg title/text length: {avg_len:.1f}")
            else:
                log_warning(f"[{name.upper()}] No valid items extracted after validation.")
            # Filtering
            filtered = []
            filtered_out = []
            for d in data:
                print(f"[DEBUG][{name.upper()}] candidate: {d}")
                title = d.get("title", d.get("text", ""))
                # Only filter out if title is None or empty after stripping
                if title is None or (isinstance(title, str) and len(title.strip()) == 0) or len(title) > max_title_len:
                    filtered_out.append((d, "missing or invalid title/text"))
                    continue
                if required_fields:
                    if not any(d.get(f) for f in required_fields):
                        filtered_out.append((d, f"missing required fields: {required_fields}"))
                        continue
                filtered.append(d)
            if filtered_out:
                log_warning(f"[{name.upper()}] Filtered out {len(filtered_out)} items due to missing/invalid fields.")
                for d, reason in filtered_out[:5]:
                    log_warning(f"  [Filtered] {reason}: {str(d)[:100]}...")
            if not filtered:
                log_warning(f"[{name.upper()}] No items with required fields for downstream output.")
            return filtered

        # --- Robust extraction for all key segment types ---
        state = context_result.get("state")
        county = context_result.get("county")
        year = context_result.get("year")

        # Contests
        raw_contests = []
        for seg in _extract_segments_by_label(segments_with_attrs, "contest_title"):
            text = seg["text"]
            # Split long blocks into possible contests
            for possible in split_possible_contests(text):
                seg_year, seg_type, cleaned_title = extract_year_and_type(possible)
                # Only treat as contest if it contains a contest keyword or a valid type
                if any(kw in possible.lower() for kw in CONTEST_KEYWORDS) or seg_type:
                    raw_contests.append({
                        "title": cleaned_title,
                        "state": state,
                        "county": county,
                        "year": seg_year or year,
                        "type_": seg_type,
                        "segment_hash": seg["segment_hash"],
                    })
        contests = diagnostics_and_filter(
            raw_contests, "contest",
            required_fields=["title"],
            max_title_len=500,
        )
        context_result["contests"] = contests

        # Panels
        raw_panels = [
            {
                "panel_text": seg["text"],
                "panel_html": seg["raw_html"],
                "segment_hash": seg["segment_hash"],
            }
            for seg in _extract_segments_by_label(segments_with_attrs, "panel")
        ]
        panels = diagnostics_and_filter(
            raw_panels, "panel",
            required_fields=["panel_text"],
            max_title_len=1000,
        )
        context_result["panels"] = panels

        # Tables
        raw_tables = []
        for seg in _extract_segments_by_label(segments_with_attrs, "results_table"):
            text = seg["text"]
            # Optionally extract year/type if present in table captions or headings
            seg_year, seg_type, cleaned_text = extract_year_and_type(text)
            raw_tables.append({
                "table_text": cleaned_text,
                "table_html": seg["raw_html"],
                "year": seg_year,
                "type_": seg_type,
                "segment_hash": seg["segment_hash"],
            })
        tables = diagnostics_and_filter(
            raw_tables, "table",
            required_fields=["table_text"],
            max_title_len=10000,

        )
        context_result["tables"] = tables

        # Candidate Panels
        raw_candidate_panels = []
        for seg in _extract_segments_by_label(segments_with_attrs, "candidate_panel"):
            text = seg["text"]
            seg_year, seg_type, cleaned_text = extract_year_and_type(text)
            raw_candidate_panels.append({
                "candidate_panel_text": cleaned_text,
                "candidate_panel_html": seg["raw_html"],
                "year": seg_year,
                "type_": seg_type,
                "segment_hash": seg["segment_hash"],
            })
        candidate_panels = diagnostics_and_filter(
            raw_candidate_panels, "candidate_panel",
            required_fields=["candidate_panel_text"],
            max_title_len=1000,

        )
        context_result["candidate_panels"] = candidate_panels

        # Location Panels
        raw_location_panels = []
        for seg in _extract_segments_by_label(segments_with_attrs, "location_panel"):
            text = seg["text"]
            seg_year, seg_type, cleaned_text = extract_year_and_type(text)
            raw_location_panels.append({
                "location_panel_text": cleaned_text,
                "location_panel_html": seg["raw_html"],
                "year": seg_year,
                "type_": seg_type,
                "segment_hash": seg["segment_hash"],
            })
        location_panels = diagnostics_and_filter(
            raw_location_panels, "location_panel",
            required_fields=["location_panel_text"],
            max_title_len=1000,

        )
        context_result["location_panels"] = location_panels

        # Headings (with update/timestamp detection)
        raw_headings = []
        for seg in _extract_segments_by_label(segments_with_attrs, "heading"):
            text = seg["text"]
            if is_update_panel(text):
                raw_headings.append({
                    "heading_text": text,
                    "heading_html": seg["raw_html"],
                    "segment_hash": seg["segment_hash"],
                    "heading_type": "last_webpage_update"
                })
                ts_match = re.search(r'(\w+day,?\s+\w+\s+\d{1,2},\s+20\d{2}.*\d{1,2}:\d{2}(:\d{2})?\s*(AM|PM)?)', text)
                if ts_match:
                    context_result["metadata"]["last_webpage_update"] = ts_match.group(1)
            else:
                raw_headings.append({
                    "heading_text": text,
                    "heading_html": seg["raw_html"],
                    "segment_hash": seg["segment_hash"],
                    "heading_type": "content"
                })

        headings = diagnostics_and_filter(
            raw_headings, "heading",
            required_fields=["heading_text"],
            max_title_len=500,

        )
        context_result["headings"] = headings

        # Ballot Types
        raw_ballot_types = []
        for seg in _extract_segments_by_label(segments_with_attrs, "ballot_types"):
            text = seg["text"]
            seg_year, seg_type, cleaned_text = extract_year_and_type(text)
            raw_ballot_types.append({
                "ballot_types_text": cleaned_text,
                "ballot_types_html": seg["raw_html"],
                "year": seg_year,
                "type_": seg_type,
                "segment_hash": seg["segment_hash"],
            })
        ballot_types = diagnostics_and_filter(
            raw_ballot_types, "ballot_types",
            required_fields=["ballot_types_text"],
            max_title_len=200,

        )
        context_result["ballot_types"] = ballot_types

        # Results Timestamps
        raw_results_timestamps = []
        for seg in _extract_segments_by_label(segments_with_attrs, "results_timestamp"):
            text = seg["text"]
            # Try to extract timestamp string for metadata
            ts_match = re.search(r'(\w+day,?\s+\w+\s+\d{1,2},\s+20\d{2}.*\d{1,2}:\d{2}(:\d{2})?\s*(AM|PM)?)', text)
            if ts_match:
                context_result["metadata"]["results_last_updated"] = ts_match.group(1)
            raw_results_timestamps.append({
                "timestamp_text": text,
                "timestamp_html": seg["raw_html"],
                "segment_hash": seg["segment_hash"],
            })
        results_timestamps = diagnostics_and_filter(
            raw_results_timestamps, "results_timestamp",
            required_fields=["timestamp_text"],
            max_title_len=200,

        )
        context_result["results_timestamps"] = results_timestamps

        # Party Labels
        raw_party_labels = []
        for seg in _extract_segments_by_label(segments_with_attrs, "party_label"):
            text = seg["text"]
            raw_party_labels.append({
                "party_label_text": text,
                "party_label_html": seg["raw_html"],
                "segment_hash": seg["segment_hash"],
            })
        party_labels = diagnostics_and_filter(
            raw_party_labels, "party_label",
            required_fields=["party_label_text"],
            max_title_len=200,

        )
        context_result["party_labels"] = party_labels

        # Vote Methods
        raw_vote_methods = []
        for seg in _extract_segments_by_label(segments_with_attrs, "vote_method"):
            text = seg["text"]
            raw_vote_methods.append({
                "vote_method_text": text,
                "vote_method_html": seg["raw_html"],
                "segment_hash": seg["segment_hash"],
            })
        vote_methods = diagnostics_and_filter(
            raw_vote_methods, "vote_method",
            required_fields=["vote_method_text"],
            max_title_len=200,

        )
        context_result["vote_methods"] = vote_methods

        # --- 3. ML-driven DOM pattern clustering and tagging ---
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
            if seg.get("ml_confidence", 0.0) < 0.7 or seg.get("ml_label", "unknown") == "unknown":
                # Use coordinator as oracle if available
                user_label = None
                if coordinator and hasattr(coordinator, "auto_label_segment"):
                    try:
                        user_label = coordinator.auto_label_segment(seg)
                    except Exception:
                        user_label = None
                if not user_label:
                    user_label = prompt_for_segment_label(seg, context_library=context_library)
                seg["ml_label"] = user_label
                seg["ml_confidence"] = 1.0
                seg["pattern_id"] = f"pattern_{hashlib.sha256(seg['html'].encode('utf-8')).hexdigest()[:10]}"
                emb = get_segment_embedding(model, seg, cache_hits=embedding_cache_hits, cache_misses=embedding_cache_misses)
                if emb is not None:
                    emb = emb.tolist()
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
                if context_library is not None and seg.get("segment_hash", []):
                    update_context_library(
                        CONTEXT_LIBRARY_PATH,
                        lambda lib: lib.setdefault("cached_segments", []).append({
                            "segment_hash": seg["segment_hash"],
                            "ml_label": user_label,
                        })
                    )
                    valid_hashes = set(seg["segment_hash"] for seg in context_library.get("cached_segments", []))
                    prune_embedding_cache(valid_hashes)
            else:
                pattern_matches.append({
                    "pattern_id": seg["pattern_id"],
                    "label": seg["ml_label"],
                    "confidence": seg["ml_confidence"],
                    "segment_html": seg["html"][:200],
                })

        context_result["pattern_kb_matches"] = pattern_matches
        context_result["segments_needing_review"] = segments_needing_review

        # --- 4. Dynamic tagging and context enrichment ---
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
            "pattern_kb_size": len(pattern_kb) if pattern_kb else 0,
        })

        if debug:
            log_debug("\n[orange][DEBUG] Extracted HTML segments with ML labels:[/orange]")
            for seg in segments_with_attrs:
                log_info(f"{seg['tag']} {seg['attrs']} [label={seg['ml_label']}, conf={seg['ml_confidence']:.2f}] {seg['html'][:80]}{'...' if len(seg['html']) > 80 else ''}")
            if segments_needing_review:
                log_debug(f"\n[red][DEBUG] {len(segments_needing_review)} segments flagged for review.[/red]")

        # --- 5. Update context library with new segments for future runs ---
        if context_library is not None:
            if "cached_segments" not in context_library:
                context_library["cached_segments"] = []
            known_hashes = {seg.get("segment_hash", []) for seg in context_library["cached_segments"]}
            for seg in segments_with_attrs:
                if seg.get("segment_hash", []) and seg["segment_hash"] not in known_hashes:
                    context_library["cached_segments"].append({
                        "segment_hash": seg["segment_hash"],
                        "ml_label": seg["ml_label"],
                        "ml_confidence": seg["ml_confidence"],
                        "pattern_id": seg["pattern_id"],
                    })
            update_context_library(
                CONTEXT_LIBRARY_PATH,
                lambda lib: lib.setdefault("cached_segments", []).extend([
                    {
                        "segment_hash": seg["segment_hash"],
                        "ml_label": seg["ml_label"],
                        "ml_confidence": seg["ml_confidence"],
                        "pattern_id": seg["pattern_id"],
                    }
                    for seg in segments_with_attrs
                    if seg.get("segment_hash", []) and seg["segment_hash"] not in known_hashes
                ])
            )
            valid_hashes = set(seg["segment_hash"] for seg in context_library.get("cached_segments", []))
            prune_embedding_cache(valid_hashes)
    except Exception as e:
        tb = traceback.format_exc()
        log_error(f"[SCAN ERROR] HTML parsing failed: {e}\n{tb}")
        log_error(f"[SCAN ERROR] HTML parsing failed: {e}", extra={"traceback": tb, "url": getattr(page, 'url', None)})
        context_result["error"] = f"[SCAN ERROR] HTML parsing failed: {e}\n{tb}"

    if context_cache is not None:
        context_cache[page_hash] = context_result
        save_context_cache_to_disk(context_cache)
    if embedding_cache_hits and not embedding_cache_misses:
        log_info(f"[bold green][CACHE] All segment embeddings loaded from cache.[/bold green]")
    elif embedding_cache_hits:
        log_warning(f"[yellow][CACHE] {len(embedding_cache_hits)} embeddings loaded from cache, {len(embedding_cache_misses)} computed.[/yellow]")
    log_info(f"[PROFILE] scan_html_for_context completed in {time.time() - start_time:.2f} seconds.")
    return context_result

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
    coordinator=None
) -> List[Dict[str, Any]]:
    """
    Extract DOM segments with attributes and ML-driven semantic labels.
    Uses selectolax for DOM, leverages context, pattern KB, and coordinator for optimal labeling.
    """
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
        def walk(node, parent_idx=None, heading_idx=None, panel_idx=None):
            tag = node.tag
            if not tag or tag.lower() not in HTML_TAGS:
                log_unknown_tag(tag, context_library)
                for child in node.iter(include_text=True):
                    walk(child, parent_idx, heading_idx, panel_idx)
                return
            attrs = dict(node.attributes)
            if include_data_attrs:
                attrs.update({k: v for k, v in node.attributes.items() if k.startswith("data-")})
            classes = attrs.get("class", "").split() if "class" in attrs else []
            id_ = attrs.get("id", "")
            is_button = tag == "button" or (tag == "input" and attrs.get("type_", "").lower() in ["button", "submit"])
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
                "context_heading_idx": this_heading_idx,
                "panel_ancestor_idx": this_panel_idx,
                "panel_ancestor_heading": None,
            }
            # Now safe to reference seg
            for k in attrs:
                if any(pat.match(k) for pat in custom_attr_patterns):
                    seg["has_custom_attr"] = True
                log_unknown_attr(k, context_library)
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
            for child in node.iter(include_text=True):
                child_idx = walk(child, this_idx, this_heading_idx, this_panel_idx)
                if child_idx is not None:
                    seg["children"].append(child_idx)
            return this_idx

        root = tree.body or tree.html or tree.root
        walk(root)

        # --- Batch embedding for all segments ---
        seg_hashes = [segment_hash(seg.get("html", "")) for seg in segments]
        seg_htmls = [seg.get("html", "") for seg in segments]
        total_segments = len(seg_hashes)
        log_info(f"[EMBED] Total segments: {total_segments}")

        hash_to_embedding = {}
        CHUNK_SIZE = 1024
        for i in range(0, total_segments, CHUNK_SIZE):
            chunk_hashes = seg_hashes[i:i+CHUNK_SIZE]
            chunk_result = load_embeddings_batch(chunk_hashes)
            hash_to_embedding.update(chunk_result)
            hits = sum(1 for v in chunk_result.values() if v is not None)
            log_debug(f"[EMBED] Batch {i//CHUNK_SIZE+1}: {hits} hits, {len(chunk_hashes)-hits} misses")
        missing = [(h, html) for h, html in zip(seg_hashes, seg_htmls) if hash_to_embedding.get(h) is None]
        if missing:
            log_info(f"[EMBED] Computing {len(missing)} missing embeddings in chunks of {CHUNK_SIZE}")
            for i in range(0, len(missing), CHUNK_SIZE):
                chunk = missing[i:i+CHUNK_SIZE]
                missing_hashes, missing_htmls = zip(*chunk)
                try:
                    new_embs = model.encode(list(missing_htmls), convert_to_numpy=True, show_progress_bar=False)
                except Exception as e:
                    log_error(f"[EMBED] Batch embedding computation failed: {e}")
                    continue
                save_embeddings_batch(list(zip(missing_hashes, new_embs)))
                for h, emb in zip(missing_hashes, new_embs):
                    hash_to_embedding[h] = emb
                log_debug(f"[EMBED] Saved {len(chunk)} new embeddings to cache.")
        for seg, h in zip(segments, seg_hashes):
            seg["_embedding"] = hash_to_embedding[h]
        log_info(f"[EMBED] Embedding assignment complete for {len(segments)} segments.")

        # --- Label segments using all available context, pattern KB, and coordinator ---
        for seg in segments:
            text = seg.get("html", "").lower()
            seg["contains_election_keyword"] = any(
                kw in text for kw in (list(location_keywords) + list(candidate_keywords) + list(ballot_types))
            )
            seg["contains_candidate"] = any(
                cand in text for cand in candidate_keywords
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
            seg["pattern_id"] = f"pattern_{hashlib.sha256(seg['html'].encode('utf-8')).hexdigest()[:10]}"
            # --- Downstream actionable flags ---
            seg["is_actionable"] = label in ("results_table", "contest_title", "candidate_panel", "location_panel")
            seg["is_election_result"] = label == "results_table"
            seg["is_contest_title"] = label == "contest_title"
        return segments

    except Exception as e:
        log_error(f"[FALLBACK] selectolax failed: {e}", extra={"traceback": traceback.format_exc(), "html_snippet": html[:200]})
        if not fallback_on_error:
            raise
        return []

def get_page_hash(page):
    content = page.content()
    return hashlib.sha256(content.encode("utf-8")).hexdigest()

def load_context_cache_from_disk(filename=None):
    global _context_cache
    if filename is None:
        filename = os.path.basename(CONTEXT_CACHE_PATH)
    path = safe_cache_path(filename)
    print(f"[DEBUG] Loading context cache from: {path}")
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                raw_cache = robust_orjson_loads(f.read())
                _context_cache = {k: v for k, v in raw_cache.items() if isinstance(v, dict)}
                return _context_cache
        except Exception as e:
            log_error(f"[ERROR] Failed to load {filename}: {e}")
            return {}
    _context_cache = {}
    return {}

def save_context_cache_to_disk(context_cache, path=CONTEXT_CACHE_PATH):
    print(f"[DEBUG] Saving context cache to: {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    context_cache = convert_ndarrays(context_cache)
    with open(path, "wb") as f:
        f.write(orjson.dumps(context_cache))

def clean_cache_inplace(cache):
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

def append_pattern_kb(entry):
    if not isinstance(entry, dict):
        raise ValueError("Only dict entries can be written to dom_pattern_kb.jsonl")
    entry = convert_ndarrays(entry)
    if "embedding" in entry and isinstance(entry["embedding"], np.ndarray):
        entry["embedding"] = entry["embedding"].tolist()
    path = safe_log_path("dom_pattern_kb.jsonl")
    with open(path, "ab") as f:
        f.write(orjson.dumps(entry, option=orjson.OPT_INDENT_2) + b"\n")

def append_feedback_log(entry):
    if not isinstance(entry, dict):
        raise ValueError("Only dict entries can be written to segment_feedback_log.jsonl")
    entry = convert_ndarrays(entry)
    if "embedding" in entry and isinstance(entry["embedding"], np.ndarray):
        entry["embedding"] = entry["embedding"].tolist()
    path = safe_log_path("segment_feedback_log.jsonl")
    with open(path, "ab") as f:
        f.write(orjson.dumps(entry, option=orjson.OPT_INDENT_2) + b"\n")
    global _pattern_kb_cache
    if "pattern_id" in entry and "label" in entry and "html" in entry:
        seg_hash = segment_identity_hash({"tag": entry.get("tag", ""), "attrs": entry.get("attrs", {}), "html": entry["html"]})
        kb_entry = {
            "pattern_id": entry["pattern_id"],
            "label": entry["label"],
            "embedding": entry.get("embedding", []),
            "example_html": entry["html"][:500],
            "segment_hash": seg_hash,
            "timestamp": entry.get("timestamp", 0),
        }
        if _pattern_kb_cache is not None and isinstance(_pattern_kb_cache, list):
            _pattern_kb_cache.append(kb_entry)

def prompt_for_segment_label(segment, context_library=None):
    seg_hash = segment_identity_hash(segment)
    cached_label = get_cached_segment_label(seg_hash)
    if cached_label:
        return cached_label
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
    if not html_preview:
        html_preview = f"[No HTML] tag={segment.get('tag', [])} attrs={segment.get('attrs', [])}"
    log_warning(f"\n[bold yellow]Segment needs review:[/bold yellow]\n{html_preview[:200]}{'...' if len(html_preview) > 200 else ''}")
    log_info(
        "[cyan]What is the semantic role of this segment? (e.g., results_table, ballot_toggle, heading, panel, candidate_panel, location_panel, ballot_types, results_timestamp, download_link, clickable, footer, legend, contest_title, party_label, vote_method, reporting_status, summary, error_message, warning, info_box, navigation, pagination, tab, modal, tooltip, ignore, unknown, etc.)[/cyan]"
    )
    label = prompt_user_input("> ").strip()
    cache_segment_label(seg_hash, label)
    return label

def segment_hash(html):
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
    def sort_attrs(match):
        tag = match.group(1)
        attrs = match.group(2)
        # Split attributes, sort, and rejoin
        attrs_list = re.findall(r'(\S+="[^"]*"|\S+=\'[^\']*\')', attrs)
        attrs_sorted = ' '.join(sorted(attrs_list))
        return f"<{tag} {attrs_sorted}>"

    html = re.sub(r'<(\w+)\s+([^>]+)>', sort_attrs, html)

    return html