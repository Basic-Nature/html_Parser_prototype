# shared_logic.py - Common parsing utilities for context-integrated pipeline
from __future__ import annotations
import difflib
import os
import platform
import re
import time
import numpy as np
import inspect
from sqlalchemy.orm import Session
from ..utils.shared_logger import SharedLogger, RichConsoleProxy
from ..utils.user_prompt import UserPrompt
from playwright.sync_api import Page, Locator, ElementHandle
from sentence_transformers import SentenceTransformer
from ..Context_Integration.Context_Library.constants import (
    STATE_ABBR, STATE_MODULE_MAP, KNOWN_STATE_TO_COUNTY_MAP, KNOWN_COUNTY_TO_PRECINCTS_MAP
)
from typing import (
    TYPE_CHECKING, Optional, Generator, Any, Iterable, Dict, 
    Union, Iterable, Collection, Protocol, Awaitable, TypedDict,
    List, Callable, Mapping, Sequence
)
if TYPE_CHECKING:
    from ..Context_Integration.context_coordinator import ContextCoordinator

assert set(STATE_MODULE_MAP.keys()) == set(KNOWN_STATE_TO_COUNTY_MAP.keys()), \
    "STATE_MODULE_MAP and KNOWN_STATE_TO_COUNTY_MAP keys are out of sync!"
console = RichConsoleProxy()   
prompt = UserPrompt()
logger = SharedLogger()

class HasAllMethod(Protocol):
    def all(self) -> List[Any]:
        """        Returns:
            List[Any]: All items from the result set.
        """
        ...

class PredictionResult(TypedDict, total=False):
    year: Optional[int]
    type_: Optional[str]
    election_types: Optional[List[str]]
    state: Optional[str]
    state_abbr: Optional[str]
    county: Optional[str]
    county_fips: Optional[str]
    district: Optional[str]
    office: Optional[str]
    office_level: Optional[str]
    party: Optional[str]
    candidate: Optional[str]
    precinct: Optional[str]
    ballot_type: Optional[str]
    vote_method: Optional[str]
    timestamp: Optional[str]
    source_url: Optional[str]
    confidence: Optional[float]
    extra: Optional[Dict[str, Any]]
    # Add more fields as needed

class EventLike(Protocol):
    def is_set(self) -> bool:
        """
        Checks if the event is set.
        Returns:
            bool: True if set, False otherwise.
        """
        ...

class Predictable(Protocol):
    def predict(self, text: str) -> Union[PredictionResult, Dict[str, Any]]:
        """
        Predicts structured contest metadata from input text.
        Args:
            text (str): The contest label or description.
        Returns:
            PredictionResult: Dict-like object with keys such as 'year', 'type_', 'state', 'county'.
        Raises:
            Exception: If prediction fails.
        """
        ...

    # Optionally support async models
    def apredict(self, text: str) -> Awaitable[Union[PredictionResult, Dict[str, Any]]]:
        """
        Asynchronously predicts structured contest metadata from input text.
        Args:
            text (str): The contest label or description.
        Returns:
            Awaitable[PredictionResult]: Awaitable dict-like object with keys such as 'year', 'type_', 'state', 'county'.
        Raises:
            Exception: If prediction fails.
        """
        ...

def safe_merge_defaults(existing: dict, defaults: dict) -> bool:
    """
    Recursively merge defaults into existing dict, only setting missing keys.
    Uses safe_get for robust access.
    Returns True if any changes were made.
    """
    changed = False
    for k, v in defaults.items():
        if safe_get(existing, k, None) is None:
            existing[k] = v
            changed = True
        elif isinstance(v, dict) and isinstance(existing[k], dict):
            if safe_merge_defaults(existing[k], v):
                changed = True
    return changed

def safe_strip(val) -> str:
    try:
        return val.strip() if isinstance(val, str) else str(val).strip()
    except Exception:
        return ""

def safe_setdefault(d: dict, key, default):
    """
    Robust setdefault: returns d[key] if present, else sets d[key]=default and returns default.
    Handles None and missing keys safely.
    """
    val = safe_get(d, key, None)
    if val is None:
        d[key] = default
        return default
    return val

def safe_tolist(val):
    """
    Safely convert val to a list.
    Handles numpy arrays, tuples, sets, and returns [val] for scalars.
    Returns an empty list for None.
    """
    import numpy as np
    if val is None:
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, (tuple, set)):
        return list(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    try:
        # Try to convert if it's an iterable (but not string/bytes)
        if hasattr(val, '__iter__') and not isinstance(val, (str, bytes)):
            return list(val)
    except Exception:
        pass
    return [val]

def safe_execute(session: Session, stmt) -> Optional[Any]:
    """
    Safely execute a SQLAlchemy statement.
    Returns result or None if not supported.
    """
    if hasattr(session, "execute") and callable(session.execute):
        try:
            return session.execute(stmt)
        except Exception:
            return None
    return None

def safe_all(rows: HasAllMethod) -> list:
    """
    Safely call .all() on a SQLAlchemy result/scalars object.
    Returns list or empty list if not supported.
    """
    if hasattr(rows, "all") and callable(getattr(rows, "all", None)):
        try:
            return rows.all()
        except Exception:
            return []
    return []

def safe_attr_keys(attrs) -> list:
    """
    Safely get a list of attribute keys from a dict-like object.
    Handles None, non-dict, and exceptions robustly.
    Returns a list of strings (keys), lowercased for consistency.
    """
    try:
        if isinstance(attrs, dict):
            return [safe_lower(str(k)) for k in attrs.keys()]
        # Try to convert to dict if possible
        return [safe_lower(str(k)) for k in dict(attrs).keys()]
    except Exception:
        return []

def safe_replace(val: str, old: str, new: str) -> str:
    """
    Safely call .replace on a string-like object.
    Returns the replaced string, or the original value if not a string or error occurs.
    """
    try:
        if isinstance(val, str):
            return val.replace(old, new)
        return str(val).replace(old, new)
    except Exception as e:
        logger.error(f"[safe_replace] Error: {e}")
        return str(val)

def safe_isdigit(val: Any) -> bool:
    """Safely check if val is a string and isdigit()."""
    try:
        return isinstance(val, str) and val.isdigit()
    except Exception:
        return False

def safe_get(dct: dict, key: str, default: Any = None) -> Any:
    """Safely get a value from a dict-like object."""
    try:
        if isinstance(dct, dict):
            return dct.get(key, default)
    except Exception:
        pass
    return default

def safe_values(obj: Mapping[Any, Any]) -> list:
    """
    Safely get .values() from a dict-like object.
    Returns an empty list if obj is not a dict or .values() fails.
    """
    try:
        if hasattr(obj, "values") and callable(obj.values):
            return list(obj.values())
    except Exception as e:
        logger.error(f"[safe_values] .values() failed: {e}")
    return []

def safe_is_set(event_like: EventLike) -> bool:
    """
    Safely check if an object has a callable is_set() method and call it.
    Returns False if not supported or any error occurs.
    """
    try:
        if hasattr(event_like, "is_set") and callable(event_like.is_set):
            return event_like.is_set()
    except Exception:
        pass
    return False

def safe_append_cached_segment(lib, seg_hash, user_label) -> None:
    """
    Safely append a segment to lib['cached_segments'].
    Ensures lib is a dict, 'cached_segments' is a list, and avoids duplicates.
    """
    if not isinstance(lib, dict):
        return
    # Ensure 'cached_segments' exists and is a list
    if "cached_segments" not in lib or not isinstance(lib["cached_segments"], list):
        lib["cached_segments"] = []
    # Avoid duplicate segment_hash
    if any(s.get("segment_hash") == seg_hash for s in lib["cached_segments"] if isinstance(s, dict)):
        return
    lib["cached_segments"].append({
        "segment_hash": seg_hash,
        "ml_label": user_label,
    })

def safe_db_call(callable_fn: Callable, *args: Any, default=None, logger: SharedLogger = None, **kwargs) -> Any:
    """
    Safely call a DB function, logging any exceptions and returning a safe default.
    Args:
        callable_fn: The function to call.
        *args, **kwargs: Arguments for the function.
        default: Value to return on error (default: None).
        logger: Optional logger instance.
    Returns:
        Result of callable_fn or default on error.
    """
    try:
        return callable_fn(*args, **kwargs)
    except Exception as e:
        func_name = getattr(callable_fn, "__name__", str(callable_fn))
        if logger:
            logger.error(f"[DB] Exception in {func_name}: {e}", exc_info=True)
        else:
            print(f"[DB] Exception in {func_name}: {e}")
        return default
    
def safe_append(lst, value, logger: SharedLogger, deduplicate=False) -> bool:
    """
    Safely append a value to a list.
    - If lst is not a list, does nothing and logs a warning.
    - Optionally deduplicates (does not append if value already exists).
    - Returns True if appended, False otherwise.
    - Logs errors if append fails.
    """
    if not isinstance(lst, list):
        if logger:
            logger.warning(f"[safe_append] Target is not a list: {type(lst)}")
        return False
    try:
        if deduplicate and value in lst:
            if logger:
                logger.info(f"[safe_append] Value already exists in list, skipping append.")
            return False
        lst.append(value)
        return True
    except Exception as e:
        if logger:
            logger.error(f"[safe_append] Error appending value: {e}")
        return False

def safe_url(page) -> str:
    """Safely get the URL from a Playwright page object."""
    try:
        url = getattr(page, "url", "")
        return str(url) if isinstance(url, str) else ""
    except Exception as e:
        logger.error(f"[safe_url] Error accessing page.url: {e}")
        return ""

def safe_update(dct, updates, logger: SharedLogger) -> None:
    """
    Safely update a dict with another dict.
    - Only updates if both are dicts.
    - Handles nested dicts recursively.
    - Logs errors if update fails.
    """
    if not isinstance(dct, dict):
        if logger:
            logger.warning(f"[safe_update] Target is not a dict: {type(dct)}")
        return
    if not isinstance(updates, dict):
        if logger:
            logger.warning(f"[safe_update] Updates is not a dict: {type(updates)}")
        return
    try:
        for k, v in updates.items():
            if isinstance(v, dict) and isinstance(dct.get(k), dict):
                safe_update(dct[k], v, logger)
            else:
                dct[k] = v
    except Exception as e:
        if logger:
            logger.error(f"[safe_update] Error updating dict: {e}")

def safe_extend(lib: dict, key: str, values: Iterable[dict]) -> None:
    """
    Safely extend a list at lib[key] with values.
    Ensures lib is a dict, lib[key] is a list, and values is an iterable (but not a string/bytes).
    Filters out None and non-dict items for safety.
    """
    import collections.abc
    if not isinstance(lib, dict):
        return
    if key not in lib or not isinstance(lib[key], list):
        lib[key] = []
    # Check values is an iterable but not a string/bytes
    if values is None or isinstance(values, (str, bytes)):
        return
    try:
        if not isinstance(values, collections.abc.Iterable):
            return
        filtered = [v for v in values if isinstance(v, dict)]
        if filtered:
            # Double-check type before extending, and use a helper if you want
            def _safe_list_extend(lst, items) -> None:
                if isinstance(lst, list) and isinstance(items, list):
                    lst.extend(items)
            try:
                _safe_list_extend(lib[key], filtered)
            except Exception as e:
                logger.error(f"[safe_extend] Failed to extend list at key '{key}': {e}")
    except Exception as e:
        logger.error(f"[safe_extend] Exception during extend: {e}")

def convert_ndarrays(obj) -> Any:
    if isinstance(obj, dict):
        return {k: convert_ndarrays(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarrays(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def _sanitize_log_filename(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_\-\.]', '_', name)

def _normalize_html_for_hash(html: str, maxlen: int = 256) -> str:
    html = re.sub(r'\s(_ngcontent-[^=]+|ng-version|ng-star-inserted|_nghost-[^=]+|_ngcontent-[^=]+|aria-checked|tabindex|style|data-[^=]+|id|class)="[^"]*"', '', html)
    html = re.sub(r'\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}', '', html)
    html = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}', '', html)
    html = re.sub(r'\d{1,2}:\d{2}(:\d{2})? ?(am|pm|AM|PM)?', '', html)
    html = re.sub(r'\s+', ' ', html.strip())
    return html[:maxlen]

def clean_cache_inplace(cache) -> int:
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

def _to_json_safe(obj) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_safe(v) for v in obj]
    return obj

def _sync_type_and_election_types(obj, fallback_types=None, fallback_type=None):
    """
    Ensures obj['type_'] and obj['election_types'] are consistent.
    Uses fallback_types and fallback_type if needed.
    """
    obj_dict = obj if isinstance(obj, dict) else {}
    etypes = obj_dict.get("election_types")
    t = obj_dict.get("type_")
    # Normalize election_types to list
    if not isinstance(etypes, list):
        etypes = [etypes] if etypes else []
    # If missing, use type_ or fallback
    if not etypes:
        if t:
            etypes = [t]
        elif fallback_types:
            etypes = fallback_types
    # If type_ missing, use first election_types or fallback
    if not t:
        t = etypes[0] if etypes else fallback_type
    # If mismatch, prefer first election_types
    if etypes and t and t != etypes[0]:
        t = etypes[0]
    obj_dict["election_types"] = etypes
    obj_dict["type_"] = t

def _keyword_in_text(text, keywords) -> bool:
    """Check if any keyword is present in the text (case-insensitive, word-boundary)."""
    text = safe_lower(text)
    for kw in keywords:
        if re.search(rf'\b{re.escape(safe_lower(kw))}\b', text):
            return True
    return False

def safe_lower(val) -> str:
    try:
        return val.lower() if isinstance(val, str) else str(val).lower()
    except Exception:
        return ""
    
def safe_encode(val, encoding="utf-8") -> bytes:
    """Safely encode a string to bytes."""
    if isinstance(val, bytes):
        return val
    if isinstance(val, str):
        return val.encode(encoding, errors="replace")
    return str(val).encode(encoding, errors="replace")
    
def safe_startswith(val, prefix) -> bool:
    try:
        return val.startswith(prefix) if isinstance(val, str) else False
    except Exception:
        return False

def safe_add(container, item) -> bool:
    """
    Safely add an item to a set-like container.
    Returns True if added, False otherwise.
    """
    def _safe_add_call(c: set, i: Any) -> bool:
        try:
            c.add(i)
            return True
        except Exception as e:
            logger.error(f"[safe_add] .add() failed: {e}")
            return False
    if container is not None and hasattr(container, "add"):
        return _safe_add_call(container, item)
    return False

def safe_predict(model: Predictable, text: str, logger: SharedLogger = None) -> Any:
    try:
        if hasattr(model, "predict") and callable(model.predict):
            return model.predict(text)
        else:
            logger.error("[safe_predict] Model has no callable 'predict' method.")
            return None
    except Exception as e:
        logger.error(f"[safe_predict] Error during predict: {e}")
        return None

def safe_split(val: Any, sep: Union[str, None] = None, maxsplit: int = -1) -> List[str]:
    """
    Safely split a string-like object.
    - Returns an empty list if val is None or not a string.
    - Handles exceptions gracefully.
    - If val is already a sequence (not a string/bytes), returns list(val).
    """
    try:
        if val is None:
            return []
        if isinstance(val, str):
            return val.split(sep, maxsplit) if sep is not None else val.split()
        if isinstance(val, bytes):
            return val.decode(errors="replace").split(sep, maxsplit) if sep is not None else val.decode(errors="replace").split()
        if isinstance(val, Sequence) and not isinstance(val, (str, bytes)):
            return list(val)
        return [str(val)]
    except Exception:
        return []

def safe_capitalize(val: object) -> str:
    """Safely capitalize a string, returns '' if not a string."""
    return val.capitalize() if isinstance(val, str) else ""

def safe_items(obj) -> Iterable:
    """
    Safely get items from a dict-like object.
    Returns an empty list if obj is not a dict or .items() fails.
    """
    def _safe_items_call(o: dict) -> Iterable:
        try:
            return o.items()
        except Exception as e:
            logger.error(f"[safe_items] .items() failed: {e}")
            return []
    if isinstance(obj, dict):
        return _safe_items_call(obj)
    # Try to convert to dict if possible
    try:
        return _safe_items_call(dict(obj))
    except Exception:
        return []

def safe_model_encode(model: SentenceTransformer, text: str | list[str], **kwargs: Any) -> Any:
    """
    Safely encode text or list of text using a model, handling edge cases.
    Returns: np.ndarray or list[np.ndarray] or None
    IDE-friendly: always returns consistent types, logs errors, and handles batch/single input.
    """
    import numpy as np

    def _normalize_text(val: Union[str, list[str]]) -> Union[str, list[str]]:
        # Accept str, bytes, or list/tuple of str/bytes
        if isinstance(val, (str, bytes)):
            return str(val)
        if isinstance(val, (list, tuple)):
            return [str(v) if not isinstance(v, str) else v for v in val]
        return str(val)

    def _safe_model_encode_call(val: Union[str, list[str]]) -> Any:
        try:
            return model.encode(val, **kwargs)
        except Exception as e:
            logger.error(f"[safe_model_encode] model.encode failed for input {repr(val)[:80]}: {e}")
            return None

    # Handle None input
    if text is None:
        logger.error("[safe_model_encode] Input text is None.")
        return None

    # Handle batch input
    if isinstance(text, (list, tuple)):
        norm_text = _normalize_text(text)
        result = _safe_model_encode_call(norm_text)
        if result is not None:
            # Ensure not string
            if isinstance(result, str):
                logger.error("[safe_model_encode] Model returned a string for batch input. Returning None.")
                return None
            return result
        # Fallback: encode each item individually
        try:
            result = [_safe_model_encode_call(_normalize_text(t)) for t in text]
            # Remove any string elements
            result = [r for r in result if not isinstance(r, str)]
            if not result:
                return None
            return result
        except Exception as e2:
            logger.error(f"[safe_model_encode] Batch encode fallback also failed: {e2}")
            return [None for _ in text]

    # Handle single string input
    norm_text = _normalize_text(text)
    result = _safe_model_encode_call(norm_text)
    if result is not None:
        # Ensure not string
        if isinstance(result, str):
            logger.error("[safe_model_encode] Model returned a string for single input. Returning None.")
            return None
        return result

    # Fallback: try as [string]
    try:
        batch_result = _safe_model_encode_call([norm_text])
        if batch_result is not None and isinstance(batch_result, (list, tuple, np.ndarray)) and len(batch_result) > 0:
            first = safe_get_first(batch_result, "batch_result", None, logger)
            # Ensure not string
            if isinstance(first, str):
                logger.error("[safe_model_encode] Model returned a string in batch fallback. Returning None.")
                return None
            return first
    except Exception as e2:
        logger.error(f"[safe_model_encode] Batch fallback failed: {e2}")

    # Extra safety: try to encode each character (rare fallback)
    try:
        logger.error(f"[safe_model_encode] All string encode attempts failed. Trying per-char fallback.")
        result = [safe_get_first(_safe_model_encode_call([c]), "char_encode", None, logger) for c in norm_text if isinstance(c, str)]
        # Remove any string elements
        result = [r for r in result if not isinstance(r, str)]
        if not result:
            return None
        return result
    except Exception as e3:
        logger.error(f"[safe_model_encode] All encode attempts failed: {e3}")
        return None

def safe_get_first(lst, label, url, logger: SharedLogger, default=None, allow_nonlist=False):
    """
    Safely get the first item from a list-like object.
    - Handles empty lists, None, non-list types, and index errors.
    - Optionally returns a default value if not found.
    - Optionally allows non-list types (returns the value itself if not a list and allow_nonlist=True).
    - Logs detailed context for debugging.
    """
    if lst is None:
        logger.error(f"[DOM_PARTS] '{label}' is None for URL: {url}")
        return default
    if isinstance(lst, list):
        if not lst:
            logger.error(f"[DOM_PARTS] Empty list for '{label}' at URL: {url}")
            return default
        try:
            return lst[0]
        except Exception as e:
            logger.error(f"[DOM_PARTS] Exception accessing first item of '{label}' for URL: {url}: {e}")
            return default
    if allow_nonlist:
        logger.warning(f"[DOM_PARTS] '{label}' is not a list for URL: {url} (type: {type(lst).__name__})")
        return lst
    logger.error(f"[DOM_PARTS] '{label}' is not a list for URL: {url} (type: {type(lst).__name__})")
    return default

def safe_is_visible(obj: Union[Locator, ElementHandle], logger: SharedLogger = None) -> bool:
    """Safely call .is_visible on a Playwright element handle or locator."""
    try:
        if hasattr(obj, "is_visible"):
            return obj.is_visible()
        return False
    except Exception as e:
        if logger: logger.error(f"[safe_is_visible] Error: {e}")
        return False

def safe_is_enabled(obj: Union[Locator, ElementHandle], logger: SharedLogger = None) -> bool:
    """Safely call .is_enabled on a Playwright element handle or locator."""
    try:
        if hasattr(obj, "is_enabled"):
            return obj.is_enabled()
        return False
    except Exception as e:
        if logger: logger.error(f"[safe_is_enabled] Error: {e}")
        return False

def safe_parse(handler: Optional[Union["ContextCoordinator", Any]], *args: Any, coordinator: Optional["ContextCoordinator"] = None, logger: SharedLogger = None, **kwargs: Any) -> Any:
    """
    Safely call handler.parse, injecting coordinator if supported.
    Handles missing parse method, non-callable, and exceptions.
    """
    try:
        if handler is None:
            if logger: logger.error("[safe_parse] Handler is None.")
            return None
        parse_method = getattr(handler, "parse", None)
        if not callable(parse_method):
            if logger: logger.error("[safe_parse] Handler has no callable 'parse' method.")
            return None
        sig = inspect.signature(parse_method)
        if logger:
            logger.debug(f"[safe_parse] Calling handler.parse with args: {[type(a) for a in args]}, kwargs: {kwargs}")
        if 'coordinator' in sig.parameters:
            return parse_method(*args, coordinator, **kwargs)
        else:
            return parse_method(*args, **kwargs)
    except Exception as e:
        if logger: logger.error(f"[safe_parse] Error calling handler.parse: {e}")
        return None

def safe_startswith(obj: Union[str, bytes], prefix: Union[str, bytes], logger: SharedLogger = None) -> bool:
    """Safely call .startswith on a string-like object."""
    try:
        if isinstance(obj, (str, bytes)):
            return obj.startswith(prefix)
        return False
    except Exception as e:
        if logger: logger.error(f"[safe_startswith] Error: {e}")
        return False

def safe_endswith(obj: Union[str, bytes], suffix: Union[str, bytes], logger: SharedLogger = None) -> bool:
    """Safely call .endswith on a string-like object."""
    try:
        if isinstance(obj, (str, bytes)):
            return obj.endswith(suffix)
        return False
    except Exception as e:
        if logger: logger.error(f"[safe_endswith] Error: {e}")
        return False

def safe_locator(page: Page, selector: str, logger: SharedLogger = None) -> Optional[Locator]:
    """Safely call .locator on a Playwright page."""
    try:
        if hasattr(page, "locator"):
            return page.locator(selector)
        return None
    except Exception as e:
        if logger: logger.error(f"[safe_locator] Error: {e}")
        return None

def safe_count(obj: Union[Locator, Collection], logger: SharedLogger = None) -> int:
    """Safely call .count() on a locator or collection."""
    try:
        if hasattr(obj, "count"):
            return obj.count()
        if hasattr(obj, "__len__"):
            return len(obj)
        return 0
    except Exception as e:
        if logger: logger.error(f"[safe_count] Error: {e}")
        return 0

def safe_evaluate(obj: Union[Locator, ElementHandle], script: str, logger: SharedLogger = None) -> Any:
    """Safely call .evaluate on a Playwright element handle."""
    try:
        if hasattr(obj, "evaluate"):
            return obj.evaluate(script)
        return None
    except Exception as e:
        if logger: logger.error(f"[safe_evaluate] Error: {e}")
        return None

def safe_is_visible(obj: Union[Locator, ElementHandle], logger: SharedLogger = None) -> bool:
    """Safely call .is_visible on a Playwright element handle."""
    try:
        if hasattr(obj, "is_visible"):
            return obj.is_visible()
        return False
    except Exception as e:
        if logger: logger.error(f"[safe_is_visible] Error: {e}")
        return False

def safe_is_enabled(obj: Union[Locator, ElementHandle], logger: SharedLogger = None) -> bool:
    """Safely call .is_enabled on a Playwright element handle."""
    try:
        if hasattr(obj, "is_enabled"):
            return obj.is_enabled()
        return False
    except Exception as e:
        if logger: logger.error(f"[safe_is_enabled] Error: {e}")
        return False

def safe_click(obj: Union[Locator, ElementHandle], logger: SharedLogger = None) -> bool:
    """Safely call .click on a Playwright element handle."""
    try:
        if hasattr(obj, "click"):
            obj.click()
            return True
        return False
    except Exception as e:
        if logger: logger.error(f"[safe_click] Error: {e}")
        return False

def safe_wait_for_timeout(page: Page, ms: int, logger: SharedLogger = None) -> bool:
    """Safely call .wait_for_timeout on a Playwright page."""
    try:
        if hasattr(page, "wait_for_timeout"):
            page.wait_for_timeout(ms)
            return True
        return False
    except Exception as e:
        if logger: logger.error(f"[safe_wait_for_timeout] Error: {e}")
        return False

def safe_get_attribute(obj: Union[Locator, ElementHandle], attr: str, logger: SharedLogger = None) -> Optional[str]:
    """Safely call .get_attribute on a Playwright element handle."""
    try:
        if hasattr(obj, "get_attribute"):
            return obj.get_attribute(attr)
        return None
    except Exception as e:
        if logger: logger.error(f"[safe_get_attribute] Error: {e}")
        return None

def safe_inner_text(obj: Union[Locator, ElementHandle], logger: SharedLogger = None) -> str:
    """Safely call .inner_text on a Playwright element handle."""
    try:
        if hasattr(obj, "inner_text"):
            return obj.inner_text()
        return ""
    except Exception as e:
        if logger: logger.error(f"[safe_inner_text] Error: {e}")
        return ""

def safe_nth(obj: Union[Locator, ElementHandle], index: int, logger: SharedLogger = None) -> Optional[Union[Locator, ElementHandle]]:
    """Safely call .nth on a Playwright locator."""
    try:
        if hasattr(obj, "nth"):
            return obj.nth(index)
        # Fallback for lists
        if isinstance(obj, (list, tuple)) and 0 <= index < len(obj):
            return obj[index]
        return None
    except Exception as e:
        if logger: logger.error(f"[safe_nth] Error: {e}")
        return None

def safe_isupper(obj, logger: SharedLogger = None):
    """Safely call .isupper() on a string-like object."""
    try:
        if isinstance(obj, str):
            return obj.isupper()
        return False
    except Exception as e:
        if logger: logger.error(f"[safe_isupper] Error: {e}")
        return False

def resolve_county_alias(county_name: str, state: Optional[str] = None) -> str:
    """
    Resolve a county name to its canonical form using known counties and aliases.
    Optionally, provide a state for more accurate mapping.
    """
    county_norm = normalize_county_name(county_name)
    # If state is provided, check only that state's counties
    if state:
        state_norm = safe_replace(safe_strip(safe_lower(state)), " ", "_")
        counties = KNOWN_STATE_TO_COUNTY_MAP.get(state_norm, [])
        if county_norm in counties:
            return county_norm
        # Fuzzy match if not found
        import difflib
        matches = difflib.get_close_matches(county_norm, counties, n=1, cutoff=0.8)
        if matches:
            return safe_get_first(matches, "county_match", None, logger)
    else:
        # Search all counties
        for counties in KNOWN_STATE_TO_COUNTY_MAP.values():
            if county_norm in counties:
                return county_norm
        # Fuzzy match across all counties
        all_counties = [c for counties in KNOWN_STATE_TO_COUNTY_MAP.values() for c in counties]
        import difflib
        matches = difflib.get_close_matches(county_norm, all_counties, n=1, cutoff=0.8)
        if matches:
            return safe_get_first(matches, "county_match", None, logger)
    # If no match, return normalized input
    return county_norm

def normalize_county_name(name) -> Optional[str]:
    """
    Normalize county names for comparison.
    Handles embedded county names, removes 'county' suffix, underscores, dashes, and extra spaces.
    E.g. 'Miami-Dade County', 'miami_dade-county', 'ResultsMiamiDadeCounty2024' -> 'miami dade'
    """
    if not name:
        return None
    name = safe_strip(safe_lower(name))
    name = safe_replace(name, "_", " ")
    name = safe_replace(name, "-", " ")
    # Remove 'county' suffix if present
    name = re.sub(r"\s+county$", "", name)
    name = re.sub(r"\s+", " ", name)
    # Try to extract county name from within a longer string (e.g., ResultsMiamiDadeCounty2024)
    match = re.search(r'([a-z ]+?)\s*county', name)
    if match:
        name = match.group(1).strip()
    # Remove any leading/trailing non-alpha chars
    name = re.sub(r"^[^a-z]+|[^a-z]+$", "", name)
    return name

def flatten_raw_field(contest) -> dict:
    """
    Recursively flatten the 'raw' field in a contest dict so that only the base/original raw data is kept.
    """
    if not isinstance(contest, dict) or "raw" not in contest:
        return contest
    base = dict(contest)
    while isinstance(base.get("raw"), dict) and "raw" in base["raw"]:
        # Go deeper until 'raw' is not a dict
        base = base["raw"]
    # Remove any nested 'raw' from the base
    if isinstance(base, dict) and "raw" in base:
        base = {k: v for k, v in base.items() if k != "raw"}
    return base

def normalize_state_name(name) -> Optional[str]:
    """
    Normalize state names and abbreviations to snake_case full state name.
    Handles abbreviations, full names, snake_case, and embedded state names in longer strings.
    E.g. 'ny', 'NY', 'New York', 'new york', 'new_york', 'ElecResultsFL.xls' -> 'new_york' or 'florida'
    """
    if not name:
        return None
    name = safe_replace(safe_strip(safe_lower(name)), " ", "_")
    # Try abbreviation lookup first
    if name in STATE_ABBR:
        return STATE_ABBR[name]
    # Try to match snake_case full name
    for full_name in STATE_ABBR.values():
        if name == full_name:
            return full_name
    # Try to match with spaces replaced by underscores
    for full_name in STATE_ABBR.values():
        if name.replace("_", " ") == full_name.replace("_", " "):
            return full_name
    # Try to find state abbreviation or name inside a longer string (e.g., filenames)
    for abbr, full_name in STATE_ABBR.items():
        pattern = r'\b' + re.escape(abbr) + r'\b'
        if re.search(pattern, name):
            return full_name
        pattern_snake = r'\b' + re.escape(full_name) + r'\b'
        if re.search(pattern_snake, name.replace("_", " ")):
            return full_name
    # Try to match state abbreviation at end of string (e.g., ElecResultsFL.xls)
    for abbr, full_name in STATE_ABBR.items():
        if name.endswith(abbr):
            return full_name
        if name.endswith("_" + abbr):
            return full_name
    return name

def infer_state_county_from_url(url: str) -> tuple:
    """
    Robustly infer state and county from a URL using regex, mappings, and context library.
    Returns (state, county) or (None, None) if not found.
    """
    url = safe_lower(url)
    url_norm = safe_replace(safe_replace(url, "-", "_"), " ", "_")
    state_map = STATE_MODULE_MAP
    county_map = KNOWN_STATE_TO_COUNTY_MAP
    IGNORED_TLDS = {
        "com", "org", "net", "gov", "edu", "co", "us", "info", "biz", "io", "me", "ca", "uk", "de", "fr", "jp"
    }
    state = None
    county = None

    # Try all state abbreviations and names (robust patterns)
    for abbr, name in STATE_ABBR.items():
        abbr_pattern = rf"/{abbr}(/|_|-|$)"
        name_repl = name.replace(' ', '[_\\-_]?')
        name_pattern = rf"/{name_repl}(/|_|-|$)"
        if re.search(abbr_pattern, url_norm) or re.search(name_pattern, url_norm):
            state = name
            break

    # Try mapping from context library
    if not state and state_map:
        for key in state_map:
            key_repl = key.replace(' ', '[_\\-_]?')
            key_pattern = rf"/{key_repl}(/|_|-|$)"
            mapped_repl = state_map[key].replace(' ', '[_\\-_]?')
            mapped_pattern = rf"/{mapped_repl}(/|_|-|$)"
            if re.search(key_pattern, url_norm) or re.search(mapped_pattern, url_norm):
                state = key
                break

    # Fuzzy match as last resort, but skip TLDs and common suffixes
    if not state:
        all_states = set(list(STATE_ABBR.values()) + list(state_map.keys()) + list(STATE_ABBR.keys()))
        url_parts = re.split(r'[/_.\-]', url_norm)
        url_parts = [part for part in url_parts if part and part not in IGNORED_TLDS]
        for part in url_parts:
            matches = difflib.get_close_matches(part, all_states, n=1, cutoff=0.8)
            if matches:
                match = safe_get_first(matches, "state_match", None, logger)
                # If match is an abbreviation, convert to full name
                state = STATE_ABBR.get(match, match)
                break

    # --- 2. Try to match county (only if state is found) ---
    if state:
        state_norm = normalize_state_name(state)
        if state_norm not in county_map:
            logger.warning(f"State '{state_norm}' not found in county map")
        counties = county_map.get(state_norm, [])
        counties_norm = [normalize_county_name(c) for c in counties]
        # Try to match "-county" or "_county" in URL
        county_match = re.search(r'/([a-z0-9_\-]+)[-_]?county', url_norm)
        if county_match:
            county_candidate = normalize_county_name(county_match.group(1))
            # Exact or fuzzy match
            if county_candidate in counties_norm:
                county = counties[counties_norm.index(county_candidate)]
            else:
                matches = difflib.get_close_matches(county_candidate, counties_norm, n=1, cutoff=0.7)
                if matches:
                    match = safe_get_first(matches, "county_match", None, logger)
                    if match in counties_norm:
                        county = counties[counties_norm.index(match)]
        # Try to match county names directly in URL
        if not county:
            for i, c_norm in enumerate(counties_norm):
                if c_norm and c_norm in url_norm:
                    county = counties[i]
                    break

    # Normalize before returning
    if state:
        state = normalize_state_name(state)
    if county:
        county = normalize_county_name(county)

    return state, county

def get_county_precincts(county_name) -> Optional[list]:
    county_norm = normalize_county_name(county_name)
    return KNOWN_COUNTY_TO_PRECINCTS_MAP.get(county_norm)

def get_state_counties(state_name) -> Optional[list]:
    state_norm = normalize_state_name(state_name)
    return KNOWN_STATE_TO_COUNTY_MAP.get(state_norm)

def scan_environment() -> dict:
    return {
        "os": platform.system(),
        "os_version": platform.version(),
        "python_version": platform.python_version(),
        "cwd": os.getcwd()
    }

def get_title_embedding_features(contests, model_name="all-MiniLM-L6-v2") -> Any:
    from ..utils.model_registry import ModelRegistry
    model = ModelRegistry.get_sentence_transformer(model_name)
    titles = []
    for c in contests:
        c_dict = c if isinstance(c, dict) else {}
        titles.append(c_dict.get("title", ""))
    return model.encode(titles, show_progress_bar=False)

def show_progress_bar(task_desc, total, update_iter) -> Generator[Any, Any, Any]:
    """
    Show a progress bar for any iterable, compatible with CLI and webapp (SocketIO) modes.
    Yields each item from update_iter.
    """
    with logger.progress_bar(task_desc, total=total) as update_progress:
        for idx, n in enumerate(update_iter):
            update_progress(idx + 1)
            yield n

def coordinator_feedback(domain, scrolls, step, incomplete=False) -> None:
    logger.info(f"[COORDINATOR] Scroll pattern for {domain}: {scrolls} scrolls, step {step}, incomplete={incomplete}")

def normalize_text(text) -> str:
    return re.sub(r"\s+", " ", safe_strip(safe_lower(text)))

def match_any(label, keywords) -> bool:
    label = normalize_text(label)
    return any(safe_lower(k) in label for k in keywords)

def build_csv_headers(rows) -> list[str]:
    headers = set()
    for row in rows:
        for k, _ in safe_items(row):
            headers.add(k)
    return sorted(headers)

def autoscroll_until_stable(
    page,
    max_stable_frames=5,
    step=8000,
    delay_ms=200,
    max_total_time=10000,
    wait_for_selector=None,
    domain=None,
    coordinator_feedback=None,
) -> bool:
    """
    Continuously scrolls a Playwright page until its scroll height and visible content stabilize
    for at least 5 consecutive measurements, or until max_total_time is reached.
    Optionally waits for a selector to appear.
    Shows a dynamic progress bar using rich or emits progress via SocketIO in webapp mode.
    Does NOT use or save any cached scroll pattern.
    """

    start_time = time.time()
    safe_evaluate(page, "window.scrollTo(0, 0)", logger)
    safe_wait_for_timeout(page, delay_ms, logger)

    stable = 0
    last_heights = []
    last_texts = []
    scroll_attempts = 0
    max_scrolls = max_total_time // delay_ms
    url_str = safe_url(page)
    domain = domain or (
        safe_get_first(url_str.split("/"), "domain_split", None, logger, default="")
        if not ("://" in url_str) else
        safe_get_first(url_str.split("/"), "domain_split", None, logger, default="")
    )
    if "://" in url_str and len(url_str.split("/")) > 2:
        domain = safe_get_first(url_str.split("/"), "domain_split", None, logger, default="", allow_nonlist=True)
        if isinstance(domain, list) and len(domain) > 2:
            domain = domain[2]

    def get_main_text() -> str:
        try:
            main_div = safe_locator(page, "main, .main-content, #main-content, body", logger)
            if main_div:
                return safe_inner_text(main_div, logger)
            else:
                return safe_inner_text(page, logger)
        except Exception:
            return ""

    with logger.progress_bar("[cyan]Scrolling page...", total=max_scrolls) as update_progress:
        while stable < max_stable_frames and scroll_attempts < max_scrolls:
            current_height = safe_evaluate(page, "document.body.scrollHeight", logger)
            current_text = get_main_text()
            last_heights.append(current_height)
            last_texts.append(current_text)
            if len(last_heights) > max_stable_frames:
                last_heights.pop(0)
                last_texts.pop(0)
            # Check if the last N heights and texts are all the same
            if (
                len(last_heights) == max_stable_frames
                and all(h == safe_get_first(last_heights, "last_heights", None, logger) for h in last_heights)
                and all(t == safe_get_first(last_texts, "last_texts", None, logger) for t in last_texts)
            ):
                stable += 1
            else:
                stable = 0
            safe_evaluate(page, f"window.scrollBy(0, {step})", logger)
            safe_wait_for_timeout(page, delay_ms, logger)
            scroll_attempts += 1
            update_progress(scroll_attempts)
            if wait_for_selector and safe_locator(page, wait_for_selector, logger):
                logger.info(f"[SCROLL] Selector '{wait_for_selector}' found. Stopping scroll.")
                break
            elapsed = (time.time() - start_time) * 1000
            if elapsed > max_total_time * 0.8 and scroll_attempts % 10 == 0:
                console.print("[bold yellow]Scrolling is taking longer than expected. Continue waiting? (y/N)[/bold yellow]")
                resp = prompt.prompt_input("Continue scrolling? (y/N): ").strip().lower()
                if resp != "y":
                    logger and logger.warning("[SCROLL] User aborted scrolling.")
                    break
        # Ensure progress bar is completed
        update_progress(max_scrolls)

    if stable >= max_stable_frames:
        logger and logger.info("[SCROLL] Completed scrolling until page height/content stabilized.")
        if coordinator_feedback:
            coordinator_feedback(domain, scroll_attempts, step)
        return True
    else:
        logger and logger.warning("[SCROLL] Max scroll time/attempts exceeded. Page may not be fully loaded.")
        if coordinator_feedback:
            coordinator_feedback(domain, scroll_attempts, step, incomplete=True)
        return False

def scan_buttons_with_progress(buttons, scan_callback=None) -> None:
    """
    Scan a list of buttons with a single-line progress bar or emits progress via SocketIO in webapp mode.
    Optionally, provide a scan_callback(button, idx) for custom logic.
    """
    total = len(buttons)
    with logger.progress_bar("Scanning buttons...", total=total) as update_progress:
        for idx, btn in enumerate(buttons):
            label = ""
            try:
                label = safe_inner_text(btn, logger)[:60]
            except Exception:
                label = str(btn)[:60]
            update_progress(idx + 1, extra={"label": label})
            if scan_callback:
                scan_callback(btn, idx)

def keyphrase_match(label, keyphrase, min_words=2, fuzzy_cutoff=0.8) -> bool:
    """
    Returns True if the label matches the keyphrase as a whole (regex or fuzzy),
    or if at least min_words from the keyphrase are present in the label.
    """
    label_norm = safe_strip(safe_lower(label))
    keyphrase_norm = safe_strip(safe_lower(keyphrase))
    # 1. Try full phrase regex (allowing whitespace, punctuation, : or \n at end)
    pattern = re.sub(r"\s+", r"\\s+", re.escape(keyphrase_norm)) + r"[\s:]*$"
    if re.search(pattern, label_norm):
        return True
    # 2. Try fuzzy full phrase
    if difflib.SequenceMatcher(None, label_norm, keyphrase_norm).ratio() >= fuzzy_cutoff:
        return True
    # 3. Require at least min_words from keyphrase to be present
    words = [w for w in re.split(r"\W+", keyphrase_norm) if w]
    matches = sum(1 for w in words if w in label_norm)
    if len(words) >= min_words and matches >= min_words:
        return True
    return False

def infer_contest_fields(
    contest: dict,
    context_library: dict,
    db_service=None,
    embedding_model=None,
    log=None
) -> tuple:
    """
    Infer missing fields for a contest using (in order):
    1. Direct field on contest
    2. Context library lookup (by normalized title)
    3. Database lookup (by normalized title, if db_service provided)
    4. ML/NLP model (if available)
    5. Fallback: extract_year_and_type (only if all else fails)
    Logs the inference path if log is provided.
    """
    from ..utils.db_utils import normalize_label
    if db_service is None:
        try:
            from ..services.election_data_services import ElectionDataService
            db_service = ElectionDataService
        except ImportError:
            db_service = None

    contest_dict = contest if isinstance(contest, dict) else {}
    title = contest_dict.get("title") or contest_dict.get("label") or ""
    norm_title = normalize_label(title)
    year = contest_dict.get("year")
    type_ = contest_dict.get("type_")
    state = contest_dict.get("state")
    county = contest_dict.get("county")
    inference_path = []

    # 1. Context library lookup
    context_contests = context_library.get("contests", []) if isinstance(context_library, dict) else []
    for c in context_contests:
        c_dict = c if isinstance(c, dict) else {}
        c_title = c_dict.get("title") or c_dict.get("label") or ""
        if normalize_label(c_title) == norm_title:
            if not year and c_dict.get("year"):
                year = c_dict.get("year")
                inference_path.append("year:context_library")
            if not type_ and c_dict.get("type_"):
                type_ = c_dict.get("type_")
                inference_path.append("type_:context_library")
            if not state and c_dict.get("state"):
                state = c_dict.get("state")
                inference_path.append("state:context_library")
            if not county and c_dict.get("county"):
                county = c_dict.get("county")
                inference_path.append("county:context_library")
            if year and type_ and state and county:
                break

    # 2. Database lookup (if db_service provided)
    if db_service and (not year or not type_ or not state or not county):
        try:
            db_contests = db_service.get_contests_by_advanced_filter(
                filters={"title": title}, limit=1
            )
            db_c = None
            if db_contests and isinstance(db_contests, list):
                db_c = db_contests[0] if db_contests and isinstance(db_contests[0], dict) else {}
            if db_c:
                if not year and db_c.get("year"):
                    year = db_c.get("year")
                    inference_path.append("year:db")
                if not type_ and db_c.get("type_"):
                    type_ = db_c.get("type_")
                    inference_path.append("type_:db")
                if not state and db_c.get("state"):
                    state = db_c.get("state")
                    inference_path.append("state:db")
                if not county and db_c.get("county"):
                    county = db_c.get("county")
                    inference_path.append("county:db")
        except Exception as e:
            if log is not None:
                safe_append(log, f"[infer_contest_fields] DB lookup failed: {e}", logger)

    # 3. ML/NLP model (if available)
    if embedding_model and (not year or not type_ or not state or not county):
        try:
            pred = safe_predict(embedding_model, title, logger)
            pred_dict = pred if isinstance(pred, dict) else {}
            def get_pred_value(field):
                field_dict = pred_dict.get(field, {})
                if isinstance(field_dict, dict):
                    return field_dict.get("value")
                return None
            if not year and get_pred_value("year"):
                year = get_pred_value("year")
                inference_path.append("year:ml")
            if not type_ and get_pred_value("type_"):
                type_ = get_pred_value("type_")
                inference_path.append("type_:ml")
            if not state and get_pred_value("state"):
                state = get_pred_value("state")
                inference_path.append("state:ml")
            if not county and get_pred_value("county"):
                county = get_pred_value("county")
                inference_path.append("county:ml")
        except Exception as e:
            if log is not None:
                safe_append(log, f"[infer_contest_fields] ML/NLP model failed: {e}", logger)

    # 4. Fallback: extract_year_and_type (only if still missing)
    if (not year or not type_) and title:
        try:
            from ..utils.html_scanner import extract_year_and_type
            y, t, _, _ = extract_year_and_type(title, url=None)
            if not year and y:
                year = y
                inference_path.append("year:extract_year_and_type")
            if not type_ and t:
                type_ = t
                inference_path.append("type_:extract_year_and_type")
        except Exception as e:
            if log is not None:
                safe_append(log, f"[infer_contest_fields] extract_year_and_type failed: {e}", logger)

    # Log the inference path if requested
    if log is not None:
        safe_append(log, f"[infer_contest_fields] {title} → {inference_path}", logger)

    # --- PATCH: Ensure type_ and election_types are synced ---
    contest_dict["year"] = year
    contest_dict["type_"] = type_
    contest_dict["state"] = state
    contest_dict["county"] = county
    _sync_type_and_election_types(contest_dict)

    return year, type_, state, county