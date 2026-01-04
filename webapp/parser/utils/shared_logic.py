from __future__ import annotations

import copy

# webapp/parser/utils/shared_logic.py
# -----------------------------------------------------------------------------------
# Common parsing utilities for context-integrated pipeline
# -----------------------------------------------------------------------------------
import difflib
import gc
import inspect
import os
import platform
import re
import shutil
import textwrap
import time
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    Type,
    TypedDict,
    TypeVar,
    Union,
    runtime_checkable,
)
from urllib.parse import ParseResult, SplitResult

import numpy as np
import orjson
from flask import request, session

# Optional Sentencetransformers dependency (graceful fallback when missing)
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:  # pragma: no cover - dependency optional in some environments
    SentenceTransformer = None  # type: ignore
from sqlalchemy.engine import ScalarResult
from sqlalchemy.orm import Query, Session

from ..Context_Integration.Context_Library.constants import (
    KNOWN_COUNTY_TO_PRECINCTS_MAP,
    KNOWN_STATE_TO_COUNTY_MAP,
    STATE_ABBR,
    STATE_MODULE_MAP,
    build_camelot_row_filter,
    normalize_party_label,
)
from ..utils.logger_singleton import logger

if TYPE_CHECKING:
    from ..Context_Integration.context_coordinator import ContextCoordinator

assert set(STATE_MODULE_MAP.keys()) == set(KNOWN_STATE_TO_COUNTY_MAP.keys()), \
    "STATE_MODULE_MAP and KNOWN_STATE_TO_COUNTY_MAP keys are out of sync!"

class ExtractPlugin(Protocol):
    def extract(self, page: Any, extraction_context: Any) -> List[Any]: ...

class Saveable(Protocol):
    def save(self, path: str) -> Any: ...

class GCModule(Protocol):
    def collect(self) -> Any: ...

class ShutilModule(Protocol):
    def rmtree(self, path: str, ignore_errors: bool = ...) -> Any: ...
    def move(self, src: str, dst: str) -> Any: ...

class TimeModule(Protocol):
    def sleep(self, seconds: float) -> Any: ...

@runtime_checkable
class HasItem(Protocol):
    def item(self) -> float:
        """Returns a single item, typically a float."""
        ...

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

def safe_filename(
    name: str,
    max_length: int = 255,
    allow_unicode: bool = False,
    reserved_names: set = None,
    default: str = "file",
    strict_mode: bool = False,
) -> str:
    """
    Robustly sanitize a string for use as a safe filename.
    - Removes or replaces unsafe characters.
    - Optionally restricts to ASCII.
    - Handles reserved device names (Windows).
    - Trims to max_length.
    - Returns a default if the result is empty.
    - strict_mode: tighten allowed chars (no spaces, no path separators or traversal tokens).
    """
    if not isinstance(name, str):
        name = str(name) if name is not None else default

    # Strip whitespace and null bytes early
    name = (name or "").strip().replace("\x00", "")

    # Optionally restrict to ASCII
    if not allow_unicode:
        name = name.encode("ascii", "ignore").decode("ascii")

    # Remove path separators entirely to avoid accidental traversal joins
    name = name.replace("/", "").replace("\\", "")

    # Normalize traversal patterns (".." -> "_")
    name = re.sub(r"\.{2,}", "_", name)

    # Replace unsafe characters with underscores (keep dots/underscores/hyphens/spaces temporarily)
    name = re.sub(r"[^A-Za-z0-9._\-\s]", "_", name)

    # Collapse whitespace to underscores
    name = name.replace(" ", "_")

    # Collapse repeated underscores and dots separately
    name = re.sub(r"_+", "_", name)
    name = re.sub(r"\.{2,}", ".", name)

    # Trim leading/trailing punctuation
    name = name.strip("._")

    # Re-split into base/ext to decide whether to keep the dot
    base, ext = (name.rsplit(".", 1) + [""])[:2] if "." in name else (name, "")

    # If the base ends with an underscore, treat the separator as an underscore to avoid "file_.ext" edge cases
    if ext and base and base.endswith("_"):
        base = base.rstrip("_") or default
        name = f"{base}_{ext}"
    elif ext and base:
        name = f"{base}.{ext}"
    else:
        name = base or ext

    # Strict mode: tighten remaining punctuation but keep single dots (extensions)
    if strict_mode:
        name = re.sub(r"_+", "_", name)
        name = re.sub(r"\.{2,}", ".", name)
        name = name.strip("._")

    # Handle reserved device names (Windows)
    reserved = reserved_names or {
        "CON", "PRN", "AUX", "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
    if name.upper() in reserved:
        name = f"_{name}_"

    # Prevent empty filename
    if not name:
        name = default

    # Truncate to max_length, preserving extension if possible
    if len(name) > max_length:
        p = Path(name)
        ext = p.suffix
        base = p.stem[: max_length - len(ext)]
        name = base + ext

    return name

def is_path_safe(path: Union[str, Path], allowed_bases: Union[str, Path, List[Union[str, Path]]] | None = None) -> bool:
    """Return True if resolved path is within any allowed base directories."""
    try:
        target = Path(path).resolve()
    except Exception:
        return False
    bases: List[Path] = []
    if allowed_bases is None:
        return True
    if isinstance(allowed_bases, (str, Path)):
        bases = [Path(allowed_bases)]
    else:
        try:
            bases = [Path(b) for b in allowed_bases]
        except Exception:
            return False
    for base in bases:
        try:
            if target.is_relative_to(base.resolve()):
                return True
        except AttributeError:
            # Python <3.9 fallback
            base_res = base.resolve()
            try:
                if os.path.commonpath([str(base_res)]) == os.path.commonpath([str(base_res), str(target)]):
                    return True
            except Exception:
                continue
        except Exception:
            continue
    return False


def safe_resolve_path(
    path: Union[str, Path],
    base: Union[str, Path, None] = None,
    *,
    must_exist: bool = False,
    create: bool = False,
) -> Path:
    """Resolve a path while enforcing base confinement and optional existence checks."""
    base_path = Path(base).expanduser().resolve() if base is not None else None
    raw_path = Path(path)
    target = raw_path if raw_path.is_absolute() else (base_path or Path.cwd()).joinpath(raw_path)
    try:
        resolved = target.resolve(strict=must_exist)
    except FileNotFoundError:
        raise ValueError(f"Path does not exist: {target}")
    except RuntimeError:
        # In rare cases (e.g., cyclical symlinks), fall back to non-strict resolution
        resolved = target.resolve(strict=False)

    if base_path and not is_path_safe(resolved, [base_path]):
        raise ValueError("Path traversal detected")

    if must_exist and not resolved.exists():
        raise ValueError(f"Path does not exist: {resolved}")

    if create:
        resolved.mkdir(parents=True, exist_ok=True)

    return resolved


def safe_join_path(base: Union[str, Path], *paths: str) -> Path:
    """Join paths under a base directory with sanitization and traversal protection."""
    base_path = Path(base).expanduser().resolve()

    sanitized_parts: List[str] = []
    for raw in paths:
        if raw is None:
            continue
        text = str(raw)
        for piece in re.split(r"[\\/]+", text):
            if not piece:
                continue
            cleaned = safe_filename(piece, strict_mode=True)
            if cleaned:
                sanitized_parts.append(cleaned)

    candidate = base_path.joinpath(*sanitized_parts)
    try:
        resolved = candidate.resolve(strict=False)
    except Exception:
        resolved = candidate

    if not is_path_safe(resolved, [base_path]):
        raise ValueError("Path traversal detected")

    return resolved


def validate_directory_path(path: Union[str, Path], create_if_missing: bool = False) -> Path:
    """Ensure a directory exists (optionally creating it) and is not a file."""
    candidate = Path(path).expanduser().resolve(strict=False)

    if candidate.exists() and not candidate.is_dir():
        raise ValueError(f"Path is not a directory: {candidate}")

    if not candidate.exists():
        if not create_if_missing:
            raise ValueError(f"Path does not exist: {candidate}")
        candidate.mkdir(parents=True, exist_ok=True)

    return candidate

T = TypeVar("T")

def safe_slug(text: str, max_len: int = 100) -> str:
    """
    Make a filesystem-friendly slug:
    - Keep alnum, space, underscore, hyphen; replace others with '_'
    - Collapse repeated underscores/spaces; convert spaces to underscores
    - Trim length to max_len
    """
    if not isinstance(text, str):
        return ""
    stem = os.path.splitext(text)[0]
    s = "".join(c if c.isalnum() or c in " _-" else "_" for c in stem)
    s = re.sub(r"[ _]+", " ", s).strip()
    s = s.replace(" ", "_")
    s = re.sub(r"_+", "_", s)
    return s[:max_len] or "untitled"

def safe_query(session: Session, model: Type[T]) -> Optional[Query]:
    """
    Safely create a SQLAlchemy query for a model.
    Returns the query or None if an error occurs.
    """
    try:
        return session.query(model)
    except Exception as e:
        logger.warning(f"[safe_query] session.query({model}) failed: {e}")
        return None

def safe_key(val) -> str:
    """
    Safely convert a value to a string key, handling None and exceptions.
    """
    try:
        if val is None:
            return ""
        return str(val)
    except Exception:
        return ""
    
def _filter_valid_kwargs(model: Type[Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter kwargs to only include valid column names for the model.
    Uses safe_key for robust key extraction and safe access to mapper.
    """
    try:
        # Defensive: safely get mapper and column_attrs
        mapper = getattr(inspect(model), "mapper", None)
        if mapper is None:
            logger.warning(f"[safe_filter_by] No mapper found for model {model}")
            return {}
        column_attrs = list(getattr(mapper, "column_attrs", []))
        valid_columns = set(safe_key(getattr(c, "key", None)) for c in column_attrs if hasattr(c, "key"))
        return {k: v for k, v in kwargs.items() if k in valid_columns}
    except Exception as e:
        logger.warning(f"[safe_filter_by] Could not inspect model {model}: {e}")
        return {}

def safe_filter_by(query: Optional[Query], model: Type, **kwargs) -> Optional[Query]:
    """
    Safely apply filter_by to a SQLAlchemy query, only allowing valid columns.
    Returns the filtered query or the original query if an error occurs.
    """
    if query is None:
        return None
    safe_kwargs = _filter_valid_kwargs(model, kwargs)
    try:
        return query.filter_by(**safe_kwargs)
    except Exception as e:
        logger.warning(f"[safe_filter_by] filter_by failed: {e}")
        return query

def safe_first(query: Optional[Query]) -> Optional[Any]:
    """
    Safely call .first() on a SQLAlchemy query.
    Returns the first result or None if an error occurs.
    """
    if query is None:
        return None
    try:
        return query.first()
    except Exception as e:
        logger.warning(f"[safe_first] query.first() failed: {e}")
        return None

def get_or_create(
    session: Session,
    model: Type[T],
    defaults: Optional[dict] = None,
    **kwargs
) -> T:
    """
    Safely get or create a SQLAlchemy model instance.
    Uses safe_query, safe_filter_by, safe_first, safe_add, and safe_commit.
    Only allows valid model columns in filter_by to prevent SQL injection.
    """
    query = safe_query(session, model)
    query = safe_filter_by(query, model, **kwargs)
    instance = safe_first(query)
    if instance:
        return instance
    params = _filter_valid_kwargs(model, kwargs)
    params.update(defaults or {})
    instance = model(**params)
    safe_add(session, instance)
    safe_commit(session)
    return instance

def safe_translate(val: str, table) -> str:
    """
    Safely call .translate on a string-like object.
    Returns the translated string, or the original value if not a string or error occurs.
    """
    try:
        if isinstance(val, str):
            return val.translate(table)
        return str(val).translate(table)
    except Exception:
        return str(val)

def safe_scheme(parsed: Union[ParseResult, SplitResult, object]) -> str:
    try:
        if hasattr(parsed, "scheme"):
            return parsed.scheme
        return getattr(parsed, "scheme", "")
    except Exception:
        return ""

def safe_netloc(parsed: Union[ParseResult, SplitResult, object]) -> str:
    try:
        if hasattr(parsed, "netloc"):
            return parsed.netloc
        return getattr(parsed, "netloc", "")
    except Exception:
        return ""

def safe_geturl(parsed: Union[ParseResult, SplitResult]) -> str:
    try:
        if hasattr(parsed, "geturl"):
            return parsed.geturl()
        return getattr(parsed, "geturl", "")
    except Exception:
        return ""

def safe_extract(plugin: ExtractPlugin, page: Any, extraction_context: Any) -> List[Any]:
    """
    Safely call the extract method of a plugin, handling missing methods and exceptions.
    """
    try:
        if hasattr(plugin, "extract") and callable(getattr(plugin, "extract")):
            return plugin.extract(page, extraction_context)
        else:
            logger.warning(f"[PLUGIN EXTRACTION] Plugin {plugin} has no callable 'extract' method.")
            return []
    except Exception as e:
        logger.error(f"[PLUGIN EXTRACTION] Error in plugin {plugin}: {e}")
        return []

def safe_isalpha(val: Any) -> bool:
    """
    Safely check if val is a string and .isalpha() returns True.
    Returns False for non-strings or on error.
    """
    try:
        return isinstance(val, str) and val.isalpha()
    except Exception:
        return False

def safe_pop(dct: dict, key: str, default=None) -> Any:
    try:
        if isinstance(dct, dict):
            return dct.pop(key, default)
    except Exception:
        pass
    return default

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

def safe_setdefault(d: dict, key, default) -> Any:
    """
    Robust setdefault: returns d[key] if present, else sets d[key]=default and returns default.
    Handles None and missing keys safely.
    """
    val = safe_get(d, key, None)
    if val is None:
        d[key] = default
        return default
    return val

def safe_tolist(val) -> List[Any]:
    """
    Safely convert val to a list.
    Handles numpy arrays, tuples, sets, and returns [val] for scalars.
    Returns an empty list for None.
    """
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

def safe_commit(session: Session) -> bool:
    try:
        session.commit()
        return True
    except Exception as e:
        logger.error(f"[safe_commit] Commit failed: {e}")
        session.rollback()
        return False

def safe_scalar_one_or_none(result: ScalarResult[T]) -> Optional[T]:
    """
    Safely call scalar_one_or_none on a SQLAlchemy ScalarResult.
    Returns the scalar value or None if not available or on error.
    """
    try:
        return result.scalar_one_or_none()
    except Exception as e:
        logger.error(f"[safe_scalar_one_or_none] Error: {e}")
        return None

def safe_model_save(
    model: Saveable,
    model_save_path: str,
    retries: int = 3,
    logger=logger,
    gc_module: GCModule = gc,
    shutil_module: ShutilModule = shutil,
    time_module: TimeModule = time
) -> bool:
    """
    Safely save a model to disk, retrying on failure and using a temp path workaround if needed.
    Returns True if save succeeded, False otherwise.
    """
    for attempt in range(1, retries + 1):
        try:
            if gc_module:
                gc_module.collect()
            model.save(model_save_path)
            if logger:
                logger.info(f"[INFO] Model saved successfully on attempt {attempt}.")
            return True
        except Exception as e:
            if logger:
                logger.warning(f"[WARN] Model save failed (attempt {attempt}): {e}")
            if time_module:
                time_module.sleep(2 * attempt)
            if gc_module:
                gc_module.collect()
    # Try saving to a temp dir and moving
    tmp_path = model_save_path + "_tmp"
    try:
        if gc_module:
            gc_module.collect()
        model.save(tmp_path)
        if shutil_module:
            shutil_module.rmtree(model_save_path, ignore_errors=True)
            shutil_module.move(tmp_path, model_save_path)
        if logger:
            logger.info("[INFO] Model saved via temp path workaround.")
        return True
    except Exception as e:
        if logger:
            logger.error(f"[ERROR] Final model save failed: {e}\nIf you see repeated save failures, close any file explorers or editors viewing the model directory.")
        return False
    
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

def safe_copy(obj: Any) -> Any:
    """
    Robustly copy an object.
    - Tries copy.deepcopy first.
    - Falls back to copy.copy.
    - Falls back to manual shallow copy for lists/dicts.
    - Returns the object itself if all else fails.
    """
    try:
        return copy.deepcopy(obj)
    except Exception:
        try:
            return copy.copy(obj)
        except Exception:
            try:
                if isinstance(obj, list):
                    return obj[:]
                if isinstance(obj, dict):
                    return dict(obj)
            except Exception:
                pass
    return obj

def safe_isalnum(val) -> bool:
    """
    Safely check if val is a string and .isalnum() returns True.
    Returns False for non-strings or on error.
    """
    try:
        return isinstance(val, str) and val.isalnum()
    except Exception:
        return False

def safe_keys(obj) -> list:
    """
    Safely get .keys() from a dict-like object.
    Returns an empty list if obj is not a dict or .keys() fails.
    """
    try:
        if isinstance(obj, dict):
            return list(obj.keys())
        # Try to convert to dict if possible
        return list(dict(obj).keys())
    except Exception:
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

def safe_set(event_like: EventLike) -> None:
    """
    Safely call .set() on an event-like object.
    """
    try:
        if hasattr(event_like, "set") and callable(event_like.set):
            event_like.set()
    except Exception as e:
        logger.error(f"[safe_set] Error calling .set(): {e}")

def safe_clear(event_like: EventLike) -> None:
    """
    Safely call .clear() on an event-like object.
    """
    try:
        if hasattr(event_like, "clear") and callable(event_like.clear):
            event_like.clear()
    except Exception as e:
        logger.error(f"[safe_clear] Error calling .clear(): {e}")

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

def safe_db_call(callable_fn: Callable, *args: Any, default=None, logger=logger, **kwargs) -> Any:
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
    
def safe_append(lst, value, logger=None, deduplicate: bool = False) -> list:
    """
    Append value to a list and return the list (never raises).
    """
    try:
        if not isinstance(lst, list):
            if logger:
                try:
                    logger.warning(f"[safe_append] Target is not a list: {type(lst)}; coercing to list.")
                except Exception:
                    pass
            lst = [] if lst is None else list(lst) if isinstance(lst, (tuple, set)) else []
        if not (deduplicate and value in lst):
            lst.append(value)
        return lst
    except Exception:
        try:
            return lst if isinstance(lst, list) else [value]
        except Exception:
            return []

def safe_update(dct, updates, logger=logger) -> None:
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

def safe_extend(lst, values, logger=None, deduplicate: bool = False) -> list:
    """
    Extend a list with iterable values and return the list (never raises).
    """
    try:
        if not isinstance(lst, list):
            if logger:
                try:
                    logger.warning(f"[safe_extend] Target is not a list: {type(lst)}; coercing to list.")
                except Exception:
                    pass
            lst = [] if lst is None else list(lst) if isinstance(lst, (tuple, set)) else []
        if values is None:
            return lst
        for v in values:
            if not (deduplicate and v in lst):
                lst.append(v)
        return lst
    except Exception:
        return lst if isinstance(lst, list) else []

def convert_ndarrays(obj) -> Any:
    if isinstance(obj, dict):
        return {k: convert_ndarrays(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarrays(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def normalize_html_for_hash(html: str, maxlen: int = 256) -> str:
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

def sync_type_and_election_types(obj, fallback_types=None, fallback_type=None):
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

def keyword_in_text(text, keywords) -> bool:
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
    
def safe_startswith(
    obj: Union[str, bytes],
    prefix: Union[str, bytes],
    logger=logger,
) -> bool:
    """Safely call .startswith on a string-like object."""
    try:
        if isinstance(obj, (str, bytes)):
            return obj.startswith(prefix)
        return False
    except Exception as exc:
        if logger:
            logger.error(f"[safe_startswith] Error: {exc}")
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

def safe_predict(model: Predictable, text: str, logger=logger) -> Any:
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

def safe_item(val: Union[HasItem, float, int], logger=logger) -> float:
    """
    Safely call .item() on a numpy scalar or similar object.
    Returns float value or 0.0 on error.
    """
    try:
        if hasattr(val, "item") and callable(val.item):
            return float(val.item())
        return float(val)
    except Exception as e:
        if logger:
            logger.error(f"[safe_item] Error calling .item(): {e}")
        return 0.0

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

def safe_similarity(model: Any, a: str, b: str, logger=logger) -> float:
    """
    Safely compute similarity between two strings using model.similarity.
    Returns a float between 0.0 and 1.0, or 0.0 on error.
    """
    if model is None:
        if logger:
            logger.debug("[safe_similarity] sentence-transformers model unavailable; returning 0.0")
        return 0.0
    try:
        sim = model.similarity(a, b)
        # Handle numpy scalars, lists, etc.
        if isinstance(sim, (float, int)):
            return float(sim)
        if hasattr(sim, "item"):
            return safe_item(sim, logger)
        if isinstance(sim, (list, tuple)) and sim:
            return float(sim[0])
        if logger:
            logger.error(f"[safe_similarity] Unexpected similarity type: {type(sim)}")
        return 0.0
    except Exception as e:
        if logger:
            logger.error(f"[safe_similarity] Exception: {e}")
        return 0.0

def safe_model_encode(model: Any, text: str, **kwargs: Any) -> Union[np.ndarray, List[np.ndarray], None]:
    """
    Safely encode text or list of text using a model, handling edge cases.
    Returns: np.ndarray or list[np.ndarray] or None
    Always returns consistent types, logs errors, and handles batch/single input.
    """
    if model is None:
        logger.debug("[safe_model_encode] sentence-transformers model unavailable; returning None")
        return None
    def _normalize_text(val):
        if isinstance(val, (str, bytes)):
            return str(val)
        if isinstance(val, (list, tuple)):
            return [str(v) if not isinstance(v, str) else v for v in val]
        return str(val)

    def _encode(val):
        try:
            result = model.encode(val, **kwargs)
            # Defensive: If result is a string, error
            if isinstance(result, str):
                logger.error(f"[safe_model_encode] Model.encode returned a string for input {repr(val)[:80]}")
                return None
            # If result is a list/tuple, filter and convert to np.ndarray
            if isinstance(result, (list, tuple)):
                arrs = [np.array(r) if not isinstance(r, np.ndarray) and not isinstance(r, str) else r
                        for r in result if not isinstance(r, str)]
                arrs = [r for r in arrs if isinstance(r, np.ndarray)]
                return arrs if arrs else None
            # If already np.ndarray
            if isinstance(result, np.ndarray):
                return result
            # Try to convert to np.ndarray
            try:
                arr = np.array(result)
                if arr.dtype.kind in {'U', 'S', 'O'}:
                    logger.error(f"[safe_model_encode] Model.encode returned non-numeric array: {arr.dtype}")
                    return None
                return arr
            except Exception:
                logger.error(f"[safe_model_encode] Could not convert result to np.ndarray: {type(result)}")
                return None
        except Exception as e:
            logger.error(f"[safe_model_encode] model.encode failed for input {repr(val)[:80]}: {e}")
            return None

    if text is None:
        logger.error("[safe_model_encode] Input text is None.")
        return None

    # Handle batch input
    if isinstance(text, (list, tuple)):
        norm_text = _normalize_text(text)
        result = _encode(norm_text)
        if result is not None:
            return result
        # Fallback: encode each item individually
        try:
            result = [_encode(_normalize_text(t)) for t in text]
            result = [r for r in result if isinstance(r, np.ndarray)]
            return result if result else None
        except Exception as e2:
            logger.error(f"[safe_model_encode] Batch encode fallback also failed: {e2}")
            return [None for _ in text]

    # Handle single string input
    norm_text = _normalize_text(text)
    result = _encode(norm_text)
    if result is not None:
        if isinstance(result, np.ndarray):
            return result
        if isinstance(result, (list, tuple)):
            arrs = [r for r in result if isinstance(r, np.ndarray)]
            return arrs if arrs else None
    # Fallback: try as [string]
    try:
        batch_result = _encode([norm_text])
        if batch_result and isinstance(batch_result, (list, tuple, np.ndarray)) and len(batch_result) > 0:
            first = safe_get_first(batch_result, "batch_result", None, logger)
            if isinstance(first, np.ndarray):
                return first
    except Exception as e2:
        logger.error(f"[safe_model_encode] Batch fallback failed: {e2}")

    # Extra safety: try to encode each character (rare fallback)
    try:
        logger.error("[safe_model_encode] All string encode attempts failed. Trying per-char fallback.")
        result = [_encode([c]) for c in norm_text if isinstance(c, str)]
        result = [r for r in result if isinstance(r, np.ndarray)]
        return result if result else None
    except Exception as e3:
        logger.error(f"[safe_model_encode] All encode attempts failed: {e3}")
        return None

def safe_get_first(lst, label, url, logger=logger, default=None, allow_nonlist=False):
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

def safe_parse(handler: Optional[Union["ContextCoordinator", Any]], *args: Any, coordinator: Optional["ContextCoordinator"] = None, logger=logger, **kwargs: Any) -> Any:
    """
    Safely call handler.parse, injecting coordinator if supported.
    Handles missing parse method, non-callable, and exceptions.
    """
    try:
        if handler is None:
            if logger:
                logger.error("[safe_parse] Handler is None.")
            return None
        parse_method = getattr(handler, "parse", None)
        if not callable(parse_method):
            if logger:
                logger.error("[safe_parse] Handler has no callable 'parse' method.")
            return None
        sig = inspect.signature(parse_method)
        param_names = list(sig.parameters.keys())

        # Build positional and keyword arguments safely
        call_args = list(args)
        call_kwargs = dict(kwargs)

        # Only add coordinator if not already in args
        if 'coordinator' in param_names:
            # Find the index of 'coordinator' in the signature
            coord_idx = param_names.index('coordinator')
            # If not enough args to fill coordinator, add it positionally
            if len(call_args) <= coord_idx:
                call_args.insert(coord_idx, coordinator)
            # Remove from kwargs if present
            call_kwargs.pop('coordinator', None)

        # Remove any kwargs that are already filled by positional args
        for i, name in enumerate(param_names[:len(call_args)]):
            if name in call_kwargs:
                call_kwargs.pop(name)

        if logger:
            logger.debug(f"[safe_parse] Calling handler.parse with args: {[type(a) for a in call_args]}, kwargs: {call_kwargs}")

        return parse_method(*call_args, **call_kwargs)
    except Exception as e:
        if logger:
            logger.error(f"[safe_parse] Error calling handler.parse: {e}")
        return None

def safe_endswith(obj: Union[str, bytes], suffix: Union[str, bytes], logger=logger) -> bool:
    """Safely call .endswith on a string-like object."""
    try:
        if isinstance(obj, (str, bytes)):
            return obj.endswith(suffix)
        return False
    except Exception as e:
        if logger:
            logger.error(f"[safe_endswith] Error: {e}")
        return False

def safe_isupper(obj: Union[str, bytes], logger=logger) -> bool:
    """Safely call .isupper() on a string-like object."""
    try:
        if isinstance(obj, str):
            return obj.isupper()
        return False
    except Exception as e:
        if logger:
            logger.error(f"[safe_isupper] Error: {e}")
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
        matches = difflib.get_close_matches(county_norm, all_counties, n=1, cutoff=0.8)
        if matches:
            return safe_get_first(matches, "county_match", None, logger)
    # If no match, return normalized input
    return county_norm

def safe_sid() -> str:
    """
    Returns a valid SocketIO session ID for the current request/session.
    Tries request.sid first (SocketIO context), then flask session storage.
    Raises RuntimeError if none found.
    """
    # Prefer request.sid (present in SocketIO event context)
    try:
        sid = getattr(request, 'sid', None)
        if isinstance(sid, str) and sid:
            return sid
    except Exception:
        pass
    # Fallback: flask session (may not be set)
    try:
        sid = session.get('sid')
        if isinstance(sid, str) and sid:
            return sid
    except Exception:
        pass
    raise RuntimeError("No valid session ID found for SocketIO connection.")

def safe_rsplit(val, sep=None, maxsplit=-1) -> list[str]:
    """
    Safely call .rsplit on a string-like object.
    Returns a list, or [str(val)] if not a string or error occurs.
    """
    try:
        if isinstance(val, str):
            return val.rsplit(sep, maxsplit)
        if isinstance(val, bytes):
            return val.decode(errors="replace").rsplit(sep, maxsplit)
        return [str(val)]
    except Exception:
        return [str(val)]

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

def resolve_state_county_from_context(context: Optional[dict]) -> tuple[Optional[str], Optional[str]]:
    """Resolve normalized state and county from a context dict.

    Checks explicit fields first, then attempts to infer from common URL keys.
    Returns (state, county) normalized to canonical snake_case (state) and lowercased county name without suffix.
    """
    context = context or {}
    # Direct fields
    state = context.get("state") or context.get("state_abbr")
    county = context.get("county") or context.get("county_name")
    # Fallback to URL-based inference
    if not state or not county:
        for k in ("source_url", "url", "page_url", "origin_url"):
            u = context.get(k)
            if u:
                s2, c2 = infer_state_county_from_url(str(u))
                state = state or s2
                county = county or c2
                if state and county:
                    break
    # Normalize
    state = normalize_state_name(state) if state else None
    county = normalize_county_name(county) if county else None
    return state, county


def format_state_label(raw: Optional[str]) -> str:
    """Return a human-readable state label from a raw or normalized value."""
    if not raw:
        return ""
    candidate = normalize_state_name(raw)
    if candidate and candidate in set(STATE_ABBR.values()):
        return candidate.replace("_", " ").title()
    text = str(raw).strip()
    if not text or text.lower() == "unknown":
        return ""
    text = re.sub(r"\s+state$", "", text, flags=re.I)
    text = text.replace("_", " ")
    return text.title()


def canonicalize_county_label(state: Optional[str], county: Optional[str]) -> Optional[str]:
    """Return the canonical county/parish label for a state, if known."""
    state_norm = normalize_state_name(state) if state else None
    county_norm = normalize_county_name(county) if county else None
    if not state_norm or not county_norm:
        return None
    for candidate in KNOWN_STATE_TO_COUNTY_MAP.get(state_norm, []):
        if normalize_county_name(candidate) == county_norm:
            return candidate
    return None


def format_county_label(raw: Optional[str], state: Optional[str] = None) -> str:
    """Return a human-readable county label from a raw or normalized value."""
    if not raw:
        return ""

    canonical = canonicalize_county_label(state, raw)
    text_source = canonical if canonical is not None else raw
    candidate = normalize_county_name(text_source)
    text = str(text_source).strip()
    if not text or text.lower() == "unknown":
        return ""

    base = ""
    if candidate and candidate not in {"unknown", "statewide", "total"}:
        base = candidate.replace("_", " ").title()
    else:
        base = re.sub(r"\s+", " ", text.replace("_", " ")).title()

    reference_lower = f"{raw or ''} {canonical or ''}".lower()
    lower_text = reference_lower
    if "county" in lower_text and not base.lower().endswith("county"):
        base = f"{base} County"
    elif "parish" in lower_text and not base.lower().endswith("parish"):
        base = f"{base} Parish"
    elif "borough" in lower_text and not base.lower().endswith("borough"):
        base = f"{base} Borough"

    return base.strip()


def _table_sample_text(
    headers: Sequence[Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    max_rows: int = 200,
    max_chars: int = 80_000,
) -> str:
    parts: List[str] = []
    header_line = " | ".join(str(h).strip() for h in headers if isinstance(h, str) and h.strip())
    if header_line:
        parts.append(header_line)
    for row in rows[:max_rows]:
        if not isinstance(row, Mapping):
            continue
        values: List[str] = []
        for header in headers:
            if not isinstance(header, str):
                continue
            raw_val = row.get(header)
            if raw_val is None:
                continue
            text = str(raw_val).strip()
            if text:
                values.append(text)
        if values:
            parts.append(" | ".join(values))
        if sum(len(p) for p in parts) > max_chars:
            break
    return "\n".join(parts)


def derive_state_county_from_table(
    headers: Sequence[Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    context: Optional[Mapping[str, Any]] = None,
    filename: Optional[str] = None,
    sample_text: Optional[str] = None,
    use_dynamic_detection: bool = True,
) -> tuple[Optional[str], Optional[str], Dict[str, Any]]:
    """Infer state/county for tabular payloads using context, columns, filename, and NLP detection."""

    diagnostics: Dict[str, Any] = {
        "state_sources": [],
        "county_sources": [],
        "state_normalized": None,
        "county_normalized": None,
        "detection_log": [],
    }

    valid_states: Set[str] = set(STATE_ABBR.values())
    state_norm: Optional[str] = None
    county_norm: Optional[str] = None

    def _record_state(value: Optional[str], source: str) -> None:
        nonlocal state_norm
        if not value:
            return
        candidate = normalize_state_name(value)
        if not candidate or candidate not in valid_states or candidate in {"unknown", "statewide"}:
            return
        if state_norm == candidate:
            if source not in diagnostics["state_sources"]:
                diagnostics["state_sources"].append(source)
            return
        state_norm = candidate
        diagnostics["state_sources"].append(source)
        diagnostics["state_normalized"] = candidate

    def _record_county(value: Optional[str], source: str) -> None:
        nonlocal county_norm
        if not value:
            return
        candidate = normalize_county_name(value)
        if not candidate or candidate in {"unknown", "total", "overall", "statewide", "all"}:
            return
        if county_norm == candidate:
            if source not in diagnostics["county_sources"]:
                diagnostics["county_sources"].append(source)
            return
        county_norm = candidate
        diagnostics["county_sources"].append(source)
        diagnostics["county_normalized"] = candidate

    ctx_state, ctx_county = resolve_state_county_from_context(dict(context or {}))
    if ctx_state:
        _record_state(ctx_state, "context")
    if ctx_county:
        _record_county(ctx_county, "context")

    def _first_non_empty(column: str) -> str:
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            val = row.get(column)
            if val is None:
                continue
            text = str(val).strip()
            if text:
                return text
        return ""

    for header in headers:
        if not isinstance(header, str):
            continue
        stripped = header.strip()
        low = stripped.lower()
        if re.search(r"\bstate\b", low):
            candidate = _first_non_empty(stripped)
            _record_state(candidate, f"column:{stripped}")
        elif re.search(r"\b(county|parish)\b", low) or "jurisdiction" in low:
            candidate = _first_non_empty(stripped)
            _record_county(candidate, f"column:{stripped}")

    if filename:
        stem = os.path.splitext(os.path.basename(filename))[0]
        _record_state(stem, "filename")
        county_match = re.search(r"([a-z][a-z\s]+?)\s+county\b", stem.lower())
        if county_match:
            _record_county(county_match.group(1), "filename")

    text_blob = sample_text or _table_sample_text(headers, rows)

    if use_dynamic_detection and (not state_norm or not county_norm):
        try:
            from ..Context_Integration.context_coordinator import dynamic_state_county_detection

            detection_context = dict(context or {})
            if filename:
                detection_context.setdefault("source_file", os.path.basename(filename))
            if state_norm:
                detection_context.setdefault("state", format_state_label(state_norm))
            if county_norm:
                detection_context.setdefault("county", format_county_label(county_norm, state_norm))
            if detection_context.get("contest") and "contests" not in detection_context:
                detection_context["contests"] = [{"title": detection_context.get("contest")}]  # type: ignore

            det_county, det_state, _handler, det_log = dynamic_state_county_detection(
                detection_context,
                text_blob,
                debug=False,
            )
            _record_state(det_state, "dynamic_detection")
            _record_county(det_county, "dynamic_detection")
            if det_log:
                diagnostics["detection_log"] = list(det_log[:25])
        except Exception:
            pass

    state_display = format_state_label(state_norm or ctx_state) if (state_norm or ctx_state) else None
    county_display = format_county_label(county_norm or ctx_county, state_norm or ctx_state) if (county_norm or ctx_county) else None
    diagnostics["state_display"] = state_display
    diagnostics["county_display"] = county_display

    return state_display, county_display, diagnostics


def derive_candidate_party_metadata(
    headers: Sequence[Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    max_rows: int = 250,
) -> tuple[Dict[str, str], List[Dict[str, Any]], Dict[str, Any]]:
    """Infer candidate→party relationships from tabular rows."""

    candidate_headers: List[str] = []
    party_headers: List[str] = []
    for header in headers:
        if not isinstance(header, str):
            continue
        stripped = header.strip()
        if not stripped:
            continue
        low = stripped.lower()
        if any(token in low for token in ("party", "affiliation", "political", "designation")):
            if stripped not in party_headers:
                party_headers.append(stripped)
            continue
        if any(token in low for token in ("candidate", "choice", "option", "nominee")):
            if stripped not in candidate_headers:
                candidate_headers.append(stripped)

    if not candidate_headers or not party_headers:
        return {}, [], {
            "candidate_columns": candidate_headers,
            "party_columns": party_headers,
            "candidate_count": 0,
        }

    def _clean(value: Any) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        return text

    skip_prefixes = ("total", "all", "overall", "combined")
    candidate_map: Dict[str, Set[str]] = {}
    for row in rows[:max_rows]:
        if not isinstance(row, Mapping):
            continue
        candidate_val = ""
        for header in candidate_headers:
            candidate_val = _clean(row.get(header))
            if candidate_val:
                break
        if not candidate_val:
            continue
        c_norm = candidate_val.lower()
        if any(c_norm.startswith(prefix) for prefix in skip_prefixes):
            continue
        party_val = ""
        for header in party_headers:
            party_val = _clean(row.get(header))
            if party_val:
                break
        bucket = candidate_map.setdefault(candidate_val, set())
        if party_val:
            bucket.add(party_val)

    diagnostics = {
        "candidate_columns": candidate_headers,
        "party_columns": party_headers,
        "candidate_count": len(candidate_map),
    }
    if not candidate_map:
        return {}, [], diagnostics

    candidate_label_map: Dict[str, str] = {}
    candidate_metadata: List[Dict[str, Any]] = []
    party_summary: Dict[str, List[str]] = {}

    for candidate in sorted(candidate_map.keys()):
        raw_parties = sorted({p for p in candidate_map[candidate] if p.strip()})
        normalized_parties = []
        for raw_party in raw_parties:
            try:
                norm = normalize_party_label(raw_party)
            except Exception:
                norm = raw_party.strip().title()
            if norm:
                normalized_parties.append(norm)
        normalized_parties = sorted({p for p in normalized_parties if p})
        if len(normalized_parties) == 1:
            party_display = normalized_parties[0]
        elif len(normalized_parties) > 1:
            party_display = "/".join(normalized_parties)
        else:
            party_display = ""
        display_label = f"{candidate} ({party_display})" if party_display else candidate
        candidate_label_map[candidate] = display_label
        candidate_metadata.append({
            "id": safe_slug(candidate, 80),
            "raw_name": candidate,
            "display_label": display_label,
            "party": party_display,
            "party_raw_values": raw_parties,
            "derived_from": "row_party_columns",
        })
        party_summary[candidate] = normalized_parties

    diagnostics["party_map"] = party_summary
    diagnostics["candidate_label_map_size"] = len(candidate_label_map)

    return candidate_label_map, candidate_metadata, diagnostics

def build_camelot_row_filter_for_context(context: Optional[dict], candidate_keys: tuple[str, ...] = ("Candidate", "Party")):
    """Build a Camelot row-noise predicate using jurisdiction-aware overrides resolved from context.

    The returned callable(row: dict) -> bool indicates whether a row should be treated as noise.
    """
    s, c = resolve_state_county_from_context(context)
    return build_camelot_row_filter(candidate_keys=candidate_keys, state=s, county=c)

def record_noise_suggestion(state: Optional[str], county: Optional[str], snippet: str, category: str = "row") -> None:
    """Record a dropped-noise snippet for future analysis and pattern suggestions.

    Appends counts to output/noise_suggestions.json with structure:
    { state: { county: { category: { snippet: count } } } }
    """
    try:
        state = normalize_state_name(state) if state else "__unknown__"
        county = normalize_county_name(county) if county else "__unknown__"
        snippet = (snippet or "").strip()
        if not snippet:
            return
        if len(snippet) > 160:
            snippet = snippet[:160]
        out_dir = Path("output")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "noise_suggestions.json"
        data = {}
        if out_path.exists():
            try:
                data = orjson.loads(out_path.read_bytes())
            except Exception:
                data = {}
        data.setdefault(state, {}).setdefault(county, {}).setdefault(category, {})
        bucket = data[state][county][category]
        bucket[snippet] = int(bucket.get(snippet, 0)) + 1
        out_path.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))
    except Exception:
        # Best-effort only
        pass

def get_county_precincts(county_name) -> Optional[list]:
    county_norm = normalize_county_name(county_name)
    return KNOWN_COUNTY_TO_PRECINCTS_MAP.get(county_norm)


def normalize_county_key(county: Optional[str]) -> str:
    """Normalize free-form county text to a lookup key for precinct maps."""
    if not county:
        return ""
    normalized = normalize_county_name(county)
    if normalized:
        return normalized
    lowered = re.sub(r"[^a-z0-9]+", " ", county.strip().lower())
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def lookup_precinct_aliases_for_county(county: Optional[str]) -> list[str]:
    """Return known precinct/municipality aliases for a county, if available."""
    key = normalize_county_key(county)
    if not key:
        return []
    aliases = KNOWN_COUNTY_TO_PRECINCTS_MAP.get(key, [])
    # Return a shallow copy to prevent accidental mutation of global constants.
    return list(aliases) if aliases else []


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
    from .model_registry import ModelRegistry
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

def normalize_label(label) -> str:
    if not label:
        return ""
    return re.sub(r"\W+", "", str(label).strip().lower())

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
            from .html_scanner import extract_year_and_type
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

    # --- Ensure type_ and election_types are synced ---
    contest_dict["year"] = year
    contest_dict["type_"] = type_
    contest_dict["state"] = state
    contest_dict["county"] = county
    sync_type_and_election_types(contest_dict)

    return year, type_, state, county

# =============================================================================
# Project inventory and architecture.md updater
# =============================================================================

def _infer_category(rel_path: str) -> str:
    p = rel_path.replace("\\", "/")
    # Webapp areas
    if "/webapp/parser/handlers/states/" in p:
        return "State Handlers"
    if "/webapp/parser/handlers/formats/" in p:
        return "Format Handlers"
    if "/webapp/parser/Context_Integration/" in p:
        return "Context & Integrity"
    if "/webapp/parser/services/" in p:
        return "Services"
    if "/webapp/parser/health/" in p:
        return "Health"
    if "/webapp/parser/utils/" in p:
        return "Utilities"
    if "/webapp/templates/" in p:
        return "Templates"
    if "/webapp/static/" in p:
        return "Static Assets"
    if "/tests/" in p or p.endswith("/tests"):
        return "Tests"
    if p.startswith("docs/"):
        return "Docs"
    return "Misc"

def _read_module_summary(abs_path: Path) -> tuple[str, int, int]:
    """Return (summary, func_count, class_count) from a Python module."""
    try:
        import ast
        src = abs_path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(src)
        doc = ast.get_docstring(tree) or ""
        # Derive a one-line summary
        summary = (doc.strip().splitlines()[0] if doc.strip() else "")
        funcs = sum(isinstance(n, ast.FunctionDef) for n in tree.body)
        classes = sum(isinstance(n, ast.ClassDef) for n in tree.body)
        return summary, funcs, classes
    except Exception:
        # Fallback: first non-empty comment line
        try:
            for line in abs_path.read_text(encoding="utf-8", errors="replace").splitlines():
                ls = line.strip()
                if ls.startswith("# ") and len(ls) > 2:
                    return ls[2:], 0, 0
        except Exception:
            pass
        return "", 0, 0

def _is_ignored_dir(name: str) -> bool:
    low = name.lower()
    return low in {".git", "__pycache__", ".venv", "venv", "node_modules", ".mypy_cache", ".pytest_cache"}

def generate_project_inventory(project_root: str | Path = ".") -> Dict[str, List[Dict[str, Any]]]:
    """Walk the repository and build a categorized inventory of files.

    Returns a dict: {category: [ {path, summary, functions, classes, loc} ]}
    """
    root = Path(project_root).resolve()
    inventory: Dict[str, List[Dict[str, Any]]] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        # prune ignored directories in-place for performance
        dirnames[:] = [d for d in dirnames if not _is_ignored_dir(d)]
        for fname in filenames:
            rel = Path(dirpath).joinpath(fname).relative_to(root)
            rel_s = str(rel).replace("\\", "/")
            category = _infer_category(rel_s)
            item: Dict[str, Any] = {"path": rel_s}
            abs_path = Path(dirpath) / fname
            # Simple LOC (non-empty)
            try:
                loc = sum(1 for line in abs_path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip())
            except Exception:
                loc = 0
            item["loc"] = loc
            if fname.endswith(".py"):
                summary, funcs, classes = _read_module_summary(abs_path)
                item.update({
                    "summary": summary,
                    "functions": funcs,
                    "classes": classes,
                })
            inventory.setdefault(category, []).append(item)
    # Sort within categories by path
    for k in list(inventory.keys()):
        inventory[k].sort(key=lambda x: x.get("path", ""))
    return inventory

def _render_inventory_md(inv: Dict[str, List[Dict[str, Any]]]) -> str:
    lines: List[str] = []
    total_files = sum(len(v) for v in inv.values())
    total_loc = sum(sum(i.get("loc", 0) for i in v) for v in inv.values())
    lines.append(f"Inventory summary: {total_files} files, ~{total_loc} non-empty LOC")
    lines.append("")
    for category in sorted(inv.keys()):
        lines.append(f"### {category}")
        lines.append("")
        for item in inv[category]:
            path = item.get("path", "")
            summary = item.get("summary") or ""
            funcs = item.get("functions")
            classes = item.get("classes")
            loc = item.get("loc")
            meta = []
            if isinstance(funcs, int):
                meta.append(f"funcs: {funcs}")
            if isinstance(classes, int):
                meta.append(f"classes: {classes}")
            if isinstance(loc, int):
                meta.append(f"loc: {loc}")
            meta_s = f" ({', '.join(meta)})" if meta else ""
            bullet = f"- `{path}`{meta_s}"
            if summary:
                bullet += f": {summary}"
            lines.append(bullet)
        lines.append("")
    return "\n".join(lines).strip() + "\n"

def _finalize_markdown_lines(lines: list[str]) -> str:
    """Ensure markdown headings/lists have required blank lines, wrap long lines, and collapse extras."""
    processed: list[str] = []
    total = len(lines)
    for idx, line in enumerate(lines):
        if line.startswith("#"):
            if processed and processed[-1] != "":
                processed.append("")
            processed.append(line)
            if idx + 1 < total and lines[idx + 1] != "":
                processed.append("")
        else:
            processed.append(line)
    deduped: list[str] = []
    prev_blank = False
    for line in processed:
        if line == "":
            if not prev_blank:
                deduped.append(line)
            prev_blank = True
        else:
            deduped.append(line)
            prev_blank = False

    wrapped: list[str] = []
    in_fence = False
    fence_delim = ""
    fence_pattern = re.compile(r"^(```|~~~)")
    bullet_pattern = re.compile(r"^(\s*(?:[-*]|\d+\.)\s+)(.+)$")
    for line in deduped:
        stripped = line.strip()
        if line.startswith("#"):
            wrapped.append(line)
            continue
        fence_match = fence_pattern.match(stripped)
        if fence_match:
            delim = fence_match.group(1)
            if not in_fence:
                in_fence = True
                fence_delim = delim
            elif delim == fence_delim:
                in_fence = False
                fence_delim = ""
            wrapped.append(line)
            continue
        if in_fence or not stripped:
            wrapped.append(line)
            continue
        if stripped.startswith(">") or stripped.startswith("|"):
            wrapped.append(line)
            continue
        if stripped.startswith("{:"):
            wrapped.append(line)
            continue
        bullet_match = bullet_pattern.match(line)
        if line.startswith("    ") and not bullet_match:
            wrapped.append(line)
            continue
        if bullet_match:
            prefix, content = bullet_match.groups()
            wrapped.extend(
                textwrap.wrap(
                    content.strip(),
                    width=80,
                    initial_indent=prefix,
                    subsequent_indent=" " * len(prefix),
                    break_long_words=False,
                    break_on_hyphens=False,
                )
                or [line]
            )
            continue
        wrapped.extend(
            textwrap.wrap(
                line.strip(),
                width=80,
                initial_indent="",
                subsequent_indent="",
                break_long_words=False,
                break_on_hyphens=False,
            )
            or [line]
        )
    return "\n".join(wrapped).rstrip() + "\n"

def update_architecture_md(project_root: str | Path = ".", md_path: str | Path = "docs/architecture.md") -> bool:
    """Replace the AUTO-INVENTORY block in architecture.md with a fresh inventory."""
    try:
        root = Path(project_root).resolve()
        md_file = (root / md_path).resolve()
        if not md_file.exists():
            logger.warning(f"[inventory] architecture.md not found at {md_file}")
            return False
        text = md_file.read_text(encoding="utf-8", errors="replace")
        begin = "<!-- AUTO-INVENTORY:START -->"
        end = "<!-- AUTO-INVENTORY:END -->"
        if begin not in text or end not in text:
            logger.warning("[inventory] Markers not found in architecture.md; aborting replace.")
            return False
        inv = generate_project_inventory(root)
        block = _render_inventory_md(inv)
        new_text = text.split(begin)[0] + begin + "\n\n" + block + "\n" + end + text.split(end)[1]
        md_file.write_text(new_text, encoding="utf-8")
        return True
    except Exception as e:
        logger.error(f"[inventory] Failed to update architecture.md: {e}")
        return False

def generate_project_map(project_root: str | Path = ".", out_markdown: str | Path = "docs/architecture.md") -> None:
    """Compatibility wrapper referenced in docs; updates architecture.md in-place."""
    ok = update_architecture_md(project_root=project_root, md_path=out_markdown)
    if not ok:
        logger.warning("[inventory] generate_project_map completed with warnings; check markers and path.")

# =============================================================================
# Static code audit: import graph, symbol defs, cross-module calls
# =============================================================================

def _posix(p: Path) -> str:
    return str(p).replace("\\", "/")

def _read_file_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""

def _extract_top_comment_block(src: str) -> str:
    """Return leading comment block (lines starting with '#') before first code/docstring.

    Stops when encountering a non-comment, non-blank line or a triple-quoted string start.
    """
    lines = src.splitlines()
    out: list[str] = []
    in_block = True
    for line in lines:
        ls = line.strip()
        if not ls:
            if in_block:
                out.append("")
            else:
                break
            continue
        # stop on docstring opening
        if ls.startswith("\"\"") or ls.startswith("'''"):
            break
        if ls.startswith("#"):
            out.append(line)
            continue
        # first code-ish line ends the top comment block
        break
    # Trim trailing blank lines
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out)


def _harvest_todos(src: str) -> list[tuple[int, str, str]]:
    """Find lines containing TODO/FIXME/WARN and similar keywords (case-insensitive). Returns list of (lineno, keyword, cleaned_text)."""
    hits: list[tuple[int, str, str]] = []
    pat = re.compile(r"\b(TODO|FIXME|WARN|WARNING|NOTE|HACK|XXX|BUG)\b", re.IGNORECASE)
    for i, line in enumerate(src.splitlines(), start=1):
        match = pat.search(line)
        if match:
            keyword = match.group(1).upper()
            # Extract text after the keyword, clean it
            after = line[match.end():].strip()
            # Remove leading punctuation like :, -, etc.
            after = re.sub(r'^[:\-\s]*', '', after).strip()
            hits.append((i, keyword, after or line.rstrip()))
    return hits


def _module_info_from_ast(src: str, file_path: Path) -> dict:
    import ast
    info: dict = {
        "path": _posix(file_path),
        "doc": "",
        "defs": [],            # list of {type, name, lineno}
        "imports": [],         # list of {type, module, name, alias, lineno}
        "aliases": {},         # Name -> fully qualified (module[.name])
        "module_aliases": {},  # alias -> module
        "calls": [],           # list of {func, kind, lineno}
        "loc": 0,
        "top_comment": "",
        "todo_lines": [],      # list[(lineno, text)]
    }
    try:
        tree = ast.parse(src)
    except Exception:
        return info
    info["doc"] = (ast.get_docstring(tree) or "").strip()
    info["loc"] = sum(1 for line in src.splitlines() if line.strip())
    info["top_comment"] = _extract_top_comment_block(src)
    info["todo_lines"] = _harvest_todos(src)

    # Collect defs and imports
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            info["defs"].append({"type": "function", "name": node.name, "lineno": getattr(node, "lineno", 0)})
        elif isinstance(node, ast.AsyncFunctionDef):
            info["defs"].append({"type": "async_function", "name": node.name, "lineno": getattr(node, "lineno", 0)})
        elif isinstance(node, ast.ClassDef):
            info["defs"].append({"type": "class", "name": node.name, "lineno": getattr(node, "lineno", 0)})
        elif isinstance(node, ast.Import):
            for alias in node.names:
                info["imports"].append({"type": "import", "module": alias.name, "name": None, "alias": alias.asname, "lineno": getattr(node, "lineno", 0)})
                if alias.asname:
                    info["module_aliases"][alias.asname] = alias.name
                else:
                    # bare import foo means name 'foo' binds to module
                    base = alias.name.split(".")[0]
                    info["module_aliases"].setdefault(base, alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for alias in node.names:
                info["imports"].append({"type": "from", "module": mod, "name": alias.name, "alias": alias.asname, "lineno": getattr(node, "lineno", 0)})
                bound = alias.asname or alias.name
                fq = f"{mod}.{alias.name}" if mod else alias.name
                info["aliases"][bound] = fq

    # Walk for calls
    import ast
    class CallVisitor(ast.NodeVisitor):
        def visit_Call(self, call: ast.Call):
            tgt = None
            kind = "local"
            # f(...) where f is Name
            if isinstance(call.func, ast.Name):
                nm = call.func.id
                if nm in info["aliases"]:
                    tgt = info["aliases"][nm]
                    kind = "call:alias"
                elif nm in info["module_aliases"]:
                    tgt = info["module_aliases"][nm]
                    kind = "call:module"
                else:
                    tgt = nm
                    kind = "call:local"
            # alias.attr(...) where alias is a module alias
            elif isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
                base = call.func.value.id
                attr = call.func.attr
                if base in info["module_aliases"]:
                    tgt = f"{info['module_aliases'][base]}.{attr}"
                    kind = "call:module.attr"
                elif base in info["aliases"]:
                    # e.g., from x import y as z; z.attr()
                    tgt = f"{info['aliases'][base]}.{attr}"
                    kind = "call:alias.attr"
                else:
                    tgt = f"{base}.{attr}"
                    kind = "call:attr"
            if tgt:
                info["calls"].append({
                    "func": tgt,
                    "kind": kind,
                    "lineno": getattr(call, "lineno", 0),
                })
            self.generic_visit(call)

    try:
        CallVisitor().visit(tree)
    except Exception:
        pass
    return info

def _scan_webapp_modules(project_root: Path) -> list[dict]:
    webapp = (project_root / "webapp").resolve()
    if not webapp.exists():
        return []
    modules = []
    for dirpath, dirnames, filenames in os.walk(webapp):
        # prune
        dirnames[:] = [d for d in dirnames if not _is_ignored_dir(d)]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            p = Path(dirpath) / fn
            src = _read_file_text(p)
            if not src:
                continue
            info = _module_info_from_ast(src, p)
            modules.append(info)
    return modules

def _index_defs(modules: list[dict]) -> dict:
    """Map fully-qualified guess for definitions to file path and line."""
    idx: dict[str, dict] = {}
    for m in modules:
        mod_path = m.get("path", "")
        mod_name = mod_path.replace("/", ".").rstrip(".py")
        # derive package-ish name by trimming project root parts until 'webapp'
        try:
            i = mod_name.index("webapp")
            mod_name = mod_name[i:].removesuffix(".py")
        except ValueError:
            pass
        for d in m.get("defs", []):
            nm = d.get("name")
            if not nm:
                continue
            key = f"{mod_name}:{nm}"
            idx[key] = {"path": m.get("path"), "lineno": d.get("lineno"), "type": d.get("type")}
    return idx

def _resolve_targets(modules: list[dict], def_index: dict) -> tuple[list[dict], dict[str, list[dict]]]:
    """Return edges and inbound map.
    edges: list of {src_path, src_line, target, resolved_path?, resolved_line?}
    inbound: target_key -> list[edge]
    """
    edges: list[dict] = []
    inbound: dict[str, list[dict]] = {}
    for m in modules:
        src_path = m.get("path")
        for c in m.get("calls", []):
            tgt = c.get("func")
            # Try to resolve to known defs by matching suffix ':name'
            resolved_key = None
            if ":" in tgt:
                # already module:name form
                if tgt in def_index:
                    resolved_key = tgt
            else:
                # find by name across modules (could be many; pick first)
                for k in def_index.keys():
                    if k.endswith(":" + tgt):
                        resolved_key = k
                        break
            edge = {
                "src_path": src_path,
                "src_line": c.get("lineno"),
                "target": tgt,
            }
            if resolved_key:
                di = def_index[resolved_key]
                edge["resolved_path"] = di.get("path")
                edge["resolved_line"] = di.get("lineno")
                inbound.setdefault(resolved_key, []).append(edge)
            edges.append(edge)
    return edges, inbound

def _render_audit_md(modules: list[dict], def_index: dict, edges: list[dict], inbound: dict[str, list[dict]], root: Path) -> str:
    import re
    import os
    
    # Diagram rendering constants
    MAX_DIAGRAM_EDGES = 15  # Maximum number of edges to show in mermaid diagrams
    MAX_SUBGRAPH_NODES = 8  # Maximum nodes per cluster subgraph
    
    # Summary
    lines: list[str] = []
    # Add YAML front matter for GitHub Pages
    lines.append("---")
    lines.append("layout: default")
    lines.append('title: "Project Audit"')
    lines.append("---")
    lines.append("")
    total = len(modules)
    total_loc = sum(m.get("loc", 0) for m in modules)
    lines.append("Audit scope: `webapp/parser/` modules.")
    lines.append("")
    lines.append(f"Modules scanned: {total} | ~{total_loc} non-empty LOC")
    lines.append("")

    # Helper function to build cluster nodes
    def _build_cluster_nodes() -> dict[str, set[str]]:
        cluster_nodes: dict[str, set[str]] = {k: set() for k in ["Entry","Pipeline","Routing","Handlers","Services","Utils","Context_Integration","Health","Other"]}
        for e in edges:
            sp = e.get("src_path")
            dp = e.get("resolved_path")
            # Only include edges where both src and dst resolve to actual modules
            if not sp or not dp:
                continue
            sm = _to_mod(sp)
            dm = _to_mod(dp)
            if not sm or not dm or dm == "unknown" or sm == "unknown" or sm == dm:
                continue
            sc = _cluster_for_path(sp)
            dc = _cluster_for_path(dp)
            cluster_nodes[sc].add(sm)
            cluster_nodes[dc].add(dm)
        return cluster_nodes

    def _to_mod(p: str | None) -> str:
        if not p:
            return "unknown"
        # Collapse to package-like name from webapp/ onward
        s = p.replace("\\", "/")
        i = s.find("webapp/")
        if i >= 0:
            s = s[i:]
        full = s.replace("/", ".").removesuffix(".py")
        parts = full.split(".")
        # Use last 1 part to shorten
        return ".".join(parts[-1:]) if len(parts) > 1 else full

    def _cluster_for_path(p: str) -> str:
        if not p:
            return "Other"
        p = p.replace("\\", "/")
        if p.endswith("/webapp/Smart_Elections_Parser_Webapp.py"):
            return "Entry"
        if "/webapp/parser/web_pipeline.py" in p:
            return "Pipeline"
        if "/webapp/parser/state_router.py" in p:
            return "Routing"
        if "/webapp/parser/handlers/states/" in p:
            return "State Handlers"
        if "/webapp/parser/handlers/formats/" in p:
            return "Format Handlers"
        if "/webapp/parser/handlers/shared/" in p:
            return "Shared Handlers"
        if "/webapp/parser/services/" in p:
            return "Services"
        if "/webapp/parser/utils/" in p:
            return "Utils"
        if "/webapp/parser/Context_Integration/" in p:
            return "Context Integration"
        if "/webapp/parser/health/" in p:
            return "Health"
        return "Other"

    # High-level call graph (top 25 edges)
    lines.append("## Pipeline map (Mermaid)")
    lines.append("")
    # Build module-level edges using resolved paths ONLY (skip if no resolved_path)
    edge_counts: dict[tuple[str, str], int] = {}
    for e in edges:
        sp = e.get("src_path")
        dp = e.get("resolved_path")
        # Only include edges where both src and dst resolve to actual modules
        if not sp or not dp:
            continue
        src = _to_mod(sp)
        dst = _to_mod(dp)
        if src == dst or dst == "unknown" or src == "unknown":
            continue
        edge_counts[(src, dst)] = edge_counts.get((src, dst), 0) + 1
    # Top edges (limited for readability)
    top_edges = sorted(edge_counts.items(), key=lambda kv: -kv[1])[:MAX_DIAGRAM_EDGES]
    
    # Build node-to-cluster mapping from edges to ensure all edge nodes are included
    node_to_cluster: dict[str, str] = {}
    cluster_pair_counts: dict[tuple[str, str], int] = {}
    for e in edges:
        sp = e.get("src_path")
        dp = e.get("resolved_path")
        if sp:
            sm = _to_mod(sp)
            if sm != "unknown":
                node_to_cluster[sm] = _cluster_for_path(sp)
        if dp:
            dm = _to_mod(dp)
            if dm != "unknown":
                node_to_cluster[dm] = _cluster_for_path(dp)
        if sp and dp:
            sc = _cluster_for_path(sp)
            dc = _cluster_for_path(dp)
            cluster_pair_counts[(sc, dc)] = cluster_pair_counts.get((sc, dc), 0) + 1
    
    # Collect all edge nodes
    edge_nodes: set[str] = set()
    for (src, dst), _ in top_edges:
        edge_nodes.add(src)
        edge_nodes.add(dst)
    
    lines.append("")
    lines.append("```mermaid")
    lines.append("graph LR")
    # Subgraphs - prioritize nodes that appear in edges
    cluster_nodes = _build_cluster_nodes()
    for cname in ["Entry","Pipeline","Routing","Handlers","Services","Utils","Context_Integration","Health"]:
        all_cluster_nodes = cluster_nodes.get(cname, set())
        # Prioritize edge nodes in this cluster
        priority_nodes = [n for n in edge_nodes if node_to_cluster.get(n) == cname]
        other_nodes = [n for n in all_cluster_nodes if n not in priority_nodes]
        # Include all priority nodes first, then fill up to limit
        max_nodes = max(MAX_SUBGRAPH_NODES, len(priority_nodes))
        nodes = sorted(priority_nodes) + sorted(other_nodes)
        nodes = nodes[:max_nodes]
        if not nodes:
            continue
        lines.append(f"  subgraph {cname}[\"{cname}\"]")
        for n in nodes:
            lines.append(f"    {n.replace('.', '_')}[\"{n}\"]")
        lines.append("  end")
    for (src, dst), cnt in top_edges:
        lines.append(f"  {src.replace('.', '_')} -->|{cnt}| {dst.replace('.', '_')}")
    if not top_edges:
        lines.append("  A[no data] --> B[no data]")
    lines.append("```")
    lines.append("")

    # Connection summary to emphasize critical paths
    lines.append("## Connection highlights")
    lines.append("")
    lines.append("Key module-to-module and cluster relationships to watch during refactors.")
    lines.append("")
    if top_edges:
        lines.append("### Top module edges")
        lines.append("")
        for (src, dst), cnt in top_edges[:10]:
            sc = node_to_cluster.get(src, "Other")
            dc = node_to_cluster.get(dst, "Other")
            lines.append(f"- `{src}` → `{dst}` ({cnt} refs, {sc} → {dc})")
        lines.append("")
    else:
        lines.append("- No module-level edges detected.")
        lines.append("")
    if cluster_pair_counts:
        lines.append("### Cluster flow summary")
        lines.append("")
        for (sc, dc), cnt in sorted(cluster_pair_counts.items(), key=lambda kv: -kv[1])[:10]:
            relation = "intra-cluster" if sc == dc else "cross-cluster"
            lines.append(f"- {sc} → {dc}: {cnt} edges ({relation})")
        lines.append("")
    else:
        lines.append("- Not enough cluster links to summarize.")
        lines.append("")

    # Compact pipeline focus (entry → pipeline → routing → handlers → utils)
    lines.append("## Pipeline focus (compact)")
    lines.append("")
    def _is_pipeline_path(p: str) -> bool:
        if not p:
            return False
        p = p.replace("\\", "/")
        return (
            p.endswith("/webapp/Smart_Elections_Parser_Webapp.py") or
            "/webapp/parser/web_pipeline.py" in p or
            "/webapp/parser/state_router.py" in p or
            "/webapp/parser/handlers/" in p or
            "/webapp/parser/utils/" in p
        )
    pipe_counts: dict[tuple[str, str], int] = {}
    for e in edges:
        sp = e.get("src_path")
        dp = e.get("resolved_path")
        # Only include edges where both src and dst are resolved module paths
        if not sp or not dp:
            continue
        if _is_pipeline_path(sp) and _is_pipeline_path(dp):
            src = _to_mod(sp)
            dst = _to_mod(dp)
            if src != dst and dst != "unknown" and src != "unknown":
                pipe_counts[(src, dst)] = pipe_counts.get((src, dst), 0) + 1
    top_pipe = sorted(pipe_counts.items(), key=lambda kv: -kv[1])[:MAX_DIAGRAM_EDGES]
    
    # Build node-to-cluster mapping for pipe edges
    pipe_node_to_cluster: dict[str, str] = {}
    for e in edges:
        sp = e.get("src_path")
        dp = e.get("resolved_path")
        if sp and _is_pipeline_path(sp):
            sm = _to_mod(sp)
            if sm != "unknown":
                pipe_node_to_cluster[sm] = _cluster_for_path(sp)
        if dp and _is_pipeline_path(dp):
            dm = _to_mod(dp)
            if dm != "unknown":
                pipe_node_to_cluster[dm] = _cluster_for_path(dp)
    
    # Collect all pipe edge nodes
    pipe_edge_nodes: set[str] = set()
    for (src, dst), _ in top_pipe:
        pipe_edge_nodes.add(src)
        pipe_edge_nodes.add(dst)
    
    lines.append("```mermaid")
    lines.append("graph LR")
    # Subgraphs - prioritize nodes that appear in pipe edges
    cluster_nodes = _build_cluster_nodes()
    for cname in ["Entry","Pipeline","Routing","Handlers","Services","Utils","Context_Integration","Health"]:
        all_cluster_nodes = cluster_nodes.get(cname, set())
        # Prioritize edge nodes in this cluster
        priority_nodes = [n for n in pipe_edge_nodes if pipe_node_to_cluster.get(n) == cname]
        other_nodes = [n for n in all_cluster_nodes if n not in priority_nodes]
        # Include all priority nodes first, then fill up to limit
        max_nodes = max(MAX_SUBGRAPH_NODES, len(priority_nodes))
        nodes = sorted(priority_nodes) + sorted(other_nodes)
        nodes = nodes[:max_nodes]
        if not nodes:
            continue
        lines.append(f"  subgraph {cname}[\"{cname}\"]")
        for n in nodes:
            lines.append(f"    {n.replace('.', '_')}[\"{n}\"]")
        lines.append("  end")
    for (src, dst), cnt in top_pipe:
        lines.append(f"  {src.replace('.', '_')} -->|{cnt}| {dst.replace('.', '_')}")
    if not top_pipe:
        lines.append("  A[no data] --> B[no data]")
    lines.append("```")
    lines.append("")

    # Cross-module hotspots (by inbound refs)
    lines.append("## Cross-module hotspots")
    lines.append("")
    hotspot = sorted(((k, len(v)) for k, v in inbound.items()), key=lambda x: -x[1])[:MAX_DIAGRAM_EDGES]
    if hotspot:
        for key, cnt in hotspot:
            path = def_index.get(key, {}).get("path", "")
            if path:
                try:
                    path = Path(path).name
                except:
                    pass
            lines.append(f"- {key} ← {cnt} refs ({path})")
    else:
        lines.append("- No cross-module references resolved.")
    lines.append("")

    # Leaf/legacy modules (zero inbound to any defs), excluding tests and __init__.py
    lines.append("## Leaf modules (candidates for review)")
    lines.append("")
    mod_has_inbound: dict[str, bool] = {}
    for key, refs in inbound.items():
        p = def_index.get(key, {}).get("path")
        if p:
            mod_has_inbound[p] = True
    leaves: list[str] = []
    for m in modules:
        p = m.get("path", "")
        if not p or p.endswith("__init__.py") or "/tests/" in p:
            continue
        if not mod_has_inbound.get(p):
            leaves.append(p)
    if leaves:
        for p in sorted(leaves)[:50]:
            try:
                rel_p = Path(p).name
                lines.append(f"- `{rel_p}`")
            except:
                lines.append(f"- `{p}`")
        if len(leaves) > 50:
            lines.append(f"- (+{len(leaves)-50} more hidden)")
    else:
        lines.append("- None detected.")
    lines.append("")

    # Clustered pipeline view (compact with subgraphs)
    lines.append("## Pipeline clusters (Mermaid)")
    lines.append("")
    def _cluster_for_path(p: str) -> str:
        if not p:
            return "Other"
        p = p.replace("\\", "/")
        if p.endswith("/webapp/Smart_Elections_Parser_Webapp.py"):
            return "Entry"
        if "/webapp/parser/web_pipeline.py" in p:
            return "Pipeline"
        if "/webapp/parser/state_router.py" in p:
            return "Routing"
        if "/webapp/parser/handlers/" in p:
            return "Handlers"
        if "/webapp/parser/services/" in p:
            return "Services"
        if "/webapp/parser/utils/" in p:
            return "Utils"
        if "/webapp/parser/Context_Integration/" in p:
            return "Context_Integration"
        if "/webapp/parser/health/" in p:
            return "Health"
        return "Other"
    # Build node sets by cluster and limited edges between them
    cluster_nodes: dict[str, set[str]] = {k: set() for k in ["Entry","Pipeline","Routing","Handlers","Services","Utils","Context_Integration","Health","Other"]}
    cluster_edges: dict[tuple[str,str], int] = {}
    node_to_cluster_map: dict[str, str] = {}
    for e in edges:
        sp = e.get("src_path")
        dp = e.get("resolved_path")
        # Only include edges where both src and dst are resolved module paths
        if not sp or not dp:
            continue
        sm = _to_mod(sp)
        dm = _to_mod(dp)
        if not sm or not dm or dm == "unknown" or sm == "unknown" or sm == dm:
            continue
        sc = _cluster_for_path(sp)
        dc = _cluster_for_path(dp)
        cluster_nodes[sc].add(sm)
        cluster_nodes[dc].add(dm)
        node_to_cluster_map[sm] = sc
        node_to_cluster_map[dm] = dc
        cluster_edges[(sm, dm)] = cluster_edges.get((sm, dm), 0) + 1
    # Keep only top edges for compactness
    top_cluster_edges = sorted(cluster_edges.items(), key=lambda kv: -kv[1])[:MAX_DIAGRAM_EDGES]
    
    # Collect all cluster edge nodes
    cluster_edge_nodes: set[str] = set()
    for (src, dst), _ in top_cluster_edges:
        cluster_edge_nodes.add(src)
        cluster_edge_nodes.add(dst)
    
    lines.append("```mermaid")
    lines.append("graph LR")
    # Subgraphs - prioritize nodes that appear in edges
    for cname in ["Entry","Pipeline","Routing","Handlers","Services","Utils","Context_Integration","Health"]:
        all_cluster_nodes = cluster_nodes.get(cname, set())
        # Prioritize edge nodes in this cluster
        priority_nodes = [n for n in cluster_edge_nodes if node_to_cluster_map.get(n) == cname]
        other_nodes = [n for n in all_cluster_nodes if n not in priority_nodes]
        # Include all priority nodes first, then fill up to limit
        max_nodes = max(MAX_SUBGRAPH_NODES, len(priority_nodes))
        nodes = sorted(priority_nodes) + sorted(other_nodes)
        nodes = nodes[:max_nodes]
        if not nodes:
            continue
        lines.append(f"  subgraph {cname}[\"{cname}\"]")
        for n in nodes:
            lines.append(f"    {n.replace('.', '_')}[\"{n}\"]")
        lines.append("  end")
    # Edges
    for (src, dst), cnt in top_cluster_edges:
        lines.append(f"  {src.replace('.', '_')} -->|{cnt}| {dst.replace('.', '_')}")
    if not top_cluster_edges:
        lines.append("  A[no data] --> B[no data]")
    lines.append("```")
    lines.append("")

    # Per-module detail
    lines.append("## Modules")
    for m in sorted(modules, key=lambda x: x.get("path", "")):
        path = m.get("path", "")
        if path:
            root_str = str(root).replace("\\", "/").lower()
            path_str = path.replace("\\", "/").lower()
            if path_str.startswith(root_str + "/") or path_str == root_str:
                # Use the original path_str for the display, but strip
                orig_path_str = path.replace("\\", "/")
                orig_root_str = str(root).replace("\\", "/")
                if orig_path_str.startswith(orig_root_str + "/"):
                    path = orig_path_str[len(orig_root_str):].lstrip("/")
                else:
                    path = orig_path_str  # fallback, but should not happen
            else:
                path = Path(path).name  # this should not happen now
        display_path = path.replace("webapp/parser/", "", 1) if path else "unknown"
        heading_label = display_path or path or "unknown"
        md_heading = heading_label.replace('_', r'\_')
        raw_ref = path or heading_label
        raw_id = re.sub(r'[^a-zA-Z0-9]+', '-', raw_ref).strip('-').lower()
        if not raw_id:
            raw_id = f"module-{abs(hash(raw_ref))}"
        lines.append(f"### {md_heading} {{#{raw_id}}}")
        lines.append("")
        if m.get("doc"):
            lines.append(f"> {m['doc'].splitlines()[0]}")
        # Top-of-file comments
        if m.get("top_comment"):
            lines.append("")
            lines.append("- Top-of-file comments:")
            lines.append("")
            lines.append("```python")
            for ln in m["top_comment"].splitlines():
                ln = re.sub(r'(\*|_)\s+(.+?)\s+(\*|_)', r'\1\2\3', ln)
                ln = re.sub(r'(\*|_)\s+', r'\1', ln)
                ln = re.sub(r'\s+(\*|_)', r'\1', ln)
                ln = re.sub(r'(\*|_)\s*([^ *]+)\s*(\*|_)', r'\1\2\3', ln)
                ln = ln.replace('\t', '    ').replace('<', '&lt;').replace('>', '&gt;').replace('*', '\\*').replace('_', '\\_')
                lines.append(ln)
            lines.append("```")
        lines.append("")
        # Definitions
        defs = m.get("defs", [])
        if defs:
            lines.append("- Definitions:")
            for d in defs:
                safe_name = d['name'].replace('_', '\\_')
                lines.append(f"  - {d['type']}: `{safe_name}` (line {d.get('lineno', '?')})")
        # Imports
        imps = m.get("imports", [])
        if imps:
            # Categorize imports
            stdlib_imports = []
            third_party_imports = []
            local_imports = []
            
            # Standard library modules (built-in)
            stdlib_modules = {
                'abc', 'argparse', 'ast', 'asyncio', 'base64', 'collections', 'contextlib', 
                'copy', 'csv', 'dataclasses', 'datetime', 'decimal', 'enum', 'functools', 
                'hashlib', 'heapq', 'html', 'http', 'inspect', 'io', 'itertools', 'json', 
                'logging', 'math', 'multiprocessing', 'operator', 'os', 'pathlib', 'pickle', 
                'platform', 'queue', 'random', 're', 'shutil', 'socket', 'sqlite3', 'ssl', 
                'statistics', 'string', 'subprocess', 'sys', 'tempfile', 'threading', 'time', 
                'timeit', 'traceback', 'typing', 'unittest', 'urllib', 'uuid', 'warnings', 
                'weakref', 'xml', 'zipfile', 'zlib'
            }
            
            for im in imps:
                module_name = im['module'].split('.')[0]
                if module_name in stdlib_modules:
                    stdlib_imports.append(im)
                elif module_name in ('webapp', 'flask', 'werkzeug', 'jinja2', 'click', 'itsdangerous',
                                   'blinker', 'markupsafe', 'orjson', 'python-dotenv', 'playwright',
                                   'PIL', 'pytesseract', 'spacy', 'transformers', 'torch', 'numpy',
                                   'pandas', 'requests', 'beautifulsoup4', 'lxml', 'selenium',
                                   'pytest', 'coverage', 'black', 'flake8', 'mypy', 'bandit',
                                   'sqlalchemy', 'psycopg2', 'pymongo', 'redis', 'celery',
                                   'twilio', 'sendgrid', 'boto3', 'google', 'azure', 'openai',
                                   'huggingface', 'sentence-transformers', 'scikit-learn', 'nltk'):
                    third_party_imports.append(im)
                else:
                    local_imports.append(im)
            
            lines.append("- Imports:")
            
            if stdlib_imports:
                lines.append("  - **Standard Library** (%d):" % len(stdlib_imports))
                for im in stdlib_imports[:50]:
                    if im["type"] == "import":
                        lines.append(f"    - `import {im['module']} as {im.get('alias') or im['module'].split('.')[0]}` (line {im.get('lineno','?')})")
                    else:
                        alias = im.get('alias')
                        alias_s = f" as {alias}" if alias else ""
                        lines.append(f"    - `from {im['module']} import {im['name']}{alias_s}` (line {im.get('lineno','?')})")
            
            if third_party_imports:
                lines.append("  - **Third-party** (%d):" % len(third_party_imports))
                for im in third_party_imports[:50]:
                    if im["type"] == "import":
                        lines.append(f"    - `import {im['module']} as {im.get('alias') or im['module'].split('.')[0]}` (line {im.get('lineno','?')})")
                    else:
                        alias = im.get('alias')
                        alias_s = f" as {alias}" if alias else ""
                        lines.append(f"    - `from {im['module']} import {im['name']}{alias_s}` (line {im.get('lineno','?')})")
            
            if local_imports:
                lines.append("  - **Local/Project** (%d):" % len(local_imports))
                for im in local_imports[:50]:
                    if im["type"] == "import":
                        lines.append(f"    - `import {im['module']} as {im.get('alias') or im['module'].split('.')[0]}` (line {im.get('lineno','?')})")
                    else:
                        alias = im.get('alias')
                        alias_s = f" as {alias}" if alias else ""
                        lines.append(f"    - `from {im['module']} import {im['name']}{alias_s}` (line {im.get('lineno','?')})")
        # TODO/FIXME/WARN
        todos = m.get("todo_lines", [])
        if todos:
            lines.append("- TODO/FIXME/WARN:")
            for ln, keyword, cleaned_txt in todos[:50]:
                safe_txt = cleaned_txt.replace("`", "\u2063`").replace("[", "\\[").replace("]", "\\]").replace('\t', ' ').replace('<', '&lt;').replace('>', '&gt;')  # avoid MD inline code breaks, link issues, tabs, inline HTML
                safe_txt = re.sub(r'(\*|_)\s+', r'\1', safe_txt)
                safe_txt = re.sub(r'\s+(\*|_)', r'\1', safe_txt)
                safe_txt = re.sub(r'(\*|_)\s*([^ *]*)\s*(\*|_)', r'\1\2\3', safe_txt)
                safe_txt = re.sub(r'(\*|_)\s*([^ *]*)\s*(\*|_)', r'\1\2\3', safe_txt)
                safe_txt = safe_txt.replace('_', '*')  # Replace any remaining _ with *
                lines.append(f"  - L{ln} **{keyword}**: {safe_txt}")
        # Outgoing calls (cross-module)
        calls = [c for c in m.get("calls", []) if any(sep in c.get("func"," ") for sep in (".", ":"))]
        if calls:
            lines.append("- Outgoing cross-module calls (sample):")
            for c in calls[:50]:
                tgt = c.get("func")
                tgt = tgt.replace('_', '\\_')
                res = ""
                # try to find a resolved match
                for k, di in def_index.items():
                    if k == tgt or k.endswith(":" + tgt.split(":")[-1]):
                        res_path = di.get("path")
                        if res_path:
                            try:
                                rel_res = Path(res_path).name
                                res = f" → {rel_res}:{di.get('lineno')}"
                            except:
                                res = f" → {res_path}:{di.get('lineno')}"
                        break
                lines.append(f"  - {tgt} (line {c.get('lineno','?')}){res}")
        # Inbound references to defs in this module
        local_keys = [k for k in def_index.keys() if def_index[k].get("path") == m.get("path", "")]
        inbound_here = []
        for k in local_keys:
            inbound_here.extend(inbound.get(k, []))
        if inbound_here:
            lines.append("- Inbound references:")
            for e in inbound_here[:50]:
                src = e.get("src_path")
                tgt = e.get("target")
                tgt = tgt.replace('_', '\\_')
                if src:
                    try:
                        src = Path(src).name
                    except:
                        pass
                lines.append(f"  - {tgt} ← {src}:{e.get('src_line','?')}")
        lines.append("")
    return _finalize_markdown_lines(lines)

def generate_project_audit(project_root: str | Path = ".", out_markdown: str | Path = "docs/project_audit.md") -> bool:
    """Scan webapp/ for Python modules and produce a first-pass audit report.

    Report includes per-file summaries, defs, imports, outgoing cross-module calls,
    and inbound references to local defs. Static, AST-only (no imports executed).
    """
    try:
        root = Path(project_root).resolve()
        modules = _scan_webapp_modules(root)
        def_index = _index_defs(modules)
        edges, inbound = _resolve_targets(modules, def_index)
        md = _render_audit_md(modules, def_index, edges, inbound, root)
        out = (root / out_markdown).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        return True
    except Exception as e:
        logger.error(f"[audit] Failed to generate project audit: {e}")
        return False

def generate_todos_index(project_root: str | Path = ".", out_markdown: str | Path = "docs/todos.md") -> bool:
    """Aggregate TODO/FIXME/WARN lines from webapp/ into a compact index.

    Writes a markdown file with a summary and per-module annotated lines.
    """
    try:
        root = Path(project_root).resolve()
        modules = _scan_webapp_modules(root)
        total = sum(len(m.get("todo_lines", [])) for m in modules)
        
        # Define priorities
        high_keywords = ['FIXME', 'BUG']
        medium_keywords = ['TODO', 'HACK', 'XXX']
        low_keywords = ['WARN', 'WARNING', 'NOTE']
        
        # Collect todos by priority
        priority_todos = {
            'high': [],
            'medium': [],
            'low': []
        }
        
        for m in modules:
            path = m.get("path", "")
            if path:
                try:
                    rel_path = Path(path).relative_to(root)
                    path = str(rel_path)
                except:
                    pass
            todos = m.get("todo_lines", [])
            for ln, keyword, cleaned_txt in todos:
                safe_txt = (cleaned_txt or "").replace("`", "\u2063`").replace("[", "\\[").replace("]", "\\]").replace('<', '&lt;').replace('>', '&gt;').replace('\t', ' ')
                # Fix emphasis spaces
                safe_txt = re.sub(r'(\*|_)\s+', r'\1', safe_txt)
                safe_txt = re.sub(r'\s+(\*|_)', r'\1', safe_txt)
                safe_txt = re.sub(r'(\*|_)\s*([^ *]*)\s*(\*|_)', r'\1\2\3', safe_txt)
                # Fix reversed links
                safe_txt = re.sub(r'\(([^)]+)\)\[:(\d+)\]', r'[\1][:\2]', safe_txt)
                safe_txt = safe_txt.replace('_', '*')  # Replace any remaining _ with *
                item = (path, ln, keyword, safe_txt)
                if keyword in high_keywords:
                    priority_todos['high'].append(item)
                elif keyword in medium_keywords:
                    priority_todos['medium'].append(item)
                elif keyword in low_keywords:
                    priority_todos['low'].append(item)
        
        lines: list[str] = []
        # Add YAML front matter for GitHub Pages
        lines.append("---")
        lines.append("layout: default")
        lines.append('title: "TODO/FIXME Index"')
        lines.append("---")
        lines.append("")
        lines.append("Index scope: TODO/FIXME annotations under `webapp/`.")
        lines.append("")
        lines.append(f"Total annotations: {total}")
        lines.append("")

        # Priority snapshot to emphasize hotspots
        lines.append("## Priority highlights")
        lines.append("")
        for priority, label in [('high', 'High Priority'), ('medium', 'Medium Priority'), ('low', 'Low Priority')]:
            todos = priority_todos[priority]
            if not todos:
                lines.append(f"- **{label}:** None outstanding.")
                continue
            file_counts: dict[str, int] = {}
            for path, *_ in todos:
                file_counts[path] = file_counts.get(path, 0) + 1
            top_files = ", ".join(
                f"{(p or 'unknown').replace('webapp/', '', 1)} ({cnt})" for p, cnt in sorted(file_counts.items(), key=lambda kv: -kv[1])[:3]
            )
            lines.append(f"- **{label}:** {len(todos)} items across {len(file_counts)} files. Focus: {top_files}.")
        lines.append("")
        
        # Output by priority
        for priority, label in [('high', 'High Priority'), ('medium', 'Medium Priority'), ('low', 'Low Priority')]:
            todos = priority_todos[priority]
            if not todos:
                continue
            lines.append(f"## {label}")
            # Group by file
            file_groups = {}
            for path, ln, keyword, safe_txt in todos:
                if path not in file_groups:
                    file_groups[path] = []
                file_groups[path].append((ln, keyword, safe_txt))
            for path in sorted(file_groups.keys()):
                display_path = (path or "unknown")
                display_path = display_path.replace("\\", "/")
                if display_path.startswith("webapp/"):
                    display_path = display_path[len("webapp/"):]
                if len(display_path) > 60:
                    parts = display_path.split("/")
                    tail = "/".join(parts[-4:]) if len(parts) >= 4 else display_path
                    display_path = f".../{tail}" if tail != display_path else tail
                heading = display_path.replace('_', r'\_')
                raw_id = re.sub(r'[^a-zA-Z0-9]+', '-', f"{path}-{priority}").strip('-').lower()
                if not raw_id:
                    raw_id = f"todo-{priority}-{abs(hash(path))}"
                elif len(raw_id) > 60:
                    suffix = abs(hash(path)) % 100000
                    raw_id = f"{raw_id[:50].rstrip('-')}-{suffix}"
                lines.append(f"### {heading} ({label})")
                lines.append(f"{{: #{raw_id} }}")
                lines.append("")
                for ln, keyword, safe_txt in file_groups[path]:
                    lines.append(f"- L{ln} *{keyword}*: {safe_txt}")
                lines.append("")
        
        md = _finalize_markdown_lines(lines)
        out = (root / out_markdown).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        return True
    except Exception as e:
        logger.error(f"[audit] Failed to generate todos index: {e}")
        return False

def generate_noise_override_suggestions(
    project_root: str | Path = ".",
    noise_json: str | Path = "output/noise_suggestions.json",
    out_markdown: str | Path = "docs/noise_override_suggestions.md",
    min_count: int = 3,
) -> bool:
    r"""Produce ready-to-paste override suggestions from aggregated noise snippets.

    Reads output/noise_suggestions.json and emits two code blocks:
    - CAMELOT_STATE_NOISE_OVERRIDES additions
    - CAMELOT_COUNTY_NOISE_OVERRIDES additions

    Each suggested regex uses re.escape(snippet) and is wrapped as ^\s*...\s*$ to avoid over-matching.
    Only includes snippets with frequency >= min_count.
    """
    try:
        root = Path(project_root).resolve()
        path = (root / noise_json).resolve()
        if not path.exists():
            logger.warning(f"[noise] No suggestions file found at {path}")
            return False
        try:
            data = orjson.loads(path.read_bytes())
        except Exception as e:
            logger.error(f"[noise] Failed to parse suggestions json: {e}")
            return False
        # Build state-level and county-level maps
        state_map: dict[str, dict[str, list[str]]] = {}
        county_map: dict[tuple[str, str], dict[str, list[str]]] = {}
        import re as _re
        for state, counties in (data or {}).items():
            if not isinstance(counties, dict):
                continue
            # Aggregate state-level snippets from __unknown__ county if present
            for county, cats in counties.items():
                if not isinstance(cats, dict):
                    continue
                for category, snippets in cats.items():
                    # Map pseudo categories to 'row'
                    cat = "row" if category in ("pseudo_party",) else category or "row"
                    if not isinstance(snippets, dict):
                        continue
                    for snippet, cnt in snippets.items():
                        try:
                            if int(cnt) < int(min_count):
                                continue
                        except Exception:
                            continue
                        patt = rf"^\s*{_re.escape(str(snippet))}\s*$"
                        if county == "__unknown__":
                            state_map.setdefault(state, {}).setdefault(cat, []).append(patt)
                        else:
                            county_map.setdefault((state, county), {}).setdefault(cat, []).append(patt)

        # Render markdown
        lines: list[str] = []
        # Add YAML front matter for GitHub Pages
        lines.append("---")
        lines.append("layout: default")
        lines.append('title: "Noise Override Suggestions"')
        lines.append("---")
        lines.append("")
        lines.append("Suggested Camelot noise overrides for Camelot parsers.")
        lines.append("")
        lines.append(f"Min count cutoff: {min_count}")
        lines.append("")
        # State-level
        if state_map:
            lines.append("")
            lines.append("## State-level additions")
            lines.append("")
            lines.append("```python")
            lines.append("CAMELOT_STATE_NOISE_OVERRIDES.update({")
            for state, cats in sorted(state_map.items()):
                # Use string concatenation to safely include literal '{' in f-string
                lines.append(f"    \"{state}\": " + "{")
                for cat, patterns in sorted(cats.items()):
                    lines.append(f"        \"{cat}\": [")
                    for p in sorted(set(patterns)):
                        lines.append(f"            r\"{p}\",")
                    lines.append("        ],")
                lines.append("    },")
            lines.append("})")
            lines.append("```")
            lines.append("")
        else:
            lines.append("")
            lines.append("## State-level additions")
            lines.append("None above threshold.")
            lines.append("")
        # County-level
        if county_map:
            lines.append("")
            lines.append("## County-level additions")
            lines.append("")
            lines.append("```python")
            lines.append("CAMELOT_COUNTY_NOISE_OVERRIDES.update({")
            for (state, county), cats in sorted(county_map.items()):
                # Use string concatenation to safely include literal '{' in f-string
                lines.append(f"    (\"{state}\", \"{county}\"): " + "{")
                for cat, patterns in sorted(cats.items()):
                    lines.append(f"        \"{cat}\": [")
                    for p in sorted(set(patterns)):
                        lines.append(f"            r\"{p}\",")
                    lines.append("        ],")
                lines.append("    },")
            lines.append("})")
            lines.append("```")
            lines.append("")
        else:
            lines.append("")
            lines.append("## County-level additions")
            lines.append("None above threshold.")
            lines.append("")

        md = _finalize_markdown_lines(lines)
        out = (root / out_markdown).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        return True
    except Exception as e:
        logger.error(f"[noise] Failed to generate override suggestions: {e}")
        return False

def generate_pipeline_map(project_root: str | Path = ".", out_markdown: str | Path = "docs/pipeline_map.md") -> bool:
    """Emit a comprehensive pipeline audit with graph, TOC, interactive elements, and detailed file contexts.

    Includes hyperlinks, collapsible sections, thorough connection maps, and automated audit for optimizations.
    """
    # Diagram rendering constants
    MAX_DIAGRAM_EDGES = 20  # Maximum number of edges to show in mermaid diagrams
    MAX_SUBGRAPH_NODES = 10  # Maximum nodes per cluster subgraph
    
    try:
        root = Path(project_root).resolve()
        modules = _scan_webapp_modules(root)
        def_index = _index_defs(modules)
        edges, _inbound = _resolve_targets(modules, def_index)
        def _to_mod(p: str | None) -> str:
            if not p:
                return "unknown"
            s = p.replace("\\", "/")
            i = s.find("webapp/")
            if i >= 0:
                s = s[i:]
            full = s.replace("/", ".").removesuffix(".py")
            parts = full.split(".")
            # Use last 1 part to shorten
            return ".".join(parts[-1:]) if len(parts) > 1 else full
        def _is_target_path(p: str) -> bool:
            if not p:
                return False
            p = p.replace("\\", "/")
            return "/webapp/parser/" in p  # Broadened to all parser files
        def _cluster_for_path(p: str) -> str:
            if not p:
                return "Other"
            p = p.replace("\\", "/")
            if "/webapp/parser/html_election_parser.py" in p:
                return "Entry"
            if "/webapp/parser/web_pipeline.py" in p:
                return "Pipeline"
            if "/webapp/parser/state_router.py" in p:
                return "Routing"
            if "/webapp/parser/handlers/states/" in p:
                return "State Handlers"
            if "/webapp/parser/handlers/formats/" in p:
                return "Format Handlers"
            if "/webapp/parser/handlers/shared/" in p:
                return "Shared Handlers"
            if "/webapp/parser/services/" in p:
                return "Services"
            if "/webapp/parser/utils/" in p:
                return "Utils"
            if "/webapp/parser/Context_Integration/" in p:
                return "Context Integration"
            if "/webapp/parser/health/" in p:
                return "Health"
            return "Other"
        # Build nodes and edges with clusters
        cluster_nodes: dict[str, set[str]] = {c: set() for c in ["Entry", "Pipeline", "Routing", "State Handlers", "Format Handlers", "Shared Handlers", "Services", "Utils", "Context Integration", "Health", "Other"]}
        cluster_edges: dict[tuple[str, str], int] = {}
        cluster_pair_counts: dict[tuple[str, str], int] = {}
        node_to_cluster: dict[str, str] = {}
        for e in edges:
            sp = e.get("src_path")
            dp = e.get("resolved_path")
            # Only include edges where both src and dst are resolved module paths
            if not sp or not dp:
                continue
            if not _is_target_path(sp) or not _is_target_path(dp):
                continue
            src = _to_mod(sp)
            dst = _to_mod(dp)
            if src == dst or dst == "unknown" or src == "unknown":
                continue
            sc = _cluster_for_path(sp)
            dc = _cluster_for_path(dp)
            cluster_nodes[sc].add(src)
            cluster_nodes[dc].add(dst)
            node_to_cluster[src] = sc
            node_to_cluster[dst] = dc
            cluster_edges[(src, dst)] = cluster_edges.get((src, dst), 0) + 1
            cluster_pair_counts[(sc, dc)] = cluster_pair_counts.get((sc, dc), 0) + 1
        # Top edges (limited for readability)
        top_edges = sorted(cluster_edges.items(), key=lambda kv: -kv[1])[:MAX_DIAGRAM_EDGES]
        
        # Collect all edge nodes
        edge_nodes: set[str] = set()
        for (src, dst), _ in top_edges:
            edge_nodes.add(src)
            edge_nodes.add(dst)
        
        lines: list[str] = []
        # Add YAML front matter for GitHub Pages
        lines.append("---")
        lines.append("layout: default")
        lines.append('title: "Comprehensive Pipeline Audit & Map"')
        lines.append("---")
        lines.append("")
        lines.append("Comprehensive pipeline audit for `webapp/parser/`.")
        lines.append("")
        lines.append("## 📋 Table of Contents")
        lines.append("- [Overview](#overview)")
        lines.append("- [Interactive Pipeline Graph](#interactive-pipeline-graph)")
        lines.append("- [File Connection Map](#file-connection-map)")
        lines.append("- [Detailed Module Contexts](#detailed-module-contexts)")
        lines.append("")
        lines.append("## Overview")
        total_modules = sum(len(nodes) for nodes in cluster_nodes.values())
        total_edges = len(cluster_edges)
        lines.append(f"- **Total Modules Audited:** {total_modules}")
        lines.append(f"- **Total Connections:** {total_edges}")
        lines.append("- **Clusters:** Entry, Pipeline, Routing, State Handlers, Format Handlers, Shared Handlers, Services, Utils, Context Integration, Health")
        lines.append("- **Audit Scope:** All `webapp/parser/` files with full context, imports, dependencies, and optimization insights.")
        lines.append("")
        lines.append("## Interactive Pipeline Graph")
        lines.append("")
        lines.append("```mermaid")
        lines.append("graph TD")
        # Subgraphs - prioritize nodes that appear in edges
        for cname in ["Entry", "Pipeline", "Routing", "State Handlers", "Format Handlers", "Shared Handlers", "Services", "Utils", "Context Integration", "Health"]:
            all_cluster_nodes = cluster_nodes.get(cname, set())
            # Prioritize edge nodes in this cluster
            priority_nodes = [n for n in edge_nodes if node_to_cluster.get(n) == cname]
            other_nodes = [n for n in all_cluster_nodes if n not in priority_nodes]
            # Include all priority nodes first, then fill up to limit
            max_nodes = max(MAX_SUBGRAPH_NODES, len(priority_nodes))
            nodes = sorted(priority_nodes) + sorted(other_nodes)
            nodes = nodes[:max_nodes]
            if not nodes:
                continue
            lines.append(f"  subgraph {cname.replace(' ', '_')}[\"{cname}\"]")
            for n in nodes:
                lines.append(f"    {n.replace('.', '_')}[\"{n}\"]")
            lines.append("  end")
        # Edges
        for (src, dst), cnt in top_edges:
            lines.append(f"  {src.replace('.', '_')} -->|{cnt}| {dst.replace('.', '_')}")
        if not top_edges:
            lines.append("  A[no data] --> B[no data]")
        lines.append("```")
        lines.append("")
        lines.append("**✨ Legend:** Colors indicate module categories with metallic accents. Click nodes for details below.")
        lines.append("")
        # Highlight major connections to emphasize cross-cutting concerns
        lines.append("## Connection Highlights")
        lines.append("Key integration points across major parser aspects to simplify tracking relevance.")
        lines.append("")
        lines.append("### Top Module Links")
        lines.append("")
        if top_edges:
            for (src, dst), cnt in top_edges[:10]:
                sc = node_to_cluster.get(src, "Other")
                dc = node_to_cluster.get(dst, "Other")
                lines.append(
                    f"- `{src}` → `{dst}` ({cnt} refs, {sc} → {dc}) — review `{dst}` whenever `{src}` changes."
                )
        else:
            lines.append("- No module-level connections detected.")
        lines.append("")
        if cluster_pair_counts:
            lines.append("### Cluster Flow Summary")
            lines.append("")
            for (sc, dc), cnt in sorted(cluster_pair_counts.items(), key=lambda kv: -kv[1])[:10]:
                relation = "intra-cluster" if sc == dc else "cross-cluster"
                lines.append(f"- {sc} → {dc}: {cnt} edges ({relation} flow to monitor.)")
            lines.append("")
        else:
            lines.append("### Cluster Flow Summary")
            lines.append("")
            lines.append("- Not enough cluster cross-links to summarize.")
            lines.append("")
        # File Connection Map
        lines.append("## File Connection Map")
        lines.append("Detailed import/export relationships and dependencies.")
        lines.append("")
        # Build reverse dependencies
        reverse_deps: dict[str, set[str]] = {}
        for m in modules:
            path = m.get("path", "")
            if not _is_target_path(path):
                continue
            for imp in m.get("imports", []):
                imp_mod = imp.get("module", "")
                if imp_mod.startswith("webapp.parser."):
                    imp_mod = imp_mod.replace("webapp.parser.", "").replace(".", "/") + ".py"
                    if imp_mod in [p.replace("\\", "/").replace("webapp/parser/", "") for p in [m["path"] for m in modules]]:
                        rev_key = imp_mod
                        src_key = path.replace("\\", "/").replace("webapp/parser/", "")
                        if rev_key not in reverse_deps:
                            reverse_deps[rev_key] = set()
                        reverse_deps[rev_key].add(src_key)
        for mod, deps in sorted(reverse_deps.items()):
            lines.append(f"- **{mod}** is imported by: {', '.join(sorted(deps))}")
        lines.append("")
        # Add detailed module summaries with collapsible sections
        lines.append("## Detailed Module Contexts")
        lines.append("Click to expand each module for full audit details.")
        lines.append("")
        pipeline_modules = [m for m in modules if _is_target_path(m.get("path", ""))]
        for m in sorted(pipeline_modules, key=lambda x: x.get("path", "")):
            path = m.get("path", "")
            if path:
                try:
                    rel_path = Path(path).relative_to(root)
                    path_str = str(rel_path).replace("\\", "/")
                    link = f"../{path_str}"
                except:
                    path_str = path.replace("\\", "/")
                    link = path_str
            else:
                path_str = "unknown"
                link = "#"
            display_path = path_str.replace("webapp/parser/", "") or path_str
            mod_name = display_path.replace("/", "_").replace(".py", "")
            # Emit a normal Markdown heading (avoid inline HTML to satisfy
            # markdownlint MD033). Rely on the rendered heading id that
            # Jekyll/kramdown will generate for linking.
            md_heading = display_path.replace('_', r'\_')
            # Create a stable id from the module path: lowercase, replace
            # non-alphanum with hyphens. Use kramdown-style header id
            # attribute (e.g. "### Title {#my-id}") which is markdown,
            # not inline HTML (avoids MD033).
            raw_id = re.sub(r'[^a-zA-Z0-9]+', '-', path_str).strip('-').lower()
            if not raw_id:
                raw_id = f"module-{abs(hash(path_str))}"
            lines.append(f"### {md_heading} {{#{raw_id}}}")
            lines.append("")
            if m.get("doc"):
                safe_doc = m['doc'].splitlines()[0].replace('_', '*')
                lines.append(f"> {safe_doc}")
            lines.append("")
            # Key functions and classes
            defs = [d for d in m.get("defs", []) if d.get("type") in ("function", "async_function", "class")]
            if defs:
                lines.append(f"#### 🔧 Key Functions & Classes ({mod_name})")
                lines.append("")
                for d in defs[:25]:  # Increased
                    lines.append(f"- `{d['name']}` ({d.get('type', 'def')}, line {d.get('lineno', '?')})")
                lines.append("")
            # Imports
            imports = m.get("imports", [])
            if imports:
                lines.append(f"#### 📦 Key Imports ({mod_name})")
                lines.append("")
                for imp in imports[:20]:
                    mod = imp.get('module', '')
                    lines.append(f"- `{mod}`")
                lines.append("")
            # Dependencies
            deps = reverse_deps.get(path_str.replace("webapp/parser/", ""), set())
            if deps:
                lines.append(f"#### 🔗 Reverse Dependencies ({mod_name})")
                lines.append("")
                lines.append(f"Imported by: {', '.join(sorted(deps))}")
                lines.append("")
            # Top comments
            if m.get("top_comment"):
                lines.append(f"#### 💬 Top-of-file Comments ({mod_name})")
                lines.append("")
                lines.append("```python")
                for ln in m["top_comment"].splitlines()[:25]:  # Increased
                    ln = ln.replace('\t', '    ').replace('<', '&lt;').replace('>', '&gt;').replace('*', '\\*').replace('_', '\\_')
                    lines.append(ln)
                lines.append("```")
                lines.append("")
            # TODOs
            todos = m.get("todo_lines", [])
            if todos:
                lines.append(f"#### ⚠️ TODO/FIXME/WARN ({mod_name})")
                lines.append("")
                for ln, keyword, cleaned_txt in todos[:20]:  # Increased
                    safe_txt = (cleaned_txt or "").replace("`", "\u2063`").replace("[", "\\[").replace("]", "\\]").replace('<', '&lt;').replace('>', '&gt;').replace('\t', ' ')
                    safe_txt = re.sub(r'(\*|_)\s+', r'\1', safe_txt)
                    safe_txt = re.sub(r'\s+(\*|_)', r'\1', safe_txt)
                    safe_txt = re.sub(r'(\*|_)\s*([^ *]*)\s*(\*|_)', r'\1\2\3', safe_txt)
                    safe_txt = re.sub(r'\(([^)]+)\)\[:(\d+)\]', r'[\1][:\2]', safe_txt)
                    safe_txt = safe_txt.replace('_', '*')  # Replace any remaining _ with *
                    lines.append(f"- L{ln} **{keyword}**: {safe_txt}")
                lines.append("")
        # Post-process lines for markdownlint compliance
        def wrap_line(line: str, max_len: int = 80) -> list[str]:
            if len(line) <= max_len:
                return [line]
            words = line.split()
            wrapped = []
            current = ""
            for word in words:
                if len(current) + len(word) + 1 <= max_len:
                    current += (" " + word) if current else word
                else:
                    if current:
                        wrapped.append(current)
                    current = word
            if current:
                wrapped.append(current)
            return wrapped
        new_lines = []
        for line in lines:
            if line.startswith("#"):
                # Ensure blank line before heading
                if new_lines and new_lines[-1] != "":
                    new_lines.append("")
                new_lines.append(line)  # Don't wrap headings
                new_lines.append("")  # Blank after heading
            elif line.startswith("- "):
                # List item, wrap if long
                wrapped = wrap_line(line, 78)  # Leave space for -
                new_lines.extend(wrapped)
            else:
                new_lines.extend(wrap_line(line))
        # Remove multiple consecutive blanks
        final_lines = []
        prev_blank = False
        for line in new_lines:
            if line == "":
                if not prev_blank:
                    final_lines.append(line)
                prev_blank = True
            else:
                final_lines.append(line)
                prev_blank = False
        out = (root / out_markdown).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(final_lines).rstrip() + "\n", encoding="utf-8")
        return True
    except Exception as e:
        logger.error(f"[pipeline] Failed to generate pipeline map: {e}")
        return False


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m webapp.parser.utils.shared_logic <command>")
        print("Commands: generate_project_audit, generate_todos_index, generate_noise_override_suggestions, generate_pipeline_map")
        sys.exit(1)
    command = sys.argv[1]
    if command == "generate_project_audit":
        success = generate_project_audit()
        sys.exit(0 if success else 1)
    elif command == "generate_todos_index":
        success = generate_todos_index()
        sys.exit(0 if success else 1)
    elif command == "generate_noise_override_suggestions":
        success = generate_noise_override_suggestions()
        sys.exit(0 if success else 1)
    elif command == "generate_pipeline_map":
        success = generate_pipeline_map()
        sys.exit(0 if success else 1)
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)