from __future__ import annotations
# ==============================================================
# 🗳️ Smart Elections: Universal PDF Election Results Parser
# ==============================================================
import os
import re
import csv
import time
import math
import platform
import shutil
import importlib
import hashlib
import atexit
import gc
import tempfile
from typing import Any
from collections import Counter, OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from ...Context_Integration.location_inference import infer_county_from_lines
from ...config import (
    ENABLE_OCR,
    ENABLE_PARALLEL,
    OUTPUT_DIR,
    # OCR Tuning Parameters (centralized in config.py)
    OCR_CONFIDENCE_THRESHOLD,
    OCR_MIN_ALPHA_SIGNAL,
    OCR_AVG_CONF_ACCEPT,
    OCR_DPI_MIN,
    OCR_DPI_MAX,
    OCR_DPI_STEP,
    OCR_PSM_LIST,
    OCR_OEM_LIST,
    OCR_PREPROCESS_VARIANTS,
    OCR_SAMPLE_BUDGET,
    OCR_MAX_RUNS,
    OCR_ORIENTATION_THRESHOLD,
    OCR_DENSE_LINE_THRESHOLD,
    OCR_TABLE_SIGNAL_MIN_COLS,
    OCR_TABLE_SIGNAL_MIN_ROWS,
    OCR_MARKUP_HTML_TAG_RATIO,
    OCR_DEBUG_SAVE_IMAGES,
    OCR_FAST_MODE_DPI_LIMIT,
    OCR_FAST_MODE_SAMPLE_LIMIT,
    PDF_FAST_MODE,
    PDF_PROBE_MAX_PAGES,
)
import html
# Optional flags/paths; provide safe defaults if missing
try:
    from ...config import ENABLE_OCR_FORCE, OCR_DEBUG_DIR
except Exception:
    ENABLE_OCR_FORCE = False
    OCR_DEBUG_DIR = os.path.join(OUTPUT_DIR, "ocr_debug")
    os.makedirs(OCR_DEBUG_DIR, exist_ok=True)

try:
    from ...config import POPPLER_PATH as CONFIG_POPPLER_PATH
except Exception:
    CONFIG_POPPLER_PATH = None
try:
    from ...config import TESSERACT_CMD as CONFIG_TESSERACT_CMD
except Exception:
    CONFIG_TESSERACT_CMD = None

try:
    from ...config import ENABLE_CAMELOT
except Exception:
    ENABLE_CAMELOT = True

try:
    from ...config import ENABLE_SANITIZE_DEBUG_LOG, SANITIZE_LOGGING_LIMIT
except Exception:
    ENABLE_SANITIZE_DEBUG_LOG = False
    SANITIZE_LOGGING_LIMIT = 200

from ...utils.camelot_utils import (
    attempt_camelot_extraction,
    hybrid_fill_camelot,
)
from ...utils.pdf_table_utils import (
    best_title_match_idx as utils_best_title_match_idx,
    coerce_vote_value_for_reconstruction as utils_coerce_vote_value_for_reconstruction,
    compute_header_richness as utils_compute_header_richness,
    compute_numeric_fill as utils_compute_numeric_fill,
    evaluate_table_candidate_quality as utils_evaluate_table_candidate_quality,
    extract_candidate_totals_from_lines as utils_extract_candidate_totals_from_lines,
    extract_contest_block as utils_extract_contest_block,
    extract_party_lookup_from_lines as utils_extract_party_lookup_from_lines,
    extract_table_by_whitespace as utils_extract_table_by_whitespace,
    find_best_header_match as utils_find_best_header_match,
    find_header_line as utils_find_header_line,
    detect_district_heading as utils_detect_district_heading,
    header_signature as utils_header_signature,
    is_bad_header_line as utils_is_bad_header_line,
    is_numeric_like as utils_is_numeric_like,
    looks_like_candidate_header as utils_looks_like_candidate_header,
    matches_anchor_header as utils_matches_anchor_header,
    merge_camelot_with_text as utils_merge_camelot_with_text,
    normalize_anchor_value as utils_normalize_anchor_value,
    normalize_numeric_token as utils_normalize_numeric_token,
    normalize_text_token as utils_normalize_text_token,
    parse_candidate_header_with_party as utils_parse_candidate_header_with_party,
    parse_candidate_line as utils_parse_candidate_line,
    reconstruct_columnar_block as utils_reconstruct_columnar_block,
    consume_reconstruction_debug_events as utils_consume_reconstruction_debug_events,
    split_ws_blocks as utils_split_ws_blocks,
    table_looks_bad as utils_table_looks_bad,
    token_set as utils_token_set,
)
from ...utils.logger_singleton import logger
from ...Context_Integration.Context_Library.constants import (
    LOCATION_KEYWORDS,
    CANDIDATE_KEYWORDS,
    BALLOT_TYPES,
    PARTY_KEYWORDS,
    TOTAL_KEYWORDS,
    MISC_FOOTER_KEYWORDS,
    CONTEST_KEYWORDS,
    CONTEST_TITLE_SKIP_PHRASES,
    CONTEST_HEADER_KEYWORDS,
    CONTEST_HEADER_PREFERENCE,
    KNOWN_STATE_TO_COUNTY_MAP,
    normalize_party_label,
)
from ...utils.table_core import harmonize_headers_and_data, robust_table_extraction
from ...utils.location_helpers import (
    attach_precinct_column,
    collect_location_headers,
    is_strict_location_header,
)
from ...Context_Integration.librarian import parse_filename_for_location
from ...utils.contest_detection import (
    CONTEST_PATTERN as _CONTEST_RX,
    detect_contest_titles_from_text,
)
from ...utils.contest_selector import select_contest_auto_first, select_contest_noninteractive
from ...utils.table_builder import build_table_noninteractive
from ...utils.output_utils import finalize_election_output
from ...utils.shared_logic import (
    format_county_label,
    format_state_label,
    normalize_county_name,
    normalize_state_name,
    safe_get,
    safe_is_set,
    safe_slug,
)
from ...utils.pivot import expand_single_rawjson_row, transform_wide_to_smart_standard
from ...Context_Integration.context_coordinator import dynamic_state_county_detection
from ...utils.header_utils import normalize_table_headers
try:
    from ...parse_trace import record_parse_observation as _record_parse_observation
    from ...profiling.pdf_structure_profiler import (
        StructureObservationPhase as _StructureObservationPhase,
    )
except Exception:
    _record_parse_observation = None
    _StructureObservationPhase = None

# Added optional tuning flags (safe defaults if not in config)
try:
    from ...config import CAMELOT_MIN_SCORE, CAMELOT_HYBRID_FILL, CAMELOT_MERGE_COMPAT
except Exception:
    CAMELOT_MIN_SCORE = 0.9
    CAMELOT_HYBRID_FILL = True
    CAMELOT_MERGE_COMPAT = True

# Optional Camelot import
try:
    import camelot
    _CAMELOT_AVAILABLE = True
except Exception:
    _CAMELOT_AVAILABLE = False

_MIN_PYMUPDF_VERSION = (1, 26, 5)
_FITZ_MODULE = None

_SANITIZE_CACHE_VERSION = "v2"
_SANITIZE_CACHE_LIMIT = 128
_SANITIZE_CACHE_MIN_CONFIDENCE = 0.20
_SANITIZE_VERTICAL_JOIN_LIMIT = 50
_SANITIZE_CACHE: OrderedDict[str, tuple[str, float]] = OrderedDict()
_POPPLER_WARNING_SHOWN = False
_PDF2IMAGE_DISABLED_REASON: str | None = None
_PAGE_ORIENTATION_CACHE: dict[str, dict[int, int]] = {}
_PAGE_ORIENTATION_DEFAULT: dict[str, int] = {}
_PAGE_ORIENTATION_LOGGED: set[str] = set()
_PAGE_ORIENTATION_APPLIED: set[tuple[str, int, int]] = set()

# PDF resource cleanup tracking (prevent Windows file lock errors)
_PDF_IMAGE_REFS: list[Image.Image] = []
_PDF_TEMP_DIRS: set[str] = set()
_PDF_CLEANUP_REGISTERED = False


def _env_truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


_CONTEST_PROMPTS_ENABLED = _env_truthy(os.getenv("SMART_ELECTIONS_ENABLE_CONTEST_PROMPTS"), False)

# OCR/runtime guardrails — tuned for typical 5-150 page statements.
_OCR_FOCUS_WINDOW_EXPAND = 2
_OCR_SAMPLE_PAGE_TARGET = 6
_OCR_CONTEST_PROBE_MIN_PAGES = 12
_OCR_CONTEST_PROBE_MAX_PAGES = 90
_OCR_CONTEST_PROBE_STRIDE = 5
_OCR_CONTEST_PROBE_DPI = 220
_OCR_CONTEST_PROBE_MAX_HITS = 6
_OCR_FULLDOC_BATCH_PAGES = 8
_OCR_FULLDOC_MAX_PAGES = 240
_LAYOUT_SCAN_PAGE_LIMIT = 28
_STATEMENT_SCAN_PAGE_LIMIT = 20


class PDFParseCancelled(RuntimeError):
    """Raised when a cooperative cancel flag requests early exit."""


def _cleanup_pdf_resources() -> None:
    """
    Safe cleanup of PDF resources before exit.
    Closes PIL Image objects and removes temp directories to prevent Windows file locks.
    """
    global _PDF_IMAGE_REFS, _PDF_TEMP_DIRS
    
    # Close all PIL Image objects (releases pdfium handles)
    for img in _PDF_IMAGE_REFS:
        try:
            if hasattr(img, 'close'):
                img.close()
        except Exception:
            pass
    _PDF_IMAGE_REFS.clear()
    
    # Force garbage collection to release file handles
    gc.collect()
    
    # Small delay for Windows file system to release locks
    time.sleep(0.1)
    
    # Clean up temp directories with retry logic
    for temp_dir in list(_PDF_TEMP_DIRS):
        if not os.path.exists(temp_dir):
            continue
        for attempt in range(3):
            try:
                shutil.rmtree(temp_dir, ignore_errors=False)
                break
            except PermissionError:
                if attempt < 2:
                    time.sleep(0.2)
                    gc.collect()
                else:
                    # Last resort: mark for deletion on reboot (Windows only)
                    try:
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    except Exception:
                        pass
            except Exception:
                break
    _PDF_TEMP_DIRS.clear()


def _register_pdf_cleanup() -> None:
    """Register PDF cleanup handler (once only)."""
    global _PDF_CLEANUP_REGISTERED
    if not _PDF_CLEANUP_REGISTERED:
        atexit.register(_cleanup_pdf_resources)
        _PDF_CLEANUP_REGISTERED = True



def _sanitize_cache_get(key: str) -> str | None:
    entry = _SANITIZE_CACHE.get(key)
    if not entry:
        return None
    value, confidence = entry
    if confidence < _SANITIZE_CACHE_MIN_CONFIDENCE:
        return None
    _SANITIZE_CACHE.move_to_end(key)
    return value


def _sanitize_cache_set(key: str, value: str, confidence: float) -> None:
    if not key:
        return
    if confidence < _SANITIZE_CACHE_MIN_CONFIDENCE:
        return
    _SANITIZE_CACHE[key] = (value, confidence)
    _SANITIZE_CACHE.move_to_end(key)
    while len(_SANITIZE_CACHE) > _SANITIZE_CACHE_LIMIT:
        _SANITIZE_CACHE.popitem(last=False)


def _normalize_angle(angle: float) -> float:
    while angle <= -180.0:
        angle += 360.0
    while angle > 180.0:
        angle -= 360.0
    return angle


def _quantize_angle(angle: float) -> int:
    angle = _normalize_angle(angle)
    quantized = int(round(angle / 45.0)) * 45
    if quantized <= -180:
        quantized += 360
    if quantized > 180:
        quantized -= 360
    return quantized


def _collect_page_orientation(page) -> tuple[int | None, float, int]:
    try:
        raw = page.get_text("rawdict") or {}
        page_rect = getattr(page, "rect", None) or getattr(page, "bound", lambda: None)()
        page_height = float(getattr(page_rect, "height", 0.0) or 0.0)
    except Exception:
        return None, 0.0, 0

    blocks = raw.get("blocks", [])
    body_weights: Counter[int] = Counter()
    header_weights: Counter[int] = Counter()
    total_weight = 0.0
    sample_votes = 0

    header_cutoff = page_height * 0.22 if page_height else None

    for block in blocks:
        if not isinstance(block, dict):
            continue
        lines = block.get("lines")
        if not isinstance(lines, list):
            continue
        for line in lines:
            spans = line.get("spans") if isinstance(line, dict) else None
            if not isinstance(spans, list):
                continue
            for span in spans:
                if not isinstance(span, dict):
                    continue
                direction = span.get("dir")
                if not direction or not isinstance(direction, (tuple, list)) or len(direction) != 2:
                    continue
                dx, dy = direction
                if dx == 0 and dy == 0:
                    continue
                bbox = span.get("bbox")
                if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                    continue
                x0, y0, x1, y1 = bbox
                span_width = abs(float(x1) - float(x0))
                span_height = abs(float(y1) - float(y0))
                if span_width <= 0 and span_height <= 0:
                    continue
                try:
                    angle = math.degrees(math.atan2(dy, dx))
                except Exception:
                    continue
                angle = _quantize_angle(angle)
                region_is_header = bool(header_cutoff and min(y0, y1) <= header_cutoff)
                # Ignore narrow header glyphs (vertical titles) to avoid mis-rotations.
                if region_is_header and span_width < span_height * 0.6:
                    continue
                weight = max(1.0, min(2000.0, (span_width * 0.6) + (span_height * 0.4)))
                sample_votes += 1
                total_weight += weight
                if region_is_header:
                    header_weights[angle] += weight
                else:
                    body_weights[angle] += weight

    if not (body_weights or header_weights):
        return None, 0.0, sample_votes

    def _dominant(counter: Counter[int]) -> tuple[int | None, float]:
        if not counter:
            return None, 0.0
        angle, weight = counter.most_common(1)[0]
        coverage = weight / max(1.0, total_weight)
        return angle, coverage

    body_angle, body_cov = _dominant(body_weights)
    header_angle, header_cov = _dominant(header_weights)

    if body_angle is not None and (body_cov >= 0.45 or body_cov >= header_cov):
        return body_angle, body_cov, sample_votes
    if header_angle is not None:
        return header_angle, max(body_cov, header_cov), sample_votes
    return None, 0.0, sample_votes


def _get_page_orientation_map(pdf_path: str, session_id=None) -> tuple[dict[int, int], int]:
    """Analyze up to 120 pages to infer per-page and default rotations."""
    abs_path = os.path.abspath(pdf_path)
    cached = _PAGE_ORIENTATION_CACHE.get(abs_path)
    if cached is not None:
        return cached, _PAGE_ORIENTATION_DEFAULT.get(abs_path, 0)

    orientation_map: dict[int, int] = {}
    orientation_counts: Counter[int] = Counter()
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        logger.debug({
            "level": "DEBUG",
            "type": "handler",
            "message": f"[DEBUG] Page orientation analysis skipped: {exc}",
            "session_id": session_id,
        })
        _PAGE_ORIENTATION_CACHE[abs_path] = orientation_map
        _PAGE_ORIENTATION_DEFAULT[abs_path] = 0
        return orientation_map, 0

    try:
        total_pages = len(doc)
        max_pages = min(total_pages, 120)
        for page_index in range(max_pages):
            page = doc[page_index]
            angle, coverage, vote_count = _collect_page_orientation(page)
            if angle is None or vote_count < 5:
                continue
            if abs(angle) < 45:
                continue
            if coverage < 0.35:
                continue
            orientation_map[page_index] = angle
            orientation_counts[angle] += 1
    finally:
        try:
            doc.close()
        except Exception:
            pass

    default_angle = 0
    if orientation_counts:
        dominant_angle, count = orientation_counts.most_common(1)[0]
        total_votes = sum(orientation_counts.values())
        if count / max(1, total_votes) >= 0.6:
            default_angle = dominant_angle

    _PAGE_ORIENTATION_CACHE[abs_path] = orientation_map
    _PAGE_ORIENTATION_DEFAULT[abs_path] = default_angle
    if abs_path not in _PAGE_ORIENTATION_LOGGED:
        if orientation_counts or default_angle:
            logged_angles = sorted(set(orientation_counts.keys()) | ({default_angle} if default_angle else set()))
            logger.info({
                "level": "INFO",
                "type": "handler",
                "message": f"[INFO] Page orientation hints detected: {logged_angles} (default={default_angle})",
                "session_id": session_id,
            })
        _PAGE_ORIENTATION_LOGGED.add(abs_path)
    return orientation_map, default_angle


def _log_orientation_application(pdf_path: str, page_index: int, angle: int, session_id=None) -> None:
    key = (pdf_path, page_index, angle)
    if key in _PAGE_ORIENTATION_APPLIED:
        return
    _PAGE_ORIENTATION_APPLIED.add(key)
    logger.info({
        "level": "INFO",
        "type": "handler",
        "message": f"[INFO] Applied rotation={angle}deg for page={page_index}",
        "session_id": session_id,
    })


def _apply_page_orientation(
    image: Image.Image,
    page_index: int,
    pdf_path: str,
    orientation_map: dict[int, int] | None,
    default_angle: int,
    *,
    session_id=None,
):
    angle = 0
    if orientation_map and page_index in orientation_map:
        angle = orientation_map.get(page_index, 0) or 0
    elif default_angle:
        angle = default_angle
    angle = _quantize_angle(angle)
    if angle in {0, 360, -360}:
        return image
    try:
        rotated = image.rotate(-angle, expand=True)
    except Exception as exc:
        logger.debug({
            "level": "DEBUG",
            "type": "handler",
            "message": f"[DEBUG] Failed to apply rotation={angle}deg on page={page_index}: {exc}",
            "session_id": session_id,
        })
        return image
    _log_orientation_application(pdf_path, page_index, angle, session_id=session_id)
    return rotated


def _expand_focus_windows(
    page_hits: list[int] | None,
    page_count: int | None,
    *,
    expand: int | None = None,
) -> list[tuple[int, int]]:
    if not page_hits or not page_count:
        return []
    expand = expand if expand is not None else _OCR_FOCUS_WINDOW_EXPAND
    normalized = []
    for hit in sorted({idx for idx in page_hits if isinstance(idx, int)}):
        start = max(0, hit - expand)
        end = min(page_count, hit + expand + 1)
        if start < end:
            normalized.append((start, end))
    if not normalized:
        return []
    merged: list[list[int]] = []
    for start, end in normalized:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _normalize_contest_key(value: str | None) -> str:
    if not value:
        return ""
    lowered = re.sub(r"[^a-z0-9 ]+", " ", value.lower())
    return re.sub(r"\s+", " ", lowered).strip()


def _contest_title_tokens(value: str | None) -> set[str]:
    if not value:
        return set()
    return {tok for tok in re.findall(r"[a-z0-9]+", value.lower()) if len(tok) >= 2}


def _ensure_not_cancelled(cancel_flag, session_id, stage: str) -> None:
    """Raise PDFParseCancelled if the cooperative flag indicates an abort."""
    if cancel_flag is None:
        return

    cancelled = False
    reason = None

    if isinstance(cancel_flag, bool):
        cancelled = cancel_flag
    elif safe_is_set(cancel_flag):
        cancelled = True
    elif callable(cancel_flag):
        try:
            cancelled = bool(cancel_flag())
        except Exception:
            cancelled = False
    elif isinstance(cancel_flag, dict):
        for key in ("cancelled", "is_cancelled", "value", "flag"):
            if bool(cancel_flag.get(key)):
                cancelled = True
                break
        reason = cancel_flag.get("reason") or cancel_flag.get("message")
    else:
        for attr in ("cancelled", "is_cancelled", "value"):
            try:
                attr_value = getattr(cancel_flag, attr)
            except Exception:
                continue
            if callable(attr_value):
                try:
                    attr_value = attr_value()
                except Exception:
                    continue
            if isinstance(attr_value, bool):
                cancelled = attr_value
            else:
                cancelled = bool(attr_value)
            if cancelled:
                break
        if not reason and hasattr(cancel_flag, "reason"):
            try:
                reason = getattr(cancel_flag, "reason")
            except Exception:
                reason = None

    if not cancelled:
        return

    message = str(stage or "pdf_handler")
    if reason:
        message = f"{message} :: {reason}"
    logger.info({
        "level": "INFO",
        "type": "handler",
        "message": f"[INFO] PDF parsing cancelled at {stage}: {reason or 'user-request'}",
        "session_id": session_id,
    })
    raise PDFParseCancelled(message)


def _cancelled_result(
    pdf_path: str | None,
    metadata_seed: dict | None,
    reason: str | None,
    *,
    session_id=None,
):
    metadata = {
        "handler": "pdf_handler",
        "cancelled": True,
        "cancel_reason": reason or "User cancelled",
    }
    if pdf_path:
        metadata.setdefault("input_file", os.path.basename(pdf_path))
    if isinstance(metadata_seed, dict):
        metadata.update(metadata_seed)
    logger.info({
        "level": "INFO",
        "type": "handler",
        "message": f"[INFO] Returning cancel result for {metadata.get('input_file')} ({metadata.get('cancel_reason')})",
        "session_id": session_id,
    })
    return None, None, None, metadata


def _estimate_ocr_time_budgets(page_total: int | None) -> tuple[int, int]:
    """Return (sample_budget_seconds, stream_budget_seconds) based on page count."""
    effective_pages = 1
    if isinstance(page_total, int) and page_total > 0:
        effective_pages = min(page_total, _OCR_FULLDOC_MAX_PAGES)

    sample_budget = int(min(210, max(45, 25 + math.sqrt(effective_pages) * 18)))
    stream_budget = int(min(900, max(180, 60 + effective_pages * 4)))
    return sample_budget, stream_budget


def _refine_focus_windows_for_contest(
    selected_title: str | None,
    contest_probe_info: dict[str, Any] | None,
    page_count: int | None,
    *,
    expand: int | None = None,
    min_score: float = 0.35,
) -> list[tuple[int, int]] | None:
    if not selected_title or not contest_probe_info or not page_count:
        return None
    probe_pages = contest_probe_info.get("pages") or []
    if not isinstance(probe_pages, list):
        return None
    target_tokens = _contest_title_tokens(selected_title)
    if not target_tokens:
        target_norm = _normalize_contest_key(selected_title)
    else:
        target_norm = ""
    focused_hits: list[int] = []
    for entry in probe_pages:
        page_idx = safe_get(entry, "page")
        if not isinstance(page_idx, int):
            continue
        entry_titles = safe_get(entry, "titles") or []
        entry_lines = safe_get(entry, "lines") or []
        combined_text = " ".join([*entry_titles, *entry_lines])
        tokens = _contest_title_tokens(combined_text)
        score = 0.0
        if target_tokens and tokens:
            intersection = len(target_tokens & tokens)
            union = len(target_tokens | tokens) or 1
            score = intersection / union
        elif target_norm:
            combined_norm = _normalize_contest_key(combined_text)
            if combined_norm and target_norm:
                score = 1.0 if target_norm in combined_norm else 0.0
        if score >= min_score:
            focused_hits.append(page_idx)
    if not focused_hits:
        return None
    return _expand_focus_windows(focused_hits, page_count, expand=expand)


def _focus_windows_from_line_records(
    line_records: list[dict],
    candidate_titles: list[str],
    page_count: int | None,
    *,
    expand: int | None = None,
    limit_windows: int = 12,
    min_score: float = 0.55,
) -> list[tuple[int, int]] | None:
    """Derive focus windows by locating contest titles in sanitized line records."""
    if not line_records or not candidate_titles or not isinstance(page_count, int) or page_count <= 0:
        return None

    title_tokens_map: list[tuple[str, set[str], str]] = []
    for title in candidate_titles:
        if not title:
            continue
        tokens = _contest_title_tokens(title)
        normalized = _normalize_contest_key(title)
        if not tokens and not normalized:
            continue
        title_tokens_map.append((title, tokens, normalized))
    if not title_tokens_map:
        return None

    hits: list[int] = []
    for record in line_records:
        page = record.get("page")
        if not isinstance(page, int) or page < 0:
            continue
        text = record.get("text") or ""
        if not text:
            continue
        text_tokens = _contest_title_tokens(text)
        text_norm = "" if text_tokens else _normalize_contest_key(text)
        for _title, tokens, norm in title_tokens_map:
            score = 0.0
            if tokens and text_tokens:
                intersection = len(tokens & text_tokens)
                union = len(tokens | text_tokens) or 1
                score = intersection / union
            elif norm and text_norm:
                score = 1.0 if norm and norm in text_norm else 0.0
            if score >= min_score:
                hits.append(page)
                break

    if not hits:
        return None

    windows = _expand_focus_windows(hits, page_count, expand=expand)
    if not windows:
        return None

    windows = windows[:limit_windows]
    return windows


def _merge_focus_windows(
    existing: list[tuple[int, int]] | None,
    extra: list[tuple[int, int]] | None,
) -> list[tuple[int, int]] | None:
    if not existing and not extra:
        return None
    combined = []
    for bucket in (existing, extra):
        for window in bucket or []:
            if not isinstance(window, tuple) or len(window) != 2:
                continue
            start, end = window
            if start is None or end is None:
                continue
            combined.append((int(start), int(end)))
    if not combined:
        return None
    combined.sort()
    merged: list[list[int]] = []
    for start, end in combined:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _autopick_contest_from_probe(
    pdf_path: str,
    contest_probe_info: dict[str, Any] | None,
    *,
    coordinator=None,
    session_id=None,
) -> dict | None:
    if not contest_probe_info:
        return None
    titles = contest_probe_info.get("titles") or []
    if not titles:
        return None
    selector_context = {
        "selector_data": {
            "contests": [{"title": t, "source": "contest_probe"} for t in titles if t],
            "noisy_patterns": [s.lower() for s in (CONTEST_TITLE_SKIP_PHRASES or set())],
        },
        "input_file": os.path.basename(pdf_path),
        "contest_probe": {
            "hits": contest_probe_info.get("hits"),
            "sample_lines": contest_probe_info.get("sample_lines"),
        },
    }
    auto = select_contest_noninteractive(
        coordinator=coordinator,
        context=selector_context,
        session_id=session_id,
        prefer_year_match=False,
        return_mode="objects",
    )
    auto_list = auto if isinstance(auto, list) else []
    if not auto_list:
        return None
    top = auto_list[0]
    picked_title = safe_get(top, "title") or safe_get(top, "metadata", {}).get("display_title")
    if not picked_title:
        return None
    confidence = None
    try:
        confidence = float(safe_get(top, "confidence") or 0.0)
    except Exception:
        confidence = None
    return {
        "title": picked_title,
        "confidence": confidence,
        "source": "contest_probe",
    }


def _compute_sample_page_indices(
    page_count: int | None,
    *,
    page_windows: list[tuple[int, int]] | None = None,
    max_samples: int | None = None,
) -> list[int]:
    if not page_count or page_count <= 0:
        return [0]
    max_samples = max_samples or _OCR_SAMPLE_PAGE_TARGET
    candidates: list[int] = []
    if page_windows:
        for start, end in page_windows:
            start = max(0, min(page_count - 1, start))
            end = max(start + 1, min(page_count, end))
            midpoint = (start + end - 1) // 2
            candidates.extend([start, midpoint, end - 1])
    else:
        candidates.extend([0, page_count - 1])
        if page_count > 1:
            candidates.append(page_count // 2)
        if page_count > 6:
            candidates.extend([page_count // 4, (3 * page_count) // 4])
    deduped = sorted({max(0, min(page_count - 1, idx)) for idx in candidates})
    if not deduped:
        return [0]
    if len(deduped) <= max_samples:
        return deduped
    step = max(1, len(deduped) // max_samples)
    sampled = deduped[::step][:max_samples]
    return sampled or [deduped[0]]


def _contest_probe_scan(
    pdf_path: str,
    *,
    session_id=None,
    max_pages: int | None = None,
    stride: int | None = None,
    dpi: int | None = None,
    max_hits: int | None = None,
    cancel_flag=None,
) -> dict:
    if not pytesseract or not ENABLE_OCR:
        return {}
    stride = stride or _OCR_CONTEST_PROBE_STRIDE
    dpi = dpi or _OCR_CONTEST_PROBE_DPI
    max_hits = max_hits or _OCR_CONTEST_PROBE_MAX_HITS
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        logger.debug({
            "level": "DEBUG",
            "type": "handler",
            "message": f"[DEBUG] Contest probe skipped: {exc}",
            "session_id": session_id,
        })
        return {}

    total_pages = len(doc)
    limit = min(total_pages, max_pages or total_pages)
    if limit <= 0:
        doc.close()
        return {}
    if limit <= stride * 2:
        stride = max(1, stride // 2)
    if limit <= stride:
        stride = 1
    orientation_map, default_angle = _get_page_orientation_map(pdf_path, session_id=session_id)
    hits: list[int] = []
    titles: list[str] = []
    sample_lines: dict[int, list[str]] = {}
    page_summaries: list[dict[str, Any]] = []

    probe_indices = list(range(0, limit, stride))
    if probe_indices[-1] != limit - 1:
        probe_indices.append(limit - 1)

    for page_index in probe_indices:
        _ensure_not_cancelled(cancel_flag, session_id, f"pdf:contest_probe_page:{page_index}")
        try:
            page = doc[page_index]
            pix = page.get_pixmap(dpi=dpi)
            mode = "RGBA" if pix.alpha else "RGB"
            img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
            oriented = _apply_page_orientation(
                img,
                page_index,
                pdf_path,
                orientation_map,
                default_angle,
                session_id=session_id,
            )
            gray = ImageOps.grayscale(oriented)
            text = pytesseract.image_to_string(gray, config="--oem 1 --psm 6")
        except Exception as exc:
            logger.debug({
                "level": "DEBUG",
                "type": "handler",
                "message": f"[DEBUG] Contest probe OCR failed for page {page_index}: {exc}",
                "session_id": session_id,
            })
            continue

        lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
        if not lines:
            continue
        detected = _dedupe_contest_titles(detect_contest_titles_from_text(lines, pdf_path))
        if detected:
            hits.append(page_index)
            titles.extend(detected)
            sample_lines[page_index] = lines[:8]
            page_summaries.append({
                "page": page_index,
                "titles": detected,
                "lines": lines[:15],
            })
            if len(hits) >= max_hits:
                break

    doc.close()
    if not hits and not titles:
        return {}
    return {
        "hits": hits,
        "titles": _dedupe_contest_titles(titles),
        "sample_lines": sample_lines,
        "pages": page_summaries,
        "probe_stride": stride,
        "probe_dpi": dpi,
    }


def _yield_full_pass_batches(
    pdf_path: str,
    *,
    dpi: int,
    session_id=None,
    batch_pages: int = 8,
    max_pages: int | None = None,
    page_windows: list[tuple[int, int]] | None = None,
    cancel_flag=None,
):
    orientation_map, default_angle = _get_page_orientation_map(pdf_path, session_id=session_id)
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        logger.error({
            "level": "ERROR",
            "type": "handler",
            "message": f"[ERROR] Unable to open PDF for full-pass OCR: {exc}",
            "session_id": session_id,
        })
        return

    limit = len(doc)
    if max_pages and max_pages > 0:
        limit = min(limit, max_pages)
    batch_pages = max(1, batch_pages)

    ranges: list[tuple[int, int]]
    if page_windows:
        ranges = []
        for start, end in page_windows:
            start = max(0, min(limit, start))
            end = max(start + 1, min(limit, end))
            ranges.append((start, end))
    else:
        ranges = [(0, limit)]

    try:
        for window_start, window_end in ranges:
            start_index = window_start
            while start_index < window_end:
                end_index = min(window_end, start_index + batch_pages)
                _ensure_not_cancelled(cancel_flag, session_id, f"ocr:stream_window:{window_start}-{window_end}")
                images: list[Image.Image] = []
                for page_index in range(start_index, end_index):
                    try:
                        _ensure_not_cancelled(cancel_flag, session_id, f"ocr:stream_page:{page_index}")
                        page = doc[page_index]
                        pix = page.get_pixmap(dpi=dpi)
                        mode = "RGBA" if pix.alpha else "RGB"
                        img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
                        oriented = _apply_page_orientation(
                            img,
                            page_index,
                            pdf_path,
                            orientation_map,
                            default_angle,
                            session_id=session_id,
                        )
                        images.append(oriented)
                    except Exception as exc:
                        logger.warning({
                            "level": "WARNING",
                            "type": "handler",
                            "message": f"[WARN] Skipping page {page_index} during OCR batch render: {exc}",
                            "session_id": session_id,
                        })
                if images:
                    yield start_index, images
                start_index = end_index
    finally:
        doc.close()
def _camelot_signal_sets() -> tuple[set[str], set[str]]:
    """Return the Camelot signal and noise keyword sets used for scoring tables."""
    signal: set[str] = set()
    for group in (
        CANDIDATE_KEYWORDS,
        PARTY_KEYWORDS,
        TOTAL_KEYWORDS,
        BALLOT_TYPES,
        CONTEST_KEYWORDS,
        {"percent", "%", "election day", "early", "absentee", "provisional", "total vote", "grand total"},
    ):
        for token in group or []:
            if isinstance(token, str) and token.strip():
                signal.add(token.lower())
    signal.update({
        "candidate",
        "candidates",
        "name",
        "votes",
        "vote",
        "ballot",
        "total",
        "totals",
        "party",
        "precinct",
        "ward",
        "district",
        "absentee",
        "early",
        "provisional",
        "grand",
        "write",
        "turnout",
    })

    noise = {
        str(token).lower()
        for token in (MISC_FOOTER_KEYWORDS or [])
        if isinstance(token, str) and token.strip()
    }
    noise.update({"page", "sheet", "summary", "report", "statement", "certificate"})
    return signal, noise

def _split_ws_blocks(s: str) -> list[str]:
    return utils_split_ws_blocks(s)


def _is_bad_header_line(line: str) -> bool:
    return utils_is_bad_header_line(line)


def _prepare_output_context(base: dict | None, extra: dict | None = None) -> dict:
    """Merge context dictionaries while excluding non-serializable helpers."""
    ctx: dict = {}
    if isinstance(base, dict):
        for key, value in base.items():
            if key == "coordinator":
                continue
            ctx[key] = value
    if isinstance(extra, dict):
        ctx.update(extra)
    return ctx


def _build_ocr_evidence(
    pdf_path: str,
    fitz_mode: str | None,
    page_text_map: list[dict] | None,
    fitz_text_len: int,
    clean_text: str | None,
    raw_text: str | None,
    ocr_used: bool,
    ocr_confidence_avg: float | None,
    ocr_params: dict | None,
    ocr_runs: list[dict] | None,
    page_summaries: list[dict] | None,
    line_records: list[dict] | None,
    metadata: dict,
) -> dict:
    ocr_text_len = None
    if ocr_used and isinstance(page_summaries, list):
        ocr_text_len = int(sum((entry.get("raw_chars") or 0) for entry in page_summaries))
    elif isinstance(raw_text, str):
        ocr_text_len = len(raw_text)

    evidence: dict[str, object] = {
        "provenance": {
            "input_file": os.path.basename(pdf_path),
            "pdf_path": pdf_path,
            "pdf_page_total": metadata.get("pdf_page_total"),
            "fitz_mode": fitz_mode,
            "fitz_page_count": len(page_text_map or []),
            "fitz_char_count": sum((entry.get("char_count") or 0) for entry in (page_text_map or [])),
            "ocr_used": bool(ocr_used),
            "ocr_focus_windows": metadata.get("ocr_focus_windows"),
            "contest_probe": metadata.get("contest_probe"),
        },
        "alternatives": {
            "fitz_text_length": fitz_text_len,
            "ocr_text_length": ocr_text_len,
            "raw_text_length": len(raw_text or ""),
            "clean_text_length": len(clean_text or ""),
            "fitz_mode_used": fitz_mode,
            "ocr_params": ocr_params or {},
            "ocr_runs": ocr_runs or [],
            "ocr_raw_text_path": metadata.get("ocr_raw_text_path"),
            "ocr_clean_text_path": metadata.get("ocr_clean_text_path"),
        },
        "confidence": {
            "ocr_confidence_avg": ocr_confidence_avg,
            "ocr_run_count": len(ocr_runs or []),
        },
        "review": {
            "page_line_source": metadata.get("page_line_source"),
            "page_line_total": len(line_records or []),
            "page_line_pages": len(page_summaries or []),
            "page_line_summary": (page_summaries or [])[:10],
            "line_record_sample": [],
        },
    }

    if ocr_used and isinstance(page_summaries, list):
        evidence["alternatives"]["ocr_text_length"] = int(sum((entry.get("raw_chars") or 0) for entry in page_summaries))

    if line_records:
        sample = []
        for rec in (line_records or [])[:10]:
            sample.append({
                "page": rec.get("page"),
                "page_line_index": rec.get("page_line_index"),
                "global_line_index": rec.get("global_line_index"),
                "text": (rec.get("text") or "")[:200],
            })
        evidence["review"]["line_record_sample"] = sample

    # Preserve high-level diagnostics already captured in metadata
    if metadata.get("contest_detection"):
        evidence["contest_detection"] = metadata.get("contest_detection")
    if metadata.get("contest_probe"):
        evidence["contest_probe"] = metadata.get("contest_probe")
    if metadata.get("contest_probe_autopick"):
        evidence["contest_probe_autopick"] = metadata.get("contest_probe_autopick")

    return evidence

def _record_page_text_structure_observation(
    *,
    pdf_page_total: int | None,
    line_records: list[dict] | None,
    page_summaries: list[dict] | None,
    page_lines_fallback: bool,
    page_text_map: list[dict] | None,
    fitz_mode: str | None,
) -> bool:
    # Emit bounded page-text structure evidence without affecting parsing.
    try:
        if (
            not callable(_record_parse_observation)
            or _StructureObservationPhase is None
        ):
            return False
        return bool(
            _record_parse_observation(
                kind="pdf_structure_phase_observed",
                value_summary={
                    "phase": _StructureObservationPhase.PAGE_TEXT_STRUCTURE.value,
                    "page_count": (
                        pdf_page_total
                        if isinstance(pdf_page_total, int)
                        and not isinstance(pdf_page_total, bool)
                        and pdf_page_total >= 0
                        else None
                    ),
                    "page_line_total": len(line_records or []),
                    "page_line_pages": len(page_summaries or []),
                    "page_line_source": (
                        "fallback"
                        if page_lines_fallback
                        else "page_map"
                    ),
                    "page_line_index_available": bool(line_records),
                    "page_lines_fallback": bool(page_lines_fallback),
                    "page_text_map_entries": len(page_text_map or []),
                    "fitz_mode": (
                        str(fitz_mode)[:80]
                        if fitz_mode is not None
                        else None
                    ),
                },
                provenance="OBSERVED",
                source_location=(
                    "pdf_handler.parse_pdf_election_results:"
                    "page_text_structure"
                ),
            )
        )
    except Exception:
        return False



def _table_looks_bad(headers: list[str], rows: list[dict]) -> bool:
    return utils_table_looks_bad(headers, rows)


def _find_header_line(lines: list[str], hints: set[str]) -> tuple[list[str], int]:
    return utils_find_header_line(lines, hints)


def _extract_table_by_whitespace(lines: list[str], start_idx: int, headers: list[str]) -> list[dict]:
    return utils_extract_table_by_whitespace(lines, start_idx, headers)


def _record_table_stage(metadata: dict, stage: str, details: dict | None = None) -> None:
    if not metadata:
        return
    if metadata.get("table_failure_stage"):
        return
    metadata["table_failure_stage"] = stage
    if details:
        metadata["table_failure_details"] = details

try:
    import pandas as pd  # type: ignore
    _PANDAS_AVAILABLE = True
except Exception:  # pragma: no cover - pandas is optional but strongly recommended
    pd = None
    _PANDAS_AVAILABLE = False


def _ensure_fitz():
    """Import PyMuPDF and validate that the installed version is supported."""
    global _FITZ_MODULE
    if _FITZ_MODULE is not None:
        return _FITZ_MODULE

    try:
        module = importlib.import_module("fitz")
    except ImportError as exc:
        raise ImportError("You must install PyMuPDF to use the PDF handler: pip install pymupdf") from exc

    _check_pymupdf_version(module)
    _FITZ_MODULE = module
    return module


def _coerce_version_tuple(raw) -> tuple[int, ...]:
    """Normalize heterogenous version representations into an int tuple."""
    parts: list[int] = []
    if isinstance(raw, (list, tuple)):
        for item in raw:
            parts.extend(_coerce_version_tuple(item))
        return tuple(parts)
    if isinstance(raw, (int, float)):
        if isinstance(raw, float):
            # Split float like 1.26 into (1, 26)
            raw_str = str(raw)
        else:
            return (int(raw),)
    if isinstance(raw, str):
        raw_str = raw
    else:
        return tuple(parts)

    for token in re.findall(r"\d+", raw_str or ""):
        try:
            parts.append(int(token))
        except ValueError:
            continue
    return tuple(parts)


def _check_pymupdf_version(module) -> None:
    """Ensure the PyMuPDF version meets the minimum requirements for this handler."""
    version = getattr(module, "version", None)
    version_tuple = _coerce_version_tuple(version)
    if not version_tuple:
        return

    needed = tuple(int(part) for part in _MIN_PYMUPDF_VERSION)
    detected_version = ".".join(str(part) for part in version_tuple) or str(version)
    if version_tuple[: len(needed)] < needed:
        logger.warning({
            "level": "WARNING",
            "type": "dependency",
            "message": (
                "[WARN] Detected PyMuPDF %s. Upgrade to %s or newer to avoid parser instability."
                % (detected_version, ".".join(str(v) for v in _MIN_PYMUPDF_VERSION))
            ),
        })

fitz = _ensure_fitz()

try:
    import pytesseract
    import pdf2image
    if CONFIG_TESSERACT_CMD:
        # Allow Windows to work without adding Tesseract to PATH
        try:
            pytesseract.pytesseract.tesseract_cmd = CONFIG_TESSERACT_CMD
        except Exception:
            pass
except ImportError:
    pytesseract = None
    pdf2image = None

# -------------------- Semantic contest block + candidate-line extractors --------------------

_CAMELOT_SIGNAL_SET, _CAMELOT_NOISE_SET = _camelot_signal_sets()

_NUM_TOKEN_RE = re.compile(r"^\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*$")
_PCT_TOKEN_RE = re.compile(r"^\s*\d{1,3}(?:\.\d+)?\s*%\s*$")

def _score_camelot_table(df) -> float:
    """
    Scoring components:
      header_signal = (# header tokens in keyword set / columns) * 2.0
      candidate_row_density = rows with >=1 header-signal token / rows * 1.5
      numeric_density = numeric-ish cells / total cells * 1.2
      percent_presence bonus
      width_penalties
    """
    try:
        import pandas as pd  # noqa
        if df is None or df.empty:
            return 0.0
        cols = [str(c).strip().lower() for c in df.columns]
        col_hits = sum(1 for c in cols if any(k in c for k in _CAMELOT_SIGNAL_SET))
        header_signal = (col_hits / max(1, len(cols))) * 2.0

        total_cells = int(df.shape[0] * df.shape[1])
        numeric_cells = 0
        candidate_like_rows = 0
        percent_cells = 0
        for _, row in df.iterrows():
            row_tokens = 0
            row_has_signal = False
            for v in row.tolist():
                vs = str(v).strip().lower()
                if not vs:
                    continue
                if _NUM_TOKEN_RE.match(vs):
                    numeric_cells += 1
                elif _PCT_TOKEN_RE.match(vs):
                    numeric_cells += 1
                    percent_cells += 1
                else:
                    # detect numeric with commas
                    core = vs.replace(",", "")
                    if core.isdigit():
                        numeric_cells += 1
                if any(k in vs for k in _CAMELOT_SIGNAL_SET):
                    row_has_signal = True
                row_tokens += 1
            if row_has_signal:
                candidate_like_rows += 1

        numeric_density = (numeric_cells / max(1, total_cells)) * 1.2
        candidate_row_density = (candidate_like_rows / max(1, df.shape[0])) * 1.5
        pct_bonus = 0.3 if percent_cells > 0 else 0.0

        width_penalty = 0.0
        if df.shape[1] > 40:
            width_penalty += 0.6
        elif df.shape[1] > 28:
            width_penalty += 0.3
        if df.shape[1] < 2:
            width_penalty += 0.6

        noise_penalty = 0.0
        if any(any(n in c for n in _CAMELOT_NOISE_SET) for c in cols):
            noise_penalty += 0.15

        score = header_signal + candidate_row_density + numeric_density + pct_bonus - width_penalty - noise_penalty
        return round(score, 4)
    except Exception:
        return 0.0

def _normalize_camelot_headers(raw_headers):
    norm = []
    seen = set()
    for i, h in enumerate(raw_headers):
        hs = (str(h) or "").strip() or f"Column {i+1}"
        hs = re.sub(r"\s+", " ", hs)
        base = hs
        k = 2
        while hs.lower() in seen:
            hs = f"{base}_{k}"
            k += 1
        seen.add(hs.lower())
        norm.append(hs)
    return norm

def _camelot_table_to_rows(table, session_id=None, limit_rows=1200):
    try:
        df = table.df
    except Exception:
        return [], []
    try:
        df = df.replace(r"^\s*$", "", regex=True)
    except Exception:
        pass

    # Promote first row to headers if it looks stronger
    orig_cols = [str(c).strip() for c in df.columns]
    first_row = [str(x).strip() for x in (df.iloc[0].tolist() if len(df) else [])]
    def _sig(lst): return sum(1 for c in lst if any(k in c.lower() for k in _CAMELOT_SIGNAL_SET))
    if first_row and _sig(first_row) > _sig(orig_cols):
        df.columns = first_row
        df = df.iloc[1:]

    headers = _normalize_camelot_headers(df.columns.tolist())
    out = []
    for _, r in df.head(limit_rows).iterrows():
        vals = r.tolist()
        rec = {}
        empty = True
        for h, v in zip(headers, vals):
            if isinstance(v, str):
                vs = v.strip()
                if vs.replace(",", "").replace("%", "").isdigit():
                    rec[h] = vs
                else:
                    rec[h] = vs
            else:
                rec[h] = v
            if rec[h] not in ("", None):
                empty = False
        if not empty:
            out.append(rec)
    return headers, out

def _merge_camelot_tables_if_compatible(entries: list[dict]) -> list[dict]:
    """
    Merge tables with identical normalized header sets (order-insensitive) to reduce fragmentation.
    """
    if not entries:
        return entries
    buckets = {}
    for e in entries:
        key = tuple(sorted([h.lower() for h in e["headers"]]))
        buckets.setdefault(key, []).append(e)
    merged = []
    for key, group in buckets.items():
        if len(group) == 1:
            merged.append(group[0])
            continue
        # Combine rows
        rows = []
        headers_ref = group[0]["headers"]
        for g in group:
            rows.extend(g["rows"])
        merged.append({
            "headers": headers_ref,
            "rows": rows,
            "score": max(g["score"] for g in group),
            "flavor": ",".join(sorted({g["flavor"] for g in group})),
            "page_range": ",".join(str(g.get("page_range") or "") for g in group)
        })
    # Preserve ordering by score
    merged.sort(key=lambda x: (-x["score"], -len(x["rows"])))
    return merged

def _extract_camelot_tables(pdf_path: str, session_id=None, max_tables=15, pages="1-end"):
    results = []
    if not _CAMELOT_AVAILABLE:
        return results
    for flavor in ("lattice", "stream"):
        try:
            tables = camelot.read_pdf(
                pdf_path,
                pages=pages,
                flavor=flavor,
                strip_text="\n",
                suppress_stdout=True
            )
        except Exception as e:
            logger.debug({
                "level": "DEBUG",
                "type": "handler",
                "message": f"[DEBUG] Camelot {flavor} failed: {e}",
                "session_id": session_id
            })
            continue
        for t in list(tables)[:max_tables]:
            try:
                score = _score_camelot_table(t.df)
            except Exception:
                score = 0.0
            h, r = _camelot_table_to_rows(t, session_id=session_id)
            if h and r:
                results.append({
                    "headers": h,
                    "rows": r,
                    "score": score,
                    "flavor": flavor,
                    "page_range": getattr(t, "page", None)
                })
        if len(results) >= max_tables:
            break
    results.sort(key=lambda x: (-x["score"], -len(x["rows"])))
    if CAMELOT_MERGE_COMPAT:
        results = _merge_camelot_tables_if_compatible(results)
    return results

# Hybrid fill: enrich Camelot rows using OCR text lines for missing numeric fields
def _hybrid_fill_camelot(top_table: dict, ocr_lines: list[str]):
    if not CAMELOT_HYBRID_FILL or not top_table or not ocr_lines:
        return
    headers = top_table["headers"]
    rows = top_table["rows"]
    numeric_cols = [h for h in headers if any(k in h.lower() for k in ("vote", "total", "absentee", "early", "election day", "provisional", "%"))]
    if not numeric_cols:
        return
    # Build fast search index of OCR lines
    ocr_index = {}
    for ln in ocr_lines:
        low = ln.lower()
        # candidate-like heuristic: contains alphabetic and a number
        if sum(ch.isalpha() for ch in low) >= 3 and re.search(r"\d", low):
            ocr_index[len(ocr_index)] = low
    for row in rows:
        name_candidate = None
        for k in ("Candidate", "Name"):
            if k in row and isinstance(row[k], str):
                name_candidate = row[k].strip()
                break
        if not name_candidate:
            continue
        low_name = name_candidate.lower()
        # Skip if row mostly filled
        empties = [c for c in numeric_cols if not str(row.get(c, "")).strip()]
        if not empties:
            continue
        # Find best OCR line containing name token(s)
        tokens = [t for t in re.findall(r"[a-z0-9]+", low_name) if len(t) > 2]
        best = None
        best_hits = 0
        for _, ln in ocr_index.items():
            hits = sum(1 for t in tokens if t in ln)
            if hits > best_hits:
                best_hits = hits
                best = ln
        if not best or best_hits == 0:
            continue
        # Extract numbers in sequence and map to empty columns heuristically
        nums = re.findall(r"\d{1,3}(?:,\d{3})*(?:\.\d+)?%?", best)
        if not nums:
            continue
        # Simple left-to-right fill for empty numeric cols
        fill_iter = iter(nums)
        for col in empties:
            try:
                val = next(fill_iter)
            except StopIteration:
                break
            if val.endswith("%"):
                # Only put into percent-like column
                if "%" in col or "percent" in col.lower():
                    row[col] = val
                else:
                    # Skip placing % into a vote column
                    continue
            else:
                clean = val.replace(",", "")
                if clean.isdigit():
                    row[col] = int(clean)
                else:
                    row[col] = val

_NUM_RE = re.compile(r"(?P<num>\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
_PCT_RE = re.compile(r"(?P<pct>\d{1,3}(?:\.\d+)?)\s*%")

def _norm_txt(s: str) -> str:
    return utils_normalize_text_token(s)


def _token_set(s: str) -> set[str]:
    return utils_token_set(s)


def _header_signature(label: str) -> set[str]:
    return utils_header_signature(label)


def _looks_like_candidate_header(label: str) -> bool:
    return utils_looks_like_candidate_header(label)


def _compute_header_richness(candidate_headers: list[str]) -> dict[str, float]:
    return utils_compute_header_richness(candidate_headers)


def _compute_numeric_fill(rows: list[dict], candidate_headers: list[str]) -> float:
    return utils_compute_numeric_fill(rows, candidate_headers)


def _evaluate_table_candidate_quality(headers: list[str], rows: list[dict], contest_title: str) -> dict[str, object]:
    result = utils_evaluate_table_candidate_quality(headers, rows, contest_title)
    # Boost score for tables with location headers if no contest title present
    if not contest_title and headers:
        header_text = " ".join(h.lower() for h in headers if h)
        has_location_keywords = any(kw.lower() in header_text for kw in LOCATION_KEYWORDS)
        if has_location_keywords and isinstance(result, dict) and "score" in result:
            result["score"] = (result["score"] or 0) + 10  # Boost by 10 points
    return result


def _find_best_header_match(source: str, targets: list[str]) -> str | None:
    return utils_find_best_header_match(source, targets)


def _normalize_anchor_value(value) -> str:
    return utils_normalize_anchor_value(value)


def _merge_camelot_with_text(
    camelot_table: dict,
    text_headers: list[str],
    text_rows: list[dict],
) -> tuple[list[str], list[dict]] | None:
    return utils_merge_camelot_with_text(camelot_table, text_headers, text_rows)


def _best_title_match_idx(lines: list[str], selected_title: str) -> int:
    """Find the index of the line that best matches the selected title by token overlap."""
    return utils_best_title_match_idx(lines, selected_title)

def _extract_contest_block(
    lines: list[str],
    selected_contest_title: str,
    *,
    line_records: list[dict] | None = None,
    include_metadata: bool = False,
):
    return utils_extract_contest_block(
        lines,
        selected_contest_title,
        _CONTEST_RX,
        line_records=line_records,
        include_metadata=include_metadata,
    )

def _parse_candidate_line(line: str, ballot_types: list[str]) -> dict | None:
    """Proxy to shared candidate line parser."""
    return utils_parse_candidate_line(line, ballot_types)

def extract_candidate_totals_from_lines(lines: list[str], selected_title: str) -> tuple[list[str], list[dict]]:
    """Shared extraction for candidate totals tables with local ballot defaults."""
    ballot_types = list(BALLOT_TYPES) if BALLOT_TYPES else None
    return utils_extract_candidate_totals_from_lines(lines, selected_title, ballot_types, _CONTEST_RX)

def _is_numeric_like(token: str) -> bool:
    return utils_is_numeric_like(token)


def _normalize_numeric_token(value: str) -> str:
    return utils_normalize_numeric_token(value)


def _matches_anchor_header(raw: str) -> bool:
    return utils_matches_anchor_header(raw)


def _reconstruct_columnar_block(lines: list[str]) -> tuple[list[str], list[dict]]:
    return utils_reconstruct_columnar_block(lines, _CONTEST_RX)


def _extract_party_lookup_from_lines(lines: list[str] | None) -> dict[str, str]:
    return utils_extract_party_lookup_from_lines(lines)


def _parse_candidate_header_with_party(header: str, party_lookup: dict[str, str]) -> tuple[str, str, dict]:
    return utils_parse_candidate_header_with_party(header, party_lookup)


def _coerce_vote_value_for_reconstruction(value):
    return utils_coerce_vote_value_for_reconstruction(value)



_DENSE_PRECINCT_LINE = re.compile(r"\b\d{3,4}\s*-\s*[a-z0-9]", re.IGNORECASE)


_PRECINCT_INLINE_ANCHOR = re.compile(r"^\s*precincts?\b", re.IGNORECASE)


def _split_dense_precinct_segments(line: str) -> list[str]:
    """Split a single dense line containing many precinct rows into separate segments."""
    matches = list(_DENSE_PRECINCT_LINE.finditer(line))
    if len(matches) <= 1:
        return [line]
    segments: list[str] = []
    first_start = matches[0].start()
    prefix = line[:first_start].strip()
    if prefix:
        segments.append(prefix)
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(line)
        chunk = line[start:end].strip()
        if chunk:
            segments.append(chunk)
    return segments or [line]


def _expand_dense_precinct_block(block: list[str]) -> tuple[list[str], dict]:
    """Detect and expand dense precinct listings into individual lines."""
    if not block:
        return block, {"expanded": False}
    expanded: list[str] = []
    expanded_any = False
    for line in block:
        if len(line) >= 400 and _DENSE_PRECINCT_LINE.search(line):
            pieces = _split_dense_precinct_segments(line)
            if len(pieces) > 1:
                expanded.extend(pieces)
                expanded_any = True
                continue
        expanded.append(line)
    if not expanded_any:
        return block, {"expanded": False}
    meta = {
        "expanded": True,
        "original_lines": len(block),
        "expanded_lines": len(expanded),
    }
    return expanded, meta


def _normalize_precinct_inline_rows(block: list[str]) -> tuple[list[str], dict]:
    """Split lines where the precinct anchor and first rows were jammed together by OCR."""
    if not block:
        return block, {}
    normalized: list[str] = []
    stats = {
        "inline_headers_normalized": 0,
        "inline_rows_injected": 0,
    }
    for raw in block:
        text = (raw or "").strip()
        anchor_match = _PRECINCT_INLINE_ANCHOR.match(text)
        if anchor_match:
            row_match = _DENSE_PRECINCT_LINE.search(text, anchor_match.end())
            if row_match:
                anchor_token = text[anchor_match.start():anchor_match.end()].strip()
                anchor_label = anchor_token.title() or "Precinct"
                header_segment = text[anchor_match.end():row_match.start()].strip()
                data_segment = text[row_match.start():].strip()
                if anchor_label:
                    normalized.append(anchor_label)
                else:
                    normalized.append("Precinct")
                if header_segment:
                    header_clean = re.sub(r"[\|\[\]{}<>]+", " ", header_segment)
                    header_clean = re.sub(r"\s{2,}", " ", header_clean).strip()
                    if header_clean:
                        normalized.append(header_clean)
                split_rows = [seg for seg in _split_dense_precinct_segments(data_segment) if seg and seg.strip()]
                if split_rows:
                    normalized.extend(split_rows)
                    stats["inline_rows_injected"] += max(0, len(split_rows) - 1)
                elif data_segment:
                    normalized.append(data_segment)
                stats["inline_headers_normalized"] += 1
                continue
        normalized.append(raw)
    return normalized, stats


def _prepare_dense_precinct_lines(block: list[str]) -> tuple[list[str], dict]:
    """Normalize inline precinct headers and expand dense lines before reconstruction."""
    if not block:
        return block, {}
    inline_normalized, inline_meta = _normalize_precinct_inline_rows(block)
    expanded_block, expand_meta = _expand_dense_precinct_block(inline_normalized)
    meta: dict[str, Any] = {}
    if inline_meta.get("inline_headers_normalized"):
        meta.update(inline_meta)
    if expand_meta.get("expanded"):
        meta.setdefault("dense_line_expansion", expand_meta)
    return expanded_block, meta


def _try_columnar_reconstruction(
    pdf_path: str,
    lines: list[str],
    line_records: list[dict] | None,
    selected_contest_title: str,
    state: str,
    county: str,
    metadata: dict,
    coordinator,
    session_id: str | None,
):
    if metadata is None:
        metadata = {}

    debug_events: list[dict] = []
    recon_attempts: list[dict[str, Any]] = []

    def _commit_attempts(extra_failure: dict | None = None) -> None:
        metadata["columnar_reconstruction_attempts"] = recon_attempts
        if debug_events:
            metadata["reconstruction_debug_events"] = debug_events
        else:
            metadata["reconstruction_debug_events"] = []
        if extra_failure:
            failure_detail = {
                "reason": extra_failure.get("reason", "unknown"),
                "contest": selected_contest_title,
                "state": state,
                "county": county,
                "attempts": len(recon_attempts),
                "scopes_tried": [entry.get("scope") for entry in recon_attempts],
            }
            failure_detail.update({k: v for k, v in extra_failure.items() if k != "reason"})
            metadata["columnar_reconstruction_failure"] = failure_detail

    if not lines:
        _commit_attempts({"reason": "no_lines_available"})
        return None

    contest_block: list[str] = []
    contest_block_meta: dict = {}
    try:
        contest_result = _extract_contest_block(
            lines,
            selected_contest_title,
            line_records=line_records,
            include_metadata=True,
        )
        if isinstance(contest_result, tuple):
            contest_block = contest_result[0] or []
            contest_block_meta = contest_result[1] or {}
        else:
            contest_block = contest_result or []
    except Exception:
        contest_block = []
        contest_block_meta = {}

    if contest_block_meta:
        contest_block_meta = dict(contest_block_meta)
        contest_block_meta.setdefault("selected_title", selected_contest_title)
        contest_block_meta.setdefault("line_count", len(contest_block))
        contest_block_meta.setdefault("line_records_available", bool(line_records))
        metadata.setdefault("contest_segments", {})[selected_contest_title] = contest_block_meta

    search_spaces: list[tuple[str, list[str]]] = []
    if contest_block:
        search_spaces.append(("contest_block", contest_block))
    if lines:
        search_spaces.append(("document", lines))

    recon_headers: list[str] = []
    recon_rows: list[dict] = []
    contest_scope: str | None = None

    for scope, candidate_lines in search_spaces:
        prepared_lines, prep_meta = _prepare_dense_precinct_lines(candidate_lines or [])
        attempt_entry: dict[str, Any] = {
            "scope": scope,
            "line_count": len(candidate_lines or []),
            "prepared_line_count": len(prepared_lines),
        }
        if prep_meta:
            metadata.setdefault("dense_line_normalization", []).append({
                "scope": scope,
                **prep_meta,
            })
            attempt_entry["normalization"] = prep_meta
        headers, rows = _reconstruct_columnar_block(prepared_lines)
        scope_events = utils_consume_reconstruction_debug_events()
        if scope_events:
            for event in scope_events:
                event.setdefault("scope", scope)
            debug_events.extend(scope_events)
        attempt_entry.update({
            "headers_detected": len(headers or []),
            "rows_detected": len(rows or []),
            "debug_event_count": len(scope_events or []),
        })
        if headers and rows:
            attempt_entry["success"] = True
            recon_attempts.append(attempt_entry)
            recon_headers = headers
            recon_rows = rows
            contest_scope = scope
            break
        attempt_entry["success"] = False
        recon_attempts.append(attempt_entry)

    if contest_block_meta and contest_scope:
        contest_block_meta = metadata.get("contest_segments", {}).get(selected_contest_title, {})
        if isinstance(contest_block_meta, dict):
            contest_block_meta.setdefault("reconstruction_scope", contest_scope)

    if not recon_headers or not recon_rows:
        _commit_attempts({"reason": "no_columnar_table_detected"})
        return None

    party_lookup: dict[str, str] = {}
    if contest_block:
        party_lookup.update(_extract_party_lookup_from_lines(contest_block))
    if lines:
        doc_lookup = _extract_party_lookup_from_lines(lines)
        for code, label in doc_lookup.items():
            party_lookup.setdefault(code, label)

    location_header = recon_headers[0]
    candidate_headers = recon_headers[1:]

    candidate_infos: list[dict] = []
    for cand_header in candidate_headers:
        candidate_label, party_label, info = _parse_candidate_header_with_party(cand_header, party_lookup)
        info["candidate_label"] = candidate_label
        info["party_label"] = normalize_party_label(party_label) if party_label else ""
        candidate_infos.append(info)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for info in candidate_infos:
        key = (info.get("candidate_label") or info.get("source_header") or "").strip().lower()
        grouped[key].append(info)

    for bucket in grouped.values():
        if len(bucket) == 1:
            info = bucket[0]
            display = info.get("candidate_label") or info.get("source_header") or ""
            info["display_label"] = display.strip()
            continue
        for idx, info in enumerate(bucket, start=1):
            base = (info.get("candidate_label") or info.get("source_header") or "").strip()
            suffix = (info.get("party_label") or info.get("party_code") or str(idx)).strip()
            info["display_label"] = f"{base} ({suffix})".strip()

    normalized_headers = [location_header]
    for info in candidate_infos:
        display = info.get("display_label") or info.get("candidate_label") or info.get("source_header") or ""
        display = re.sub(r"\s{2,}", " ", display).strip()
        info["display_label"] = display or (info.get("source_header") or "")
        total_key = f"{info['display_label']} - Total Vote"
        party_key = f"{info['display_label']} - Party"
        info["total_key"] = total_key
        info["party_key"] = party_key
        normalized_headers.extend([total_key, party_key])
        raw_party = info.get("party_label", "")
        info["party_label"] = normalize_party_label(raw_party) if raw_party else ""

    def _normalize_candidate_row(row: dict) -> dict:
        normalized = {location_header: row.get(location_header, "")}
        for info in candidate_infos:
            source_header = info.get("source_header")
            total_val = _coerce_vote_value_for_reconstruction(row.get(source_header, ""))
            normalized[info["total_key"]] = total_val
            normalized[info["party_key"]] = info.get("party_label", "")
        return normalized

    segments_map: OrderedDict[str, dict] = OrderedDict()
    default_rows: list[dict] = []

    for row in recon_rows:
        sub_label = row.pop("_subcontest_label", None)
        sub_number_raw = row.pop("_subcontest_number", None)
        number_val: int | None = None
        if sub_number_raw is not None:
            try:
                number_val = int(sub_number_raw)
            except Exception:
                try:
                    number_val = int(str(sub_number_raw).strip())
                except Exception:
                    number_val = None
        normalized_row = _normalize_candidate_row(row)
        if sub_label or number_val is not None:
            label_clean = re.sub(r"\s{2,}", " ", str(sub_label or "")).strip()
            if not label_clean and number_val is not None:
                label_clean = f"District {number_val}"
            if not label_clean:
                label_clean = "Segment"
            key = f"{label_clean.lower()}::{number_val if number_val is not None else ''}"
            segment_entry = segments_map.setdefault(key, {
                "label": label_clean,
                "number": number_val,
                "rows": [],
            })
            if not segment_entry.get("label"):
                segment_entry["label"] = label_clean
            if segment_entry.get("number") is None and number_val is not None:
                segment_entry["number"] = number_val
            segment_entry["rows"].append(normalized_row)
        else:
            default_rows.append(normalized_row)

    candidate_meta = [{
        "source_header": info.get("source_header"),
        "display_label": info.get("display_label"),
        "party": info.get("party_label", ""),
        "party_code": info.get("party_code"),
        "party_inference": info.get("party_inference"),
        "total_column": info.get("total_key"),
        "party_column": info.get("party_key"),
    } for info in candidate_infos]

    base_context = {
        "state": state,
        "county": county,
        "contest": selected_contest_title,
    }

    def _run_single_output(rows: list[dict], contest_title: str, extra_context: dict | None = None):
        context_extra = {
            "handler": "pdf_handler",
            "input_file": os.path.basename(pdf_path),
            "session_id": session_id,
            "columnar_reconstruction": True,
        }
        if extra_context:
            context_extra.update(extra_context)
        export_context = _prepare_output_context(base_context, context_extra)
        rows_copy = [dict(r) for r in rows]
        transformed_headers, transformed_rows, smart_applied = transform_wide_to_smart_standard(
            list(normalized_headers),
            rows_copy,
            export_context,
        )
        if smart_applied:
            final_headers_local = transformed_headers
            final_rows_local = transformed_rows
        else:
            final_headers_local = list(normalized_headers)
            final_rows_local = rows_copy
        result_paths = finalize_election_output(
            headers=final_headers_local,
            data=final_rows_local,
            coordinator=coordinator,
            contest=contest_title,
            state=state,
            county=county,
            context=export_context,
            enable_user_feedback=False,
            session_id=session_id,
        )
        return final_headers_local, final_rows_local, export_context, bool(smart_applied), result_paths

    if not segments_map:
        final_headers, final_rows, export_context, smart_applied, result_paths = _run_single_output(default_rows, selected_contest_title)
        columnar_meta = metadata.get("columnar_reconstruction") or {}
        columnar_meta.update({
            "rows": len(recon_rows),
            "columns": len(recon_headers),
            "scope": contest_scope,
            "wide_rows": len(default_rows),
            "final_rows": len(final_rows),
            "party_lookup": party_lookup,
            "party_lookup_keys": sorted(party_lookup.keys()),
            "candidate_columns": candidate_meta,
            "location_header": location_header,
            "wide_headers": normalized_headers,
            "final_headers": final_headers,
            "smart_standard_applied": bool(smart_applied),
            "attempts": recon_attempts,
        })
        if debug_events:
            columnar_meta["debug_events"] = debug_events
        metadata["columnar_reconstruction"] = columnar_meta
        export_context["columnar_reconstruction_details"] = columnar_meta
        metadata.setdefault("decision_trace", []).append({
            "stage": "columnar_reconstruction",
            "contest": selected_contest_title,
            "scope": contest_scope,
            "candidates": len(candidate_infos),
            "smart_standard_applied": bool(smart_applied),
            "location_header": location_header,
        })
        metadata.update({
            "output_file": os.path.basename(result_paths.get("csv_path", "")),
            "headers": final_headers,
            "row_count": len(final_rows),
            "csv_path": result_paths.get("csv_path"),
            "metadata_path": result_paths.get("metadata_path"),
            "columnar_reconstruction": columnar_meta,
        })
        metadata.pop("columnar_reconstruction_failure", None)
        _commit_attempts(None)
        logger.info({
            "level": "INFO",
            "type": "output",
            "message": "[OUTPUT] Columnar reconstruction normalized to smart-standard rows.",
            "session_id": session_id,
            "rows": len(final_rows),
            "candidates": len(candidate_infos),
            "smart_standard_applied": bool(smart_applied),
        })
        return final_headers, final_rows, selected_contest_title, metadata

    segments_sequence: list[dict] = []
    for entry in segments_map.values():
        rows_group = entry.get("rows") or []
        if rows_group:
            segments_sequence.append({
                "label": entry.get("label"),
                "number": entry.get("number"),
                "rows": rows_group,
            })
    if default_rows:
        segments_sequence.append({
            "label": "Summary",
            "number": None,
            "rows": default_rows,
        })

    segments_with_rows = [segment for segment in segments_sequence if segment.get("rows")]
    if not segments_with_rows:
        final_headers, final_rows, export_context, smart_applied, result_paths = _run_single_output(default_rows, selected_contest_title)
        columnar_meta = metadata.get("columnar_reconstruction") or {}
        columnar_meta.update({
            "rows": len(recon_rows),
            "columns": len(recon_headers),
            "scope": contest_scope,
            "wide_rows": len(default_rows),
            "final_rows": len(final_rows),
            "party_lookup": party_lookup,
            "party_lookup_keys": sorted(party_lookup.keys()),
            "candidate_columns": candidate_meta,
            "location_header": location_header,
            "wide_headers": normalized_headers,
            "final_headers": final_headers,
            "smart_standard_applied": bool(smart_applied),
            "attempts": recon_attempts,
        })
        if debug_events:
            columnar_meta["debug_events"] = debug_events
        metadata["columnar_reconstruction"] = columnar_meta
        export_context["columnar_reconstruction_details"] = columnar_meta
        metadata.setdefault("decision_trace", []).append({
            "stage": "columnar_reconstruction",
            "contest": selected_contest_title,
            "scope": contest_scope,
            "candidates": len(candidate_infos),
            "smart_standard_applied": bool(smart_applied),
            "location_header": location_header,
        })
        metadata.update({
            "output_file": os.path.basename(result_paths.get("csv_path", "")),
            "headers": final_headers,
            "row_count": len(final_rows),
            "csv_path": result_paths.get("csv_path"),
            "metadata_path": result_paths.get("metadata_path"),
            "columnar_reconstruction": columnar_meta,
        })
        metadata.pop("columnar_reconstruction_failure", None)
        _commit_attempts(None)
        logger.info({
            "level": "INFO",
            "type": "output",
            "message": "[OUTPUT] Columnar reconstruction normalized to smart-standard rows.",
            "session_id": session_id,
            "rows": len(final_rows),
            "candidates": len(candidate_infos),
            "smart_standard_applied": bool(smart_applied),
        })
        return final_headers, final_rows, selected_contest_title, metadata

    def _normalize_spaces(value: str | None) -> str:
        return re.sub(r"\s{2,}", " ", (value or "")).strip()

    def _to_ordinal(number: int | None) -> str:
        if number is None:
            return ""
        if 10 <= number % 100 <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
        return f"{number}{suffix}"

    def _segment_contest_title(base: str, label: str | None, number: int | None) -> str:
        base_norm = _normalize_spaces(base)
        base_lower = base_norm.lower()
        label_norm = _normalize_spaces(label)
        ordinal = _to_ordinal(number).lower()
        if label_norm and label_norm.lower() in base_lower:
            return base_norm
        if number is not None:
            phrase = f"district {number}".lower()
            if phrase in base_lower or ordinal in base_lower or f"{number} district" in base_lower:
                return base_norm
        suffix = label_norm or (f"District {number}" if number is not None else "Segment")
        return f"{base_norm} - {suffix}"

    def _segment_matches_title(segment: dict, title: str) -> bool:
        title_low = _normalize_spaces(title).lower()
        label_low = _normalize_spaces(segment.get("label")).lower()
        if label_low and label_low in title_low:
            return True
        number = segment.get("number")
        if number is not None:
            ordinal = _to_ordinal(number).lower()
            if (
                f"district {number}" in title_low
                or f"{number} district" in title_low
                or ordinal in title_low
            ):
                return True
        return False

    bundle_seed = f"{os.path.abspath(pdf_path)}::{selected_contest_title}"
    bundle_key = hashlib.sha1(bundle_seed.encode("utf-8", errors="ignore")).hexdigest()[:16]

    for segment in segments_with_rows:
        segment["contest_title"] = _segment_contest_title(
            selected_contest_title,
            segment.get("label"),
            segment.get("number"),
        )

    bundle_members = []
    for idx, segment in enumerate(segments_with_rows):
        bundle_members.append({
            "title": segment.get("contest_title"),
            "row_count": len(segment.get("rows") or []),
            "metadata": {
                "sub_contest_label": segment.get("label"),
                "sub_contest_number": segment.get("number"),
                "bundle_index": idx,
            },
        })

    bundle_metadata = {
        "bundle_mode": "aggregate",
        "bundle_key": bundle_key,
        "bundle_size": len(bundle_members),
        "display_title": selected_contest_title,
        "summary": f"{len(bundle_members)} segments",
        "members": bundle_members,
    }

    selected_segment = None
    for segment in segments_with_rows:
        if _segment_matches_title(segment, selected_contest_title):
            selected_segment = segment
            break
    if selected_segment is None:
        selected_segment = segments_with_rows[0]

    bundle_outputs: list[dict] = []
    selected_final_headers: list[str] = []
    selected_final_rows: list[dict] = []
    selected_result_paths: dict[str, str] = {"csv_path": "", "metadata_path": ""}
    selected_export_context: dict | None = None
    selected_smart_applied = False

    total_wide_rows = sum(len(segment.get("rows") or []) for segment in segments_with_rows)

    for idx, segment in enumerate(segments_with_rows):
        label = segment.get("label")
        number = segment.get("number")
        contest_title = segment.get("contest_title") or selected_contest_title
        extra_context = {
            "bundle_mode": "aggregate",
            "bundle_key": bundle_key,
            "bundle_size": len(bundle_members),
            "bundle_metadata": bundle_metadata,
            "bundle_member_index": idx,
            "bundle_member_label": label,
            "bundle_member_number": number,
            "sub_contest_label": label,
            "sub_contest_number": number,
            "contest": contest_title,
        }
        final_headers_seg, final_rows_seg, export_context_seg, smart_applied_seg, result_seg = _run_single_output(segment.get("rows") or [], contest_title, extra_context)
        bundle_outputs.append({
            "label": label,
            "number": number,
            "contest": contest_title,
            "csv_path": result_seg.get("csv_path"),
            "metadata_path": result_seg.get("metadata_path"),
            "row_count": len(final_rows_seg),
            "headers": final_headers_seg,
            "smart_standard_applied": smart_applied_seg,
        })
        if segment is selected_segment:
            selected_final_headers = final_headers_seg
            selected_final_rows = final_rows_seg
            selected_result_paths = result_seg
            selected_export_context = export_context_seg
            selected_smart_applied = smart_applied_seg

    if selected_export_context is None:
        selected_export_context = _prepare_output_context(base_context, {
            "handler": "pdf_handler",
            "input_file": os.path.basename(pdf_path),
            "session_id": session_id,
            "columnar_reconstruction": True,
        })

    columnar_meta = metadata.get("columnar_reconstruction") or {}
    columnar_meta.update({
        "rows": len(recon_rows),
        "columns": len(recon_headers),
        "scope": contest_scope,
        "wide_rows": total_wide_rows,
        "final_rows": len(selected_final_rows),
        "party_lookup": party_lookup,
        "party_lookup_keys": sorted(party_lookup.keys()),
        "candidate_columns": candidate_meta,
        "location_header": location_header,
        "wide_headers": normalized_headers,
        "final_headers": selected_final_headers,
        "smart_standard_applied": bool(selected_smart_applied),
        "bundle_mode": "aggregate",
        "bundle_outputs": bundle_outputs,
        "bundle_key": bundle_key,
        "bundle_metadata": bundle_metadata,
        "selected_sub_contest": {
            "label": selected_segment.get("label"),
            "number": selected_segment.get("number"),
            "contest": selected_segment.get("contest_title"),
        },
        "attempts": recon_attempts,
    })
    if debug_events:
        columnar_meta["debug_events"] = debug_events
    metadata["columnar_reconstruction"] = columnar_meta
    selected_export_context["columnar_reconstruction_details"] = columnar_meta

    metadata.setdefault("decision_trace", []).append({
        "stage": "columnar_reconstruction",
        "contest": selected_contest_title,
        "scope": contest_scope,
        "candidates": len(candidate_infos),
        "smart_standard_applied": bool(selected_smart_applied),
        "location_header": location_header,
        "bundle_mode": "aggregate",
        "segments": len(bundle_outputs),
    })

    metadata.update({
        "output_file": os.path.basename(selected_result_paths.get("csv_path", "")),
        "headers": selected_final_headers,
        "row_count": len(selected_final_rows),
        "csv_path": selected_result_paths.get("csv_path"),
        "metadata_path": selected_result_paths.get("metadata_path"),
        "columnar_reconstruction": columnar_meta,
        "bundle_outputs": bundle_outputs,
        "bundle_metadata": bundle_metadata,
        "sub_contest_label": selected_segment.get("label"),
        "sub_contest_number": selected_segment.get("number"),
        "contest": selected_segment.get("contest_title"),
    })
    metadata.pop("columnar_reconstruction_failure", None)
    _commit_attempts(None)

    logger.info({
        "level": "INFO",
        "type": "output",
        "message": "[OUTPUT] Columnar reconstruction emitted bundled sub-contests.",
        "session_id": session_id,
        "segments": len(bundle_outputs),
        "candidates": len(candidate_infos),
        "smart_standard_applied": bool(selected_smart_applied),
    })

    return selected_final_headers, selected_final_rows, selected_segment.get("contest_title"), metadata

def _log_ocr_environment(session_id=None):
    try:
        resolved_tesseract = None
        try:
            resolved_tesseract = getattr(pytesseract.pytesseract, "tesseract_cmd", None)
        except Exception:
            resolved_tesseract = None

        info = {
            "platform": platform.platform(),
            "pytesseract": bool(pytesseract),
            "pdf2image": bool(pdf2image),
            "poppler_path_env": bool(CONFIG_POPPLER_PATH),
            "poppler_path_resolved": _detect_poppler_path(),
            "pdftoppm_in_path": bool(shutil.which("pdftoppm")),
            "pdftocairo_in_path": bool(shutil.which("pdftocairo")),
            "tesseract_cmd_env": bool(CONFIG_TESSERACT_CMD),
            "tesseract_cmd_resolved": resolved_tesseract,
            "tesseract_in_path": bool(shutil.which("tesseract")),
            "ENABLE_OCR": bool(ENABLE_OCR),
            "ENABLE_OCR_FORCE": bool(ENABLE_OCR_FORCE),
        }
        logger.info({
            "level": "INFO",
            "type": "handler",
            "message": "[ENV] PDF/OCR capabilities detected",
            "session_id": session_id,
            "env": info
        })
    except Exception:
        pass

def _detect_poppler_path() -> str | None:
    """
    Cross-platform Poppler locator.
    - Windows: try config POPPLER_PATH, env POPPLER_PATH, common install dirs.
    - Linux/macOS: if pdftoppm is in PATH, return None (pdf2image will use PATH).
    """
    # Config or env
    if CONFIG_POPPLER_PATH and os.path.isdir(CONFIG_POPPLER_PATH):
        return CONFIG_POPPLER_PATH
    env_path = os.environ.get("POPPLER_PATH")
    if env_path and os.path.isdir(env_path):
        return env_path

    system = platform.system().lower()
    if system.startswith("win"):
        candidates = [
            r"C:\Program Files\poppler\Library\bin",
            r"C:\Program Files\poppler-24.08.0\Library\bin",
            r"C:\Program Files\poppler-24.07.0\Library\bin",
            r"C:\Program Files\poppler-24.06.0\Library\bin",
            r"C:\Program Files\poppler\bin",
            r"C:\poppler\bin",
        ]
        for p in candidates:
            if os.path.isdir(p):
                return p
        return None
    else:
        # On Linux/macOS we rely on PATH
        if shutil.which("pdftoppm") or shutil.which("pdftocairo"):
            return None
        return None

def _detect_contest_positions(line_records: list[dict]) -> list[dict]:
    """Detect contest titles and their positions by page from line records."""
    contest_positions = []
    for record in line_records:
        text = record.get("text", "").strip()
        if not text:
            continue
        page = record.get("page")
        if page is None:
            continue
        # Check if line matches contest keywords
        if _CONTEST_RX.search(text):
            contest_positions.append({
                "title": text,
                "page": page,
                "line_index": record.get("global_line_index", 0)
            })
    return contest_positions


def _associate_tables_with_contests(contest_positions: list[dict], tables: list[dict], page_text_map: list[dict]) -> list[dict]:
    """Associate tables with the nearest previous contest and evaluate quality."""
    associated = []
    for table in tables:
        page = table.get('page', 0)
        # Find the nearest previous contest
        contest = None
        for pos in reversed(contest_positions):
            if pos['page'] <= page:
                contest = pos
                break
        if contest:
            contest_title = contest['title']
            headers = table.get('headers', [])
            data = table.get('data', [])
            score = _evaluate_table_candidate_quality(headers, data, contest_title).get('score', 0.0)
            associated.append({
                'score': score,
                'headers': headers,
                'data': data,
                'contest_title': contest_title
            })
    return associated

def _is_mostly_markup(text: str) -> bool:
    """
    Return True if the extracted 'text' is actually markup-wrappers (e.g., <img> tags)
    with little real text. This function intentionally strips base64 image payloads
    and common HTML wrappers before estimating remaining alphabetic signal to avoid
    misclassifying large data:image blobs as textual content.
    """
    if not isinstance(text, str):
        return False
    s = text.strip()
    if not s:
        return False

    s_lower = s.lower()
    # Fast path: if no obvious markup tokens, bail out early
    has_tags = any(tok in s_lower for tok in ("<img", "<div", "<span", "<html", "<svg", "<p", "<table", "data:image/"))
    if not has_tags:
        return False

    # Strip data:image payloads and very long base64-like runs (noise)
    try:
        s_wo_b64 = re.sub(r'src\s*=\s*"data:image/[^\"]+"', 'src="[image]"', s, flags=re.IGNORECASE)
        s_wo_b64 = re.sub(r'[A-Za-z0-9+/=]{200,}', ' ', s_wo_b64)
    except Exception:
        s_wo_b64 = s

    # Remove tags to evaluate remaining plain text signal
    try:
        s_plain = re.sub(r'<[^>]+>', ' ', s_wo_b64)
    except Exception:
        s_plain = s_wo_b64

    # Compute alphabetic characters in a limited window to keep it cheap
    alpha = sum(1 for ch in s_plain[:8000] if ch.isalpha())
    return alpha < OCR_MIN_ALPHA_SIGNAL

def _sanitize_extracted_text(text: str) -> str:
    """
    Convert raw extracted content (which may contain XHTML, <img src="data:image..."> etc.)
    into neat, readable lines for downstream steps.
    - Remove data:image/base64 payloads and HTML tags
    - Unescape entities
    - Collapse whitespace
    - Drop extremely noisy/empty lines
    """
    if not isinstance(text, str):
        return ""
    cache_key = None
    try:
        cache_key = f"{_SANITIZE_CACHE_VERSION}:{hashlib.sha1(text.encode('utf-8', errors='ignore')).hexdigest()}"
        cached_value = _sanitize_cache_get(cache_key)
        if cached_value is not None:
            return cached_value
    except Exception:
        cache_key = None
    # Remove data:image base64 attributes entirely
    text = re.sub(r'src\s*=\s*"data:image/[^"]+"', 'src="[image]"', text, flags=re.IGNORECASE)
    # Remove long base64-like runs that may appear outside attributes
    text = re.sub(r'[A-Za-z0-9+/=]{200,}', ' ', text)
    # Strip all HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Unescape HTML entities
    try:
        text = html.unescape(text)
    except Exception:
        pass
    raw_lines = text.splitlines()
    candidate_lines = []
    vertical_buffer: list[str] = []
    vertical_chunks = 0

    def _vertical_buffer_length(buffer: list[str]) -> int:
        return sum(len(token.strip()) for token in buffer)

    def flush_vertical_buffer():
        nonlocal vertical_chunks
        if not vertical_buffer:
            return
        joined = "".join(token.strip() for token in vertical_buffer)
        joined = re.sub(r'([^\w\s])\1{2,}', r'\1', joined)
        joined = joined.strip()
        if len(joined) > 1 and re.search(r"[A-Za-z0-9]", joined):
            candidate_lines.append(joined)
            vertical_chunks += 1
        else:
            for token in vertical_buffer:
                candidate_lines.append(token)
        vertical_buffer.clear()

    for raw in raw_lines:
        stripped = raw.strip()
        if not stripped:
            flush_vertical_buffer()
            continue
        collapsed = re.sub(r'\s+', ' ', stripped)
        alnum = sum(ch.isalnum() for ch in collapsed)
        is_vertical_char = bool(alnum == 1 and len(collapsed) <= 3 and re.search(r"[A-Za-z]", collapsed))
        if is_vertical_char:
            if _vertical_buffer_length(vertical_buffer) + len(collapsed) > _SANITIZE_VERTICAL_JOIN_LIMIT:
                flush_vertical_buffer()
            vertical_buffer.append(collapsed)
            continue
        flush_vertical_buffer()
        candidate_lines.append(collapsed)

    flush_vertical_buffer()

    drop_stats = Counter()
    kept_stats = Counter()
    lines = []
    candidate_total = len(candidate_lines)
    for s in candidate_lines:
        s = s.strip()
        if not s:
            drop_stats["empty"] += 1
            continue
        if s in {"[image]", "[data]"}:
            drop_stats["placeholder"] += 1
            continue
        alnum = sum(ch.isalnum() for ch in s)
        if alnum < 2:
            if not re.search(r"[A-Za-z]", s):
                drop_stats["lonely_low_signal"] += 1
                continue
            kept_stats["alpha_low_signal"] += 1
            lines.append(s)
            continue
        punct = sum(not ch.isalnum() and not ch.isspace() for ch in s)
        if alnum and punct / max(1, len(s)) > 0.6:
            drop_stats["punct_ratio"] += 1
            continue
        lines.append(s)

    neat = []
    last = None
    for value in lines:
        if value != last:
            neat.append(value)
            last = value

    confidence = len(neat) / max(1, candidate_total)
    try:
        total_dropped = sum(drop_stats.values())
        if (ENABLE_SANITIZE_DEBUG_LOG or candidate_total <= SANITIZE_LOGGING_LIMIT) and (total_dropped or vertical_chunks or kept_stats):
            drop_preview = dict(list(drop_stats.items())[:6])
            kept_preview = dict(list(kept_stats.items())[:6])
            logger.info({
                "level": "INFO",
                "type": "handler",
                "message": (
                    f"[INFO] sanitize_extracted_text kept={len(neat)} dropped={total_dropped} "
                    f"vertical_chunks={vertical_chunks} confidence={confidence:.2f} "
                    f"drop_reasons={drop_preview} kept_flags={kept_preview}"
                )
            })
    except Exception:
        pass

    result = "\n".join(neat)
    # Fallback: if sanitization collapsed almost everything, prefer a gentler pass
    # to keep textual signal for downstream contest/title detection.
    if len(result) < 200 and len(text) > 5000:
        try:
            gentle_lines = []
            for raw in raw_lines:
                collapsed = re.sub(r'\s+', ' ', raw).strip()
                if collapsed:
                    gentle_lines.append(collapsed)
            fallback_result = "\n".join(gentle_lines)
            if len(fallback_result) > len(result):
                logger.info({
                    "level": "INFO",
                    "type": "handler",
                    "message": "[INFO] sanitize fallback engaged; returning gentle cleaned text.",
                })
                result = fallback_result
        except Exception:
            pass
    try:
        if cache_key:
            _sanitize_cache_set(cache_key, result, confidence)
    except Exception:
        pass
    return result


_PRECINCT_ROW_SPLIT_PATTERN = re.compile(r"\b\d{3,4}\s*-\s*[0-9A-Z]{1,3}\b")
_DENSE_LINE_MIN_LENGTH = 320


def _split_dense_precinct_line(line: str) -> list[str]:
    """Split extremely long OCR lines that contain multiple precinct blocks."""
    if not line:
        return []
    if len(line) < _DENSE_LINE_MIN_LENGTH:
        return [line]
    matches = list(_PRECINCT_ROW_SPLIT_PATTERN.finditer(line))
    if len(matches) <= 1:
        return [line]

    segments: list[str] = []
    first_start = matches[0].start()
    if first_start > 0:
        prefix = line[:first_start].strip()
        if prefix:
            segments.append(prefix)

    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(line)
        chunk = line[start:end].strip()
        if chunk:
            segments.append(chunk)

    return segments or [line]


def _explode_dense_ocr_lines(
    lines: list[str],
    line_records: list[dict] | None,
) -> tuple[list[str], list[dict] | None, bool]:
    """Expand long OCR lines into multiple entries for downstream parsing."""
    if not lines:
        return lines, line_records, False

    changed = False
    new_lines: list[str] = []
    new_records: list[dict] = []
    for idx, line in enumerate(lines):
        fragments = _split_dense_precinct_line(line)
        if len(fragments) > 1:
            changed = True
        record = line_records[idx] if line_records and idx < len(line_records) else None
        for frag_idx, fragment in enumerate(fragments):
            new_lines.append(fragment)
            if record is None:
                new_records.append({
                    "page": None,
                    "page_line_index": None,
                    "global_line_index": None,
                    "text": fragment,
                    "fragment_index": frag_idx if len(fragments) > 1 else None,
                })
            else:
                frag_record = dict(record)
                frag_record["text"] = fragment
                if len(fragments) > 1:
                    frag_record["fragment_index"] = frag_idx
                new_records.append(frag_record)

    if not changed:
        return lines, line_records, False

    for new_idx, rec in enumerate(new_records):
        rec["global_line_index"] = new_idx
    return new_lines, new_records, True


def _summarize_pages_from_records(
    line_records: list[dict] | None,
    raw_char_lookup: dict | None = None,
) -> list[dict]:
    if not line_records:
        return []
    raw_char_lookup = raw_char_lookup or {}
    summary_order: "OrderedDict[int | None, dict]" = OrderedDict()
    for idx, record in enumerate(line_records):
        page = record.get("page") if isinstance(record, dict) else None
        entry = summary_order.setdefault(page, {
            "page": page,
            "lines": 0,
            "start_index": None,
            "end_index": None,
            "raw_chars": raw_char_lookup.get(page),
        })
        entry["lines"] += 1
        if entry["start_index"] is None:
            entry["start_index"] = idx
        entry["end_index"] = idx
    return list(summary_order.values())


def _assemble_page_line_records(
    page_text_map: list[dict],
    fallback_text: str,
) -> tuple[str, list[str], list[dict], list[dict], bool]:
    """Build sanitized line records with page context.

    Returns the joined clean text, the list of lines, a parallel list of line
    records (with page and index metadata), per-page summaries, and a boolean
    indicating whether a fallback (page-agnostic) build was required.
    """

    aggregated_lines: list[str] = []
    line_records: list[dict] = []
    page_summaries: list[dict] = []

    for entry in page_text_map or []:
        page_index = entry.get("page")
        raw_text = entry.get("raw_text") or ""
        clean_page_text = _sanitize_extracted_text(raw_text)
        entry["clean_text"] = clean_page_text
        page_lines = clean_page_text.splitlines()
        start_offset = len(aggregated_lines)

        for page_line_idx, text in enumerate(page_lines):
            aggregated_lines.append(text)
            line_records.append({
                "page": page_index,
                "page_line_index": page_line_idx,
                "global_line_index": start_offset + page_line_idx,
                "text": text,
            })

        page_summaries.append({
            "page": page_index,
            "lines": len(page_lines),
            "start_index": start_offset if page_lines else None,
            "end_index": (start_offset + len(page_lines) - 1) if page_lines else None,
            "raw_chars": len(raw_text),
        })

    used_fallback = False

    if not aggregated_lines:
        fallback_clean = _sanitize_extracted_text(fallback_text)
        aggregated_lines = fallback_clean.splitlines()
        line_records = [{
            "page": None,
            "page_line_index": idx,
            "global_line_index": idx,
            "text": text,
        } for idx, text in enumerate(aggregated_lines)]
        page_summaries = [{
            "page": None,
            "lines": len(aggregated_lines),
            "start_index": 0 if aggregated_lines else None,
            "end_index": (len(aggregated_lines) - 1) if aggregated_lines else None,
            "raw_chars": len(fallback_text),
        }]
        used_fallback = True

    clean_text = "\n".join(aggregated_lines)
    return clean_text, aggregated_lines, line_records, page_summaries, used_fallback

def _pdf_to_images(
    pdf_path: str,
    session_id=None,
    *,
    dpi: int = 200,
    page_indices: list[int] | None = None,
    max_pages: int | None = None,
    cancel_flag=None,
    return_indices: bool = False,
):
    """
    Convert PDF pages to PIL Images.
    - If page_indices or max_pages provided, render only that subset (via PyMuPDF to avoid full-doc raster).
    - Else, try pdf2image (Poppler) then fallback to PyMuPDF for all pages.
    """
    orientation_map, default_angle = _get_page_orientation_map(pdf_path, session_id=session_id)
    results: list = []

    def _store(page_idx: int, pil_img: Image.Image) -> None:
        oriented = _apply_page_orientation(
            pil_img,
            page_idx,
            pdf_path,
            orientation_map,
            default_angle,
            session_id=session_id,
        )
        if return_indices:
            results.append((page_idx, oriented))
        else:
            results.append(oriented)

    # Try pdf2image first if available
    # If specific pages requested, use PyMuPDF directly (efficient random-page render)
    if page_indices is not None or max_pages is not None:
        try:
            doc = fitz.open(pdf_path)
            count = len(doc)
            if page_indices is None:
                # render first max_pages pages
                take = min(max_pages or count, count)
                idxs = list(range(take))
            else:
                idxs = sorted({i for i in page_indices if isinstance(i, int) and 0 <= i < count})
            for i in idxs:
                _ensure_not_cancelled(cancel_flag, session_id, f"render:target_page:{i}")
                page = doc[i]
                pix = page.get_pixmap(dpi=dpi)
                mode = "RGBA" if pix.alpha else "RGB"
                pil_img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
                _store(i, pil_img)
            doc.close()
            return results
        except Exception as e:
            logger.error({
                "level": "ERROR",
                "type": "handler",
                "message": f"[ERROR] Targeted page render failed: {e}",
                "session_id": session_id
            })
            results.clear()

    global _POPPLER_WARNING_SHOWN, _PDF2IMAGE_DISABLED_REASON
    # Try full-document pdf2image first if available and not previously disabled
    if pdf2image and _PDF2IMAGE_DISABLED_REASON is None and (page_indices is None and max_pages is None):
        try:
            poppler_path = _detect_poppler_path()
            is_windows = platform.system().lower().startswith("win")
            if is_windows and not poppler_path:
                if not _POPPLER_WARNING_SHOWN:
                    logger.warning({
                        "level": "WARNING",
                        "type": "handler",
                        "message": "[WARN] Poppler binaries not detected; skipping pdf2image and using PyMuPDF fallback.",
                        "session_id": session_id
                    })
                    _POPPLER_WARNING_SHOWN = True
                _PDF2IMAGE_DISABLED_REASON = "poppler_not_found"
            else:
                # Register cleanup handler for temp files
                _register_pdf_cleanup()
                
                # Track temp directory created by pdf2image
                temp_dir = tempfile.mkdtemp(prefix="pdf2image_")
                _PDF_TEMP_DIRS.add(temp_dir)
                
                kwargs = {"dpi": dpi, "output_folder": temp_dir}
                if poppler_path and is_windows:
                    kwargs["poppler_path"] = poppler_path
                
                images_raw = []
                try:
                    images_raw = pdf2image.convert_from_path(pdf_path, **kwargs)
                    if images_raw:
                        for idx, img in enumerate(images_raw):
                            # Track image refs for cleanup
                            _PDF_IMAGE_REFS.append(img)
                            _store(idx, img)
                        return results
                finally:
                    # Explicitly close images after use
                    for img in images_raw:
                        try:
                            if hasattr(img, 'close'):
                                img.close()
                        except Exception:
                            pass
        except Exception as e:
            reason = (str(e) or "pdf2image_failed").strip()
            reason_lower = reason.lower()
            disable_future = any(token in reason_lower for token in ("poppler", "nonetype", "win32", "pdftoppm", "pdftocairo"))
            if disable_future:
                _PDF2IMAGE_DISABLED_REASON = reason[:160]
            logger.warning({
                "level": "WARNING",
                "type": "handler",
                "message": (
                    "[WARN] pdf2image conversion failed; "
                    + ("disabling future attempts and " if disable_future else "")
                    + "falling back to PyMuPDF render. "
                    + f"reason={reason}"
                ),
                "session_id": session_id
            })
            results.clear()

    # Fallback: render via PyMuPDF (no Poppler needed)
    _register_pdf_cleanup()
    doc = None
    try:
        doc = fitz.open(pdf_path)
        total = len(doc)
        rng = range(total)
        if max_pages is not None:
            rng = range(min(max_pages, total))
        for i in rng:
            _ensure_not_cancelled(cancel_flag, session_id, f"render:full_pass_page:{i}")
            page = doc[i]
            # Render to pixmap at requested DPI for OCR
            pix = page.get_pixmap(dpi=dpi)
            mode = "RGBA" if pix.alpha else "RGB"
            pil_img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
            # Track for cleanup
            _PDF_IMAGE_REFS.append(pil_img)
            _store(i, pil_img)
            # Explicitly release pixmap memory
            pix = None
        if doc:
            doc.close()
            doc = None
    except Exception as e:
        logger.error({
            "level": "ERROR",
            "type": "handler",
            "message": f"[ERROR] PyMuPDF render fallback failed: {e}",
            "session_id": session_id
        })
        results.clear()
    finally:
        # Ensure doc is closed even on exception
        if doc:
            try:
                doc.close()
            except Exception:
                pass
        # Force cleanup of any partial results
        gc.collect()
    return results

def _prep_variants(images):
    """
    Yield (name, images_variant) for multiple preprocessing paths.
    """
    yield "none", images
    gray = [ImageOps.grayscale(img) for img in images]
    try:
        yield "gray", gray
        thresh = [
            ImageOps.autocontrast(ImageOps.grayscale(img)).point(
                lambda p: 255 if p > 180 else 0,
                mode='1'
            )
            for img in images
        ]
        try:
            yield "thresh", thresh
        finally:
            del thresh
        sharp = [ImageEnhance.Contrast(img.filter(ImageFilter.SHARPEN)).enhance(1.5) for img in gray]
        try:
            yield "sharp_contrast", sharp
        finally:
            del sharp
    finally:
        del gray

def _dedupe_contest_titles(titles: list[str]) -> list[str]:
    """
    Deduplicate contest-like titles by normalized token set and fuzzy containment.
    Keeps the first occurrence of near-duplicates.
    """
    def norm(s: str) -> str:
        s = (s or "").lower().strip()
        s = re.sub(r'[\s\-_\/]+', ' ', s)
        s = re.sub(r'[^a-z0-9 ]+', '', s)
        return re.sub(r'\s+', ' ', s).strip()
    seen = []
    out = []
    for t in (titles or []):
        nt = norm(t)
        if not nt:
            continue
        dup = False
        for st in seen:
            # treat as duplicate if token overlap is high or containment
            a = set(nt.split())
            b = set(st.split())
            inter = len(a & b)
            union = max(1, len(a | b))
            jacc = inter / union
            if jacc >= 0.85 or nt in st or st in nt:
                dup = True
                break
        if not dup:
            seen.append(nt)
            out.append(t)
    return out

def _ocr_images(images, tesseract_config: str, confidence_threshold=None):
    """
    Run pytesseract on a list of PIL images and return combined text and avg confidence.
    """
    if confidence_threshold is None:
        confidence_threshold = OCR_CONFIDENCE_THRESHOLD
    if not pytesseract:
        return "", 0.0, []

    page_texts = []
    confs_all = []
    # Per-page confidences for debugging
    per_page = []

    for img in images:
        text = ""
        details = {}
        if hasattr(pytesseract, "Output"):
            try:
                details = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=tesseract_config)
            except Exception:
                # Fallback to plain text if data API fails
                try:
                    text = pytesseract.image_to_string(img, config=tesseract_config)
                except Exception:
                    text = ""
            else:
                # Prefer confidences from the data API when available
                words = details.get("text", []) or []
                confs = details.get("conf", []) or []
                for j in range(len(words)):
                    word = (words[j] or "").strip()
                    conf_raw = confs[j] if j < len(confs) else "-1"
                    if word:
                        try:
                            conf_val = float(conf_raw)
                        except Exception:
                            conf_val = -1.0
                        confs_all.append(conf_val)
                        if conf_val >= confidence_threshold:
                            text += word + " "
        else:
            try:
                text = pytesseract.image_to_string(img, config=tesseract_config)
            except Exception:
                text = ""

        page_texts.append(text)
        if details:
            vals = []
            for c in details.get("conf", []):
                try:
                    vals.append(float(c))
                except Exception:
                    pass
            per_page.append(sum(vals) / len(vals) if vals else 0.0)
        else:
            per_page.append(0.0)

    avg_conf = sum(confs_all) / len(confs_all) if confs_all else 0.0
    return "\n".join(page_texts), avg_conf, per_page

def adaptive_ocr_pipeline(
    pdf_path,
    session_id=None,
    target_conf=None,
    max_seconds: int | None = None,
    max_runs=None,
    *,
    doc_page_count: int | None = None,
    page_focus_windows: list[tuple[int, int]] | None = None,
    cancel_flag=None,
    stream_time_budget: int | None = None,
    metadata_bucket: dict | None = None,
    shared_raster_cache: dict | None = None,
):
    """
    Adaptive OCR loop:
    - Try different DPIs, preprocessors, and Tesseract configs (psm/oem)
    - Keep the best result by avg confidence
    - Early stop on reaching target_conf or exceeding budgets
    Returns: best_text, best_conf, runs_summary(list of dict)
    """
    if target_conf is None:
        target_conf = OCR_AVG_CONF_ACCEPT
    if max_runs is None:
        max_runs = OCR_MAX_RUNS
    
    start = time.time()
    sample_budget = max_seconds if (isinstance(max_seconds, (int, float)) and max_seconds > 0) else None
    sample_deadline = (start + sample_budget) if sample_budget else None
    sample_timeout = False
    stream_timeout = False
    timeout_phase: str | None = None
    runs_summary = []
    best = {"text": "", "conf": 0.0, "params": {}}

    _ensure_not_cancelled(cancel_flag, session_id, "ocr:init")

    page_count = doc_page_count
    if page_count is None:
        try:
            _ensure_not_cancelled(cancel_flag, session_id, "ocr:page_probe")
            doc = fitz.open(pdf_path)
            page_count = len(doc)
            doc.close()
        except Exception:
            page_count = None

    fast_mode = os.environ.get("PDF_FAST_MODE", "0").lower() in {"1", "true", "yes"}
    if fast_mode:
        dpi_max_fast = OCR_FAST_MODE_DPI_LIMIT
        dpi_list = [d for d in range(OCR_DPI_MIN, dpi_max_fast + 1, OCR_DPI_STEP)]
        if not dpi_list:
            dpi_list = [250]
        psm_list = OCR_PSM_LIST[:3]
    elif page_count and page_count >= 180:
        # Very long docs: reduce DPI to avoid exhaustion
        dpi_list = [d for d in range(OCR_DPI_MIN, 301, OCR_DPI_STEP)]
        if not dpi_list:
            dpi_list = [250, 300]
        psm_list = OCR_PSM_LIST[:3]
    else:
        dpi_list = list(range(OCR_DPI_MIN, OCR_DPI_MAX + 1, OCR_DPI_STEP))
        if not dpi_list:
            dpi_list = [200, 250, 300, 350]
        psm_list = OCR_PSM_LIST
    oem_list = OCR_OEM_LIST
    conf_threshold_word = OCR_CONFIDENCE_THRESHOLD
    # Precompute sample page indices (first/middle/last up to 5 pages)
    sample_indices = _compute_sample_page_indices(
        page_count,
        page_windows=page_focus_windows,
        max_samples=_OCR_SAMPLE_PAGE_TARGET,
    )
    # Caches to avoid rerendering
    cache_sample = {}  # dpi -> [PIL.Image]
    logger.info({
        "level": "INFO",
        "type": "handler",
        "message": (
            "[INFO] OCR param search on sample pages "
            f"{sample_indices}; focus_windows={page_focus_windows or 'all'}; final pass on full document."
        ),
        "session_id": session_id,
    })

    exit_param_search = False
    for dpi in dpi_list:
        if (sample_deadline and time.time() > sample_deadline) or len(runs_summary) >= max_runs:
            sample_timeout = True
            timeout_phase = timeout_phase or "sample_param_search"
            break

        # Get sample images for trials (fast)
        if dpi not in cache_sample:
            rendered = list(_pdf_to_images(
                pdf_path,
                session_id=session_id,
                dpi=dpi,
                page_indices=sample_indices,
                cancel_flag=cancel_flag,
                return_indices=True,
            ))
            cache_sample[dpi] = [img for _, img in rendered]
            if shared_raster_cache is not None:
                sample_bucket = shared_raster_cache.setdefault("sample_images", {})
                sample_bucket[dpi] = list(rendered)
                shared_raster_cache.setdefault("sample_page_indices", list(sample_indices))
        images = cache_sample[dpi]
        if not images:
            continue
        logger.debug({
            "level": "DEBUG",
            "type": "handler",
            "message": f"[DEBUG] OCR trials at dpi={dpi} on {len(images)} sample page(s).",
            "session_id": session_id
        })

        for prep_name, prep_imgs in _prep_variants(images):
            _ensure_not_cancelled(cancel_flag, session_id, f"ocr:prep:{prep_name}@{dpi}")
            if (sample_deadline and time.time() > sample_deadline) or len(runs_summary) >= max_runs:
                sample_timeout = True
                timeout_phase = timeout_phase or "sample_param_search"
                exit_param_search = True
                break

            for oem in oem_list:
                for psm in psm_list:
                    _ensure_not_cancelled(cancel_flag, session_id, f"ocr:trial:dpi{dpi}")
                    if (sample_deadline and time.time() > sample_deadline) or len(runs_summary) >= max_runs:
                        sample_timeout = True
                        timeout_phase = timeout_phase or "sample_param_search"
                        exit_param_search = True
                        break
                    config = f"--oem {oem} --psm {psm}"
                    text, avg_conf, per_page = _ocr_images(prep_imgs, config, confidence_threshold=conf_threshold_word)

                    # Record run
                    run = {
                        "dpi": dpi,
                        "prep": prep_name,
                        "oem": oem,
                        "psm": psm,
                        "avg_conf": round(avg_conf, 2),
                        "per_page": [round(c, 2) for c in per_page]
                    }
                    runs_summary.append(run)
                    if len(runs_summary) % 5 == 0:
                        logger.info({
                            "level": "INFO",
                            "type": "handler",
                            "message": f"[INFO] OCR trials progress: {len(runs_summary)} run(s), best={round(best['conf'],2)} conf",
                            "session_id": session_id
                        })
                    # Update best
                    if avg_conf > best["conf"]:
                        best = {"text": text, "conf": avg_conf, "params": {"dpi": dpi, "prep": prep_name, "oem": oem, "psm": psm}}

                    # Early stop when good enough
                    if avg_conf >= target_conf:
                        exit_param_search = True
                        break
                if exit_param_search:
                    break
            if exit_param_search:
                break
        if exit_param_search:
            break

    # Combine high-confidence lines across top runs to improve recall
    if runs_summary:
        # sort by confidence
        top = sorted(runs_summary, key=lambda r: r["avg_conf"], reverse=True)[:5]
        # Re-run OCR quickly for those top settings to collect lines
        line_sets = []
        for r in top:
            _ensure_not_cancelled(cancel_flag, session_id, "ocr:topline")
            # Use sample images for quick combination
            if r["dpi"] not in cache_sample:
                rendered = list(_pdf_to_images(
                    pdf_path,
                    session_id=session_id,
                    dpi=r["dpi"],
                    page_indices=sample_indices,
                    cancel_flag=cancel_flag,
                    return_indices=True,
                ))
                cache_sample[r["dpi"]] = [img for _, img in rendered]
                if shared_raster_cache is not None:
                    sample_bucket = shared_raster_cache.setdefault("sample_images", {})
                    sample_bucket[r["dpi"]] = list(rendered)
            imgs = cache_sample.get(r["dpi"]) or []
            if not imgs:
                continue
            # reapply preprocess
            prep_variants = dict(_prep_variants(imgs))
            imgs2 = prep_variants.get(r["prep"], imgs)
            txt, _, _ = _ocr_images(imgs2, f"--oem {r['oem']} --psm {r['psm']}", confidence_threshold=conf_threshold_word)
            line_sets.append(set((txt or "").splitlines()))
        if line_sets:
            combined = sorted(set.union(*line_sets))
            combined_text = "\n".join(combined)
            # Keep the better of combined vs. best raw
            if len(combined_text) > len(best["text"]):
                best["text"] = combined_text

    # Final assurance: run a streaming full-document pass with the best params (covers all pages without loading entire PDF)
    try:
        params = best.get("params") or {}
        if params and sample_deadline and time.time() > sample_deadline and not stream_time_budget:
            sample_timeout = True
            timeout_phase = timeout_phase or "sample_param_search"
            logger.warning({
                "level": "WARNING",
                "type": "handler",
                "message": "[WARN] Skipping full-document OCR pass due to expired sample budget.",
                "session_id": session_id,
            })
            params = {}
        if params:
            dpi = params.get("dpi", 300)
            cfg = f"--oem {params.get('oem', 3)} --psm {params.get('psm', 6)}"
            chunk_pages = _OCR_FULLDOC_BATCH_PAGES
            max_pages = _OCR_FULLDOC_MAX_PAGES if _OCR_FULLDOC_MAX_PAGES > 0 else None
            text_fragments: list[str] = []
            total_pages_rendered = 0
            chunk_counter = 0
            stream_start = time.time()
            stream_deadline = (stream_start + stream_time_budget) if stream_time_budget else None

            for batch_start, images in _yield_full_pass_batches(
                pdf_path,
                dpi=dpi,
                session_id=session_id,
                batch_pages=chunk_pages,
                max_pages=max_pages,
                page_windows=page_focus_windows if page_focus_windows else None,
                cancel_flag=cancel_flag,
            ):
                if not images:
                    continue
                prep_variants = dict(_prep_variants(images))
                imgs2 = prep_variants.get(params.get("prep", "none"), images)
                chunk_text, _, _ = _ocr_images(imgs2, cfg, confidence_threshold=conf_threshold_word)
                if chunk_text:
                    text_fragments.append(chunk_text)
                total_pages_rendered += len(images)
                chunk_counter += 1

                if chunk_counter % 5 == 0 or len(images) >= chunk_pages:
                    logger.info({
                        "level": "INFO",
                        "type": "handler",
                        "message": (
                            "[INFO] OCR full-pass progress: "
                            f"chunks={chunk_counter}, pages_rendered={total_pages_rendered}"
                        ),
                        "session_id": session_id,
                    })

                if stream_deadline and time.time() > stream_deadline:
                    stream_timeout = True
                    timeout_phase = timeout_phase or "stream_full_pass"
                    logger.warning({
                        "level": "WARNING",
                        "type": "handler",
                        "message": "[WARN] Aborting full-document OCR pass due to timeout budget.",
                        "session_id": session_id,
                    })
                    break

                _ensure_not_cancelled(
                    cancel_flag,
                    session_id,
                    f"ocr:stream:dpi{dpi}:chunk{chunk_counter}",
                )

            text_full = "\n".join(fragment for fragment in text_fragments if fragment)
            if text_full and len(text_full) > len(best["text"] or ""):
                best["text"] = text_full

            logger.info({
                "level": "INFO",
                "type": "handler",
                "message": (
                    "[INFO] Streaming OCR full-document pass complete "
                    f"(pages_rendered={total_pages_rendered}, chunks={chunk_counter}, dpi={params.get('dpi')}, "
                    f"prep={params.get('prep')}, oem={params.get('oem')}, psm={params.get('psm')})."
                ),
                "session_id": session_id,
            })

            if max_pages and total_pages_rendered >= max_pages:
                logger.warning({
                    "level": "WARNING",
                    "type": "handler",
                    "message": (
                        "[WARN] Full-document OCR pass truncated due to OCR_FULLDOC_MAX_PAGES limit. "
                        f"Processed {total_pages_rendered} / {max_pages} pages."
                    ),
                    "session_id": session_id,
                })
    except Exception:
        logger.debug({
            "level": "DEBUG",
            "type": "handler",
            "message": "[DEBUG] Streaming OCR full-document pass failed; continuing with best partial text.",
            "session_id": session_id,
        })
    total_runtime = time.time() - start
    diag_entry = {
        "sample_budget_seconds": sample_budget,
        "stream_budget_seconds": stream_time_budget,
        "sample_timeout": sample_timeout,
        "stream_timeout": stream_timeout,
        "timeout_phase": timeout_phase,
        "total_seconds": round(total_runtime, 2),
        "runs_attempted": len(runs_summary),
    }
    if metadata_bucket is not None:
        attempts = metadata_bucket.setdefault("ocr_attempts", [])
        attempts.append(diag_entry)
        metadata_bucket["ocr_timeout_triggered"] = metadata_bucket.get("ocr_timeout_triggered", False) or bool(timeout_phase)
        if timeout_phase:
            metadata_bucket["ocr_timeout_phase"] = timeout_phase
        metadata_bucket["ocr_time_spent_seconds"] = diag_entry["total_seconds"]
    return best["text"], best["conf"], runs_summary, best.get("params", {})

def ocr_multi_pass(images, passes=3, confidence_threshold=30, session_id=None):
    ocr_runs = []
    pass_confidences = []

    def process_image_ocr(img):
        page_text = ""
        confidences = []
        if pytesseract:
            details = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT) if hasattr(pytesseract, "Output") else {}
            for j in range(len(details.get("text", []))):
                word = details["text"][j].strip()
                conf = details["conf"][j]
                if word:
                    try:
                        conf_val = float(conf)
                        confidences.append(conf_val)
                        if conf_val >= confidence_threshold:
                            page_text += word + " "
                    except ValueError:
                        continue
        return page_text, confidences

    for i in range(passes):
        logger.info({
            "level": "INFO",
            "type": "handler",
            "message": f"[INFO] OCR pass {i+1} of {passes}",
            "session_id": session_id
        })
        ocr_text = ""
        confidences = []
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(process_image_ocr, images))
        for text, conf_list in results:
            ocr_text += text + "\n"
            confidences.extend(conf_list)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        pass_confidences.append(avg_conf)
        ocr_runs.append(ocr_text)

    line_sets = [set(text.splitlines()) for text in ocr_runs]
    combined_lines = sorted(set.union(*line_sets))
    all_text = "\n".join(combined_lines)
    overall_avg = sum(pass_confidences) / len(pass_confidences) if pass_confidences else 0.0
    return all_text, overall_avg, ocr_runs

def _extract_text_multi(pdf_path, session_id=None):
    """Try multiple PyMuPDF extract modes and return the richest text + per-page map."""
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.warning({
            "level": "WARNING",
            "type": "handler",
            "message": f"[WARN] Multi-mode text extraction failed: {e}",
            "session_id": session_id
        })
        return "", "error", []

    modes = ["text", "raw", "html", "xhtml"]
    best_text = ""
    best_mode = None
    best_page_map: list[dict] = []
    try:
        page_total = len(doc)
        for mode in modes:
            buf: list[str] = []
            page_entries: list[dict] = []
            for page_index in range(page_total):
                try:
                    page_text = doc[page_index].get_text(mode)
                except Exception:
                    page_text = ""
                if not isinstance(page_text, str):
                    page_text = ""
                buf.append(page_text)
                page_entries.append({
                    "page": page_index,
                    "raw_text": page_text,
                    "char_count": len(page_text),
                })
            combined = "\n".join(buf)
            if len(combined) > len(best_text):
                best_text = combined
                best_mode = mode
                best_page_map = page_entries
    finally:
        try:
            doc.close()
        except Exception:
            pass

    return best_text, best_mode or "text", best_page_map

def _save_ocr_debug_images(pdf_path, session_id=None, dpi=300, limit=2, cancel_flag=None):
    try:
        _ensure_not_cancelled(cancel_flag, session_id, "ocr:debug_raster")
        # Render only first N pages instead of rasterizing the entire document
        idxs = list(range(max(0, limit)))
        imgs = _pdf_to_images(
            pdf_path,
            session_id=session_id,
            dpi=dpi,
            page_indices=idxs,
            cancel_flag=cancel_flag,
        )
        saved = []
        base = safe_slug(os.path.basename(pdf_path))
        for idx, img in enumerate(imgs):
            out = os.path.join(
                OCR_DEBUG_DIR,
                f"{base}_p{idx+1}_{dpi}dpi.png"
            )
            try:
                img.save(out)
                saved.append(str(out))
            except Exception:
                pass
        if saved:
            logger.info({
                "level": "INFO",
                "type": "handler",
                "message": "[DEBUG] Saved OCR debug raster(s)",
                "session_id": session_id,
                "files": saved
            })
    except Exception:
        pass


def _write_debug_text(pdf_path: str, text: str | None, suffix: str, session_id=None) -> str | None:
    """Persist sanitized/diagnostic OCR text for manual inspection."""
    if not text:
        return None
    try:
        base = safe_slug(os.path.splitext(os.path.basename(pdf_path))[0])
        out_file = os.path.join(OCR_DEBUG_DIR, f"{base}__{suffix}.txt")
        with open(out_file, "w", encoding="utf-8") as handle:
            handle.write(text)
        logger.debug({
            "level": "DEBUG",
            "type": "handler",
            "message": f"[DEBUG] Wrote OCR debug text ({suffix}) to {out_file}",
            "session_id": session_id
        })
        return out_file
    except Exception:
        return None


_LAYOUT_HEADER_KEYWORDS = {
    "candidate", "party", "district", "ward", "precinct", "total", "votes", "vote",
    "absentee", "affidavit", "military", "emergency", "machine", "counter", "early",
    "mail", "provisional", "ballots", "election", "day", "overall", "write", "count"
}

_NUMERIC_VALUE_RE = re.compile(r"^-?\d[\d,]*(?:\.\d+)?%?$")


def _header_token_score(line_text: str) -> int:
    tokens = _token_set(line_text)
    return sum(1 for tok in tokens if tok in _LAYOUT_HEADER_KEYWORDS)


def _line_has_digits(words: list[dict]) -> bool:
    for w in words:
        if _NUM_RE.search(w.get("text", "")):
            return True
    return False


def _line_is_candidate_only(line_text: str) -> bool:
    if not line_text:
        return False
    digits = sum(ch.isdigit() for ch in line_text)
    letters = sum(ch.isalpha() for ch in line_text)
    return digits == 0 and letters >= 6


def _group_words_by_gaps(words: list[dict], gap_threshold: int = 28) -> list[list[dict]]:
    groups: list[list[dict]] = []
    current: list[dict] = []
    prev_right: float | None = None
    for word in sorted(words, key=lambda w: w.get("left", 0)):
        if prev_right is not None and word.get("left", 0) - prev_right > gap_threshold:
            if current:
                groups.append(current)
            current = []
        current.append(word)
        prev_right = word.get("right", word.get("left", 0))
    if current:
        groups.append(current)
    return groups


def _build_layout_columns(header_words: list[dict], image_width: int) -> list[dict]:
    grouped = _group_words_by_gaps(header_words)
    columns: list[dict] = []
    seen: set[str] = set()
    for idx, group in enumerate(grouped):
        raw_label = " ".join(w.get("text", "").strip() for w in group if w.get("text"))
        clean_label = re.sub(r"\s+", " ", raw_label).strip() or f"Column {idx + 1}"
        base_key = clean_label.lower()
        if base_key in seen:
            suffix = 2
            while f"{base_key}__{suffix}" in seen:
                suffix += 1
            clean_label = f"{clean_label} {suffix}"
            base_key = f"{base_key}__{suffix}"
        seen.add(base_key)
        left = min(w.get("left", 0) for w in group)
        right = max((w.get("right") or (w.get("left", 0) + w.get("width", 0))) for w in group)
        columns.append({
            "label": clean_label,
            "key": clean_label,
            "left": max(0, left - 4),
            "right": right + 6,
        })
    if not columns:
        return []
    columns[0]["left"] = 0
    columns[-1]["right"] = max(image_width, int(columns[-1]["right"]))
    return columns


def _assign_words_to_columns(columns: list[dict], words: list[dict]) -> dict:
    row: dict[str, str] = {col["label"]: "" for col in columns}
    if not columns:
        return row
    for word in sorted(words, key=lambda w: w.get("left", 0)):
        text = (word.get("text") or "").strip()
        if not text:
            continue
        center = word.get("center_x")
        if center is None:
            center = word.get("left", 0) + word.get("width", 0) / 2
        target = None
        for col in columns:
            if col["left"] - 2 <= center <= col["right"] + 2:
                target = col
                break
        if target is None:
            target = columns[0] if center < columns[0]["left"] else columns[-1]
        existing = row[target["label"]]
        row[target["label"]] = f"{existing} {text}".strip() if existing else text
    return row


def _clean_numeric_cell(value: str) -> str | int:
    if not isinstance(value, str):
        return value
    raw = value.strip()
    if not raw:
        return ""
    pct = False
    if raw.endswith("%"):
        pct = True
        raw = raw[:-1]
    cleaned = raw.replace(",", "").strip()
    if cleaned.isdigit():
        num = int(cleaned)
        return f"{num}%" if pct else num
    return value


def _split_key_value_line(text: str) -> tuple[str, str] | None:
    """Split a 'Label value' line emitted by statement-and-return style pages."""
    if not text:
        return None
    parts = text.rsplit(" ", 1)
    if len(parts) != 2:
        return None
    label, value = parts[0].strip().rstrip(":"), parts[1].strip()
    if not label or not value:
        return None
    if not _NUMERIC_VALUE_RE.match(value):
        return None
    return label, value


def _finalize_layout_table(columns: list[dict], rows: list[dict]) -> tuple[list[str], list[dict]]:
    headers = [col["label"] for col in columns]
    normalized_rows: list[dict] = []
    for row in rows:
        normalized = {}
        for col in columns:
            val = row.get(col["label"], "")
            if col is columns[0]:
                normalized[col["label"]] = re.sub(r"\s+", " ", val.strip())
                continue
            normalized[col["label"]] = _clean_numeric_cell(val)
        normalized_rows.append(normalized)
    return headers, normalized_rows


def _merge_layout_tables(tables: list[dict]) -> list[dict]:
    if not tables:
        return tables
    buckets: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for table in tables:
        key = tuple(h.lower() for h in table.get("headers", []))
        buckets[key].append(table)
    merged: list[dict] = []
    for key, group in buckets.items():
        if len(group) == 1:
            merged.append(group[0])
            continue
        rows: list[dict] = []
        for item in group:
            rows.extend(item.get("rows", []))
        merged.append({
            "headers": group[0].get("headers", []),
            "rows": rows,
            "pages": sorted({g.get("page") for g in group if g.get("page") is not None}),
            "score": sum(len(g.get("rows", [])) for g in group)
        })
    merged.sort(key=lambda t: (-len(t.get("rows", [])), -len(t.get("headers", []))))
    return merged


def _extract_tables_via_layout(
    pdf_path: str,
    session_id=None,
    ocr_params: dict | None = None,
    max_pages: int | None = None,
    page_indices: list[int] | None = None,
    pre_rendered: list[tuple[int, Image.Image]] | None = None,
    *,
    cancel_flag=None,
):
    if not pytesseract or not _PANDAS_AVAILABLE:
        return []
    try:
        tess_output = pytesseract.Output.DATAFRAME
    except Exception:
        return []

    dpi = int((ocr_params or {}).get("dpi", 300))
    oem = (ocr_params or {}).get("oem", 1)
    psm = (ocr_params or {}).get("psm", 6)
    config = f"--oem {oem} --psm {psm} -c preserve_interword_spaces=1"

    layout_tables: list[dict] = []
    seen_pages: set[int] = set()

    reused_pages: list[tuple[int, Image.Image]] = []
    if pre_rendered:
        target = set(page_indices) if page_indices else None
        for idx, image in pre_rendered:
            if idx is None:
                continue
            if target is not None and idx not in target:
                continue
            if idx in seen_pages:
                continue
            seen_pages.add(idx)
            reused_pages.append((idx, image))

    def _iter_remaining():
        remaining_indices = None
        remaining_max = None
        if page_indices:
            remaining_indices = [i for i in page_indices if i not in seen_pages]
            if not remaining_indices:
                return
        elif max_pages is not None:
            remaining = max_pages - len(seen_pages)
            if remaining <= 0:
                return
            remaining_max = remaining
        yield from _pdf_to_images(
            pdf_path,
            session_id=session_id,
            dpi=dpi,
            max_pages=remaining_max,
            page_indices=remaining_indices,
            cancel_flag=cancel_flag,
            return_indices=True,
        )

    def _process_page(actual_page, image):
        page_index = actual_page if isinstance(actual_page, int) else 0
        _ensure_not_cancelled(cancel_flag, session_id, f"layout:page:{page_index}")
        try:
            df = pytesseract.image_to_data(image, output_type=tess_output, config=config)
        except Exception as exc:
            logger.debug({
                "level": "DEBUG",
                "type": "handler",
                "message": f"[DEBUG] Tesseract DATAFRAME extraction failed on page {page_index}: {exc}",
                "session_id": session_id
            })
            return

        if df is None or df.empty:
            return
        try:
            df = df[df["conf"].fillna(-1) > -1]
            df["text"] = df["text"].fillna("").astype(str).str.strip()
            df = df[df["text"] != ""]
        except Exception:
            return
        if df.empty:
            return

        df["right"] = df["left"] + df["width"]
        df["center_x"] = df["left"] + (df["width"] / 2)
        df["center_y"] = df["top"] + (df["height"] / 2)

        line_records = []
        group_cols = ["page_num", "block_num", "par_num", "line_num"]
        try:
            grouped = df.groupby(group_cols)
        except Exception:
            return

        for (_, _, _, line_num), group in grouped:
            if group.empty:
                continue
            words: list[dict] = []
            for _, row in group.sort_values("left").iterrows():
                text = str(row.get("text", "")).strip()
                if not text:
                    continue
                left = int(row.get("left", 0))
                width = int(row.get("width", 0))
                words.append({
                    "text": text,
                    "left": left,
                    "right": left + width,
                    "width": width,
                    "center_x": float(row.get("center_x", left + width / 2)),
                })
            if not words:
                continue
            text = " ".join(w["text"] for w in words)
            line_records.append({
                "words": words,
                "text": text,
                "page": page_index,
                "avg_top": float(group["top"].mean()),
                "image_width": image.width,
            })

        line_records.sort(key=lambda item: (item["page"], item["avg_top"]))

        current_table: dict | None = None
        pending_candidate_words: list[dict] | None = None
        blank_streak = 0

        for line in line_records:
            text = line["text"]
            words = line["words"]
            header_score = _header_token_score(text)
            digits_present = _line_has_digits(words)

            if header_score >= 2 and not _is_bad_header_line(text):
                if current_table and current_table.get("rows"):
                    headers, rows = _finalize_layout_table(current_table["columns"], current_table["rows"])
                    layout_tables.append({
                        "headers": headers,
                        "rows": rows,
                        "page": current_table.get("page"),
                        "score": len(rows)
                    })
                columns = _build_layout_columns(words, line["image_width"])
                if len(columns) < 2:
                    current_table = None
                    pending_candidate_words = None
                    continue
                current_table = {
                    "columns": columns,
                    "rows": [],
                    "page": page_index,
                }
                pending_candidate_words = None
                blank_streak = 0
                continue

            if not current_table:
                continue

            if not text.strip():
                blank_streak += 1
                if blank_streak >= 2 and current_table.get("rows"):
                    headers, rows = _finalize_layout_table(current_table["columns"], current_table["rows"])
                    layout_tables.append({
                        "headers": headers,
                        "rows": rows,
                        "page": current_table.get("page"),
                        "score": len(rows)
                    })
                    current_table = None
                    pending_candidate_words = None
                continue

            blank_streak = 0

            if digits_present:
                active_words = list(words)
                if pending_candidate_words:
                    active_words = pending_candidate_words + active_words
                row = _assign_words_to_columns(current_table["columns"], active_words)
                if any(row.get(col["label"]) for col in current_table["columns"]):
                    current_table["rows"].append(row)
                pending_candidate_words = None
            else:
                if _line_is_candidate_only(text):
                    pending_candidate_words = list(words)
                    continue
                if current_table["rows"]:
                    first_col = current_table["columns"][0]["label"]
                    existing = current_table["rows"][-1].get(first_col, "")
                    current_table["rows"][-1][first_col] = f"{existing} {text}".strip()

        if current_table and current_table.get("rows"):
            headers, rows = _finalize_layout_table(current_table["columns"], current_table["rows"])
            layout_tables.append({
                "headers": headers,
                "rows": rows,
                "page": current_table.get("page"),
                "score": len(rows)
            })

    for actual_page, image in reused_pages:
        _process_page(actual_page, image)

    for actual_page, image in _iter_remaining() or []:
        _process_page(actual_page, image)

    return _merge_layout_tables(layout_tables)


_STATEMENT_AD_RE = re.compile(r"assembly\s+district\s+(\d+)", re.I)
_STATEMENT_ED_RE = re.compile(r"election\s+district\s+(\d+)", re.I)
_STATEMENT_PRECINCT_RE = re.compile(r"precinct\s+(\d+)", re.I)

_STATEMENT_SUMMARY_KEYWORDS = (
    "public counter",
    "manually counted",
    "absentee",
    "military",
    "affidavit",
    "total ballots",
    "total votes",
    "total vote",
    "total applicable",
    "less - inapplicable",
    "inapplicable",
    "unrecorded",
    "vote for",
    "scanner",
    "ballots cast",
    "ballots counted",
)

def _identify_statement_location_columns(headers: list[str]) -> list[str]:
    return collect_location_headers(headers or [], ensure_precinct=False)


def _statement_value_has_payload(value: object) -> bool:
    if value in (None, ""):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return False
        core = (
            stripped.replace(",", "")
            .replace("%", "")
            .replace("-", "")
            .replace("(", "")
            .replace(")", "")
        )
        if core.replace(".", "", 1).isdigit():
            return True
        return len(stripped) >= 3
    return True


def _coerce_statement_numeric(value: object) -> object:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        try:
            return int(value)
        except Exception:
            return value
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        if cleaned.replace(".", "", 1).isdigit():
            try:
                return int(float(cleaned))
            except Exception:
                return value
    return value


def _remap_statement_summary_header(header: str) -> str:
    low = (header or "").strip().lower()
    if "public counter" in low:
        return "Precinct Public Counter"
    if "manually counted" in low and "emergency" in low:
        return "Precinct Emergency Ballots"
    if "absentee" in low or "military" in low:
        return "Precinct Absentee Military"
    if "affidavit" in low:
        return "Precinct Affidavit"
    if "total applicable" in low:
        return "Precinct Total Applicable Ballots"
    if "less - inapplicable" in low or "inapplicable" in low:
        return "Precinct Inapplicable Ballots"
    if low == "total vote" or "total votes" in low:
        return "Precinct Total Votes"
    if "total ballots" in low:
        return "Precinct Total Ballots"
    if "unrecorded" in low:
        return "Precinct Unrecorded"
    return header


def _is_statement_summary_header(header: str) -> bool:
    if is_strict_location_header(header):
        return False
    low = (header or "").strip().lower()
    if not low:
        return False
    return any(token in low for token in _STATEMENT_SUMMARY_KEYWORDS)


def _parse_statement_candidate_header(header: str) -> tuple[str, str | None, str | None]:
    text = re.sub(r"\s+", " ", (header or "").strip())
    candidate_type = "Write-In" if re.search(r"write[-\s]?in", text, re.I) else "Candidate"
    party = None

    without_write_in = re.sub(r"\s*\(write[-\s]?in\)\s*", "", text, flags=re.I)
    without_write_in = re.sub(r"/\s*write[-\s]?in", "", without_write_in, flags=re.I)

    party_match = re.search(r"\(([^)]+)\)$", without_write_in)
    if party_match and not re.search(r"write", party_match.group(1), re.I):
        party = party_match.group(1).strip()
        candidate_label = without_write_in[: party_match.start()].strip()
    else:
        candidate_label = without_write_in.strip()

    if candidate_label.isupper():
        candidate_label = candidate_label.title()

    return candidate_label or text, candidate_type, party


def _normalize_statement_candidate_results(
    headers: list[str],
    rows: list[dict],
) -> tuple[list[str], list[dict], dict]:
    if not headers or not rows:
        return [], [], {}

    location_cols = _identify_statement_location_columns(headers)
    if "Precinct" not in location_cols:
        location_cols = ["Precinct", *location_cols]
    summary_cols = [h for h in headers if _is_statement_summary_header(h)]
    base_cols = [h for h in location_cols if h in headers or h == "Precinct"]
    candidate_cols = [h for h in headers if h not in base_cols and h not in summary_cols]

    if not candidate_cols:
        return [], [], {}

    summary_targets: dict[str, str] = {}
    summary_order: list[str] = []
    for col in summary_cols:
        mapped = _remap_statement_summary_header(col)
        summary_targets[col] = mapped
        if mapped not in summary_order:
            summary_order.append(mapped)

    normalized_rows: list[dict] = []
    for rec in rows:
        base = {col: rec.get(col, "") for col in base_cols}
        for cand in candidate_cols:
            value = rec.get(cand, "")
            if not _statement_value_has_payload(value):
                continue
            candidate_label, candidate_type, candidate_party = _parse_statement_candidate_header(cand)
            normalized = dict(base)
            normalized["Candidate"] = candidate_label
            if candidate_type:
                normalized["Candidate Type"] = candidate_type
            if candidate_party:
                normalized["Party"] = candidate_party
            normalized["Votes"] = _coerce_statement_numeric(value)
            for src, dest in summary_targets.items():
                summary_val = rec.get(src, "")
                if not _statement_value_has_payload(summary_val):
                    continue
                normalized[dest] = _coerce_statement_numeric(summary_val)
            normalized_rows.append(normalized)

    if not normalized_rows:
        return [], [], {}

    header_candidates = []
    if "Precinct" in location_cols:
        header_candidates.append("Precinct")
    for loc in location_cols:
        if loc != "Precinct":
            header_candidates.append(loc)
    header_candidates.append("Candidate")
    if any(row.get("Candidate Type") for row in normalized_rows):
        header_candidates.append("Candidate Type")
    if any(row.get("Party") for row in normalized_rows):
        header_candidates.append("Party")
    header_candidates.append("Votes")
    header_candidates.extend(h for h in summary_order if h not in header_candidates)

    keep_headers: list[str] = []
    for header in header_candidates:
        if header in {"Precinct", "Assembly District", "Election District", "Candidate", "Votes"}:
            keep_headers.append(header)
            continue
        if header in {"Candidate Type", "Party"}:
            if any(row.get(header) for row in normalized_rows):
                keep_headers.append(header)
            continue
        if any(_statement_value_has_payload(row.get(header)) for row in normalized_rows):
            keep_headers.append(header)

    finalized_rows = [{h: row.get(h, "") for h in keep_headers} for row in normalized_rows]

    diagnostics = {
        "source_headers": len(headers),
        "candidate_columns": len(candidate_cols),
        "summary_columns": len(summary_cols),
        "normalized_rows": len(finalized_rows),
        "location_headers": location_cols,
    }

    return keep_headers, finalized_rows, diagnostics


def _extract_statement_return_blocks(
    pdf_path: str,
    session_id=None,
    ocr_params: dict | None = None,
    max_pages: int | None = None,
    page_indices: list[int] | None = None,
    pre_rendered: list[tuple[int, Image.Image]] | None = None,
    *,
    cancel_flag=None,
):
    """Parse statement & return style PDF pages into structured key/value rows."""
    if not pytesseract:
        return [], []

    dpi = int(max(360, (ocr_params or {}).get("dpi", 300)))
    oem = (ocr_params or {}).get("oem", 3)
    config = f"--oem {oem} --psm 4 -c preserve_interword_spaces=1"

    seen_pages: set[int] = set()

    reused_pages: list[tuple[int, Image.Image]] = []
    if pre_rendered:
        target = set(page_indices) if page_indices else None
        for idx, image in pre_rendered:
            if idx is None:
                continue
            if target is not None and idx not in target:
                continue
            if idx in seen_pages:
                continue
            seen_pages.add(idx)
            reused_pages.append((idx, image))

    remaining_indices = None
    remaining_max = None
    images_iter = None
    if page_indices:
        remaining_indices = [i for i in page_indices if i not in seen_pages]
        if remaining_indices:
            images_iter = _pdf_to_images(
                pdf_path,
                session_id=session_id,
                dpi=dpi,
                page_indices=remaining_indices,
                cancel_flag=cancel_flag,
                return_indices=True,
            )
    else:
        if max_pages is not None:
            remaining_max = max_pages - len(reused_pages)
            if remaining_max > 0:
                images_iter = _pdf_to_images(
                    pdf_path,
                    session_id=session_id,
                    dpi=dpi,
                    max_pages=remaining_max,
                    cancel_flag=cancel_flag,
                    return_indices=True,
                )
        else:
            images_iter = _pdf_to_images(
                pdf_path,
                session_id=session_id,
                dpi=dpi,
                cancel_flag=cancel_flag,
                return_indices=True,
            )

    records_map: dict[tuple[str, str, str], dict] = {}
    current_record: dict[str, object] | None = None
    current_ad: str | None = None
    current_ed: str | None = None

    def _ensure_record() -> bool:
        nonlocal current_record
        if current_record is None:
            if not (current_ad or current_ed):
                return False
            current_record = {"Assembly District": current_ad or ""}
            if current_ed:
                current_record["Election District"] = current_ed
        else:
            if current_ad and not current_record.get("Assembly District"):
                current_record["Assembly District"] = current_ad
            if current_ed and not current_record.get("Election District"):
                current_record["Election District"] = current_ed
        return True

    def commit_record():
        nonlocal current_record
        if not current_record:
            return
        ad = str(current_record.get("Assembly District") or "").strip()
        ed = str(current_record.get("Election District") or "").strip()
        precinct_label = str(current_record.get("Precinct") or "").strip()
        if not ad:
            current_record = None
            return
        has_payload = any(
            _statement_value_has_payload(value)
            for key, value in current_record.items()
            if key not in {"Assembly District", "Election District", "Precinct"}
        )
        if not has_payload and not ed and not precinct_label:
            current_record = None
            return
        key = (ad, ed, precinct_label)
        bucket = records_map.setdefault(key, {"Assembly District": ad})
        if ed:
            bucket["Election District"] = ed
        if precinct_label:
            bucket["Precinct"] = precinct_label
        for k, v in current_record.items():
            if isinstance(v, str) and not v.strip():
                continue
            if v in (None, ""):
                continue
            bucket[k] = v
        current_record = None

    def _process_statement_page(actual_page, image):
        nonlocal current_ad, current_ed, current_record
        page_index = actual_page if isinstance(actual_page, int) else 0
        _ensure_not_cancelled(cancel_flag, session_id, f"statement_blocks:page:{page_index}")
        try:
            df = pytesseract.image_to_data(image, output_type=pytesseract.Output.DATAFRAME, config=config)
        except Exception as exc:
            logger.debug({
                "level": "DEBUG",
                "type": "handler",
                "message": f"[DEBUG] Statement block OCR failed on page {page_index}: {exc}",
                "session_id": session_id
            })
            return

        if df is None or df.empty:
            return
        df = df[df["text"].notna()]
        df["text"] = df["text"].astype(str).str.strip()
        df = df[df["text"] != ""]
        if df.empty:
            return

        df = df.assign(center_x=df["left"] + (df["width"] / 2))
        group_cols = ["page_num", "block_num", "par_num", "line_num"]
        try:
            grouped = df.groupby(group_cols)
        except Exception:
            return

        # Sort by layout ordering
        lines = []
        for _, group in grouped:
            words = []
            for _, row in group.sort_values("left").iterrows():
                token = str(row.get("text", "")).strip()
                if not token:
                    continue
                words.append(token)
            if not words:
                continue
            text = re.sub(r"\s+", " ", " ".join(words)).strip()
            if text:
                top = float(group["top"].mean())
                lines.append((top, text))

        lines.sort(key=lambda item: item[0])

        for _, raw_text in lines:
            text = raw_text.strip()
            if not text:
                continue
            # Skip boilerplate page indicators
            low = text.lower()
            if low.startswith("page ") and " of " in low:
                continue

            ad_match = _STATEMENT_AD_RE.search(text)
            if ad_match:
                commit_record()
                current_ad = ad_match.group(1)
                current_ed = None
                current_record = {"Assembly District": current_ad}
                continue

            if current_ad is None and current_record is None:
                # Ignore until we know which assembly the block belongs to
                continue

            ed_match = _STATEMENT_ED_RE.search(text)
            if ed_match:
                commit_record()
                current_ed = ed_match.group(1)
                current_record = {"Assembly District": current_ad or "", "Election District": current_ed}
                continue

            prec_match = _STATEMENT_PRECINCT_RE.search(text)
            if prec_match:
                if _ensure_record():
                    # Check if text contains aggregated precinct info with "/"
                    if "/" in text.lower() and "precinct" in text.lower():
                        # Extract all precinct numbers from aggregated strings like "AD 37 / Precinct 71 / Precinct 652"
                        precinct_nums = re.findall(r'precinct\s+(\d+)', text, re.I)
                        if len(precinct_nums) > 1:
                            current_record["Precinct"] = " / ".join(precinct_nums)
                        else:
                            current_record["Precinct"] = prec_match.group(1)
                    else:
                        current_record["Precinct"] = prec_match.group(1)
                continue

            key_val = _split_key_value_line(text)
            if not key_val:
                continue

            label, value_token = key_val
            parsed_value = _clean_numeric_cell(value_token)
            if _ensure_record():
                current_record[label] = parsed_value

        # Continue accumulating across pages; do not commit yet to allow multi-page sections

    saw_page = False
    if reused_pages:
        for actual_page, image in reused_pages:
            saw_page = True
            _process_statement_page(actual_page, image)

    if images_iter:
        for actual_page, image in images_iter:
            saw_page = True
            _process_statement_page(actual_page, image)

    if not saw_page:
        return [], []

    commit_record()

    if not records_map:
        return [], []

    records = list(records_map.values())

    if not records:
        return [], []

    def _sort_key(rec: dict) -> tuple:
        ad_raw = str(rec.get("Assembly District", "")).strip()
        ed_raw = str(rec.get("Election District", "")).strip()
        def _coerce(val: str) -> tuple[int, str, str]:
            stripped = (val or "").strip()
            if stripped.isdigit():
                padded = stripped.zfill(6)
                return (0, padded, stripped)
            return (1, stripped.lower(), stripped)
        return (_coerce(ad_raw), _coerce(ed_raw))

    try:
        records.sort(key=_sort_key)
    except Exception:
        records.sort(key=lambda r: (str(r.get("Assembly District", "")), str(r.get("Election District", ""))))

    ordered_headers: list[str] = []
    priority = ["Precinct", "Assembly District", "Election District"]
    for key in priority:
        ordered_headers.append(key)
    seen = {h for h in ordered_headers}
    for rec in records:
        for key in rec.keys():
            if key not in seen:
                ordered_headers.append(key)
                seen.add(key)

    rows = []
    for rec in records:
        row = {h: rec.get(h, "") for h in ordered_headers}
        rows.append(row)

    return ordered_headers, rows


def _attach_statement_precinct(
    headers: list[str],
    rows: list[dict],
    *,
    location_headers: list[str] | None = None,
) -> tuple[list[str], list[dict], bool]:
    """Ensure statement rows expose a precinct-like label for downstream pivots."""
    if not rows:
        return headers, rows, False

    sanitized_extras = [
        header.strip()
        for header in (location_headers or [])
        if isinstance(header, str) and header.strip()
    ]

    detected = collect_location_headers(
        headers or [],
        ensure_precinct=True,
        extra_headers=sanitized_extras,
    )

    updated_headers, updated_rows, added_any = attach_precinct_column(
        list(headers or []),
        rows,
        location_headers=detected,
        column_name="Precinct",
    )

    return updated_headers, updated_rows, added_any


def _should_prefer_statement_blocks(
    headers: list[str] | None,
    rows: list[dict] | None,
    *,
    camelot_tables: list[dict] | None = None,
    layout_tables: list[dict] | None = None,
) -> bool:
    if not rows:
        return False
    populated = sum(1 for row in rows if any(str(val).strip() for val in row.values()))
    if len(rows) >= 5 and populated >= max(3, len(rows) // 3):
        return True
    if (not camelot_tables) and (not layout_tables) and populated:
        return True
    return False


def _finalize_structured_table_output(
    pdf_path: str,
    headers: list[str],
    data: list[dict],
    selected_contest_title: str,
    state: str,
    county: str,
    year: int | None,
    contest_slug: str,
    metadata: dict,
    coordinator,
    session_id=None,
):
    statement_used = bool(metadata.get("statement_blocks_used"))
    precinct_attached = False
    if statement_used:
        diag_info = metadata.get("statement_blocks_normalized")
        location_headers = list(diag_info.get("location_headers") or []) if isinstance(diag_info, dict) else []
        headers, data, precinct_attached = _attach_statement_precinct(
            headers,
            data,
            location_headers=location_headers,
        )

    headers, data = normalize_table_headers(headers, data)

    headers_adj, data_adj = harmonize_headers_and_data(
        headers,
        data,
        context={
            "contest": selected_contest_title,
            "state": state,
            "county": county,
        }
    )

    domain = safe_slug(os.path.basename(pdf_path))
    context = {
        "contest": selected_contest_title,
        "state": state,
        "county": county,
        "year": year,
        "session_id": session_id,
        "handler": "pdf_handler",
        "ocr_confidence_avg": metadata.get("ocr_confidence_avg"),
        "ocr_used": metadata.get("ocr_used"),
        "contest_slug": contest_slug,
        "source_slug": domain
    }

    if statement_used:
        context.setdefault("include_all_precincts_row", False)
        context["skip_pivot"] = True
        context["skip_row_noise_filter"] = True
        if precinct_attached:
            context.setdefault("precinct_sort", "natural")
        metadata["statement_blocks_precinct_attached"] = precinct_attached

    headers_exp, data_exp = expand_single_rawjson_row(headers_adj, data_adj, context=context)

    headers_final, data_final, _entity_info = build_table_noninteractive(
        domain=domain,
        headers=headers_exp,
        data=data_exp,
        coordinator=coordinator,
        context=context,
        pivot_to_wide=True,
        debug=False
    )

    export_context = _prepare_output_context(
        context,
        {
            "handler": "pdf_handler",
            "input_file": os.path.basename(pdf_path),
            "session_id": session_id,
            "ocr_confidence_avg": metadata.get("ocr_confidence_avg"),
            "ocr_used": metadata.get("ocr_used"),
            "ocr_evidence": metadata.get("ocr_evidence"),
        }
    )

    result = finalize_election_output(
        headers=headers_final,
        data=data_final,
        coordinator=coordinator,
        contest=selected_contest_title,
        state=state,
        county=county,
        context=export_context,
        enable_user_feedback=False,
        session_id=session_id
    )

    metadata.update({
        "output_file": os.path.basename(result.get("csv_path", "")),
        "headers": headers_final,
        "row_count": len(data_final),
        "csv_path": result.get("csv_path"),
        "metadata_path": result.get("metadata_path"),
    })

    logger.info({
        "level": "INFO",
        "type": "output",
        "message": f"[OUTPUT] Wrote {len(data_final)} rows to: {result.get('csv_path')}",
        "session_id": session_id
    })
    logger.info({
        "level": "INFO",
        "type": "output",
        "message": f"[OUTPUT] Metadata written to: {result.get('metadata_path')}",
        "session_id": session_id
    })

    return headers_final, data_final, result

def infer_headers_and_methods(lines, table_hints):
    hints = {h.lower() for h in (table_hints or []) if isinstance(h, str)}
    headers, header_idx = _find_header_line(lines, hints)
    return headers, (lines[header_idx] if header_idx >= 0 else "")

def _should_force_ocr(raw_text: str, clean_text: str) -> bool:
    """
    Force OCR if the sanitized signal is too small relative to raw text length,
    even if fitz returned non-empty content (common with markup/xhtml dumps).
    """
    try:
        raw_len = len(raw_text or "")
        clean_len = len(clean_text or "")
        if raw_len == 0:
            return False
        # Force when clean text is tiny compared to raw or trivially short
        ratio = clean_len / max(1, raw_len)
        return (clean_len < 500 and raw_len > 100_000) or ratio < 0.005
    except Exception:
        return False

def _should_auto_select(titles: list[str]) -> bool:
    """
    True if multiple detected titles are effectively near-duplicates
    (i.e., one contest repeated across the PDF).
    Uses token-set Jaccard similarity across all pairs.
    """
    def norm_tokens(s: str) -> set[str]:
        s = (s or "").lower()
        s = re.sub(r'[^a-z0-9 ]+', '', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return set(s.split())
    if not titles or len(titles) <= 1:
        return True
    toks = [norm_tokens(t) for t in titles if isinstance(t, str)]
    toks = [t for t in toks if t]
    if len(toks) <= 1:
        return True
    # Require high similarity across all pairs
    thresh = 0.80
    for i in range(len(toks)):
        for j in range(i + 1, len(toks)):
            a, b = toks[i], toks[j]
            inter = len(a & b)
            union = max(1, len(a | b))
            jacc = inter / union
            if jacc < thresh:
                return False
    return True

def _pick_representative_title(titles: list[str]) -> str:
    """
    Pick a stable representative from near-duplicates:
    prefer shortest meaningful title with max token overlap to others.
    """
    def norm_tokens(s: str) -> set[str]:
        s = (s or "").lower()
        s = re.sub(r'[^a-z0-9 ]+', '', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return set(s.split())
    if not titles:
        return ""
    toks = [norm_tokens(t) for t in titles]
    # Score: average token overlap with others, then length as tiebreaker
    best = (float("-inf"), "")
    for idx, t in enumerate(titles):
        a = toks[idx]
        if not a:
            continue
        score = 0.0
        for j, b in enumerate(toks):
            if j == idx or not b:
                continue
            inter = len(a & b)
            union = max(1, len(a | b))
            score += inter / union
        # Prefer longer representative on tie
        key = (score, len(t.strip()))
        if key > (best[0], -len(best[1].strip())):
            best = (score, t)
    return best[1] or titles[0]

def _dedupe_contest_titles(titles):
    return list(dict.fromkeys(titles))
   
def parse_pdf_election_results(pdf_path, session_id=None, coordinator=None, cancel_flag=None) -> tuple[list[str], list[dict], str, dict]:
    """ Main PDF handler function."""
    # Log active OCR tuning config at parse start for diagnostics
    from ...config import log_ocr_config_summary, get_ocr_config_dict, log_extraction_quality  # type: ignore[attr-defined]
    from ... import config as cfg_module
    log_ocr_config_summary(cfg_module, logger, session_id=session_id)
    
    def _finalize_with_quality(headers, data, contest, metadata):
        """Wrapper to add ML quality logging before returning results."""
        # Add quality metrics to metadata
        quality = log_extraction_quality(
            headers, data, metadata, "pdf_handler", logger, session_id
        )
        metadata["quality_metrics"] = quality
        return headers, data, contest, metadata
    
    _log_ocr_environment(session_id=session_id)
    all_text = ""
    page_text_map: list[dict] = []
    metadata = {"ocr_config": get_ocr_config_dict(cfg_module)}
    tried_associated = False
    headers = []
    ocr_score = 0.0
    ocr_runs = []
    pdf_page_total: int | None = None
    contest_probe_info: dict[str, Any] = {}
    page_focus_windows: list[tuple[int, int]] | None = None
    probe_preselect: dict[str, Any] | None = None
    shared_raster_cache: dict[str, Any] = {}

    _ensure_not_cancelled(cancel_flag, session_id, "pdf:bootstrap")

    def _ensure_contest_focus():
        nonlocal contest_probe_info, page_focus_windows
        # Fast mode: limit to first N pages to accelerate OCR/debug
        fast_mode = str(os.environ.get("PDF_FAST_MODE", "")).lower() in {"1", "true", "yes"}
        if fast_mode:
            try:
                n = int(os.environ.get("PDF_FAST_PAGES", "5"))
            except Exception:
                n = 5
            total = pdf_page_total or n
            end_page = max(1, min(n, int(total)))
            page_focus_windows = [(1, end_page)]
            return
        if page_focus_windows is not None:
            return
        if not pdf_page_total or pdf_page_total < _OCR_CONTEST_PROBE_MIN_PAGES:
            page_focus_windows = None
            return
        _ensure_not_cancelled(cancel_flag, session_id, "pdf:contest_probe_init")
        if not contest_probe_info:
            # Allow env override to cap probe workload
            max_pages_override = os.environ.get("PDF_PROBE_MAX_PAGES")
            try:
                max_pages_override = int(max_pages_override) if max_pages_override else None
            except Exception:
                max_pages_override = None
            contest_probe_info = _contest_probe_scan(
                pdf_path,
                session_id=session_id,
                max_pages=max_pages_override or _OCR_CONTEST_PROBE_MAX_PAGES,
                stride=_OCR_CONTEST_PROBE_STRIDE,
                dpi=_OCR_CONTEST_PROBE_DPI,
                max_hits=_OCR_CONTEST_PROBE_MAX_HITS,
                cancel_flag=cancel_flag,
            ) or {}
            if contest_probe_info:
                metadata["contest_probe"] = contest_probe_info
        hits = contest_probe_info.get("hits") or []
        if hits:
            page_focus_windows = _expand_focus_windows(hits, pdf_page_total)
        else:
            page_focus_windows = []

    def _apply_probe_preselection():
        nonlocal page_focus_windows, probe_preselect
        if probe_preselect is not None:
            return
        if not pdf_page_total or not contest_probe_info:
            return
        _ensure_not_cancelled(cancel_flag, session_id, "pdf:contest_probe_selection")
        picked = _autopick_contest_from_probe(
            pdf_path,
            contest_probe_info,
            coordinator=coordinator,
            session_id=session_id,
        )
        if not picked:
            return
        probe_preselect = picked
        metadata["contest_probe_autopick"] = picked
        refined = _refine_focus_windows_for_contest(
            picked.get("title"),
            contest_probe_info,
            pdf_page_total,
            expand=_OCR_FOCUS_WINDOW_EXPAND,
        )
        if refined:
            page_focus_windows = refined
            metadata["ocr_focus_windows"] = refined

    def _page_indices_from_windows(windows: list[tuple[int, int]] | None, limit: int | None = None) -> list[int]:
        if not windows:
            return []
        indices: list[int] = []
        for start, end in windows:
            if start is None or end is None:
                continue
            cursor = max(0, int(start))
            stop = max(cursor, int(end))
            while cursor < stop:
                if limit is not None and len(indices) >= limit:
                    return indices
                indices.append(cursor)
                cursor += 1
        return sorted(set(indices))

    # Try standard text first
    try:
        doc = fitz.open(pdf_path)
        pdf_page_total = len(doc)
        for i in range(pdf_page_total):
            _ensure_not_cancelled(cancel_flag, session_id, f"pdf:text_extract:{i}")
            page_text = doc[i].get_text()
            if not isinstance(page_text, str):
                page_text = str(page_text or "")
            all_text += page_text
            page_text_map.append({
                "page": i,
                "raw_text": page_text,
                "char_count": len(page_text),
            })
        doc.close()
    except Exception as e:
        logger.warning({
            "level": "WARNING",
            "type": "handler",
            "message": f"[WARN] fitz text extraction failed: {e}",
            "session_id": session_id
        })
        all_text = ""
        page_text_map = []

    # If empty or forced, try alternative extract modes
    if (not all_text.strip()) or ENABLE_OCR_FORCE:
        alt_text, mode_used, alt_page_map = _extract_text_multi(pdf_path, session_id=session_id)
        if len(alt_text) > len(all_text):
            all_text = alt_text
            metadata["fitz_mode_used"] = mode_used
            if alt_page_map:
                page_text_map = alt_page_map

    # If the "text" is markup-only, treat as empty to force OCR
    if _is_mostly_markup(all_text):
        logger.info({
            "level": "INFO",
            "type": "handler",
            "message": "[INFO] Detected markup-only PDF text — switching to OCR.",
            "session_id": session_id
        })
        all_text = ""

    metadata["pdf_page_total"] = pdf_page_total
    ocr_sample_budget, ocr_stream_budget = _estimate_ocr_time_budgets(pdf_page_total)
    metadata["ocr_time_budget"] = {
        "sample_seconds": ocr_sample_budget,
        "stream_seconds": ocr_stream_budget,
        "pdf_pages": pdf_page_total,
    }

    # OCR fallback (adaptive, cross‑platform)
    has_text = bool((all_text or "").strip())
    need_ocr = (not has_text) and bool(pytesseract) and ENABLE_OCR

    # If forcing OCR but pytesseract is unavailable, log and skip the loop
    if (not pytesseract) and ENABLE_OCR_FORCE:
        logger.warning({
            "level": "WARNING",
            "type": "handler",
            "message": "[WARN] ENABLE_OCR_FORCE is set but Tesseract is unavailable; skipping OCR fallback.",
            "session_id": session_id
        })

    if need_ocr or (ENABLE_OCR_FORCE and pytesseract):
        if not has_text:
            logger.info({
                "level": "INFO",
                "type": "handler",
                "message": "[INFO] Empty/forced OCR — attempting adaptive OCR fallback.",
                "session_id": session_id
            })
        _ensure_contest_focus()
        _apply_probe_preselection()
        _save_ocr_debug_images(
            pdf_path,
            session_id=session_id,
            dpi=300,
            limit=2,
            cancel_flag=cancel_flag,
        )
        best_text, best_conf, runs_summary, ocr_params = adaptive_ocr_pipeline(
            pdf_path,
            session_id=session_id,
            target_conf=70.0,
            max_seconds=ocr_sample_budget,
            max_runs=28,
            doc_page_count=pdf_page_total,
            page_focus_windows=page_focus_windows if page_focus_windows else None,
            cancel_flag=cancel_flag,
            stream_time_budget=ocr_stream_budget,
            metadata_bucket=metadata,
            shared_raster_cache=shared_raster_cache,
        )
        candidate_ocr_params = dict(ocr_params or {})
        if candidate_ocr_params:
            metadata["ocr_params"] = candidate_ocr_params
        # Prefer OCR text if we had none, or if significantly better
        if (not has_text) or (len(best_text) > len(all_text) * 1.25):
            all_text = best_text or all_text
            ocr_score = best_conf or 0.0
            ocr_runs = runs_summary or []
            metadata["ocr_confidence_avg"] = round(ocr_score, 2)
            metadata["ocr_runs"] = ocr_runs
            metadata["ocr_used"] = True
            metadata["ocr_params"] = dict(ocr_params or {})
        else:
            metadata["ocr_used"] = False
        if page_focus_windows:
            metadata["ocr_focus_windows"] = page_focus_windows

    clean_text = _sanitize_extracted_text(all_text)

    # Force OCR when sanitized signal is very low vs raw fitz text (markup dump case)
    low_signal_force = False
    try_low_signal = _should_force_ocr(all_text, clean_text)
    ocr_already_used = bool(metadata.get("ocr_used"))
    if try_low_signal:
        low_signal_force = True
        metadata["ocr_low_signal"] = True
        if not pytesseract or not ENABLE_OCR:
            # Record why we can't run OCR in this scenario
            metadata["ocr_used"] = False
            metadata["ocr_reason"] = "low_signal_but_ocr_disabled"
            logger.warning({
                "level": "WARNING",
                "type": "handler",
                "message": "[WARN] Low-signal text detected but OCR is unavailable or disabled.",
                "session_id": session_id
            })
        elif not ocr_already_used:
            logger.info({
                "level": "INFO",
                "type": "handler",
                "message": "[INFO] Low-signal text detected from fitz (markup-heavy). Forcing OCR.",
                "session_id": session_id
            })
            _save_ocr_debug_images(
                pdf_path,
                session_id=session_id,
                dpi=300,
                limit=2,
                cancel_flag=cancel_flag,
            )
            _ensure_contest_focus()
            _apply_probe_preselection()
            best_text, best_conf, runs_summary, ocr_params = adaptive_ocr_pipeline(
                pdf_path,
                session_id=session_id,
                target_conf=70.0,
                max_seconds=ocr_sample_budget,
                max_runs=18,
                doc_page_count=pdf_page_total,
                page_focus_windows=page_focus_windows if page_focus_windows else None,
                cancel_flag=cancel_flag,
                stream_time_budget=ocr_stream_budget,
                metadata_bucket=metadata,
                shared_raster_cache=shared_raster_cache,
            )
            candidate_ocr_params = dict(ocr_params or {})
            if candidate_ocr_params:
                metadata["ocr_params"] = candidate_ocr_params
            if best_text:
                all_text = best_text
                clean_text = _sanitize_extracted_text(all_text)
                ocr_score = best_conf or 0.0
                ocr_runs = runs_summary or []
                metadata["ocr_confidence_avg"] = round(ocr_score, 2)
                metadata["ocr_runs"] = ocr_runs
                metadata["ocr_used"] = True
                metadata["ocr_params"] = dict(ocr_params or {})
                metadata["ocr_reason"] = "low_signal_fitx_markup"
            else:
                metadata["ocr_used"] = False
                metadata["ocr_reason"] = "low_signal_ocr_no_text"
            if page_focus_windows:
                metadata["ocr_focus_windows"] = page_focus_windows
        else:
            # OCR already performed earlier; do not run twice
            logger.info({
                "level": "INFO",
                "type": "handler",
                "message": "[INFO] Low-signal detected, but OCR already completed earlier. Skipping second OCR.",
                "session_id": session_id
            })

    clean_text, lines, line_records, page_summaries, page_lines_fallback = _assemble_page_line_records(
        page_text_map,
        all_text or "",
    )

    if not clean_text:
        fallback_label = os.path.splitext(os.path.basename(pdf_path))[0]
        clean_text = fallback_label
        lines = clean_text.splitlines()
        line_records = [{
            "page": None,
            "page_line_index": idx,
            "global_line_index": idx,
            "text": text,
        } for idx, text in enumerate(lines)]
        page_summaries = [{
            "page": None,
            "lines": len(lines),
            "start_index": 0 if lines else None,
            "end_index": (len(lines) - 1) if lines else None,
            "raw_chars": len(clean_text),
        }]
        page_lines_fallback = True

    raw_char_lookup = {
        entry.get("page"): entry.get("raw_chars")
        for entry in page_summaries or []
        if isinstance(entry, dict)
    }
    lines, line_records, dense_split_applied = _explode_dense_ocr_lines(lines, line_records)
    if dense_split_applied:
        clean_text = "\n".join(lines)
        page_summaries = _summarize_pages_from_records(line_records, raw_char_lookup)
        metadata["ocr_dense_lines_split"] = True

    metadata["page_line_total"] = len(line_records)
    metadata["page_line_source"] = "fallback" if page_lines_fallback else "page_map"
    metadata["page_line_pages"] = len(page_summaries)
    metadata["page_line_index_available"] = bool(line_records)
    metadata["page_lines_fallback"] = bool(page_lines_fallback)
    if page_summaries:
        metadata["page_line_summary"] = page_summaries[:25]
    _record_page_text_structure_observation(
        pdf_page_total=pdf_page_total,
        line_records=line_records,
        page_summaries=page_summaries,
        page_lines_fallback=page_lines_fallback,
        page_text_map=page_text_map,
        fitz_mode=metadata.get("fitz_mode_used"),
    )

    metadata["ocr_evidence"] = _build_ocr_evidence(
        pdf_path=pdf_path,
        fitz_mode=metadata.get("fitz_mode_used"),
        page_text_map=page_text_map,
        fitz_text_len=sum((entry.get("char_count") or 0) for entry in (page_text_map or [])),
        clean_text=clean_text,
        raw_text=all_text,
        ocr_used=bool(metadata.get("ocr_used")),
        ocr_confidence_avg=metadata.get("ocr_confidence_avg"),
        ocr_params=metadata.get("ocr_params"),
        ocr_runs=metadata.get("ocr_runs"),
        page_summaries=page_summaries,
        line_records=line_records,
        metadata=metadata,
    )

    try:
        logger.info({
            "level": "INFO",
            "type": "handler",
            "message": f"[INFO] Text lengths — raw={len(all_text or '')}, clean={len(clean_text or '')} \n Please wait a moment for data to compile.",
            "session_id": session_id
        })
    except Exception:
        pass

    if clean_text:
        debug_path = _write_debug_text(pdf_path, clean_text, "clean", session_id=session_id)
        if debug_path:
            metadata["ocr_clean_text_path"] = debug_path
    if all_text and len(all_text) <= 2_500_000:
        raw_debug_path = _write_debug_text(pdf_path, all_text, "raw", session_id=session_id)
        if raw_debug_path:
            metadata["ocr_raw_text_path"] = raw_debug_path

    logger.debug({
        "level": "DEBUG",
        "type": "handler",
        "message": "[DEBUG] PDF extracted text preview (first 500 chars):" + (
            clean_text[:500] if isinstance(clean_text, str) else str(clean_text)[:500]
        ),
        "session_id": session_id
    })

    ocr_params_raw = metadata.get("ocr_params")
    ocr_params = ocr_params_raw if isinstance(ocr_params_raw, dict) else {}
    layout_dpi = int(max(250, ocr_params.get("dpi", 300)))
    sample_image_cache = {}
    if isinstance(shared_raster_cache, dict):
        cached_samples = shared_raster_cache.get("sample_images")
        if isinstance(cached_samples, dict):
            sample_image_cache = cached_samples

    table_hints = list(
        set(LOCATION_KEYWORDS) | set(CANDIDATE_KEYWORDS) | set(BALLOT_TYPES) |
        set(PARTY_KEYWORDS) | set(TOTAL_KEYWORDS) | set(MISC_FOOTER_KEYWORDS) | set(CONTEST_KEYWORDS)
    )
    _ensure_not_cancelled(cancel_flag, session_id, "pdf:table_candidates")
    camelot_tables = attempt_camelot_extraction(pdf_path, session_id=session_id)
    layout_tables: list[dict] = []
    layout_focus_indices = _page_indices_from_windows(
        page_focus_windows,
        limit=_LAYOUT_SCAN_PAGE_LIMIT,
    )
    layout_max_pages = None
    if layout_focus_indices:
        metadata["layout_focus_pages"] = layout_focus_indices[:50]
    elif pdf_page_total and pdf_page_total > _LAYOUT_SCAN_PAGE_LIMIT:
        layout_max_pages = _LAYOUT_SCAN_PAGE_LIMIT
        metadata["layout_scan_limited"] = {
            "limit": _LAYOUT_SCAN_PAGE_LIMIT,
            "total_pages": pdf_page_total,
        }
    layout_tables = _extract_tables_via_layout(
        pdf_path,
        session_id=session_id,
        ocr_params=ocr_params,
        max_pages=layout_max_pages,
        page_indices=layout_focus_indices or None,
        pre_rendered=sample_image_cache.get(layout_dpi),
        cancel_flag=cancel_flag,
    )

    statement_headers: list[str] | None = None
    statement_rows: list[dict] | None = None
    statement_focus_indices = _page_indices_from_windows(
        page_focus_windows,
        limit=_STATEMENT_SCAN_PAGE_LIMIT,
    )
    statement_max_pages = None
    if statement_focus_indices:
        metadata["statement_focus_pages"] = statement_focus_indices[:50]
    elif pdf_page_total and pdf_page_total > _STATEMENT_SCAN_PAGE_LIMIT:
        statement_max_pages = _STATEMENT_SCAN_PAGE_LIMIT
        metadata["statement_scan_limited"] = {
            "limit": _STATEMENT_SCAN_PAGE_LIMIT,
            "total_pages": pdf_page_total,
        }
    statement_dpi = int(max(360, ocr_params.get("dpi", 300)))
    statement_headers, statement_rows = _extract_statement_return_blocks(
        pdf_path,
        session_id=session_id,
        ocr_params=ocr_params,
        max_pages=statement_max_pages,
        page_indices=statement_focus_indices or None,
        pre_rendered=sample_image_cache.get(statement_dpi),
        cancel_flag=cancel_flag,
    )
    statement_headers = statement_headers or []
    statement_rows = statement_rows or []

    statement_headers_raw = list(statement_headers or [])
    statement_rows_raw = [dict(row) for row in statement_rows] if statement_rows else []
    statement_headers_norm, statement_rows_norm, statement_norm_diag = _normalize_statement_candidate_results(
        statement_headers_raw,
        [dict(row) for row in statement_rows_raw],
    )

    statement_headers_use = statement_headers_raw
    statement_rows_use = statement_rows_raw
    if statement_rows_norm:
        statement_headers_use = statement_headers_norm
        statement_rows_use = statement_rows_norm
        diag = dict(statement_norm_diag or {})
        diag["raw_rows"] = len(statement_rows_raw)
        metadata["statement_blocks_normalized"] = diag

    statement_headers_copy = list(statement_headers_use)
    statement_rows_copy = [dict(row) for row in statement_rows_use]
    statement_headers = list(statement_headers_use)
    statement_rows = [dict(row) for row in statement_rows_use]
    if layout_tables:
        metadata["layout_tables_available"] = [
            {
                "headers": tbl.get("headers", [])[:8],
                "rows": len(tbl.get("rows", [])),
                "page": tbl.get("page"),
            }
            for tbl in layout_tables[:5]
        ]
    if statement_rows_copy:
        metadata["statement_blocks_available"] = {
            "rows": len(statement_rows_copy),
            "headers": statement_headers_copy[:10]
        }
        metadata["statement_blocks_diagnostic"] = {
            "normalized_rows": len(statement_rows_copy),
            "raw_rows": len(statement_rows_raw)
        }

    statement_priority = _should_prefer_statement_blocks(
        statement_headers_copy,
        statement_rows_copy,
        camelot_tables=camelot_tables,
        layout_tables=layout_tables,
    )

    table_candidates = []
    for entry in camelot_tables or []:
        table_candidates.append({
            'page': entry.get('page', 0),
            'headers': entry.get('headers', []),
            'data': entry.get('rows', []),
            'source': 'camelot'
        })
    for entry in layout_tables or []:
        table_candidates.append({
            'page': entry.get('page', 0),
            'headers': entry.get('headers', []),
            'data': entry.get('rows', []),
            'source': 'layout'
        })

    headers, header_candidate = infer_headers_and_methods(lines, table_hints)

    # Detect potential contests from text as hints
    contest_detection_diag: dict[str, Any] = {}
    detected_titles = detect_contest_titles_from_text(
        lines,
        pdf_path,
        diagnostics=contest_detection_diag,
    )
    # Deduplicate aggressively – single-race PDFs often repeat the same heading
    detected_titles = _dedupe_contest_titles(detected_titles)
    if contest_detection_diag:
        contest_detection_diag["dedup_titles"] = detected_titles[:25]
        contest_detection_diag["dedup_count"] = len(detected_titles)
        metadata["contest_detection"] = contest_detection_diag
    probe_titles = contest_probe_info.get("titles") if contest_probe_info else []
    if probe_titles:
        for title in probe_titles:
            if title and title not in detected_titles:
                detected_titles.append(title)
    if probe_preselect and probe_preselect.get("title"):
        preferred = probe_preselect["title"]
        if preferred in detected_titles:
            detected_titles = [preferred] + [t for t in detected_titles if t != preferred]
        else:
            detected_titles.insert(0, preferred)
    if not detected_titles:
        detected_titles = [os.path.basename(pdf_path).replace(".pdf", "")]

    text_focus_windows = _focus_windows_from_line_records(
        line_records,
        detected_titles,
        pdf_page_total,
        expand=_OCR_FOCUS_WINDOW_EXPAND,
        limit_windows=12,
    )
    if text_focus_windows:
        combined_windows = _merge_focus_windows(page_focus_windows, text_focus_windows)
        if combined_windows:
            page_focus_windows = combined_windows
            metadata["ocr_focus_windows"] = page_focus_windows[:50]
        if probe_preselect is None and len(detected_titles) == 1:
            probe_preselect = {
                "title": detected_titles[0],
                "confidence": None,
                "source": "text_detection",
            }
            metadata["contest_text_autopick"] = probe_preselect

    # Derive light context from filename for better selection (before prompting)
    fname = os.path.basename(pdf_path).lower()
    parsed_location = parse_filename_for_location(fname)
    state = parsed_location.get("state", "Unknown")
    county = parsed_location.get("county", "Unknown")
    year = parsed_location.get("year")
    state_normalized = None
    county_normalized = None

    # Single contest fast-path or unified selector pass (no duplicate prompts)
    selector_context = {
        "selector_data": {
            "contests": [{"title": t} for t in detected_titles],
            "noisy_patterns": [s.lower() for s in (CONTEST_TITLE_SKIP_PHRASES or set())]
        },
        "input_file": os.path.basename(pdf_path),
        "selector_options": {
            "allow_prompt": _CONTEST_PROMPTS_ENABLED,
            "fallback_title": detected_titles[0] if detected_titles else None,
        },
    }
    if probe_preselect:
        selector_context["probe_preselect"] = probe_preselect
    force_contest_prompt = bool(os.environ.get("SMART_ELECTIONS_FORCE_CONTEST_PROMPT"))
    single_contest_detected = len(detected_titles) == 1
    selected_contest_title: str | None = None
    shortcut_statement = (
        statement_priority
        and not force_contest_prompt
        and single_contest_detected
    )
    if shortcut_statement:
        selected_contest_title = detected_titles[0]
        metadata["contest_selection_mode"] = "statement_priority"
    elif single_contest_detected and not force_contest_prompt:
        selected_contest_title = detected_titles[0]
        metadata["contest_selection_mode"] = "single_detected"
    else:
        allow_parallel_auto = ENABLE_PARALLEL and not force_contest_prompt
        auto_kwargs = {
            "coordinator": coordinator,
            "context": selector_context,
            "session_id": session_id,
            "allow_multiple": False,
            "force_interactive": force_contest_prompt,
            "prefer_year_match": False,
        }
        if allow_parallel_auto:
            if _should_auto_select(detected_titles):
                auto_kwargs["auto_confidence_threshold"] = 0.0
            else:
                auto_kwargs["auto_confidence_threshold"] = 0.82
        auto_pick = select_contest_auto_first(**auto_kwargs)
        mode = "auto" if (allow_parallel_auto and auto_pick) else "prompt"
        metadata["contest_selection_mode"] = mode
        if not auto_pick:
            if mode == "auto":
                logger.warning({
                    "level": "WARNING",
                    "type": "handler",
                    "message": "[WARN] Auto contest selection failed in batch mode; falling back to filename.",
                    "session_id": session_id
                })
            selected_contest_title = os.path.basename(pdf_path).replace(".pdf", "")
        else:
            selected_contest_title = safe_get(auto_pick[0], "title") or detected_titles[0]
    if not selected_contest_title:
        selected_contest_title = os.path.basename(pdf_path).replace(".pdf", "")
    if probe_preselect and probe_preselect.get("title"):
        metadata["contest_probe_autopick_match"] = (
            probe_preselect["title"].strip().lower()
            == (selected_contest_title or "").strip().lower()
        )
    
    # Override with formatted filename if available and detected
    filename_formatted = None
    year = None
    if pdf_path:
        parsed_location = parse_filename_for_location(os.path.basename(pdf_path))
        contest = parsed_location.get('contest', '')
        location = parsed_location.get('location', '')
        year = parsed_location.get('year')
        if contest and location:
            filename_formatted = f"{contest} ({location})"
    if filename_formatted and filename_formatted in detected_titles:
        selected_contest_title = filename_formatted
    contest_slug = safe_slug(selected_contest_title, 80)

    if statement_priority and statement_rows_copy:
        metadata["statement_blocks_used"] = True
        metadata["statement_blocks_rows"] = len(statement_rows_copy)
        metadata["statement_blocks_reason"] = "priority_preempt"
        metadata["table_source"] = "statement"
        _record_table_stage(metadata, "statement_priority", {"rows": len(statement_rows_copy)})
        headers_final, data_final, _ = _finalize_structured_table_output(
            pdf_path,
            list(statement_headers_copy),
            [dict(row) for row in statement_rows_copy],
            selected_contest_title,
            state,
            county,
            year,
            contest_slug,
            metadata,
            coordinator,
            session_id=session_id,
        )
        return _finalize_with_quality(headers_final, data_final, selected_contest_title, metadata)
    
    # Special handling for New York location
    if location == "New York":
        county = "New York"    # Update metadata using derived context
    metadata.update({
        "source_file": os.path.basename(pdf_path),
        "state": state,
        "county": county,
        "handler": "pdf_handler",
        "contest": selected_contest_title,
        "state_normalized": state_normalized,
        "county_normalized": county_normalized,
    })

    location_context = {
        "state": state,
        "county": county,
        "contest": selected_contest_title,
        "input_file": os.path.basename(pdf_path),
    }

    try:
        det_county, det_state, _handler_path, det_log = dynamic_state_county_detection(
            location_context,
            clean_text,
            debug=False
        )
        location_log = metadata.setdefault("location_detection_log", [])
        if det_state:
            state_normalized = det_state
            formatted_state = format_state_label(det_state)
            if formatted_state:
                state = formatted_state
        if det_county:
            county_normalized = det_county
            formatted_county = format_county_label(det_county, det_state or state_normalized or state)
            if formatted_county:
                county = formatted_county
        if det_log:
            location_log.extend(det_log)
    except Exception:
        pass

    normalized_state_hint = state_normalized or normalize_state_name(state)
    placeholder_county = (county or "").strip().lower()
    county_missing = (not county) or placeholder_county in {"unknown", "n/a", "na", "unspecified"}
    if county_missing and normalized_state_hint:
        inferred_county_norm, county_hits = infer_county_from_lines(normalized_state_hint, lines)
        if inferred_county_norm:
            county_normalized = inferred_county_norm
            formatted_county = format_county_label(
                inferred_county_norm,
                normalized_state_hint,
            ) or inferred_county_norm.replace("_", " ").title()
            county = formatted_county
            log_entry = (
                f"County '{formatted_county}' inferred from text scan for state '{normalized_state_hint}'"
                f" (hits={county_hits})."
            )
            metadata.setdefault("location_detection_log", []).append(log_entry)
            metadata["location_detection_source"] = "county_text_scan"

    metadata.update({
        "state": state,
        "county": county,
        "state_normalized": state_normalized,
        "county_normalized": county_normalized,
    })

    contest_column = None
    if headers:
        # Auto-detect a contest-like column; avoid interactive re-prompt
        contest_header_keywords = CONTEST_HEADER_KEYWORDS
        candidates = [h for h in headers if any(kw in h.lower() for kw in contest_header_keywords)]

        if len(candidates) == 1:
            contest_column = candidates[0]
            logger.info({
                "level": "INFO",
                "type": "handler",
                "message": f"[INFO] Auto-selected contest column: {contest_column}",
                "session_id": session_id
            })
        elif len(candidates) > 1:
            # Rank by preference order from constants
            pref = CONTEST_HEADER_PREFERENCE
            def rank(h):
                low = h.lower()
                for i, kw in enumerate(pref):
                    if kw in low:
                        return i
                return len(pref)
            candidates.sort(key=rank)
            if rank(candidates[0]) < rank(candidates[1]):
                contest_column = candidates[0]
                logger.info({
                    "level": "INFO",
                    "type": "handler",
                    "message": f"[INFO] Auto-selected contest column (ranked): {contest_column}",
                    "session_id": session_id
                })
            else:
                logger.info({
                    "level": "INFO",
                    "type": "handler",
                    "message": "[INFO] Multiple possible contest columns detected; skipping auto-selection to avoid extra prompts.",
                    "session_id": session_id
                })

    data = []
    table_source = "text"
    table_quality_meta = metadata.setdefault("table_quality", {})
    camelot_best_score = 0.0

    if headers:
        # Locate the header line index
        header_line_idx = -1
        if header_candidate:
            try:
                header_line_idx = lines.index(header_candidate)
            except ValueError:
                header_line_idx = -1
        if header_line_idx < 0:
            # heuristic search: first line whose split matches header cells
            for idx, line in enumerate(lines):
                if _split_ws_blocks(line) == headers:
                    header_line_idx = idx
                    break
            if header_line_idx < 0:
                header_line_idx = 0

        # First pass: strict width-split like before
        for line in lines[header_line_idx + 1:]:
            if not line.strip():
                continue
            row = _split_ws_blocks(line)
            if len(row) == len(headers):
                data.append(dict(zip(headers, row)))

        # Fallback pass: whitespace-table extractor if strict pass yielded no rows
        if not data:
            data = _extract_table_by_whitespace(lines, header_line_idx, headers)
        # Try semantic candidate totals extraction if still empty
        if not data:
            _record_table_stage(metadata, "text_rows_empty", {"header_idx": header_line_idx})
            cand_headers, cand_rows = extract_candidate_totals_from_lines(lines, selected_contest_title)
            if cand_headers and cand_rows:
                export_context = _prepare_output_context(
                    None,
                    {
                        "handler": "pdf_handler",
                        "input_file": os.path.basename(pdf_path),
                        "session_id": session_id,
                        "semantic_extraction": True,
                    }
                )
                result = finalize_election_output(
                    headers=cand_headers,
                    data=cand_rows,
                    coordinator=coordinator,
                    contest=selected_contest_title,
                    state=state,
                    county=county,
                    context=export_context,
                    enable_user_feedback=False,
                    session_id=session_id
                )
                metadata.update({
                    "output_file": os.path.basename(result.get("csv_path", "")),
                    "headers": cand_headers,
                    "row_count": len(cand_rows),
                    "csv_path": result.get("csv_path"),
                    "metadata_path": result.get("metadata_path"),
                })
                metadata["table_source"] = "semantic_candidates"
                _record_table_stage(metadata, "semantic_candidates_local", {"rows": len(cand_rows)})
                logger.info({
                    "level": "INFO",
                    "type": "output",
                    "message": f"[OUTPUT] Wrote semantic candidate totals: {result.get('csv_path')}",
                    "session_id": session_id
                })
                return cand_headers, cand_rows, selected_contest_title, metadata

        text_quality = _evaluate_table_candidate_quality(headers, data, selected_contest_title)
        table_quality_meta["text"] = text_quality

        if camelot_tables:
            camelot_evals: list[tuple[dict, dict]] = []
            for entry in camelot_tables:
                quality = _evaluate_table_candidate_quality(entry.get("headers") or [], entry.get("rows") or [], selected_contest_title)
                camelot_evals.append((entry, quality))
            camelot_evals = [item for item in camelot_evals if isinstance(item[1], dict)]
            if camelot_evals:
                camelot_evals.sort(key=lambda item: item[1]["score"], reverse=True)
                top_c, top_quality = camelot_evals[0]
                if metadata.get("ocr_used"):
                    try:
                        hybrid_fill_camelot(top_c, lines)
                        metadata["camelot_hybrid_fill"] = True
                        top_quality = _evaluate_table_candidate_quality(top_c.get("headers") or [], top_c.get("rows") or [], selected_contest_title)
                        camelot_evals[0] = (top_c, top_quality)
                    except Exception:
                        pass

                camelot_best_score = float(top_quality.get("score", 0.0))
                text_score = float(text_quality.get("score", 0.0))
                top_score = camelot_best_score
                table_quality_meta["camelot_top"] = {
                    **top_quality,
                    "flavor": top_c.get("flavor"),
                    "raw_score": float(top_c.get("score", 0.0)),
                }
                table_quality_meta["camelot_candidates"] = [
                    {
                        "score": eval_info.get("score", 0.0),
                        "rows": eval_info.get("rows", 0),
                        "candidate_columns": eval_info.get("candidate_columns", 0),
                        "details": eval_info.get("details", {}),
                        "flavor": entry.get("flavor"),
                        "raw_score": float(entry.get("score", 0.0)),
                    }
                    for entry, eval_info in camelot_evals[:5]
                ]

                top_details = top_quality.get("details", {}) if isinstance(top_quality, dict) else {}
                text_details = text_quality.get("details", {}) if isinstance(text_quality, dict) else {}
                header_gain = top_details.get("richness", 0.0) - text_details.get("richness", 0.0)
                camelot_reason = None
                camelot_used_direct = False

                if not data and top_score >= max(0.25, text_score + 0.1):
                    camelot_used_direct = True
                    camelot_reason = "no_text_rows"
                elif top_score >= text_score + 0.12 and top_score >= 0.35:
                    camelot_used_direct = True
                    camelot_reason = "higher_quality_score"
                elif header_gain >= 0.2 and text_score >= top_score - 0.05:
                    text_row_count = len(data)
                    camelot_row_count = len(top_c.get("rows") or [])
                    merged = _merge_camelot_with_text(top_c, headers, data)
                    if merged:
                        headers, data = merged
                        table_source = "camelot_merged"
                        metadata["camelot_merge_applied"] = True
                        metadata["camelot_merge_reason"] = "header_richness"
                        metadata["camelot_merge_gain"] = round(header_gain, 4)
                        metadata["camelot_merge_rows"] = {
                            "text": text_row_count,
                            "camelot": camelot_row_count,
                        }
                        metadata["camelot_merge_flavor"] = top_c.get("flavor")
                        metadata["camelot_merge_raw_score"] = float(top_c.get("score", 0.0))
                        metadata["camelot_used"] = False
                    else:
                        camelot_reason = "merge_failed"

                if camelot_used_direct:
                    headers = list(top_c.get("headers") or [])
                    data = list(top_c.get("rows") or [])
                    table_source = "camelot"
                    metadata["camelot_used"] = True
                    metadata["camelot_flavor"] = top_c.get("flavor")
                    metadata["camelot_score"] = float(top_c.get("score", 0.0))
                    metadata["camelot_rows"] = len(top_c.get("rows") or [])
                    if camelot_reason:
                        metadata["camelot_reason"] = camelot_reason
                else:
                    metadata["camelot_available"] = True
                    metadata["camelot_top_score"] = float(top_c.get("score", 0.0))
                    metadata["camelot_alt_count"] = len(camelot_tables)
                    metadata["camelot_tables_summary"] = [
                        {
                            "quality_score": eval_info.get("score", 0.0),
                            "raw_score": float(entry.get("score", 0.0)),
                            "rows": len(entry.get("rows") or []),
                            "cols": len(entry.get("headers") or []),
                            "flavor": entry.get("flavor"),
                        }
                        for entry, eval_info in camelot_evals[:5]
                    ]
                    if camelot_reason:
                        metadata["camelot_reason"] = camelot_reason
            else:
                metadata["camelot_available"] = True
                metadata["camelot_alt_count"] = len(camelot_tables)
                metadata["camelot_reason"] = "no_viable_candidates"

        active_quality = _evaluate_table_candidate_quality(headers, data, selected_contest_title)
        table_quality_meta["active"] = {**active_quality, "source": table_source}

        # If we got some rows, but the table looks low-quality, switch to semantic extraction
        if data and _table_looks_bad(headers, data):
            _record_table_stage(metadata, "text_rows_low_quality", {"rows": len(data), "headers": len(headers)})
            logger.info({
                "level": "INFO",
                "type": "handler",
                "message": "[INFO] Detected low-quality header/rows; attempting semantic candidate extraction instead.",
                "session_id": session_id
            })
            cand_headers, cand_rows = extract_candidate_totals_from_lines(lines, selected_contest_title)
            if cand_headers and cand_rows and len(cand_rows) >= len(data):
                export_context = _prepare_output_context(
                    None,
                    {
                        "handler": "pdf_handler",
                        "input_file": os.path.basename(pdf_path),
                        "session_id": session_id,
                        "semantic_extraction": True,
                        "replaced_noisy_table": True,
                    }
                )
                result = finalize_election_output(
                    headers=cand_headers,
                    data=cand_rows,
                    coordinator=coordinator,
                    contest=selected_contest_title,
                    state=state,
                    county=county,
                    context=export_context,
                    enable_user_feedback=False,
                    session_id=session_id
                )
                metadata.update({
                    "output_file": os.path.basename(result.get("csv_path", "")),
                    "headers": cand_headers,
                    "row_count": len(cand_rows),
                    "csv_path": result.get("csv_path"),
                    "metadata_path": result.get("metadata_path"),
                })
                metadata["table_source"] = "semantic_candidates"
                _record_table_stage(metadata, "semantic_candidates_local", {"rows": len(cand_rows)})
                logger.info({
                    "level": "INFO",
                    "type": "output",
                    "message": f"[OUTPUT] Replaced noisy table with semantic candidate totals: {result.get('csv_path')}",
                    "session_id": session_id
                })
                return cand_headers, cand_rows, selected_contest_title, metadata
            else:
                logger.info({
                    "level": "INFO",
                    "type": "handler",
                    "message": "[INFO] Semantic extraction did not improve over noisy table; keeping extracted rows.",
                    "session_id": session_id
                })

        if layout_tables:
            best_layout = layout_tables[0]
            layout_headers = best_layout.get("headers") or []
            layout_rows = best_layout.get("rows") or []
            layout_quality = _evaluate_table_candidate_quality(layout_headers, layout_rows, selected_contest_title)
            table_quality_meta["layout_top"] = {
                **layout_quality,
                "page": best_layout.get("page"),
            }
            best_competing = max(active_quality["score"], camelot_best_score, text_quality["score"])
            use_layout = False
            if layout_headers and layout_rows:
                if not data and layout_quality["score"] >= 0.3:
                    use_layout = True
                elif layout_quality["score"] >= best_competing + 0.12:
                    use_layout = True
                elif layout_quality["score"] >= 0.4 and best_competing < 0.3:
                    use_layout = True
                elif layout_quality["score"] >= 0.35 and active_quality["score"] < 0.25 and camelot_best_score < 0.35:
                    use_layout = True
            if use_layout:
                headers = layout_headers
                data = layout_rows
                contest_column = None
                table_source = "layout"
                metadata["layout_table_used"] = True
                metadata["layout_table_rows"] = len(layout_rows)
                metadata["layout_table_page"] = best_layout.get("page")
            elif layout_rows:
                metadata["layout_table_candidate_rows"] = len(layout_rows)

        active_quality = _evaluate_table_candidate_quality(headers, data, selected_contest_title)
        table_quality_meta["active"] = {**active_quality, "source": table_source}

        if statement_rows_copy:
            use_statement = False
            pre_statement_rows = len(data) if isinstance(data, list) else 0
            if not data:
                use_statement = True
            elif len(statement_rows_copy) >= max(len(data), 5):
                use_statement = True
            elif len(data) <= 3 and len(statement_rows_copy) >= len(data) * 2:
                use_statement = True
            if use_statement:
                headers = list(statement_headers_copy)
                data = [dict(row) for row in statement_rows_copy]
                contest_column = None
                table_source = "statement"
                metadata["statement_blocks_used"] = True
                metadata["statement_blocks_rows"] = len(statement_rows_copy)
                _record_table_stage(metadata, "statement_promoted", {"pre_rows": pre_statement_rows})
            metadata["statement_blocks_decision"] = {
                "pre_rows": pre_statement_rows,
                "use_statement": use_statement,
                "statement_rows": len(statement_rows_copy),
            }
            logger.debug({
                "level": "DEBUG",
                "type": "handler",
                "message": "[DEBUG] Statement-return heuristic",
                "session_id": session_id,
                "pre_rows": pre_statement_rows,
                "statement_rows": len(statement_rows_copy),
                "use_statement": use_statement,
            })

        active_quality = _evaluate_table_candidate_quality(headers, data, selected_contest_title)
        table_quality_meta["active"] = {**active_quality, "source": table_source}

        # If we have a contest column, filter to the selected contest
        contest = selected_contest_title
        if contest_column:
            def _norm_title(s: str) -> str:
                s = (s or "").lower().strip()
                s = re.sub(r'[\s\-_/]+', ' ', s)
                s = re.sub(r'[^a-z0-9 ]+', '', s)
                return re.sub(r'\s+', ' ', s).strip()

            def _tokens(s: str) -> set[str]:
                return set(re.findall(r'[a-z0-9]+', (s or "").lower()))

            norm_selected = _norm_title(contest)
            present_values = sorted({(r.get(contest_column, "") or "").strip() for r in data if r.get(contest_column)})
            norm_map = {v: _norm_title(v) for v in present_values}

            # 1) exact normalized match
            exact = [v for v, nv in norm_map.items() if nv == norm_selected]
            chosen_value = exact[0] if exact else None

            # 2) token-overlap fallback if no exact
            if not chosen_value:
                sel_tok = _tokens(contest)
                scored = []
                for v in present_values:
                    vt = _tokens(v)
                    inter = len(sel_tok & vt)
                    union = len(sel_tok | vt) or 1
                    jacc = inter / union
                    # small boost for prefix/substring matches
                    if norm_selected and norm_map[v].startswith(norm_selected):
                        jacc += 0.15
                    if norm_selected and norm_selected in norm_map[v]:
                        jacc += 0.10
                    scored.append((jacc, v))
                scored.sort(reverse=True)
                if scored and scored[0][0] >= 0.45:
                    chosen_value = scored[0][1]

            if chosen_value:
                data = [r for r in data if _norm_title(r.get(contest_column, "")) == _norm_title(chosen_value)]
                logger.info({
                    "level": "INFO",
                    "type": "handler",
                    "message": f"[INFO] Filtered rows to contest '{chosen_value}' via column '{contest_column}'.",
                    "session_id": session_id
                })
            else:
                logger.warning({
                    "level": "WARNING",
                    "type": "handler",
                    "message": f"[WARN] Selected contest '{contest}' not found in column '{contest_column}'. Skipping row filter.",
                    "session_id": session_id,
                    "present": present_values[:25]
                })

        if data:
            # Deduplicate exact duplicate rows
            data = [dict(t) for t in dict.fromkeys(tuple(sorted(row.items())) for row in data)]
            metadata["pre_finalize_row_count"] = len(data)
            metadata["table_source"] = table_source
            headers_final, data_final, _ = _finalize_structured_table_output(
                pdf_path,
                headers,
                data,
                selected_contest_title,
                state,
                county,
                year,
                contest_slug,
                metadata,
                coordinator,
                session_id=session_id,
            )
            if statement_rows_copy and statement_headers_copy:
                promote_statement = False
                if metadata.get("statement_blocks_used"):
                    promote_statement = True
                elif len(data_final) <= 3 and len(statement_rows_copy) >= max(len(data_final) * 2, 5):
                    promote_statement = True
                if promote_statement:
                    metadata["statement_blocks_used"] = True
                    metadata["statement_blocks_rows"] = len(statement_rows_copy)
                    metadata["statement_blocks_promoted"] = {
                        "rows_before": len(data_final),
                        "rows_statement": len(statement_rows_copy),
                    }
                    metadata["table_source"] = "statement"
                    headers_final, data_final, _ = _finalize_structured_table_output(
                        pdf_path,
                        list(statement_headers_copy),
                        [dict(row) for row in statement_rows_copy],
                        selected_contest_title,
                        state,
                        county,
                        year,
                        contest_slug,
                        metadata,
                        coordinator,
                        session_id=session_id,
                    )
            return _finalize_with_quality(headers_final, data_final, selected_contest_title, metadata)

        else:
            tried_associated = True
            # Try associated tables as fallback
            contest_positions = _detect_contest_positions(page_text_map)
            associated_tables = _associate_tables_with_contests(contest_positions, table_candidates, page_text_map)
            if associated_tables:
                best_associated = max(associated_tables, key=lambda x: x['score'])
                if best_associated['score'] >= 0.3:
                    headers, data = best_associated['headers'], best_associated['data']
                    metadata['associated_table'] = True
                    metadata['associated_contest'] = best_associated['contest_title']
                    return headers, data, all_text, metadata
            # Attempt columnar reconstruction before falling back to raw text export
            recon_result = _try_columnar_reconstruction(
                pdf_path,
                lines,
                line_records,
                selected_contest_title,
                state,
                county,
                metadata,
                coordinator,
                session_id,
            )
            if recon_result:
                return recon_result

            if statement_rows_copy and statement_headers_copy:
                metadata["statement_blocks_used"] = True
                metadata["statement_blocks_rows"] = len(statement_rows_copy)
                metadata["table_source"] = "statement"
                headers_final, data_final, _ = _finalize_structured_table_output(
                    pdf_path,
                    list(statement_headers_copy),
                    [dict(row) for row in statement_rows_copy],
                    selected_contest_title,
                    state,
                    county,
                    year,
                    contest_slug,
                    metadata,
                    coordinator,
                    session_id=session_id,
                )
                return _finalize_with_quality(headers_final, data_final, selected_contest_title, metadata)

            unmatched_count = len(lines[header_line_idx + 1:])
            logger.warning({
                "level": "WARNING",
                "type": "output",
                "message": f"[WARN] No structured rows matched the inferred column count of {len(headers)}. Total lines scanned: {unmatched_count}",
                "session_id": session_id
            })
            fallback_rows = [{"raw_line": line} for line in lines[header_line_idx + 1:]]
            _record_table_stage(metadata, "raw_line_fallback", {"unmatched_lines": unmatched_count})
            fallback_context = {
                "contest": selected_contest_title,
                "state": state,
                "county": county,
                "session_id": session_id,
                "handler": "pdf_handler",
            }
            export_context = _prepare_output_context(
                fallback_context,
                {
                    "handler": "pdf_handler",
                    "input_file": os.path.basename(pdf_path),
                    "session_id": session_id,
                    "fallback": True,
                }
            )
            metadata["table_source"] = "raw_lines"
            result = finalize_election_output(
                headers=["raw_line"],
                data=fallback_rows,
                coordinator=coordinator,
                contest=selected_contest_title,
                state=state,
                county=county,
                context=export_context,
                enable_user_feedback=False,
                session_id=session_id
            )
            metadata.update({
                "output_file": os.path.basename(result.get("csv_path", "")),
                "headers": ["raw_line"],
                "row_count": len(fallback_rows),
                "csv_path": result.get("csv_path"),
                "metadata_path": result.get("metadata_path"),
            })
            logger.warning({
                "level": "WARNING",
                "type": "output",
                "message": f"[OUTPUT] Wrote fallback rows to: {result.get('csv_path')}",
                "session_id": session_id
            })
            return ["raw_line"], fallback_rows, selected_contest_title, metadata

    if layout_tables and not metadata.get("layout_table_used"):
        best_layout = layout_tables[0]
        layout_headers = best_layout.get("headers") or []
        layout_rows = best_layout.get("rows") or []
        if layout_headers and layout_rows:
            prefer_statement = False
            if statement_rows_copy and statement_headers_copy:
                if len(layout_rows) <= 3 and len(statement_rows_copy) >= max(len(layout_rows) * 2, 5):
                    prefer_statement = True
                elif len(statement_rows_copy) >= max(len(layout_rows) + 5, int(len(layout_rows) * 1.5)):
                    prefer_statement = True
            metadata["layout_table_used"] = True
            metadata["layout_table_rows"] = len(layout_rows)
            metadata["layout_table_page"] = best_layout.get("page")
            metadata["table_source"] = "layout"
            headers_final, data_final, _ = _finalize_structured_table_output(
                pdf_path,
                layout_headers,
                layout_rows,
                selected_contest_title,
                state,
                county,
                year,
                contest_slug,
                metadata,
                coordinator,
                session_id=session_id,
            )
            if (
                statement_rows_copy
                and statement_headers_copy
                and (
                    prefer_statement
                    or (
                        len(data_final) <= 3
                        and len(statement_rows_copy) >= max(len(data_final) * 2, 5)
                    )
                )
            ):
                metadata["layout_table_available_rows"] = len(layout_rows)
                metadata["layout_table_available_page"] = best_layout.get("page")
                metadata["layout_table_used"] = False
                metadata["statement_blocks_used"] = True
                metadata["statement_blocks_rows"] = len(statement_rows_copy)
                metadata["table_source"] = "statement"
                _record_table_stage(metadata, "statement_promoted", {"layout_rows": len(layout_rows)})
                headers_final, data_final, _ = _finalize_structured_table_output(
                    pdf_path,
                    list(statement_headers_copy),
                    [dict(row) for row in statement_rows_copy],
                    selected_contest_title,
                    state,
                    county,
                    year,
                    contest_slug,
                    metadata,
                    coordinator,
                    session_id=session_id,
                )
            return _finalize_with_quality(headers_final, data_final, selected_contest_title, metadata)

    if statement_rows and statement_headers:
        # Ensure contest metadata reflects statement extraction path
        metadata["statement_blocks_available"] = {
            "rows": len(statement_rows),
            "headers": statement_headers[:10]
        }
        metadata["statement_blocks_used"] = True
        metadata["statement_blocks_rows"] = len(statement_rows)
        metadata["table_source"] = "statement"
        _record_table_stage(metadata, "statement_terminal", {"rows": len(statement_rows)})
        headers_final, data_final, _ = _finalize_structured_table_output(
            pdf_path,
            list(statement_headers),
            [dict(row) for row in statement_rows],
            selected_contest_title,
            state,
            county,
            year,
            contest_slug,
            metadata,
            coordinator,
            session_id=session_id,
        )
        return _finalize_with_quality(headers_final, data_final, selected_contest_title, metadata)

    if not tried_associated:
        # Try associated tables as fallback
        contest_positions = _detect_contest_positions(page_text_map)
        associated_tables = _associate_tables_with_contests(contest_positions, table_candidates, page_text_map)
        if associated_tables:
            best_associated = max(associated_tables, key=lambda x: x['score'])
            if best_associated['score'] >= 0.3:
                headers, data = best_associated['headers'], best_associated['data']
                metadata['associated_table'] = True
                metadata['associated_contest'] = best_associated['contest_title']
                return headers, data, all_text, metadata

    recon_result = _try_columnar_reconstruction(
        pdf_path,
        lines,
        line_records,
        selected_contest_title,
        state,
        county,
        metadata,
        coordinator,
        session_id,
    )
    if recon_result:
        return recon_result

    # No headers at all: still try semantic candidate totals from entire text
    cand_headers, cand_rows = extract_candidate_totals_from_lines(lines, selected_contest_title)
    if cand_headers and cand_rows:
        _record_table_stage(metadata, "semantic_totals", {"rows": len(cand_rows)})
        export_context = _prepare_output_context(
            None,
            {
                "handler": "pdf_handler",
                "input_file": os.path.basename(pdf_path),
                "session_id": session_id,
                "semantic_extraction": True,
            }
        )
        result = finalize_election_output(
            headers=cand_headers,
            data=cand_rows,
            coordinator=coordinator,
            contest=selected_contest_title,
            state=state,
            county=county,
            context=export_context,
            enable_user_feedback=False,
            session_id=session_id
        )
        metadata.update({
            "output_file": os.path.basename(result.get("csv_path", "")),
            "headers": cand_headers,
            "row_count": len(cand_rows),
            "csv_path": result.get("csv_path"),
            "metadata_path": result.get("metadata_path"),
        })
        logger.info({
            "level": "INFO",
            "type": "output",
            "message": f"[OUTPUT] Wrote semantic candidate totals: {result.get('csv_path')}",
            "session_id": session_id
        })
        return cand_headers, cand_rows, selected_contest_title, metadata

    # Plain text fallback
    export_context = _prepare_output_context(
        None,
        {
            "handler": "pdf_handler",
            "input_file": os.path.basename(pdf_path),
            "session_id": session_id,
            "text_sanitized": True,
            "raw_text_len": len(all_text or ""),
            "clean_text_len": len(clean_text or ""),
        }
    )
    result = finalize_election_output(
        headers=["text"],
        data=[{"text": clean_text}],
        coordinator=coordinator,
        contest=selected_contest_title,
        state=state,
        county=county,
        context=export_context,
        enable_user_feedback=False,
        session_id=session_id
    )
    metadata.update({
        "output_file": os.path.basename(result.get("csv_path", "")),
        "headers": ["text"],
        "row_count": 1,
        "text_sanitized": True,
        "raw_text_len": len(all_text or ""),
        "clean_text_len": len(clean_text or ""),
        "csv_path": result.get("csv_path"),
        "metadata_path": result.get("metadata_path")
    })
    _record_table_stage(metadata, "text_fallback", {"raw_len": len(all_text or "")})
    logger.warning({
        "level": "WARNING",
        "type": "output",
        "message": f"[OUTPUT] Wrote plain text to: {result.get('csv_path')}",
        "session_id": session_id
    })
    return _finalize_with_quality(["text"], [{"text": clean_text}], selected_contest_title, metadata)

def parse(page=None, coordinator=None, html_context=None, manual_file=None, session_id=None, **kwargs):
    """
    Universal pipeline entry: Accepts a PDF file path (manual_file) from the format router.
    Returns: headers, data, contest, metadata
    """
    html_context = html_context or {}
    # Parity guard: allow provided_tables + skip_pivot to bypass file requirement
    provided_tables = html_context.get("provided_tables")
    if isinstance(provided_tables, list) and provided_tables:
        ctx = dict(html_context)
        ctx.update({
            "session_id": session_id,
            "coordinator": coordinator,
        })
        merged_headers, merged_rows = robust_table_extraction(page=None, extraction_context=ctx)

        contest = html_context.get("contest") or "Provided Tables"
        state = html_context.get("state") or "Unknown"
        county = html_context.get("county") or "Unknown"
        year = html_context.get("year")
        domain = html_context.get("source_slug") or safe_slug(contest)

        headers_final, data_final, _entity_info = build_table_noninteractive(
            domain=domain,
            headers=merged_headers,
            data=merged_rows,
            coordinator=coordinator,
            context={
                **ctx,
                "contest": contest,
                "state": state,
                "county": county,
                "year": year,
                "handler": "pdf_handler",
            },
            pivot_to_wide=not bool(html_context.get("skip_pivot")),
            debug=False,
        )

        export_context = _prepare_output_context(
            ctx,
            {
                "handler": "pdf_handler",
                "session_id": session_id,
                "race": contest,
                "provided_tables": True,
                "skip_pivot": bool(html_context.get("skip_pivot")),
                "optional_deps": {
                    "camelot_available": bool(_CAMELOT_AVAILABLE),
                    "pytesseract_available": bool(pytesseract),
                    "pdf2image_available": bool(pdf2image),
                },
            }
        )

        result = finalize_election_output(
            headers=headers_final,
            data=data_final,
            coordinator=coordinator,
            contest=contest,
            state=state,
            county=county,
            context=export_context,
            enable_user_feedback=False,
            session_id=session_id,
        )

        metadata = {
            "race": contest,
            "input_file": html_context.get("input_file") or "<provided>",
            "output_file": os.path.basename(result.get("csv_path", "")),
            "headers": headers_final,
            "row_count": len(data_final),
            "handler": "pdf_handler",
            "state": state,
            "county": county,
            "year": year,
            "csv_path": result.get("csv_path"),
            "metadata_path": result.get("metadata_path"),
            "optional_deps": {
                "camelot_available": bool(_CAMELOT_AVAILABLE),
                "pytesseract_available": bool(pytesseract),
                "pdf2image_available": bool(pdf2image),
            },
        }
        # Add quality metrics for provided_tables path
        from ...config import log_extraction_quality  # type: ignore[attr-defined]
        quality = log_extraction_quality(
            headers_final, data_final, metadata, "pdf_handler", logger, session_id
        )
        metadata["quality_metrics"] = quality
        return headers_final, data_final, contest, metadata
    cancel_flag = kwargs.get("cancel_flag")

    if html_context.get("skip_format") or html_context.get("manual_skip"):
        logger.info({
            "level": "INFO",
            "type": "handler",
            "message": "[SKIP] PDF parsing intentionally skipped via context flag.",
            "session_id": session_id
        })
        return None, None, None, {"skipped": True}

    if not manual_file or not os.path.isfile(manual_file):
        logger.error({
            "level": "ERROR",
            "type": "handler",
            "message": "[ERROR] No PDF file provided to parse().",
            "session_id": session_id
        })
        return None, None, None, {"skipped": True}

    try:
        result = parse_pdf_election_results(
            manual_file,
            session_id=session_id,
            coordinator=coordinator,
            cancel_flag=cancel_flag,
        )
    except PDFParseCancelled as exc:
        meta_seed = {
            "input_file": os.path.basename(manual_file) if manual_file else None,
        }
        return _cancelled_result(manual_file, meta_seed, str(exc), session_id=session_id)

    # Defensive: always return a 4-tuple, never a bool
    if not (isinstance(result, tuple) and len(result) == 4):
        logger.error({
            "level": "ERROR",
            "type": "handler",
            "message": "[ERROR] Invalid result from parse_pdf_election_results (expected 4-tuple).",
            "session_id": session_id,
            "got_type": type(result).__name__
        })
        return None, None, None, {"error": "Invalid parse result"}
    return result
