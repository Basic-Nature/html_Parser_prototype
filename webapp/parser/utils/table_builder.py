from __future__ import annotations
# ===================================================================
# table_builder.py
# Election Data Cleaner - Table Extraction and Cleaning Orchestrator
# Centralizes user feedback, ML learning, and structure confirmation.
# ===================================================================
import copy
import os
import orjson
import time
from typing import List, Dict, Tuple, Any, TYPE_CHECKING, Optional
from rich.table import Table

from ..Context_Integration.Context_Library.constants import (
    PERCENT_KEYWORDS,
)
from .shared_logic import (
    safe_get, safe_append, safe_isalnum, safe_copy, safe_strip, safe_replace, safe_lower,
    safe_values
)
from .logger_singleton import logger
from ..config import CACHE_DIR
from .table_core import (
    harmonize_headers_and_data,
    nlp_entity_annotate_table,
    pivot_to_wide_format,
    table_signature,
    cache_table_structure,
    normalize_header, merge_table_data
)
from .dynamic_table_extractor import dynamic_table_extractor

if TYPE_CHECKING:
    from ..Context_Integration.context_coordinator import ContextCoordinator

# ===================================================================
# Helper: Structured logging wrapper
# ===================================================================

def _emit(level: str, msg_type: str, message: str, session_id: Optional[str] = None, **fields):
    """
    Emit a structured log payload using SharedLogger with consistent keys.
    """
    payload = {
        "level": level.upper(),
        "type": msg_type,
        "message": message,
        "session_id": session_id,
    }
    # Attach extra fields only if not None
    for k, v in fields.items():
        if v is not None:
            payload[k] = v
    # Route to appropriate logger method
    level_l = level.lower()
    log_fn = getattr(logger, level_l, logger.info)
    log_fn(payload)

# ===================================================================
# MAIN TABLE BUILDING PIPELINE
# ===================================================================

def build_dynamic_table(
    domain: str,
    headers: List[str],
    data: List[Dict[str, Any]],
    coordinator: "ContextCoordinator",
    context: dict = None,
    max_feedback_loops: int = 2,
    learning_mode: bool = True,
    confirm_table_structure_callback=None,
    pivot_to_wide: bool = True,
    debug: bool = False,
) -> Tuple[List[str], List[Dict[str, Any]], dict]:
    """
    Orchestrates robust, multi-source, entity-aware table extraction and harmonization.
    Always merges, harmonizes, and pivots all panel tables before any feedback/confirmation.
    Returns (headers, data, entity_info) for downstream enrichment.
    """
    from ..Context_Integration.context_coordinator import ContextCoordinator

    # Normalize inputs / coordinator / context
    coordinator = coordinator or ContextCoordinator()
    context = context or {}
    data = data or []
    headers = headers or []
    if "coordinator" not in context or context["coordinator"] is None:
        context["coordinator"] = coordinator
    # session_id pass-through for logs and prompts
    session_id = safe_get(context, "session_id", None)
    if safe_get(context, "panel_heading") and not safe_get(context, "Precinct"):
        context["Precinct"] = context["panel_heading"]
    page = safe_get(context, "page", [])

    _emit("info", "builder", "[TABLE_BUILDER] Starting dynamic build", session_id, domain=domain, pivot_to_wide=pivot_to_wide, learning_mode=learning_mode)

    # --- 1. Gather all panel tables if present ---
    all_panel_tables = []
    if safe_get(context, "panels"):
        for panel in context["panels"]:
            for table in safe_get(panel, "tables", []):
                table_context = context.copy()
                table_context["panel_heading"] = safe_get(panel, "panel_heading", [])
                table_context["Precinct"] = safe_get(panel, "Precinct", [])
                table_context["table_html"] = safe_get(table, "table_html", [])
                try:
                    h, d = dynamic_table_extractor(page, table_context, coordinator, table_html=safe_get(table, "table_html"))
                except Exception as e:
                    _emit("warning", "builder", "[TABLE_BUILDER] dynamic_table_extractor failed for panel table", session_id, error=str(e))
                    h, d = [], []
                if h and d:
                    # NOTE: avoid assigning result of safe_append; ensure list type
                    all_panel_tables.append((h, d))
        _emit("debug", "builder", "[TABLE_BUILDER] Collected panel tables", session_id, count=len(all_panel_tables))
    elif headers and data:
        # Ensure explicit append to keep list type
        all_panel_tables.append((headers, data))
        _emit("debug", "builder", "[TABLE_BUILDER] Using provided headers/data as sole table", session_id, headers=len(headers), rows=len(data))
    else:
        try:
            h, d = dynamic_table_extractor(page, context, coordinator)
        except Exception as e:
            _emit("warning", "builder", "[TABLE_BUILDER] dynamic_table_extractor failed (no panels path)", session_id, error=str(e))
            h, d = [], []
        if h and d:
            all_panel_tables.append((h, d))
        _emit("debug", "builder", "[TABLE_BUILDER] Fallback extractor path", session_id, found=bool(h and d))

    # Defensive: coerce/validate structure before iterating
    if not isinstance(all_panel_tables, list):
        _emit("warning", "builder", "[TABLE_BUILDER] all_panel_tables was not a list; coercing to empty list", session_id, got_type=str(type(all_panel_tables)))
        all_panel_tables = []
    else:
        # Keep only (headers, rows) tuple pairs
        fixed = []
        for item in all_panel_tables:
            if isinstance(item, tuple) and len(item) == 2:
                fixed.append(item)
            else:
                _emit("warning", "builder", "[TABLE_BUILDER] Dropping invalid table entry", session_id, entry_type=str(type(item)))
        all_panel_tables = fixed
        
    # --- 2. Merge and harmonize all tables (advanced logic) ---
    if all_panel_tables:
        all_headers = []
        all_data = []
        for h, d in all_panel_tables:
            # If h or d are lists of lists, flatten them
            if isinstance(h, list) and any(isinstance(x, list) for x in h):
                for sub_h in h:
                    all_headers = safe_append(all_headers, sub_h)
            else:
                all_headers = safe_append(all_headers, h)
            if isinstance(d, list) and any(isinstance(x, dict) for x in d):
                all_data.extend(d)
            else:
                all_data = safe_append(all_data, d)

        # Merge headers and data with advanced deduplication and alignment
        try:
            merged_headers, merged_data = merge_table_data(all_headers, all_data)
        except Exception as e:
            _emit("warning", "builder", "[TABLE_BUILDER] merge_table_data failed; falling back to raw", session_id, error=str(e))
            merged_headers, merged_data = all_headers, all_data
    else:
        merged_headers, merged_data = [], []

    try:
        merged_headers, merged_data = harmonize_headers_and_data(merged_headers, merged_data, context=context)
    except Exception as e:
        _emit("warning", "builder", "[TABLE_BUILDER] harmonize_headers_and_data (initial) failed", session_id, error=str(e))
        # Keep original if harmonization fails
        pass

    # --- Ensure all required percent columns are present ---
    norm_headers = set(normalize_header(h) for h in merged_headers)
    for percent_col in PERCENT_KEYWORDS:
        norm_percent_col = normalize_header(percent_col)
        if norm_percent_col not in norm_headers:
            merged_headers = safe_append(merged_headers, percent_col)
            for row in merged_data:
                row[percent_col] = ""
    # Mark presence for downstream logic
    context["has_percent_reported"] = any(
        any(normalize_header(h) == normalize_header(pc) for h in merged_headers)
        for pc in PERCENT_KEYWORDS
    )
    # --- 3. NLP entity annotation ---
    try:
        annotated_headers, annotated_data, entity_info = nlp_entity_annotate_table(
            merged_headers, merged_data, context=context, coordinator=coordinator
        )
    except Exception as e:
        _emit("warning", "builder", "[TABLE_BUILDER] NLP entity annotation failed", session_id, error=str(e), contest=safe_get(context, "contest", "Unknown"))
        annotated_headers, annotated_data = merged_headers, merged_data
        entity_info = {}

    try:
        headers, data = harmonize_headers_and_data(annotated_headers, annotated_data, context=context)
    except Exception as e:
        _emit("warning", "builder", "[TABLE_BUILDER] harmonize_headers_and_data (post-NLP) failed; using annotated directly", session_id, error=str(e))
        headers, data = annotated_headers, annotated_data

    # --- 4. Pivot to wide format before feedback ---
    if pivot_to_wide:
        try:
            wide_headers, wide_data = pivot_to_wide_format(headers, data, entity_info, coordinator, context)
            headers, data = harmonize_headers_and_data(wide_headers, wide_data, context=context)
        except Exception as e:
            _emit("warning", "builder", "[TABLE_BUILDER] Pivot to wide format failed", session_id, error=str(e), contest=safe_get(context, "contest", "Unknown"))

    # --- 5. Optionally load from cache for debugging ---
    if debug:
        cached = _load_table_builder_cache(domain, latest=True)
        if cached:
            _emit("info", "builder", "[TABLE_BUILDER] Loaded cached table", session_id, domain=domain)
            headers, data = safe_get(cached, "headers", headers), safe_get(cached, "data", data)
            entity_info = safe_get(cached, "entity_info", entity_info)

    # --- 6. User/ML confirmation and learning (if enabled) ---
    if learning_mode:
        contest = safe_get(context, "contest", []) or "Unknown Contest"
        feedback_loops = 0
        while feedback_loops < max_feedback_loops:
            try:
                if confirm_table_structure_callback:
                    headers_confirmed, data_confirmed = confirm_table_structure_callback(
                        headers, data, domain, contest, coordinator, session_id=session_id
                    )
                else:
                    headers_confirmed, data_confirmed = prompt_user_to_confirm_table_structure(
                        headers, data, domain, contest, coordinator, session_id=session_id
                    )
            except Exception as e:
                _emit("warning", "builder", "[TABLE_BUILDER] Confirmation callback failed; skipping interactive step", session_id, error=str(e))
                break

            if debug:
                _save_table_builder_cache(domain, {
                    "headers": headers_confirmed,
                    "data": data_confirmed,
                    "entity_info": entity_info,
                    "timestamp": time.time()
                })

            # Accept if user made changes or confirmed, else loop for feedback
            if headers_confirmed != headers or data_confirmed != data or not learning_mode:
                headers, data = headers_confirmed, data_confirmed
                break
            feedback_loops += 1

        # Ensure all user-added columns are present in all rows
        for h in headers:
            for row in data:
                if h not in row:
                    row[h] = ""

    _emit("info", "builder", "[TABLE_BUILDER] Completed dynamic build", session_id, headers=len(headers), rows=len(data))
    return headers, data, entity_info

# Non-interactive wrapper for format handlers

def build_table_noninteractive(
    domain: str,
    headers: List[str] | None,
    data: List[Dict[str, Any]] | None,
    coordinator: "ContextCoordinator" = None,
    context: dict | None = None,
    pivot_to_wide: bool = True,
    debug: bool = False
) -> Tuple[List[str], List[Dict[str, Any]], dict]:
    """
    Convenience wrapper around build_dynamic_table with learning_mode disabled.
    Use this in format handlers (CSV/JSON/PDF) after contest selection to
    harmonize, annotate, and optionally pivot tables without any prompts.
    """
    context = context or {}
    session_id = safe_get(context, "session_id", None)
    _emit("info", "builder", "[TABLE_BUILDER] build_table_noninteractive called", session_id, domain=domain)
    return build_dynamic_table(
        domain=domain,
        headers=headers or [],
        data=data or [],
        coordinator=coordinator,
        context=context,
        max_feedback_loops=0,
        learning_mode=False,
        confirm_table_structure_callback=None,
        pivot_to_wide=pivot_to_wide,
        debug=debug
    )

# ===================================================================
# CACHE MANAGEMENT STRATEGY
# ===================================================================

def _get_table_builder_cache_dir():
    """
    Returns the directory for table builder cache files, nested under CACHE_DIR for consistency.
    """
    cache_dir = os.path.join(CACHE_DIR, "table_builder_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

def _save_table_builder_cache(domain, persistent_cache, keep_last_n=5):
    """
    Save the persistent cache for debugging/recovery only.
    Keeps only the last N cache files per domain to avoid stale data buildup.
    """
    cache_dir = _get_table_builder_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    safe_domain = "".join(c for c in domain if safe_isalnum(c) or c in ("-", "_"))
    timestamp = int(safe_get(persistent_cache, "timestamp", time.time()))
    cache_path = os.path.join(cache_dir, f"{safe_domain}_{timestamp}_table.json")
    with open(cache_path, "wb") as f:
        f.write(orjson.dumps(persistent_cache, option=orjson.OPT_INDENT_2))
    # Cleanup
    files = sorted(
        [f for f in os.listdir(cache_dir) if f.startswith(safe_domain)],
        key=lambda fn: os.path.getmtime(os.path.join(cache_dir, fn)),
        reverse=True
    )
    for old_file in files[keep_last_n:]:
        try:
            os.remove(os.path.join(cache_dir, old_file))
        except Exception:
            pass

def _list_table_builder_cache(domain=None):
    """
    List available cache files for a domain (or all if domain is None).
    """
    cache_dir = _get_table_builder_cache_dir()
    if not os.path.exists(cache_dir):
        return []
    files = os.listdir(cache_dir)
    if domain:
        safe_domain = "".join(c for c in domain if safe_isalnum(c) or c in ("-", "_"))
        files = [f for f in files if f.startswith(safe_domain)]
    return sorted(files, key=lambda fn: os.path.getmtime(os.path.join(cache_dir, fn)), reverse=True)

def _load_table_builder_cache(domain, latest=True):
    """
    Load the latest (or all) cache files for a domain.
    """
    files = _list_table_builder_cache(domain)
    if not files:
        return None
    cache_dir = _get_table_builder_cache_dir()
    if latest:
        with open(os.path.join(cache_dir, files[0]), "rb") as f:
            return orjson.loads(f.read())
    else:
        caches = []
        for fn in files:
            with open(os.path.join(cache_dir, fn), "rb") as f:
                caches.append(orjson.loads(f.read()))
        return caches

# ===================================================================
# USER FEEDBACK, CONFIRMATION, AND LEARNING
# ===================================================================

def prompt_user_to_confirm_table_structure(
    headers,
    data,
    domain,
    contest,
    coordinator,
    session_id: Optional[str] = None
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Interactive CLI for user to confirm, correct, or reject table structure.
    Ensures 'Percent Reported' is always included if present in data or context.
    """
    from ..Context_Integration.context_coordinator import ContextCoordinator
    coordinator = coordinator or ContextCoordinator()
    should_log = True
    columns_changed = False
    new_headers = safe_copy(headers)

    # Always include 'Percent Reported' if present in any row
    if any("Percent Reported" in row for row in data) and "Percent Reported" not in new_headers:
        new_headers = safe_append(new_headers, "Percent Reported")

    os.makedirs(CACHE_DIR, exist_ok=True)

    # Denied structures cache
    denied_structures_path = os.path.join(CACHE_DIR, "denied_table_structures.json")
    denied_structures = {}
    if os.path.exists(denied_structures_path):
        with open(denied_structures_path, "rb") as f:
            denied_structures = orjson.loads(f.read())
    sig = f"{domain}:{table_signature(headers)}"
    denied_count = safe_get(denied_structures, sig, 0)

    # Removed columns cache
    removed_columns_log_path = os.path.join(CACHE_DIR, "removed_columns_cache.json")
    removed_columns_log = {}
    if os.path.exists(removed_columns_log_path):
        with open(removed_columns_log_path, "rb") as f:
            removed_columns_log = orjson.loads(f.read())

    # ML/NLP suggestions
    ml_scores = []
    nlp_suggestions = []
    for h in new_headers:
        try:
            score = coordinator.score_header(h, {"contest": contest})
        except Exception:
            score = 0.0
        ml_scores = safe_append(ml_scores, score)
        try:
            ents = coordinator.extract_entities(h)
        except Exception:
            ents = []
        if ents:
            ent, label = ents[0]
            nlp_suggestions = safe_append(nlp_suggestions, (h, ent, label))
        else:
            nlp_suggestions = safe_append(nlp_suggestions, (h, None, None))

    avg_score = sum(ml_scores) / len(ml_scores) if ml_scores else 0.0
    auto_accept_threshold = 0.93  # Accept automatically if ML is very confident

    # If ML confidence is low and NLP suggests better header names, auto-apply those suggestions
    if avg_score < 0.7 and any(ent and ent != h for h, ent, label in nlp_suggestions):
        _emit("info", "builder", "[TABLE_BUILDER] ML confidence low; auto-applying NLP header suggestions", session_id, contest=contest)
        alt = safe_copy(new_headers)
        for idx, (h, ent, label) in enumerate(nlp_suggestions):
            if ent and ent != h and idx < len(alt):
                alt[idx] = ent
                new_headers[idx] = ent
        new_headers, data = harmonize_headers_and_data(new_headers, data)
        try:
            ml_scores = [coordinator.score_header(h, {"contest": contest}) for h in new_headers]
        except Exception:
            ml_scores = [0.0 for _ in new_headers]
        avg_score = sum(ml_scores) / len(ml_scores) if ml_scores else 0.0

    # Multiple structure candidates (if available)
    structure_candidates = [safe_copy(new_headers)]
    alt_headers = []
    for idx, (h, ent, label) in enumerate(nlp_suggestions):
        if ent and ent != h and idx < len(new_headers):
            alt = safe_copy(new_headers)
            alt[idx] = ent
            alt_headers = safe_append(alt_headers, alt)
    if alt_headers:
        structure_candidates += alt_headers

    candidate_idx = 0
    while True:
        candidate_headers = structure_candidates[candidate_idx]

        # Preview log
        preview = {
            "headers": candidate_headers,
            "rows_preview": [
                {h: safe_get(row, h, "") for h in candidate_headers} for row in data[:5]
            ]
        }
        _emit("info", "builder", "[TABLE_BUILDER] Candidate structure preview", session_id, contest=contest, candidate_index=candidate_idx+1, candidates_total=len(structure_candidates), preview=preview, ml_avg_confidence=round(avg_score, 3))

        # Also show in rich table for CLI users (optional)
        try:
            preview_table = Table(show_header=True, header_style="bold magenta")
            for h in candidate_headers:
                preview_table.add_column(h)
            for row in data[:5]:
                preview_table.add_row(*(str(safe_get(row, h, "")) for h in candidate_headers))
            logger.alert(preview_table)
        except Exception:
            pass

        # Auto-accept if ML is very confident
        if avg_score >= auto_accept_threshold:
            _emit("info", "builder", "[TABLE_BUILDER] Auto-accepting structure due to high ML confidence", session_id, confidence=round(avg_score, 3))
            new_headers = candidate_headers
            break

        # Interactive options (CLI)
        logger.info({
            "level": "INFO",
            "type": "builder",
            "message": "[TABLE_BUILDER] Options: [Y] Accept | [N] Reject | [C] Remove columns | [O] Reorder | [R] Rename | [A] Add | [Next]/[Prev]",
            "session_id": session_id
        })

        # Raw input prompt (CLI only)
        resp = input("Accept, Reject, mark Columns, reorder, Rename, Add, Next, or Prev? [Y/n/c/o/r/a/next/prev]: ").strip().lower()
        if resp in ("", "y", "yes"):
            if should_log and hasattr(coordinator, "log_table_structure"):
                try:
                    coordinator.log_table_structure(domain, new_headers, data)
                except Exception:
                    pass
            new_headers, data = harmonize_headers_and_data(new_headers, data)
            if columns_changed:
                _emit("info", "builder", "[TABLE_BUILDER] Columns were changed by user before acceptance", session_id)
            return new_headers, data
        elif resp in ("n", "no"):
            denied_structures[sig] = safe_get(denied_structures, sig, 0) + 1
            denied_count = denied_structures[sig]
            with open(denied_structures_path, "wb") as f:
                f.write(orjson.dumps(denied_structures, option=orjson.OPT_INDENT_2))
            _emit("info", "builder", "[TABLE_BUILDER] User declined structure", session_id, contest=contest, denied_count=denied_count)
            retry = input("Would you like to retry correction? [y/N]: ").strip().lower()
            if retry in ("y", "yes"):
                continue
            else:
                return headers, data
        elif resp == "c":
            print("Enter column numbers (comma-separated) that are incorrect (starting from 1):")
            for idx, h in enumerate(candidate_headers):
                print(f"  {idx+1}: {h}")
            wrong_cols = input("Columns to mark as incorrect: ")
            if wrong_cols:
                wrong_idxs = [int(i)-1 for i in wrong_cols.split(",") if i.strip().isdigit()]
                for idx in wrong_idxs:
                    if 0 <= idx < len(candidate_headers):
                        col_name = candidate_headers[idx]
                        _emit("warning", "builder", f"[TABLE_BUILDER] Column marked incorrect: {col_name}", session_id, contest=contest)
                        removed_columns_log.setdefault(contest, {})
                        removed_columns_log[contest][col_name] = safe_get(removed_columns_log[contest], col_name, 0) + 1
                candidate_headers = [h for i, h in enumerate(candidate_headers) if i not in wrong_idxs]
                data = [{h: safe_get(row, h, "") for h in candidate_headers} for row in data]
                columns_changed = True
                structure_candidates[candidate_idx] = candidate_headers
            with open(removed_columns_log_path, "wb") as f:
                f.write(orjson.dumps(removed_columns_log, option=orjson.OPT_INDENT_2))
        elif resp == "o":
            print("Enter new order of columns as space/comma-separated numbers (starting from 1):")
            for idx, h in enumerate(candidate_headers):
                print(f"  {idx+1}: {h}")
            order = input("New order: ").replace(",", " ").split()
            try:
                new_order = [candidate_headers[int(i)-1] for i in order if i.strip().isdigit() and 0 < int(i) <= len(candidate_headers)]
                if new_order:
                    candidate_headers = new_order
                    data = [{h: safe_get(row, h, "") for h in candidate_headers} for row in data]
                    columns_changed = True
                    structure_candidates[candidate_idx] = candidate_headers
                    _emit("info", "builder", "[TABLE_BUILDER] Columns reordered", session_id)
            except Exception as e:
                _emit("error", "builder", "[TABLE_BUILDER] Invalid reorder sequence", session_id, error=str(e))
        elif resp == "r":
            print("Enter column numbers (comma-separated) to rename (starting from 1):")
            for idx, h in enumerate(candidate_headers):
                print(f"  {idx+1}: {h}")
            col_nums = input("Columns to rename: ").strip()
            if col_nums:
                rename_idxs = [int(i)-1 for i in col_nums.split(",") if i.strip().isdigit() and 0 <= int(i)-1 < len(candidate_headers)]
                for idx in rename_idxs:
                    old_name = candidate_headers[idx]
                    new_name = input(f"Rename column '{old_name}' to: ").strip()
                    if new_name:
                        candidate_headers[idx] = new_name
                data = [{h: safe_get(row, h, "") for h in candidate_headers} for row in data]
                columns_changed = True
                structure_candidates[candidate_idx] = candidate_headers
        elif resp == "a":
            print("Enter names of columns to add, separated by commas:")
            add_cols = input("Columns to add: ").split(",")
            for col in add_cols:
                col = col.strip()
                if col and col not in candidate_headers:
                    candidate_headers = safe_append(candidate_headers, col)
                    for row in data:
                        row[col] = safe_get(row, col, "")
                    _emit("info", "builder", f"[TABLE_BUILDER] Added column '{col}'", session_id)
            columns_changed = True
            structure_candidates[candidate_idx] = candidate_headers
            for row in data:
                for col in candidate_headers:
                    if col not in row:
                        row[col] = ""
        elif resp in ("next", "nxt"):
            candidate_idx = (candidate_idx + 1) % len(structure_candidates)
            continue
        elif resp in ("prev", "previous"):
            candidate_idx = (candidate_idx - 1) % len(structure_candidates)
            continue
        else:
            _emit("error", "builder", "[TABLE_BUILDER] Unknown option in confirmation prompt", session_id)

        # Always harmonize after user modification
        candidate_headers, data = harmonize_headers_and_data(candidate_headers, data)

    # Save user-confirmed structure for future ML learning
    if should_log and hasattr(coordinator, "log_table_structure"):
        try:
            coordinator.log_table_structure(contest, new_headers, context={"domain": domain})
            cache_table_structure(domain, new_headers, new_headers)
            _emit("info", "builder", "[TABLE_BUILDER] Logged confirmed table structure", session_id, contest=contest)
            if hasattr(coordinator, "save_table_structure_to_db"):
                coordinator.save_table_structure_to_db(
                    contest=contest,
                    headers=new_headers,
                    context={"domain": domain},
                    ml_confidence=avg_score if 'avg_score' in locals() else None,
                    confirmed_by_user=True
                )
        except Exception as e:
            _emit("warning", "builder", "[TABLE_BUILDER] Failed to persist table structure logs", session_id, error=str(e))

    # Always harmonize before returning
    new_headers, data = harmonize_headers_and_data(new_headers, data)
    if columns_changed:
        _emit("info", "builder", "[TABLE_BUILDER] Final structure had user column changes", session_id)
    return new_headers, data

# ===================================================================
# OPTIONAL: BATCH OPERATIONS AND SUGGESTIONS
# ===================================================================

def interactive_batch_operations(headers, data) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Allow batch renaming, reordering, or removal of columns in the CLI.
    """
    history = []
    while True:
        print("\n[Batch Ops] [R]ename, [O]rder, [D]elete, [U]ndo, [Q]uit")
        cmd = input("Choose operation: ").strip().lower()
        if cmd == "r":
            print("Enter column numbers (comma-separated) to rename:")
            for idx, h in enumerate(headers):
                print(f"  {idx+1}: {h}")
            col_nums = input("Columns to rename: ").strip()
            if col_nums:
                rename_idxs = [int(i)-1 for i in col_nums.split(",") if i.strip().isdigit() and 0 <= int(i)-1 < len(headers)]
                history.append((copy.deepcopy(headers), copy.deepcopy(data)))
                for idx in rename_idxs:
                    old_name = headers[idx]
                    new_name = input(f"Rename column '{old_name}' to: ").strip()
                    if new_name:
                        headers[idx] = new_name
                data = [{h: row.get(h, "") for h in headers} for row in data]
        elif cmd == "o":
            print("Enter new order of columns as space/comma-separated numbers (starting from 1):")
            for idx, h in enumerate(headers):
                print(f"  {idx+1}: {h}")
            order = input("New order: ").replace(",", " ").split()
            try:
                new_order = [headers[int(i)-1] for i in order if i.strip().isdigit() and 0 < int(i) <= len(headers)]
                if new_order:
                    history.append((copy.deepcopy(headers), copy.deepcopy(data)))
                    headers = new_order
                    data = [{h: row.get(h, "") for h in headers} for row in data]
            except Exception as e:
                _emit("error", "builder", "[TABLE_BUILDER] Invalid order in batch ops", None, error=str(e))
        elif cmd == "d":
            print("Enter column numbers (comma-separated) to delete:")
            for idx, h in enumerate(headers):
                print(f"  {idx+1}: {h}")
            del_nums = input("Columns to delete: ").strip()
            if del_nums:
                del_idxs = [int(i)-1 for i in del_nums.split(",") if i.strip().isdigit() and 0 <= int(i)-1 < len(headers)]
                history.append((copy.deepcopy(headers), copy.deepcopy(data)))
                headers = [h for i, h in enumerate(headers) if i not in del_idxs]
                data = [{h: row.get(h, "") for h in headers} for row in data]
        elif cmd == "u":
            if history:
                headers, data = history.pop()
                print("[Batch Ops] Undo successful.")
            else:
                print("[Batch Ops] Nothing to undo.")
        elif cmd == "q":
            break
        else:
            print("[Batch Ops] Unknown option.")
    return headers, data

def auto_suggest_corrections(headers, data, coordinator):
    """
    Suggest likely corrections based on previous user feedback or ML confidence.
    Examines both headers and the data content for issues.
    """
    from ..Context_Integration.context_coordinator import ContextCoordinator
    coordinator = coordinator or ContextCoordinator()
    suggestions = []

    # Suggest based on ML confidence for headers
    for h in headers:
        try:
            score = coordinator.score_header(h, {})
        except Exception:
            score = 0.0
        if score < 0.7:
            suggestions.append((h, "Low ML confidence"))

    # Suggest if any header is empty or non-alphanumeric
    for h in headers:
        h_clean = safe_replace(safe_strip(h), " ", "")
        if not h_clean or not safe_isalnum(h_clean):
            suggestions.append((h, "Header is empty or not alphanumeric"))

    # Suggest if any column is mostly empty in the data
    for h in headers:
        empty_count = sum(1 for row in data if not safe_strip(safe_get(row, h, "")))
        if data and empty_count > len(data) * 0.7:
            suggestions.append((h, "Column is mostly empty in data"))

    # Suggest if any row is missing values for required headers
    for idx, row in enumerate(data):
        missing = [h for h in headers if not safe_strip(safe_get(row, h, ""))]
        if missing:
            suggestions.append((f"Row {idx+1}", f"Missing values for columns: {missing}"))

    # Feedback log based suggestions (if available)
    if hasattr(coordinator, "get_feedback_log"):
        try:
            feedback_log = coordinator.get_feedback_log()
        except Exception:
            feedback_log = {}
        for h in headers:
            h_norm = safe_lower(safe_strip(h))
            if h_norm in feedback_log.get("removed_columns", {}):
                count = feedback_log["removed_columns"][h_norm]
                if count > 2:
                    suggestions.append((h, f"Column '{h}' was removed {count} times in past feedback"))
            if h_norm in feedback_log.get("renamed_columns", {}):
                new_name = feedback_log["renamed_columns"][h_norm]
                suggestions.append((h, f"Column '{h}' was often renamed to '{new_name}'"))

    return suggestions

def dynamic_confidence_threshold(history, coordinator=None, default=0.93):
    """
    Adjust threshold for auto-accepting structures based on past accuracy and feedback log.
    If a ContextCoordinator is provided, use its feedback analytics for smarter adjustment.
    Uses safe_get and safe_values for robustness.
    """
    if coordinator is None:
        from ..Context_Integration.context_coordinator import ContextCoordinator
        coordinator = ContextCoordinator()
    threshold = default

    if history:
        correct = sum(1 for h in history[-5:] if safe_get(h, "accepted"))
        if correct >= 4:
            threshold = min(0.98, threshold + 0.02)
        elif correct <= 2:
            threshold = max(0.85, threshold - 0.05)

    if coordinator and hasattr(coordinator, "get_feedback_log"):
        try:
            feedback = coordinator.get_feedback_log()
        except Exception:
            feedback = {}
        denials = sum(safe_values(safe_get(feedback, "structure_denials", {})))
        removals = sum(safe_values(safe_get(feedback, "removed_columns", {})))
        total_feedback = denials + removals
        if total_feedback > 10:
            threshold = min(0.99, threshold + 0.03)
        elif denials > removals and denials > 5:
            threshold = min(0.99, threshold + 0.04)
        elif removals > 10:
            threshold = max(0.85, threshold - 0.03)
    return threshold

# ===================================================================
# END OF FILE
# ===================================================================