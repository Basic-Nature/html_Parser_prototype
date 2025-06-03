# table_builder.py
# ===================================================================
# Election Data Cleaner - Table Extraction and Cleaning Utilities
# Context-integrated version: uses ContextCoordinator for config
# ===================================================================

import re
import os
import json
from typing import List, Dict, Tuple, Any, Optional
from .logger_instance import logger
from .shared_logger import rprint
from .shared_logic import normalize_text
from typing import TYPE_CHECKING
from .dynamic_table_extractor import (
    extract_table_data,
    extract_rows_and_headers_from_dom,
    fallback_nlp_candidate_vote_scan,
    extract_with_patterns,
    guess_headers_from_row,
    harmonize_headers_and_data,
    detect_table_structure,
    handle_candidate_major,
    handle_precinct_major,
    handle_ambiguous,
    merge_table_data
)
if TYPE_CHECKING:
    from ..Context_Integration.context_coordinator import ContextCoordinator
context_cache = {}
from ..config import BASE_DIR

def build_dynamic_table(
    domain: str,
    headers: List[str],
    data: List[Dict[str, Any]],
    coordinator: "ContextCoordinator",
    context: dict = None,
    max_feedback_loops: int = 3,
    learning_mode: bool = True,
    confirm_table_structure_callback=None
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Dynamically builds a uniform, context-aware table using DOM, NLP, and ML.
    - Scans the DOM for all possible tabular structures (tables, repeated divs, ARIA regions, etc.)
    - Uses NLP/NER to identify headers/entities and cross-references DOM hierarchy for context
    - Ranks and merges candidate structures using ML/NLP confidence and keyword/entity matching
    - Always harmonizes and allows for user correction if needed
    """
    if context is None:
        context = {}

    # 1. Ensure contest title is set for context and logging
    if "contest_title" not in context or not context["contest_title"]:
        context["contest_title"] = (
            context.get("selected_race")
            or context.get("title")
            or "Unknown Contest"
        )
    contest_title = context["contest_title"]
    logger.info(f"[TABLE BUILDER] Using contest_title: '{contest_title}'")

    # 2. Try to use a learned structure if available
    learned_structure = None
    if hasattr(coordinator, "get_table_structure"):
        learned_structure = coordinator.get_table_structure(contest_title, context=context, learning_mode=True)
    if not learned_structure and hasattr(coordinator, "get_table_structure_from_db"):
        learned_structure = coordinator.get_table_structure_from_db(contest_title, context=context)
    learned_headers = []
    if learned_structure:
        if isinstance(learned_structure, list):
            if learned_structure and isinstance(learned_structure[0], dict):
                learned_structure = learned_structure[0]
            else:
                learned_structure = {}
        learned_headers = learned_structure.get("headers", [])
        if learned_headers and data:
            merged_headers = list(learned_headers)
            for h in headers:
                if h not in merged_headers:
                    merged_headers.append(h)
            merged_headers, merged_data = harmonize_headers_and_data(merged_headers, data)
            logger.info(f"[TABLE BUILDER] Applied merged learned structure for '{contest_title}'.")
            return merged_headers, merged_data

    # 3. If no headers/data, attempt all extraction strategies in order of reliability
    extraction_attempts = [
        # 1. Standard HTML table extraction
        lambda page, ctx: extract_table_data(page.locator("table").first) if page.locator("table").count() > 0 else ([], []),
        # 2. Pattern-based extraction using learned DOM patterns
        lambda page, ctx: (
            (lambda pr: (
                guess_headers_from_row(pr[0][1]),
                [
                    {
                        guess_headers_from_row(pr[0][1])[idx] if idx < len(guess_headers_from_row(pr[0][1])) else f"Column {idx+1}": row.locator("> *").nth(idx).inner_text().strip()
                        for idx in range(row.locator("> *").count())
                    }
                    for _, row, _ in pr
                ]
            ) if pr else ([], []))(extract_with_patterns(page, ctx))
        ),
        # 3. Extraction from repeated DOM structures (divs, lists, etc.)
        lambda page, ctx: extract_rows_and_headers_from_dom(page, coordinator=coordinator),
        # 4. NLP/keyword-based fallback extraction
        lambda page, ctx: fallback_nlp_candidate_vote_scan(page),
    ]

    # 4. Try all extraction attempts and collect candidates
    page = context.get("page")
    candidates = []
    for extract_fn in extraction_attempts:
        try:
            cand_headers, cand_data = extract_fn(page, context)
            if cand_headers and cand_data:
                # Score using ML/NLP
                ml_scores = [coordinator.score_header(h, context) for h in cand_headers]
                avg_score = sum(ml_scores) / len(ml_scores) if ml_scores else 0
                candidates.append({
                    "headers": cand_headers,
                    "data": cand_data,
                    "ml_score": avg_score,
                    "source": extract_fn.__name__,
                })
        except Exception as e:
            logger.warning(f"[TABLE BUILDER] Extraction attempt {extract_fn.__name__} failed: {e}")

    # 5. Rank candidates by ML/NLP confidence and keyword/entity matching
    def candidate_rank(cand):
        # Prefer higher ML score, more rows, more columns
        return (cand["ml_score"], len(cand["data"]), len(cand["headers"]))
    candidates = sorted(candidates, key=candidate_rank, reverse=True)

    # 6. Pick the best candidate, or fallback to ambiguous handler
    if candidates:
        best = candidates[0]
        headers, data = best["headers"], best["data"]
        logger.info(f"[TABLE BUILDER] Selected extraction from {best['source']} with ML score {best['ml_score']:.2f}")
    else:
        logger.warning("[TABLE BUILDER] No extraction method succeeded, using fallback handler.")
        headers, data = [], []

    # 7. Harmonize headers/data after extraction
    headers, data = harmonize_headers_and_data(headers, data)

    # 8. Use NLP/NER to cross-reference DOM and refine structure
    structure_info = detect_table_structure(headers, data, coordinator)
    logger.info(f"[TABLE BUILDER] Detected table structure: {structure_info}")

    if structure_info["type"] == "candidate-major":
        return handle_candidate_major(headers, data, coordinator, context)
    elif structure_info["type"] == "precinct-major":
        return handle_precinct_major(headers, data, coordinator, context)
    else:
        # As a last resort, try ambiguous handler (which tries both major types and picks the best)
        return handle_ambiguous(headers, data, coordinator, context)


def robust_table_extraction(page, extraction_context=None, existing_headers=None, existing_data=None):
    """
    Attempts all extraction strategies in order, merging partial results.
    Uses extraction_context for all steps and logs context/anomalies.
    DOM pattern extraction is prioritized.

    - If existing_headers/data are provided and non-empty, merges new results into them.
    - Never overwrites non-empty headers/data unless extraction fails.
    - Always harmonizes headers after merging.
    """
    import types

    def safe_json(obj):
        """Recursively remove non-serializable objects (like functions, classes, custom objects) from dicts/lists."""
        import types
        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                if k in ("coordinator", "ContextCoordinator"):
                    continue
                if isinstance(v, (types.FunctionType, types.ModuleType)) or hasattr(v, "__dict__"):
                    continue
                try:
                    json.dumps(v)
                    result[k] = safe_json(v)
                except Exception:
                    continue
            return result
        elif isinstance(obj, list):
            return [safe_json(v) for v in obj if not hasattr(v, "__dict__")]
        else:
            return obj

    headers_list = []
    data_list = []
    extraction_logs = []

    # If existing headers/data are provided and non-empty, add them first
    if existing_headers and existing_data and len(existing_headers) > 0 and len(existing_data) > 0:
        headers_list.append(existing_headers)
        data_list.append(existing_data)

    def log_page_html(page, context, prefix=""):
        """Save the current page HTML for debugging extraction issues."""
        try:
            html = page.content()
            contest_title = context.get("selected_race") or context.get("contest_title") or "unknown"
            safe_title = re.sub(r'[^a-zA-Z0-9_\-]', '_', contest_title)[:40]
            fname = f"debug_{prefix}{safe_title}.html"
            fpath = os.path.join(BASE_DIR, "log", fname)
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(html)
            logger.info(f"[TABLE BUILDER] Saved page HTML to {fpath}")
        except Exception as e:
            logger.error(f"[TABLE BUILDER] Could not save page HTML: {e}")

    # 1. Pattern-based extraction (approved DOM patterns, prioritized)
    try:
        pattern_rows = extract_with_patterns(page, extraction_context)
        if pattern_rows:
            headers = guess_headers_from_row(pattern_rows[0][1])
            data = []
            for heading, row, pat in pattern_rows:
                cells = row.locator("> *")
                row_data = {}
                for idx in range(cells.count()):
                    row_data[headers[idx] if idx < len(headers) else f"Column {idx+1}"] = cells.nth(idx).inner_text().strip()
                if row_data:
                    data.append(row_data)
            extraction_logs.append({
                "method": "pattern",
                "headers": headers,
                "rows": len(data),
                "columns": len(headers),
                "success": bool(headers and data),
                "context": extraction_context
            })
            if headers == ["Label", "Votes"] and data:
                possible_headers = list(data[0].values())
                if all(isinstance(h, str) and len(h) > 0 for h in possible_headers):
                    headers = possible_headers
                    data = data[1:]
    except Exception as e:
        logger.error(f"[TABLE BUILDER] Pattern extraction failed: {e}")
        extraction_logs.append({
            "method": "pattern",
            "error": str(e),
            "success": False,
            "context": extraction_context
        })

    # 2. Standard HTML table extraction
    try:
        tables = page.locator("table")
        for i in range(tables.count()):
            headers, data = extract_table_data(tables.nth(i))
            extraction_logs.append({
                "method": "table",
                "headers": headers,
                "rows": len(data),
                "columns": len(headers),
                "success": bool(headers and data),
                "context": extraction_context
            })
            if headers == ["Label", "Votes"] and data:
                possible_headers = list(data[0].values())
                if all(isinstance(h, str) and len(h) > 0 for h in possible_headers):
                    headers = possible_headers
                    data = data[1:]
    except Exception as e:
        logger.error(f"[TABLE BUILDER] Table extraction failed: {e}")
        extraction_logs.append({
            "method": "table",
            "error": str(e),
            "success": False,
            "context": extraction_context
        })

    # 3. Repeated DOM structures (divs, lists, etc.)
    try:
        headers, data = extract_rows_and_headers_from_dom(page)
        logger.info(f"[TABLE BUILDER] DOM structure headers: {headers}")
        logger.info(f"[TABLE BUILDER] First 3 rows: {data[:3]}")
        extraction_logs.append({
            "method": "repeated_dom",
            "headers": headers,
            "rows": len(data),
            "columns": len(headers),
            "success": bool(headers and data),
            "context": extraction_context
        })
        if headers == ["Label", "Votes"] and data:
            possible_headers = list(data[0].values())
            if all(isinstance(h, str) and len(h) > 0 for h in possible_headers):
                headers = possible_headers
                data = data[1:]
    except Exception as e:
        logger.error(f"[TABLE BUILDER] Repeated DOM extraction failed: {e}")
        extraction_logs.append({
            "method": "repeated_dom",
            "error": str(e),
            "success": False,
            "context": extraction_context
        })

    # 4. NLP/keyword-based fallback extraction
    try:
        headers, data = fallback_nlp_candidate_vote_scan(page)
        # PATCH: If first data row looks like headers, use it
        if headers == ["Label", "Votes"] and data:
            possible_headers = list(data[0].values())
            if all(isinstance(h, str) and len(h) > 0 for h in possible_headers):
                headers = possible_headers
                data = data[1:]
        extraction_logs.append({
            "method": "nlp_fallback",
            "headers": headers,
            "rows": len(data),
            "columns": len(headers),
            "success": bool(headers and data),
            "context": extraction_context
        })
        if headers and data:
            headers_list.append(headers)
            data_list.append(data)
            logger.warning("[TABLE BUILDER] Fallback NLP extraction used. Only candidate/vote pairs extracted.")
    except Exception as e:
        logger.error(f"[TABLE BUILDER] NLP fallback extraction failed: {e}")
        extraction_logs.append({
            "method": "nlp_fallback",
            "error": str(e),
            "success": False,
            "context": extraction_context
        })

    # --- Safe JSON logging ---
    logger.info(f"[TABLE BUILDER] Extraction summary: {json.dumps(safe_json(extraction_logs), indent=2)}")

    # Merge all results (including any existing headers/data)
    if headers_list and data_list:
        merged_headers, merged_data = merge_table_data(headers_list, data_list)
        merged_headers, merged_data = harmonize_headers_and_data(merged_headers, merged_data)
        logger.info(f"[TABLE BUILDER] Merged extraction: {len(merged_data)} rows, {len(merged_headers)} columns.")
        return merged_headers, merged_data

    logger.warning("[TABLE BUILDER] No extraction method succeeded.")
    return [], []


def is_likely_header(row):
    known_fields = {"candidate", "votes", "percent", "party", "district"}
    return sum(1 for cell in row if cell.lower() in known_fields) >= 2

EXTRACTION_ATTEMPTS = [
    # 1. Standard HTML table extraction
    lambda page, context: extract_table_data(page.locator("table").first) if page.locator("table").count() > 0 else ([], []),
    # 2. Pattern-based extraction using learned DOM patterns
    lambda page, context: (
        (lambda pr: (
            guess_headers_from_row(pr[0][1]),
            [
                {
                    guess_headers_from_row(pr[0][1])[idx] if idx < len(guess_headers_from_row(pr[0][1])) else f"Column {idx+1}": row.locator("> *").nth(idx).inner_text().strip()
                    for idx in range(row.locator("> *").count())
                }
                for _, row, _ in pr
            ]
        ) if pr else ([], []))(extract_with_patterns(page, context))
    ),
    # 3. Extraction from repeated DOM structures (divs, lists, etc.)
    lambda page, context: extract_rows_and_headers_from_dom(page),
    # 4. NLP/keyword-based fallback extraction
    lambda page, context: fallback_nlp_candidate_vote_scan(page),
]

def extract_table_from_headers(headers, data, context):
    """
    Converts headers and data into a uniform table format.
    Harmonizes all rows to the union of all keys, preserving order.
    """
    if not headers or not data:
        headers, data = robust_table_extraction(context.get("page"), context)
        if not headers or not data:
            logger.error("[TABLE BUILDER] No data could be extracted from the page.")
            return [], []
    return harmonize_headers_and_data(headers, data)
