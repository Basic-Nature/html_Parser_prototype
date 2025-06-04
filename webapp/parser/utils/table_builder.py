# table_builder.py
# ===================================================================
# Election Data Cleaner - Table Extraction and Cleaning Utilities
# Context-integrated version: uses ContextCoordinator for config
# ===================================================================

from typing import List, Dict, Tuple, Any, Optional
from .logger_instance import logger

from typing import TYPE_CHECKING
from .table_core import (
    robust_table_extraction,
    extract_table_data,
    harmonize_headers_and_data,
    guess_headers_from_row,
    detect_table_structure,
    handle_candidate_major,
    handle_precinct_major,
    extract_rows_and_headers_from_dom,
    fallback_nlp_candidate_vote_scan,
    extract_with_patterns,
    handle_ambiguous
    
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
    logger.info(f"[TABLE BUILDER][BUILD_DYNAMIC_TABLE] Using contest_title: '{contest_title}'")

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
            logger.info(f"[TABLE BUILDER][BUILD_DYNAMIC_TABLE] Applied merged learned structure for '{contest_title}'.")
            logger.info(f"[TABLE BUILDER][BUILD_DYNAMIC_TABLE] Output headers: {merged_headers}")
            logger.info(f"[TABLE BUILDER][BUILD_DYNAMIC_TABLE] Output sample row: {merged_data[0] if merged_data else 'NO DATA'}")
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
            logger.info(f"[TABLE BUILDER][BUILD_DYNAMIC_TABLE] Extraction attempt {extract_fn.__name__}: headers={cand_headers}, rows={len(cand_data)}")
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
            logger.warning(f"[TABLE BUILDER][BUILD_DYNAMIC_TABLE] Extraction attempt {extract_fn.__name__} failed: {e}")

    # 5. Rank candidates by ML/NLP confidence and keyword/entity matching
    def candidate_rank(cand):
        # Prefer higher ML score, more rows, more columns
        return (cand["ml_score"], len(cand["data"]), len(cand["headers"]))
    candidates = sorted(candidates, key=candidate_rank, reverse=True)

    # 6. Pick the best candidate, or fallback to ambiguous handler
    if candidates:
        best = candidates[0]
        headers, data = best["headers"], best["data"]
        logger.info(f"[TABLE BUILDER][BUILD_DYNAMIC_TABLE] Selected extraction from {best['source']} with ML score {best['ml_score']:.2f}")
        logger.info(f"[TABLE BUILDER][BUILD_DYNAMIC_TABLE] Output headers: {headers}")
        logger.info(f"[TABLE BUILDER][BUILD_DYNAMIC_TABLE] Output sample row: {data[0] if data else 'NO DATA'}")
    else:
        logger.warning("[TABLE BUILDER][BUILD_DYNAMIC_TABLE] No extraction method succeeded, using fallback handler.")
        headers, data = [], []

    # 7. Harmonize headers/data after extraction
    headers, data = harmonize_headers_and_data(headers, data)
    logger.info(f"[TABLE BUILDER][BUILD_DYNAMIC_TABLE] Harmonized headers: {headers}")
    logger.info(f"[TABLE BUILDER][BUILD_DYNAMIC_TABLE] Harmonized sample row: {data[0] if data else 'NO DATA'}")

    # 8. Use NLP/NER to cross-reference DOM and refine structure
    structure_info = detect_table_structure(headers, data, coordinator)
    logger.info(f"[TABLE BUILDER][BUILD_DYNAMIC_TABLE] Detected table structure: {structure_info}")

    from .table_core import progressive_table_verification, interactive_feedback_loop
    headers, data, structure_info = progressive_table_verification(headers, data, coordinator, context)
    if not structure_info.get("verified"):
        headers, data, structure_info = interactive_feedback_loop(headers, data, structure_info)

    if structure_info["type"] == "candidate-major":
        logger.info("[TABLE BUILDER][BUILD_DYNAMIC_TABLE] Using candidate-major handler.")
        return handle_candidate_major(headers, data, coordinator, context)
    elif structure_info["type"] == "precinct-major":
        logger.info("[TABLE BUILDER][BUILD_DYNAMIC_TABLE] Using precinct-major handler.")
        return handle_precinct_major(headers, data, coordinator, context)
    else:
        logger.info("[TABLE BUILDER][BUILD_DYNAMIC_TABLE] Using ambiguous handler.")
        return handle_ambiguous(headers, data, coordinator, context)




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
