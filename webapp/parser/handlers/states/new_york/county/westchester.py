"""
Westchester County Handler (New York)

Auto-generated county-level handler for Westchester County, New York.

County handlers extend state-level logic with county-specific:
- Custom navigation sequences
- Button/toggle interactions
- Vendor-specific UI patterns
- Table structure variations

Based on Rockland County, NY reference implementation.

To customize:
1. Implement button toggle logic in parse()
2. Add navigation recipe for automation
3. Override extraction strategy if needed

Generated with: python scripts/generate_county_handler.py "New York" "Westchester"
"""
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from playwright.sync_api import Page
from webapp.parser.Context_Integration.librarian import clean_for_json
from webapp.parser.utils.browser_utils import (
    autoscroll_until_stable,
)
from webapp.parser.utils.contest_selector import select_contest_auto_first
from webapp.parser.utils.html_scanner import scan_html_for_context
from webapp.parser.utils.logger_singleton import logger
from webapp.parser.utils.shared_logic import safe_get
from webapp.parser.utils.table_core import robust_table_extraction

if TYPE_CHECKING:
    from webapp.parser.Context_Integration.context_coordinator import ContextCoordinator


# Context cache for this county
context_cache = {}
accepted_buttons_cache = {}


def parse(
    page: Page = None,
    html_context: Dict[str, Any] = None,
    coordinator: "ContextCoordinator" = None,
    context: Dict[str, Any] = None,
    session_id: str = None,
    **kwargs,
) -> Tuple[List[str], List[Dict[str, Any]], str, Dict[str, Any]]:
    """
    Westchester County handler for New York.
    
    Workflow:
    1. Scan HTML for context and contests
    2. Select contest
    3. Perform county-specific navigation (buttons, toggles, etc.)
    4. Extract tables
    5. Return results
    
    TODO: Customize this handler for Westchester County's specific UI.
    """
    if html_context is None:
        html_context = {}
    
    from webapp.parser.Context_Integration.context_coordinator import ContextCoordinator
    
    if coordinator is None:
        coordinator = ContextCoordinator()
    
    logger.info("[bold cyan][Westchester County, NY] Starting parse...[/bold cyan]")
    
    # === 1. Scan HTML for context ===
    context_result = scan_html_for_context(
        target_url=getattr(page, "url", None) if page else html_context.get("url"),
        page=page,
        coordinator=coordinator,
        session_id=session_id or getattr(coordinator, "session_id", None),
        allow_duplicates=getattr(coordinator, "allow_duplicates", False),
        context_cache=context_cache,
        debug=html_context.get("debug", False),
        **kwargs,
    )
    
    # Ensure location fields
    state = context_result.get("state") or "NY"
    county = context_result.get("county") or "Westchester"
    year = context_result.get("year")
    
    # Propagate to contests
    for contest in safe_get(context_result, "contests", []):
        contest.setdefault("state", state)
        contest.setdefault("county", county)
        if year:
            contest.setdefault("year", year)
        if session_id:
            contest["session_id"] = session_id
    
    # Clean and enrich
    context_result = clean_for_json(context_result)
    coordinator.organize_and_enrich(context_result)
    
    # === 2. Select contest ===
    context_for_selector = {
        "state": state,
        "county": county,
        "year": year,
        "contests": context_result.get("contests", []),
        "session_id": session_id,
        **{k: v for k, v in html_context.items() if k not in ("state", "county", "year", "contests")},
    }
    
    selected = select_contest_auto_first(
        coordinator=coordinator,
        context=context_for_selector,
        session_id=session_id,
        allow_multiple=False,
        force_interactive=html_context.get("force_interactive", False),
    )
    
    if not selected:
        logger.warning("[Westchester County] No contest selected")
        return None, None, None, {"skipped": True, "county": county}
    
    # Handle list return
    if isinstance(selected, list) and selected:
        selected = selected[0]
    
    logger.info(f"[Westchester County] Processing contest: {selected.get('title')}")
    
    # === 3. County-specific navigation ===
    # TODO: Add button toggles, navigation sequences, etc. specific to Westchester County
    #
    # Example button toggle pattern (based on Rockland County reference):
    #
    # toggle_keywords = ["View results by district", "Show detailed results"]
    # btn, idx = coordinator.get_best_button_advanced(
    #     page=page,
    #     contest=selected,
    #     keywords=toggle_keywords,
    #     context=html_context,
    #     learning_mode=True,
    # )
    # if btn and "element_handle" in btn:
    #     element = btn["element_handle"]
    #     if safe_is_visible(element, logger) and safe_is_enabled(element, logger):
    #         safe_click(element, logger)
    #         page.wait_for_timeout(2000)  # Wait for content to load
    
    # === 4. Autoscroll if needed ===
    if page and html_context.get("autoscroll", False):
        logger.info("[Westchester County] Autoscrolling to load all content...")
        autoscroll_until_stable(page, max_scrolls=10, scroll_delay=1000)
    
    # === 5. Extract tables ===
    result = robust_table_extraction(
        page=page,
        coordinator=coordinator,
        html_context={**html_context, "selected_contest": selected},
        session_id=session_id,
        **kwargs,
    )
    
    headers = result.get("headers", [])
    data_rows = result.get("data", [])
    
    # === 6. Build metadata ===
    metadata = {
        "state": state,
        "county": county,
        "year": year,
        "contest_title": selected.get("title"),
        "source_url": html_context.get("url") or context_result.get("url"),
        "session_id": session_id,
        "handler": f"{state.lower()}.county.westchester",
        "row_count": len(data_rows),
        "column_count": len(headers),
    }
    
    contest_title = selected.get("title", "Unknown Contest")
    logger.info(f"[green][Westchester County] Extracted {len(data_rows)} rows[/green]")
    
    return headers, data_rows, contest_title, metadata
