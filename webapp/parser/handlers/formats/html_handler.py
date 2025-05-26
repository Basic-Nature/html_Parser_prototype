from ...state_router import get_handler_from_context
from ...utils.contest_selector import select_contest
from ...utils.table_builder import extract_table_data, calculate_grand_totals
from ...utils.html_scanner import scan_html_for_context, get_detected_races_from_context
from ...utils.shared_logger import logger
from ...utils.shared_logic import (
    normalize_text, find_and_click_toggle,
    ALL_SELECTORS
)
from rich import print as rprint
import os
import re
import json

# --- Load tag configs from context library ---
CONTEXT_LIBRARY_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "Context_Integration", "context_library.json"
)
if os.path.exists(CONTEXT_LIBRARY_PATH):
    with open(CONTEXT_LIBRARY_PATH, "r", encoding="utf-8") as f:
        CONTEXT_LIBRARY = json.load(f)
    PRECINCT_ELEMENT_TAGS = CONTEXT_LIBRARY.get("precinct_element_tags", ["h3", "h4", "h5", "strong", "b", "span", "div"])
    CONTEST_PANEL_TAGS = CONTEXT_LIBRARY.get("contest_panel_tags", ["div", "section", "article"])
    
else:
    logger.error("[html_handler] context_library.json not found. Using default tags.")
    PRECINCT_ELEMENT_TAGS = ["h3", "h4", "h5", "strong", "b", "span", "div"]
    CONTEST_PANEL_TAGS = ["div", "section", "article"]

def extract_contest_panel(page, contest_title, panel_tags=None):
    """
    Returns a Playwright Locator for the contest panel matching contest_title.
    Tries all tags in panel_tags (default: CONTEST_PANEL_TAGS), then falls back to ALL_SELECTORS.
    Implements early break logic for efficiency.
    """
    panel_tags = panel_tags or CONTEST_PANEL_TAGS
    selectors_to_try = [
        ", ".join(panel_tags),  # Try the specific panel tags first
        ALL_SELECTORS           # Fallback: try all selectors
    ]
    for selector in selectors_to_try:
        panels = page.locator(selector)
        if panels.count() == 0:
            continue
        for i in range(panels.count()):
            panel = panels.nth(i)
            header_text = panel.inner_text().strip().lower()
            if contest_title and contest_title.lower() in header_text:
                return panel  # EARLY BREAK: found a match
    return None

def extract_precinct_tables(panel):
    """
    Returns a list of (precinct_name, table_element) tuples from a contest panel.
    Accepts a Playwright Locator as panel.
    Tries default tags, then falls back to ALL_SELECTORS.
    Implements early break logic for efficiency.
    """
    selectors_to_try = [
        ', '.join(PRECINCT_ELEMENT_TAGS),
        ALL_SELECTORS
    ]
    precincts = []
    for selector in selectors_to_try:
        elements = panel.locator(selector)
        if elements.count() == 0:
            continue
        current_precinct = None
        for i in range(elements.count()):
            el = elements.nth(i)
            tag = el.evaluate("e => e.tagName").strip().lower()
            if tag in PRECINCT_ELEMENT_TAGS:
                label = el.inner_text().strip()
                current_precinct = label
            elif tag == "table" and current_precinct:
                precincts.append((current_precinct, el))
                current_precinct = None
        if precincts:
            break  # EARLY BREAK: found precincts, stop searching
    return precincts

def parse(page, html_context=None):
    """
    Generic HTML handler: cleans, filters, and extracts data from HTML using shared utilities.
    Delegates to state/county handler if available.
    Returns headers, data, contest_title, metadata.
    """
    # 1. Scan the page for context and races
    context = scan_html_for_context(page, debug=False)
    if html_context is None:
        html_context = {}
    html_context.update(context)

    # 2. Try to delegate to state/county handler if available
    state_handler = get_handler_from_context(
        context=html_context
    )
    if state_handler and hasattr(state_handler, "parse"):
        logger.info(f"[HTML Handler] Redirecting to state handler: {getattr(state_handler, '__name__', str(state_handler))}...")
        return state_handler.parse(page, html_context)

    # 3. Let user select contest if not already set
    contest_title = html_context.get("selected_race")
    if not contest_title:
        detected = get_detected_races_from_context(html_context)
        contest_title = select_contest(detected)
        if not contest_title:
            return None, None, None, {"skipped": True}
        # If select_contest returns a list, pick the first contest's title
        if isinstance(contest_title, list):
            if contest_title and isinstance(contest_title[0], tuple):
                contest_title = contest_title[0][2]  # (year, etype, race)
            elif contest_title and isinstance(contest_title[0], str):
                contest_title = contest_title[0]
        html_context["selected_race"] = contest_title

    # 4. Try to find and click toggles dynamically using handler-supplied keywords
    handler_keywords = html_context.get("toggle_keywords", ["View Results", "Vote Method", "Show Results"])
    clicked = find_and_click_toggle(
        page,
        container=None,
        handler_keywords=handler_keywords,
        logger=logger,
        verbose=True,
        interactive=True
    )
    if not clicked:
        logger.warning("[HTML Handler] No toggle/button clicked automatically or interactively.")

    # 5. Try to extract contest panel and precinct tables
    headers, data = [], []
    try:
        contest_panel = extract_contest_panel(page, contest_title)
        if contest_panel:
            precinct_tables = extract_precinct_tables(contest_panel)
            if precinct_tables:
                # Try to get method names from the first table header
                first_table = precinct_tables[0][1]
                method_names = []
                header_locator = first_table.locator("thead tr th")
                if header_locator.count() == 0:
                    header_locator = first_table.locator("tbody tr:first-child th")
                for i in range(1, header_locator.count() - 1):  # skip first (candidate), last (total)
                    method_names.append(header_locator.nth(i).inner_text().strip())
                data = []
                for precinct_name, table in precinct_tables:
                   headers, rows = extract_table_data(table)
                   for row in rows:
                       row["Precinct"] = precinct_name
                       data.append(row)
                if data:
                    # Build headers from all keys
                    all_keys = set()
                    for row in data:
                        all_keys.update(row.keys())
                    headers = ["Precinct", "% Precincts Reporting"] + sorted([k for k in all_keys if k not in ("Precinct", "% Precincts Reporting")])
                    # Add grand total row
                    data.append(calculate_grand_totals(data))
                    logger.info(f"[HTML Handler] Extracted {len(data)-1} precinct rows from contest panel.")
                    # Output metadata
                    clean_race = normalize_text(contest_title).strip().lower() if contest_title else "unknown"
                    clean_race = re.sub(r'[\s:]+', ' ', clean_race).strip()
                    clean_race = re.sub(r'[\\/:*?"<>|]', '_', clean_race)
                    metadata = {
                        "race": clean_race,
                        "source": page.url,
                        "handler": "html_handler",
                        "state": html_context.get("state", "Unknown"),
                        "county": html_context.get("county", None),
                        "year": html_context.get("year", "Unknown")
                    }
                    return headers, data, contest_title, metadata
        # Fallback: Try to extract the first table on the page
        table = page.query_selector("table")
        if not table:
            table = page.query_selector("table#resultsTable, table.results-table")
        if not table:
            raise RuntimeError("No table found on the page.")
        headers, data = extract_table_data(table)
        logger.info(f"[HTML Handler] Extracted {len(data)} rows from the table.")
    except Exception as e:
        logger.error(f"[ERROR] Failed to extract table from page: {e}")
        return None, None, None, {"error": str(e)}

    # 6. Output metadata
    clean_race = normalize_text(contest_title).strip().lower() if contest_title else "unknown"
    clean_race = re.sub(r'[\s:]+', ' ', clean_race).strip()
    clean_race = re.sub(r'[\\/:*?"<>|]', '_', clean_race)
    metadata = {
        "race": clean_race,
        "source": page.url,
        "handler": "html_handler",
        "state": html_context.get("state", "Unknown"),
        "county": html_context.get("county", None),
        "year": html_context.get("year", "Unknown")
    }
    return headers, data, contest_title, metadata