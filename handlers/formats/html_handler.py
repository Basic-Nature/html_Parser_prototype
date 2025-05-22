# handlers/formats/html_handler.py
# ==============================================================
# Fallback handler for generic HTML parsing.
# Used when structured formats (JSON, CSV, PDF) are not present.
# This routes through the state_router using html_scanner context.
# ==============================================================
from utils.contest_selector import select_contest
from utils.html_table_extractor import extract_table_data
from utils.shared_logic import normalize_text
from state_router import get_handler as get_state_handler
from utils.format_router import detect_format_from_links, route_format_handler
from utils.html_scanner import CONTEST_PANEL_TAGS, PRECINCT_ELEMENT_TAGS, CONTEST_HEADER_SELECTORS
from utils.download_utils import download_confirmed_file
from utils.shared_logger import logger
from rich import print as rprint
from utils.html_scanner import get_detected_races_from_context
import os
import re
# This fallback HTML handler is invoked when no state or county-specific handler is matched.
# It handles race prompting, HTML table fallback extraction, and potential re-routing.
def extract_contest_panel(page, contest_title):
    """
    Returns the Playwright element for the contest panel matching contest_title.
    """
    header_selector = ", ".join(CONTEST_HEADER_SELECTORS)
    for tag in CONTEST_PANEL_TAGS:
        panels = page.query_selector_all(tag)
        for panel in panels:
            header = panel.query_selector(header_selector)
            if header and contest_title.lower() in header.inner_text().strip().lower():
                return panel
    return None

def extract_precinct_tables(panel):
    """
    Returns a list of (precinct_name, table_element) tuples from a contest panel.
    """
    selector = ', '.join(PRECINCT_ELEMENT_TAGS)
    elements = panel.query_selector_all(selector)
    precincts = []
    current_precinct = None
    for el in elements:
        tag = el.evaluate("e => e.tagName").strip().upper()
        if tag in [t.upper() for t in PRECINCT_ELEMENT_TAGS if t != "table"]:
            label = el.inner_text().strip()
            current_precinct = label
        elif tag == "TABLE" and current_precinct:
            precincts.append((current_precinct, el))
            current_precinct = None
    return precincts
# --- 1. Detect structured files and prompt user ---
def parse(page, html_context=None):
  
    found_files = detect_format_from_links(page)
    for fmt, url in found_files:
        filename = os.path.basename(url)
        rprint(f"[bold green]Discovered {fmt.upper()} file:[/bold green] {filename}")
        choice = input(f"[PROMPT] Download and parse {filename}? [y/n] (n): ").strip().lower()
        if choice == "y":
            handler = route_format_handler(fmt)
            if handler:
                file_path = download_confirmed_file(url, page.url)
                if not file_path:
                    rprint(f"[red]Failed to download {filename}. Continuing with HTML parsing.[/red]")
                    continue
                dummy_page = None  # Structured handlers don't need a Playwright page
                result = handler.parse(dummy_page, {"filename": file_path})
                # Defensive: always expect a 4-tuple
                if not isinstance(result, tuple) or len(result) != 4:
                    rprint(f"[red]Handler for {fmt} did not return expected structure. Skipping.[/red]")
                    continue
                return result
            else:
                rprint(f"[red]No handler for {fmt}[/red]")
                continue
        else:
            rprint(f"[yellow]Skipping {filename}, continuing with HTML parsing.[/yellow]")
            # Continue to HTML parsing below


    # STEP 2 — Infer state dynamically if not present
    if not html_context or 'state' not in html_context or html_context['state'] == 'Unknown':
        logger.info("[INFO] Inferring state from URL or page text...")
    if html_context is None:
        html_context = {}
    contest_title = html_context.get("selected_race")
    if not contest_title and "available_races" in html_context:
        detected = get_detected_races_from_context(html_context) 
        contest_title = select_contest(detected)
        if not contest_title:
            return None, None, None, {"skipped": True}
        html_context["selected_race"] = contest_title
    # STEP 3 — Redirect to state/county-specific handler if matched
    # Check if the state handler is already resolved
    state_handler = get_state_handler(
        state_abbreviation=html_context.get("state"),
        county_name=html_context.get("county")
    )
    if state_handler and hasattr(state_handler, "parse"):
        logger.info(f"[HTML Handler] Redirecting to state handler: {state_handler.__name__}...")
        return state_handler.parse(page, html_context)    

    state = html_context.get("state")
    county = html_context.get("county")
    if state or county:
        state_handler = get_state_handler(state_abbreviation=state, county_name=county)
        if state_handler and hasattr(state_handler, "parse"):
            logger.info(f"[HTML Handler] Redirecting to state handler: {state_handler.__name__}...")
            return state_handler.parse(page, html_context)
    # If state and county are not set, check if the context has them
    if not state and not county:
        # Attempt to resolve state and county from the page text
        # This is a fallback if the URL doesn't provide enough context
        # Note: This is a basic heuristic and may not be accurate
        # In a real-world scenario, you might want to use a more sophisticated method
        # to extract the state and county from the text
        # For example, using regex to find state names or abbreviations
        # or using a library that can parse and understand the text better
        # This is a placeholder for the actual logic
        # that would be used to extract the state and county from the text
        # For now, we'll just log the raw text for debugging
        logger.debug(f"[DEBUG] Raw text for state/county inference: {html_context['raw_text']}")    
    if html_context.get("state") and html_context.get("county"):
        # If state and county are already set, use them directly
        logger.debug(f"[DEBUG] State and county already set in context: {html_context['state']}, {html_context['county']}")
    logger.debug(f"[DEBUG] Attempting to route using get_handler(state='{html_context.get('state')}', county='{html_context.get('county')}')")
    state_handler = get_state_handler(
        state_abbreviation=html_context.get("state"),
        county_name=html_context.get("county")
    )
    if state_handler:
        # Check if the handler has a parse method
        if hasattr(state_handler, "parse"):
            logger.info(f"[HTML Handler] Redirecting to state handler: {state_handler.__name__}...")
            return state_handler.parse(page, html_context)
        else:
            logger.warning(f"[WARN] State handler {state_handler.__name__} does not have a parse method.")
    elif html_context.get("state") and not html_context.get("county"):
        # If only the state is set, use the state handler directly
        logger.info(f"[HTML Handler] Redirecting to state handler: {html_context['state']}...")
        state_handler = get_state_handler(state_abbreviation=html_context["state"])
        logger.info(f"[HTML Handler] Redirecting to handler for {html_context.get('state')} / {html_context.get('county') or 'state-default'}...")
        return state_handler.parse(page, html_context)

    # STEP 4 — Final fallback: Basic HTML table extraction if no route matched
    headers = []
    try:
        table = page.query_selector("table")
        if not table:
            # Attempt to find a table with a specific class or ID
            table = page.query_selector("table#resultsTable, table.results-table")
        if not table:
            raise RuntimeError("No table found on the page.")
        # Extract
        headers, data = extract_table_data(table)
        logger.info(f"[HTML Handler] Extracted {len(data)} rows from the table.")
    except Exception as e:    
        logger.error(f"[ERROR] Failed to extract table from page: {e}")
        return None, None, None, {"error": str(e)}
        # Extract rows
        # Use a list comprehension to extract data from each row

    # STEP 5 — Output metadata for this HTML-based parse session
    # --- 5. Output metadata ---
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

# End of file
