# handlers/formats/html_handler.py
# ==============================================================
# Fallback handler for generic HTML parsing.
# Used when structured formats (JSON, CSV, PDF) are not present.
# This routes through the state_router using html_scanner context.
# ==============================================================

from webapp.parser.state_router import get_handler as get_state_handler
from utils.contest_selector import select_contest
from utils.download_utils import download_confirmed_file
from utils.format_router import detect_format_from_links, route_format_handler
from utils.html_table_extractor import extract_table_data
from utils.html_scanner import scan_html_for_context, get_detected_races_from_context
from utils.shared_logger import logger
from utils.shared_logic import (
    normalize_text, click_dynamic_toggle,
    CONTEST_HEADER_SELECTORS, CONTEST_PANEL_TAGS,
    PRECINCT_ELEMENT_TAGS, CONTEST_PANEL_TAGS
)
from rich import print as rprint
import os
import re
# This fallback HTML handler is invoked when no state or county-specific handler is matched.
# It handles race prompting, HTML table fallback extraction, and potential re-routing.
def extract_contest_panel(page, contest_title, panel_tags=None):
    """
    Returns a Playwright Locator for the contest panel matching contest_title.
    Tries all tags in panel_tags (default: CONTEST_PANEL_TAGS).
    """
    header_selector = ", ".join(CONTEST_HEADER_SELECTORS)
    panel_tags = panel_tags or CONTEST_PANEL_TAGS
    for tag in panel_tags:
        # Use locator, not query_selector_all, for robust downstream use
        panels = page.locator(tag)
        for i in range(panels.count()):
            panel = panels.nth(i)
            header = panel.locator(header_selector)
            if header.count() > 0:
                header_text = header.first.inner_text().strip().lower()
                if contest_title.lower() in header_text:
                    return panel
    pass
def extract_precinct_tables(panel):
    """
    Returns a list of (precinct_name, table_element) tuples from a contest panel.
    Accepts a Playwright Locator as panel.
    """
    selector = ', '.join(PRECINCT_ELEMENT_TAGS)
    elements = panel.locator(selector)
    precincts = []
    current_precinct = None
    for i in range(elements.count()):
        el = elements.nth(i)
        tag = el.evaluate("e => e.tagName").strip().upper()
        if tag in [t.upper() for t in PRECINCT_ELEMENT_TAGS if t != "table"]:
            label = el.inner_text().strip()
            current_precinct = label
        elif tag == "TABLE" and current_precinct:
            precincts.append((current_precinct, el))
            current_precinct = None
    pass
# --- 1. Detect structured files and prompt user ---
def parse(page, html_context=None):
    # 1. Check for structured files first
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
                result = handler.parse(None, {"filename": file_path})
                if not isinstance(result, tuple) or len(result) != 4:
                    rprint(f"[red]Handler for {fmt} did not return expected structure. Skipping.[/red]")
                    continue
                return result
            else:
                rprint(f"[red]No handler for {fmt}[/red]")
                continue
        else:
            rprint(f"[yellow]Skipping {filename}, continuing with HTML parsing.[/yellow]")

    # 2. Scan the page for context and races
    context = scan_html_for_context(page, debug=False)
    if html_context is None:
        html_context = {}
    html_context.update(context)

    # 3. Let user select contest if not already set
    contest_title = html_context.get("selected_race")
    if not contest_title:
        detected = get_detected_races_from_context(html_context)
        contest_title = select_contest(detected)
        if not contest_title:
            return None, None, None, {"skipped": True}
        html_context["selected_race"] = contest_title

    # 4. Delegate to state/county handler if available
    state_handler = get_state_handler(
        state_abbreviation=html_context.get("state"),
        county_name=html_context.get("county")
    )
    if state_handler and hasattr(state_handler, "parse"):
        logger.info(f"[HTML Handler] Redirecting to state handler: {state_handler.__name__}...")
        return state_handler.parse(page, html_context)

    # 5. Try to find and click toggles dynamically using handler-supplied keywords
    # Example: handler_keywords could come from config, context, or be generic
    handler_keywords = html_context.get("toggle_keywords", ["View Results", "Vote Method", "Show Results"])
    clicked = click_dynamic_toggle(
        page,
        container=None,  # or a panel if you have one
        handler_keywords=handler_keywords,
        logger=logger,
        verbose=True,
        interactive=True  # Enable prompt fallback
    )
    if not clicked:
        logger.warning("[HTML Handler] No toggle/button clicked automatically or interactively.")

    # 6. Fallback: Try to extract the first table
    headers, data = [], []
    try:
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

    # 7. Output metadata
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
