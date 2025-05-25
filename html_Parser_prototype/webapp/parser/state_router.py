# filepath: c:\Users\olivi\OneDrive\Desktop\html_Parser_prototype\webapp\parser\html_election_parser.py

from playwright.sync_api import sync_playwright
from .utils.shared_logger import logger
from .utils.html_scanner import scan_html_for_context, get_detected_races_from_context
from .utils.contest_selector import select_contest
from .utils.table_builder import extract_table_data
from .utils.output_utils import finalize_election_output
from .utils.format_router import detect_format_from_links, route_format_handler
from .state_router import get_handler_from_context
import os

def initialize_playwright():
    """Initialize Playwright and return the browser instance."""
    playwright = sync_playwright().start()
    return playwright

def parse_election_data(page, html_context):
    """Main function to parse election data from the given page."""
    # 1. Scan the page for context and races
    context = scan_html_for_context(page)
    html_context.update(context)

    # 2. Try to delegate to state/county handler if available
    state_handler = get_handler_from_context(
        state=html_context.get("state"),
        county=html_context.get("county")
    )
    if state_handler and hasattr(state_handler, "parse"):
        logger.info("[INFO] Delegating to state handler for parsing.")
        state_handler.parse(page, html_context)
        return

    # 3. Let user select contest if not already set
    contest_title = html_context.get("selected_race")
    if not contest_title:
        contest_title = select_contest(get_detected_races_from_context(context))
        html_context["selected_race"] = contest_title

    # 4. Extract tables and data
    precinct_tables = extract_precinct_tables(page)
    all_data = []
    headers = []

    for precinct_name, table_element in precinct_tables:
        logger.info(f"[INFO] Extracting data from precinct: {precinct_name}")
        table_headers, table_data = extract_table_data(table_element)
        headers = headers or table_headers  # Set headers from the first table
        all_data.extend(table_data)

    # 5. Finalize output
    if all_data:
        metadata = {
            "state": html_context.get("state"),
            "county": html_context.get("county"),
            "contest_title": contest_title
        }
        finalize_election_output(headers, all_data, contest_title, metadata)
    else:
        logger.warning("[WARN] No data extracted from precinct tables.")

def main():
    """Main entry point for the election data parsing pipeline."""
    with initialize_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False)
        page = browser.new_page()

        # Navigate to the election results page
        url = "https://example.com/election-results"  # Replace with the actual URL
        page.goto(url)

        # Initialize context for HTML parsing
        html_context = {}

        try:
            parse_election_data(page, html_context)
        except Exception as e:
            logger.error(f"[ERROR] An error occurred during parsing: {e}")
        finally:
            page.close()
            browser.close()

if __name__ == "__main__":
    main()