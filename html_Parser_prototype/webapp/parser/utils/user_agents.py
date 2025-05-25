# filepath: c:\Users\olivi\OneDrive\Desktop\html_Parser_prototype\webapp\parser\html_election_parser.py

from playwright.sync_api import sync_playwright
from .utils.shared_logger import logger
from .utils.download_utils import ensure_input_directory, ensure_output_directory
from .utils.html_scanner import scan_html_for_context, get_detected_races_from_context
from .utils.contest_selector import select_contest
from .utils.table_builder import extract_table_data
from .utils.output_utils import finalize_election_output
from .utils.format_router import detect_format_from_links, route_format_handler
from .utils.user_prompt import prompt_user_input
from .state_router import get_handler_from_context

def initialize_directories():
    """Ensure necessary directories exist for input and output."""
    ensure_input_directory()
    ensure_output_directory()

def parse_html_page(page):
    """Main function to parse the HTML page for election data."""
    logger.info("[INFO] Starting HTML parsing...")

    # 1. Scan the page for context and races
    context = scan_html_for_context(page)
    detected_races = get_detected_races_from_context(context)
    
    # 2. Update context with detected races
    context['available_races'] = detected_races

    # 3. Let user select contest if not already set
    contest_title = context.get("selected_race")
    if not contest_title:
        contest_title = select_contest(detected_races)
        context["selected_race"] = contest_title

    # 4. Try to delegate to state/county handler if available
    state_handler = get_handler_from_context(
        state=context.get("state"),
        county=context.get("county")
    )
    if state_handler and hasattr(state_handler, "parse"):
        logger.info(f"[INFO] Delegating to state handler for {context['state']}, {context['county']}.")
        state_handler.parse(page, context)
        return

    # 5. Extract tables and data
    tables = extract_precinct_tables(page)
    if not tables:
        logger.warning("[WARN] No precinct tables found.")
        return

    # 6. Process each table and finalize output
    for precinct_name, table_element in tables:
        headers, data = extract_table_data(table_element)
        if headers and data:
            metadata = {
                "state": context.get("state"),
                "county": context.get("county"),
                "contest_title": contest_title
            }
            finalize_election_output(headers, data, contest_title, metadata)
        else:
            logger.warning(f"[WARN] No data extracted for precinct: {precinct_name}")

def main():
    """Main entry point for the HTML election parser."""
    initialize_directories()
    
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False)
        page = browser.new_page()
        
        # Navigate to the election results page (URL should be provided)
        url = "http://example.com/election-results"  # Replace with actual URL
        page.goto(url)

        # Parse the HTML page
        parse_html_page(page)

        # Close the browser
        browser.close()

if __name__ == "__main__":
    main()