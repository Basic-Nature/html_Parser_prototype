# filepath: c:\Users\olivi\OneDrive\Desktop\html_Parser_prototype\webapp\parser\html_election_parser.py

from playwright.sync_api import sync_playwright
from .utils.shared_logger import logger
from .utils.download_utils import ensure_input_directory, ensure_output_directory
from .utils.format_router import detect_format_from_links, route_format_handler
from .utils.html_scanner import scan_html_for_context, get_detected_races_from_context
from .utils.output_utils import finalize_election_output
from .utils.user_prompt import prompt_user_input
from .utils.table_builder import extract_table_data
from .utils.contest_selector import select_contest
from .state_router import get_handler_from_context

def initialize_directories():
    """Ensure necessary directories for input and output exist."""
    ensure_input_directory()
    ensure_output_directory()

def parse_html_page(page):
    """Main function to parse the HTML page for election data."""
    logger.info("[INFO] Starting HTML page parsing...")

    # 1. Scan the page for context and races
    context = scan_html_for_context(page)
    logger.debug(f"[DEBUG] Context scanned: {context}")

    # 2. Detect available formats and prompt user if necessary
    detected_formats = detect_format_from_links(page)
    if not detected_formats:
        logger.warning("[WARN] No downloadable formats found.")
        return

    # 3. Route to appropriate format handler
    for format_str, url in detected_formats:
        handler = route_format_handler(format_str)
        if handler:
            logger.info(f"[INFO] Handling format: {format_str}")
            handler.parse(page, context)
        else:
            logger.warning(f"[WARN] No handler found for format: {format_str}")

    # 4. Extract races and allow user to select contests
    detected_races = get_detected_races_from_context(context)
    selected_contests = select_contest(detected_races)
    if not selected_contests:
        logger.info("[INFO] No contests selected. Exiting.")
        return

    # 5. Process each selected contest
    for contest in selected_contests:
        logger.info(f"[INFO] Processing contest: {contest}")
        contest_panel = extract_contest_panel(page, contest)
        precinct_tables = extract_precinct_tables(contest_panel)

        # 6. Extract data from precinct tables
        for precinct_name, table_element in precinct_tables:
            headers, data = extract_table_data(table_element)
            logger.debug(f"[DEBUG] Extracted data for {precinct_name}: {data}")

            # 7. Finalize output for the election data
            metadata = {
                "state": context.get("state"),
                "county": context.get("county"),
                "contest_title": contest,
            }
            finalize_election_output(headers, data, contest, metadata)

def main():
    """Main entry point for the HTML election parser."""
    initialize_directories()
    
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False)
        page = browser.new_page()
        
        # Navigate to the election results page
        url = "https://example.com/election-results"  # Replace with actual URL
        page.goto(url)

        # Parse the HTML page
        parse_html_page(page)

        # Close the browser
        page.close()
        browser.close()

if __name__ == "__main__":
    main()