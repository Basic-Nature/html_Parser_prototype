# filepath: c:\Users\olivi\OneDrive\Desktop\html_Parser_prototype\webapp\parser\html_election_parser.py

from playwright.sync_api import sync_playwright
from .utils.shared_logger import logger
from .utils.download_utils import ensure_input_directory, ensure_output_directory
from .utils.html_scanner import scan_html_for_context, get_detected_races_from_context
from .utils.contest_selector import select_contest
from .utils.table_builder import extract_table_data
from .utils.output_utils import finalize_election_output
from .state_router import get_handler_from_context
import os

def setup_directories():
    """Ensure necessary directories exist for input and output."""
    ensure_input_directory()
    ensure_output_directory()

def parse_html_page(page):
    """Main function to parse the HTML page for election data."""
    logger.info("[INFO] Starting HTML parsing...")
    
    # 1. Scan the page for context and races
    context = scan_html_for_context(page)
    logger.debug(f"[DEBUG] Context scanned: {context}")

    # 2. Update HTML context with scanned data
    html_context = context.get("html_context", {})
    
    # 3. Try to delegate to state/county handler if available
    state_handler = get_handler_from_context(
        state=html_context.get("state"),
        county=html_context.get("county")
    )
    if state_handler and hasattr(state_handler, "parse"):
        logger.info("[INFO] Delegating to state handler for parsing...")
        state_handler.parse(page, html_context)
        return

    # 4. Let user select contest if not already set
    contest_title = html_context.get("selected_race")
    if not contest_title:
        detected_races = get_detected_races_from_context(context)
        contest_title = select_contest(detected_races)
        if not contest_title:
            logger.warning("[WARN] No contest selected. Skipping parsing.")
            return

    # 5. Extract tables and data
    precinct_tables = extract_precinct_tables(page)
    if not precinct_tables:
        logger.warning("[WARN] No precinct tables found. Skipping parsing.")
        return

    # 6. Process each precinct table
    all_data = []
    for precinct_name, table_element in precinct_tables:
        headers, data = extract_table_data(table_element)
        all_data.extend(data)

    # 7. Finalize output
    if all_data:
        metadata = {
            "state": html_context.get("state"),
            "county": html_context.get("county"),
            "contest_title": contest_title
        }
        headers = headers  # Assuming headers are consistent across tables
        finalize_election_output(headers, all_data, contest_title, metadata)
    else:
        logger.warning("[WARN] No data extracted from tables.")

def main():
    """Main entry point for the HTML election parser."""
    setup_directories()
    
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False)
        page = browser.new_page()
        
        # Navigate to the election results page
        url = "https://example.com/election-results"  # Replace with actual URL
        page.goto(url)
        
        # Parse the HTML page
        parse_html_page(page)
        
        # Close the browser
        browser.close()
        logger.info("[INFO] Parsing completed.")

if __name__ == "__main__":
    main()