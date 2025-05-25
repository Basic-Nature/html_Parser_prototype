# filepath: c:\Users\olivi\OneDrive\Desktop\html_Parser_prototype\webapp\parser\html_election_parser.py

from playwright.sync_api import sync_playwright
from .utils.shared_logger import logger
from .utils.download_utils import ensure_input_directory, ensure_output_directory
from .utils.format_router import detect_format_from_links, route_format_handler
from .utils.html_scanner import scan_html_for_context, get_detected_races_from_context
from .utils.output_utils import finalize_election_output
from .utils.user_prompt import prompt_user_input
from .utils.contest_selector import select_contest
from .utils.table_builder import extract_table_data
from .utils.shared_logic import normalize_text
import os

def setup_directories():
    """Ensure necessary directories exist for input and output."""
    ensure_input_directory()
    ensure_output_directory()

def parse_html_page(page):
    """Main function to parse the HTML page for election data."""
    logger.info("[INFO] Starting HTML page parsing...")

    # 1. Scan the page for context and races
    context = scan_html_for_context(page)
    logger.debug(f"[DEBUG] Context scanned: {context}")

    # 2. Detect available races
    detected_races = get_detected_races_from_context(context)
    logger.info(f"[INFO] Detected races: {detected_races}")

    # 3. Prompt user to select a contest if not already set
    contest_title = context.get("selected_race")
    if not contest_title:
        contest_title = select_contest(detected_races)
        if not contest_title:
            logger.warning("[WARN] No contest selected. Exiting.")
            return

    # 4. Extract contest panel and precinct tables
    contest_panel = extract_contest_panel(page, contest_title)
    precinct_tables = extract_precinct_tables(contest_panel)

    # 5. Parse each precinct table and collect data
    all_data = []
    for precinct_name, table_element in precinct_tables:
        headers, data = extract_table_data(table_element)
        all_data.extend(data)

    # 6. Finalize output
    if all_data:
        metadata = {
            "state": context.get("state"),
            "county": context.get("county"),
            "contest_title": contest_title
        }
        headers = headers  # Assuming headers are consistent across tables
        output_paths = finalize_election_output(headers, all_data, contest_title, metadata)
        logger.info(f"[INFO] Output files created: {output_paths}")
    else:
        logger.warning("[WARN] No data extracted from precinct tables.")

def main():
    """Main entry point for the HTML election parser."""
    setup_directories()

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