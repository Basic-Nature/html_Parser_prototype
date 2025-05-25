# filepath: c:\Users\olivi\OneDrive\Desktop\html_Parser_prototype\webapp\parser\html_election_parser.py

from playwright.sync_api import sync_playwright
from .utils.shared_logger import logger
from .utils.html_scanner import scan_html_for_context, get_detected_races_from_context
from .utils.contest_selector import select_contest
from .utils.output_utils import finalize_election_output
from .utils.download_utils import ensure_input_directory
from .utils.format_router import detect_format_from_links, route_format_handler
from .utils.shared_logic import normalize_text
from .handlers.formats.html_handler import parse as parse_html
from .utils.user_prompt import prompt_user_input

def initialize_playwright():
    """Initialize Playwright and return the browser instance."""
    playwright = sync_playwright().start()
    return playwright

def launch_browser(playwright):
    """Launch the browser and return the page object."""
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    return page

def close_browser(browser, context):
    """Close the browser and context."""
    context.close()
    browser.close()

def parse_election_data(page):
    """Main function to parse election data from the webpage."""
    logger.info("[INFO] Starting election data parsing...")

    # 1. Navigate to the target URL
    target_url = "https://example.com/election-results"  # Replace with the actual URL
    page.goto(target_url)

    # 2. Scan the page for context and races
    context = scan_html_for_context(page)
    detected_races = get_detected_races_from_context(context)

    # 3. Allow user to select contests
    contest_title = select_contest(detected_races)
    if not contest_title:
        logger.warning("[WARN] No contest selected. Exiting.")
        return

    # 4. Extract and parse the relevant HTML data
    html_context = {"selected_race": contest_title}
    parse_html(page, html_context)

    # 5. Detect available formats for download
    formats = detect_format_from_links(page)
    if not formats:
        logger.warning("[WARN] No downloadable formats found.")
        return

    # 6. Prompt user for format selection
    selected_format = prompt_user_input("Select a format to download:", default=formats[0][0])
    handler = route_format_handler(selected_format)
    if handler:
        # Process the selected format
        logger.info(f"[INFO] Processing format: {selected_format}")
        # Call the appropriate handler's parse function here
        # handler.parse()  # Uncomment and implement as needed

    # 7. Finalize output
    metadata = {"state": context.get("state"), "county": context.get("county")}
    finalize_election_output(headers=[], data=[], contest_title=contest_title, metadata=metadata)

    logger.info("[INFO] Election data parsing completed.")

def main():
    """Main entry point for the election parser."""
    ensure_input_directory()  # Ensure input directory exists
    with initialize_playwright() as playwright:
        page = launch_browser(playwright)
        try:
            parse_election_data(page)
        finally:
            close_browser(page.browser, page.context)

if __name__ == "__main__":
    main()