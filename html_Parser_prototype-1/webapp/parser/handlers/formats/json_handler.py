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

def setup_directories():
    """Ensure necessary directories exist for input and output."""
    ensure_input_directory()
    ensure_output_directory()

def parse_html_page(page):
    """Main function to parse the HTML page and extract election data."""
    logger.info("[INFO] Starting HTML parsing...")

    # 1. Scan the page for context and races
    context = scan_html_for_context(page)
    logger.debug(f"[DEBUG] Context scanned: {context}")

    # 2. Detect available formats for download
    available_formats = detect_format_from_links(page)
    logger.info(f"[INFO] Available formats detected: {available_formats}")

    # 3. Prompt user to select a format if multiple formats are available
    if len(available_formats) > 1:
        selected_format = prompt_user_input(
            "Multiple formats detected. Please select a format:",
            default=available_formats[0][0]
        )
    else:
        selected_format = available_formats[0][0] if available_formats else None

    # 4. Route to the appropriate handler based on the selected format
    if selected_format:
        handler = route_format_handler(selected_format)
        if handler:
            logger.info(f"[INFO] Using handler for format: {selected_format}")
            handler.parse(page, context)
        else:
            logger.error(f"[ERROR] No handler found for format: {selected_format}")
            return

    # 5. Extract contest information
    contest_title = context.get("selected_race")
    if not contest_title:
        contest_title = select_contest(context.get("available_races"))
    
    # 6. Extract tables and finalize output
    precinct_tables = extract_table_data(page)
    if precinct_tables:
        headers, data = precinct_tables
        metadata = {
            "state": context.get("state"),
            "county": context.get("county"),
            "contest_title": contest_title
        }
        output_paths = finalize_election_output(headers, data, contest_title, metadata)
        logger.info(f"[INFO] Output files created: {output_paths}")
    else:
        logger.warning("[WARN] No precinct tables found.")

def main():
    """Main entry point for the HTML election parser."""
    setup_directories()
    
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False)
        page = browser.new_page()
        
        # Navigate to the election results page (URL should be provided)
        url = "https://example.com/election-results"  # Replace with actual URL
        page.goto(url)
        
        parse_html_page(page)
        
        # Close the browser after parsing
        page.close()
        browser.close()

if __name__ == "__main__":
    main()