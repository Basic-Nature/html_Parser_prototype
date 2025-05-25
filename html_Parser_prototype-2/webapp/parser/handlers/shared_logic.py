# filepath: c:\Users\olivi\OneDrive\Desktop\html_Parser_prototype\webapp\parser\html_election_parser.py

from playwright.sync_api import sync_playwright
from .utils.shared_logger import logger
from .utils.download_utils import ensure_input_directory
from .utils.format_router import detect_format_from_links, route_format_handler
from .utils.html_scanner import scan_html_for_context, get_detected_races_from_context
from .utils.output_utils import finalize_election_output
from .utils.user_prompt import prompt_user_input
from .handlers.formats.html_handler import parse as parse_html
from .utils.contest_selector import select_contest

def setup_playwright():
    """Initialize Playwright and return the browser instance."""
    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(headless=False)  # Set headless=True for production
    return playwright, browser

def close_playwright(playwright, browser):
    """Close the Playwright browser and instance."""
    browser.close()
    playwright.stop()

def orchestrate_parsing(page_url):
    """Main orchestration function for the election data parsing pipeline."""
    playwright, browser = setup_playwright()
    page = browser.new_page()
    
    try:
        logger.info(f"[INFO] Navigating to {page_url}")
        page.goto(page_url)

        # Step 1: Ensure input directory exists
        ensure_input_directory()

        # Step 2: Scan the page for context and races
        context = scan_html_for_context(page)
        logger.debug(f"[DEBUG] Scanned context: {context}")

        # Step 3: Detect available formats for download
        detected_formats = detect_format_from_links(page)
        logger.debug(f"[DEBUG] Detected formats: {detected_formats}")

        # Step 4: Prompt user for format selection
        selected_format = prompt_user_for_format(detected_formats)
        if not selected_format:
            logger.warning("[WARN] No format selected. Exiting.")
            return

        # Step 5: Route to the appropriate format handler
        format_handler = route_format_handler(selected_format)
        if format_handler is None:
            logger.error(f"[ERROR] No handler found for format: {selected_format}")
            return

        # Step 6: Parse the HTML content
        html_context = {}
        parse_html(page, html_context)

        # Step 7: Finalize output
        metadata = {
            "state": html_context.get("state"),
            "county": html_context.get("county"),
            "election_type": html_context.get("election_type"),
            "source_url": page_url
        }
        headers, data = extract_table_data(page)  # Assuming this function is defined elsewhere
        output_paths = finalize_election_output(headers, data, "Election Results", metadata)
        logger.info(f"[INFO] Output files created: {output_paths}")

    except Exception as e:
        logger.error(f"[ERROR] An error occurred during parsing: {e}")
    finally:
        close_playwright(playwright, browser)

def prompt_user_for_format(detected_formats):
    """Prompt the user to select a format from the detected formats."""
    if not detected_formats:
        logger.warning("[WARN] No formats detected.")
        return None

    format_options = [f"{fmt[0].upper()} ({fmt[1]})" for fmt in detected_formats]
    logger.info("[INFO] Available formats:")
    for i, opt in enumerate(format_options):
        logger.info(f"  [{i}] {opt}")

    selection = prompt_user_input(
        "[PROMPT] Select a format to parse (0-{len(format_options)-1}): ",
        default="0",
        validator=lambda x: x.isdigit() and 0 <= int(x) < len(format_options)
    )
    return detected_formats[int(selection)][0]  # Return the selected format

if __name__ == "__main__":
    page_url = input("[PROMPT] Enter the URL of the election results page: ").strip()
    orchestrate_parsing(page_url)