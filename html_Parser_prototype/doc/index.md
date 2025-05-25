### Refactored `html_election_parser.py`

```python
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
    logger.info("[INFO] Starting HTML parsing process...")

    # 1. Scan the page for context and races
    context = scan_html_for_context(page)
    logger.debug(f"[DEBUG] Scanned context: {context}")

    # 2. Update the context with detected races
    detected_races = get_detected_races_from_context(context)
    logger.info(f"[INFO] Detected races: {detected_races}")

    # 3. Let user select contest if not already set
    contest_title = context.get("selected_race")
    if not contest_title:
        contest_title = select_contest(detected_races)
        if not contest_title:
            logger.warning("[WARN] No contest selected. Exiting parsing.")
            return

    # 4. Try to delegate to state/county handler if available
    state_handler = get_handler_from_context(context.get("state"), context.get("county"))
    if state_handler and hasattr(state_handler, "parse"):
        logger.info(f"[INFO] Delegating to state handler for {contest_title}.")
        state_handler.parse(page, context)
        return

    # 5. Extract tables and finalize output
    headers, data = extract_table_data(page)
    if not headers or not data:
        logger.warning("[WARN] No data extracted from tables.")
        return

    # 6. Finalize output
    metadata = {
        "state": context.get("state"),
        "county": context.get("county"),
        "contest_title": contest_title,
        "source_url": context.get("source_url"),
    }
    output_paths = finalize_election_output(headers, data, contest_title, metadata)
    logger.info(f"[INFO] Output files created: {output_paths}")

def main():
    """Main entry point for the HTML election parser."""
    setup_directories()

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False)
        page = browser.new_page()
        
        # Navigate to the election results page (replace with actual URL)
        url = "https://example.com/election-results"
        page.goto(url)

        parse_html_page(page)

        # Close the browser after parsing
        page.close()
        browser.close()

if __name__ == "__main__":
    main()
```

### Explanation of the Refactored Code

1. **Setup Directories**: The `setup_directories` function ensures that the necessary input and output directories exist before starting the parsing process.

2. **Main Parsing Function**: The `parse_html_page` function orchestrates the parsing of the HTML page:
   - It scans the page for context and races.
   - It allows the user to select a contest if not already set.
   - It delegates to a state/county handler if available.
   - It extracts table data and finalizes the output.

3. **Main Entry Point**: The `main` function serves as the entry point for the script, initializing the Playwright browser, navigating to the election results page, and calling the parsing function.

4. **Modularity and Clarity**: The code is structured into clear, modular functions, making it easier to maintain and extend in the future. Each function has a single responsibility, which enhances readability and testability.

This refactored structure should effectively orchestrate the entire election data parsing pipeline while maintaining clarity and modularity.