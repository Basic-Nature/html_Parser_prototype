### Refactored `html_election_parser.py`

```python
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
    logger.info("[INFO] Starting HTML parsing...")

    # 1. Scan the page for context and races
    context = scan_html_for_context(page)
    logger.debug(f"[DEBUG] Context scanned: {context}")

    # 2. Detect available races
    detected_races = get_detected_races_from_context(context)
    logger.info(f"[INFO] Detected races: {detected_races}")

    # 3. Prompt user to select contests if necessary
    selected_contests = select_contest(detected_races)
    if not selected_contests:
        logger.warning("[WARN] No contests selected. Exiting.")
        return

    # 4. Extract tables and data for each selected contest
    for contest in selected_contests:
        logger.info(f"[INFO] Processing contest: {contest}")
        contest_panel = extract_contest_panel(page, contest)
        precinct_tables = extract_precinct_tables(contest_panel)

        for precinct_name, table_element in precinct_tables:
            headers, data = extract_table_data(table_element)
            logger.debug(f"[DEBUG] Extracted data for {precinct_name}: {data}")

            # 5. Finalize output for the extracted data
            metadata = {
                "state": context.get("state"),
                "county": context.get("county"),
                "contest_title": contest,
            }
            finalize_election_output(headers, data, contest, metadata)

def main():
    """Main entry point for the HTML election parser."""
    setup_directories()

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False)
        page = browser.new_page()

        # Navigate to the election results page
        url = input("[PROMPT] Enter the URL of the election results page: ")
        page.goto(url)

        # Parse the HTML page
        parse_html_page(page)

        # Close the browser
        browser.close()
        logger.info("[INFO] Parsing completed.")

if __name__ == "__main__":
    main()
```

### Explanation of the Refactored Code

1. **Imports**: The necessary modules and functions are imported at the top for clarity.

2. **Setup Directories**: The `setup_directories` function ensures that the input and output directories exist before any parsing begins.

3. **Main Parsing Function**: The `parse_html_page` function orchestrates the parsing process:
   - It scans the HTML page for context and races.
   - It detects available races and prompts the user to select contests.
   - For each selected contest, it extracts tables and data, and finalizes the output.

4. **Main Entry Point**: The `main` function serves as the entry point for the script, handling browser setup and navigation to the specified URL.

5. **Modularity**: Each task is encapsulated in its own function, promoting clarity and modularity. This structure allows for easier testing and maintenance.

6. **Logging**: The use of logging at various levels (info, debug, warning) provides insight into the parsing process and helps with debugging.

This refactored structure should effectively orchestrate the entire election data parsing pipeline while maintaining clarity and modularity.