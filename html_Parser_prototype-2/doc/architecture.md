### Refactored `html_election_parser.py`

```python
# filepath: c:\Users\olivi\OneDrive\Desktop\html_Parser_prototype\webapp\parser\html_election_parser.py

from playwright.sync_api import sync_playwright
from .utils.shared_logger import logger
from .utils.html_scanner import scan_html_for_context, get_detected_races_from_context
from .utils.contest_selector import select_contest
from .utils.table_builder import extract_table_data
from .utils.output_utils import finalize_election_output
from .utils.format_router import detect_format_from_links, route_format_handler
from .utils.download_utils import download_confirmed_file
from .utils.shared_logic import normalize_text
from .state_router import get_handler_from_context

def initialize_playwright():
    """Initialize Playwright and return the browser instance."""
    playwright = sync_playwright().start()
    return playwright

def parse_html_page(page, html_context):
    """Main parsing function for the HTML page."""
    logger.info("[INFO] Starting HTML parsing...")

    # 1. Scan the page for context and races
    context = scan_html_for_context(page)
    html_context.update(context)

    # 2. Delegate to state/county handler if available
    state_handler = get_handler_from_context(
        state=html_context.get("state"),
        county=html_context.get("county")
    )
    if state_handler and hasattr(state_handler, "parse"):
        logger.info("[INFO] Delegating to state handler...")
        state_handler.parse(page, html_context)
        return

    # 3. Let user select contest if not already set
    contest_title = html_context.get("selected_race")
    if not contest_title:
        contest_title = select_contest(get_detected_races_from_context(context))
        html_context["selected_race"] = contest_title

    # 4. Extract tables and data
    precinct_tables = extract_precinct_tables(page)
    for precinct_name, table_element in precinct_tables:
        headers, data = extract_table_data(table_element)
        # Process and finalize the election output
        finalize_election_output(headers, data, contest_title, html_context)

def main():
    """Main function to orchestrate the election data parsing pipeline."""
    with initialize_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False)
        page = browser.new_page()

        # Navigate to the election results page
        url = "https://example.com/election-results"  # Replace with actual URL
        page.goto(url)

        # Initialize context for HTML parsing
        html_context = {}

        # Parse the HTML page
        parse_html_page(page, html_context)

        # Clean up
        page.close()
        browser.close()
        logger.info("[INFO] Parsing completed.")

if __name__ == "__main__":
    main()
```

### Explanation of the Refactored Code

1. **Modular Functions**: The code is organized into functions that handle specific tasks, such as initializing Playwright, parsing the HTML page, and orchestrating the main flow. This modularity enhances readability and maintainability.

2. **Context Management**: The `html_context` dictionary is used to store relevant information throughout the parsing process, making it easy to pass data between functions.

3. **Dynamic Handling**: The code checks for available state handlers and allows the user to select contests dynamically, ensuring that the parsing adapts to the content of the page.

4. **Logging**: The use of logging provides insights into the parsing process, making it easier to debug and understand the flow of execution.

5. **Finalization**: The `finalize_election_output` function is called to handle the output of the parsed data, ensuring that results are saved in the desired format.

This refactored structure provides a clear and effective orchestration of the election data parsing pipeline while maintaining clarity and modularity in the code.