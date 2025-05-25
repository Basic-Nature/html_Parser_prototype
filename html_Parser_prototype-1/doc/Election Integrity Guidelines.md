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
from .state_router import get_handler_from_context

def initialize_playwright():
    """Initialize Playwright and return the browser instance."""
    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(headless=False)  # Set headless=True for production
    return playwright, browser

def close_playwright(playwright, browser):
    """Close the Playwright browser and instance."""
    browser.close()
    playwright.stop()

def parse_election_data(page, html_context):
    """Main function to parse election data from the HTML page."""
    # 1. Scan the page for context and races
    context = scan_html_for_context(page)
    html_context.update(context)

    # 2. Try to delegate to state/county handler if available
    state_handler = get_handler_from_context(
        state=html_context.get("state"),
        county=html_context.get("county")
    )
    if state_handler and hasattr(state_handler, "parse"):
        logger.info("[INFO] Delegating to state handler for parsing.")
        state_handler.parse(page, html_context)
        return

    # 3. Let user select contest if not already set
    contest_title = html_context.get("selected_race")
    if not contest_title:
        contest_title = select_contest(get_detected_races_from_context(context))
        html_context["selected_race"] = contest_title

    # 4. Extract tables and data
    precinct_tables = extract_precinct_tables(page)
    all_data = []
    for precinct_name, table_element in precinct_tables:
        headers, data = extract_table_data(table_element)
        all_data.extend(data)

    # 5. Finalize output
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
    """Main entry point for the election data parsing pipeline."""
    playwright, browser = initialize_playwright()
    try:
        page = browser.new_page()
        url = "http://example.com/election-results"  # Replace with the actual URL
        page.goto(url)

        html_context = {}
        parse_election_data(page, html_context)

    except Exception as e:
        logger.error(f"[ERROR] An error occurred during parsing: {e}")
    finally:
        close_playwright(playwright, browser)

if __name__ == "__main__":
    main()
```

### Explanation of the Refactor:

1. **Initialization and Cleanup**: The `initialize_playwright` and `close_playwright` functions handle the setup and teardown of the Playwright browser instance, ensuring that resources are managed properly.

2. **Main Parsing Logic**: The `parse_election_data` function orchestrates the parsing process:
   - It scans the HTML for context and races.
   - It delegates to a state handler if available.
   - It allows the user to select a contest if not already set.
   - It extracts tables and compiles the data.
   - Finally, it finalizes the output by writing the data to files.

3. **Modularity**: Each part of the process is encapsulated in functions, making the code easier to read, maintain, and test.

4. **Error Handling**: The main function includes error handling to catch and log any issues that arise during the parsing process.

5. **User Interaction**: The code allows for user interaction when selecting contests, ensuring that the pipeline is flexible and can adapt to different scenarios.

This structure provides a clear and modular approach to orchestrating the election data parsing pipeline, integrating all components effectively.