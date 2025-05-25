# filepath: c:\Users\olivi\OneDrive\Desktop\html_Parser_prototype\webapp\parser\html_election_parser.py

from playwright.sync_api import Page
from .state_router import get_handler_from_context
from .utils.html_scanner import scan_html_for_context, get_detected_races_from_context
from .utils.contest_selector import select_contest
from .utils.table_builder import extract_table_data
from .utils.shared_logger import logger
from .utils.output_utils import finalize_election_output

def orchestrate_election_parsing(page: Page, html_context=None):
    """
    Orchestrates the entire election data parsing pipeline.
    
    Args:
        page (Page): The Playwright page object containing the election data.
        html_context (dict, optional): Context for the HTML parsing.
    """
    # Step 1: Scan the page for context and races
    context = scan_html_for_context(page)
    if html_context is None:
        html_context = context
    else:
        html_context.update(context)

    # Step 2: Delegate to state/county handler if available
    state_handler = get_handler_from_context(
        state=html_context.get("state"),
        county=html_context.get("county")
    )
    if state_handler and hasattr(state_handler, "parse"):
        logger.info("[INFO] Delegating to state handler for parsing.")
        state_handler.parse(page, html_context)
        return

    # Step 3: Let user select contest if not already set
    contest_title = html_context.get("selected_race")
    if not contest_title:
        logger.info("[INFO] No contest selected. Prompting user for selection.")
        detected_races = get_detected_races_from_context(context)
        selected_contests = select_contest(detected_races)
        if not selected_contests:
            logger.warning("[WARN] No contests selected. Exiting parsing.")
            return
        contest_title = selected_contests[0]  # Assuming single selection for simplicity

    # Step 4: Extract contest panel and precinct tables
    logger.info(f"[INFO] Extracting data for contest: {contest_title}")
    contest_panel = extract_contest_panel(page, contest_title)
    precinct_tables = extract_precinct_tables(contest_panel)

    # Step 5: Parse each precinct table and collect results
    all_data = []
    headers = []
    for precinct_name, table_element in precinct_tables:
        logger.info(f"[INFO] Parsing precinct table for: {precinct_name}")
        headers, data = extract_table_data(table_element)
        all_data.extend(data)

    # Step 6: Finalize and output the election results
    if all_data:
        logger.info("[INFO] Finalizing election output.")
        metadata = {
            "state": html_context.get("state"),
            "county": html_context.get("county"),
            "contest_title": contest_title
        }
        finalize_election_output(headers, all_data, contest_title, metadata)
    else:
        logger.warning("[WARN] No data extracted from precinct tables.")

def extract_contest_panel(page, contest_title):
    """
    Extracts the contest panel from the page based on the contest title.
    
    Args:
        page (Page): The Playwright page object.
        contest_title (str): The title of the contest to extract.
    
    Returns:
        Locator: The Playwright Locator for the contest panel.
    """
    # Implementation of extracting contest panel goes here
    pass

def extract_precinct_tables(panel):
    """
    Extracts precinct tables from the contest panel.
    
    Args:
        panel: The contest panel Locator.
    
    Returns:
        List[Tuple[str, Locator]]: A list of tuples containing precinct names and their corresponding table elements.
    """
    # Implementation of extracting precinct tables goes here
    pass
```

### Key Changes and Structure:
1. **Orchestration Function**: The `orchestrate_election_parsing` function serves as the main entry point for the parsing pipeline, coordinating the various steps involved in parsing election data.

2. **Modularity**: Each step of the process is clearly defined, with dedicated functions for extracting contest panels and precinct tables. This modularity enhances readability and maintainability.

3. **Dynamic Handling**: The code dynamically handles the selection of contests and delegates parsing to state-specific handlers when available.

4. **Logging**: Comprehensive logging is included to track the progress and any issues encountered during the parsing process.

5. **Error Handling**: The code includes checks for user selections and data extraction, ensuring that the process can gracefully handle cases where no data is available or no contests are selected.

This refactored structure provides a clear and organized approach to orchestrating the election data parsing pipeline, making it easier to understand and extend in the future.