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
        html_context (dict): Optional context for the HTML parsing.
    
    Returns:
        dict: Contains paths to the output files generated.
    """
    # Step 1: Scan the page for context and races
    context = scan_html_for_context(page)
    if html_context is None:
        html_context = {}
    html_context.update(context)

    # Step 2: Delegate to state/county handler if available
    state_handler = get_handler_from_context(
        state=html_context.get("state"),
        county=html_context.get("county")
    )
    if state_handler and hasattr(state_handler, "parse"):
        logger.info("[INFO] Delegating to state handler for parsing.")
        return state_handler.parse(page, html_context)

    # Step 3: Let user select contest if not already set
    contest_title = html_context.get("selected_race")
    if not contest_title:
        logger.info("[INFO] No contest selected. Prompting user for selection.")
        detected_races = get_detected_races_from_context(context)
        selected_contests = select_contest(detected_races)
        if selected_contests is None:
            logger.warning("[WARN] No contests selected. Skipping parsing.")
            return {}

        # Update the context with selected contests
        html_context["selected_race"] = selected_contests

    # Step 4: Extract tables and data
    logger.info("[INFO] Extracting tables from the page.")
    precinct_tables = extract_precinct_tables(page)
    if not precinct_tables:
        logger.warning("[WARN] No precinct tables found. Exiting.")
        return {}

    # Step 5: Process each precinct table and gather data
    all_data = []
    for precinct_name, table_element in precinct_tables:
        headers, data = extract_table_data(table_element)
        all_data.extend(data)

    # Step 6: Finalize output
    if all_data:
        logger.info("[INFO] Finalizing election output.")
        metadata = {
            "state": html_context.get("state"),
            "county": html_context.get("county"),
            "contest_title": contest_title,
        }
        return finalize_election_output(headers, all_data, contest_title, metadata)
    else:
        logger.warning("[WARN] No data extracted from precinct tables.")
        return {}

# Example usage
# if __name__ == "__main__":
#     with sync_playwright() as p:
#         browser = p.chromium.launch(headless=False)
#         page = browser.new_page()
#         page.goto("URL_TO_ELECTION_DATA")
#         output = orchestrate_election_parsing(page)
#         print(output)
#         browser.close()