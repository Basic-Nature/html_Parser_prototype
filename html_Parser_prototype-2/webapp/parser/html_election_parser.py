# filepath: c:\Users\olivi\OneDrive\Desktop\html_Parser_prototype\webapp\parser\html_election_parser.py

from playwright.sync_api import Page
from .state_router import get_handler_from_context
from .utils.html_scanner import scan_html_for_context, get_detected_races_from_context
from .utils.contest_selector import select_contest
from .utils.table_builder import extract_table_data
from .utils.shared_logger import logger
from .utils.output_utils import finalize_election_output
from .utils.user_prompt import prompt_user_input

def orchestrate_election_parsing(page: Page, html_context=None):
    """
    Orchestrates the entire election data parsing pipeline.
    
    Args:
        page (Page): The Playwright page object containing the HTML content.
        html_context (dict): Optional context for the HTML parsing.
    
    Returns:
        None
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
        available_races = get_detected_races_from_context(context)
        contest_title = select_contest(available_races)
        if contest_title is None:
            logger.warning("[WARN] No contest selected. Skipping parsing.")
            return

    # Step 4: Extract contest panel and precinct tables
    contest_panel = extract_contest_panel(page, contest_title)
    precinct_tables = extract_precinct_tables(contest_panel)

    # Step 5: Parse each precinct table and collect data
    all_data = []
    headers = []
    for precinct_name, table_element in precinct_tables:
        logger.info(f"[INFO] Parsing precinct table for: {precinct_name}")
        table_headers, table_data = extract_table_data(table_element)
        headers = headers or table_headers  # Initialize headers
        all_data.extend(table_data)

    # Step 6: Finalize and output the election data
    if all_data:
        logger.info("[INFO] Finalizing election output.")
        metadata = {
            "state": html_context.get("state"),
            "county": html_context.get("county"),
            "contest_title": contest_title
        }
        finalize_election_output(headers, all_data, contest_title, metadata)
    else:
        logger.warning("[WARN] No data collected from precinct tables.")

def extract_contest_panel(page: Page, contest_title: str):
    """
    Extracts the contest panel from the page based on the contest title.
    
    Args:
        page (Page): The Playwright page object.
        contest_title (str): The title of the contest to extract.
    
    Returns:
        Playwright Locator: The contest panel element.
    """
    # Implementation of extracting the contest panel
    # This function should return the appropriate Playwright Locator for the contest panel
    pass

def extract_precinct_tables(panel):
    """
    Extracts precinct tables from the contest panel.
    
    Args:
        panel: The contest panel element.
    
    Returns:
        List[Tuple[str, Any]]: A list of tuples containing precinct names and their corresponding table elements.
    """
    # Implementation of extracting precinct tables
    # This function should return a list of tuples (precinct_name, table_element)
    pass