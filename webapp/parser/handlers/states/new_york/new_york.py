import importlib
from playwright.sync_api import Page
from typing import Optional
from ....utils.shared_logger import logger

def parse(page: Page, html_context: Optional[dict] = None):
    if html_context is None:
        html_context = {}
    """
    State-level handler for New York.
    Dynamically delegates to the correct county parser if available.
    Logs a warning if the county is not yet implemented.

    Args:
        page (Page): Playwright browser page with the election results loaded.
        html_context (dict): Pre-scanned context with races, years, etc.

    Returns:
        Tuple: (contest_title, headers, rows, metadata)
    """
    county = (html_context.get("county") or "").strip().lower().replace(" ", "_")
    if not county:
        logger.warning("[NY Handler] No county specified in html_context.")
        raise NotImplementedError("No county specified for NY handler.")

    module_path = f"webapp.parser.handlers.states.new_york.county.{county}"

    try:
        county_module = importlib.import_module(module_path)
        logger.info(f"[NY Handler] Routing to county parser: {module_path}")
        return county_module.parse(page, html_context)
    except ModuleNotFoundError:
        logger.warning(f"[NY Handler] No specific parser implemented for county: '{county}'. Please add it under {module_path}.py")
        raise NotImplementedError(f"No handler found for NY county: '{county}'")