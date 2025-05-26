import importlib
from playwright.sync_api import Page
from typing import Optional, Tuple, Any
from ....utils.shared_logger import logger

def parse(page: Page, html_context: Optional[dict] = None) -> Tuple[Any, Any, Any, dict]:
    """
    State-level handler for New York.
    Delegates to the correct county parser if available.
    Returns standardized (headers, data, contest_title, metadata).
    """
    if html_context is None:
        html_context = {}

    county = (html_context.get("county") or "").strip().lower().replace(" ", "_")
    if not county:
        logger.warning("[NY Handler] No county specified in html_context.")
        return None, None, None, {"error": "No county specified for NY handler."}

    module_path = f"webapp.parser.handlers.states.new_york.county.{county}"

    try:
        county_module = importlib.import_module(module_path)
        logger.info(f"[NY Handler] Routing to county parser: {module_path}")
        # Expect county_module.parse to return (headers, data, contest_title, metadata)
        return county_module.parse(page, html_context)
    except ModuleNotFoundError:
        logger.warning(f"[NY Handler] No specific parser implemented for county: '{county}'. Please add it under {module_path}.py")
        return None, None, None, {"error": f"No handler found for NY county: '{county}'"}
    except Exception as e:
        logger.error(f"[NY Handler] Error in county parser: {e}")
        return None, None, None, {"error": str(e)}