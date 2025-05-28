# state_router.py
# ===============================================
# Dynamically routes to the correct state or county-specific handler module
# Uses importlib for auto-resolution from folder structure.
# Now uses context_library.json for state/county mapping.
# Also provides state/county info for format_router and download_utils.
# ===============================================
import os
import importlib
from typing import Optional, Dict, Any, List, Tuple
from .utils.logger_instance import logger
from .config import CONTEXT_LIBRARY_PATH
 
import json

def import_handler(module_path: str):
    """
    Dynamically import a handler module by its dotted path.
    Returns the module if found, else None.
    """
    try:
        if module_path in LOADED_HANDLERS:
            return LOADED_HANDLERS[module_path]
        module = importlib.import_module(module_path)
        LOADED_HANDLERS[module_path] = module
        return module
    except Exception as e:
        logger.debug(f"[Router] Could not import {module_path}: {e}")
        return None
    
if os.path.exists(CONTEXT_LIBRARY_PATH):
    with open(CONTEXT_LIBRARY_PATH, "r", encoding="utf-8") as f:
        CONTEXT_LIBRARY = json.load(f)
    STATE_MODULE_MAP = CONTEXT_LIBRARY.get("state_module_map", {})
else:
    logger.error("[State Router] context_library.json not found. State routing will fail.")
    STATE_MODULE_MAP = {}

LOADED_HANDLERS: Dict[str, Any] = {}

STATE_HANDLER_BASE_PATH = os.path.join(
    os.path.dirname(__file__), "handlers", "states"
)

def get_handler(context: Dict[str, Any], url: Optional[str] = None) -> Optional[Any]:
    """
    Dynamically resolves and returns the best handler module for the given context.
    Uses dynamic_state_county_detection for robust detection.
    Logs all routing attempts and fallbacks.
    """
    import importlib
    from .Context_Integration.context_coordinator import ContextCoordinator, dynamic_state_county_detection

    # Step 1: Enrich context using the coordinator (NLP, ML, etc.)
    coordinator = ContextCoordinator(use_library=True, enable_ml=False, alert_monitor=False)
    enriched = coordinator.organize_and_enrich(context)
    # Try to get raw_html for NLP if available
    html = context.get("raw_html", "") or (enriched.get("raw_html") if enriched else "")
    # Load context library for detection
    context_library = coordinator.library

    # Step 2: Use dynamic_state_county_detection for best guess
    county, state, handler_path, detection_log = dynamic_state_county_detection(
        context, html, context_library, debug=True
    )

    # Step 3: Update context with detected values for downstream use
    if state:
        context["state"] = state
    if county:
        context["county"] = county

    # Step 4: Log detection steps
    logger.info("[Router] Detection log:")
    for log_entry in detection_log:
        logger.info(f"    {log_entry}")

    # Step 5: Attempt to import the handler module
    if handler_path:
        logger.info(f"[Router] Attempting to import handler: {handler_path}")
        module = import_handler(handler_path)
        if module:
            logger.info(f"[Router] Routed to handler: {handler_path}")
            return module
        else:
            logger.warning(f"[Router] Could not import handler: {handler_path}")

    # Step 6: Fallback to state-level handler if county handler not found
    if state:
        normalized_state = state.strip().lower().replace(" ", "_")
        fallback_path = f"webapp.parser.handlers.states.{normalized_state}"
        logger.info(f"[Router] Attempting fallback to state handler: {fallback_path}")
        module = import_handler(fallback_path)
        if module:
            logger.info(f"[Router] Routed to fallback state handler: {fallback_path}")
            return module

    logger.warning("[Router] No suitable handler found for context.")
    return None

def list_available_states() -> List[str]:
    """List all available state handler modules."""
    base_path = STATE_HANDLER_BASE_PATH
    if not os.path.isdir(base_path):
        logger.warning("[Router] handlers/states directory not found.")
        return []
    return sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])

def list_available_counties(state_key: str) -> List[str]:
    """List all available county handler modules for a given state."""
    base_path = os.path.join(STATE_HANDLER_BASE_PATH, state_key, "county")
    if not os.path.isdir(base_path):
        logger.warning(f"[Router] counties directory not found for state: {state_key}")
        return []
    return sorted([d for d in os.listdir(base_path) if d.endswith(".py") and not d.startswith("__")])

def cli():
    """Simple CLI for state_router utilities."""
    import argparse
    parser = argparse.ArgumentParser(description="State Router CLI Utility")
    parser.add_argument("--list-states", action="store_true", help="List all available state handlers")
    parser.add_argument("--list-counties", metavar="STATE", help="List all available county handlers for a state")
    args = parser.parse_args()

    if args.list_states:
        print("Available states:")
        for state in list_available_states():
            print(f" - {state}")
    elif args.list_counties:
        print(f"Available counties for {args.list_counties}:")
        for county in list_available_counties(args.list_counties):
            print(f" - {county}")
    else:
        parser.print_help()

if __name__ == "__main__":
    cli()