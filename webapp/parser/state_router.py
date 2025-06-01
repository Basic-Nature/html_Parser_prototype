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
import difflib 
import json
from .utils.shared_logic import normalize_state_name, normalize_county_name
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
    Now validates detected county against available handlers and context library.
    """
    import importlib
    from .Context_Integration.context_coordinator import ContextCoordinator, dynamic_state_county_detection

    # Step 1: Enrich context using the coordinator (NLP, ML, etc.)
    coordinator = ContextCoordinator(use_library=True, enable_ml=False, alert_monitor=False)
    enriched = coordinator.organize_and_enrich(context)
    html = context.get("raw_html", "") or (enriched.get("raw_html") if enriched else "")
    context_library = coordinator.library

    # Step 2: Use dynamic_state_county_detection for best guess
    county, state, handler_path, detection_log = dynamic_state_county_detection(
        context, html, context_library, debug=True
    )

    # --- NEW: Validate detected county before setting in context ---
    valid_county = None
    valid_state = None
    available_states = list_available_states()
    # Normalize state
    if state:
        normalized_state = normalize_state_name(state)
        if normalized_state in available_states:
            valid_state = normalized_state
        else:
            # Fuzzy match state
            matches = fuzzy_match_handler(normalized_state, available_states)
            if matches:
                valid_state = matches[0]
    else:
        valid_state = None

    available_counties = list_available_counties(valid_state) if valid_state else []
    if county:
        normalized_county = normalize_county_name(county)
        if normalized_county in available_counties:
            valid_county = normalized_county
        else:
            # Check if county is actually a district of a known county (context_library)
            known_county_to_district = context_library.get("Known_county_to_district_map", {})
            for county_name, districts in known_county_to_district.items():
                if normalized_county in [normalize_county_name(d) for d in districts]:
                    valid_county = normalize_county_name(county_name)
                    logger.info(f"[Router] '{county}' matched as district of county '{county_name}'. Using '{county_name}'.")
                    break
            if not valid_county:
                # Fuzzy match county
                matches = fuzzy_match_handler(normalized_county, available_counties)
                if matches:
                    valid_county = matches[0]
                    logger.info(f"[Router] '{county}' not found. Fuzzy matched to '{valid_county}'.")

    # Step 3: Update context with validated values for downstream use
    if valid_state:
        context["state"] = valid_state
    if valid_county:
        context["county"] = valid_county

    # Step 4: Log detection steps
    logger.info("[Router] Detection log:")
    for log_entry in detection_log:
        logger.info(f"    {log_entry}")

    # Step 5: Attempt to import the handler module
    # Try county handler first if both are valid
    if valid_state and valid_county:
        handler_path = f"webapp.parser.handlers.states.{valid_state}.county.{valid_county}"
        logger.info(f"[Router] Attempting to import handler: {handler_path}")
        module = import_handler(handler_path)
        if module:
            logger.info(f"[Router] Routed to handler: {handler_path}")
            return module
        else:
            logger.warning(f"[Router] Could not import handler: {handler_path}")

    # Fallback to state-level handler if county handler not found
    if valid_state:
        fallback_path = f"webapp.parser.handlers.states.{valid_state}"
        logger.info(f"[Router] Attempting fallback to state handler: {fallback_path}")
        module = import_handler(fallback_path)
        if module:
            logger.info(f"[Router] Routed to fallback state handler: {fallback_path}")
            return module

    logger.warning("[Router] No suitable handler found for context.")
    return None

def list_available_handlers(level: str = "state", state: str = None) -> list:
    """
    List all available handler modules for a given level.
    level: "state" or "county"
    If level is "county", state must be provided (normalized, e.g., "new_york").
    """
    if level == "state":
        return list_available_states()
    elif level == "county" and state:
        return list_available_counties(state)
    else:
        return []

def list_available_states() -> list:
    """List all available state handler modules (normalized names)."""
    base_path = STATE_HANDLER_BASE_PATH
    if not os.path.isdir(base_path):
        logger.warning("[Router] handlers/states directory not found.")
        return []
    return sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])

def list_available_counties(state_key: str) -> list:
    """
    List all available county handler modules for a given state (normalized names, no .py).
    """
    base_path = os.path.join(STATE_HANDLER_BASE_PATH, state_key, "county")
    if not os.path.isdir(base_path):
        logger.warning(f"[Router] counties directory not found for state: {state_key}")
        return []
    counties = []
    for fname in os.listdir(base_path):
        if fname.endswith(".py") and not fname.startswith("__"):
            counties.append(fname[:-3])  # strip .py
        elif os.path.isdir(os.path.join(base_path, fname)):
            counties.append(fname)
    return sorted(counties)

def fuzzy_match_handler(query: str, choices: list, n=3, cutoff=0.6) -> list:
    """
    Return a list of close matches for query from choices.
    """
    if not query or not choices:
        return []
    return difflib.get_close_matches(query, choices, n=n, cutoff=cutoff)

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