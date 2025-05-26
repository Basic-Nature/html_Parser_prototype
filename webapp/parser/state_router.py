# state_router.py
# ===============================================
# Dynamically routes to the correct state or county-specific handler module
# Uses importlib for auto-resolution from folder structure.
# Now uses context_library.json for state/county mapping.
# ===============================================
import os
import importlib
from typing import Optional, Dict, Any, List
from .utils.shared_logger import logger

# --- Load state/county mapping from context library JSON ---
import json

CONTEXT_LIBRARY_PATH = os.path.join(
    os.path.dirname(__file__), "Context_Integration", "context_library.json"
)
if os.path.exists(CONTEXT_LIBRARY_PATH):
    with open(CONTEXT_LIBRARY_PATH, "r", encoding="utf-8") as f:
        CONTEXT_LIBRARY = json.load(f)
    STATE_MODULE_MAP = CONTEXT_LIBRARY.get("state_module_map", {})
else:
    logger.error("[State Router] context_library.json not found. State routing will fail.")
    STATE_MODULE_MAP = {}

LOADED_HANDLERS: Dict[str, Any] = {}

def import_handler(module_path: str):
    """Dynamically import and return a handler module from the given dotted path."""
    try:
        if module_path not in LOADED_HANDLERS:
            logger.debug(f"[Router] Importing handler module: {module_path}")
            module = importlib.import_module(module_path)
            LOADED_HANDLERS[module_path] = module
        return LOADED_HANDLERS[module_path]
    except ModuleNotFoundError as e:
        logger.warning(f"[Router] Module not found: {module_path} — {e}")
        return None

def get_handler(state_abbreviation: Optional[str], county_name: Optional[str] = None):
    """
    Dynamically loads and returns a handler based on state or state+county keys.
    Returns the handler module or None if not found.
    """
    if not state_abbreviation:
        logger.warning("[Router] Missing state_abbreviation — skipping handler resolution.")
        return None

    normalized_state = state_abbreviation.strip().lower().replace(" ", "_")
    state_key = STATE_MODULE_MAP.get(normalized_state, normalized_state)
    module_path = f"webapp.parser.handlers.states.{state_key}"

    # --- COUNTY HANDLER PRIORITY ---
    if county_name:
        normalized_county = county_name.strip().lower().replace(" ", "_")
        composite_path = f"{module_path}.county.{normalized_county}"
        logger.debug(f"[Router] Attempting county-level handler: {composite_path}")
        module = import_handler(composite_path)
        if module:
            logger.info(f"[Router] Routed to handler for {state_key}_{normalized_county}")
            return module

    # --- STATE HANDLER FALLBACK ---
    module = import_handler(module_path)
    if module:
        logger.info(f"[Router] Routed to state handler for {state_key}")
        return module
    else:
        logger.warning(f"[Router] Could not import module for state '{state_abbreviation}'")
        return None

def get_handler_from_context(context: Dict[str, Any]):
    """
    Extracts state and county info from a context dictionary and routes accordingly.
    Returns the handler module or None if not found.
    """
    state = context.get("state")
    county = context.get("county")
    if not state and not county:
        logger.warning("[Router] No state or county provided in context.")
        return None

    # Try to infer state from filename if missing
    if not state:
        filename = context.get("filename", "").lower()
        tokens = filename.replace("_", " ").split()
        for token in tokens:
            if token in STATE_MODULE_MAP:
                state = token
                context["state"] = token
                logger.info(f"[Router] Inferred state '{state}' from filename: {filename}")
                break
        if not state:
            logger.warning("[Router] No state provided in context.")
            return None

    handler = get_handler(state_abbreviation=state, county_name=county)
    if handler:
        logger.info(f"[Router] Handler resolved for state '{state}' and county '{county}'")
        return handler
    else:
        logger.warning("[Router] No handler could be resolved.")
    return None

# --- CLI Utilities (optional, can be removed if not needed) ---

def list_available_states() -> List[str]:
    """List all available state handler modules."""
    base_path = os.path.join(os.path.dirname(__file__), "handlers", "states")
    if not os.path.isdir(base_path):
        logger.warning("[Router] handlers/states directory not found.")
        return []
    return sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])

def list_available_counties(state_key: str) -> List[str]:
    """List all available county handler modules for a given state."""
    base_path = os.path.join(os.path.dirname(__file__), "handlers", "states", state_key, "county")
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