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

STATE_HANDLER_BASE_PATH = os.path.join(
    os.path.dirname(__file__), "handlers", "states"
)

def get_handler(context: Dict[str, Any], url: Optional[str] = None) -> Optional[Any]:
    """
    Dynamically resolves and returns the best handler module for the given context.
    - Uses ContextCoordinator to enrich and normalize context.
    - Tries state/county from context, filename, or URL.
    - Logs all routing attempts and fallbacks.
    """
    from .Context_Integration.context_coordinator import ContextCoordinator
    # Step 1: Enrich context using the coordinator (NLP, ML, etc.)
    coordinator = ContextCoordinator(use_library=True, enable_ml=False, alert_monitor=False)
    enriched = coordinator.organize_and_enrich(context)
    state = context.get("state") or (enriched.get("metadata", {}).get("state") if enriched else None)
    county = context.get("county") or (enriched.get("metadata", {}).get("county") if enriched else None)

    # Step 2: Try to infer state/county from filename if missing
    if not state:
        filename = context.get("filename", "").lower()
        tokens = filename.replace("_", " ").split()
        for token in tokens:
            if token in STATE_MODULE_MAP:
                state = token
                context["state"] = token
                logger.info(f"[Router] Inferred state '{state}' from filename: {filename}")
                break

    # Step 3: Try to infer state/county from URL if still missing
    if not state and url:
        # Example: look for '/ny/' or '/new_york/' in the URL
        for key in STATE_MODULE_MAP:
            if f"/{key}/" in url or f"/{STATE_MODULE_MAP[key]}/" in url:
                state = key
                context["state"] = key
                logger.info(f"[Router] Inferred state '{state}' from URL: {url}")
                break

    if not state:
        logger.warning("[Router] No state provided in context, filename, or URL.")
        return None

    # Step 4: Normalize for module path
    normalized_state = state.strip().lower().replace(" ", "_")
    state_key = STATE_MODULE_MAP.get(normalized_state, normalized_state)
    module_path = f"webapp.parser.handlers.states.{state_key}"

    # Step 5: Try county-level handler first
    if county:
        normalized_county = county.strip().lower().replace(" ", "_")
        composite_path = f"{module_path}.county.{normalized_county}"
        logger.debug(f"[Router] Attempting county-level handler: {composite_path}")
        module = import_handler(composite_path)
        if module:
            logger.info(f"[Router] Routed to handler for {state_key}_{normalized_county}")
            return module

    # Step 6: Fallback to state-level handler
    module = import_handler(module_path)
    if module:
        logger.info(f"[Router] Routed to state handler for {state_key}")
        return module

    logger.warning(f"[Router] Could not import module for state '{state}'")
    return None
# --- CLI Utilities (optional, can be removed if not needed) ---

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