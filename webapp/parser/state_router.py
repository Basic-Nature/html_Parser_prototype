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
from .utils.shared_logger import logger
from .config import CONTEXT_LIBRARY_PATH, BASE_DIR
import difflib 
import json
import re
from .utils.shared_logic import normalize_state_name, normalize_county_name

# Preload on import
import time

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

STATE_HANDLER_BASE_PATH = os.path.join(BASE_DIR, "parser", "handlers", "states")

# === Handler Map Caching ===
HANDLER_MAP = {
    "states": [],
    "counties_by_state": {},
    "last_loaded": None
}
FUZZY_MATCH_THRESHOLD = 0.6  # Default, can be overridden
DEBUG_MODE = False

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

def preload_handler_map():
    """Scan and cache all available state/county handlers."""
    states = list_available_states()
    counties_by_state = {s: list_available_counties(s) for s in states}
    HANDLER_MAP["states"] = states
    HANDLER_MAP["counties_by_state"] = counties_by_state
    HANDLER_MAP["last_loaded"] = time.time()
    logger.info(f"[Router] Handler map preloaded: {len(states)} states, {sum(len(c) for c in counties_by_state.values())} counties.")

def reload_handler_map():
    preload_handler_map()
    logger.info("[Router] Handler map reloaded.")



def scan_url_for_state_county(url: str, available_states: List[str], available_counties_by_state: Dict[str, List[str]]) -> Tuple[Optional[str], Optional[str], List[str]]:
    """
    Scan the URL for state/county clues using regex and keyword heuristics.
    Returns (state, county, log_entries)
    """
    log_entries = []
    if not url:
        log_entries.append("[URL Scan] No URL provided.")
        return None, None, log_entries
    url_lower = url.lower()
    # Try to match state in URL
    state_match = None
    for state in available_states:
        if state in url_lower:
            state_match = state
            log_entries.append(f"[URL Scan] Matched state '{state}' in URL.")
            break
    # Try to match county in URL (if state found)
    county_match = None
    if state_match:
        counties = available_counties_by_state.get(state_match, [])
        for county in counties:
            if county in url_lower:
                county_match = county
                log_entries.append(f"[URL Scan] Matched county '{county}' in URL.")
                break
    # Fuzzy match if not found
    if not state_match:
        matches = fuzzy_match_handler(url_lower, available_states)
        if matches:
            state_match = matches[0]
            log_entries.append(f"[URL Scan] Fuzzy matched state '{state_match}' in URL.")
    if state_match and not county_match:
        counties = available_counties_by_state.get(state_match, [])
        matches = fuzzy_match_handler(url_lower, counties)
        if matches:
            county_match = matches[0]
            log_entries.append(f"[URL Scan] Fuzzy matched county '{county_match}' in URL.")
    return state_match, county_match, log_entries


def fuzzy_match_handler(query: str, choices: list, n=3, cutoff=None, debug=False) -> list:
    """
    Return a list of close matches for query from choices.
    cutoff: float, fuzzy match threshold (default: FUZZY_MATCH_THRESHOLD)
    debug: bool, if True, log match scores
    """
    if not query or not choices:
        return []
    cutoff = cutoff if cutoff is not None else FUZZY_MATCH_THRESHOLD
    matches = difflib.get_close_matches(query, choices, n=n, cutoff=cutoff)
    if debug and matches:
        logger.info(f"[Router][Fuzzy] Query '{query}' matches: {matches} (cutoff={cutoff})")
    return matches

def get_handler(context: Dict[str, Any], url: Optional[str] = None, debug: bool = False, fuzzy_cutoff: float = None) -> Any:
    """
    Dynamically resolves and returns the best handler module for the given context.
    Now scans the URL first for state/county clues, then context, then context library, then filesystem.
    Logs all routing attempts, available options, and fallbacks.
    Returns a dict with keys: 'handler', 'summary', 'log', 'error' (if any)
    """
    from .Context_Integration.context_coordinator import ContextCoordinator, dynamic_state_county_detection
    summary = {"attempts": [], "final": None, "error": None}
    log = []
    # Use preloaded handler map
    available_states = HANDLER_MAP["states"]
    available_counties_by_state = HANDLER_MAP["counties_by_state"]
    context_library_states = list(STATE_MODULE_MAP.keys()) if STATE_MODULE_MAP else []
    if debug:
        logger.info(f"[Router] Available states (filesystem): {available_states}")
        logger.info(f"[Router] Available states (context library): {context_library_states}")
        for s in available_states:
            logger.info(f"[Router] Counties for state '{s}': {available_counties_by_state[s]}")
    # Step 1: Scan URL for clues first
    url_state, url_county, url_log = scan_url_for_state_county(url or context.get('url', ''), available_states, available_counties_by_state)
    for entry in url_log:
        log.append(entry)
        if debug:
            logger.info(entry)
    # Step 2: Enrich context using the coordinator (NLP, ML, etc.)
    coordinator = ContextCoordinator(use_library=True, enable_ml=False, alert_monitor=False)
    enriched = coordinator.organize_and_enrich(context)
    html = context.get("raw_html", "") or (enriched.get("raw_html") if enriched else "")
    context_library = coordinator.library
    # Step 3: Use dynamic_state_county_detection for best guess (context, html, context library)
    county, state, handler_path, detection_log = dynamic_state_county_detection(
        context, html, context_library, debug=True
    )
    for log_entry in detection_log:
        log.append(f"[Context Detection] {log_entry}")
        if debug:
            logger.info(f"[Router] [Context Detection] {log_entry}")
    # Step 4: Decide on state/county using priority: URL > context > context library > filesystem
    valid_state = None
    valid_county = None
    fuzzy_cutoff = fuzzy_cutoff if fuzzy_cutoff is not None else FUZZY_MATCH_THRESHOLD
    # 1. Use URL scan if found
    if url_state:
        valid_state = url_state
        if url_county:
            valid_county = url_county
        summary["attempts"].append(f"URL scan: state={url_state}, county={url_county}")
    # 2. Else use context detection
    if not valid_state and state:
        normalized_state = normalize_state_name(state)
        if normalized_state in available_states:
            valid_state = normalized_state
        else:
            matches = fuzzy_match_handler(normalized_state, available_states, cutoff=fuzzy_cutoff, debug=debug)
            if matches:
                valid_state = matches[0]
                summary["attempts"].append(f"Fuzzy matched state '{normalized_state}' to '{valid_state}'")
    if valid_state and not valid_county and county:
        normalized_county = normalize_county_name(county)
        counties = available_counties_by_state.get(valid_state, [])
        if normalized_county in counties:
            valid_county = normalized_county
        else:
            # Check if county is a district of a known county (context_library)
            known_county_to_district = context_library.get("Known_county_to_district_map", {})
            for county_name, districts in known_county_to_district.items():
                if normalized_county in [normalize_county_name(d) for d in districts]:
                    valid_county = normalize_county_name(county_name)
                    log.append(f"'{county}' matched as district of county '{county_name}'. Using '{county_name}'.")
                    break
            if not valid_county:
                matches = fuzzy_match_handler(normalized_county, counties, cutoff=fuzzy_cutoff, debug=debug)
                if matches:
                    valid_county = matches[0]
                    log.append(f"'{county}' not found. Fuzzy matched to '{valid_county}'.")
    # Step 5: Update context with validated values for downstream use
    if valid_state:
        context["state"] = valid_state
    if valid_county:
        context["county"] = valid_county
    summary["final"] = {"state": valid_state, "county": valid_county}
    log.append(f"Final resolved state: {valid_state}, county: {valid_county}")
    # Step 6: Attempt to import the handler module
    handler = None
    error = None
    attempted_paths = []
    if valid_state and valid_county:
        handler_path = f"webapp.parser.handlers.states.{valid_state}.county.{valid_county}"
        attempted_paths.append(handler_path)
        log.append(f"Attempting to import handler: {handler_path}")
        handler = import_handler(handler_path)
        if handler:
            log.append(f"Routed to handler: {handler_path}")
        else:
            log.append(f"Could not import handler: {handler_path}")
    # Fallback to state-level handler if county handler not found
    if not handler and valid_state:
        fallback_path = f"webapp.parser.handlers.states.{valid_state}"
        attempted_paths.append(fallback_path)
        log.append(f"Attempting fallback to state handler: {fallback_path}")
        handler = import_handler(fallback_path)
        if handler:
            log.append(f"Routed to fallback state handler: {fallback_path}")
        else:
            log.append(f"Could not import fallback state handler: {fallback_path}")
    if not handler:
        error = {
            "message": "No suitable handler found for context.",
            "attempted_paths": attempted_paths,
            "final_state": valid_state,
            "final_county": valid_county,
            "log": log
        }
        log.append("No suitable handler found for context.")
    summary["log"] = log
    summary["error"] = error
    return {"handler": handler, "summary": summary}

def list_available_handlers(level=None, state=None, fuzzy=False, refresh=False, debug=False):
    """
    List available handlers dynamically, with options for level (state/county), fuzzy matching, refresh, and diagnostics.
    Returns sorted, deduplicated results. Normalizes input for robust lookup.
    """
    if refresh:
        preload_handler_map()
    handlers = {}
    states = HANDLER_MAP["states"] if HANDLER_MAP["states"] else list_available_states()
    counties_by_state = HANDLER_MAP["counties_by_state"] if HANDLER_MAP["counties_by_state"] else {s: list_available_counties(s) for s in states}
    # Normalize state input
    norm_state = normalize_state_name(state) if state else None
    if debug:
        logger.info(f"[list_available_handlers] level={level}, state={state}, fuzzy={fuzzy}, refresh={refresh}")
    for s in states:
        counties = counties_by_state.get(s, [])
        handlers[s] = sorted(set(counties))
    if level == "state":
        return sorted(set(handlers.keys()))
    if level == "county" and norm_state:
        counties = handlers.get(norm_state, [])
        if fuzzy and state:
            # Fuzzy match state if not found
            import difflib
            matches = difflib.get_close_matches(norm_state, handlers.keys(), n=3, cutoff=FUZZY_MATCH_THRESHOLD)
            if matches:
                counties = handlers.get(matches[0], [])
                if debug:
                    logger.info(f"[list_available_handlers] Fuzzy matched state '{state}' to '{matches[0]}'")
        return sorted(set(counties))
    if fuzzy and state:
        # Fuzzy match state at top level
        import difflib
        matches = difflib.get_close_matches(norm_state, handlers.keys(), n=3, cutoff=FUZZY_MATCH_THRESHOLD)
        if matches:
            if debug:
                logger.info(f"[list_available_handlers] Fuzzy matched state '{state}' to '{matches[0]}'")
            return {matches[0]: handlers[matches[0]]}
    return handlers

# --- CLI improvements for diagnostics and robust output ---
def cli():
    """CLI for state_router utilities."""
    import argparse
    parser = argparse.ArgumentParser(description="State Router CLI Utility")
    parser.add_argument("--list-states", action="store_true", help="List all available state handlers")
    parser.add_argument("--list-counties", metavar="STATE", help="List all available county handlers for a state")
    parser.add_argument("--test-route", metavar="URL_OR_CONTEXT", help="Test routing for a given URL or JSON context file")
    parser.add_argument("--debug", action="store_true", help="Enable debug/verbose logging")
    parser.add_argument("--reload", action="store_true", help="Reload handler map before running")
    parser.add_argument("--fuzzy-cutoff", type=float, default=None, help="Fuzzy match threshold (default: 0.6)")
    parser.add_argument("--fuzzy", action="store_true", help="Enable fuzzy matching for handler listing")
    parser.add_argument("--refresh", action="store_true", help="Refresh handler map before listing")
    args = parser.parse_args()
    global DEBUG_MODE
    if args.debug:
        DEBUG_MODE = True
    if args.reload or args.refresh:
        reload_handler_map()
    if args.list_states:
        print("Available states:")
        for state in list_available_handlers(level="state", fuzzy=args.fuzzy, refresh=args.refresh, debug=args.debug):
            print(f" - {state}")
    elif args.list_counties:
        state = args.list_counties
        counties = list_available_handlers(level="county", state=state, fuzzy=args.fuzzy, refresh=args.refresh, debug=args.debug)
        if counties:
            print(f"Available counties for {state}:")
            for county in counties:
                print(f" - {county}")
        else:
            print(f"No counties found for state '{state}'. Try --fuzzy for fuzzy matching.")
    elif args.test_route:
        # Try to load as JSON context, else treat as URL
        test_input = args.test_route
        context = None
        url = None
        import json
        if os.path.isfile(test_input):
            with open(test_input, "r", encoding="utf-8") as f:
                try:
                    context = json.load(f)
                except Exception as e:
                    print(f"Failed to load context from file: {e}")
                    return
        else:
            url = test_input
        result = get_handler(context or {}, url=url, debug=args.debug, fuzzy_cutoff=args.fuzzy_cutoff)
        print("Routing result:")
        print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
        if result["handler"]:
            print(f"Handler module: {getattr(result['handler'], '__name__', str(result['handler']))}")
        else:
            print("No suitable handler found.")
    else:
        parser.print_help()