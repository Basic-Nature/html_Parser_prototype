# state_router.py
# ===============================================
# Dynamically routes to the correct state or county-specific handler module
# Uses importlib for auto-resolution from folder structure.
# Now uses librarian.py for state/county mapping.
# Also provides state/county info for format_router and download_utils.
# ===============================================
import os
import importlib
from typing import Optional, Dict, Any, List, Tuple
from .utils.shared_logger import log_info, log_warning, log_debug, log_error
import traceback
from .config import BASE_DIR
from .bots.librarian import STATE_MODULE_MAP, KNOWN_COUNTY_TO_PRECINCTS_MAP
import difflib
import time
from .utils.shared_logic import normalize_state_name, normalize_county_name
import orjson
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
        log_warning("[Router] handlers/states directory not found.")
        return []
    return sorted([
        normalize_state_name(d)
        for d in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, d))
    ])

def list_available_counties(state_key: str, suppress_warning: bool = False) -> list:
    """
    List all available county handler modules for a given state (normalized names, no .py).
    If suppress_warning is True, do not log warnings if the counties directory is missing.
    """
    state_key = normalize_state_name(state_key)
    base_path = os.path.join(STATE_HANDLER_BASE_PATH, state_key, "county")
    if not os.path.isdir(base_path):
        if not suppress_warning:
            log_warning(f"[Router] counties directory not found for state: {state_key}")
        return []
    counties = []
    for fname in os.listdir(base_path):
        if fname.endswith(".py") and not fname.startswith("__"):
            counties.append(normalize_county_name(fname[:-3]))
        elif os.path.isdir(os.path.join(base_path, fname)):
            counties.append(normalize_county_name(fname))
    return sorted(counties)

def import_handler(module_or_file_path: str):
    """
    Import a handler module by either dotted module path (preferred)
    or filesystem path (ending in .py). Returns the module if found, else None.
    Logs detailed errors and gives usage hints.
    """
    try:
        # If already loaded, return cached
        if module_or_file_path in LOADED_HANDLERS:
            return LOADED_HANDLERS[module_or_file_path]

        # Detect if it's a filesystem path (endswith .py or contains os.sep)
        is_file_path = module_or_file_path.endswith('.py') or os.sep in module_or_file_path

        if is_file_path:
            # Convert file path to module path
            abs_path = os.path.abspath(module_or_file_path)
            if not os.path.exists(abs_path):
                log_error(f"[HTML Handler] Handler file does not exist: {abs_path}")
                log_info("[HTML Handler] Example of valid file path: webapp\\parser\\handlers\\states\\new_york\\county\\rockland.py")
                return None
            # Remove BASE_DIR and .py, convert to dotted path
            rel_path = os.path.relpath(abs_path, BASE_DIR)
            module_path = rel_path.replace(os.sep, ".").replace("/", ".")
            if module_path.endswith(".py"):
                module_path = module_path[:-3]
            log_info(f"[HTML Handler] Converted file path to module path: {module_path}")
        else:
            module_path = module_or_file_path

        try:
            module = importlib.import_module(module_path)
            LOADED_HANDLERS[module_or_file_path] = module
            return module
        except Exception as e:
            log_error(f"[HTML Handler] Failed to import handler from path '{module_or_file_path}': {e}")
            log_debug(f"[HTML Handler] Traceback:\n{traceback.format_exc()}")
            log_info("[HTML Handler] Example of valid module path: webapp.parser.handlers.states.new_york.county.rockland")
            log_info("[HTML Handler] Example of valid file path: webapp\\parser\\handlers\\states\\new_york\\county\\rockland.py")
            return None
    except Exception as e:
        log_error(f"[HTML Handler] Unexpected error importing handler: {e}\n{traceback.format_exc()}")
        return None

def prompt_for_handler_fallback(available_states, available_counties_by_state, last_error=None, max_attempts=3):
    """
    Prompt the user for manual state/county selection with robust fallback.
    Shows last error, allows cancel, and limits attempts.
    """
    attempts = 0
    state = None
    county = None
    while attempts < max_attempts:
        if last_error:
            print(f"\n[ERROR] Last import failed: {last_error}\n")
        print("Available states:", ", ".join(available_states))
        state = input("Enter state (or leave blank to cancel): ").strip().lower()
        if state == "cancel" or not state:
            print("Aborted by user.")
            return None, None
        if state not in available_states:
            print(f"State '{state}' not found. Try again.")
            attempts += 1
            continue
        counties = available_counties_by_state.get(state, [])
        print("Available counties:", ", ".join(counties))
        county = input("Enter county (or leave blank to skip county): ").strip().lower()
        if county == "cancel":
            print("Aborted by user.")
            return None, None
        if county and county not in counties:
            print(f"County '{county}' not found for state '{state}'. Try again.")
            attempts += 1
            continue
        return state, county if county else None
    print("Too many failed attempts. Exiting fallback.")
    return None, None

def preload_handler_map(restrict_to_states=None):
    """
    Scan and cache all available state/county handlers.
    If restrict_to_states is provided, only scan those states.
    """
    if restrict_to_states:
        states = [normalize_state_name(s) for s in restrict_to_states]
    else:
        states = list_available_states()
    counties_by_state = {
        normalize_state_name(s): list_available_counties(s, suppress_warning=True)
        for s in states
    }
    HANDLER_MAP["states"] = [normalize_state_name(s) for s in states]
    HANDLER_MAP["counties_by_state"] = counties_by_state
    HANDLER_MAP["last_loaded"] = time.time()
    log_info(f"[Router] Handler map preloaded: {len(states)} states, {sum(len(c) for c in counties_by_state.values())} counties.")

def reload_handler_map():
    """
    Reload the handler map cache.
    """
    preload_handler_map()
    log_info("[Router] Handler map reloaded.")

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
    state_match = None
    for state in available_states:
        if state in url_lower:
            state_match = state
            log_entries.append(f"[URL Scan] Matched state '{state}' in URL.")
            break
    county_match = None
    if state_match:
        counties = available_counties_by_state.get(state_match, [])
        for county in counties:
            if county in url_lower:
                county_match = county
                log_entries.append(f"[URL Scan] Matched county '{county}' in URL.")
                break
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
        log_info(f"[Router][Fuzzy] Query '{query}' matches: {matches} (cutoff={cutoff})")
    return matches

def list_available_handlers(level=None, state=None, fuzzy=False, refresh=False, debug=False):
    """
    List available handlers dynamically, with options for level (state/county), fuzzy matching, refresh, and diagnostics.
    Returns sorted, deduplicated results. Normalizes input for robust lookup.
    """
    if refresh:
        preload_handler_map()
    handlers = {}
    states = HANDLER_MAP["states"] if HANDLER_MAP["states"] else [normalize_state_name(s) for s in list_available_states()]
    counties_by_state = HANDLER_MAP["counties_by_state"] if HANDLER_MAP["counties_by_state"] else {normalize_state_name(s): list_available_counties(s) for s in states}
    norm_state = normalize_state_name(state) if state else None
    if debug:
        log_info(f"[list_available_handlers] level={level}, state={state}, fuzzy={fuzzy}, refresh={refresh}")
    for s in states:
        counties = counties_by_state.get(s, [])
        handlers[s] = sorted(set(counties))
    if level == "state":
        return sorted(set(handlers.keys()))
    if level == "county" and norm_state:
        counties = handlers.get(norm_state, [])
        if fuzzy and state:
            matches = difflib.get_close_matches(norm_state, handlers.keys(), n=3, cutoff=FUZZY_MATCH_THRESHOLD)
            if matches:
                counties = handlers.get(matches[0], [])
                if debug:
                    log_info(f"[list_available_handlers] Fuzzy matched state '{state}' to '{matches[0]}'")
        return sorted(set(counties))
    if fuzzy and state:
        matches = difflib.get_close_matches(norm_state, handlers.keys(), n=3, cutoff=FUZZY_MATCH_THRESHOLD)
        if matches:
            if debug:
                log_info(f"[list_available_handlers] Fuzzy matched state '{state}' to '{matches[0]}'")
            return {matches[0]: handlers[matches[0]]}
    return handlers

def get_handler(context: Dict[str, Any], url: Optional[str] = None, debug: bool = False, fuzzy_cutoff: float = None, non_interactive=False) -> Any:
    """
    Dynamically resolves and returns the best handler module for the given context.
    Uses context_coordinator's dynamic_state_county_detection as the primary source.
    Returns a dict with keys: 'handler', 'summary', 'log', 'error' (if any)
    """
    from .Context_Integration.context_coordinator import ContextCoordinator, dynamic_state_county_detection
    if not HANDLER_MAP["states"] or not HANDLER_MAP["counties_by_state"]:
        preload_handler_map()
    summary = {"attempts": [], "final": None, "error": None}
    log = []
    # Use preloaded handler map
    available_states = [normalize_state_name(s) for s in HANDLER_MAP["states"]]
    available_counties_by_state = {
        normalize_state_name(s): [normalize_county_name(c) for c in HANDLER_MAP["counties_by_state"].get(s, [])]
        for s in HANDLER_MAP["states"]
    }
    librarian_states = [normalize_state_name(s) for s in STATE_MODULE_MAP.keys()] if STATE_MODULE_MAP else []
    if debug:
        log_info(f"[Router] Available states (filesystem): {available_states}")
        log_info(f"[Router] Available states (context library): {librarian_states}")
        for s in available_states:
            log_info(f"[Router] Counties for state '{s}': {available_counties_by_state[s]}")
    # Step 1: Scan URL for clues first
    url_state, url_county, url_log = scan_url_for_state_county(url or context.get('url', ''), available_states, available_counties_by_state)
    for entry in url_log:
        log.append(entry)
        if debug:
            log_info(entry)
    # Step 2: Enrich context using the coordinator (NLP, ML, etc.)
    coordinator = ContextCoordinator(use_library=True, enable_ml=False, alert_monitor=False)
    enriched = coordinator.organize_and_enrich(context)
    html = context.get("raw_html", "") or (enriched.get("raw_html") if enriched else "")
    # Step 3: Use dynamic_state_county_detection for best guess (context, html)
    county, state, handler_path, detection_log = dynamic_state_county_detection(
        context, html, debug=True
    )
    for log_entry in detection_log:
        log.append(f"[Context Detection] {log_entry}")
        if debug:
            log_info(f"[Router] [Context Detection] {log_entry}")
    # Step 4: Decide on state/county using priority: URL > context > context library > filesystem
    valid_state = None
    valid_county = None
    fuzzy_cutoff = fuzzy_cutoff if fuzzy_cutoff is not None else FUZZY_MATCH_THRESHOLD
    # 1. Use URL scan if found
    if url_state:
        valid_state = normalize_state_name(url_state)
        if url_county:
            valid_county = normalize_county_name(url_county)
        summary["attempts"].append(f"URL scan: state={valid_state}, county={valid_county}")
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
    if debug:
        log_info(f"[Router] Available states (filesystem): {available_states}")
        log_info(f"[Router] Counties for state '{valid_state}': {available_counties_by_state.get(valid_state, [])}")               
    if valid_state and not valid_county and county:
        normalized_county = normalize_county_name(county)
        counties = available_counties_by_state.get(valid_state, [])
        if normalized_county in counties:
            valid_county = normalized_county
        else:
            # Check if county is a precinct of a known county (librarian)
            known_county_to_precinct = KNOWN_COUNTY_TO_PRECINCTS_MAP
            for county_name, precincts in known_county_to_precinct.items():
                if normalized_county in [normalize_county_name(d) for d in precincts]:
                    valid_county = normalize_county_name(county_name)
                    log.append(f"'{county}' matched as precinct of county '{county_name}'. Using '{county_name}'.")
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
        if handler and hasattr(handler, "parse"):
            log.append(f"Routed to handler: {handler_path}")
        else:
            log.append(f"Could not import handler or missing 'parse': {handler_path}")
            handler = None
    # Fallback to state-level handler if county handler not found
    if not handler and valid_state:
        fallback_path = f"webapp.parser.handlers.states.{valid_state}"
        attempted_paths.append(fallback_path)
        log.append(f"Attempting fallback to state handler: {fallback_path}")
        handler = import_handler(fallback_path)
        if handler and hasattr(handler, "parse"):
            log.append(f"Routed to fallback state handler: {fallback_path}")
        else:
            log.append(f"Could not import fallback state handler or missing 'parse': {fallback_path}")
            handler = None
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
        if os.path.isfile(test_input):
            with open(test_input, "rb") as f:
                try:
                    context = orjson.loads(f.read())
                except Exception as e:
                    print(f"Failed to load context from file: {e}")
                    return
        else:
            url = test_input
        result = get_handler(context or {}, url=url, debug=args.debug, fuzzy_cutoff=args.fuzzy_cutoff)
        print("Routing result:")
        print(orjson.dumps(result["summary"], option=orjson.OPT_INDENT_2))
        if result["handler"]:
            print(f"Handler module: {getattr(result['handler'], '__name__', str(result['handler']))}")
        else:
            print("No suitable handler found.")
            # --- Add fallback prompt here ---
            available_states = list_available_states()
            available_counties_by_state = {s: list_available_counties(s) for s in available_states}
            import_error_message = result["summary"]["error"]["message"] if result["summary"].get("error") else "Unknown error"
            state, county = prompt_for_handler_fallback(available_states, available_counties_by_state, last_error=import_error_message)
            if not state:
                print("No handler selected. Exiting.")
                return
            handler_path = f"webapp.parser.handlers.states.{state}.county.{county}" if county else f"webapp.parser.handlers.states.{state}"
            handler = import_handler(handler_path)
            if handler and hasattr(handler, "parse"):
                print(f"Handler module: {getattr(handler, '__name__', str(handler))}")
            else:
                print("Still could not import a suitable handler.")