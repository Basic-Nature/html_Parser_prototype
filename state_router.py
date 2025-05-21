# state_router.py
# ===============================================
# Dynamically routes to the correct state or county-specific handler module
# Uses importlib for auto-resolution from folder structure.
# Also used by JSON, CSV, and PDF format handlers.
# ===============================================
from utils.shared_logger import log_info, log_debug, log_warning
import os
import importlib

# Optional hardcoded URL hint map to bypass detection logic (loaded from external file)
import json

URL_HINT_OVERRIDES = {}
# Load URL hint overrides from a JSON file if it exists
# This file should contain a JSON object with key-value pairs of URL patterns and module paths
# Example: {"pattern1": "module.path1", "pattern2": "module.path2"}
# This allows for easy customization of URL routing without modifying the codebase. 
# The file should be in the same directory as this script.
try:
    with open("url_hint_overrides.txt", "r", encoding="utf-8") as f:
        URL_HINT_OVERRIDES = json.load(f)
        log_info(f"[Router] Loaded URL_HINT_OVERRIDES with {len(URL_HINT_OVERRIDES)} entries.")
except FileNotFoundError:
    log_warning("[Router] url_hint_overrides.txt not found — using default empty mapping.")
except Exception as e:
    log_warning(f"[Router] Failed to load URL_HINT_OVERRIDES: {e}")

# Mapping of state abbreviations (modern and traditional) to folder module names
STATE_MODULE_MAP = {
    # Modern then Traditional postal abbreviations
    "al": "alabama", "ala": "alabama",
    "ak": "alaska",
    "az": "arizona", "ariz": "arizona",
    "ar": "arkansas", "ark": "arkansas",
    "ca": "california", "calif": "california",
    "co": "colorado", "colo": "colorado",
    "ct": "connecticut", "conn": "connecticut",
    "de": "delaware", "del": "delaware",
    "dc": "district_of_columbia", "d.c.": "district_of_columbia",
    "fl": "florida", "fla": "florida",
    "ga": "georgia", "ga.": "georgia",
    "hi": "hawaii",
    "id": "idaho",
    "il": "illinois", "ill": "illinois",
    "in": "indiana", "ind": "indiana",
    "ia": "iowa",
    "ks": "kansas", "kans": "kansas",
    "ky": "kentucky", "ky.": "kentucky",
    "la": "louisiana", "la.": "louisiana",
    "me": "maine",
    "md": "maryland", "md.": "maryland",
    "ma": "massachusetts", "mass": "massachusetts",
    "mi": "michigan", "mich.": "michigan",
    "mn": "minnesota", "minn": "minnesota",
    "ms": "mississippi", "miss": "mississippi",
    "mo": "missouri", "mo.": "missouri",
    "mt": "montana", "mont": "montana",
    "ne": "nebraska", "nebr": "nebraska",
    "nv": "nevada", "nev": "nevada",
    "nh": "new_hampshire", "n.h.": "new_hampshire",
    "nj": "new_jersey", "n.j.": "new_jersey",
    "nm": "new_mexico", "n. mex.": "new_mexico",
    "ny": "new_york", "n.y.": "new_york",
    "nc": "north_carolina", "n.c.": "north_carolina",
    "nd": "north_dakota", "n. dak.": "north_dakota",
    "oh": "ohio",
    "ok": "oklahoma", "okla": "oklahoma",
    "or": "oregon", "ore": "oregon",
    "pa": "pennsylvania", "pa.": "pennsylvania",
    "ri": "rhode_island", "r.i.": "rhode_island",
    "sc": "south_carolina", "s.c.": "south_carolina",
    "sd": "south_dakota", "s. dak.": "south_dakota",
    "tn": "tennessee", "tenn": "tennessee",
    "tx": "texas", "tex": "texas",
    "ut": "utah",
    "vt": "vermont", "vt.": "vermont",
    "va": "virginia", "va.": "virginia",
    "wa": "washington", "wash": "washington",
    "wv": "west_virginia", "w. va.": "west_virginia",
    "wi": "wisconsin", "wis": "wisconsin",
    "wy": "wyoming", "wyo": "wyoming",
}


# Cache for already-imported handlers
LOADED_HANDLERS = {}

def import_handler(module_path):
    """
    Dynamically import and return a handler module from the given dotted path.
    Returns None if the module is not found.
    """
    try:
        if module_path not in LOADED_HANDLERS:
            module = importlib.import_module(module_path)
            LOADED_HANDLERS[module_path] = module
        return LOADED_HANDLERS[module_path]
    except ModuleNotFoundError as e:
        log_warning(f"[Router] Module not found: {module_path} — {e}")
        # Retry using closest matching submodule from parent package
        try:
            parent = module_path.rsplit(".", 1)[0]
            target_name = module_path.rsplit(".", 1)[-1]
            pkg = importlib.import_module(parent)
            possibilities = getattr(pkg, "__all__", dir(pkg))
            close = get_close_matches(target_name, possibilities, n=1, cutoff=0.6)
            if close:
                retry_path = f"{parent}.{close[0]}"
                log_info(f"[Router] Retrying with closest match: {retry_path}")
                return importlib.import_module(retry_path)
        except Exception as inner:
            log_debug(f"[Router] Retry suggestion failed: {inner}")
        return None

def resolve_state_handler(url_or_text):
    """
    Tries to match known state identifiers in a URL or HTML snippet
    to route to the appropriate handler module.
    """
    lower = url_or_text.lower()
    # Check for known state identifiers in the URL or text
    for state_abbr, state_key in STATE_MODULE_MAP.items():
        if state_abbr in lower:
            # Check if the state is a known abbreviation
            if state_abbr in STATE_MODULE_MAP:
                log_info(f"[INFO] [State Router] URL/text matched state '{state_abbr}' → {state_key}")
                module_path = f"handlers.states.{state_key}"
                module = import_handler(module_path)
                if module:
                    return module
            module_path = f"handlers.states.{state_key}"
            module = import_handler(module_path)
            if module:
                log_info(f"[INFO] [State Router] URL/text matched state '{state_abbr}' → {module_path}")
                return module
            

    for pattern, module_path in URL_HINT_OVERRIDES.items():
        if pattern in lower:
            module = import_handler(module_path)
            if module:
                log_info(f"[INFO] [State Router] URL pattern '{pattern}' matched → {module_path}")
                return module

    log_info("[INFO] [State Router] No matching state-specific handler found.")
    return None

from difflib import get_close_matches

# Fuzzy matching threshold (adjustable via .env)
FUZZY_MATCH_CUTOFF = float(os.getenv("FUZZY_MATCH_CUTOFF", "0.7"))
# Fallback to default if not set
if FUZZY_MATCH_CUTOFF < 0.5 or FUZZY_MATCH_CUTOFF > 1.0:
    log_warning(f"[Router] Invalid FUZZY_MATCH_CUTOFF value: {FUZZY_MATCH_CUTOFF}. Using default 0.7.")
    FUZZY_MATCH_CUTOFF = 0.7

def get_handler(state_abbreviation, county_name=None):
    """
    Dynamically loads and returns a handler based on state or state+county keys.
    Accepts state abbreviations and maps them to full folder names where needed.
    Prioritizes flat state-level handlers that implement their own parse() logic.
    """
    """
    Dynamically loads and returns a handler based on state or state+county keys.
    Accepts state abbreviations and maps them to full folder names where needed.
    """
    if not state_abbreviation:
        log_warning("[Router] Missing state_abbreviation — skipping handler resolution.")
        return None

    normalized_state = state_abbreviation.strip().lower().replace(" ", "_")
    # Check if the state is a known abbreviation
    if normalized_state in STATE_MODULE_MAP:
        log_info(f"[Router] State '{state_abbreviation}' is a known abbreviation.")
        return STATE_MODULE_MAP[normalized_state]
    # Check if the state is a known full name
    if normalized_state in STATE_MODULE_MAP.values():
        log_info(f"[Router] State '{state_abbreviation}' is a known full name.")
        return normalized_state
    # Check if the state is a known full name with spaces
    if normalized_state in [name.replace("_", " ") for name in STATE_MODULE_MAP.values()]:
        log_info(f"[Router] State '{state_abbreviation}' is a known full name with spaces.")
        return normalized_state
    # Check if the state is a known full name with dashes
    if normalized_state in [name.replace("-", "_") for name in STATE_MODULE_MAP.values()]:
        log_info(f"[Router] State '{state_abbreviation}' is a known full name with dashes.")
        return normalized_state
    # Check if the state is a known full name with underscores
    if normalized_state in [name.replace("_", "-") for name in STATE_MODULE_MAP.values()]:
        log_info(f"[Router] State '{state_abbreviation}' is a known full name with underscores.")
        return normalized_state
    # Check if the state is a known full name with periods
    if normalized_state in [name.replace(".", "_") for name in STATE_MODULE_MAP.values()]:
        log_info(f"[Router] State '{state_abbreviation}' is a known full name with periods.")
        return normalized_state 
    # Check if the state is a known full name with spaces and dashes
    if normalized_state in [name.replace(" ", "-") for name in STATE_MODULE_MAP.values()]:
        log_info(f"[Router] State '{state_abbreviation}' is a known full name with spaces and dashes.")
        return normalized_state
    # Check if the state is a known full name with spaces and underscores
    if normalized_state in [name.replace(" ", "_") for name in STATE_MODULE_MAP.values()]:
        log_info(f"[Router] State '{state_abbreviation}' is a known full name with spaces and underscores.")
        return normalized_state

    # Fuzzy match to closest known state abbreviation if needed
    if normalized_state not in STATE_MODULE_MAP:
        close = get_close_matches(normalized_state, STATE_MODULE_MAP.keys(), n=1, cutoff=FUZZY_MATCH_CUTOFF)
        if close:
            log_info(f"[Router] Fuzzy matched state '{state_abbreviation}' → '{close[0]}'")
            normalized_state = close[0]
        else:
            log_warning(f"[Router] No close match found for state '{state_abbreviation}'")
            return None
    state_key = STATE_MODULE_MAP.get(normalized_state, normalized_state)
    module_path = f"handlers.states.{state_key}"
    log_debug(f"[DEBUG] Module path resolved to: {module_path}")
    # Check if the module exists
    try:
        importlib.import_module(module_path)
    except ModuleNotFoundError:
        log_warning(f"[Router] Module '{module_path}' not found.")
        return None

    # First check if a top-level state handler has a parse() function
    try:
        state_module = import_handler(module_path)
        if hasattr(state_module, "parse"):
            if getattr(state_module, "county_mode", True) is False:
                log_info(f"[Router] county_mode = False on {module_path}. Using flat state handler.")
            log_info(f"[Router] Using top-level parse() from {module_path} — skipping county lookup.")
            return state_module
    except Exception as e:
        log_debug(f"[Router] State-level parse() check failed: {e}")

    if county_name:
        normalized_county = county_name.strip().lower().replace(" ", "_")
        composite_path = f"{module_path}.county.{normalized_county}"
        log_debug(f"[DEBUG] Attempting county-level handler: {composite_path}")

        # Optional: validate known counties in the state package
        try:
            county_pkg = importlib.import_module(f"{module_path}.county")
            valid_counties = getattr(county_pkg, "__all__", dir(county_pkg))
            if normalized_county not in valid_counties and normalized_county not in composite_path:
                log_warning(f"[Router] County '{normalized_county}' not found in declared modules for {state_key}")
        except ModuleNotFoundError:
            log_debug(f"[Router] No .county package found under {module_path}")
        except Exception as inner:
            log_debug(f"[Router] County validation error: {inner}")

        module = import_handler(composite_path)
        if module:
            log_info(f"[INFO] [State Router] Routed to handler for {state_key}_{normalized_county}")
            return module

    # Fallback to state-only handler
    module = import_handler(module_path)
    if module:
        log_info(f"[INFO] [State Router] Routed to fallback state-level handler for '{state_key}'")
        return module

    log_warning(f"[WARN] [State Router] No handler found for state '{state_key}' or county '{county_name}'")
    return None

def get_handler_from_context(context):
    """
    Extracts state and county info from a context dictionary and routes accordingly.
    Supports filename inference when no state is given.
    """
    """
    Extracts state and county info from a context dictionary and routes accordingly.
    Falls back to resolve_state_handler using the context URL/text if state fails.
    """
    """
    Extracts state and county info from a context dictionary and routes accordingly.
    Used by format handlers like JSON, CSV, and PDF to dynamically find state/county handlers.
    """
    state = context.get("state")
    county = context.get("county")
    if not state and not county:
        log_warning("[Router] No state or county provided in context.")
        return None

    if not state:
        filename = context.get("filename", "").lower()
        tokens = filename.replace("_", " ").split()
        for token in tokens:
            if token in STATE_MODULE_MAP:
                state = token
                context["state"] = token
                log_info(f"[Router] Inferred state '{state}' from filename: {filename}")
                break
        log_warning("[Router] No state provided in context.")
        return None

    handler = get_handler(state_abbreviation=state, county_name=county)
    if handler:
        log_info(f"[Router] Handler resolved via get_handler for state '{state}' and county '{county}'")
        return handler

    # Fallback if fuzzy match + module lookup failed
    url_text = context.get("url", "") + " " + context.get("raw_text", "")
    fallback = resolve_state_handler(url_text)
    if fallback:
        log_info("[Router] Fallback to resolve_state_handler succeeded.")
    else:
        log_warning("[Router] No handler could be resolved even via fallback.")
    return fallback
# End of state_router.py
