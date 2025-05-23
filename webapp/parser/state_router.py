# state_router.py
# ===============================================
# Dynamically routes to the correct state or county-specific handler module
# Uses importlib for auto-resolution from folder structure.
# Also used by JSON, CSV, and PDF format handlers.
# Enhanced: Caching, reload-on-change, CLI utilities, batch tracking.
# ===============================================
import os
import importlib
import json
import time
from difflib import get_close_matches
from typing import Optional, Dict, Any, Set, List

from .utils.shared_logger import logger

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
LOADED_HANDLERS: Dict[str, Any] = {}
FUZZY_MATCH_CUTOFF = float(os.getenv("FUZZY_MATCH_CUTOFF", "0.7"))
if FUZZY_MATCH_CUTOFF < 0.5 or FUZZY_MATCH_CUTOFF > 1.0:
    logger.warning(f"[Router] Invalid FUZZY_MATCH_CUTOFF value: {FUZZY_MATCH_CUTOFF}. Using default 0.7.")
    FUZZY_MATCH_CUTOFF = 0.7

def get_hint_file_path() -> str:
    """Returns the absolute path to the url_hint_overrides.txt file."""
    return os.path.join(os.path.dirname(__file__), "url_hint_overrides.txt")

class UrlHintOverridesCache:
    """Caches and reloads url_hint_overrides.txt on change."""
    def __init__(self):
        self._path = get_hint_file_path()
        self._last_mtime = None
        self._cache: Dict[str, str] = {}

    def get(self) -> Dict[str, str]:
        try:
            mtime = os.path.getmtime(self._path)
            if self._last_mtime != mtime:
                with open(self._path, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
                self._last_mtime = mtime
                logger.info(f"[Router] Reloaded URL_HINT_OVERRIDES ({len(self._cache)} entries).")
        except FileNotFoundError:
            if self._cache:
                logger.warning("[Router] url_hint_overrides.txt disappeared, using last cached version.")
            else:
                logger.info("[Router] url_hint_overrides.txt not found — using default empty mapping.")
                self._cache = {}
        except Exception as e:
            logger.warning(f"[Router] Failed to load URL_HINT_OVERRIDES: {e}")
        return self._cache

URL_HINT_OVERRIDES_CACHE = UrlHintOverridesCache()

def import_handler(module_path: str):
    """Dynamically import and return a handler module from the given dotted path."""
    try:
        if module_path not in LOADED_HANDLERS:
            module = importlib.import_module(module_path)
            LOADED_HANDLERS[module_path] = module
        return LOADED_HANDLERS[module_path]
    except ModuleNotFoundError as e:
        logger.warning(f"[Router] Module not found: {module_path} — {e}")
        # Retry using closest matching submodule from parent package
        try:
            parent = module_path.rsplit(".", 1)[0]
            target_name = module_path.rsplit(".", 1)[-1]
            pkg = importlib.import_module(parent)
            possibilities = getattr(pkg, "__all__", dir(pkg))
            close = get_close_matches(target_name, possibilities, n=1, cutoff=0.6)
            if close:
                retry_path = f"{parent}.{close[0]}"
                logger.info(f"[Router] Retrying with closest match: {retry_path}")
                return importlib.import_module(retry_path)
        except Exception as inner:
            logger.debug(f"[Router] Retry suggestion failed: {inner}")
        return None

def resolve_state_handler(url_or_text: str):
    """Tries to match known state identifiers in a URL or HTML snippet to route to the appropriate handler module."""
    lower = url_or_text.lower()
    for state_abbr, state_key in STATE_MODULE_MAP.items():
        if state_abbr in lower:
            logger.info(f"[State Router] URL/text matched state '{state_abbr}' → {state_key}")
            module_path = f"handlers.states.{state_key}"
            module = import_handler(module_path)
            if module:
                return module

    url_hint_overrides = URL_HINT_OVERRIDES_CACHE.get()
    for pattern, module_path in url_hint_overrides.items():
        if pattern in lower:
            module = import_handler(module_path)
            if module:
                logger.info(f"[State Router] URL pattern '{pattern}' matched → {module_path}")
                return module

    logger.info("[State Router] No matching state-specific handler found.")
    return None

def get_handler(state_abbreviation: Optional[str], county_name: Optional[str] = None):
    """Dynamically loads and returns a handler based on state or state+county keys."""
    if not state_abbreviation:
        logger.warning("[Router] Missing state_abbreviation — skipping handler resolution.")
        return None

    normalized_state = state_abbreviation.strip().lower().replace(" ", "_")
    state_key = STATE_MODULE_MAP.get(normalized_state, normalized_state)
    module_path = f"handlers.states.{state_key}"

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
        return module
    else:
        logger.warning(f"[Router] Could not import module for state '{state_abbreviation}'")
        return None

def get_handler_from_context(context: Dict[str, Any]):
    """Extracts state and county info from a context dictionary and routes accordingly."""
    state = context.get("state")
    county = context.get("county")
    if not state and not county:
        logger.warning("[Router] No state or county provided in context.")
        return None

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
        logger.info(f"[Router] Handler resolved via get_handler for state '{state}' and county '{county}'")
        return handler

    # Fallback if fuzzy match + module lookup failed
    url_text = context.get("url", "") + " " + context.get("raw_text", "")
    fallback = resolve_state_handler(url_text)
    if fallback:
        logger.info("[Router] Fallback to resolve_state_handler succeeded.")
    else:
        logger.warning("[Router] No handler could be resolved even via fallback.")
    return fallback

# =========================
# CLI Utilities & Batch Mode
# =========================

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
        return []
    return sorted([d for d in os.listdir(base_path) if d.endswith(".py") and not d.startswith("__")])

def batch_run(states: Optional[List[str]] = None, counties: Optional[List[str]] = None):
    """
    Batch run handlers for a list of states and/or counties.
    Keeps track of which have already been processed.
    """
    processed: Set[str] = set()
    available_states = states or list_available_states()
    for state in available_states:
        state_key = STATE_MODULE_MAP.get(state, state)
        handler = get_handler(state_key)
        if not handler:
            logger.warning(f"[Batch] No handler found for state: {state}")
            continue
        if counties:
            for county in counties:
                key = f"{state}_{county}"
                if key in processed:
                    continue
                county_handler = get_handler(state_key, county)
                if county_handler:
                    logger.info(f"[Batch] Running handler for {state} - {county}")
                    # You can call a standard entry point here, e.g.:
                    # county_handler.parse(...)
                    processed.add(key)
        else:
            if state in processed:
                continue
            logger.info(f"[Batch] Running handler for {state}")
            # handler.parse(...)
            processed.add(state)

def cli():
    """Simple CLI for state_router utilities."""
    import argparse
    parser = argparse.ArgumentParser(description="State Router CLI Utility")
    parser.add_argument("--list-states", action="store_true", help="List all available state handlers")
    parser.add_argument("--list-counties", metavar="STATE", help="List all available county handlers for a state")
    parser.add_argument("--batch", nargs="*", metavar="STATE", help="Batch run handlers for given states")
    args = parser.parse_args()

    if args.list_states:
        print("Available states:")
        for state in list_available_states():
            print(f" - {state}")
    elif args.list_counties:
        print(f"Available counties for {args.list_counties}:")
        for county in list_available_counties(args.list_counties):
            print(f" - {county}")
    elif args.batch is not None:
        print("Batch running handlers...")
        batch_run(states=args.batch)
    else:
        parser.print_help()

if __name__ == "__main__":
    cli()