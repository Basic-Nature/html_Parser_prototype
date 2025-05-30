# shared_logic.py - Common parsing utilities for context-integrated pipeline
from datetime import datetime
import difflib
import json
import os
import platform
import re
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, SpinnerColumn
import time
from ..utils.logger_instance import logger
from ..utils.shared_logger import rprint
from ..utils.user_prompt import prompt_user_input
from ..config import BASE_DIR
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..Context_Integration.context_coordinator import ContextCoordinator


def load_state_county_mappings():
    context_lib_path = os.path.join(
        os.path.dirname(__file__), "..", "Context_Integration", "context_library.json"
    )
    if os.path.exists(context_lib_path):
        with open(context_lib_path, "r", encoding="utf-8") as f:
            context_lib = json.load(f)
        state_map = context_lib.get("state_module_map", {})
        county_map = context_lib.get("Known_state_to_county_map", {})
        return state_map, county_map
    return {}, {}

STATE_ABBR = {
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
    "wy": "wyoming", "wyo": "wyoming"
}
def normalize_state_name(name):
    """
    Normalize state names and abbreviations to snake_case full state name.
    Handles abbreviations, full names, snake_case, and embedded state names in longer strings.
    E.g. 'ny', 'NY', 'New York', 'new york', 'new_york', 'ElecResultsFL.xls' -> 'new_york' or 'florida'
    """
    if not name:
        return None
    name = name.strip().lower().replace(" ", "_")
    # Try abbreviation lookup first
    if name in STATE_ABBR:
        return STATE_ABBR[name]
    # Try to match snake_case full name
    for full_name in STATE_ABBR.values():
        if name == full_name:
            return full_name
    # Try to match with spaces replaced by underscores
    for full_name in STATE_ABBR.values():
        if name.replace("_", " ") == full_name.replace("_", " "):
            return full_name
    # Try to find state abbreviation or name inside a longer string (e.g., filenames)
    for abbr, full_name in STATE_ABBR.items():
        pattern = r'\b' + re.escape(abbr) + r'\b'
        if re.search(pattern, name):
            return full_name
        pattern_snake = r'\b' + re.escape(full_name) + r'\b'
        if re.search(pattern_snake, name.replace("_", " ")):
            return full_name
    # Try to match state abbreviation at end of string (e.g., ElecResultsFL.xls)
    for abbr, full_name in STATE_ABBR.items():
        if name.endswith(abbr):
            return full_name
        if name.endswith("_" + abbr):
            return full_name
    return name

def normalize_county_name(name):
    """
    Normalize county names for comparison.
    Handles embedded county names, removes 'county' suffix, underscores, dashes, and extra spaces.
    E.g. 'Miami-Dade County', 'miami_dade-county', 'ResultsMiamiDadeCounty2024' -> 'miami dade'
    """
    if not name:
        return None
    name = name.lower().replace("_", " ").replace("-", " ").strip()
    # Remove 'county' suffix if present
    name = re.sub(r"\s+county$", "", name)
    name = re.sub(r"\s+", " ", name)
    # Try to extract county name from within a longer string (e.g., ResultsMiamiDadeCounty2024)
    match = re.search(r'([a-z ]+?)\s*county', name)
    if match:
        name = match.group(1).strip()
    # Remove any leading/trailing non-alpha chars
    name = re.sub(r"^[^a-z]+|[^a-z]+$", "", name)
    return name


def infer_state_county_from_url(url: str):
    """
    Robustly infer state and county from a URL using regex, mappings, and context library.
    Returns (state, county) or (None, None) if not found.
    """
    url = url.lower()
    url_norm = url.replace("-", "_").replace(" ", "_")
    state_map, county_map = load_state_county_mappings()
    IGNORED_TLDS = {
        "com", "org", "net", "gov", "edu", "co", "us", "info", "biz", "io", "me", "ca", "uk", "de", "fr", "jp"
    }
    state = None
    county = None

    # Try all state abbreviations and names (robust patterns)
    for abbr, name in STATE_ABBR.items():
        abbr_pattern = rf"/{abbr}(/|_|-|$)"
        name_repl = name.replace(' ', '[_\\-_]?')
        name_pattern = rf"/{name_repl}(/|_|-|$)"
        if re.search(abbr_pattern, url_norm) or re.search(name_pattern, url_norm):
            state = name
            break

    # Try mapping from context library
    if not state and state_map:
        for key in state_map:
            key_repl = key.replace(' ', '[_\\-_]?')
            key_pattern = rf"/{key_repl}(/|_|-|$)"
            mapped_repl = state_map[key].replace(' ', '[_\\-_]?')
            mapped_pattern = rf"/{mapped_repl}(/|_|-|$)"
            if re.search(key_pattern, url_norm) or re.search(mapped_pattern, url_norm):
                state = key
                break

    # Fuzzy match as last resort, but skip TLDs and common suffixes
    if not state:
        all_states = set(list(STATE_ABBR.values()) + list(state_map.keys()) + list(STATE_ABBR.keys()))
        url_parts = re.split(r'[/_.\-]', url_norm)
        url_parts = [part for part in url_parts if part and part not in IGNORED_TLDS]
        for part in url_parts:
            matches = difflib.get_close_matches(part, all_states, n=1, cutoff=0.8)
            if matches:
                match = matches[0]
                # If match is an abbreviation, convert to full name
                state = STATE_ABBR.get(match, match)
                break

    # --- 2. Try to match county (only if state is found) ---
    if state:
        counties = county_map.get(state, [])
        counties_norm = [normalize_county_name(c) for c in counties]
        # Try to match "-county" or "_county" in URL
        county_match = re.search(r'/([a-z0-9_\-]+)[-_]?county', url_norm)
        if county_match:
            county_candidate = normalize_county_name(county_match.group(1))
            # Exact or fuzzy match
            if county_candidate in counties_norm:
                county = counties[counties_norm.index(county_candidate)]
            else:
                matches = difflib.get_close_matches(county_candidate, counties_norm, n=1, cutoff=0.7)
                if matches:
                    county = counties[counties_norm.index(matches[0])]
        # Try to match county names directly in URL
        if not county:
            for i, c_norm in enumerate(counties_norm):
                if c_norm and c_norm in url_norm:
                    county = counties[i]
                    break

    # Normalize before returning
    if state:
        state = normalize_state_name(state)
    if county:
        county = normalize_county_name(county)

    return state, county


def scan_environment():
    return {
        "os": platform.system(),
        "os_version": platform.version(),
        "python_version": platform.python_version(),
        "cwd": os.getcwd()
    }

def get_title_embedding_features(contests, model_name="all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    titles = [c.get("title", "") for c in contests]
    return model.encode(titles)

def show_progress_bar(task_desc, total, update_iter):
    with Progress(
        SpinnerColumn(style="bold cyan"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None, style="bold cyan"),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task(task_desc, total=total)
        for n in update_iter:
            progress.update(task, advance=1)
            yield n

def coordinator_feedback(domain, scrolls, step, incomplete=False):
    logger.info(f"[COORDINATOR] Scroll pattern for {domain}: {scrolls} scrolls, step {step}, incomplete={incomplete}")

def normalize_text(text):
    return re.sub(r"\s+", " ", text.strip().lower())

def match_any(label, keywords):
    label = normalize_text(label)
    return any(k.lower() in label for k in keywords)

def build_csv_headers(rows):
    headers = set()
    for row in rows:
        headers.update(row.keys())
    return sorted(headers)

def safe_join(base, *paths):
    final_path = os.path.abspath(os.path.join(base, *paths))
    if not final_path.startswith(os.path.abspath(base)):
        print(f"DEBUG: Attempted to join {paths} to base {base} -> {final_path}")
        raise ValueError("Attempted Path Traversal Detected!")
    return final_path

def load_context_library(path=None):
    if path is None:
        from ..Context_Integration.context_organizer import CONTEXT_LIBRARY_PATH
        path = CONTEXT_LIBRARY_PATH
    safe_path = safe_join(BASE_DIR, os.path.relpath(path, BASE_DIR))
    if not os.path.exists(safe_path):
        return {}
    with open(safe_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_context_library(lib, path=None):
    if path is None:
        from ..Context_Integration.context_organizer import CONTEXT_LIBRARY_PATH
        path = CONTEXT_LIBRARY_PATH
    safe_path = safe_join(BASE_DIR, os.path.relpath(path, BASE_DIR))
    with open(safe_path, "w", encoding="utf-8") as f:
        json.dump(lib, f, indent=2, ensure_ascii=False)

def update_domain_selector_cache(domain, selector, label, success=True):
    lib = load_context_library()
    lib.setdefault("domain_selectors", {})
    entry = {
        "selector": selector,
        "label": label,
        "success_count": 1 if success else 0,
        "last_used": datetime.utcnow().isoformat(),
    }
    found = False
    for e in lib["domain_selectors"].get(domain, []):
        if e["selector"] == selector:
            e["success_count"] += 1 if success else 0
            e["last_used"] = entry["last_used"]
            found = True
            break
    if not found:
        lib["domain_selectors"].setdefault(domain, []).append(entry)
    save_context_library(lib)

def get_domain_selectors(domain):
    lib = load_context_library()
    return lib.get("domain_selectors", {}).get(domain, [])

# --- ContextCoordinator Integration ---

def get_contextual_buttons(coordinator, contest_title=None, keywords=None):
    """
    Use ContextCoordinator to get the best button(s) for a contest, optionally filtered by keywords.
    """
    if not coordinator or not hasattr(coordinator, "get_best_button"):
        return []
    return coordinator.get_best_button(contest_title, keywords=keywords)

def get_contextual_selectors(coordinator, contest_title=None):
    """
    Use ContextCoordinator to get selectors for a contest or globally.
    """
    if not coordinator or not hasattr(coordinator, "get_for_html_handler"):
        return []
    selectors = coordinator.get_for_html_handler().get("all_selectors", [])
    return selectors

def get_contextual_contests(coordinator, filters=None):
    """
    Use ContextCoordinator to get contests, optionally filtered.
    """
    if not coordinator or not hasattr(coordinator, "get_contests"):
        return []
    return coordinator.get_contests(filters=filters)

def get_precinct_headers_from_coordinator(coordinator):
    """
    Use ContextCoordinator to get precinct headers for table parsing.
    """
    if not coordinator or not hasattr(coordinator, "get_for_table_builder"):
        return []
    return coordinator.get_for_table_builder().get("precinct_headers", [])

# Example usage in your pipeline:
# coordinator = ContextCoordinator()
# contests = get_contextual_contests(coordinator)
# selectors = get_contextual_selectors(coordinator)
# find_and_click_toggle(page, coordinator=coordinator, ...)

# --- Button/Toggle Logic (Context-Driven) ---

def find_and_click_toggle(
    page,
    coordinator: "ContextCoordinator",
    container=None,
    handler_selectors=None,
    handler_keywords=None,
    post_toggle_check=None,
    logger=None,
    verbose=False,
    max_attempts=3,
    wait_after_click=0,
    fallback_selectors=None,
    fallback_keywords=None,
    context_title=None,
    domain_cache=None,
):
    """
    Attempts to click a toggle using handler/coordinator-supplied selectors/keywords first.
    Returns True if a toggle was clicked and post_toggle_check passes (or table appears).
    """
    search_root = container if container else page
    domain = page.url.split("/")[2] if "://" in page.url else page.url.split("/")[0]

    # 1. Use coordinator for selectors/keywords if available
    if coordinator:
        ctx_selectors = get_contextual_selectors(coordinator, contest_title=context_title)
        if ctx_selectors:
            for selector in ctx_selectors:
                elements = search_root.locator(selector)
                for i in range(elements.count()):
                    el = elements.nth(i)
                    if el.is_visible() and el.is_enabled():
                        el.scroll_into_view_if_needed()
                        el.click()
                        if wait_after_click:
                            page.wait_for_timeout(wait_after_click)
                        if post_toggle_check and post_toggle_check(page):
                            if logger:
                                logger.info(f"[TOGGLE] Clicked coordinator selector: {selector}")
                            return True
                        if page.query_selector("table"):
                            if logger:
                                logger.info(f"[TOGGLE] Table found after coordinator selector: {selector}")
                            return True

    # Try cached selectors first
    cached_selectors = [e["selector"] for e in get_domain_selectors(domain)]
    for selector in cached_selectors:
        elements = search_root.locator(selector)
        for i in range(elements.count()):
            el = elements.nth(i)
            if el.is_visible() and el.is_enabled():
                el.scroll_into_view_if_needed()
                el.click()
                if wait_after_click:
                    page.wait_for_timeout(wait_after_click)
                if post_toggle_check and post_toggle_check(page):
                    update_domain_selector_cache(domain, el.selector, el.inner_text(), success=True)
                    if logger:
                        logger.info(f"[TOGGLE] Clicked cached selector: {selector}")
                    return True
                if page.query_selector("table"):
                    update_domain_selector_cache(domain, el.selector, el.inner_text(), success=True)
                    if logger:
                        logger.info(f"[TOGGLE] Table found after cached selector: {selector}")
                    return True

    # 2. Try handler-supplied selectors (most specific, fastest)
    if handler_selectors:
        for selector in handler_selectors:
            elements = search_root.locator(selector)
            for i in range(elements.count()):
                el = elements.nth(i)
                if el.is_visible() and el.is_enabled():
                    el.scroll_into_view_if_needed()
                    el.click()
                    if wait_after_click:
                        page.wait_for_timeout(wait_after_click)
                    if post_toggle_check and post_toggle_check(page):
                        if logger:
                            logger.info(f"[TOGGLE] Clicked handler selector: {selector}")
                        return True
                    if page.query_selector("table"):
                        if logger:
                            logger.info(f"[TOGGLE] Table found after handler selector: {selector}")
                        return True

    # 3. Try handler-supplied keywords (text/aria-label)
    if handler_keywords:
        for kw in handler_keywords:
            elements = search_root.locator(f"*:has-text('{kw}')")
            for i in range(elements.count()):
                el = elements.nth(i)
                if el.is_visible() and el.is_enabled():
                    el.scroll_into_view_if_needed()
                    el.click()
                    if wait_after_click:
                        page.wait_for_timeout(wait_after_click)
                    if post_toggle_check and post_toggle_check(page):
                        if logger:
                            logger.info(f"[TOGGLE] Clicked handler keyword: {kw}")
                        return True
                    if page.query_selector("table"):
                        if logger:
                            logger.info(f"[TOGGLE] Table found after handler keyword: {kw}")
                        return True

    # 4. Fallback: Try generic selectors/keywords if provided
    if fallback_selectors:
        for selector in fallback_selectors:
            elements = search_root.locator(selector)
            for i in range(elements.count()):
                el = elements.nth(i)
                if el.is_visible() and el.is_enabled():
                    el.scroll_into_view_if_needed()
                    el.click()
                    if wait_after_click:
                        page.wait_for_timeout(wait_after_click)
                    if post_toggle_check and post_toggle_check(page):
                        if logger:
                            logger.info(f"[TOGGLE] Clicked fallback selector: {selector}")
                        return True
                    if page.query_selector("table"):
                        if logger:
                            logger.info(f"[TOGGLE] Table found after fallback selector: {selector}")
                        return True

    if fallback_keywords:
        for kw in fallback_keywords:
            elements = search_root.locator(f"*:has-text('{kw}')")
            for i in range(elements.count()):
                el = elements.nth(i)
                if el.is_visible() and el.is_enabled():
                    el.scroll_into_view_if_needed()
                    el.click()
                    if wait_after_click:
                        page.wait_for_timeout(wait_after_click)
                    if post_toggle_check and post_toggle_check(page):
                        if logger:
                            logger.info(f"[TOGGLE] Clicked fallback keyword: {kw}")
                        return True
                    if page.query_selector("table"):
                        if logger:
                            logger.info(f"[TOGGLE] Table found after fallback keyword: {kw}")
                        return True

    # 5. Dynamic DOM scan for clickable elements
    clickable_selectors = [
        "button", "a", "[role=button]", "[onclick]", ".btn", ".toggle", ".expand"
    ]
    elements = []
    for sel in clickable_selectors:
        elements.extend(search_root.locator(sel).all())
    elements = list({el: None for el in elements}.keys())

    # 6. Score elements by text similarity to keywords and proximity to context_title
    candidates = []
    for el in elements:
        try:
            text = el.inner_text().strip()
            score = 0
            if handler_keywords:
                matches = difflib.get_close_matches(text.lower(), [k.lower() for k in handler_keywords], n=1, cutoff=0.6)
                if matches:
                    score += 10
            if context_title and context_title.lower() in text.lower():
                score += 5
            if el.is_visible() and el.is_enabled():
                score += 2
            candidates.append((score, el, text))
        except Exception:
            continue
    candidates.sort(reverse=True, key=lambda x: x[0])

    # 7. Try clicking candidates in order of score
    for score, el, text in candidates[:max_attempts]:
        try:
            el.scroll_into_view_if_needed()
            el.click()
            if wait_after_click:
                page.wait_for_timeout(wait_after_click)
            if post_toggle_check and post_toggle_check(page):
                if logger:
                    logger.info(f"[TOGGLE] Clicked dynamic candidate: {text}")
                if domain_cache is not None:
                    domain_cache.setdefault(domain, []).append(el.selector)
                return True
            if page.query_selector("table"):
                if logger:
                    logger.info(f"[TOGGLE] Table found after dynamic candidate: {text}")
                return True
        except Exception as e:
            if logger:
                logger.warning(f"[TOGGLE] Failed to click candidate: {text} ({e})")
            continue

    if logger:
        logger.warning("[TOGGLE] No toggle found/clicked after dynamic scan.")
    elements = []
    for sel in clickable_selectors:
        elements.extend(search_root.locator(sel).all())
    elements = list({el: None for el in elements}.keys())
    choices = []
    for i, el in enumerate(elements):
        try:
            text = el.inner_text().strip()
            choices.append(f"[{i}] {text} ({el.selector})")
        except Exception:
            continue

    if choices:
        rprint("\n[bold yellow]Manual toggle selection required. Choose an element:[/bold yellow]")
        for c in choices:
            rprint(c)
        selection = prompt_user_input("Enter index of element to click (or blank to skip): ").strip()
        if selection.isdigit():
            idx = int(selection)
            if 0 <= idx < len(elements):
                el = elements[idx]
                el.scroll_into_view_if_needed()
                el.click()
                update_domain_selector_cache(domain, el.selector, el.inner_text(), success=True)
                log_selector_attempt(domain, el.selector, el.inner_text(), True)
                if logger:
                    logger.info(f"[TOGGLE] Clicked user-selected element: {el.selector}")
                return True
    return False

def log_selector_attempt(domain, selector, label, success):
    lib = load_context_library()
    lib.setdefault("selector_attempts", [])
    lib["selector_attempts"].append({
        "domain": domain,
        "selector": selector,
        "label": label,
        "success": success,
        "timestamp": datetime.utcnow().isoformat()
    })
    save_context_library(lib)

def autoscroll_until_stable(
    page,
    max_stable_frames=5,
    step=8000,
    delay_ms=200,
    max_total_time=10000,
    wait_for_selector=None,
    domain=None,
    logger=None,
    coordinator_feedback=None,
):
    """
    Continuously scrolls a Playwright page until its scroll height and visible content stabilize,
    or until max_total_time is reached. Optionally waits for a selector to appear.
    Shows a dynamic progress bar and prompts user if scrolling takes too long.
    Learns scroll patterns per domain for future runs.
    """
    from rich.console import Console
    console = Console()
    logger = logger or globals().get("logger", None)
    start_time = time.time()
    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(delay_ms)
    stable = 0
    last_height = 0
    last_text = ""
    scroll_attempts = 0
    max_scrolls = max_total_time // delay_ms
    domain = domain or (page.url.split("/")[2] if "://" in page.url else page.url.split("/")[0])

    # Try to load scroll pattern from cache
    lib = load_context_library()
    domain_scrolls = lib.get("domain_scrolls", {}).get(domain, None)
    if domain_scrolls:
        max_scrolls = domain_scrolls.get("max_scrolls", max_scrolls)
        step = domain_scrolls.get("step", step)
        logger and logger.info(f"[SCROLL] Loaded scroll pattern for {domain}: max_scrolls={max_scrolls}, step={step}")

    def get_main_text():
        try:
            main_div = page.query_selector("main, .main-content, #main-content, body")
            return main_div.inner_text() if main_div else page.inner_text()
        except Exception:
            return ""

    with Progress(
        SpinnerColumn(style="bold cyan"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None, style="bold cyan"),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        transient=True,
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Scrolling page...", total=max_scrolls)
        while stable < max_stable_frames and scroll_attempts < max_scrolls:
            current_height = page.evaluate("() => document.body.scrollHeight")
            current_text = get_main_text()
            if current_height == last_height and current_text == last_text:
                stable += 1
            else:
                stable = 0
            last_height = current_height
            last_text = current_text
            page.evaluate(f"window.scrollBy(0, {step})")
            page.wait_for_timeout(delay_ms)
            scroll_attempts += 1
            progress.update(task, advance=1)
            if wait_for_selector and page.query_selector(wait_for_selector):
                logger and logger.info(f"[SCROLL] Selector '{wait_for_selector}' found. Stopping scroll.")
                break
            elapsed = (time.time() - start_time) * 1000
            if elapsed > max_total_time * 0.8 and scroll_attempts % 10 == 0:
                console.print("[bold yellow]Scrolling is taking longer than expected. Continue waiting? (y/N)[/bold yellow]")
                resp = prompt_user_input("Continue scrolling? (y/N): ").strip().lower()
                if resp != "y":
                    logger and logger.warning("[SCROLL] User aborted scrolling.")
                    break
        progress.update(task, completed=max_scrolls)

    # Save scroll pattern for this domain
    lib.setdefault("domain_scrolls", {})
    lib["domain_scrolls"][domain] = {
        "max_scrolls": scroll_attempts,
        "step": step,
        "last_used": datetime.utcnow().isoformat(),
    }
    save_context_library(lib)

    if stable >= max_stable_frames:
        logger and logger.info("[SCROLL] Completed scrolling until page height/content stabilized.")
        if coordinator_feedback:
            coordinator_feedback(domain, scroll_attempts, step)
        return True
    else:
        logger and logger.warning("[SCROLL] Max scroll time/attempts exceeded. Page may not be fully loaded.")
        if coordinator_feedback:
            coordinator_feedback(domain, scroll_attempts, step, incomplete=True)
        return False

def parse_text_block_to_rows(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    contests = []
    current_contest = None
    current_candidates = []

    for line in lines:
        if any(kw in line.lower() for kw in ["president", "senate", "congress", "governor"]):
            if current_contest and current_candidates:
                contests.append((current_contest, current_candidates))
            current_contest = line
            current_candidates = []
        else:
            if current_contest:
                current_candidates.append(line)
    if current_contest and current_candidates:
        contests.append((current_contest, current_candidates))

    rows = []
    headers = set(["Contest"])
    for contest, candidates in contests:
        i = 0
        while i < len(candidates):
            row = {"Contest": contest}
            row["Candidate"] = candidates[i]
            i += 1
            for field in ["Party", "Percentage", "Votes"]:
                if i < len(candidates):
                    val = candidates[i]
                    if "%" in val:
                        row["Percentage"] = val
                    elif val.replace(",", "").isdigit():
                        row["Votes"] = val
                    elif any(x in val.lower() for x in ["democratic", "republican", "conservative", "working", "families", "write-in"]):
                        row["Party"] = val
                    i += 1
            headers.update(row.keys())
            rows.append(row)
    return list(headers), rows

def scan_buttons_with_progress(buttons, scan_callback=None):
    """
    Scan a list of buttons with a single-line progress bar.
    Optionally, provide a scan_callback(button, idx) for custom logic.
    """
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("Scanning buttons...", total=len(buttons))
        for idx, btn in enumerate(buttons):
            label = ""
            try:
                label = btn.inner_text()[:60]
            except Exception:
                label = str(btn)[:60]
            progress.update(task, advance=1, description=f"Scanning: {label}")
            if scan_callback:
                scan_callback(btn, idx)

def keyphrase_match(label, keyphrase, min_words=2, fuzzy_cutoff=0.8):
    """
    Returns True if the label matches the keyphrase as a whole (regex or fuzzy),
    or if at least min_words from the keyphrase are present in the label.
    """
    label_norm = label.lower().strip()
    keyphrase_norm = keyphrase.lower().strip()
    # 1. Try full phrase regex (allowing whitespace, punctuation, : or \n at end)
    pattern = re.sub(r"\s+", r"\\s+", re.escape(keyphrase_norm)) + r"[\s:]*$"
    if re.search(pattern, label_norm):
        return True
    # 2. Try fuzzy full phrase
    if difflib.SequenceMatcher(None, label_norm, keyphrase_norm).ratio() >= fuzzy_cutoff:
        return True
    # 3. Require at least min_words from keyphrase to be present
    words = [w for w in re.split(r"\W+", keyphrase_norm) if w]
    matches = sum(1 for w in words if w in label_norm)
    if len(words) >= min_words and matches >= min_words:
        return True
    return False