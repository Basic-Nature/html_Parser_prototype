# shared_logic.py - Common parsing utilities for context-integrated pipeline

import difflib
import orjson
import os
import platform
import re

from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, SpinnerColumn

from ..utils.shared_logger import rprint, logger
from ..utils.user_prompt import prompt_user_input
from ..bots.librarian import STATE_ABBR, STATE_MODULE_MAP, KNOWN_STATE_TO_COUNTY_MAP, KNOWN_COUNTY_TO_PRECINCTS_MAP
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..Context_Integration.context_coordinator import ContextCoordinator

assert set(STATE_MODULE_MAP.keys()) == set(KNOWN_STATE_TO_COUNTY_MAP.keys()), \
    "STATE_MODULE_MAP and KNOWN_STATE_TO_COUNTY_MAP keys are out of sync!"

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
    state_map = STATE_MODULE_MAP
    county_map = KNOWN_STATE_TO_COUNTY_MAP
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
        state_norm = normalize_state_name(state)
        if state_norm not in county_map:
            logger.warning(f"State '{state_norm}' not found in county map")
        counties = county_map.get(state_norm, [])
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

def get_county_precincts(county_name):
    county_norm = normalize_county_name(county_name)
    return KNOWN_COUNTY_TO_PRECINCTS_MAP.get(county_norm)

def get_state_counties(state_name):
    state_norm = normalize_state_name(state_name)
    return KNOWN_STATE_TO_COUNTY_MAP.get(state_norm)

def scan_environment():
    return {
        "os": platform.system(),
        "os_version": platform.version(),
        "python_version": platform.python_version(),
        "cwd": os.getcwd()
    }

def get_title_embedding_features(contests, model_name="all-MiniLM-L6-v2"):
    from ..utils.model_registry import ModelRegistry
    model = ModelRegistry.get_sentence_transformer(model_name)
    titles = [c.get("title", "") for c in contests]
    return model.encode(titles, show_progress_bar=False)

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
    Continuously scrolls a Playwright page until its scroll height and visible content stabilize
    for at least 5 consecutive measurements, or until max_total_time is reached.
    Optionally waits for a selector to appear.
    Shows a dynamic progress bar using rich and prompts user if scrolling takes too long.
    Does NOT use or save any cached scroll pattern.
    """
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    import time

    console = Console()
    logger = logger or globals().get("logger", None)
    start_time = time.time()
    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(delay_ms)

    stable = 0
    last_heights = []
    last_texts = []
    scroll_attempts = 0
    max_scrolls = max_total_time // delay_ms
    domain = domain or (page.url.split("/")[2] if "://" in page.url else page.url.split("/")[0])

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
            last_heights.append(current_height)
            last_texts.append(current_text)
            if len(last_heights) > max_stable_frames:
                last_heights.pop(0)
                last_texts.pop(0)
            # Check if the last N heights and texts are all the same
            if (
                len(last_heights) == max_stable_frames
                and all(h == last_heights[0] for h in last_heights)
                and all(t == last_texts[0] for t in last_texts)
            ):
                stable += 1
            else:
                stable = 0
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
