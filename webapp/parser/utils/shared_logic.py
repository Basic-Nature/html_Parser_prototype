# shared_logic.py - Common parsing utilities for context-integrated pipeline
from datetime import datetime
import difflib
import os
import json
import re
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, SpinnerColumn
import time
from ..utils.shared_logger import logger, rprint
from ..utils.user_prompt import prompt_user_input

CONTEXT_LIBRARY_PATH = os.path.join(
    os.path.dirname(__file__), "..", "Context_Integration", "context_library.json"
)
if os.path.exists(CONTEXT_LIBRARY_PATH):
    with open(CONTEXT_LIBRARY_PATH, "r", encoding="utf-8") as f:
        CONTEXT_LIBRARY = json.load(f)
    ALL_SELECTORS = ", ".join(CONTEXT_LIBRARY.get("selectors", {}).get("all_selectors", []))
else:
    ALL_SELECTORS = "h1, h2, h3, h4, h5, h6, strong, b, span, div"

# How to Use the Progress Bar Elsewhere
# for item in show_progress_bar("Processing items...", total=len(items), update_iter=items):
    # process item

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

def load_context_library():
    if os.path.exists(CONTEXT_LIBRARY_PATH):
        with open(CONTEXT_LIBRARY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_context_library(lib):
    with open(CONTEXT_LIBRARY_PATH, "w", encoding="utf-8") as f:
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

def find_and_click_toggle(
    page,
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
    context_title=None,  # New: contest or section title for proximity scoring
    domain_cache=None,   # New: dict to cache successful selectors per domain
):
    """
    Attempts to click a toggle using handler-supplied selectors/keywords first.
    If not found, falls back to generic selectors/keywords.
    Returns True if a toggle was clicked and post_toggle_check passes (or table appears).
    """
    search_root = container if container else page
    domain = page.url.split("/")[2] if "://" in page.url else page.url.split("/")[0]
    cached_selectors = [e["selector"] for e in get_domain_selectors(domain)]
    # Try cached selectors first
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

    # 1. Try handler-supplied selectors (most specific, fastest)
    if handler_selectors:
        for kw in handler_keywords:
            # Defensive: skip if kw is too long or contains newlines
            if not isinstance(kw, str) or len(kw) > 100 or "\n" in kw:
                if logger:
                    logger.warning(f"[TOGGLE] Skipping invalid keyword for selector: {repr(kw)}")
                continue
            elements = search_root.locator(f"*:has-text('{kw}')")
            for i in range(elements.count()):
                el = elements.nth(i)
                if el.is_visible() and el.is_enabled():
                    el.scroll_into_view_if_needed()
                    el.click()            
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

    # 2. Try handler-supplied keywords (text/aria-label)
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

    # 3. Fallback: Try generic selectors/keywords if provided
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
    # 2. Dynamic DOM scan for clickable elements
    clickable_selectors = [
        "button", "a", "[role=button]", "[onclick]", ".btn", ".toggle", ".expand"
    ]
    elements = []
    for sel in clickable_selectors:
        elements.extend(search_root.locator(sel).all())
    # Remove duplicates
    elements = list({el: None for el in elements}.keys())

    # 3. Score elements by text similarity to keywords and proximity to context_title
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

    # 4. Try clicking candidates in order of score
    for score, el, text in candidates[:max_attempts]:
        try:
            el.scroll_into_view_if_needed()
            el.click()
            if wait_after_click:
                page.wait_for_timeout(wait_after_click)
            if post_toggle_check and post_toggle_check(page):
                if logger:
                    logger.info(f"[TOGGLE] Clicked dynamic candidate: {text}")
                # Optionally cache selector for this domain
                if domain_cache is not None:
                    domain = page.url.split("/")[2]
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
        # List clickable elements for user
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
    wait_for_selector=None
):
    """
    Continuously scrolls a Playwright page until its scroll height stabilizes,
    or until max_total_time is reached. Optionally waits for a selector to appear.
    Returns True if stabilized or selector found, False if timed out.
    """
    logger.info("[SCROLL] Starting auto-scroll until page height stabilizes...")
    start_time = time.time()
    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(delay_ms)
    stable = 0
    last_height = 0
    while stable < max_stable_frames:
        current_height = page.evaluate("() => document.body.scrollHeight")
        if current_height == last_height:
            stable += 1
        else:
            stable = 0
        last_height = current_height
        page.evaluate(f"window.scrollBy(0, {step})")
        page.wait_for_timeout(delay_ms)
        if wait_for_selector and page.query_selector(wait_for_selector):
            logger.info(f"[SCROLL] Selector '{wait_for_selector}' found. Stopping scroll.")
            break
        if (time.time() - start_time) * 1000 > max_total_time:
            logger.warning("[SCROLL] Max scroll time exceeded. Stopping scroll.")
            break
    logger.info("[SCROLL] Completed scrolling until page height stabilized.")
    return True

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
            # Prompt user if scrolling takes too long
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
        # Coordinator feedback hook
        if coordinator_feedback:
            coordinator_feedback(domain, scroll_attempts, step)
        return True
    else:
        logger and logger.warning("[SCROLL] Max scroll time/attempts exceeded. Page may not be fully loaded.")
        if coordinator_feedback:
            coordinator_feedback(domain, scroll_attempts, step, incomplete=True)
        return False