# shared_logic.py - Common parsing utilities for context-integrated pipeline
import os
import json
import re
import time
from ..utils.shared_logger import logger, rprint

CONTEXT_LIBRARY_PATH = os.path.join(
    os.path.dirname(__file__), "..", "Context_Integration", "context_library.json"
)
if os.path.exists(CONTEXT_LIBRARY_PATH):
    with open(CONTEXT_LIBRARY_PATH, "r", encoding="utf-8") as f:
        CONTEXT_LIBRARY = json.load(f)
    ALL_SELECTORS = ", ".join(CONTEXT_LIBRARY.get("selectors", {}).get("all_selectors", []))
else:
    ALL_SELECTORS = "h1, h2, h3, h4, h5, h6, strong, b, span, div"

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

def find_and_click_toggle(
    page,
    container=None,
    handler_selectors=None,
    handler_keywords=None,
    post_toggle_check=None,
    logger=None,
    verbose=False,
    max_attempts=1,
    wait_after_click=0,
    fallback_selectors=None,
    fallback_keywords=None
):
    """
    Attempts to click a toggle using handler-supplied selectors/keywords first.
    If not found, falls back to generic selectors/keywords.
    Returns True if a toggle was clicked and post_toggle_check passes (or table appears).
    """
    search_root = container if container else page

    # 1. Try handler-supplied selectors (most specific, fastest)
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

    if logger:
        logger.warning("[TOGGLE] No toggle found/clicked.")
    return False

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

def resume_after_toggle_or_selection(page, logger=None):
    """
    After a toggle or contest selection, try to find a table.
    If not found, try to parse a text block.
    Returns (headers, data) or (None, None) if nothing found.
    """
    table = page.query_selector("table")
    if table:
        from ..utils.table_builder import extract_table_data
        headers, data = extract_table_data(table)
        if logger:
            logger.info("[RESUME] Table found after toggle/selection.")
        return headers, data
    main_text = ""
    main_div = page.query_selector("main, .main-content, #main-content, body")
    if main_div:
        main_text = main_div.inner_text()
    else:
        main_text = page.inner_text()
    headers, data = parse_text_block_to_rows(main_text)
    if logger:
        logger.info("[RESUME] No table found. Parsed text block into rows.")
    return headers, data