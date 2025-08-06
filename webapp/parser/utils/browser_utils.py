# utils/browser_utils.py
# ---------------------------------------------------------------
# Handles launching the Playwright or SeleniumBase browser and applying stealth
# options and user-agent rotation. Handles recovery from headless mode if CAPTCHA
# interaction is required, and can relaunch into stealth mode after persistent CAPTCHA.
# ---------------------------------------------------------------
from __future__ import annotations
import os
import random
import time
import inspect
import asyncio
from typing import (
    Protocol, Optional, Tuple, Union, Collection, Any
)
from flask import Response
from playwright.sync_api import (
    sync_playwright, Browser, BrowserContext, Page, BrowserType, ElementHandle, Locator
)
from selenium.webdriver.remote.webdriver import WebDriver
from .logger_singleton import logger, console, prompt
from .shared_logic import (
    safe_lower, safe_get, safe_get_first
)
from ..config import CONTEXT_LIBRARY_PATH

# Load user agents and captcha indicators from context library
if os.path.exists(CONTEXT_LIBRARY_PATH):
    import orjson
    with open(CONTEXT_LIBRARY_PATH, "rb") as f:
        CONTEXT_LIBRARY = orjson.loads(f.read())

    def get_list(context: dict, key: str) -> list:
        value = context.get(key, [])
        if isinstance(value, list):
            return value
        # Try to coerce from stringified list or other types
        if isinstance(value, str):
            try:
                import ast
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
        return []

    USER_AGENTS = get_list(CONTEXT_LIBRARY, "user_agents")
    CLOUDFLARE_CAPTCHA_INDICATORS = get_list(CONTEXT_LIBRARY, "cloudflare_captcha_indicators")
else:
    logger.error("[browser_utils] context_library.json not found. User agent rotation will be limited.")
    USER_AGENTS = []
    CLOUDFLARE_CAPTCHA_INDICATORS = []

class Closable(Protocol):
    def close(self) -> None:
        """Method to close the object, typically a browser or context."""
        ...

def safe_url(page) -> str:
    """Safely get the URL from a Playwright page object."""
    try:
        url = getattr(page, "url", "")
        return str(url) if isinstance(url, str) else ""
    except Exception as e:
        logger.error(f"[safe_url] Error accessing page.url: {e}")
        return ""

def safe_is_visible(obj: Union[Locator, ElementHandle], logger=logger) -> bool:
    """Safely call .is_visible on a Playwright element handle or locator."""
    try:
        if hasattr(obj, "is_visible"):
            return obj.is_visible()
        return False
    except Exception as e:
        if logger: logger.error(f"[safe_is_visible] Error: {e}")
        return False

def safe_is_enabled(obj: Union[Locator, ElementHandle], logger=logger) -> bool:
    """Safely call .is_enabled on a Playwright element handle or locator."""
    try:
        if hasattr(obj, "is_enabled"):
            return obj.is_enabled()
        return False
    except Exception as e:
        if logger: logger.error(f"[safe_is_enabled] Error: {e}")
        return False

def safe_locator(page: Page, selector: str, logger=logger) -> Optional[Locator]:
    """Safely call .locator on a Playwright page."""
    try:
        if hasattr(page, "locator"):
            return page.locator(selector)
        return None
    except Exception as e:
        if logger: logger.error(f"[safe_locator] Error: {e}")
        return None

def safe_count(obj: Union[Locator, Collection], logger=logger) -> int:
    """Safely call .count() on a locator or collection."""
    try:
        if hasattr(obj, "count"):
            return obj.count()
        if hasattr(obj, "__len__"):
            return len(obj)
        return 0
    except Exception as e:
        if logger: logger.error(f"[safe_count] Error: {e}")
        return 0

def safe_evaluate(obj: Union[Locator, ElementHandle], script: str, logger=logger) -> Any:
    """Safely call .evaluate on a Playwright element handle."""
    try:
        if hasattr(obj, "evaluate"):
            return obj.evaluate(script)
        return None
    except Exception as e:
        if logger: logger.error(f"[safe_evaluate] Error: {e}")
        return None

def safe_is_visible(obj: Union[Locator, ElementHandle], logger=logger) -> bool:
    """Safely call .is_visible on a Playwright element handle."""
    try:
        if hasattr(obj, "is_visible"):
            return obj.is_visible()
        return False
    except Exception as e:
        if logger: logger.error(f"[safe_is_visible] Error: {e}")
        return False

def safe_is_enabled(obj: Union[Locator, ElementHandle], logger=logger) -> bool:
    """Safely call .is_enabled on a Playwright element handle."""
    try:
        if hasattr(obj, "is_enabled"):
            return obj.is_enabled()
        return False
    except Exception as e:
        if logger: logger.error(f"[safe_is_enabled] Error: {e}")
        return False

def safe_click(obj: Union[Locator, ElementHandle], logger=logger) -> bool:
    """Safely call .click on a Playwright element handle."""
    try:
        if hasattr(obj, "click"):
            obj.click()
            return True
        return False
    except Exception as e:
        if logger: logger.error(f"[safe_click] Error: {e}")
        return False

def safe_wait_for_timeout(page: Page, ms: int, logger=logger) -> bool:
    """Safely call .wait_for_timeout on a Playwright page."""
    try:
        if hasattr(page, "wait_for_timeout"):
            page.wait_for_timeout(ms)
            return True
        return False
    except Exception as e:
        if logger: logger.error(f"[safe_wait_for_timeout] Error: {e}")
        return False

def safe_get_attribute(obj: Union[Locator, ElementHandle], attr: str, logger=logger) -> Optional[str]:
    """Safely call .get_attribute on a Playwright element handle."""
    try:
        if hasattr(obj, "get_attribute"):
            return obj.get_attribute(attr)
        return None
    except Exception as e:
        if logger: logger.error(f"[safe_get_attribute] Error: {e}")
        return None

def safe_attributes(element) -> dict:
    """
    Safely get the attributes dictionary from a selectolax element.
    Returns an empty dict if not available.
    """
    try:
        attrs = getattr(element, "attributes", {})
        if isinstance(attrs, dict):
            return attrs
        return {}
    except Exception:
        return {}

def safe_inner_text(obj: Union[Locator, ElementHandle], logger=logger) -> str:
    """Safely call .inner_text on a Playwright element handle."""
    try:
        if hasattr(obj, "inner_text"):
            return obj.inner_text()
        return ""
    except Exception as e:
        if logger: logger.error(f"[safe_inner_text] Error: {e}")
        return ""

def safe_nth(obj: Union[Locator, ElementHandle], index: int, logger=logger) -> Optional[Union[Locator, ElementHandle]]:
    """Safely call .nth on a Playwright locator."""
    try:
        if hasattr(obj, "nth"):
            return obj.nth(index)
        # Fallback for lists
        if isinstance(obj, (list, tuple)) and 0 <= index < len(obj):
            return obj[index]
        return None
    except Exception as e:
        if logger: logger.error(f"[safe_nth] Error: {e}")
        return None

def safe_query_selector_all(page: Page, selector: str, session_id: Optional[str] = None) -> list[ElementHandle]:
    try:
        if hasattr(page, "query_selector_all") and callable(page.query_selector_all):
            return page.query_selector_all(selector)
        else:
            logger.error(f"[SAFE] page does not support query_selector_all. (Session: {session_id})")
            return []
    except Exception as e:
        logger.error(f"[SAFE] Exception during query_selector_all: {e} (Session: {session_id})")
        return []

def safe_context_library(page, session_id=None):
    try:
        if hasattr(page, "context_library"):
            lib = getattr(page, "context_library")
            if isinstance(lib, dict):
                return lib
        return {}
    except Exception as e:
        logger.error(f"[SAFE] Exception accessing context_library: {e} (Session: {session_id})")
        return {}

def safe_context_result(page: Page, session_id: Optional[str] = None) -> dict:
    try:
        if hasattr(page, "context_result"):
            result = getattr(page, "context_result")
            if isinstance(result, dict):
                return result
        return {}
    except Exception as e:
        logger.error(f"[SAFE] Exception accessing context_result: {e} (Session: {session_id})")
        return {}

def safe_chromium(playwright: sync_playwright, session_id: Optional[str] = None) -> Optional[BrowserType]:
    try:
        browser_type = getattr(playwright, "chromium", None)
        if browser_type is None:
            logger.error(f"[SAFE] Playwright has no 'chromium' attribute. (Session: {session_id})")
        return browser_type
    except Exception as e:
        logger.error(f"[SAFE] Exception accessing 'chromium': {e} (Session: {session_id})")
        return None

def safe_launch(browser_type: Optional[BrowserType], headless: bool = True, args: Optional[list] = None, session_id: Optional[str] = None) -> Optional[Browser]:
    try:
        if browser_type is None:
            logger.error(f"[SAFE] browser_type is None, cannot launch. (Session: {session_id})")
            return None
        return browser_type.launch(headless=headless, args=args or [])
    except Exception as e:
        logger.error(f"[SAFE] Exception during browser launch: {e} (Session: {session_id})")
        return None

def safe_new_context(browser: Browser, user_agent: Optional[str] = None, viewport: Optional[dict] = None, locale: Optional[str] = None, session_id: Optional[str] = None) -> Optional[BrowserContext]:
    try:
        if browser is None:
            logger.error(f"[SAFE] browser is None, cannot create context. (Session: {session_id})")
            return None
        return browser.new_context(user_agent=user_agent, viewport=viewport, locale=locale)
    except Exception as e:
        logger.error(f"[SAFE] Exception during new_context: {e} (Session: {session_id})")
        return None

def safe_new_page(context: BrowserContext, session_id: Optional[str] = None) -> Optional[Page]:
    try:
        if context is None:
            logger.error(f"[SAFE] context is None, cannot create new page. (Session: {session_id})")
            return None
        return context.new_page()
    except Exception as e:
        logger.error(f"[SAFE] Exception during new_page: {e} (Session: {session_id})")
        return None

def safe_goto(page: Page, url: str, timeout: int = 60000, session_id: Optional[str] = None) -> Optional[Response]:
    try:
        if page is None:
            logger.error(f"[SAFE] page is None, cannot goto URL. (Session: {session_id})")
            return None
        return page.goto(url, timeout=timeout)
    except Exception as e:
        logger.error(f"[SAFE] Exception during page.goto: {e} (Session: {session_id})")
        return None

def safe_content(page: Page, session_id: Optional[str] = None) -> str:
    """
    Safely call .content() on a Playwright page object.
    Returns the HTML content as a string, or an empty string on error.
    Enhanced: checks for callable, handles async, and logs more detail.
    """
    try:
        if page is None:
            logger.error(f"[SAFE] page is None, cannot get content. (Session: {session_id})")
            return ""
        content_method = getattr(page, "content", None)
        if not callable(content_method):
            logger.error(f"[SAFE] page.content is not callable or missing. (Session: {session_id})")
            return ""
        # Handle both sync and async Playwright APIs   
        if inspect.iscoroutinefunction(content_method):
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            return loop.run_until_complete(content_method())
        else:
            return content_method()
    except Exception as e:
        logger.error(f"[SAFE] Exception during page.content: {e} (Session: {session_id})")
        return ""

def get_random_user_agent() -> str:
    if USER_AGENTS:
        return random.choice(USER_AGENTS)
    return "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"

def launch_minimized_playwright_browser(playwright: sync_playwright, target_url: str, wait_seconds: int = 7, session_id=None) -> Tuple[Optional[Browser], Optional[BrowserContext], Optional[Page], str]:
    user_agent = get_random_user_agent()
    browser_type = safe_chromium(playwright, session_id=session_id)
    browser = safe_launch(browser_type, headless=False, args=["--window-position=0,1000", "--window-size=1280,800"], session_id=session_id)
    context = safe_new_context(browser, user_agent=user_agent, viewport={"width": 1280, "height": 800}, locale="en-US", session_id=session_id)
    page = safe_new_page(context, session_id=session_id)
    safe_goto(page, target_url, timeout=60000, session_id=session_id)
    logger.info(f"[BROWSER] Playwright launched (minimized) with User-Agent: {user_agent} (Session: {session_id})")
    logger.info(f"[BROWSER] Waiting {wait_seconds} seconds for page to load... (Session: {session_id})")
    time.sleep(wait_seconds)
    return browser, context, page, user_agent

def detect_cloudflare_captcha(page: Page) -> bool:
    html = page.content().lower()
    for indicator in CLOUDFLARE_CAPTCHA_INDICATORS:
        if safe_lower(indicator) in html:
            logger.warning(f"[CAPTCHA] Detected Cloudflare CAPTCHA indicator: '{indicator}'")
            return True
    return False

def relaunch_maximized_for_captcha(playwright: sync_playwright, target_url: str, user_agent: str, timeout: int = 300, session_id=None) -> Tuple[Optional[Browser], Optional[BrowserContext], Optional[Page]]:
    browser_type = safe_chromium(playwright, session_id=session_id)
    browser = safe_launch(browser_type, headless=False, args=["--start-maximized"], session_id=session_id)
    context = safe_new_context(browser, user_agent=user_agent, viewport={"width": 1920, "height": 1080}, locale="en-US", session_id=session_id)
    page = safe_new_page(context, session_id=session_id)
    safe_goto(page, target_url, timeout=60000, session_id=session_id)
    logger.info(f"[CAPTCHA] Relaunched browser in maximized mode for manual CAPTCHA resolution. (Session: {session_id})")
    start_time = time.time()
    while time.time() - start_time < timeout:
        html = safe_content(page, session_id=session_id).lower()
        if not any(safe_lower(indicator) in html for indicator in CLOUDFLARE_CAPTCHA_INDICATORS):
            logger.info(f"[CAPTCHA] CAPTCHA appears to be cleared by user. (Session: {session_id})")
            return browser, context, page
        logger.info(f"[CAPTCHA] Waiting for user to solve CAPTCHA... (Session: {session_id})")
        time.sleep(5)
    logger.error(f"[CAPTCHA] Timeout waiting for user to solve CAPTCHA. (Session: {session_id})")
    return None, None, None

def prompt_user_for_selenium_retry() -> bool:
    logger.warning("[yellow][CAPTCHA] CAPTCHA could not be solved or a persistent loading screen was detected.[/yellow]")
    user_input = input("Would you like to retry in Selenium stealth mode? (y/n): ").strip().lower()
    return user_input == "y"

def launch_selenium_stealth(target_url: str, user_agent: str) -> WebDriver:
    from ..utils.seleniumbase_launcher import launch_browser as sb_launch
    _, _, driver = sb_launch(user_agent=user_agent, headless=True)
    driver.get(target_url)
    logger.info("[BROWSER] SeleniumBase launched in stealth mode.")
    return driver

def safe_browser_close(
    browser: Optional[Closable], 
    session_id: Optional[str] = None
) -> None:
    """
    Safely close a Playwright browser instance with robust type and error checks.
    """
    if browser is not None:
        browser_type = type(browser).__name__
        try:
            if hasattr(browser, "close") and callable(browser.close):
                browser.close()
            else:
                logger.warning({
                    "level": "WARNING",
                    "type": "browser",
                    "message": f"Browser object of type '{browser_type}' does not support close().",
                    "session_id": session_id
                })
        except Exception as e:
            logger.warning({
                "level": "WARNING",
                "type": "browser",
                "message": f"Exception during browser close: {e}",
                "session_id": session_id
            })

def browser_pipeline(playwright, target_url, cache_exit_callback=None, session_id=None):
    """
    Main browser utility for html_election_parser.
    Returns (browser, context, page, user_agent) or None if session should exit.
    Handles CAPTCHA detection, user intervention, and Selenium fallback.
    """
    # Step 1: Launch minimized Playwright browser and load page
    browser, context, page, user_agent = launch_minimized_playwright_browser(playwright, target_url)
    # Step 2: Detect CAPTCHA
    if not detect_cloudflare_captcha(page):
        logger.info(f"[CAPTCHA] No CAPTCHA detected. Continuing pipeline. (Session: {session_id})")
        return browser, context, page, user_agent

    # Step 3: CAPTCHA detected, relaunch maximized for user intervention
    safe_browser_close(browser, session_id=session_id)
    browser, context, page = relaunch_maximized_for_captcha(playwright, target_url, user_agent)
    if browser and not detect_cloudflare_captcha(page):
        logger.info(f"[CAPTCHA] CAPTCHA cleared after user intervention. Continuing pipeline. (Session: {session_id})")
        return browser, context, page, user_agent

    # Step 4: If still CAPTCHA or loading, prompt for Selenium retry
    retry_selenium = prompt_user_for_selenium_retry()

    if retry_selenium:
        from ..utils.seleniumbase_launcher import launch_browser, close_driver
        _, _, driver = launch_browser(user_agent=user_agent, headless=True)
        if driver:
            try:
                driver.get(target_url)
                logger.info(f"[CAPTCHA] SeleniumBase launched in stealth mode. (Session: {session_id})")
                # Check for CAPTCHA indicators in SeleniumBase page source
                html = driver.page_source.lower()
                if not any(safe_lower(indicator) in html for indicator in CLOUDFLARE_CAPTCHA_INDICATORS):
                    logger.info(f"[CAPTCHA] CAPTCHA cleared in SeleniumBase. (Session: {session_id})")
                    # Optionally, you could return the driver here if you support SeleniumBase downstream
                    close_driver(driver)
                    return None, None, None, user_agent
                else:
                    logger.warning(f"[CAPTCHA] CAPTCHA still present after SeleniumBase retry. (Session: {session_id})")
            except Exception as e:
                logger.error(f"[CAPTCHA] Exception during SeleniumBase retry: {e} (Session: {session_id})")
            finally:
                close_driver(driver)
        if cache_exit_callback:
            cache_exit_callback(target_url, status="captcha_failed", session_id=session_id)
        return None, None, None, user_agent
    else:
        logger.info(f"[CAPTCHA] User chose to exit gracefully. Exiting session. (Session: {session_id})")
        if cache_exit_callback:
            cache_exit_callback(target_url, status="captcha_exit", session_id=session_id)
        return None, None, None, user_agent

def autoscroll_until_stable(
    page,
    max_stable_frames=5,
    step=8000,
    delay_ms=200,
    max_total_time=10000,
    wait_for_selector=None,
    domain=None,
    coordinator_feedback=None,
) -> bool:
    """
    Continuously scrolls a Playwright page until its scroll height and visible content stabilize
    for at least 5 consecutive measurements, or until max_total_time is reached.
    Optionally waits for a selector to appear.
    Shows a dynamic progress bar using rich or emits progress via SocketIO in webapp mode.
    Does NOT use or save any cached scroll pattern.
    """

    start_time = time.time()
    safe_evaluate(page, "window.scrollTo(0, 0)", logger)
    safe_wait_for_timeout(page, delay_ms, logger)

    stable = 0
    last_heights = []
    last_texts = []
    scroll_attempts = 0
    max_scrolls = max_total_time // delay_ms
    url_str = safe_url(page)
    domain = domain or (
        safe_get_first(url_str.split("/"), "domain_split", None, logger, default="")
        if not ("://" in url_str) else
        safe_get_first(url_str.split("/"), "domain_split", None, logger, default="")
    )
    if "://" in url_str and len(url_str.split("/")) > 2:
        domain = safe_get_first(url_str.split("/"), "domain_split", None, logger, default="", allow_nonlist=True)
        if isinstance(domain, list) and len(domain) > 2:
            domain = domain[2]

    def get_main_text() -> str:
        try:
            main_div = safe_locator(page, "main, .main-content, #main-content, body", logger)
            if main_div:
                return safe_inner_text(main_div, logger)
            else:
                return safe_inner_text(page, logger)
        except Exception:
            return ""

    with logger.progress_bar("[cyan]Scrolling page...", total=max_scrolls) as update_progress:
        while stable < max_stable_frames and scroll_attempts < max_scrolls:
            current_height = safe_evaluate(page, "document.body.scrollHeight", logger)
            current_text = get_main_text()
            last_heights.append(current_height)
            last_texts.append(current_text)
            if len(last_heights) > max_stable_frames:
                last_heights.pop(0)
                last_texts.pop(0)
            # Check if the last N heights and texts are all the same
            if (
                len(last_heights) == max_stable_frames
                and all(h == safe_get_first(last_heights, "last_heights", None, logger) for h in last_heights)
                and all(t == safe_get_first(last_texts, "last_texts", None, logger) for t in last_texts)
            ):
                stable += 1
            else:
                stable = 0
            safe_evaluate(page, f"window.scrollBy(0, {step})", logger)
            safe_wait_for_timeout(page, delay_ms, logger)
            scroll_attempts += 1
            update_progress(scroll_attempts)
            if wait_for_selector and safe_locator(page, wait_for_selector, logger):
                logger.info(f"[SCROLL] Selector '{wait_for_selector}' found. Stopping scroll.")
                break
            elapsed = (time.time() - start_time) * 1000
            if elapsed > max_total_time * 0.8 and scroll_attempts % 10 == 0:
                console.print("[bold yellow]Scrolling is taking longer than expected. Continue waiting? (y/N)[/bold yellow]")
                resp = prompt.prompt_input("Continue scrolling? (y/N): ").strip().lower()
                if resp != "y":
                    logger and logger.warning("[SCROLL] User aborted scrolling.")
                    break
        # Ensure progress bar is completed
        update_progress(max_scrolls)

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

def scan_buttons_with_progress(buttons, scan_callback=None) -> None:
    """
    Scan a list of buttons with a single-line progress bar or emits progress via SocketIO in webapp mode.
    Optionally, provide a scan_callback(button, idx) for custom logic.
    """
    total = len(buttons)
    with logger.progress_bar("Scanning buttons...", total=total) as update_progress:
        for idx, btn in enumerate(buttons):
            label = ""
            try:
                label = safe_inner_text(btn, logger)[:60]
            except Exception:
                label = str(btn)[:60]
            update_progress(idx + 1, extra={"label": label})
            if scan_callback:
                scan_callback(btn, idx)