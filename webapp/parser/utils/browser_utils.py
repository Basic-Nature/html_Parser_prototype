from __future__ import annotations
# utils/browser_utils.py
# ---------------------------------------------------------------
# Handles launching the Playwright browser (sync or async) and applying stealth,
# user-agent rotation, and CAPTCHA handling. Supports both interactive (async)
# and batch (subprocess) use cases.
# ---------------------------------------------------------------
import os
import random
import time
import inspect
import asyncio
import re
from typing import (
    Protocol, Optional, Tuple, Union, Any, Sequence, Dict, TypeVar
)
from selectolax.parser import Node as SelectolaxNode
from playwright.sync_api import (
    sync_playwright, Browser as SyncBrowser, BrowserContext as SyncBrowserContext,
    Page as SyncPage, BrowserType as SyncBrowserType, ElementHandle as SyncElementHandle, Locator as SyncLocator
)
from playwright.async_api import (
    async_playwright, Browser as AsyncBrowser, BrowserContext as AsyncBrowserContext,
    Page as AsyncPage, BrowserType as AsyncBrowserType, ElementHandle as AsyncElementHandle, Locator as AsyncLocator
)
from selenium.webdriver.remote.webelement import WebElement as SeleniumElement
from .logger_singleton import logger, console, prompt
from .shared_logic import (
    safe_lower, safe_get_first
)
from ..config import CONTEXT_LIBRARY_PATH

# --- Type Aliases for IDE and Type Checking ---
PageType = Union[SyncPage, AsyncPage]
ElementType = Union[SyncElementHandle, AsyncElementHandle]
LocatorType = Union[SyncLocator, AsyncLocator]
BrowserType = Union[SyncBrowser, AsyncBrowser]
BrowserContextType = Union[SyncBrowserContext, AsyncBrowserContext]
EvaluateType = Union[PageType, ElementType]

T = TypeVar("T")

ElementLike = Union[
    SyncElementHandle,
    AsyncElementHandle,
    SelectolaxNode,
    SeleniumElement,
    object  # fallback for custom nodes
]

# Load user agents and captcha indicators from context library
if os.path.exists(CONTEXT_LIBRARY_PATH):
    import orjson
    with open(CONTEXT_LIBRARY_PATH, "rb") as f:
        CONTEXT_LIBRARY = orjson.loads(f.read())

    def get_list(context: dict, key: str) -> list:
        value = context.get(key, [])
        if isinstance(value, list):
            return value
        # Securely parse stringified lists, avoid code injection
        if isinstance(value, str):
            # Only allow a list of quoted strings or numbers, no code
            safe_list_pattern = r"^\[\s*('([^'\\]|\\.)*'|\"([^\"\\]|\\.)*\"|\d+)(\s*,\s*('([^'\\]|\\.)*'|\"([^\"\\]|\\.)*\"|\d+))*\s*\]$"
            if re.fullmatch(safe_list_pattern, value):
                try:
                    import ast
                    parsed = ast.literal_eval(value)
                    # Ensure all elements are str or int, not objects/code
                    if isinstance(parsed, list) and all(isinstance(x, (str, int, float)) for x in parsed):
                        return parsed
                except Exception:
                    logger.warning(f"[browser_utils] Failed to safely parse context_library value for key '{key}'")
            else:
                logger.warning(f"[browser_utils] Skipping unsafe context_library value for key '{key}'")
        return []

    USER_AGENTS = get_list(CONTEXT_LIBRARY, "user_agents")
    CLOUDFLARE_CAPTCHA_INDICATORS = get_list(CONTEXT_LIBRARY, "cloudflare_captcha_indicators")
else:
    logger.error("[browser_utils] context_library.json not found. User agent rotation will be limited.")
    USER_AGENTS = []
    CLOUDFLARE_CAPTCHA_INDICATORS = []

class Closable(Protocol):
    def close(self) -> None:
        """Protocol for objects with a close method."""
        ...

def get_random_user_agent() -> str:
    if USER_AGENTS:
        return random.choice(USER_AGENTS)
    return "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"

# -------------------- SHARED SAFE HELPERS --------------------

def safe_url(page: Optional[PageType]) -> str:
    """Safely get the URL from a Playwright page."""
    try:
        url = getattr(page, "url", "")
        return str(url) if isinstance(url, str) else ""
    except Exception as e:
        logger.error(f"[safe_url] Error accessing page.url: {e}")
        return ""

def safe_inner_text(obj: Optional[ElementType | PageType], logger=logger) -> str:
    """
    Safely get inner text from a Playwright element or page.
    Only calls .inner_text() on known Playwright ElementHandle or Page types.
    """
    try:
        if isinstance(obj, (SyncElementHandle, AsyncElementHandle)):
            return obj.inner_text()
        if isinstance(obj, (SyncPage, AsyncPage)) and hasattr(obj, "inner_text"):
            return obj.inner_text()
        logger and logger.error(f"[safe_inner_text] Object is not a Playwright ElementHandle or Page: {type(obj)}")
        return ""
    except Exception as e:
        if logger: logger.error(f"[safe_inner_text] Error: {e}")
        return ""

def safe_locator(page: Optional[PageType], selector: str, logger=logger) -> Optional[LocatorType]:
    """Safely get a locator from a Playwright page."""
    try:
        if hasattr(page, "locator"):
            return page.locator(selector)
        return None
    except Exception as e:
        if logger: logger.error(f"[safe_locator] Error: {e}")
        return None

def safe_evaluate(obj: Optional[EvaluateType], script: str, logger=logger) -> Any:
    """
    Safely evaluate a script on a Playwright Page or ElementHandle, with security checks.
    Only allows certain patterns and only calls evaluate on known Playwright types.
    """
    try:
        forbidden = [
            "import", "__", "eval", "exec", "open", "os.", "sys.", "subprocess", "pickle",
            "Function(", "constructor", "window['", "window[\"", "document['", "document[\""
        ]
        allowed_patterns = [
            r"^window\.scroll(To|By)\(\d+,\s*\d+\)$",
            r"^document\.body\.scrollHeight$"
        ]
        if not isinstance(script, str):
            logger.error(f"[safe_evaluate] Script is not a string, refusing to execute: {script}")
            return None
        if any(x in script for x in forbidden):
            logger.error(f"[safe_evaluate] Unsafe script detected, refusing to execute: {script}")
            return None
        if not any(re.fullmatch(pat, script.strip()) for pat in allowed_patterns):
            logger.error(f"[safe_evaluate] Script does not match allowed patterns, refusing to execute: {script}")
            return None

        if isinstance(obj, (SyncPage, AsyncPage, SyncElementHandle, AsyncElementHandle)):
            return obj.evaluate(script)
        else:
            logger.error(f"[safe_evaluate] Object is not a Playwright Page or ElementHandle: {type(obj)}")
            return None
    except Exception as e:
        if logger: logger.error(f"[safe_evaluate] Error: {e}")
        return None

def safe_wait_for_timeout(page: Optional[PageType], ms: int, logger=logger) -> bool:
    """Safely wait for a timeout on a Playwright page."""
    try:
        if hasattr(page, "wait_for_timeout"):
            page.wait_for_timeout(ms)
            return True
        return False
    except Exception as e:
        if logger: logger.error(f"[safe_wait_for_timeout] Error: {e}")
        return False

def safe_content(page: Optional[PageType], session_id: Optional[str] = None) -> str:
    """Safely get the HTML content from a Playwright page."""
    try:
        if page is None:
            logger.error(f"[SAFE] page is None, cannot get content. (Session: {session_id})")
            return ""
        content_method = getattr(page, "content", None)
        if not callable(content_method):
            logger.error(f"[SAFE] page.content is not callable or missing. (Session: {session_id})")
            return ""
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

def safe_nth(seq: Optional[Sequence[Any]], n: int, default: Any = None) -> Any:
    """Return the nth item of seq if it exists, else default."""
    try:
        return seq[n]
    except (IndexError, TypeError):
        return default

def safe_is_visible(element: Optional[ElementType], logger=logger) -> bool:
    """Safely check if a Playwright element is visible."""
    try:
        if hasattr(element, "is_visible"):
            return element.is_visible()
        return False
    except Exception as e:
        if logger: logger.error(f"[safe_is_visible] Error: {e}")
        return False

def safe_is_enabled(element: Optional[ElementType], logger=logger) -> bool:
    """Safely check if a Playwright element is enabled."""
    try:
        if hasattr(element, "is_enabled"):
            return element.is_enabled()
        return False
    except Exception as e:
        if logger: logger.error(f"[safe_is_enabled] Error: {e}")
        return False

def safe_click(element: Optional[ElementType], logger=logger) -> bool:
    """Safely click a Playwright element."""
    try:
        if hasattr(element, "click"):
            element.click()
            return True
        return False
    except Exception as e:
        if logger: logger.error(f"[safe_click] Error: {e}")
        return False

def safe_get_attribute(element: Optional[ElementType], attr: str, logger=logger) -> str:
    """Safely get an attribute from a Playwright element."""
    try:
        if hasattr(element, "get_attribute"):
            val = element.get_attribute(attr)
            return val if val is not None else ""
        return ""
    except Exception as e:
        if logger: logger.error(f"[safe_get_attribute] Error: {e}")
        return ""

def safe_attributes(element: ElementLike) -> Dict[str, str]:
    """
    Extract attributes from selectolax, Playwright, or Selenium elements.
    Returns a dict of attribute names to string values.
    """
    try:
        # selectolax Node
        if hasattr(element, "attributes") and isinstance(element.attributes, dict):
            return {str(k): str(v) for k, v in element.attributes.items()}

        # Playwright ElementHandle (sync or async)
        if hasattr(element, "get_attribute") and callable(getattr(element, "get_attribute", None)):
            eval_fn = getattr(element, "evaluate", None)
            if callable(eval_fn):
                try:
                    attrs = eval_fn(
                        "el => Object.fromEntries(Array.from(el.attributes).map(a => [a.name, a.value]))"
                    )
                    if isinstance(attrs, dict):
                        return {str(k): str(v) for k, v in attrs.items()}
                except Exception as e:
                    logger.warning(f"[safe_attributes] Playwright JS extraction failed: {e}")
            # Fallback: try common attributes
            try:
                attr_names = [
                    "id", "class", "name", "type", "value", "href", "src", "alt",
                    "title", "role", "style", "data-*"
                ]
                result = {}
                for attr in attr_names:
                    val = element.get_attribute(attr)
                    if val is not None:
                        result[attr] = str(val)
                return result
            except Exception as e:
                logger.warning(f"[safe_attributes] Playwright fallback extraction failed: {e}")

        # Selenium WebElement
        if hasattr(element, "get_attribute") and hasattr(element, "tag_name"):
            try:
                attrs = element.get_property("attributes")
                if attrs and isinstance(attrs, list):
                    return {str(a['name']): str(a['value']) for a in attrs if 'name' in a and 'value' in a}
            except Exception:
                # fallback: try common attributes
                attr_names = [
                    "id", "class", "name", "type", "value", "href", "src", "alt",
                    "title", "role", "style", "data-*"
                ]
                result = {}
                for attr in attr_names:
                    try:
                        val = element.get_attribute(attr)
                        if val is not None:
                            result[attr] = str(val)
                    except Exception:
                        continue
                return result

        # Try __dict__ as last resort (rare, but some custom nodes)
        if hasattr(element, "__dict__"):
            attrs = getattr(element, "__dict__")
            if isinstance(attrs, dict):
                return {str(k): str(v) for k, v in attrs.items() if isinstance(k, str) and isinstance(v, (str, int, float, bool))}

        return {}
    except Exception as e:
        logger.error(f"[safe_attributes] Error extracting attributes from {type(element)}: {e}")
        return {}

def safe_query_selector_all(page: Optional[PageType], selector: str, logger=logger) -> list:
    """Safely query all selectors on a Playwright page."""
    try:
        if hasattr(page, "query_selector_all"):
            return page.query_selector_all(selector)
        return []
    except Exception as e:
        if logger: logger.error(f"[safe_query_selector_all] Error: {e}")
        return []

def safe_context_library(page: Optional[PageType] = None, session_id: Optional[str] = None) -> dict:
    """Safely load the context library from disk or memory."""
    try:
        if os.path.exists(CONTEXT_LIBRARY_PATH):
            import orjson
            with open(CONTEXT_LIBRARY_PATH, "rb") as f:
                return orjson.loads(f.read())
        return {}
    except Exception as e:
        logger.error(f"[safe_context_library] Error: {e} (Session: {session_id})")
        return {}

def safe_count(obj: Optional[Any], logger=logger) -> int:
    """
    Safely get the count/length of a Playwright locator, element list, or any countable object.
    Returns 0 if not countable or on error.
    """
    try:
        # Playwright Locator has a count() method (sync/async)
        if hasattr(obj, "count"):
            count_method = getattr(obj, "count")
            if callable(count_method):
                # Handle async and sync
                if inspect.iscoroutinefunction(count_method):
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    return loop.run_until_complete(count_method())
                else:
                    return count_method()
        # Try len() for lists, tuples, etc.
        if hasattr(obj, "__len__"):
            return len(obj)
        # Try length property
        if hasattr(obj, "length"):
            return getattr(obj, "length")
        logger and logger.warning(f"[safe_count] Object is not countable: {type(obj)}")
        return 0
    except Exception as e:
        if logger: logger.error(f"[safe_count] Error: {e}")
        return 0

def safe_context_result(page: Optional[PageType], session_id: Optional[str] = None) -> dict:
    """
    Safely get context metadata/results for the page/session.
    Handles both sync and async Playwright Page types and checks for dict-like metadata.
    """
    try:
        ctx = None
        # Playwright Page has .context property (not always callable)
        if page is not None and hasattr(page, "context"):
            context_attr = getattr(page, "context")
            # If it's a method, call it; if property, use as is
            ctx = context_attr() if callable(context_attr) else context_attr
        if ctx is not None:
            metadata = getattr(ctx, "metadata", None)
            if isinstance(metadata, dict):
                return metadata
            # Some Playwright contexts may have .metadata() as a method
            if callable(metadata):
                meta_val = metadata()
                if isinstance(meta_val, dict):
                    return meta_val
        return {}
    except Exception as e:
        logger.error(f"[safe_context_result] Error: {e} (Session: {session_id})")
        return {}

def safe_launch(
    browser_type: Optional[object], headless: bool = False, args: list = None, logger=logger
) -> Optional[BrowserType]:
    """Safely launch a Playwright browser (sync)."""
    try:
        if browser_type is None:
            logger.error("[safe_launch] browser_type is None.")
            return None
        if not hasattr(browser_type, "launch") or not callable(getattr(browser_type, "launch", None)):
            logger.error(f"[safe_launch] browser_type does not have a callable .launch(): {type(browser_type)}")
            return None
        # Optionally, check for correct type
        if not isinstance(browser_type, SyncBrowserType):
            logger.warning(f"[safe_launch] browser_type is not a SyncBrowserType: {type(browser_type)}")
        return browser_type.launch(headless=headless, args=args or [])
    except Exception as e:
        logger.error(f"[safe_launch] Error launching browser: {e}")
        return None

async def async_safe_launch(
    browser_type: Optional[object], headless: bool = False, args: list = None, logger=logger
) -> Optional[AsyncBrowser]:
    """Safely launch a Playwright browser (async)."""
    try:
        if browser_type is None:
            logger.error("[async_safe_launch] browser_type is None.")
            return None
        if not hasattr(browser_type, "launch") or not callable(getattr(browser_type, "launch", None)):
            logger.error(f"[async_safe_launch] browser_type does not have a callable .launch(): {type(browser_type)}")
            return None
        # Optionally, check for correct type
        if not isinstance(browser_type, AsyncBrowserType):
            logger.warning(f"[async_safe_launch] browser_type is not an AsyncBrowserType: {type(browser_type)}")
        return await browser_type.launch(headless=headless, args=args or [])
    except Exception as e:
        logger.error(f"[async_safe_launch] Error launching browser: {e}")
        return None
def safe_new_context(browser: Optional[BrowserType], **kwargs) -> Optional[BrowserContextType]:
    """Safely create a new browser context (sync)."""
    try:
        if browser is None:
            logger.error("[safe_new_context] browser is None.")
            return None
        return browser.new_context(**kwargs)
    except Exception as e:
        logger.error(f"[safe_new_context] Error creating context: {e}")
        return None

async def async_safe_new_context(browser: Optional[AsyncBrowser], **kwargs) -> Optional[AsyncBrowserContext]:
    """Safely create a new browser context (async)."""
    try:
        if browser is None:
            logger.error("[async_safe_new_context] browser is None.")
            return None
        return await browser.new_context(**kwargs)
    except Exception as e:
        logger.error(f"[async_safe_new_context] Error creating context: {e}")
        return None

def safe_new_page(context: Optional[BrowserContextType]) -> Optional[PageType]:
    """Safely create a new page (sync)."""
    try:
        if context is None:
            logger.error("[safe_new_page] context is None.")
            return None
        return context.new_page()
    except Exception as e:
        logger.error(f"[safe_new_page] Error creating page: {e}")
        return None

async def async_safe_new_page(context: Optional[AsyncBrowserContext]) -> Optional[AsyncPage]:
    """Safely create a new page (async)."""
    try:
        if context is None:
            logger.error("[async_safe_new_page] context is None.")
            return None
        return await context.new_page()
    except Exception as e:
        logger.error(f"[async_safe_new_page] Error creating page: {e}")
        return None

def safe_goto(page: Optional[PageType], url: str, timeout: int = 60000) -> bool:
    """Safely navigate to a URL (sync)."""
    try:
        if page is None:
            logger.error("[safe_goto] page is None.")
            return False
        page.goto(url, timeout=timeout)
        return True
    except Exception as e:
        logger.error(f"[safe_goto] Error navigating to {url}: {e}")
        return False

async def async_safe_goto(page: Optional[AsyncPage], url: str, timeout: int = 60000) -> bool:
    """Safely navigate to a URL (async)."""
    try:
        if page is None:
            logger.error("[async_safe_goto] page is None.")
            return False
        await page.goto(url, timeout=timeout)
        return True
    except Exception as e:
        logger.error(f"[async_safe_goto] Error navigating to {url}: {e}")
        return False

async def async_safe_browser_close(browser: Optional[AsyncBrowser], session_id: Optional[str] = None) -> None:
    if browser is not None:
        try:
            await browser.close()
        except Exception as e:
            logger.warning({
                "level": "WARNING",
                "type": "browser",
                "message": f"Exception during async browser close: {e}",
                "session_id": session_id
            })

# -------------------- ASYNC PLAYWRIGHT PIPELINE --------------------

async def async_launch_browser(target_url: str, wait_seconds: int = 7, session_id=None) -> Tuple[Optional[AsyncBrowser], Optional[AsyncBrowserContext], Optional[AsyncPage], str]:
    user_agent = get_random_user_agent()
    async with async_playwright() as p:
        browser_type = getattr(p, "chromium", None)
        browser = await async_safe_launch(browser_type, headless=False, args=["--window-position=0,1000", "--window-size=1280,800"])
        context = await async_safe_new_context(browser, user_agent=user_agent, viewport={"width": 1280, "height": 800}, locale="en-US")
        page = await async_safe_new_page(context)
        await async_safe_goto(page, target_url, timeout=60000)
        logger.info(f"[BROWSER] Async Playwright launched (minimized) with User-Agent: {user_agent} (Session: {session_id})")
        logger.info(f"[BROWSER] Waiting {wait_seconds} seconds for page to load... (Session: {session_id})")
        await asyncio.sleep(wait_seconds)
        return browser, context, page, user_agent

async def async_detect_cloudflare_captcha(page: AsyncPage) -> bool:
    html = (await page.content()).lower()
    for indicator in CLOUDFLARE_CAPTCHA_INDICATORS:
        if safe_lower(indicator) in html:
            logger.warning(f"[CAPTCHA] Detected Cloudflare CAPTCHA indicator: '{indicator}'")
            return True
    return False

async def async_browser_pipeline(target_url: str, session_id=None) -> Tuple[Optional[AsyncBrowser], Optional[AsyncBrowserContext], Optional[AsyncPage], str]:
    browser, context, page, user_agent = await async_launch_browser(target_url, session_id=session_id)
    if not await async_detect_cloudflare_captcha(page):
        logger.info(f"[CAPTCHA] No CAPTCHA detected. Continuing pipeline. (Session: {session_id})")
        return browser, context, page, user_agent
    logger.warning(f"[CAPTCHA] CAPTCHA detected in async mode. Manual intervention not implemented. (Session: {session_id})")
    return browser, context, page, user_agent

# -------------------- SYNC PLAYWRIGHT PIPELINE (for subprocess/batch) --------------------

def sync_launch_browser(playwright: sync_playwright, target_url: str, wait_seconds: int = 7, session_id=None) -> Tuple[Optional[SyncBrowser], Optional[SyncBrowserContext], Optional[SyncPage], str]:
    user_agent = get_random_user_agent()
    browser_type = getattr(playwright, "chromium", None)
    browser = safe_launch(browser_type, headless=False, args=["--window-position=0,1000", "--window-size=1280,800"])
    context = safe_new_context(browser, user_agent=user_agent, viewport={"width": 1280, "height": 800}, locale="en-US")
    page = safe_new_page(context)
    safe_goto(page, target_url, timeout=60000)
    logger.info(f"[BROWSER] Playwright launched (minimized) with User-Agent: {user_agent} (Session: {session_id})")
    logger.info(f"[BROWSER] Waiting {wait_seconds} seconds for page to load... (Session: {session_id})")
    time.sleep(wait_seconds)
    return browser, context, page, user_agent

def sync_detect_cloudflare_captcha(page: SyncPage) -> bool:
    html = page.content().lower()
    for indicator in CLOUDFLARE_CAPTCHA_INDICATORS:
        if safe_lower(indicator) in html:
            logger.warning(f"[CAPTCHA] Detected Cloudflare CAPTCHA indicator: '{indicator}'")
            return True
    return False

def sync_safe_browser_close(browser: Optional[SyncBrowser], session_id: Optional[str] = None) -> None:
    if browser is not None:
        try:
            browser.close()
        except Exception as e:
            logger.warning({
                "level": "WARNING",
                "type": "browser",
                "message": f"Exception during browser close: {e}",
                "session_id": session_id
            })

def sync_browser_pipeline(playwright, target_url, cache_exit_callback=None, session_id=None) -> Tuple[Optional[SyncBrowser], Optional[SyncBrowserContext], Optional[SyncPage], str]:
    browser, context, page, user_agent = sync_launch_browser(playwright, target_url, session_id=session_id)
    if not sync_detect_cloudflare_captcha(page):
        logger.info(f"[CAPTCHA] No CAPTCHA detected. Continuing pipeline. (Session: {session_id})")
        return browser, context, page, user_agent
    logger.warning(f"[CAPTCHA] CAPTCHA detected in sync mode. Manual intervention not implemented. (Session: {session_id})")
    return browser, context, page, user_agent

# -------------------- USAGE PATTERN --------------------
# For interactive/session-based parsing (async):
#   await async_browser_pipeline(target_url, session_id=session_id)
#
# For batch/subprocess parsing (sync):
#   with sync_playwright() as p:
#       sync_browser_pipeline(p, target_url, session_id=session_id)
#
# Both pipelines return (browser, context, page, user_agent)

# -------------------- (Optional) AUTOSCROLL AND OTHER UTILITIES --------------------

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