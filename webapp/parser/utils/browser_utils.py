# utils/browser_utils.py
# ---------------------------------------------------------------
# Handles launching the Playwright or SeleniumBase browser and applying stealth
# options and user-agent rotation. Handles recovery from headless mode if CAPTCHA
# interaction is required, and can relaunch into stealth mode after persistent CAPTCHA.
# ---------------------------------------------------------------

import random
import os
import platform
import json
from ..utils.seleniumbase_launcher import launch_browser as sb_launch, relaunch_browser_fullscreen_if_needed as sb_relaunch         
from playwright.sync_api import sync_playwright, Browser, BrowserContext, Page
from ..utils.logger_instance import logger
from ..utils.shared_logger import rprint

# --- Load user agents from context library ---
CONTEXT_LIBRARY_PATH = os.path.join(
    os.path.dirname(__file__), "..", "Context_Integration", "context_library.json"
)
if os.path.exists(CONTEXT_LIBRARY_PATH):
    with open(CONTEXT_LIBRARY_PATH, "r", encoding="utf-8") as f:
        CONTEXT_LIBRARY = json.load(f)
    USER_AGENTS = CONTEXT_LIBRARY.get("user_agents", [])
else:
    logger.error("[browser_utils] context_library.json not found. User agent rotation will be limited.")
    USER_AGENTS = []

def get_random_user_agent():
    if USER_AGENTS:
        return random.choice(USER_AGENTS)
    # Fallback
    return "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"

def create_browser_context(browser_type, user_agent, headless=True, proxy=None):
    launch_args = {
        "headless": headless,
        "args": ["--disable-blink-features=AutomationControlled"]
    }
    if proxy:
        launch_args["proxy"] = proxy

    browser = browser_type.launch(**launch_args)
    context = browser.new_context(
        user_agent=user_agent,
        viewport={"width": 1280, "height": 800},
        locale="en-US"
    )
    page = context.new_page()
    return browser, context, page

def launch_browser_with_stealth(playwright, headless=True, minimized=True, user_agent=None, proxy=None, backend="playwright"):
    """
    Launches a browser instance (Playwright or SeleniumBase) with stealth and fingerprinting adjustments.
    Args:
        playwright: The Playwright instance (if using Playwright).
        headless (bool): Whether to launch in headless mode.
        minimized (bool): Whether to minimize the browser (if GUI).
        user_agent (str): Optional user-agent override.
        proxy (dict): Optional proxy settings.
        backend (str): "playwright" or "seleniumbase"
    Returns:
        (browser, context, page, user_agent): Tuple of session components.
    """
    user_agent = user_agent or get_random_user_agent()
    if backend == "seleniumbase":

        _, _, driver = sb_launch()
        logger.info(f"[BROWSER] SeleniumBase launched with User-Agent: {user_agent}")
        return None, None, driver, user_agent
    # Default: Playwright
    browser_type = playwright.chromium
    browser, context, page = create_browser_context(browser_type, user_agent, headless=headless, proxy=proxy)
    logger.info(f"[BROWSER] Playwright launched with User-Agent: {user_agent}")
    return browser, context, page, user_agent

def relaunch_browser_fullscreen_if_needed(playwright, url: str, timeout: int = 300, proxy=None, backend="playwright"):
    """
    Relaunches the browser in non-headless fullscreen mode if CAPTCHA is detected.
    Args:
        playwright: The Playwright instance (if using Playwright).
        url (str): The URL to revisit.
        timeout (int): Max seconds to allow CAPTCHA to resolve.
        proxy (dict): Optional proxy settings.
        backend (str): "playwright" or "seleniumbase"
    Returns:
        (browser, context, page, user_agent)
    """
    user_agent = get_random_user_agent()
    if backend == "seleniumbase":
        # sb_relaunch handles maximize and wait for CAPTCHA
        driver = sb_relaunch(None, url, timeout=timeout)
        logger.info("[BROWSER] SeleniumBase relaunched for CAPTCHA resolution.")
        return None, None, driver, user_agent

    logger.warning("[CAPTCHA] Relaunching Playwright browser in GUI mode for manual resolution...")
    browser_type = playwright.chromium
    launch_args = {
        "headless": False,
        "args": ["--start-maximized"]
    }
    if proxy:
        launch_args["proxy"] = proxy

    browser = browser_type.launch(**launch_args)
    context = browser.new_context(
        user_agent=user_agent,
        viewport={"width": 1280, "height": 800},
        locale="en-US"
    )
    page = context.new_page()
    try:
        page.goto(url, timeout=timeout * 1000)
        logger.info("[CAPTCHA] Waiting in GUI mode. Please resolve the CAPTCHA manually.")
        logger.info("[CAPTCHA] If nothing appears, try manually refreshing the page.")
    except Exception as e:
        logger.error(f"[CAPTCHA] Failed to load page in GUI mode: {e}")

    return browser, context, page, user_agent

def relaunch_browser_stealth(playwright, url: str, proxy=None, backend="playwright"):
    """
    Relaunches the browser in stealth mode after persistent CAPTCHA.
    """
    user_agent = get_random_user_agent()
    if backend == "seleniumbase":

        _, _, driver = sb_launch()
        driver.get(url)
        logger.info("[BROWSER] SeleniumBase relaunched in stealth mode.")
        return None, None, driver, user_agent

    browser_type = playwright.chromium
    browser, context, page = create_browser_context(browser_type, user_agent, headless=True, proxy=proxy)
    page.goto(url)
    logger.info("[BROWSER] Playwright relaunched in stealth mode.")
    return browser, context, page, user_agent

def detect_environment_for_captcha(page_or_driver):
    """
    Detects if the environment is likely to trigger persistent CAPTCHA.
    Returns "playwright", "seleniumbase", or "unknown".
    """
    # Simple detection based on object type or attributes
    if hasattr(page_or_driver, "content"):
        return "playwright"
    if hasattr(page_or_driver, "page_source"):
        return "seleniumbase"
    return "unknown"

def force_foreground(page_or_driver=None):
    """
    Forces the browser window to the foreground.
    """
    os_type = platform.system()
    try:
        # Playwright Page
        if page_or_driver and hasattr(page_or_driver, "bring_to_front"):
            page_or_driver.bring_to_front()
        # SeleniumBase Driver (Chrome/Edge)
        elif page_or_driver and hasattr(page_or_driver, "maximize_window"):
            page_or_driver.maximize_window()
        # OS-level fallback
        if os_type == "Windows":
            import ctypes
            ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 9)  # SW_RESTORE
            ctypes.windll.user32.SetForegroundWindow(ctypes.windll.kernel32.GetConsoleWindow())
        elif os_type == "Darwin":
            os.system("osascript -e 'tell application \"System Events\" to set frontmost of the first process whose unix id is (do shell script \"echo $PPID\") to true'")
        elif os_type == "Linux":
            os.system("xdotool windowactivate $(xdotool search --onlyvisible --name 'Chromium' | head -1) 2>/dev/null")
    except Exception as e:
        logger.warning(f"[BROWSER] Foreground window fallback failed: {e}")

def is_browser_open(browser: Browser) -> bool:
    try:
        return not browser.is_closed()
    except Exception as e:
        logger.error(f"[BROWSER] Error checking browser status: {e}")
        return False

def close_browser(browser: Browser):
    try:
        if is_browser_open(browser):
            browser.close()
            logger.info("[BROWSER] Browser closed successfully.")
        else:
            logger.info("[BROWSER] Browser is already closed.")
    except Exception as e:
        logger.error(f"[BROWSER] Error closing browser: {e}")

def close_context(context: BrowserContext):
    try:
        if context:
            context.close()
            logger.info("[CONTEXT] Context closed successfully.")
        else:
            logger.info("[CONTEXT] Context is already closed.")
    except Exception as e:
        logger.error(f"[CONTEXT] Error closing context: {e}")

def close_page(page: Page):
    try:
        if page:
            page.close()
            logger.info("[PAGE] Page closed successfully.")
        else:
            logger.info("[PAGE] Page is already closed.")
    except Exception as e:
        logger.error(f"[PAGE] Error closing page: {e}")

def close_all(browser: Browser, context: BrowserContext, page: Page):
    close_page(page)
    close_context(context)
    close_browser(browser)

def HEADLESS_DEFAULT():
    return os.getenv("HEADLESS", "True").lower() == "true"

def HEADLESS_MINIMIZED_DEFAULT():
    return os.getenv("HEADLESS_MINIMIZED", "True").lower() == "true"

def HEADLESS_MAXIMIZED_DEFAULT():
    return os.getenv("HEADLESS_MAXIMIZED", "False").lower() == "true"

def MINIMIZED_DEFAULT():
    return os.getenv("MINIMIZED", "True").lower() == "true"

def SCAN_WAIT_SECONDS():
    return int(os.getenv("SCAN_WAIT_SECONDS", 5))