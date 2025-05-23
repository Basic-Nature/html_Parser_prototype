# utils/browser_utils.py
# ---------------------------------------------------------------
# Handles launching the Playwright browser and applying stealth
# options and user-agent rotation. This includes recovery from
# headless mode if CAPTCHA interaction is required.
# ---------------------------------------------------------------

import random
import os
from playwright.sync_api import sync_playwright, Browser, BrowserContext, Page
from ..utils.user_agents import USER_AGENTS
from ..utils.shared_logger import logger, rprint

def create_browser_context(browser_type, user_agent, headless=True, proxy=None):
    """
    Creates a new browser and context with common settings.

    Args:
        browser_type: The Playwright browser type.
        user_agent (str): The user agent string to use.
        headless (bool): Whether the browser should be headless.
        proxy (dict): Optional proxy settings.

    Returns:
        (browser, context, page): Tuple of browser, context, and page.
    """
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


def launch_browser_with_stealth(playwright, headless=True, minimized=True, user_agent=None, proxy=None):
    """
    Launches a Playwright browser instance with stealth and fingerprinting adjustments.

    Args:
        playwright: The Playwright instance.
        headless (bool): Whether to launch in headless mode.
        minimized (bool): Whether to minimize the browser (if GUI).
        user_agent (str): Optional user-agent override.
        proxy (dict): Optional proxy settings.

    Returns:
        (browser, context, page, user_agent): Tuple of session components.
    """
    browser_type = playwright.chromium
    user_agent = user_agent or random.choice(USER_AGENTS)
    browser, context, page = create_browser_context(browser_type, user_agent, headless=headless, proxy=proxy)
    logger.info(f" Using User-Agent: {user_agent}")
    return browser, context, page, user_agent


def relaunch_browser_fullscreen_if_needed(playwright, url: str, timeout: int = 300, proxy=None):
    """
    Relaunches the browser in non-headless fullscreen mode if CAPTCHA is detected.

    Args:
        playwright: The Playwright instance.
        url (str): The URL to revisit.
        timeout (int): Max seconds to allow CAPTCHA to resolve.
        proxy (dict): Optional proxy settings.

    Returns:
        (browser, context, page, user_agent)
    """
    logger.warning("[CAPTCHA] Relaunching browser in GUI mode for manual resolution...")
    browser_type = playwright.chromium
    user_agent = random.choice(USER_AGENTS)

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
def force_foreground():
    """
    Forces the browser window to the foreground.
    """
    if os.name == 'nt':
        os.system('taskkill /F /IM chrome.exe')
        os.system('start chrome')
    else:
        os.system('xdotool search --onlyvisible --class "chrome" windowactivate')  # Linux
        os.system('xdotool search --onlyvisible --class "firefox" windowactivate')
    # macOS
    # os.system('osascript -e \'tell application "Google Chrome" to activate\'')
    # os.system('osascript -e \'tell application "Firefox" to activate\'')
    # os.system('osascript -e \'tell application "Safari" to activate\'')
    # Add more OS-specific commands as needed
    # to bring the browser to the foreground.
    # This is a placeholder and may not work on all systems.
    # You may need to use platform-specific commands or libraries.
    # For example, on macOS, you might use AppleScript to bring the app to the front.
    # On Linux, you might use xdotool or wmctrl.
    # On Windows, you might use pygetwindow or similar libraries.
    
def is_browser_open(browser: Browser) -> bool:
    """
    Checks if the browser is still open.

    Args:
        browser (Browser): The Playwright browser instance.

    Returns:
        bool: True if the browser is open, False otherwise.
    """
    try:
        return not browser.is_closed()
    except Exception as e:
        logger.error(f"[BROWSER] Error checking browser status: {e}")
        return False
def close_browser(browser: Browser):
    """
    Closes the browser instance.

    Args:
        browser (Browser): The Playwright browser instance.
    """
    try:
        if is_browser_open(browser):
            browser.close()
            logger.info("[BROWSER] Browser closed successfully.")
        else:
            logger.info("[BROWSER] Browser is already closed.")
    except Exception as e:
        logger.error(f"[BROWSER] Error closing browser: {e}")
        
def close_context(context: BrowserContext):
    """
    Closes the browser context.

    Args:
        context (BrowserContext): The Playwright browser context instance.
    """
    try:
        if context:
            context.close()
            logger.info("[CONTEXT] Context closed successfully.")
        else:
            logger.info("[CONTEXT] Context is already closed.")
    except Exception as e:
        logger.error(f"[CONTEXT] Error closing context: {e}")
        
def close_page(page: Page):
    """
    Closes the browser page.

    Args:
        page (Page): The Playwright page instance.
    """
    try:
        if page:
            page.close()
            logger.info("[PAGE] Page closed successfully.")
        else:
            logger.info("[PAGE] Page is already closed.")
    except Exception as e:
        logger.error(f"[PAGE] Error closing page: {e}")
        
def close_all(browser: Browser, context: BrowserContext, page: Page):
    """
    Closes the browser, context, and page.

    Args:
        browser (Browser): The Playwright browser instance.
        context (BrowserContext): The Playwright browser context instance.
        page (Page): The Playwright page instance.
    """
    close_page(page)
    close_context(context)
    close_browser(browser)

def HEADLESS_DEFAULT():
    """
    Returns the default headless mode setting.
    """
    return os.getenv("HEADLESS", "True").lower() == "true"

def HEADLESS_MINIMIZED_DEFAULT():
    """
    Returns the default minimized mode setting for headless browsers.
    """
    return os.getenv("HEADLESS_MINIMIZED", "True").lower() == "true"

def HEADLESS_MAXIMIZED_DEFAULT():
    """
    Returns the default maximized mode setting for headless browsers.
    """
    return os.getenv("HEADLESS_MAXIMIZED", "False").lower() == "true"
def MINIMIZED_DEFAULT():
    """
    Returns the default minimized mode setting.
    """
    return os.getenv("MINIMIZED", "True").lower() == "true"
def SCAN_WAIT_SECONDS():
    """
    Returns the default wait time for scanning.
    """
    return int(os.getenv("SCAN_WAIT_SECONDS", 5))  