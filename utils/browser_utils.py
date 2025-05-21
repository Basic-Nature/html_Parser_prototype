# utils/browser_utils.py
# ---------------------------------------------------------------
# Handles launching the Playwright browser and applying stealth
# options and user-agent rotation. This includes recovery from
# headless mode if CAPTCHA interaction is required.
# ---------------------------------------------------------------

import random
import os
from playwright.sync_api import sync_playwright, Browser, BrowserContext, Page
from utils.user_agents import USER_AGENTS
from utils.shared_logger import logger, rprint

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
