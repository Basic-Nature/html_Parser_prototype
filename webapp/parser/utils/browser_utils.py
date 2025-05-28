# utils/browser_utils.py
# ---------------------------------------------------------------
# Handles launching the Playwright or SeleniumBase browser and applying stealth
# options and user-agent rotation. Handles recovery from headless mode if CAPTCHA
# interaction is required, and can relaunch into stealth mode after persistent CAPTCHA.
# ---------------------------------------------------------------

import os
import random
import time
from playwright.sync_api import sync_playwright, Browser, BrowserContext, Page
from ..utils.logger_instance import logger
from ..utils.shared_logger import rprint
from ..config import CONTEXT_LIBRARY_PATH

# Load user agents and captcha indicators from context library
if os.path.exists(CONTEXT_LIBRARY_PATH):
    import json
    with open(CONTEXT_LIBRARY_PATH, "r", encoding="utf-8") as f:
        CONTEXT_LIBRARY = json.load(f)
    USER_AGENTS = CONTEXT_LIBRARY.get("user_agents", [])
    CLOUDFLARE_CAPTCHA_INDICATORS = CONTEXT_LIBRARY.get("cloudflare_captcha_indicators", [])
else:
    logger.error("[browser_utils] context_library.json not found. User agent rotation will be limited.")
    USER_AGENTS = []
    CLOUDFLARE_CAPTCHA_INDICATORS = []

def get_random_user_agent():
    if USER_AGENTS:
        return random.choice(USER_AGENTS)
    return "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"

def launch_minimized_playwright_browser(playwright, target_url, wait_seconds=7):
    user_agent = get_random_user_agent()
    browser_type = playwright.chromium
    browser = browser_type.launch(headless=False, args=["--window-position=0,1000", "--window-size=1280,800"])
    context = browser.new_context(user_agent=user_agent, viewport={"width": 1280, "height": 800}, locale="en-US")
    page = context.new_page()
    page.goto(target_url, timeout=60000)
    logger.info(f"[BROWSER] Playwright launched (minimized) with User-Agent: {user_agent}")
    logger.info(f"[BROWSER] Waiting {wait_seconds} seconds for page to load...")
    time.sleep(wait_seconds)
    return browser, context, page, user_agent

def detect_cloudflare_captcha(page):
    html = page.content().lower()
    for indicator in CLOUDFLARE_CAPTCHA_INDICATORS:
        if indicator.lower() in html:
            logger.warning(f"[CAPTCHA] Detected Cloudflare CAPTCHA indicator: '{indicator}'")
            return True
    return False

def relaunch_maximized_for_captcha(playwright, target_url, user_agent, timeout=300):
    browser_type = playwright.chromium
    browser = browser_type.launch(headless=False, args=["--start-maximized"])
    context = browser.new_context(user_agent=user_agent, viewport={"width": 1920, "height": 1080}, locale="en-US")
    page = context.new_page()
    page.goto(target_url, timeout=60000)
    logger.info("[CAPTCHA] Relaunched browser in maximized mode for manual CAPTCHA resolution.")
    start_time = time.time()
    while time.time() - start_time < timeout:
        html = page.content().lower()
        if not any(indicator.lower() in html for indicator in CLOUDFLARE_CAPTCHA_INDICATORS):
            logger.info("[CAPTCHA] CAPTCHA appears to be cleared by user.")
            return browser, context, page
        logger.info("[CAPTCHA] Waiting for user to solve CAPTCHA...")
        time.sleep(5)
    logger.error("[CAPTCHA] Timeout waiting for user to solve CAPTCHA.")
    return None, None, None

def prompt_user_for_selenium_retry():
    rprint("[yellow][CAPTCHA] CAPTCHA could not be solved or a persistent loading screen was detected.[/yellow]")
    user_input = input("Would you like to retry in Selenium stealth mode? (y/n): ").strip().lower()
    return user_input == "y"

def launch_selenium_stealth(target_url, user_agent):
    from ..utils.seleniumbase_launcher import launch_browser as sb_launch
    _, _, driver = sb_launch(user_agent=user_agent, headless=True)
    driver.get(target_url)
    logger.info("[BROWSER] SeleniumBase launched in stealth mode.")
    return driver

def browser_pipeline(playwright, target_url, cache_exit_callback=None):
    """
    Main browser utility for html_election_parser.
    Returns (browser, context, page, user_agent) or None if session should exit.
    """
    # Step 1: Launch minimized Playwright browser and load page
    browser, context, page, user_agent = launch_minimized_playwright_browser(playwright, target_url)
    # Step 2: Detect CAPTCHA
    if not detect_cloudflare_captcha(page):
        logger.info("[CAPTCHA] No CAPTCHA detected. Continuing pipeline.")
        return browser, context, page, user_agent

    # Step 3: CAPTCHA detected, relaunch maximized for user intervention
    browser.close()
    browser, context, page = relaunch_maximized_for_captcha(playwright, target_url, user_agent)
    if browser and not detect_cloudflare_captcha(page):
        logger.info("[CAPTCHA] CAPTCHA cleared after user intervention. Continuing pipeline.")
        return browser, context, page, user_agent

    # Step 4: If still CAPTCHA or loading, prompt for Selenium retry
    if prompt_user_for_selenium_retry():
        driver = launch_selenium_stealth(target_url, user_agent)
        # Re-run CAPTCHA detection in Selenium (pseudo-code, adapt as needed)
        # If solved, return driver; else, exit
        # For now, just exit after retry
        logger.info("[CAPTCHA] Selenium retry complete. Exiting session.")
        if cache_exit_callback:
            cache_exit_callback(target_url, status="captcha_failed")
        return None, None, None, user_agent
    else:
        logger.info("[CAPTCHA] User chose to exit gracefully. Exiting session.")
        if cache_exit_callback:
            cache_exit_callback(target_url, status="captcha_exit")
        return None, None, None, user_agent