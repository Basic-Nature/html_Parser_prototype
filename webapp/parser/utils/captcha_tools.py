# utils/captcha_tools.py
# ---------------------------------------------------------------
# CAPTCHA detection and user-intervention handler for Cloudflare
# ---------------------------------------------------------------

import time
import os
import platform
from playwright.sync_api import Page, sync_playwright, Error as PlaywrightError
from ..utils.browser_utils import relaunch_browser_fullscreen_if_needed
from ..utils.shared_logger import logger, rprint
# Load timeout from environment (default to 300 seconds)
DEFAULT_CAPTCHA_TIMEOUT = int(os.getenv("CAPTCHA_TIMEOUT", "300"))
POLL_INTERVAL = 5

# Cloudflare CAPTCHA and challenge keywords for detection
CLOUDFLARE_CAPTCHA_INDICATORS = [
    "verify you are human",
    "checking if the site connection is secure",
    "enable javascript and cookies to continue",
    "ray id:",
    "performance & security by cloudflare",
    "cf-turnstile-response",
    "challenge-platform",
    "just a moment..."
]

def detect_cloudflare_challenge(page, indicators=None):
    """
    Scans the current page content for common Cloudflare CAPTCHA/challenge keywords.

    Args:
        page: Playwright page object.
        indicators (list): Optional custom indicators to check. Defaults to CLOUDFLARE_CAPTCHA_INDICATORS.

    Returns:
        bool: True if indicators are found, False otherwise.
    """
    indicators = indicators or CLOUDFLARE_CAPTCHA_INDICATORS
    try:
        content = page.content().lower()
        return any(keyword.lower() in content for keyword in indicators)
    except PlaywrightError as e:
        logger.error(f"[CAPTCHA] Error reading content: {e}")
        return False

def force_foreground():
    """Attempt to bring the browser or console window to front, platform-specific."""
    os_type = platform.system()
    try:
        if os_type == "Windows":
            import ctypes
            ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 9)  # SW_RESTORE
            ctypes.windll.user32.SetForegroundWindow(ctypes.windll.kernel32.GetConsoleWindow())
        elif os_type == "Darwin":  # macOS
            os.system("osascript -e 'tell application \"System Events\" to set frontmost of the first process whose unix id is (do shell script \"echo $PPID\") to true'")
        elif os_type == "Linux":
            # Linux often requires external tools like wmctrl or xdotool
            os.system("xdotool windowactivate $(xdotool search --onlyvisible --name 'Chromium' | head -1) 2>/dev/null")
        else:
            logger.info(f"[CAPTCHA] No foreground handling for OS: {os_type}")
    except Exception as e:
        logger.warning(f"[CAPTCHA] Foreground window fallback failed: {e}")

def is_cloudflare_captcha_present(page: Page) -> bool:
    try:
        html = page.content().lower()
        return any(keyword.lower() in html for keyword in CLOUDFLARE_CAPTCHA_INDICATORS)
    except PlaywrightError as e:
        logger.error(f"[CAPTCHA] Failed reading page content: {e}")
        return False

def wait_for_user_to_solve_captcha(page: Page, timeout: int = DEFAULT_CAPTCHA_TIMEOUT):
    """
    Waits for manual CAPTCHA resolution by checking if challenge elements disappear.
    """
    logger.info(f"[CAPTCHA] Waiting up to {timeout} seconds for CAPTCHA to be solved...")
    start = time.time()
    retries = 0
    while time.time() - start < timeout:
        try:
            if not is_cloudflare_captcha_present(page):
                logger.info("[CAPTCHA] CAPTCHA resolved — continuing.")
                return True
            if retries % 3 == 0:
                try:
                    page.bring_to_front()
                    force_foreground()
                except Exception as e:
                    logger.debug(f"[CAPTCHA] Could not bring browser to front: {e}")
            time.sleep(POLL_INTERVAL)
            retries += 1
        except PlaywrightError as e:
            logger.error(f"[CAPTCHA] CAPTCHA monitoring failed: {e}")
            break
    logger.warning("[CAPTCHA] CAPTCHA not resolved within timeout.")
    return False

def handle_cloudflare_captcha(playwright, page: Page, original_url: str, timeout: int = DEFAULT_CAPTCHA_TIMEOUT):
    """
    Detect and respond to CAPTCHA challenge. Relaunch GUI browser if needed.
    """
    logger.info("[INFO] Checking for CAPTCHA indicators...")
    if not is_cloudflare_captcha_present(page):
        logger.info("[INFO] No CAPTCHA detected.")
        return None

    logger.warning("[CAPTCHA] CAPTCHA triggered. Relaunching browser for manual resolution.")

    browser, context, new_page, user_agent = relaunch_browser_fullscreen_if_needed(
        playwright,
        url=original_url,
        timeout=timeout
    )

    wait_for_user_to_solve_captcha(new_page, timeout)

    return browser, context, new_page, user_agent
