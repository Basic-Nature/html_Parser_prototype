# utils/captcha_tools.py
# ---------------------------------------------------------------
# CAPTCHA detection and user-intervention handler (browser-agnostic)
# ---------------------------------------------------------------

import time
import os
import platform
from ..utils.logger_instance import logger
from ..utils.shared_logger import rprint
import json
from ..config import CONTEXT_LIBRARY_PATH

# Load CAPTCHA indicators from context library
if os.path.exists(CONTEXT_LIBRARY_PATH):
    with open(CONTEXT_LIBRARY_PATH, "r", encoding="utf-8") as f:
        CONTEXT_LIBRARY = json.load(f)
    CLOUDFLARE_CAPTCHA_INDICATORS = CONTEXT_LIBRARY.get("cloudflare_captcha_indicators", [])
else:
    logger.error("[captcha_tools] context_library.json not found. CAPTCHA detection will be limited.")
    CLOUDFLARE_CAPTCHA_INDICATORS = []

DEFAULT_CAPTCHA_TIMEOUT = int(os.getenv("CAPTCHA_TIMEOUT", "300"))
POLL_INTERVAL = 5

def detect_cloudflare_challenge(page_or_driver, indicators=None):
    """
    Scans the current page content for common Cloudflare CAPTCHA/challenge keywords.
    Accepts either a Playwright page or SeleniumBase driver.
    """
    indicators = indicators or CLOUDFLARE_CAPTCHA_INDICATORS
    try:
        html = get_page_content(page_or_driver).lower()
        return any(keyword.lower() in html for keyword in indicators)
    except Exception as e:
        logger.error(f"[CAPTCHA] Error reading content: {e}")
        return False

def get_page_content(page_or_driver):
    """
    Returns the HTML content from a Playwright page or SeleniumBase driver.
    """
    # Playwright Page
    if hasattr(page_or_driver, "content"):
        return page_or_driver.content()
    # SeleniumBase Driver
    if hasattr(page_or_driver, "page_source"):
        return page_or_driver.page_source
    raise RuntimeError("Unsupported browser object for content extraction.")

def bring_to_front(page_or_driver):
    """
    Attempts to bring the browser window to the foreground.
    """
    os_type = platform.system()
    try:
        # Playwright Page
        if hasattr(page_or_driver, "bring_to_front"):
            page_or_driver.bring_to_front()
        # SeleniumBase Driver (Chrome/Edge)
        elif hasattr(page_or_driver, "maximize_window"):
            page_or_driver.maximize_window()
        # OS-level foreground
        if os_type == "Windows":
            import ctypes
            ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 9)  # SW_RESTORE
            ctypes.windll.user32.SetForegroundWindow(ctypes.windll.kernel32.GetConsoleWindow())
        elif os_type == "Darwin":  # macOS
            os.system("osascript -e 'tell application \"System Events\" to set frontmost of the first process whose unix id is (do shell script \"echo $PPID\") to true'")
        elif os_type == "Linux":
            os.system("xdotool windowactivate $(xdotool search --onlyvisible --name 'Chromium' | head -1) 2>/dev/null")
    except Exception as e:
        logger.warning(f"[CAPTCHA] Foreground window fallback failed: {e}")

def is_cloudflare_captcha_present(page_or_driver) -> bool:
    """
    Returns True if a Cloudflare CAPTCHA is detected on the page.
    """
    try:
        html = get_page_content(page_or_driver).lower()
        return any(keyword.lower() in html for keyword in CLOUDFLARE_CAPTCHA_INDICATORS)
    except Exception as e:
        logger.error(f"[CAPTCHA] Failed reading page content: {e}")
        return False

def wait_for_user_to_solve_captcha(page_or_driver, timeout: int = DEFAULT_CAPTCHA_TIMEOUT):
    """
    Waits for manual CAPTCHA resolution by checking if challenge elements disappear.
    Works for both Playwright and SeleniumBase.
    """
    logger.info(f"[CAPTCHA] Waiting up to {timeout} seconds for CAPTCHA to be solved...")
    start = time.time()
    retries = 0
    while time.time() - start < timeout:
        try:
            if not is_cloudflare_captcha_present(page_or_driver):
                logger.info("[CAPTCHA] CAPTCHA resolved — continuing.")
                return True
            if retries % 3 == 0:
                try:
                    bring_to_front(page_or_driver)
                except Exception as e:
                    logger.debug(f"[CAPTCHA] Could not bring browser to front: {e}")
            time.sleep(POLL_INTERVAL)
            retries += 1
        except Exception as e:
            logger.error(f"[CAPTCHA] CAPTCHA monitoring failed: {e}")
            break
    logger.warning("[CAPTCHA] CAPTCHA not resolved within timeout.")
    return False