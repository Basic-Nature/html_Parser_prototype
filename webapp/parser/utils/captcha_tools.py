from __future__ import annotations

import ctypes
import os
import platform

# utils/captcha_tools.py
# ---------------------------------------------------------------
# CAPTCHA detection and user-intervention handler (browser-agnostic)
# ---------------------------------------------------------------
import time
from typing import Any, Protocol, runtime_checkable

import orjson

from ..config import CONTEXT_LIBRARY_PATH, DEFAULT_CAPTCHA_TIMEOUT
from .logger_singleton import logger
from .shared_logic import safe_get, safe_lower


@runtime_checkable
class HasContent(Protocol):
    def content(self) -> str: 
        """Returns the HTML content of the page."""
        ...

@runtime_checkable
class HasPageSource(Protocol):
    @property
    def page_source(self) -> str: 
        """Returns the HTML source of the page."""
        ...

@runtime_checkable
class HasBringToFront(Protocol):
    def bring_to_front(self) -> None: 
        """Brings the browser window to the front."""
        ...

@runtime_checkable
class HasMaximizeWindow(Protocol):
    def maximize_window(self) -> None: 
        """Maximizes the browser window."""
        ...

# Load CAPTCHA indicators from context library
if os.path.exists(CONTEXT_LIBRARY_PATH):
    with open(CONTEXT_LIBRARY_PATH, "rb") as f:
        CONTEXT_LIBRARY = orjson.loads(f.read())
    CLOUDFLARE_CAPTCHA_INDICATORS = safe_get(CONTEXT_LIBRARY, "cloudflare_captcha_indicators", [])
else:
    logger.error("[captcha_tools] context_library.json not found. CAPTCHA detection will be limited.")
    CLOUDFLARE_CAPTCHA_INDICATORS = []

POLL_INTERVAL = 5

def detect_cloudflare_challenge(page_or_driver, indicators=None) -> bool:
    """
    Scans the current page content for common Cloudflare CAPTCHA/challenge keywords.
    Accepts either a Playwright page or SeleniumBase driver.
    """
    indicators = indicators or CLOUDFLARE_CAPTCHA_INDICATORS
    try:
        html = safe_lower(get_page_content(page_or_driver))
        return any(safe_lower(keyword) in html for keyword in indicators)
    except Exception as e:
        logger.error(f"[CAPTCHA] Error reading content: {e}")
        return False

def get_page_content(page_or_driver: Any) -> str:
    """
    Returns the HTML content from a Playwright page or SeleniumBase driver.
    """
    if isinstance(page_or_driver, HasContent):
        return page_or_driver.content()
    if isinstance(page_or_driver, HasPageSource):
        return page_or_driver.page_source
    raise RuntimeError("Unsupported browser object for content extraction.")

def bring_to_front(page_or_driver: Any) -> None:
    """
    Attempts to bring the browser window to the foreground.
    Only runs OS-specific code on the correct platform.
    """
    os_type = platform.system()
    try:
        if isinstance(page_or_driver, HasBringToFront):
            page_or_driver.bring_to_front()
        elif isinstance(page_or_driver, HasMaximizeWindow):
            page_or_driver.maximize_window()
        # OS-level foreground (for local dev only)
        if os_type == "Windows":
            try:
                windll = getattr(ctypes, "windll", None)
                user32 = getattr(windll, "user32", None) if windll else None
                kernel32 = getattr(windll, "kernel32", None) if windll else None
                GetConsoleWindow = getattr(kernel32, "GetConsoleWindow", None) if kernel32 else None
                ShowWindow = getattr(user32, "ShowWindow", None) if user32 else None
                SetForegroundWindow = getattr(user32, "SetForegroundWindow", None) if user32 else None
                if GetConsoleWindow and ShowWindow and SetForegroundWindow:
                    hwnd = GetConsoleWindow()
                    if hwnd:
                        ShowWindow(hwnd, 9)  # SW_RESTORE
                        SetForegroundWindow(hwnd)
            except Exception as e:
                logger.debug(f"[CAPTCHA] Windows foreground fallback failed: {e}")
        elif os_type == "Darwin":  # macOS
            try:
                os.system("osascript -e 'tell application \"System Events\" to set frontmost of the first process whose unix id is (do shell script \"echo $PPID\") to true'")
            except Exception as e:
                logger.debug(f"[CAPTCHA] macOS foreground fallback failed: {e}")
        elif os_type == "Linux":
            try:
                os.system("xdotool windowactivate $(xdotool search --onlyvisible --name 'Chromium' | head -1) 2>/dev/null")
            except Exception as e:
                logger.debug(f"[CAPTCHA] Linux foreground fallback failed: {e}")
    except Exception as e:
        logger.warning(f"[CAPTCHA] Foreground window fallback failed: {e}")

def is_cloudflare_captcha_present(page_or_driver) -> bool:
    """
    Returns True if a Cloudflare CAPTCHA is detected on the page.
    """
    try:
        html = safe_lower(get_page_content(page_or_driver))
        return any(safe_lower(keyword) in html for keyword in CLOUDFLARE_CAPTCHA_INDICATORS)
    except Exception as e:
        logger.error(f"[CAPTCHA] Failed reading page content: {e}")
        return False

def wait_for_user_to_solve_captcha(page_or_driver, timeout: int = DEFAULT_CAPTCHA_TIMEOUT) -> bool:
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