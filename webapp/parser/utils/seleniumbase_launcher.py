from __future__ import annotations

import time
from typing import Optional

# webapp/parser/utils/seleniumbase_launcher.py
# -----------------------------------------------------------------------------------
# This file contains functions to launch and manage SeleniumBase browsers
# for web scraping and automation tasks, including handling CAPTCHAs and stealth mode.
# -----------------------------------------------------------------------------------
try:
    from seleniumbase import Driver as _SeleniumBaseDriver
except ImportError as exc:  # pragma: no cover - optional dependency
    _SeleniumBaseDriver = None
    _SELENIUMBASE_IMPORT_ERROR: Optional[Exception] = exc
else:  # pragma: no cover - exercised only when dependency is installed
    _SELENIUMBASE_IMPORT_ERROR = None


class _MissingDriver:
    """Raised when SeleniumBase is requested without the optional dependency."""

    def __init__(self, *args, **kwargs) -> None:  # pylint: disable=unused-argument
        message = (
            "SeleniumBase is not installed. Install it with `pip install seleniumbase` "
            "to enable the manual CAPTCHA fallback."
        )
        raise RuntimeError(message) from _SELENIUMBASE_IMPORT_ERROR


Driver = _SeleniumBaseDriver or _MissingDriver  # type: ignore[assignment]
SELENIUMBASE_AVAILABLE = _SeleniumBaseDriver is not None

from ..config import HEADLESS_DEFAULT
from .logger_singleton import logger


def launch_browser(user_agent=None, headless=None, proxy=None):
    """
    Launch SeleniumBase browser with stealth and custom options.
    Returns (None, None, driver) for compatibility with Playwright tuple.
    """
    launch_headless = HEADLESS_DEFAULT if headless is None else headless
    driver_kwargs = {
        "uc": True,
        "headless": launch_headless,
    }
    if user_agent:
        driver_kwargs["user_agent"] = user_agent
    if proxy:
        driver_kwargs["proxy"] = proxy
    driver = Driver(**driver_kwargs)
    return None, None, driver

def relaunch_browser_fullscreen_if_needed(_, url, timeout=300, user_agent=None, proxy=None):
    """
    Relaunch SeleniumBase browser in GUI mode for manual CAPTCHA solving.
    Maximizes window, navigates to URL, and waits for user to solve CAPTCHA.
    Returns driver in (None, None, driver) tuple.
    """
    driver_kwargs = {
        "uc": True,
        "headless": False,
    }
    if user_agent:
        driver_kwargs["user_agent"] = user_agent
    if proxy:
        driver_kwargs["proxy"] = proxy
    driver = Driver(**driver_kwargs)
    driver.get(url, [])
    try:
        driver.maximize_window()
    except Exception:
        pass
    logger.info("[SeleniumBase] Please solve the CAPTCHA manually in the browser window.")
    logger.info(f"[SeleniumBase] Waiting up to {timeout} seconds...")
    start = time.time()
    while time.time() - start < timeout:
        # Simple check: look for common Cloudflare challenge indicators in page source
        html = driver.page_source.lower()
        if not any(x in html for x in [
            "verify you are human",
            "checking if the site connection is secure",
            "enable javascript and cookies to continue",
            "performance & security by cloudflare",
            "cf-turnstile-response",
            "challenge-platform",
            "just a moment..."
        ]):
            logger.info("[SeleniumBase] CAPTCHA appears to be cleared.")
            break
        time.sleep(5)
    return driver

def relaunch_browser_stealth(_, url, user_agent=None, proxy=None):
    """
    Relaunch SeleniumBase browser in stealth mode after persistent CAPTCHA.
    Returns (None, None, driver, user_agent)
    """
    driver_kwargs = {
        "uc": True,
        "headless": True,
    }
    if user_agent:
        driver_kwargs["user_agent"] = user_agent
    if proxy:
        driver_kwargs["proxy"] = proxy
    driver = Driver(**driver_kwargs)
    driver.get(url, [])
    return None, None, driver, user_agent

def close_driver(driver):
    """
    Safely close the SeleniumBase driver.
    """
    try:
        driver.quit()
    except Exception:
        pass