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
            # Capture DOM metadata after CAPTCHA resolution for pattern learning
            try:
                dom_metadata = _capture_post_captcha_dom_metadata(driver)
                if dom_metadata:
                    _log_captcha_resolution_data(url, start, dom_metadata)
            except Exception as meta_exc:
                logger.debug(f"[Selenium-NLP] DOM metadata capture failed: {meta_exc}")
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


def _capture_post_captcha_dom_metadata(driver) -> dict:
    """
    Capture DOM metadata after CAPTCHA resolution for pattern learning.
    
    This data helps train NLP models on post-CAPTCHA page structures
    and informs navigation recipes for Cloudflare-protected sites.
    
    Args:
        driver: SeleniumBase Driver instance
        
    Returns:
        Dict with DOM structure metrics and interactive element counts
    """
    try:
        dom_metadata = driver.execute_script("""
            return {
                interactive_elements: document.querySelectorAll('button, a[href], select, input[type="submit"]').length,
                form_count: document.forms.length,
                table_count: document.querySelectorAll('table').length,
                challenge_artifacts: document.querySelectorAll('[class*="cloudflare"], [id*="captcha"], [class*="challenge"]').length,
                heading_count: document.querySelectorAll('h1, h2, h3').length,
                body_text_length: document.body.innerText.length,
                viewport: {
                    width: window.innerWidth,
                    height: window.innerHeight
                }
            };
        """)
        return dom_metadata or {}
    except Exception as exc:
        logger.debug(f"[Selenium-NLP] DOM metadata JS execution failed: {exc}")
        return {}


def _log_captcha_resolution_data(url: str, start_time: float, dom_metadata: dict) -> None:
    """
    Log CAPTCHA resolution event with DOM transition data for ML analysis.
    
    Builds dataset for:
    - Automated CAPTCHA detection classifier
    - Post-challenge page structure patterns
    - Navigation recipe optimization
    
    Args:
        url: Target URL that triggered CAPTCHA
        start_time: Timestamp when CAPTCHA wait began
        dom_metadata: DOM structure after resolution
    """
    import os
    import time

    import orjson
    
    try:
        # Import config for LOG_DIR
        from ..config import LOG_DIR
        
        resolution_entry = {
            "url": url,
            "captcha_type": "cloudflare",
            "time_to_clear_seconds": time.time() - start_time,
            "dom_after_clearance": dom_metadata,
            "timestamp": int(time.time())
        }
        
        log_path = os.path.join(LOG_DIR, "captcha_resolution_log.jsonl")
        os.makedirs(LOG_DIR, exist_ok=True)
        
        with open(log_path, "ab") as f:
            f.write(orjson.dumps(resolution_entry))
            f.write(b"\n")
        
        logger.info(f"[Selenium-NLP] Logged CAPTCHA resolution: {dom_metadata.get('table_count', 0)} tables, "
                   f"{dom_metadata.get('interactive_elements', 0)} interactive elements")
        
    except Exception as exc:
        logger.debug(f"[Selenium-NLP] CAPTCHA resolution logging failed: {exc}")
