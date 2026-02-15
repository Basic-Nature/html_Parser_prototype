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
    
    Enhanced: Captures DOM structure transitions for ML training.
    """
    logger.info(f"[CAPTCHA] Waiting up to {timeout} seconds for CAPTCHA to be solved...")
    start = time.time()
    retries = 0
    initial_dom_snapshot = None
    
    while time.time() - start < timeout:
        try:
            # Capture initial challenge state on first iteration
            if retries == 0:
                try:
                    initial_dom_snapshot = _capture_captcha_dom_state(page_or_driver, "challenge_present")
                except Exception as snap_exc:
                    logger.debug(f"[CAPTCHA-NLP] Initial snapshot failed: {snap_exc}")
            
            if not is_cloudflare_captcha_present(page_or_driver):
                logger.info("[CAPTCHA] CAPTCHA resolved — continuing.")
                # Capture cleared state and log transition for ML training
                try:
                    cleared_snapshot = _capture_captcha_dom_state(page_or_driver, "challenge_cleared")
                    _log_captcha_transition(
                        initial_state=initial_dom_snapshot,
                        cleared_state=cleared_snapshot,
                        time_to_clear=time.time() - start
                    )
                except Exception as trans_exc:
                    logger.debug(f"[CAPTCHA-NLP] Transition logging failed: {trans_exc}")
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


def _capture_captcha_dom_state(page_or_driver, state_label: str) -> dict:
    """
    Capture DOM structure snapshot during CAPTCHA interaction.
    
    Args:
        page_or_driver: Playwright page or SeleniumBase driver
        state_label: "challenge_present" or "challenge_cleared"
        
    Returns:
        Dict with HTML snippet, indicators, and element counts
    """
    try:
        html_content = get_page_content(page_or_driver)
        html_snippet = html_content[:1000] if html_content else ""
        
        from .browser_utils import CLOUDFLARE_CAPTCHA_INDICATORS
        indicators_matched = [
            kw for kw in CLOUDFLARE_CAPTCHA_INDICATORS 
            if kw.lower() in html_content.lower()
        ] if html_content else []
        
        return {
            "state": state_label,
            "html_snippet": html_snippet,
            "indicators_matched": indicators_matched,
            "html_length": len(html_content) if html_content else 0,
            "timestamp": time.time()
        }
    except Exception as exc:
        logger.debug(f"[CAPTCHA-NLP] DOM state capture failed: {exc}")
        return {"state": state_label, "error": str(exc)}


def _log_captcha_transition(initial_state: dict, cleared_state: dict, time_to_clear: float) -> None:
    """
    Log CAPTCHA DOM state transition for supervised ML training.
    
    Builds dataset for:
    - Automated CAPTCHA type classification (Cloudflare vs reCAPTCHA vs custom)
    - Challenge resolution time prediction
    - Navigation recipe optimization (skip auto-actions during challenges)
    
    Args:
        initial_state: DOM snapshot when challenge was detected
        cleared_state: DOM snapshot after clearance
        time_to_clear: Seconds elapsed from detection to resolution
    """
    if not initial_state or not cleared_state:
        return
    
    try:
        import os

        import orjson

        from ..config import LOG_DIR
        
        transition_entry = {
            "captcha_type": "cloudflare",  # Could be enhanced to detect type from indicators
            "initial_indicators": initial_state.get("indicators_matched", []),
            "cleared_indicators": cleared_state.get("indicators_matched", []),
            "time_to_clear_seconds": time_to_clear,
            "html_delta_bytes": cleared_state.get("html_length", 0) - initial_state.get("html_length", 0),
            "initial_snippet": initial_state.get("html_snippet", "")[:500],
            "cleared_snippet": cleared_state.get("html_snippet", "")[:500],
            "timestamp": int(time.time())
        }
        
        log_path = os.path.join(LOG_DIR, "captcha_transition_log.jsonl")
        os.makedirs(LOG_DIR, exist_ok=True)
        
        with open(log_path, "ab") as f:
            f.write(orjson.dumps(transition_entry))
            f.write(b"\n")
        
        logger.debug(f"[CAPTCHA-NLP] Logged transition: {len(initial_state.get('indicators_matched', []))} → "
                    f"{len(cleared_state.get('indicators_matched', []))} indicators, {time_to_clear:.1f}s")
        
    except Exception as exc:
        logger.debug(f"[CAPTCHA-NLP] Transition logging failed: {exc}")
