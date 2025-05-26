from seleniumbase import Driver
import time
import os

def launch_browser(user_agent=None, headless=True, proxy=None):
    """
    Launch SeleniumBase browser with stealth and custom options.
    Returns (None, None, driver) for compatibility with Playwright tuple.
    """
    driver_kwargs = {
        "uc": True,
        "headless": headless,
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
    driver.get(url)
    try:
        driver.maximize_window()
    except Exception:
        pass
    print(f"[SeleniumBase] Please solve the CAPTCHA manually in the browser window.")
    print(f"[SeleniumBase] Waiting up to {timeout} seconds...")
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
            print("[SeleniumBase] CAPTCHA appears to be cleared.")
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
    driver.get(url)
    return None, None, driver, user_agent

def close_driver(driver):
    """
    Safely close the SeleniumBase driver.
    """
    try:
        driver.quit()
    except Exception:
        pass