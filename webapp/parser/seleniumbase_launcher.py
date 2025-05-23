# seleniumbase_launcher.py
from seleniumbase import Driver
from utils.shared_logger import log_debug, log_info, log_warning, log_error
from utils.captcha_tools import wait_for_user_to_solve_captcha
from utils.browser_utils import force_foreground
from utils.shared_logger import logger


def launch_browser():
    driver = Driver(uc=True, headless=False, browser="chrome")  # or browser="edge"
    print("[DEBUG] SeleniumBase browser launched — complete CAPTCHA if needed.")
    return None, None, driver  # mimic playwright return structure

def relaunch_browser_fullscreen_if_needed(driver, url: str, timeout: int = 300):
    """
    Relaunches the browser in non-headless fullscreen mode if CAPTCHA is detected.
    """
    driver.get(url)
    driver.maximize_window()
    logger.info("[INFO] Browser maximized for CAPTCHA resolution.")
    force_foreground(driver)
    wait_for_user_to_solve_captcha(driver, timeout)


