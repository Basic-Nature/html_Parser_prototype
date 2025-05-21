# seleniumbase_launcher.py
from seleniumbase import Driver

def launch_browser():
    driver = Driver(uc=True, headless=False, browser="chrome")  # or browser="edge"
    print("[DEBUG] SeleniumBase browser launched — complete CAPTCHA if needed.")
    return None, None, driver  # mimic playwright return structure
