import os
import sys
import time
from pathlib import Path

URL = os.environ.get('WEBAPP_URL', 'http://127.0.0.1:5000/')
VIEWPORTS = [(1280, 800), (1024, 768), (390, 844)]
OUT_DIR = Path('tools/screenshots')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Wait for server to be available
import requests

def wait_for_server(url, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code < 500:
                print(f"server OK: {url} (status={r.status_code})")
                return True
        except Exception as e:
            # print('.', end='', flush=True)
            pass
        time.sleep(1)
    return False

if not wait_for_server(URL, timeout=30):
    print(f"ERROR: Server not responding at {URL}")
    sys.exit(2)

# Try Playwright
try:
    from playwright.sync_api import sync_playwright
except Exception as e:
    print("Playwright not installed or not available. Install with: pip install playwright && playwright install")
    sys.exit(3)

try:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        for w, h in VIEWPORTS:
            print(f"Capturing {w}x{h}...")
            page.set_viewport_size({"width": w, "height": h})
            page.goto(URL, wait_until='networkidle')
            time.sleep(0.6)
            out = OUT_DIR / f"screenshot_{w}x{h}.png"
            page.screenshot(path=str(out), full_page=True)
            print(f"Saved {out}")
        browser.close()
    print("Screenshots completed.")
    sys.exit(0)
except Exception as e:
    print(f"Playwright run failed: {e}")
    sys.exit(4)
