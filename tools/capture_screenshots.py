from playwright.sync_api import sync_playwright
import os

URL = "http://127.0.0.1:5000/"
WIDTHS = [360, 768, 1024, 1280]
OUT_DIR = os.path.join(os.path.dirname(__file__), "debug_screenshots")

os.makedirs(OUT_DIR, exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    try:
        for w in WIDTHS:
            h = 900
            ctx = browser.new_context(viewport={"width": w, "height": h})
            page = ctx.new_page()
            print(f"Navigating to {URL} at {w}x{h}...")
            try:
                page.goto(URL, timeout=30000)
                # give page a moment to settle
                page.wait_for_timeout(800)
                filename = f"shot_{w}x{h}.png"
                outpath = os.path.join(OUT_DIR, filename)
                page.screenshot(path=outpath, full_page=True)
                print(f"Saved: {outpath}")
            except Exception as e:
                print(f"Failed to capture {w}px: {e}")
            finally:
                ctx.close()
    finally:
        browser.close()

print("Done.")
