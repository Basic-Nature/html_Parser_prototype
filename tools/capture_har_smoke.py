#!/usr/bin/env python3
"""Headless smoke that captures network requests/responses to a HAR-like JSON.

Usage: python tools/capture_har_smoke.py [url]
"""
from playwright.sync_api import sync_playwright
import json
import os
import time
import sys

OUT_DIR = os.path.join("tools", "debug_headless_output")
os.makedirs(OUT_DIR, exist_ok=True)

url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:5000/ballot_lens"
records = []

def safe_headers(h):
    try:
        return dict(h)
    except Exception:
        return {}

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    # Use a narrower viewport to encourage navbar overflow so the "More" toggle appears
    # Try a smaller width to trigger overflow in the navbar (mobile/compact view)
    context = browser.new_context(viewport={"width": 700, "height": 800})
    page = context.new_page()

    def on_request(req):
        try:
            pd = None
            try:
                pd = req.post_data()
            except Exception:
                pd = None
            records.append({
                "type": "request",
                "url": req.url,
                "method": req.method,
                "headers": safe_headers(req.headers),
                "post_data": pd,
                "timestamp": int(time.time() * 1000),
            })
        except Exception:
            pass

    def on_response(resp):
        try:
            body = None
            try:
                body = resp.text()
                # Truncate very large bodies
                if isinstance(body, str) and len(body) > 200000:
                    body = body[:200000] + "\n...(truncated)"
            except Exception:
                body = None
            records.append({
                "type": "response",
                "url": resp.url,
                "status": resp.status,
                "headers": safe_headers(resp.headers),
                "body": body,
                "timestamp": int(time.time() * 1000),
            })
        except Exception:
            pass

    page.on("request", on_request)
    page.on("response", on_response)

    try:
        print(f"Visiting: {url}")
        page.goto(url, timeout=30000)
    except Exception as e:
        print(f"Navigation error: {e}")

    # Allow time for background requests / socket connects
    time.sleep(6)

    def try_click_visible(selector, timeout=1500, retries=3):
        for _ in range(retries):
            try:
                page.wait_for_selector(selector, state='visible', timeout=timeout)
                loc = page.locator(selector)
                try:
                    loc.scroll_into_view_if_needed(timeout=timeout)
                except Exception:
                    pass
                loc.click()
                return True
            except Exception:
                time.sleep(0.2)
        return False

    # Try to open parent/menu toggler(s) that reveal nav-more, then click nav-more.
    try:
        toggler_selectors = [
            '#navToggle', '#btnNavToggle', '.nav-toggler', '.navbar-toggler',
            '#btnMenu', '[data-testid="nav-toggle"]', '.btn-nav-toggle', '[aria-controls="nav-more"]'
        ]
        toggled = False
        for sel in toggler_selectors:
            if try_click_visible(sel, timeout=1500, retries=2):
                print(f'Clicked toggler: {sel}')
                toggled = True
                time.sleep(0.5)
                break
            else:
                print(f'Failed clicking toggler {sel}')

        # Try clicking the nav-more control, with JS and DOM fallbacks
        if try_click_visible('#btnNavMore', timeout=3000, retries=3):
            print('Clicked #btnNavMore')
        else:
            print('Nav-more not visible or not found after toggler attempts')
    except Exception as e:
        print('Error while attempting to reveal/click nav-more:', e)

    # Allow additional time after interactions
    time.sleep(8)

    # Save snapshot and screenshot
    try:
        html = page.content()
        with open(os.path.join(OUT_DIR, "capture_page.html"), "w", encoding="utf-8") as f:
            f.write(html)
    except Exception:
        pass
    try:
        page.screenshot(path=os.path.join(OUT_DIR, "capture_screenshot.png"))
    except Exception:
        pass

    # Write records to file
    out_path = os.path.join(OUT_DIR, "network_capture.json")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        print(f"Wrote network capture to: {out_path}")
    except Exception as e:
        print(f"Failed to write capture: {e}")

    try:
        context.close()
        browser.close()
    except Exception:
        pass

print("Done")
