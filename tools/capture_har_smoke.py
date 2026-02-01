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

    # Try to open parent/menu toggler(s) that reveal nav-more, then click nav-more.
    try:
        toggler_selectors = [
            '#navToggle', '#btnNavToggle', '.nav-toggler', '.navbar-toggler',
            '#btnMenu', '[data-testid="nav-toggle"]', '.btn-nav-toggle', '[aria-controls="nav-more"]'
        ]
        toggled = False
        for sel in toggler_selectors:
            try:
                loc = page.locator(sel)
                loc.wait_for(state='visible', timeout=1500)
                loc.click()
                print(f'Clicked toggler: {sel}')
                toggled = True
                time.sleep(0.5)
                break
            except Exception:
                # fallback JS click
                try:
                    page.evaluate("(s)=>{const el=document.querySelector(s); if(el) el.click();}", sel)
                    print(f'JS-clicked toggler (fallback): {sel}')
                    toggled = True
                    time.sleep(0.5)
                    break
                except Exception as e:
                    print(f'Failed clicking toggler {sel}:', e)

        # Try clicking the nav-more control, with JS and DOM fallbacks
        try:
            nav_more = page.locator('#btnNavMore')
            nav_more.wait_for(state='visible', timeout=3000)
            nav_more.click(force=True)
            print('Clicked #btnNavMore')
        except Exception:
            # JS click fallback
            try:
                page.evaluate("() => { const el = document.querySelector('#btnNavMore'); if(el) el.click(); }")
                print('Attempted JS click on #btnNavMore (no visibility)')
            except Exception:
                # Last-resort: temporarily un-inert nav dropdown and click
                try:
                    page.evaluate("() => { const dd = document.querySelector('#navMoreDropdown'); if(dd){ dd.__backup_inert = dd.inert; dd.__backup_aria = dd.getAttribute('aria-hidden'); dd.inert = false; dd.setAttribute('aria-hidden','false'); const links = dd.querySelectorAll('a'); for(const a of links){ a.setAttribute('tabindex','0'); } } }")
                    time.sleep(0.3)
                    try:
                        page.evaluate("() => { const b = document.querySelector('#btnNavMore'); if(b) b.click(); }")
                        print('Clicked #btnNavMore after temporary un-inert')
                    except Exception as e:
                        print('Failed click after un-inert:', e)
                    try:
                        page.evaluate("() => { const dd = document.querySelector('#navMoreDropdown'); if(dd && typeof dd.__backup_inert !== 'undefined'){ dd.inert = !!dd.__backup_inert; if(dd.__backup_aria!==null){ dd.setAttribute('aria-hidden', dd.__backup_aria);} else { dd.removeAttribute('aria-hidden'); } const links = dd.querySelectorAll('a'); for(const a of links){ a.removeAttribute('tabindex'); } delete dd.__backup_inert; delete dd.__backup_aria; } }")
                    except Exception:
                        pass
                except Exception as e:
                    print('Nav-more not visible or not found after toggler attempts and fallbacks:', e)
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
