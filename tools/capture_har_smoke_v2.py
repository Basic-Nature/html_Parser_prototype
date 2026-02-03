#!/usr/bin/env python3
"""Improved headless capture (safe single-file v2).

Usage: python tools/capture_har_smoke_v2.py [url]
"""
import json
import os
import sys
import time

from playwright.sync_api import sync_playwright

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

    print(f"Visiting: {url}")
    try:
        page.goto(url, timeout=30000)
    except Exception as e:
        print("Navigation error:", e)

    time.sleep(4)

    # Try a deterministic sequence of interactions to reveal nav-more
    toggler_selectors = [
        '#navToggle', '#btnNavToggle', '.nav-toggler', '.navbar-toggler',
        '#btnMenu', '[data-testid="nav-toggle"]', '.btn-nav-toggle', '[aria-controls="nav-more"]',
        '#sidebarToggleBtn', '#btnToggleRightSidebar'
    ]

    def try_click(sel):
        try:
            nodes = page.query_selector_all(sel)
            if not nodes:
                return False
            for n in nodes:
                try:
                    n.click()
                    print('Clicked toggler:', sel)
                    return True
                except Exception:
                    try:
                        page.evaluate("(s)=>{const el=document.querySelector(s); if(el) el.click();}", sel)
                        print('JS-clicked toggler (fallback):', sel)
                        return True
                    except Exception:
                        continue
        except Exception:
            return False
        return False

    # Try each toggler quickly
    toggled = False
    for s in toggler_selectors:
        if try_click(s):
            toggled = True
            time.sleep(0.4)
            break

    # Trigger resize and hover and dispatch small events
    try:
        page.evaluate("() => { window.dispatchEvent(new Event('resize')); }")
    except Exception:
        pass
    try:
        page.evaluate("() => { const nb=document.querySelector('.navbar-content'); if(nb){ nb.dispatchEvent(new MouseEvent('mouseenter',{bubbles:true})); nb.dispatchEvent(new MouseEvent('mousemove',{bubbles:true})); } }")
    except Exception:
        pass

    # Call candidate global helpers heuristically (safe one-liner)
    try:
        page.evaluate("() => { Object.getOwnPropertyNames(window||{}).forEach(n=>{ try{ if(/nav|Nav/.test(n) && /(toggle|Overflow|overflow|sync)/i.test(n)){ const f=window[n]; if(typeof f==='function') try{ f(); }catch(e){} } }catch(e){} }); }")
    except Exception:
        pass

    time.sleep(0.8)

    # Ask app to manage overlay focus/inert state if helper is present
    try:
        page.evaluate("() => { try{ if(typeof window.manageOverlayFocus === 'function'){ window.manageOverlayFocus('#navMoreDropdown', true); } }catch(e){} }")
        try:
            page.wait_for_function("() => { const d = document.getElementById('navMoreDropdown'); return d && !d.hasAttribute('inert') && d.getAttribute('aria-hidden') === 'false'; }", timeout=800)
        except Exception:
            pass
    except Exception:
        pass

    # Try clicking #btnNavMore; wait for dropdown to become interactive, fallback to safe un-inert for debug
    try:
        btns = page.query_selector_all('#btnNavMore')
        if btns:
            try:
                btns[0].click()
                # wait until dropdown is toggled to interactive by client JS
                page.wait_for_function("() => { const d = document.getElementById('navMoreDropdown'); return d && !d.hasAttribute('inert') && d.getAttribute('aria-hidden') === 'false'; }", timeout=1500)
                print('Clicked #btnNavMore and dropdown became interactive')
            except Exception:
                # fallback: JS-click then ensure dropdown is interactive for capture
                try:
                    page.evaluate("() => { const b = document.querySelector('#btnNavMore'); if(b) b.click(); }")
                except Exception:
                    pass

        # If still inert/hidden, prefer asking the app helper to open it; only then use debug fallback
        dd = page.query_selector('#navMoreDropdown')
        if dd:
            attrs = page.evaluate("(el)=>{ return {inert: el.inert===true, aria: el.getAttribute('aria-hidden')}; }", dd)
            if attrs and (attrs.get('inert') or attrs.get('aria')=='true'):
                # Try app helper first
                try:
                    page.evaluate("() => { try{ if(typeof window.manageOverlayFocus === 'function'){ window.manageOverlayFocus('#navMoreDropdown', true); } }catch(e){} }")
                    try:
                        page.wait_for_function("() => { const d = document.getElementById('navMoreDropdown'); return d && !d.hasAttribute('inert') && d.getAttribute('aria-hidden') === 'false'; }", timeout=700)
                        print('manageOverlayFocus opened dropdown')
                    except Exception:
                        pass
                except Exception:
                    pass

                # If still not interactive, perform temporary debug-only attribute edits
                attrs2 = page.evaluate("(el)=>{ return {inert: el.inert===true, aria: el.getAttribute('aria-hidden')}; }", dd)
                if attrs2 and (attrs2.get('inert') or attrs2.get('aria')=='true'):
                    try:
                        page.evaluate("() => { const dd = document.querySelector('#navMoreDropdown'); if(!dd) return; dd.__backup_inert = dd.inert; dd.__backup_aria = dd.getAttribute('aria-hidden'); dd.inert = false; dd.setAttribute('aria-hidden','false'); dd.querySelectorAll('a').forEach(a=>{ a.setAttribute('tabindex','0'); }); }")
                        time.sleep(0.2)
                        try:
                            page.evaluate("() => { const b=document.querySelector('#btnNavMore'); if(b) b.click(); }")
                            print('Clicked #btnNavMore after un-inert fallback')
                        except Exception:
                            pass
                    finally:
                        try:
                            page.evaluate("() => { const dd = document.querySelector('#navMoreDropdown'); if(!dd) return; if(typeof dd.__backup_inert!=='undefined'){ dd.inert = !!dd.__backup_inert; if(dd.__backup_aria!==null){ dd.setAttribute('aria-hidden', dd.__backup_aria); } else { dd.removeAttribute('aria-hidden'); } dd.querySelectorAll('a').forEach(a=>{ a.removeAttribute('tabindex'); }); delete dd.__backup_inert; delete dd.__backup_aria; } }")
                        except Exception:
                            pass
    except Exception as e:
        print('Nav reveal/click flow failed:', e)

    # give it a moment
    time.sleep(4)

    # Save snapshot and screenshot
    try:
        html = page.content()
        with open(os.path.join(OUT_DIR, 'capture_page.html'), 'w', encoding='utf-8') as f:
            f.write(html)
    except Exception:
        pass
    try:
        page.screenshot(path=os.path.join(OUT_DIR, 'capture_screenshot.png'))
    except Exception:
        pass

    out_path = os.path.join(OUT_DIR, 'network_capture.json')
    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2)
        print('Wrote network capture to:', out_path)
    except Exception as e:
        print('Failed to write capture:', e)

    try:
        context.close()
        browser.close()
    except Exception:
        pass

print('Done')
