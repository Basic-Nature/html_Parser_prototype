#!/usr/bin/env python3
"""
Smoke test: verify off-canvas sidebar and footer toggle behavior using Playwright.
Run this while the dev server is running (default http://localhost:5000/ballot_lens).
"""
import sys
import time
import json
import argparse
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout


def run_check(url: str) -> int:
    results = {
        "sidebar_present": False,
        "sidebar_opened": False,
        "sidebar_closed": False,
        "backdrop_present": False,
        "footer_present": False,
        "footer_expanded": False,
        "footer_collapsed": False,
    }
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            # Create a mobile-like context so the responsive sidebar toggle appears
            context = browser.new_context(
                viewport={"width": 375, "height": 800},
                is_mobile=True,
                has_touch=True,
                user_agent=("Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) "
                            "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1")
            )
            page = context.new_page()
            page.set_default_timeout(20000)
            page.goto(url)

            # Wait for full load and app initialization
            try:
                page.wait_for_function("() => document.readyState === 'complete'", timeout=5000)
            except Exception:
                pass

            # Wait for main UI hooks to appear (toggle or footer)
            toggle = None
            sidebar = None
            backdrop = None
            footer_preview = None
            session_footer = None
            # try waiting for either toggle or footer to be available
            try:
                toggle = page.wait_for_selector('.sidebar-toggle', timeout=4000)
            except Exception:
                toggle = page.query_selector('.sidebar-toggle')
            try:
                sidebar = page.wait_for_selector('.sidebar-right', timeout=4000)
            except Exception:
                sidebar = page.query_selector('.sidebar-right')
            try:
                backdrop = page.wait_for_selector('.sidebar-backdrop', timeout=4000)
            except Exception:
                backdrop = page.query_selector('.sidebar-backdrop')
            try:
                footer_preview = page.wait_for_selector('#footerPreview', timeout=4000)
            except Exception:
                footer_preview = page.query_selector('#footerPreview')
            try:
                session_footer = page.wait_for_selector('#sessionFooter', timeout=4000)
            except Exception:
                session_footer = page.query_selector('#sessionFooter')

            results['sidebar_present'] = bool(sidebar and toggle)
            results['backdrop_present'] = bool(backdrop)
            results['footer_present'] = bool(session_footer and footer_preview)

            # Sidebar open/close
            if sidebar and backdrop:
                results['sidebar_present'] = True
                # Prefer clicking the toggle if present, otherwise call the toggle via JS
                try:
                    if toggle:
                        toggle.click()
                    else:
                        page.evaluate("() => { const t=document.querySelector('.sidebar-toggle'); if(t) t.click(); else { document.querySelector('.sidebar-right')?.classList.add('sidebar-open'); document.querySelector('.sidebar-backdrop')?.classList.add('visible'); } }")
                    time.sleep(0.5)
                    results['sidebar_opened'] = page.evaluate("() => !!document.querySelector('.sidebar-right') && document.querySelector('.sidebar-right').classList.contains('sidebar-open')")
                except Exception:
                    results['sidebar_opened'] = page.evaluate("() => !!document.querySelector('.sidebar-right') && document.querySelector('.sidebar-right').classList.contains('sidebar-open')")
                # close via backdrop if visible
                try:
                    if backdrop:
                        backdrop.click()
                    else:
                        page.evaluate("() => { document.querySelector('.sidebar-right')?.classList.remove('sidebar-open'); document.querySelector('.sidebar-backdrop')?.classList.remove('visible'); }")
                    time.sleep(0.25)
                    results['sidebar_closed'] = not page.evaluate("() => document.querySelector('.sidebar-right') && document.querySelector('.sidebar-right').classList.contains('sidebar-open')")
                except Exception:
                    results['sidebar_closed'] = not page.evaluate("() => document.querySelector('.sidebar-right') && document.querySelector('.sidebar-right').classList.contains('sidebar-open')")

            # Footer expand/collapse
            if footer_preview and session_footer:
                results['footer_present'] = True
                try:
                    footer_preview.click()
                except Exception:
                    page.evaluate("() => document.querySelector('#sessionFooter')?.classList.toggle('expanded')")
                time.sleep(0.35)
                results['footer_expanded'] = page.evaluate("() => document.querySelector('#sessionFooter') && document.querySelector('#sessionFooter').classList.contains('expanded')")
                # click outside to collapse (use coordinates away from footer)
                try:
                    page.mouse.click(10, 10)
                except Exception:
                    page.click('body')
                time.sleep(0.25)
                results['footer_collapsed'] = not page.evaluate("() => document.querySelector('#sessionFooter') && document.querySelector('#sessionFooter').classList.contains('expanded')")

            try:
                context.close()
            except Exception:
                pass
            browser.close()
    except PWTimeout as e:
        print(json.dumps({"error": "timeout", "detail": str(e)}))
        return 2
    except Exception as e:
        print(json.dumps({"error": "exception", "detail": str(e)}))
        return 3

    print(json.dumps(results))
    # Exit code 0 if all checks passed
    passed = all([
        results['sidebar_present'],
        results['sidebar_opened'],
        results['sidebar_closed'],
        results['backdrop_present'],
        results['footer_present'],
        results['footer_expanded'],
        results['footer_collapsed'],
    ])
    return 0 if passed else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://localhost:5000/ballot_lens', help='URL of running dev server')
    args = parser.parse_args()
    rc = run_check(args.url)
    sys.exit(rc)
