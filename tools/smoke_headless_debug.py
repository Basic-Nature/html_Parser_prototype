from playwright.sync_api import sync_playwright
import time
import json
import os

OUT_DIR = os.path.join("tools", "debug_headless_output")
os.makedirs(OUT_DIR, exist_ok=True)

URL = os.environ.get("PARSER_URL", "http://127.0.0.1:5000/ballot_lens")

def try_selector(page, sel):
    try:
        el = page.query_selector(sel)
        return el is not None
    except Exception:
        return False

def main():
    results = {"url": URL}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 360, "height": 800})
        page = context.new_page()
        # Capture console messages and page errors for diagnostics
        console_msgs = []
        def on_console(msg):
            try:
                loc = None
                try:
                    loc = msg.location
                except Exception:
                    loc = None
                console_msgs.append({'type': msg.type, 'text': msg.text, 'location': loc})
            except Exception:
                pass
        page.on('console', on_console)
        def on_page_error(exc):
            try:
                stack = None
                try:
                    stack = exc.stack
                except Exception:
                    stack = None
                console_msgs.append({'type': 'error', 'text': str(exc), 'stack': stack})
            except Exception:
                pass
        page.on('pageerror', on_page_error)
        try:
            page.goto(URL, timeout=60000)
        except Exception as e:
            results["error"] = f"goto_failed: {e}"
            print(json.dumps(results, indent=2))
            return

        # selectors we care about
        selectors = {
            "nav_more": "#btnNavMore",
            "nav_more_dropdown": "#navMoreDropdown",
            "sidebar_toggle": ".sidebar-toggle",
            "sidebar": "#sidebar",
            "backdrop": ".sidebar-backdrop",
            # Right-sidebar controls
            "btn_right_toggle": "#btnToggleRightSidebar",
            "sidebar_right": ".sidebar-right",
            "mobile_overlay": "#mobileSidebarOverlay",
        }

        for k, s in selectors.items():
            results[f"{k}_found"] = try_selector(page, s)

        # record body overflow before
        try:
            results["body_overflow_before"] = page.evaluate("() => window.getComputedStyle(document.body).overflow")
        except Exception as e:
            results["body_overflow_before_error"] = str(e)

        # Try force-click the nav-more button
        try:
            if results.get("nav_more_found"):
                page.click(selectors["nav_more"], timeout=10000, force=True)
                page.wait_for_timeout(800)
                # longer wait for potential animations
                page.wait_for_timeout(400)
                try:
                    results["nav_more_dropdown_post_visible"] = page.is_visible(selectors["nav_more_dropdown"])
                except Exception:
                    results["nav_more_dropdown_post_visible"] = False
            else:
                results["nav_more_skipped"] = True
        except Exception as e:
            results["nav_more_error"] = str(e)

        # Try force-click the sidebar toggle (open)
        try:
            if results.get("sidebar_toggle_found"):
                # Collect computed style and bounding rect for diagnostics
                try:
                    diag = page.evaluate("() => { const el = document.querySelector('.sidebar-toggle'); if (!el) return null; const cs = window.getComputedStyle(el); const r = el.getBoundingClientRect(); return {display: cs.display, visibility: cs.visibility, opacity: cs.opacity, width: r.width, height: r.height, top: r.top, left: r.left, inViewport: (r.top>=0 && r.left>=0 && r.bottom<=window.innerHeight && r.right<=window.innerWidth)}; }")
                    results['sidebar_toggle_computed'] = diag
                except Exception as e:
                    results['sidebar_toggle_computed_error'] = str(e)
                # Ensure the toggle is visible: try to force inline styles as a fallback for headless runs
                try:
                    page.evaluate("() => { const el = document.querySelector('.sidebar-toggle'); if (el) { el.style.position='fixed'; el.style.left='12px'; el.style.top='12px'; el.style.zIndex='12600'; el.style.display='inline-flex'; el.style.visibility='visible'; } }")
                except Exception:
                    pass
                # Prefer floating toggle if present
                if try_selector(page, '#sidebarToggleFloating'):
                    try:
                        page.click('#sidebarToggleFloating', timeout=8000, force=True)
                    except Exception:
                        pass
                else:
                    # fallback to programmatic DOM click (may trigger handlers even if element has zero-layout)
                    try:
                        page.evaluate("() => { const el = document.querySelector('.sidebar-toggle'); if (el) el.click(); }")
                    except Exception:
                        # final fallback: page.click
                        page.click(selectors["sidebar_toggle"], timeout=10000, force=True)
                # Wait for any open animation
                page.wait_for_timeout(1200)
                # Check body overflow and sidebar visibility
                try:
                    results["body_overflow_after"] = page.evaluate("() => window.getComputedStyle(document.body).overflow")
                except Exception as e:
                    results["body_overflow_after_error"] = str(e)
                try:
                    # either visible or has open class
                    vis = page.is_visible(selectors["sidebar"]) if try_selector(page, selectors["sidebar"]) else False
                    if not vis:
                        # check class
                        vis = page.eval_on_selector(selectors["sidebar"], "el => el.classList.contains('sidebar-open')") if try_selector(page, selectors["sidebar"]) else False
                    results["sidebar_open"] = bool(vis)
                except Exception as e:
                    results["sidebar_open_error"] = str(e)
            else:
                results["sidebar_toggle_skipped"] = True
        except Exception as e:
            results["sidebar_toggle_error"] = str(e)

        # As an aggressive fallback, try invoking toggleSidebar()/openLeft() directly if available in page context
        try:
            invoked = page.evaluate("() => { try { if (typeof toggleSidebar === 'function') { toggleSidebar(); return 'toggleSidebar'; } if (typeof openLeft === 'function') { openLeft(); return 'openLeft'; } return null; } catch(e) { return 'err:'+String(e); } }")
            results['toggle_invoke'] = invoked
            page.wait_for_timeout(500)
            try:
                results['sidebar_open_after_invoke'] = page.eval_on_selector(selectors['sidebar'], "el => el.classList.contains('sidebar-open')") if try_selector(page, selectors['sidebar']) else False
            except Exception:
                results['sidebar_open_after_invoke'] = False
        except Exception as e:
            results['toggle_invoke_error'] = str(e)

        # Try toggling right-sidebar via the right toggle button (floating)
        try:
            if results.get("btn_right_toggle_found"):
                page.click(selectors["btn_right_toggle"], timeout=10000, force=True)
                page.wait_for_timeout(700)
                # Check for sidebar-right visibility or open class
                try:
                    results["sidebar_right_visible"] = page.is_visible(selectors["sidebar_right"]) if try_selector(page, selectors["sidebar_right"]) else False
                except Exception:
                    # fallback: check class
                    try:
                        results["sidebar_right_open_class"] = page.eval_on_selector(selectors["sidebar_right"], "el => el.classList.contains('open') || el.classList.contains('sidebar-open')") if try_selector(page, selectors["sidebar_right"]) else False
                    except Exception as ee:
                        results["sidebar_right_check_error"] = str(ee)
                # overlay visibility
                try:
                    results["mobile_overlay_visible"] = page.is_visible(selectors["mobile_overlay"]) if try_selector(page, selectors["mobile_overlay"]) else False
                except Exception:
                    results["mobile_overlay_visible"] = False
            else:
                results["btn_right_toggle_skipped"] = True
        except Exception as e:
            results["btn_right_toggle_error"] = str(e)

        # Extra diagnostics: check body class and overlay state after interactions
        try:
            results['body_has_no_scroll_class'] = page.evaluate("() => document.body.classList.contains('no-scroll')")
        except Exception:
            results['body_has_no_scroll_class'] = False
        try:
            results['body_inline_overflow'] = page.evaluate("() => document.body.style.overflow || ''")
        except Exception:
            results['body_inline_overflow'] = ''
        try:
            results['overlay_visible_after_invoke'] = page.evaluate("() => { const o = document.getElementById('mobileSidebarOverlay') || document.querySelector('.sidebar-backdrop'); return o ? o.classList.contains('visible') : false }")
        except Exception:
            results['overlay_visible_after_invoke'] = False

        # Diagnostic: force-add right-sidebar open classes and body scroll-lock (helps determine if CSS hides it)
        try:
            page.evaluate("() => { const r = document.querySelector('.sidebar-right'); const o = document.getElementById('mobileSidebarOverlay'); if (r) { r.classList.add('open','sidebar-open'); } if (o) { o.classList.add('visible'); o.setAttribute('aria-hidden','false'); } document.body.classList.add('sidebar-right-open','no-scroll'); }")
            page.wait_for_timeout(300)
            try:
                results['sidebar_right_forced_visible'] = page.is_visible(selectors['sidebar_right']) if try_selector(page, selectors['sidebar_right']) else False
            except Exception:
                results['sidebar_right_forced_visible'] = False
        except Exception as e:
            results['sidebar_right_force_error'] = str(e)

        # If interactions didn't reveal visibility, try JS dispatch as fallback
        try:
            page.evaluate("() => { const b = document.querySelector('#btnNavMore'); if (b) b.dispatchEvent(new MouseEvent('click',{bubbles:true,cancelable:true})); }")
            page.wait_for_timeout(500)
            try:
                results["nav_more_dropdown_post_visible_js"] = page.is_visible(selectors["nav_more_dropdown"]) if try_selector(page, selectors["nav_more_dropdown"]) else False
            except Exception:
                results["nav_more_dropdown_post_visible_js"] = False
        except Exception as e:
            results["nav_more_js_error"] = str(e)

        # Save screenshot and HTML snapshot for inspection
        shot = os.path.join(OUT_DIR, "debug_screenshot.png")
        htmlp = os.path.join(OUT_DIR, "debug_page.html")
        try:
            page.screenshot(path=shot, full_page=True)
            results["screenshot"] = shot
        except Exception as e:
            results["screenshot_error"] = str(e)
        try:
            with open(htmlp, "w", encoding="utf-8") as f:
                f.write(page.content())
            results["html_snapshot"] = htmlp
        except Exception as e:
            results["html_snapshot_error"] = str(e)

        # Attach collected console messages
        if console_msgs:
            results['console'] = console_msgs

        # Close
        context.close()
        browser.close()

    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
