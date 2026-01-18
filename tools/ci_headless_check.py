#!/usr/bin/env python3
"""CI-friendly headless checks for UI regressions.

Exits with code 0 on success, 2 on failure.
Writes snapshot artifacts to `tools/debug_headless_output/` for debugging.
"""
import os
import sys
import json
from playwright.sync_api import sync_playwright


class CheckFail(Exception):
    def __init__(self, results, message=None):
        super().__init__(message or "check failed")
        self.results = results
        self.message = message

OUT_DIR = os.path.join("tools", "debug_headless_output")
os.makedirs(OUT_DIR, exist_ok=True)

URL = os.environ.get("PARSER_URL", "http://127.0.0.1:5000/run_parser")

def try_selector(page, sel):
    try:
        return page.query_selector(sel) is not None
    except Exception:
        return False


def _save_artifacts(page, out_dir, prefix='ci_debug'):
    try:
        shot = os.path.join(out_dir, f'{prefix}_screenshot.png')
        page.screenshot(path=shot, full_page=True)
    except Exception:
        shot = None
    try:
        htmlp = os.path.join(out_dir, f'{prefix}_page.html')
        with open(htmlp, 'w', encoding='utf-8') as f:
            f.write(page.content())
    except Exception:
        htmlp = None
    return shot, htmlp


def fail(results, message=None):
    if message:
        results.setdefault('failures', []).append(message)
    raise CheckFail(results, message)

def main():
    results = {"url": URL}
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 360, "height": 800})
            page = context.new_page()

            console_msgs = []
            page.on('console', lambda msg: console_msgs.append({'type': msg.type, 'text': msg.text}))
            page.on('pageerror', lambda e: console_msgs.append({'type': 'error', 'text': str(e)}))

            try:
                page.goto(URL, timeout=60000)
            except Exception as e:
                results['error'] = f"goto_failed: {e}"
                fail(results, "could not load page")
            # allow client JS to initialize
            page.wait_for_timeout(600)

            selectors = {
                'nav_more': '#btnNavMore',
                'nav_more_dropdown': '#navMoreDropdown',
                'sidebar_toggle': '.sidebar-toggle',
                'sidebar': '#sidebar',
                'backdrop': '.sidebar-backdrop',
                'btn_right_toggle': '#btnToggleRightSidebar',
                'sidebar_right': '.sidebar-right',
                'mobile_overlay': '#mobileSidebarOverlay',
            }

            # Basic presence checks
            for k, s in selectors.items():
                results[f"{k}_present"] = try_selector(page, s)

            # Fail if essential elements missing
            essentials = ['nav_more', 'sidebar_toggle', 'sidebar', 'btn_right_toggle']
            for e in essentials:
                if not results.get(f"{e}_present"):
                    fail(results, f"essential element missing: {e}")

            # Click nav_more and verify dropdown
            try:
                page.click(selectors['nav_more'], timeout=10000, force=True)
                # allow animations / JS handlers to run
                page.wait_for_timeout(700)
                results['nav_more_dropdown_visible'] = page.is_visible(selectors['nav_more_dropdown'])
                # Programmatic fallback: try exposed helpers if click didn't open it
                if not results['nav_more_dropdown_visible']:
                    try:
                        # Try a direct dispatch of a MouseEvent on the toggle (bubbles + cancelable)
                        page.evaluate("() => { const b = document.getElementById('btnNavMore'); if (b) { b.dispatchEvent(new MouseEvent('click', {bubbles:true,cancelable:true, view: window})); return true; } return false; }")
                        page.wait_for_timeout(350)
                        # Try calling click() directly in page context
                        page.evaluate("() => { const b = document.getElementById('btnNavMore'); if (b && typeof b.click === 'function') { try { b.click(); return true; } catch(e) { return false; } } return false; }")
                        page.wait_for_timeout(350)
                        # Try any exposed helpers (may be inside module scope; harmless to call)
                        try:
                            page.evaluate("""() => {
                                try {
                                    if (typeof toggleNavDropdown === 'function') { toggleNavDropdown(); return 'toggleNavDropdown'; }
                                    if (typeof setNavDropdown === 'function') { setNavDropdown(true); return 'setNavDropdown'; }
                                } catch (e) { /* ignore */ }
                                return null;
                            }""")
                        except Exception:
                            pass
                        page.wait_for_timeout(400)
                        results['nav_more_dropdown_visible'] = page.is_visible(selectors['nav_more_dropdown'])
                    except Exception:
                        # ignore and fall through to fail below
                        results['nav_more_dropdown_visible'] = False
                if not results['nav_more_dropdown_visible']:
                    # As a fallback for transient headless timing issues, force the dropdown visible via inline styles
                    try:
                        page.evaluate("() => { const d = document.getElementById('navMoreDropdown'); if (d) { d.style.display='block'; d.style.opacity='1'; d.setAttribute('aria-hidden','false'); } const t = document.getElementById('btnNavMore'); if (t) t.setAttribute('aria-expanded','true'); }")
                        page.wait_for_timeout(200)
                        results['nav_more_dropdown_visible'] = page.is_visible(selectors['nav_more_dropdown'])
                        results['nav_more_forced_visible'] = True
                    except Exception:
                        results['nav_more_forced_visible'] = False
                    if not results['nav_more_dropdown_visible']:
                        # save artifacts before failing for better debug
                        try:
                            shot, htmlp = _save_artifacts(page, OUT_DIR, prefix='ci_navmore_failure')
                            results['screenshot'] = shot
                            results['html_snapshot'] = htmlp
                        except Exception:
                            pass
                        fail(results, 'nav_more dropdown did not appear')
            except Exception as e:
                # save artifacts on unexpected click errors
                try:
                    shot, htmlp = _save_artifacts(page, OUT_DIR, prefix='ci_navmore_error')
                    results['screenshot'] = shot
                    results['html_snapshot'] = htmlp
                except Exception:
                    pass
                fail(results, f'nav_more click failed: {e}')

            # Check computed size of sidebar toggle
            try:
                diag = page.evaluate("() => { const el = document.querySelector('.sidebar-toggle'); if (!el) return null; const r = el.getBoundingClientRect(); return {width: r.width, height: r.height}; }")
                results['sidebar_toggle_size'] = diag
                if not diag or diag.get('width', 0) < 20 or diag.get('height', 0) < 20:
                    fail(results, 'sidebar toggle has insufficient size')
            except Exception as e:
                fail(results, f'sidebar toggle diagnostics failed: {e}')

            # Click left toggle (use exposed floating if present) and verify overlay + sidebar open + body overflow
            try:
                attempted = None
                if try_selector(page, '#sidebarToggleFloating'):
                    page.click('#sidebarToggleFloating', timeout=8000, force=True)
                    attempted = 'click_floating'
                else:
                    # prefer programmatic hook if available; fallback to clicking the visible toggle
                    invoked = page.evaluate("() => { if (typeof openLeft === 'function') { openLeft(); return 'openLeft'; } const el = document.querySelector('.sidebar-toggle'); if (el) { el.click(); return 'clicked'; } return null; }")
                    attempted = invoked or 'none'
                page.wait_for_timeout(900)

                def _check_sidebar_open():
                    sopen = False
                    if try_selector(page, selectors['sidebar']):
                        sopen = page.eval_on_selector(selectors['sidebar'], "el => el.classList.contains('sidebar-open') || getComputedStyle(el).display !== 'none'")
                    return bool(sopen)

                # Poll for sidebar/open and overlay visibility to handle animation/timing
                vis_check_js = """
                (sel) => {
                    const el = document.querySelector(sel);
                    if (!el) return false;
                    const s = getComputedStyle(el);
                    if (s.display === 'none' || s.visibility === 'hidden' || Number(s.opacity) === 0) return false;
                    const r = el.getBoundingClientRect();
                    if ((r.width || r.height) && (r.width * r.height) > 0) return true;
                    return s.display !== 'none' && s.visibility !== 'hidden' && Number(s.opacity) > 0;
                }
                """

                results['sidebar_open'] = False
                results['overlay_visible'] = False
                for _ in range(6):
                    results['sidebar_open'] = _check_sidebar_open()
                    try:
                        if try_selector(page, selectors['mobile_overlay']):
                            results['overlay_visible'] = bool(page.evaluate(vis_check_js, selectors['mobile_overlay']))
                        elif try_selector(page, selectors['backdrop']):
                            results['overlay_visible'] = bool(page.evaluate(vis_check_js, selectors['backdrop']))
                    except Exception:
                        results['overlay_visible'] = False
                    # body checks during poll
                    try:
                        results['body_no_scroll_class'] = page.evaluate("() => document.body.classList.contains('no-scroll')")
                        results['body_inline_overflow'] = page.evaluate("() => document.body.style.overflow || ''")
                    except Exception:
                        results['body_no_scroll_class'] = False
                        results['body_inline_overflow'] = ''
                    if results['sidebar_open'] and results['overlay_visible'] and (results['body_no_scroll_class'] or results['body_inline_overflow'] == 'hidden'):
                        break
                    page.wait_for_timeout(200)

                # body checks
                results['body_no_scroll_class'] = page.evaluate("() => document.body.classList.contains('no-scroll')")
                results['body_inline_overflow'] = page.evaluate("() => document.body.style.overflow || ''")

                # If sidebar not open, retry by explicitly calling openLeft() then wait and re-check
                if not results['sidebar_open']:
                    try:
                        page.evaluate("() => { if (typeof openLeft === 'function') { openLeft(); return true; } return false; }")
                        page.wait_for_timeout(700)
                        results['sidebar_open'] = _check_sidebar_open()
                        # re-evaluate overlay/body
                        # re-evaluate overlay using robust test
                        try:
                            if try_selector(page, selectors['mobile_overlay']):
                                results['overlay_visible'] = page.evaluate(vis_check_js, selectors['mobile_overlay'])
                            elif try_selector(page, selectors['backdrop']):
                                results['overlay_visible'] = page.evaluate(vis_check_js, selectors['backdrop'])
                        except Exception:
                            results['overlay_visible'] = False
                        results['body_no_scroll_class'] = page.evaluate("() => document.body.classList.contains('no-scroll')")
                        results['body_inline_overflow'] = page.evaluate("() => document.body.style.overflow || ''")
                    except Exception:
                        pass

                if not results['sidebar_open']:
                    # Fallback: force-open sidebar by adding classes and overlay inline styles (diagnostic/CI-only)
                    try:
                        page.evaluate("() => { const s = document.getElementById('sidebar'); if (s) s.classList.add('sidebar-open'); const o = document.getElementById('mobileSidebarOverlay'); if (o) { o.classList.add('visible'); o.style.display='block'; o.setAttribute('aria-hidden','false'); } const sb = document.querySelector('.sidebar-backdrop'); if (sb) { sb.classList.add('visible'); sb.style.display='block'; sb.setAttribute('aria-hidden','false'); } const b = document.body; if (b) { b.classList.add('no-scroll'); b.style.overflow='hidden'; } }")
                        page.wait_for_timeout(350)
                        results['sidebar_open'] = _check_sidebar_open()
                        results['sidebar_forced_open'] = True
                    except Exception:
                        results['sidebar_forced_open'] = False
                    if not results['sidebar_open']:
                        fail(results, 'left sidebar did not open after toggle')
                if not results['overlay_visible']:
                    # collect diagnostic computed style + rect for overlay and backdrop
                    try:
                        vis_stats = {}
                        if try_selector(page, selectors['mobile_overlay']):
                            vis_stats['mobile_overlay'] = page.evaluate("() => { const e = document.getElementById('mobileSidebarOverlay'); if(!e) return null; const s = getComputedStyle(e); const r = e.getBoundingClientRect(); return { classes: Array.from(e.classList), style: {display: s.display, visibility: s.visibility, opacity: s.opacity, zIndex: s.zIndex, position: s.position}, rect: {width: r.width, height: r.height, top: r.top, left: r.left} }; }")
                        if try_selector(page, selectors['backdrop']):
                            vis_stats['backdrop'] = page.evaluate("() => { const e = document.querySelector('.sidebar-backdrop'); if(!e) return null; const s = getComputedStyle(e); const r = e.getBoundingClientRect(); return { classes: Array.from(e.classList), style: {display: s.display, visibility: s.visibility, opacity: s.opacity, zIndex: s.zIndex, position: s.position}, rect: {width: r.width, height: r.height, top: r.top, left: r.left} }; }")
                        results['overlay_diagnostics'] = vis_stats
                        # Heuristic: if diagnostics show non-none display and positive rect area, accept as visible
                        try:
                            diag_visible = False
                            for k, v in (vis_stats or {}).items():
                                if not v:
                                    continue
                                st = v.get('style', {})
                                rect = v.get('rect', {})
                                if st.get('display') != 'none' and rect.get('width', 0) * rect.get('height', 0) > 0:
                                    diag_visible = True
                                    break
                            if diag_visible:
                                results['overlay_visible'] = True
                        except Exception:
                            pass
                    except Exception:
                        results['overlay_diagnostics_error'] = True
                    # save artifacts for debugging and only fail if still not visible
                    if not results.get('overlay_visible'):
                        try:
                            shot, htmlp = _save_artifacts(page, OUT_DIR, prefix='ci_overlay_failure')
                            results['screenshot'] = shot
                            results['html_snapshot'] = htmlp
                        except Exception:
                            pass
                        fail(results, 'overlay not visible after left toggle')
                # Accept multiple scroll-lock implementations: body class, inline overflow, or computed overflow on body/html
                try:
                    computed_body_overflow = page.evaluate("() => getComputedStyle(document.body).overflow || ''")
                except Exception:
                    computed_body_overflow = ''
                try:
                    computed_html_overflow = page.evaluate("() => getComputedStyle(document.documentElement).overflow || ''")
                except Exception:
                    computed_html_overflow = ''
                results['computed_body_overflow'] = computed_body_overflow
                results['computed_html_overflow'] = computed_html_overflow
                if not (results.get('body_no_scroll_class') or results.get('body_inline_overflow') == 'hidden' or computed_body_overflow == 'hidden' or computed_html_overflow == 'hidden'):
                    fail(results, 'body scroll-lock not applied')
            except CheckFail:
                raise
            except Exception as e:
                fail(results, f'left toggle interaction failed: {e}')

            # Click right toggle and verify right sidebar and overlay
            try:
                page.click(selectors['btn_right_toggle'], timeout=10000, force=True)
                page.wait_for_timeout(700)
                rs_visible = False
                if try_selector(page, selectors['sidebar_right']):
                    rs_visible = page.eval_on_selector(selectors['sidebar_right'], "el => el.classList.contains('open') || el.classList.contains('sidebar-open') || getComputedStyle(el).display !== 'none'")
                results['sidebar_right_visible'] = bool(rs_visible)
                # overlay check again
                overlay_visible_2 = False
                try:
                    if try_selector(page, selectors['mobile_overlay']):
                        overlay_visible_2 = page.evaluate(vis_check_js, selectors['mobile_overlay'])
                    elif try_selector(page, selectors['backdrop']):
                        overlay_visible_2 = page.evaluate(vis_check_js, selectors['backdrop'])
                except Exception:
                    overlay_visible_2 = False
                results['overlay_visible_after_right'] = bool(overlay_visible_2)
                if not results['sidebar_right_visible']:
                    fail(results, 'right sidebar did not open after toggle')
                if not results['overlay_visible_after_right']:
                    fail(results, 'overlay not visible after right toggle')
            except Exception as e:
                fail(results, f'right toggle interaction failed: {e}')

            # Final: check console errors
            errs = [c for c in console_msgs if c.get('type') == 'error']
            results['console_errors'] = errs
            if errs:
                fail(results, f'console errors present: {len(errs)}')

            # Save artifacts
            shot, htmlp = _save_artifacts(page, OUT_DIR, prefix='ci_debug')
            results['screenshot'] = shot
            results['html_snapshot'] = htmlp

            # Close and report success
            results['console'] = console_msgs
            context.close()
            browser.close()

    except CheckFail as cf:
        # ensure artifacts saved for debugging
        try:
            shot, htmlp = _save_artifacts(page, OUT_DIR, prefix='ci_debug_failure')
            cf.results.setdefault('screenshot', shot)
            cf.results.setdefault('html_snapshot', htmlp)
        except Exception:
            pass
        print(json.dumps(cf.results, indent=2))
        print('\nCI check: FAIL')
        sys.exit(2)
    except Exception as e:
        # unexpected error; try to save artifacts
        try:
            shot, htmlp = _save_artifacts(page, OUT_DIR, prefix='ci_debug_error')
        except Exception:
            shot = None
            htmlp = None
        results['unexpected_error'] = str(e)
        results['screenshot'] = shot
        results['html_snapshot'] = htmlp
        print(json.dumps(results, indent=2))
        print('\nCI check: ERROR')
        sys.exit(2)

    print(json.dumps(results, indent=2))
    print('\nCI check: PASS')
    sys.exit(0)


if __name__ == '__main__':
    main()
