from playwright.sync_api import sync_playwright
import json

URL = 'http://127.0.0.1:5000/run_parser'

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    ctx = browser.new_context(viewport={"width":360,"height":800})
    page = ctx.new_page()
    console_msgs = []
    page.on('console', lambda msg: console_msgs.append({'type': msg.type, 'text': msg.text}))
    page.on('pageerror', lambda e: console_msgs.append({'type': 'error', 'text': str(e)}))
    page.goto(URL)
    page.wait_for_timeout(800)
    def safe_eval(expr):
        try:
            return page.evaluate(expr)
        except Exception as e:
            return {'error': str(e)}

    info = {}
    info['btn_exists'] = page.query_selector('#btnNavMore') is not None
    info['dropdown_exists'] = page.query_selector('#navMoreDropdown') is not None
    info['btn_rect'] = safe_eval("() => { const el = document.getElementById('btnNavMore'); if (!el) return null; const r = el.getBoundingClientRect(); return {left: r.left, top: r.top, right: r.right, bottom: r.bottom, width: r.width, height: r.height}; }")
    info['btn_style'] = safe_eval("() => { const el=document.getElementById('btnNavMore'); if(!el) return null; const cs = getComputedStyle(el); return {display: cs.display, visibility: cs.visibility, opacity: cs.opacity, pointerEvents: cs.pointerEvents, zIndex: cs.zIndex}; }")
    info['element_at_center'] = safe_eval("() => { const el=document.getElementById('btnNavMore'); if(!el) return null; const r=el.getBoundingClientRect(); const cx=Math.round((r.left+r.right)/2); const cy=Math.round((r.top+r.bottom)/2); const hit=document.elementFromPoint(cx, cy); return {cx:cx, cy:cy, hit: (hit ? (hit.id||hit.className||hit.tagName) : null)} }")
    info['dropdown_style'] = safe_eval("() => { const el=document.getElementById('navMoreDropdown'); if(!el) return null; const cs=getComputedStyle(el); return {display: cs.display, visibility: cs.visibility, opacity: cs.opacity, zIndex: cs.zIndex, position: cs.position, left: cs.left, top: cs.top}; }")
    # Try to dispatch a click on the toggle and re-check dropdown style
    try:
        # Install a temporary click listener to detect if clicks are delivered
        page.evaluate("() => { window.__lastNavMoreClicked = false; const b=document.getElementById('btnNavMore'); if (b) { b.addEventListener('click', () => { window.__lastNavMoreClicked = true; }, {once:true}); return true; } return false; }")
        page.evaluate("() => { const b=document.getElementById('btnNavMore'); if (b) { b.dispatchEvent(new MouseEvent('click', {bubbles:true,cancelable:true, view:window})); return true; } return false; }")
        page.wait_for_timeout(500)
        info['dropdown_style_after_click'] = safe_eval("() => { const el=document.getElementById('navMoreDropdown'); if(!el) return null; const cs=getComputedStyle(el); return {display: cs.display, visibility: cs.visibility, opacity: cs.opacity, zIndex: cs.zIndex, position: cs.position, left: cs.left, top: cs.top}; }")
    except Exception as e:
        info['click_error'] = str(e)
    try:
        info['navmore_clicked_flag'] = page.evaluate("() => !!window.__lastNavMoreClicked")
    except Exception:
        info['navmore_clicked_flag'] = None
    # As a last-resort diagnostic, force the dropdown visible via inline styles and re-check
    try:
        page.evaluate("() => { const d = document.getElementById('navMoreDropdown'); if (d) { d.style.display='block'; d.style.opacity='1'; d.setAttribute('aria-hidden','false'); return true; } return false; }")
        page.wait_for_timeout(200)
        info['dropdown_style_forced'] = safe_eval("() => { const el=document.getElementById('navMoreDropdown'); if(!el) return null; const cs=getComputedStyle(el); return {display: cs.display, visibility: cs.visibility, opacity: cs.opacity, zIndex: cs.zIndex, position: cs.position, left: cs.left, top: cs.top}; }")
    except Exception as e:
        info['dropdown_force_error'] = str(e)
    info['console'] = console_msgs
    # Sidebar diagnostics
    info['sidebar_toggle_present'] = page.query_selector('.sidebar-toggle') is not None
    info['sidebar_rect'] = safe_eval("() => { const el=document.querySelector('.sidebar-toggle'); if(!el) return null; const r=el.getBoundingClientRect(); return {left:r.left, top:r.top, width:r.width, height:r.height}; }")
    info['sidebar_initial_class'] = safe_eval("() => { const s=document.getElementById('sidebar'); return s ? Array.from(s.classList) : null; }")
    # Try clicking the floating toggle if present, else call openLeft()
    try:
        if page.query_selector('#sidebarToggleFloating'):
            page.click('#sidebarToggleFloating', timeout=2000, force=True)
        else:
            invoked = page.evaluate("() => { if (typeof openLeft === 'function') { openLeft(); return 'openLeft'; } const b=document.querySelector('.sidebar-toggle'); if (b) { b.click(); return 'clicked'; } return null; }")
        page.wait_for_timeout(700)
        info['sidebar_open_after'] = safe_eval("() => { const s=document.getElementById('sidebar'); if(!s) return false; return s.classList.contains('sidebar-open') || getComputedStyle(s).display !== 'none'; }")
        info['overlay_visible_after'] = safe_eval("() => { const o=document.getElementById('mobileSidebarOverlay'); const b=document.querySelector('.sidebar-backdrop'); if (o) return o.classList.contains('visible') || getComputedStyle(o).display !== 'none'; if (b) return b.classList.contains('visible') || getComputedStyle(b).display !== 'none'; return false; }")
        info['body_no_scroll_after'] = safe_eval("() => document.body.classList.contains('no-scroll')")
    except Exception as e:
        info['sidebar_interaction_error'] = str(e)

    print(json.dumps(info, indent=2))
    ctx.close()
    browser.close()
