from playwright.sync_api import sync_playwright

URL='http://127.0.0.1:5000/ballot_lens'
with sync_playwright() as p:
    b=p.chromium.launch(headless=True)
    c=b.new_context(viewport={"width":360,"height":800})
    page=c.new_page()
    page.goto(URL)
    page.wait_for_timeout(800)
    def getcls(sel):
        try:
            return page.evaluate(f"() => {{ const e = document.querySelector('{sel}'); return e ? Array.from(e.classList) : null; }}")
        except Exception as e:
            return str(e)
    def get_stats(sel):
        try:
            return page.evaluate(f"() => {{ const e = document.querySelector('{sel}'); if (!e) return null; const s = getComputedStyle(e); const r = e.getBoundingClientRect(); return {{classes: Array.from(e.classList), style: {{display: s.display, visibility: s.visibility, opacity: s.opacity, zIndex: s.zIndex, position: s.position}}, rect: {{x: r.x, y: r.y, width: r.width, height: r.height, top: r.top, left: r.left}} }} }}")
        except Exception as e:
            return str(e)

    print('mobileOverlay stats before:', get_stats('#mobileSidebarOverlay'))
    print('backdrop stats before:', get_stats('.sidebar-backdrop'))
    # Try using exposed openLeft() helper or click the toggle
    try:
        invoked = page.evaluate("() => { try { if (typeof openLeft === 'function') { openLeft(); return 'openLeft'; } const f = document.getElementById('sidebarToggleFloating'); if (f) { f.click(); return 'clicked_floating'; } const s = document.querySelector('.sidebar-toggle'); if (s) { s.click(); return 'clicked_toggle'; } return null; } catch(e) { return 'err:'+e.message } }")
        print('open invoked:', invoked)
    except Exception as e:
        print('open invoke error', e)
    page.wait_for_timeout(700)
    # force add classes
    page.evaluate("() => { const o = document.getElementById('mobileSidebarOverlay'); if (o) { o.classList.add('visible'); o.style.display='block'; o.style.opacity='1'; } const sb = document.querySelector('.sidebar-backdrop'); if (sb) { sb.classList.add('visible'); sb.style.display='block'; sb.style.opacity='1'; } }")
    page.wait_for_timeout(200)
    print('mobileOverlay stats after:', get_stats('#mobileSidebarOverlay'))
    print('backdrop stats after:', get_stats('.sidebar-backdrop'))
    # Print body scroll-lock diagnostics
    try:
        body_classes = page.evaluate("() => Array.from(document.body.classList)")
    except Exception as e:
        body_classes = str(e)
    try:
        body_inline = page.evaluate("() => document.body.style.overflow || ''")
    except Exception as e:
        body_inline = str(e)
    try:
        html_inline = page.evaluate("() => document.documentElement.style.overflow || ''")
    except Exception as e:
        html_inline = str(e)
    print('body classes:', body_classes)
    print('body inline overflow:', body_inline)
    print('html inline overflow:', html_inline)
    c.close()
    b.close()
