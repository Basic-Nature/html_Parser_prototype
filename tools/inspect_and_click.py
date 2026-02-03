import os

from playwright.sync_api import sync_playwright

URL = os.environ.get('WEBAPP_URL','http://127.0.0.1:5000/')

TOGGLES = [
    ('More','btnNavMore','#navMoreDropdown'),
    ('RightTools','btnToggleRightSidebar','#parserToolsDropdown'),
]

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    # Use DOMContentLoaded rather than networkidle to avoid hangs with long-lived sockets
    page.goto(URL, wait_until='domcontentloaded', timeout=60000)
    for name, btn_id, dropdown_sel in TOGGLES:
        print('---', name)
        exists = page.query_selector(f'#{btn_id}') is not None
        print('button exists:', exists)
        if exists:
            info = page.evaluate(f"() => {{ const el = document.getElementById('{btn_id}'); if (!el) return null; const r = el.getBoundingClientRect(); const cs = getComputedStyle(el); return {{rect: r, display: cs.display, visibility: cs.visibility, opacity: cs.opacity, aria: el.getAttribute('aria-hidden'), expanded: el.getAttribute('aria-expanded')}}; }}")
            print('button info:', info)
        d_exists = page.query_selector(dropdown_sel) is not None
        print('dropdown exists:', d_exists)
        if d_exists:
            info2 = page.evaluate(f"() => {{ const el = document.querySelector('{dropdown_sel}'); if (!el) return null; const cs = getComputedStyle(el); return {{display: cs.display, visibility: cs.visibility, opacity: cs.opacity, aria: el.getAttribute('aria-hidden'), classList: el.className}}; }}")
            print('dropdown info:', info2)
        # try dispatch click event via page.evaluate
        if exists:
            try:
                page.evaluate(f"() => {{ const b = document.getElementById('{btn_id}'); if (!b) return false; b.dispatchEvent(new MouseEvent('click', {{bubbles:true,cancelable:true, view:window}})); return true; }}")
                print('dispatched click via DOM event')
                page.wait_for_timeout(400)
                if d_exists:
                    after = page.evaluate(f"() => {{ const el = document.querySelector('{dropdown_sel}'); if (!el) return null; const cs = getComputedStyle(el); return {{display: cs.display, visibility: cs.visibility, opacity: cs.opacity, aria: el.getAttribute('aria-hidden'), classList: el.className}}; }}")
                    print('dropdown after click:', after)
            except Exception as e:
                print('dispatch click failed', e)
    browser.close()
print('done')
