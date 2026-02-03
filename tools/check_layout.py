import json
import os

from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:5000/ballot_lens"
# Extended set of common device breakpoints
WIDTHS = [320, 360, 375, 412, 480, 568, 640, 768, 834, 1024, 1280, 1440, 1600]
HEIGHT = 900
THRESHOLD = 40  # px tolerance for top offset

OUT_DIR = os.path.join('tools', 'debug_screenshots')
os.makedirs(OUT_DIR, exist_ok=True)

results = []
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    try:
        for w in WIDTHS:
            ctx = browser.new_context(viewport={"width": w, "height": HEIGHT})
            page = ctx.new_page()
            page.goto(URL, timeout=30000)
            # allow initial render
            page.wait_for_timeout(900)

            # attempt to wait for selectors
            sidebar_sel = None
            main_sel = None
            try:
                sidebar_sel = page.wait_for_selector('#sidebar, .sidebar-left', timeout=3000)
            except Exception:
                sidebar_sel = page.query_selector('#sidebar') or page.query_selector('.sidebar-left')
            try:
                main_sel = page.wait_for_selector('main.main-content, .main-content', timeout=3000)
            except Exception:
                main_sel = page.query_selector('main.main-content') or page.query_selector('.main-content')

            sidebar_box = None
            main_box = None
            if sidebar_sel:
                try:
                    sidebar_box = sidebar_sel.bounding_box()
                except Exception:
                    sidebar_box = None
            if main_sel:
                try:
                    main_box = main_sel.bounding_box()
                except Exception:
                    main_box = None

            passed = True
            note = None

            # Save a screenshot for visual inspection
            shot_path = os.path.join(OUT_DIR, f'shot_{w}x{HEIGHT}.png')
            try:
                page.screenshot(path=shot_path, full_page=True)
            except Exception:
                pass

            if not sidebar_box or not main_box:
                try:
                    snippet = page.content()[:1600]
                except Exception:
                    snippet = ''
                note = f'missing selector; snippet_start={snippet!r}'
                passed = False
            else:
                sidebar_bottom = sidebar_box['y'] + sidebar_box['height']
                main_top = main_box['y']
                if main_top > sidebar_bottom + THRESHOLD:
                    passed = False
                    note = f'main_top={main_top:.1f} > sidebar_bottom+{THRESHOLD}={sidebar_bottom+THRESHOLD:.1f}'
                else:
                    note = f'main_top={main_top:.1f}, sidebar_bottom={sidebar_bottom:.1f}'

            results.append({
                'width': w,
                'height': HEIGHT,
                'screenshot': shot_path if os.path.exists(shot_path) else None,
                'sidebar_box': sidebar_box,
                'main_box': main_box,
                'passed': passed,
                'note': note,
            })

            ctx.close()
    finally:
        browser.close()

# write JSON report
report_path = os.path.join(OUT_DIR, 'layout_report.json')
with open(report_path, 'w', encoding='utf-8') as rf:
    json.dump({'url': URL, 'threshold': THRESHOLD, 'results': results}, rf, indent=2)

for r in results:
    status = 'OK' if r['passed'] else 'FAIL'
    print(f"{r['width']}px: {status} — {r['note']}")

any_failed = any(not r['passed'] for r in results)
if any_failed:
    print(f"Report saved to: {report_path}")
    raise SystemExit(2)
else:
    print(f"All checks passed. Report saved to: {report_path}")
