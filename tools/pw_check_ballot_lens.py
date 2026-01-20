from playwright.sync_api import sync_playwright
import os, time
URL = os.environ.get('WEBAPP_URL','http://127.0.0.1:5000/ballot_lens')
with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    page = b.new_page()
    page.goto(URL, wait_until='domcontentloaded', timeout=30000)
    time.sleep(1.0)
    cont = page.content()
    print('len', len(cont))
    print('has btnNavMore?', 'btnNavMore' in cont)
    print('has parserToolsDropdown?', 'parserToolsDropdown' in cont)
    # pick elements by id
    for id_ in ['btnNavMore','navMoreDropdown','btnToggleRightSidebar','parserToolsDropdown']:
        el = page.query_selector('#' + id_)
        print(id_, 'exists', el is not None)
        if el:
            try:
                bb = el.bounding_box()
            except Exception:
                bb = None
            print('  bounding', bb)
    b.close()
