from playwright.sync_api import sync_playwright

URL='http://127.0.0.1:5000/'
with sync_playwright() as p:
    b=p.chromium.launch(headless=True)
    page=b.new_page()
    page.goto(URL, wait_until='networkidle')
    page.wait_for_timeout(1500)
    content=page.content()
    print('len content', len(content))
    print('has btnNavMore?', 'btnNavMore' in content)
    print('has parserToolsDropdown?', 'parserToolsDropdown' in content)
    b.close()
