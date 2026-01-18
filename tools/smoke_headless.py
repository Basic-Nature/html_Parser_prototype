from playwright.sync_api import sync_playwright
import sys, json, time

BASE = 'http://127.0.0.1:5000'
URL = BASE + '/run_parser'

def first_selector(page, selectors):
    for s in selectors:
        try:
            el = page.query_selector(s)
            if el:
                return s
        except Exception:
            continue
    return None

def visible(page, selector):
    try:
        el = page.query_selector(selector)
        if not el:
            return False
        return el.is_visible()
    except Exception:
        return False

def main():
    results = {"url": URL, "checks": {}, "errors": []}
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width":360, "height":800})
            page = context.new_page()
            page.goto(URL, wait_until='load', timeout=15000)
            time.sleep(0.5)

            # detect nav-more toggle
            nav_more_sel = first_selector(page, ['#btnNavMore', '.nav-more-toggle', '[data-nav-more]', '.nav-more'])
            results['checks']['nav_more_selector_found'] = bool(nav_more_sel)
            results['checks']['nav_more_selector'] = nav_more_sel

            if nav_more_sel:
                try:
                    pre_vis = visible(page, '.nav-more-dropdown') or visible(page, '#navMoreDropdown')
                    page.click(nav_more_sel)
                    time.sleep(0.25)
                    post_vis = visible(page, '.nav-more-dropdown.open') or visible(page, '.nav-more-dropdown') or visible(page, '#navMoreDropdown')
                    results['checks']['nav_more_dropdown_pre_visible'] = bool(pre_vis)
                    results['checks']['nav_more_dropdown_post_visible'] = bool(post_vis)
                except Exception as e:
                    results['errors'].append(f'nav_more click error: {e}')
            else:
                results['errors'].append('nav_more toggle not found')

            # sidebar toggle
            side_sel = first_selector(page, ['.sidebar-toggle', '#sidebarToggle', '[data-sidebar-toggle]'])
            results['checks']['sidebar_toggle_selector_found'] = bool(side_sel)
            results['checks']['sidebar_toggle_selector'] = side_sel
            try:
                body_over_before = page.evaluate("() => window.getComputedStyle(document.body).overflow")
                results['checks']['body_overflow_before'] = body_over_before
            except Exception:
                results['checks']['body_overflow_before'] = None

            if side_sel:
                try:
                    page.click(side_sel)
                    time.sleep(0.35)
                    # check sidebar open state
                    sidebar_open = visible(page, '#sidebar.sidebar-open') or visible(page, '#sidebar') or visible(page, '.sidebar.sidebar-open')
                    body_over_after = page.evaluate("() => window.getComputedStyle(document.body).overflow")
                    results['checks']['sidebar_open'] = bool(sidebar_open)
                    results['checks']['body_overflow_after'] = body_over_after
                except Exception as e:
                    results['errors'].append(f'sidebar click error: {e}')
            else:
                results['errors'].append('sidebar toggle not found')

            # body scroll-lock heuristic: compare before/after to see if overflow set to hidden
            try:
                results['checks']['body_scroll_locked'] = (results['checks'].get('body_overflow_before') is not None and results['checks'].get('body_overflow_after') in ('hidden', 'hidden auto', 'hidden scroll'))
            except Exception:
                results['checks']['body_scroll_locked'] = False

            browser.close()
    except Exception as e:
        results['errors'].append(str(e))
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
