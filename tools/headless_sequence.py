import json
import os
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = os.environ.get('WEBAPP_URL', 'http://127.0.0.1:5000/')
OUT_DIR = Path('tools/screenshots')
OUT_DIR.mkdir(parents=True, exist_ok=True)

MORE_SELECTORS = [
    '#btnNavMore',
    '#navMoreToggle',
    '#navMoreButton',
    '[data-action="nav-more"]',
    '.nav-more-toggle',
    '.more-toggle',
    '[aria-controls="navMoreDropdown"]',
    '[data-target="navMoreDropdown"]',
]
MORE_OVERLAY_CANDIDATES = ['#navMoreDropdown', '.nav-dropdown', '.more-menu', '.overlay', '[data-overlay="more"]']

RIGHT_SELECTORS = [
    '#btnToggleRightSidebar',
    '#rightToolToggle',
    '.tool-toggle',
    '.tools-toggle',
    '[data-action="open-tools"]',
    '[aria-controls="parserToolsDropdown"], [aria-controls="toolPanel"]',
    '[data-target="toolPanel"]',
]
RIGHT_OVERLAY_CANDIDATES = ['#parserToolsDropdown', '#toolPanel', '.tool-panel', '.right-panel', '.panel-right', '.overlay']

VIEWPORTS = [(1280,800),(1024,768),(390,844)]


def find_and_click(page, selectors, timeout=1500):
    for sel in selectors:
        try:
            el = page.query_selector(sel)
            if el:
                # attempt DOM-dispatched MouseEvent first (bubbles, cancelable)
                try:
                    res = page.evaluate("sel => { const el = document.querySelector(sel); if(!el) return false; el.dispatchEvent(new MouseEvent('click', {bubbles:true,cancelable:true, view:window})); return true; }", sel)
                    if res:
                        print(f"Dispatched DOM click to {sel}")
                        return sel
                except Exception:
                    # fallback to element.click()
                    try:
                        print(f"Falling back to element.click() for {sel}")
                        el.click()
                        return sel
                    except Exception:
                        continue
        except Exception:
            continue
    return None


def derive_overlay_candidates_near(page, x, y, max_ancestors=6):
    """Probe the page at (x,y), walk up ancestors and return an ordered list of stable selectors.
    Prefers ids, aria-controls, data-target, then class-based selectors.
    """
    try:
        script = r"""
        (x,y,maxAnc) => {
            function cleanClassList(el){
                try{ return Array.from(el.classList).filter(Boolean).slice(0,3); }catch(e){return []}
            }
            const out = [];
            let el = document.elementFromPoint(x,y);
            let depth = 0;
            while(el && depth < maxAnc){
                // prefer id
                if(el.id){ out.push('#'+el.id); }
                // aria-controls
                const ac = el.getAttribute && el.getAttribute('aria-controls');
                if(ac){ out.push('[aria-controls="'+ac+'"]'); }
                // data-target / data-action
                for(const attr of ['data-target','data-action','data-overlay','data-panel']){
                    const v = el.getAttribute && el.getAttribute(attr);
                    if(v) out.push('['+attr+'="'+v+'"]');
                }
                // role + aria-label
                const role = el.getAttribute && el.getAttribute('role');
                const al = el.getAttribute && el.getAttribute('aria-label');
                if(role && al){ out.push(role+ '[aria-label="'+al+'"]'); }
                // class selector fallback
                const classes = cleanClassList(el);
                if(classes.length){ out.push(el.tagName.toLowerCase()+'.'+classes.join('.')); }
                el = el.parentElement;
                depth += 1;
            }
            // dedupe while preserving order
            const seen = new Set();
            return out.filter(s => { if(seen.has(s)) return false; seen.add(s); return true; });
        }
        """
        res = page.evaluate(script, x, y, max_ancestors)
        if isinstance(res, list) and res:
            return res
    except Exception:
        pass
    return []


def wait_for_any(page, candidates, timeout=2000):
    end = time.time() + timeout/1000.0
    while time.time() < end:
        for c in candidates:
            try:
                el = page.query_selector(c)
                if el:
                    b = el.evaluate("el => ({visible: !!(el.offsetParent), aria: el.getAttribute('aria-hidden')})")
                    visible = b and b.get('visible')
                    aria = b and b.get('aria')
                    if visible or aria == 'false':
                        return c
            except Exception:
                continue
        time.sleep(0.05)
    return None


with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    for w,h in VIEWPORTS:
        page.set_viewport_size({'width': w, 'height': h})
        page.goto(URL, wait_until='domcontentloaded')
        # allow client JS to initialize (socket, DOM tweaks)
        page.wait_for_timeout(800)
        base = OUT_DIR / f'seq_{w}x{h}'

        # Baseline screenshot
        bpath = base.with_suffix('.base.png')
        page.screenshot(path=str(bpath), full_page=True)
        print('Saved', bpath)

        # Open More menu
        sel = find_and_click(page, MORE_SELECTORS)
        if sel:
            found = wait_for_any(page, MORE_OVERLAY_CANDIDATES, timeout=2000)
            time.sleep(0.25)
            mpath = base.with_suffix('.more_open.png')
            page.screenshot(path=str(mpath), full_page=True)
            print('Saved', mpath, 'after clicking', sel, 'overlay found', found)
        else:
            print('No candidate clicked for More menu')

        # Open Right tool
        # Attempt to use mapped cluster centers (from tools/cluster_dom_mapping.json) to probe for selectors,
        # falling back to a heuristic point near the right edge.
        try:
            mapping_path = Path('tools/cluster_dom_mapping.json')
            probe_x = int(w * 0.85)
            probe_y = int(h * 0.15)
            if mapping_path.exists():
                try:
                    with open(mapping_path, 'r', encoding='utf-8') as fh:
                        mapping = json.load(fh)
                    key = f'seq_{w}x{h}.right_open.png'
                    entries = mapping.get(key) or mapping.get(key.replace('.png',''))
                    if entries and isinstance(entries, list) and len(entries) > 0:
                        # prefer largest cluster (index 0) if present
                        first = entries[0]
                        c = first.get('cluster') or {}
                        center = c.get('center')
                        if isinstance(center, list) and len(center) >= 2:
                            probe_x = int(center[0])
                            probe_y = int(center[1])
                except Exception:
                    pass
            derived = derive_overlay_candidates_near(page, probe_x, probe_y)
            if derived:
                print('Derived right overlay candidates:', derived)
                RIGHT_OVERLAY_CANDIDATES = derived + RIGHT_OVERLAY_CANDIDATES
        except Exception:
            pass

        sel2 = find_and_click(page, RIGHT_SELECTORS)
        if sel2:
            found2 = wait_for_any(page, RIGHT_OVERLAY_CANDIDATES, timeout=2000)
            time.sleep(0.25)
            rpath = base.with_suffix('.right_open.png')
            page.screenshot(path=str(rpath), full_page=True)
            print('Saved', rpath, 'after clicking', sel2, 'overlay found', found2)
        else:
            print('No candidate clicked for right tool')

        # Combined screenshot
        cpath = base.with_suffix('.both_open.png')
        page.screenshot(path=str(cpath), full_page=True)
        print('Saved combined', cpath)

    browser.close()
print('Done')
