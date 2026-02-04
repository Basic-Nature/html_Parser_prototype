import json
import os
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path('tools/screenshots')
OUT = Path('tools')
OUT.mkdir(exist_ok=True)
URL = os.environ.get('WEBAPP_URL', 'http://127.0.0.1:5000/ballot_lens')
MAX_W = 800

def compute_clusters(base_fp, cur_fp):
    base = Image.open(base_fp).convert('RGB')
    cur = Image.open(cur_fp).convert('RGB')
    bw, bh = base.size
    scale = 1.0
    if bw > MAX_W:
        scale = MAX_W / bw
        base_s = base.resize((int(bw*scale), int(bh*scale))).convert('RGB')
        cur_s = cur.resize((int(bw*scale), int(bh*scale))).convert('RGB')
    else:
        base_s = base
        cur_s = cur
    w,h = base_s.size
    diff_mask = [[0]*w for _ in range(h)]
    changed = 0
    for y in range(h):
        for x in range(w):
            pb = base_s.getpixel((x,y))
            pc = cur_s.getpixel((x,y))
            d = abs(pb[0]-pc[0]) + abs(pb[1]-pc[1]) + abs(pb[2]-pc[2])
            if d > 30:
                diff_mask[y][x] = 1
                changed += 1
    # cluster flood-fill
    visited = [[False]*w for _ in range(h)]
    clusters = []
    for y in range(h):
        for x in range(w):
            if visited[y][x] or diff_mask[y][x]==0:
                continue
            stack = [(x, y)]
            minx = x
            miny = y
            maxx = x
            maxy = y
            area = 0
            while stack:
                cx,cy = stack.pop()
                if cx<0 or cy<0 or cx>=w or cy>=h:
                    continue
                if visited[cy][cx] or diff_mask[cy][cx]==0:
                    continue
                visited[cy][cx]=True
                area += 1
                if cx < minx:
                    minx = cx
                if cy < miny:
                    miny = cy
                if cx > maxx:
                    maxx = cx
                if cy > maxy:
                    maxy = cy
                stack.extend([(cx+1,cy),(cx-1,cy),(cx,cy+1),(cx,cy-1)])
            if area>20:
                clusters.append((minx,miny,maxx+1,maxy+1,area))
    # map clusters to original coords
    mapped = []
    for c in clusters:
        x1,y1,x2,y2,a = c
        ox1 = int(x1/scale)
        oy1 = int(y1/scale)
        ox2 = int(x2/scale)
        oy2 = int(y2/scale)
        cx = (ox1+ox2)//2
        cy = (oy1+oy2)//2
        mapped.append({'box':(ox1,oy1,ox2,oy2),'center':(cx,cy),'area':a})
    return mapped

def find_pairs():
    files = sorted(ROOT.glob('seq_*'))
    pairs = []
    for f in files:
        name = f.name
        if name.endswith('.base.png'):
            base = f
            prefix = name.replace('.base.png','')
            for suffix in ('.more_open.png','.right_open.png','.both_open.png'):
                cand = ROOT/(prefix+suffix)
                if cand.exists():
                    pairs.append((base,cand))
    return pairs

def map_clusters_to_dom(pairs):
    results = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        for base_fp, cur_fp in pairs:
            print('Processing', cur_fp.name)
            clusters = compute_clusters(base_fp, cur_fp)
            # load page and set viewport to base image size
            img = Image.open(base_fp)
            w,h = img.size
            try:
                page.set_viewport_size({'width': w, 'height': min(h, 1200)})
            except Exception:
                pass
            page.goto(URL, wait_until='domcontentloaded', timeout=60000)
            page.wait_for_timeout(500)
            mapped_info = []
            for idx, c in enumerate(clusters):
                cx,cy = c['center']
                vh = page.evaluate('() => window.innerHeight')
                # scroll so cy is vertically centered
                scroll_y = max(0, cy - vh//2)
                page.evaluate(f'scrollTo(0, {scroll_y})')
                page.wait_for_timeout(120)
                # compute point relative to viewport
                point_x = min(max(2, cx), page.evaluate('() => document.documentElement.scrollWidth')-2)
                point_y = vh//2
                try:
                    el_info = page.evaluate("p => { const x=p[0], y=p[1]; const el = document.elementFromPoint(x,y); if(!el) return null; const r=el.getBoundingClientRect(); return {tag: el.tagName, id: el.id, class: el.className, rect: {x:r.x,y:r.y,width:r.width,height:r.height}} }", [point_x, point_y])
                except Exception as e:
                    el_info = {'error': str(e)}
                mapped_info.append({'cluster_index': idx, 'cluster': c, 'point': (point_x, point_y), 'element': el_info})
            results[cur_fp.name] = mapped_info
        browser.close()
    return results

if __name__ == '__main__':
    pairs = find_pairs()
    if not pairs:
        print('No seq_* base/open pairs found in', ROOT)
        raise SystemExit(0)
    res = map_clusters_to_dom(pairs)
    outp = OUT / 'cluster_dom_mapping.json'
    with open(outp, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2)
    print('Wrote', outp)
