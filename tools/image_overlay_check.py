from pathlib import Path

from PIL import Image

ROOT = Path('tools/screenshots')
files = sorted(ROOT.glob('*.png'))
if not files:
    print('No screenshots found in', ROOT)
    raise SystemExit(0)

MAX_W = 800  # downscale for faster processing

def dominant_color(img, region=None, samples=2000):
    if region:
        img = img.crop(region)
    small = img.resize((50,50))
    colors = small.convert('RGB').getcolors(50*50)
    colors.sort(key=lambda x: x[0], reverse=True)
    return colors[0][1] if colors else (0,0,0)


def find_bright_regions(img, downscale=1):
    # convert to grayscale and threshold for bright areas
    gray = img.convert('L')
    w, h = gray.size
    data = gray.load()
    visited = [[False]*w for _ in range(h)]
    # allow detection of both bright and dark/translucent overlays
    bright_thresh = 220
    dark_thresh = 40
    regions = []
    for y in range(h):
        for x in range(w):
            if visited[y][x]:
                continue
            visited[y][x] = True
            v = data[x,y]
            if v < bright_thresh and v > dark_thresh:
                continue
            # flood fill
            stack = [(x,y)]
            minx, miny, maxx, maxy = x, y, x, y
            area = 0
            while stack:
                cx, cy = stack.pop()
                if cx < 0 or cy < 0 or cx >= w or cy >= h:
                    continue
                if visited[cy][cx]:
                    continue
                visited[cy][cx] = True
                cv = data[cx,cy]
                if cv < bright_thresh and cv > dark_thresh:
                    continue
                area += 1
                if cx < minx: minx = cx
                if cy < miny: miny = cy
                if cx > maxx: maxx = cx
                if cy > maxy: maxy = cy
                # neighbours
                stack.append((cx+1, cy))
                stack.append((cx-1, cy))
                stack.append((cx, cy+1))
                stack.append((cx, cy-1))
            if area > 30:
                regions.append((minx, miny, maxx+1, maxy+1, area))
    # merge overlapping small regions
    regions.sort(key=lambda r: -r[4])
    merged = []
    for r in regions:
        x1,y1,x2,y2,a = r
        merged_flag = False
        for i, m in enumerate(merged):
            mx1,my1,mx2,my2,ma = m
            # if overlap or close
            if not (x2 < mx1-4 or x1 > mx2+4 or y2 < my1-4 or y1 > my2+4):
                nx1 = min(x1,mx1); ny1 = min(y1,my1); nx2 = max(x2,mx2); ny2 = max(y2,my2)
                merged[i] = (nx1,ny1,nx2,ny2, ma + a)
                merged_flag = True
                break
        if not merged_flag:
            merged.append(r)
    return merged

for fp in files:
    print('---')
    print(fp.name)
    im = Image.open(fp)
    ow, oh = im.size
    print('Size:', ow, 'x', oh, 'bytes:', fp.stat().st_size)
    # navbar region sample top 120px
    nav_h = min(120, oh//8)
    nav_region = (0,0,ow,nav_h)
    nav_color = dominant_color(im, region=nav_region)
    print('Navbar dominant color (RGB):', nav_color)

    # downscale image
    scale = 1.0
    if ow > MAX_W:
        scale = MAX_W/ow
        w = int(ow*scale); h = int(oh*scale)
        small = im.resize((w,h))
    else:
        w,h = ow,oh
        small = im.copy()
    regions = find_bright_regions(small)
    print('Detected overlay-like regions by brightness (downscaled coords):', len(regions))
    for i, r in enumerate(regions[:8]):
        x1,y1,x2,y2,a = r
        # map to original coords
        ox1 = int(x1/scale); oy1 = int(y1/scale); ox2 = int(x2/scale); oy2 = int(y2/scale)
        print(f'  Region {i}: box={ox1,oy1,ox2,oy2} area~{a} (downscaled)')
        # sample mean color in original
        crop = im.crop((ox1,oy1,ox2,oy2)).convert('RGB')
        px = crop.getdata()
        cnt = 0; rs=gs=bs=0
        for p in px:
            rs += p[0]; gs += p[1]; bs += p[2]; cnt +=1
        if cnt:
            print('    mean RGB:', (rs//cnt, gs//cnt, bs//cnt))
        # check clipping
        clipped = ox1 <= 2 or oy1 <=2 or ox2 >= ow-2 or oy2 >= oh-2
        print('    clipped to edge?', clipped)
    if not regions:
        print('  No strong bright/dark overlay-like regions detected by single-image analysis.')

    # If corresponding base image exists (same prefix .base.png), compute pixel-diff regions
    name = fp.name
    base_fp = None
    for suffix in ('.base.png','_base.png'):
        cand = ROOT / name.replace('.more_open.png', suffix).replace('.right_open.png', suffix).replace('.both_open.png', suffix)
        if cand.exists():
            base_fp = cand
            break
    if not base_fp:
        # try replacing the whole descriptor with .base
        if 'more_open' in name or 'right_open' in name or 'both_open' in name:
            cand = ROOT / name.replace('.more_open.png', '.base.png').replace('.right_open.png', '.base.png').replace('.both_open.png', '.base.png')
            if cand.exists():
                base_fp = cand
    if base_fp and base_fp.exists():
        try:
            base_im = Image.open(base_fp).convert('RGB')
            cur_im = im.convert('RGB')
            # downscale both for speed
            bw, bh = base_im.size
            scale = 1.0
            max_w = 800
            if bw > max_w:
                scale = max_w / bw
                base_small = base_im.resize((int(bw*scale), int(bh*scale))).convert('RGB')
                cur_small = cur_im.resize((int(bw*scale), int(bh*scale))).convert('RGB')
            else:
                base_small = base_im
                cur_small = cur_im

            bw, bh = base_small.size
            diff_mask = [[0]*bw for _ in range(bh)]
            changed = 0
            total = bw*bh
            for y in range(bh):
                for x in range(bw):
                    pb = base_small.getpixel((x,y))
                    pc = cur_small.getpixel((x,y))
                    d = abs(pb[0]-pc[0]) + abs(pb[1]-pc[1]) + abs(pb[2]-pc[2])
                    if d > 30:
                        diff_mask[y][x] = 1
                        changed += 1
            pct = (changed/total)*100.0 if total else 0.0
            print('Pixel-diff vs base:', base_fp.name, f'changed_pixels={changed} ({pct:.2f}%)')
            # derive bounding boxes of changed clusters (simple scan)
            clusters = []
            visited = [[False]*bw for _ in range(bh)]
            for y in range(bh):
                for x in range(bw):
                    if visited[y][x] or diff_mask[y][x]==0:
                        continue
                    # flood
                    stack=[(x,y)]
                    minx = x; miny = y; maxx = x; maxy = y; area=0
                    while stack:
                        cx,cy = stack.pop()
                        if cx<0 or cy<0 or cx>=bw or cy>=bh:
                            continue
                        if visited[cy][cx] or diff_mask[cy][cx]==0:
                            continue
                        visited[cy][cx]=True
                        area+=1
                        if cx<minx: minx=cx
                        if cy<miny: miny=cy
                        if cx>maxx: maxx=cx
                        if cy>maxy: maxy=cy
                        stack.extend([(cx+1,cy),(cx-1,cy),(cx,cy+1),(cx,cy-1)])
                    if area>20:
                        clusters.append((minx,miny,maxx+1,maxy+1,area))
            print('Detected diff clusters (downscaled coords):', len(clusters))
            for i,c in enumerate(clusters[:6]):
                x1,y1,x2,y2,a = c
                ox1=int(x1/scale); oy1=int(y1/scale); ox2=int(x2/scale); oy2=int(y2/scale)
                print(f'  Cluster {i}: box={ox1,oy1,ox2,oy2} area~{a} (downscaled)')
        except Exception as e:
            print('Diff analysis failed:', e)

print('\nDone')
