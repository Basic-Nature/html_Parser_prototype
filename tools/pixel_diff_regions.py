import json
import os

from PIL import Image, ImageChops

OUT_DIR = os.path.join('tools', 'debug_screenshots')
BASELINE_DIR = os.path.join(OUT_DIR, 'baseline')
DIFF_DIR = os.path.join(OUT_DIR, 'diffs_regions')
LAYOUT_REPORT = os.path.join(OUT_DIR, 'layout_report.json')

os.makedirs(DIFF_DIR, exist_ok=True)

HEADER_HEIGHT = 160  # px


def load_layout():
    if not os.path.exists(LAYOUT_REPORT):
        raise FileNotFoundError(f"Layout report not found: {LAYOUT_REPORT}")
    with open(LAYOUT_REPORT, 'r', encoding='utf-8') as f:
        return json.load(f)


def crop_region(img, box):
    # box: (x, y, width, height)
    x, y, w, h = box
    x = int(max(0, x))
    y = int(max(0, y))
    w = int(max(1, w))
    h = int(max(1, h))
    return img.crop((x, y, x + w, y + h))


def compare_crop(img_a, img_b):
    # assumes same size
    diff = ImageChops.difference(img_a, img_b)
    bbox = diff.getbbox()
    if bbox is None:
        return {'identical': True, 'diff_pixels': 0, 'total_pixels': img_a.size[0]*img_a.size[1], 'percent_diff': 0.0, 'bbox': None}, None
    # count non-zero pixels
    nonzero = 0
    for p in diff.getdata():
        if p[0] or p[1] or p[2] or (len(p) > 3 and p[3]):
            nonzero += 1
    total = img_a.size[0]*img_a.size[1]
    percent = (nonzero / total) * 100.0
    return {'identical': False, 'diff_pixels': nonzero, 'total_pixels': total, 'percent_diff': percent, 'bbox': bbox}, diff


def main():
    layout = load_layout()
    results = []
    for r in layout.get('results', []):
        width = r.get('width')
        shot = r.get('screenshot')
        if not shot:
            results.append({'width': width, 'error': 'no_screenshot'})
            continue
        # normalize paths
        shot_path = os.path.normpath(shot)
        shot_name = os.path.basename(shot_path)
        base_path = os.path.join(BASELINE_DIR, shot_name)
        cur_path = os.path.join(OUT_DIR, shot_name)
        entry = {'width': width, 'image': shot_name}
        if not os.path.exists(cur_path):
            entry['error'] = 'current_missing'
            results.append(entry)
            continue
        if not os.path.exists(base_path):
            entry['error'] = 'baseline_missing'
            results.append(entry)
            continue
        try:
            base_img = Image.open(base_path).convert('RGBA')
            cur_img = Image.open(cur_path).convert('RGBA')
        except Exception as e:
            entry['error'] = f'load_failed: {e}'
            results.append(entry)
            continue

        # Header region
        header_box = (0, 0, base_img.size[0], HEADER_HEIGHT)
        base_header = crop_region(base_img, header_box)
        cur_header = crop_region(cur_img, header_box)
        hdr_res, hdr_diff = compare_crop(base_header, cur_header)
        hdr_diff_path = None
        if hdr_diff is not None:
            hdr_diff_path = os.path.join(DIFF_DIR, f'diff_header_{shot_name}')
            try:
                hdr_diff.save(hdr_diff_path)
            except Exception:
                hdr_diff_path = None

        # Main grid region (use main_box from layout)
        main_box_raw = r.get('main_box')
        if not main_box_raw:
            entry['main_error'] = 'no_main_box'
            results.append(entry)
            continue
        main_box = (main_box_raw.get('x', 0), main_box_raw.get('y', 0), main_box_raw.get('width', 100), main_box_raw.get('height', 100))
        # ensure crop within image
        bw, bh = base_img.size
        mx, my, mw, mh = main_box
        if mx < 0:
            mx = 0
        if my < 0:
            my = 0
        if mx + mw > bw:
            mw = bw - mx
        if my + mh > bh:
            mh = bh - my
        main_box = (mx, my, mw, mh)
        base_main = crop_region(base_img, main_box)
        cur_main = crop_region(cur_img, main_box)
        main_res, main_diff = compare_crop(base_main, cur_main)
        main_diff_path = None
        if main_diff is not None:
            main_diff_path = os.path.join(DIFF_DIR, f'diff_main_{shot_name}')
            try:
                main_diff.save(main_diff_path)
            except Exception:
                main_diff_path = None

        entry.update({'header': hdr_res, 'header_diff': hdr_diff_path, 'main': main_res, 'main_diff': main_diff_path})
        results.append(entry)

    report = {'url': layout.get('url'), 'regions': ['header', 'main'], 'results': results}
    out_path = os.path.join(OUT_DIR, 'pixel_region_report.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print('Region report written to', out_path)
    for e in results:
        print(e)

if __name__ == '__main__':
    main()
