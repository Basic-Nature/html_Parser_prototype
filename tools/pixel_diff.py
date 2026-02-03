import json
import os
import shutil

from PIL import Image, ImageChops

OUT_DIR = os.path.join('tools', 'debug_screenshots')
BASELINE_DIR = os.path.join(OUT_DIR, 'baseline')
DIFF_DIR = os.path.join(OUT_DIR, 'diffs')

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DIFF_DIR, exist_ok=True)

def list_shot_images():
    out = []
    for fn in os.listdir(OUT_DIR):
        if fn.startswith('shot_') and fn.lower().endswith('.png'):
            out.append(fn)
    out.sort()
    return out

def ensure_baseline(shot_names):
    # If baseline folder missing, create it and copy all current shots as baseline
    if not os.path.exists(BASELINE_DIR):
        os.makedirs(BASELINE_DIR, exist_ok=True)
        for name in shot_names:
            src = os.path.join(OUT_DIR, name)
            dst = os.path.join(BASELINE_DIR, name)
            if os.path.exists(src):
                shutil.copy2(src, dst)
        return True
    return False

def compare_images(baseline_path, current_path, diff_path):
    try:
        a = Image.open(baseline_path).convert('RGBA')
        b = Image.open(current_path).convert('RGBA')
    except Exception as e:
        return {'error': str(e)}
    if a.size != b.size:
        return {'mismatch': True, 'reason': 'size', 'size_a': a.size, 'size_b': b.size}
    diff = ImageChops.difference(a, b)
    bbox = diff.getbbox()
    if bbox is None:
        return {'identical': True}
    # count non-zero pixels
    nonzero = 0
    px = diff.getdata()
    for p in px:
        if p[0] or p[1] or p[2] or p[3]:
            nonzero += 1
    total = a.size[0] * a.size[1]
    percent = (nonzero / total) * 100.0
    try:
        diff.save(diff_path)
    except Exception:
        pass
    return {'identical': False, 'bbox': bbox, 'diff_pixels': nonzero, 'total_pixels': total, 'percent_diff': percent}

def main():
    shot_names = list_shot_images()
    baseline_created = ensure_baseline(shot_names)
    report = {'baseline_created': baseline_created, 'comparisons': []}
    # compare each shot image found
    for name in shot_names:
        base = os.path.join(BASELINE_DIR, name)
        cur = os.path.join(OUT_DIR, name)
        entry = {'image': name}
        if not os.path.exists(cur):
            entry['error'] = 'current_missing'
            report['comparisons'].append(entry)
            continue
        if not os.path.exists(base):
            # auto-create baseline for this shot by copying current
            try:
                shutil.copy2(cur, base)
                entry['status'] = 'baseline_added'
                entry['note'] = 'baseline created from current'
                entry['identical'] = True
                report['comparisons'].append(entry)
            except Exception as e:
                entry['error'] = f'baseline_create_failed: {e}'
                report['comparisons'].append(entry)
            continue
        diff_path = os.path.join(DIFF_DIR, f'diff_{name}')
        res = compare_images(base, cur, diff_path)
        entry.update(res)
        if res.get('identical'):
            entry['status'] = 'identical'
        elif res.get('error'):
            entry['status'] = 'error'
        else:
            entry['status'] = 'different'
            entry['diff_image'] = diff_path if os.path.exists(diff_path) else None
        report['comparisons'].append(entry)

    report_path = os.path.join(OUT_DIR, 'pixel_diff_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f'Report written to {report_path}')
    for c in report['comparisons']:
        print(c)

if __name__ == '__main__':
    main()
