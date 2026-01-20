from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import json

ROOT = Path('tools/screenshots')
MAPPING = Path('tools/cluster_dom_mapping.json')
OUT = Path('tools/reports')
OUT.mkdir(parents=True, exist_ok=True)
ANNOT_DIR = OUT / 'annotated'
ANNOT_DIR.mkdir(exist_ok=True)

TOP_N = 8

def short_elem(el):
    if not el:
        return ''
    if 'error' in el:
        return f"ERROR:{el.get('error')[:60]}"
    parts = []
    tag = el.get('tag')
    if tag: parts.append(tag.lower())
    eid = el.get('id')
    if eid: parts.append(f"#{eid}")
    cls = el.get('class')
    if cls:
        clsshort = ' '.join(cls.split()[:2])
        parts.append(f".{clsshort}")
    return ' '.join(parts)

def annotate_image(img_path, clusters):
    im = Image.open(img_path).convert('RGBA')
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for i, item in enumerate(clusters[:TOP_N]):
        box = item['cluster']['box']
        x1,y1,x2,y2 = box
        # draw rectangle
        draw.rectangle([x1,y1,x2,y2], outline=(255,0,0,200), width=3)
        label = f"#{i} {short_elem(item.get('element'))}"
        # text background
        try:
            if font:
                tw, th = font.getsize(label)
            else:
                tw, th = draw.textbbox((0,0), label)[2:4]
        except Exception:
            try:
                tw, th = draw.textbbox((0,0), label)[2:4]
            except Exception:
                tw, th = (len(label)*6, 12)
        draw.rectangle([x1, max(0,y1-th-4), x1+tw+6, max(0,y1)], fill=(0,0,0,160))
        draw.text((x1+3, max(0,y1-th-2)), label, fill=(255,255,255,255), font=font)
    outp = ANNOT_DIR / (Path(img_path).name.replace('.png', '.annotated.png'))
    im.save(outp)
    return outp

def main():
    if not MAPPING.exists():
        print('Mapping file not found:', MAPPING)
        return
    mapping = json.loads(MAPPING.read_text(encoding='utf-8'))
    report = {}
    for shot, items in mapping.items():
        # sort clusters by area desc
        items_sorted = sorted(items, key=lambda x: x['cluster']['area'], reverse=True)
        top = items_sorted[:TOP_N]
        img_fp = ROOT / shot
        if not img_fp.exists():
            # maybe combined images have different suffix; skip if missing
            print('Missing screenshot for', shot)
            continue
        annotated = annotate_image(img_fp, top)
        summary = []
        for idx, it in enumerate(top):
            elem = it.get('element') or {}
            summary.append({
                'index': idx,
                'box': it['cluster']['box'],
                'area': it['cluster']['area'],
                'center': it['cluster']['center'],
                'mapped': {
                    'tag': elem.get('tag'),
                    'id': elem.get('id'),
                    'class': elem.get('class'),
                },
            })
        report[shot] = {'annotated': str(annotated), 'top': summary}
    out_json = OUT / 'cluster_report.json'
    out_html = OUT / 'cluster_report.html'
    out_json.write_text(json.dumps(report, indent=2), encoding='utf-8')
    # simple HTML
    with open(out_html, 'w', encoding='utf-8') as f:
        f.write('<html><body>\n')
        f.write('<h1>Cluster Mapping Report</h1>\n')
        for shot, info in report.items():
            f.write(f"<h2>{shot}</h2>\n")
            f.write(f"<img src='{Path(info['annotated']).as_posix()}' style='max-width:100%;height:auto'>\n")
            f.write('<ul>\n')
            for t in info['top']:
                f.write(f"<li>#{t['index']} box={t['box']} area={t['area']} -> {t['mapped']['tag']} {t['mapped']['id'] or ''} {t['mapped']['class'] or ''}</li>\n")
            f.write('</ul>\n')
        f.write('</body></html>')
    print('Wrote', out_json, 'and', out_html)

if __name__ == '__main__':
    main()
