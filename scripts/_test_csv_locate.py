import time
import sys
try:
    import requests
except Exception:
    requests = None
import urllib.request

BASE = 'http://127.0.0.1:5000'
ROOT = 'output'
SUBPATH = 'Alabama__Washington__Attorney_General__20260108_185043'
NAME = 'results.csv'
ROW = 2

print('Testing /csv_locate for', f'{ROOT}/{SUBPATH}/{NAME}', 'row', ROW)
# wait for server
for i in range(60):
    try:
        with urllib.request.urlopen(BASE + '/health', timeout=2) as r:
            print('Server ready')
            break
    except Exception:
        time.sleep(0.5)
else:
    print('Server did not respond on', BASE, file=sys.stderr)
    sys.exit(2)

params = f'?root={ROOT}&path={SUBPATH}&name={NAME}&row={ROW}'
loc_url = BASE + '/csv_locate' + params
print('Calling', loc_url)
try:
    if requests:
        resp = requests.get(BASE + '/csv_locate', params={'root': ROOT, 'path': SUBPATH, 'name': NAME, 'row': ROW}, timeout=10)
        print('Status:', resp.status_code)
        print('JSON:', resp.text)
        js = resp.json()
    else:
        with urllib.request.urlopen(loc_url, timeout=10) as r:
            body = r.read().decode('utf-8')
            print('Body:', body)
            import json
            js = json.loads(body)
except Exception as e:
    print('csv_locate request failed:', e)
    sys.exit(3)

viewer = js.get('viewer') if isinstance(js, dict) else None
print('Viewer URL:', viewer)
if not viewer:
    sys.exit(0)

full = BASE + viewer
print('Fetching viewer HTML:', full)
try:
    if requests:
        r2 = requests.get(full, timeout=10)
        print('Viewer status:', r2.status_code)
        text = r2.text
    else:
        with urllib.request.urlopen(full, timeout=10) as r:
            text = r.read().decode('utf-8')
    print('Viewer snippet:', text[:800].replace('\n',' '))
    if 'highlight=' in full or 'highlight=' in text:
        print('Highlight param present')
    else:
        print('No highlight param found in viewer URL or HTML')
except Exception as e:
    print('Failed to fetch viewer:', e)
    sys.exit(4)

print('Test complete')
