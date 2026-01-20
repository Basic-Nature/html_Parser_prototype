# Test csv_locate and view_csv via Flask test_client (no external server)
import json
import sys
from pathlib import Path

# Ensure repo root on sys.path
ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from webapp.Smart_Elections_Parser_Webapp import app

ROOT_PARAM = 'output'
SUBPATH = 'Alabama__Washington__Attorney_General__20260108_185043'
NAME = 'results.csv'
ROW = 2

with app.test_client() as client:
    print('Calling /csv_locate via test_client...')
    resp = client.get('/csv_locate', query_string={'root': ROOT_PARAM, 'path': SUBPATH, 'name': NAME, 'row': ROW})
    print('Status:', resp.status_code)
    try:
        j = resp.get_json()
    except Exception:
        j = None
    print('JSON:', json.dumps(j, indent=2))
    viewer = j.get('viewer') if isinstance(j, dict) else None
    if viewer:
        print('Fetching viewer path via test_client:', viewer)
        r2 = client.get(viewer)
        print('Viewer status:', r2.status_code)
        txt = r2.get_data(as_text=True)
        print('Viewer snippet:', txt[:800].replace('\n',' '))
    else:
        print('No viewer returned')
