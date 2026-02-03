import importlib
import os
import sys
import traceback

m = importlib.import_module('webapp.Smart_Elections_Parser_Webapp')
# provide minimal globals
m.INPUT_DIR = getattr(m, 'INPUT_DIR', os.path.abspath(os.path.join(os.getcwd(),'input')))
m.OUTPUT_DIR = getattr(m, 'OUTPUT_DIR', os.path.abspath(os.path.join(os.getcwd(),'output')))
m.UPLOADS_DIR = getattr(m, 'UPLOADS_DIR', os.path.abspath(os.path.join(os.getcwd(),'uploads')))
app = getattr(m, 'app', None)
if app is None:
    print('No app')
    sys.exit(2)
app.testing = True
with app.test_client() as c:
    try:
        resp = c.get('/csv_locate', query_string={'root':'output','path':'Alabama__Washington__Attorney_General__20260108_185043','name':'results.csv','row':2})
        print('STATUS', resp.status_code)
        print(resp.get_data(as_text=True)[:4000])
    except Exception:
        traceback.print_exc()
        sys.exit(3)
