import importlib, sys, os, json

try:
    m = importlib.import_module('webapp.Smart_Elections_Parser_Webapp')
    print('IMPORT OK')
    # Minimal globals
    m.INPUT_DIR = getattr(m, 'INPUT_DIR', os.path.abspath(os.path.join(os.getcwd(), 'input')))
    m.OUTPUT_DIR = getattr(m, 'OUTPUT_DIR', os.path.abspath(os.path.join(os.getcwd(), 'output')))
    m.UPLOADS_DIR = getattr(m, 'UPLOADS_DIR', os.path.abspath(os.path.join(os.getcwd(), 'uploads')))
    m.LOG_DIR = getattr(m, 'LOG_DIR', os.path.abspath(os.path.join(os.getcwd(), 'logs')))
    if not hasattr(m, 'safe_split'):
        def _ss(s, sep):
            try:
                return [t for t in str(s).split(sep) if t]
            except Exception:
                return []
        m.safe_split = _ss
    app = getattr(m, 'app', None)
    if app is None:
        print('NO FLASK APP FOUND')
        sys.exit(2)

    with app.test_client() as c:
        params = {'root':'output','path':'Alabama__Washington__Attorney_General__20260108_185043','name':'results.csv','row':2}
        r = c.get('/csv_locate', query_string=params)
        print('LOCATE STATUS', r.status_code)
        try:
            js = r.get_json()
        except Exception:
            print('LOCATE BODY', r.get_data(as_text=True)[:2000])
            sys.exit(3)
        print('LOCATE JSON', js)
        viewer = js.get('viewer')
        if not viewer:
            print('No viewer URL returned')
            sys.exit(4)
        # viewer is path like /view_csv?.... Use test_client to fetch
        r2 = c.get(viewer)
        print('VIEWER STATUS', r2.status_code)
        text = r2.get_data(as_text=True)
        print('VIEWER SNIPPET', text[:1000].replace('\n',' '))

except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(5)
