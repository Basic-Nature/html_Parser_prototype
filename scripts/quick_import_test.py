import importlib, sys

try:
    m = importlib.import_module('webapp.Smart_Elections_Parser_Webapp')
    print('IMPORT OK')
    # Ensure minimal globals expected by handlers when imported in test context
    import os
    m.INPUT_DIR = getattr(m, 'INPUT_DIR', os.path.abspath(os.path.join(os.getcwd(), 'input')))
    m.OUTPUT_DIR = getattr(m, 'OUTPUT_DIR', os.path.abspath(os.path.join(os.getcwd(), 'output')))
    m.UPLOADS_DIR = getattr(m, 'UPLOADS_DIR', os.path.abspath(os.path.join(os.getcwd(), 'uploads')))
    m.LOG_DIR = getattr(m, 'LOG_DIR', os.path.abspath(os.path.join(os.getcwd(), 'logs')))
    # Provide a simple safe_split fallback if missing
    if not hasattr(m, 'safe_split'):
        def _ss(s, sep):
            try:
                return [t for t in str(s).split(sep) if t]
            except Exception:
                return []
        m.safe_split = _ss

    app = getattr(m, 'app', None)
    if app is None:
        print('NO_FLASK_APP_FOUND')
        sys.exit(2)
    with app.test_client() as c:
        r = c.get('/view_csv?root=output&path=Alabama__Washington__Attorney_General__20260108_185043&name=results.csv')
        print('STATUS', r.status_code)
        data = r.get_data(as_text=True)
        print(data[:1000])
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(3)
