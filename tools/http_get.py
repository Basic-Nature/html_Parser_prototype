import os

import requests

URL = os.environ.get('WEBAPP_URL', 'http://127.0.0.1:5000/ballot_lens')
print('GET', URL)
try:
    r = requests.get(URL, timeout=5)
    print('status', r.status_code)
    print('len', len(r.text))
    print('has btnNavMore?', 'btnNavMore' in r.text)
    print('has parserToolsDropdown?', 'parserToolsDropdown' in r.text)
    # print a snippet
    print(r.text[:500])
except Exception as e:
    print('error', e)
