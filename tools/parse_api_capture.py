#!/usr/bin/env python3
import json
import os
from urllib.parse import urlparse

PATH = os.path.join('tools','debug_headless_output','network_capture.json')
if not os.path.exists(PATH):
    print('No capture found at', PATH)
    raise SystemExit(1)

with open(PATH,'r',encoding='utf-8') as f:
    data = json.load(f)

requests = [r for r in data if r.get('type')=='request']
responses = [r for r in data if r.get('type')=='response']

# Build quick lookup of responses by url
resp_by_url = {}
for r in responses:
    resp_by_url.setdefault(r.get('url'), []).append(r)

def is_api_like(url):
    if not url:
        return False
    p = urlparse(url)
    path = (p.path or '').lower()
    return ('/api/' in path) or ('/socket.io' in path) or ('/upload' in path) or ('/download' in path) or ('warehouse' in path)

matched = [req for req in requests if is_api_like(req.get('url'))]

if not matched:
    print('No API-like requests found in capture.')
    raise SystemExit(0)

def trunc(s, n=1000):
    if s is None:
        return ''
    s = str(s)
    return s if len(s) <= n else s[:n] + '\n...(truncated)'

for i, req in enumerate(matched, 1):
    url = req.get('url')
    print(f"[{i}] {req.get('method')} {url}")
    pd = req.get('post_data')
    if pd:
        print('  Request body:')
        print('   ', trunc(pd, 1000).replace('\n','\n    '))
    resp_list = resp_by_url.get(url, [])
    if resp_list:
        for r in resp_list:
            print(f"  Response: {r.get('status')}")
            b = r.get('body')
            if b:
                print('    Body:')
                print('     ', trunc(b, 1000).replace('\n','\n     '))
    else:
        print('  No matching response recorded in capture.')
    print('-'*60)
