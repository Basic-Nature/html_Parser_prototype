#!/usr/bin/env python3
import json
import os
from collections import Counter

PATH = os.path.join('tools','debug_headless_output','network_capture.json')
if not os.path.exists(PATH):
    print('No capture found at', PATH)
    raise SystemExit(1)

with open(PATH,'r',encoding='utf-8') as f:
    data = json.load(f)

reqs = [r for r in data if r.get('type')=='request']
resps = [r for r in data if r.get('type')=='response']
print('total_records', len(data))
print('requests', len(reqs), 'responses', len(resps))

urls = [r.get('url') for r in data]
print('\nTop requested URLs:')
for u,n in Counter(urls).most_common(12):
    print(n, u)

non200 = [r for r in resps if isinstance(r.get('status'), int) and r.get('status') != 200]
print('\nnon-200 responses:', len(non200))
for r in non200:
    print(r.get('status'), r.get('url'))

print('\nSummary written: tools/debug_headless_output/network_capture.json')
