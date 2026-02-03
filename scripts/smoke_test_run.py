#!/usr/bin/env python3
import json
import os
import time

import orjson
import webapp.parser.html_election_parser as hep

# Import the pipeline and config
from webapp.parser import web_pipeline
from webapp.parser.config import PROCESSED_URLS_FILE


# Stub out the heavy main() with a lightweight simulator
def stub_main(**kwargs):
    session_id = kwargs.get('session_id')
    emit = kwargs.get('emit_func')
    urls = kwargs.get('urls') or ['http://example/1','http://example/2','http://example/3']
    entries = []
    for i, u in enumerate(urls, 1):
        status = 'success' if i % 2 == 1 else 'fail'
        entry = {
            'url': u,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'status': status,
        }
        # For test: mark second entry as flagged with extra metadata and low confidence
        if i == 2:
            entry['flagged_for_review'] = True
            entry['flagged_reason'] = 'Low extraction confidence'
            entry['metadata'] = {
                'handler': 'stub_handler',
                'contest': 'Test Contest',
                'state': 'TestState',
                'county': 'TestCounty',
                'quality_metrics': {'extraction_confidence': 0.21}
            }
        entries.append(entry)
        if emit:
            emit({
                'type': 'processed_url',
                'session_id': session_id,
                'url': u,
                'status': status,
                'ts': time.time(),
            })
        time.sleep(0.05)

    # Persist simulated processed url cache
    os.makedirs(os.path.dirname(PROCESSED_URLS_FILE), exist_ok=True)
    with open(PROCESSED_URLS_FILE, 'wb') as f:
        f.write(orjson.dumps(entries, option=orjson.OPT_INDENT_2))
    return ([], [])

# Replace the real main with our stub
hep.main = stub_main
# Also override any already-imported reference in web_pipeline
try:
    import webapp.parser.web_pipeline as wp
    wp.main = stub_main
except Exception:
    pass

# Simple emit function that prints JSON lines
def emit_func(obj):
    print('EMIT:', json.dumps(obj))

def main():
    session_id = 'smoke_' + str(int(time.time()))
    cancel_flag = web_pipeline.cancellation_manager.get_flag(session_id)
    # Run the pipeline with our stubbed main
    web_pipeline.process_urls_for_web(
        prompt_queue=None,
        session_id=session_id,
        cancel_flag=cancel_flag,
        emit_func=emit_func,
        urls=['http://example/1','http://example/2','http://example/3'],
        output_bypass=True,
        disable_internal_heartbeat=True,
    )

    # Show generated report files
    rep_dir = os.path.join('output', 'reports')
    print('\nREPORT_DIR:', rep_dir)
    if os.path.isdir(rep_dir):
        files = sorted(os.listdir(rep_dir))
        print('FILES:', files)
        if files:
            last = os.path.join(rep_dir, files[-1])
            print('LAST_REPORT_PATH:', last)
            with open(last, 'r', encoding='utf-8') as f:
                print('\nLAST_REPORT_CONTENT:\n')
                print(f.read())
    else:
        print('No reports directory found.')

    # Print processed urls cache
    if os.path.exists(PROCESSED_URLS_FILE):
        print('\nPROCESSED_URLS_FILE:', PROCESSED_URLS_FILE)
        with open(PROCESSED_URLS_FILE, 'rb') as f:
            print(f.read().decode('utf-8'))
    else:
        print('\nPROCESSED_URLS_FILE missing')

if __name__ == '__main__':
    main()
