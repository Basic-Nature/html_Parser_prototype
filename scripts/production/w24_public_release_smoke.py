"""Bounded, read-only public endpoint checks. Not a replacement for browser acceptance."""
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request

BASE = 'https://www.electionpulse.org'
PATHS = ('/health', '/ballot_lens', '/projects', '/auth/oidc/status')


def get(path: str) -> dict:
    url = BASE + path
    req = urllib.request.Request(url, headers={'User-Agent': 'ElectionPulse-W24-Pilot-Acceptance/1.0'})
    try:
        with urllib.request.urlopen(req, timeout=12) as response:
            data = response.read(4097)
            return {'route': path, 'status': response.status, 'bounded': len(data) <= 4096}
    except urllib.error.HTTPError as exc:
        exc.read(4097)
        return {'route': path, 'status': exc.code, 'bounded': True}
    except Exception:
        return {'route': path, 'status': None, 'bounded': False}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--oidc-expected', choices=('disabled','enabled'), required=True)
    args = parser.parse_args()
    rows = [get(p) for p in PATHS]
    expected = {'/health': 200, '/ballot_lens': 200, '/projects': 200,
                '/auth/oidc/status': 404 if args.oidc_expected == 'disabled' else 200}
    passed = all(row['status'] == expected[row['route']] and row['bounded'] for row in rows)
    print(json.dumps({'result': 'PASS' if passed else 'HOLD', 'checks': rows,
                      'real_browser_parsing': 'NOT_TESTED', 'live_login': 'NOT_TESTED',
                      'download_and_socket_io': 'NOT_TESTED'}, indent=2))
    return 0 if passed else 3


if __name__ == '__main__':
    raise SystemExit(main())
