#!/usr/bin/env python3
"""Lightweight metrics server for CI/dev smoke testing.

Starts an HTTP server exposing Prometheus metrics at /metrics and a test-only
POST /test/metrics/increment that increments a test counter. Exits with 0
when verification succeeds.
"""
import http.server
import json
import socket
import threading
import time

from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, Counter, generate_latest

TEST_COUNTER = Counter('test_metrics_increment_total', 'Test-only increment counter')


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # keep output minimal
        pass

    def do_GET(self):
        if self.path.startswith('/metrics'):
            data = generate_latest(REGISTRY)
            self.send_response(200)
            self.send_header('Content-Type', CONTENT_TYPE_LATEST)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path.startswith('/test/metrics/increment'):
            try:
                TEST_COUNTER.inc()
                body = json.dumps({'success': True}).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception:
                self.send_response(500)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()


def find_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('127.0.0.1', 0))
    addr, port = s.getsockname()
    s.close()
    return port


def run_server(port, ready_evt):
    with http.server.ThreadingHTTPServer(('127.0.0.1', port), Handler) as httpd:
        ready_evt.set()
        httpd.serve_forever()


def verify(port):
    import urllib.error
    import urllib.request

    base = f'http://127.0.0.1:{port}'
    # Hit increment endpoint
    try:
        req = urllib.request.Request(base + '/test/metrics/increment', method='POST')
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.getcode() != 200:
                print('Increment endpoint failed', resp.getcode())
                return False
    except Exception as e:
        print('Increment request failed:', e)
        return False

    # Fetch metrics and ensure counter present
    try:
        with urllib.request.urlopen(base + '/metrics', timeout=5) as resp:
            if resp.getcode() != 200:
                print('Metrics endpoint not 200', resp.getcode())
                return False
            data = resp.read().decode('utf-8', errors='ignore')
            if 'test_metrics_increment_total' in data:
                print('Metrics verified')
                return True
            print('Test counter not found in metrics')
            return False
    except Exception as e:
        print('Metrics request failed:', e)
        return False


def main():
    port = find_free_port()
    ready = threading.Event()
    t = threading.Thread(target=run_server, args=(port, ready), daemon=True)
    t.start()
    if not ready.wait(timeout=3):
        print('Server failed to start')
        raise SystemExit(1)
    # Give server a moment
    time.sleep(0.2)
    ok = verify(port)
    if ok:
        print('Fallback metrics smoke test succeeded')
        raise SystemExit(0)
    else:
        print('Fallback metrics smoke test failed')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
