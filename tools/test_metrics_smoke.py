"""
Smoke test: start the webapp with ENABLE_PROMETHEUS=true and verify /metrics returns 200
and contains the expected metric names.
"""
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

HOST = os.environ.get('TEST_HOST', '127.0.0.1')
START_PORT = int(os.environ.get('TEST_PORT', '5000'))
TRIES = int(os.environ.get('TEST_PORT_TRIES', '5'))
PY = os.environ.get('PYTHON', sys.executable or 'python')


def start_server(env=None, log_path=None):
    # Start server inheriting the current environment, then overlay any overrides.
    base_env = os.environ.copy()
    if isinstance(env, dict):
        base_env.update(env)
    base_env['ENABLE_PROMETHEUS'] = 'true'
    cmd = [PY, '-m', 'webapp.Smart_Elections_Parser_Webapp']
    lf = open(log_path, 'ab') if log_path else subprocess.PIPE
    proc = subprocess.Popen(cmd, env=base_env, stdout=lf, stderr=subprocess.STDOUT)
    return proc, lf


def try_wait_for_metrics(host, port, timeout=20):
    url = f'http://{host}:{port}/metrics'
    deadline = time.time() + timeout
    last_exc = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    return resp.read().decode('utf-8', errors='ignore')
        except Exception as e:
            last_exc = e
            time.sleep(0.5)
    raise RuntimeError(f"Timed out waiting for /metrics on {host}:{port}: {last_exc}")


def main():
    print('Starting robust metrics smoke test')
    tmpdir = tempfile.mkdtemp(prefix='metrics_smoke_')
    log_path = os.path.join(tmpdir, 'server.log')
    print('Logs will be saved to', log_path)

    expected = [
        'smart_processed_total',
        'smart_processed_success',
        'smart_processed_fail',
        'smart_fallbacks_total',
    ]

    proc = None
    lf = None
    success = False
    try:
        for port in range(START_PORT, START_PORT + TRIES):
            print(f'Trying port {port}...')
            try:
                proc, lf = start_server(env={'PORT': str(port)}, log_path=log_path)
            except Exception as e:
                print('Failed to start server:', e)
                continue

            try:
                data = try_wait_for_metrics(HOST, port, timeout=25)
                print('Received /metrics; checking for expected counters...')
                missing = [m for m in expected if m not in data]
                if missing:
                    print('Missing metrics on port', port, ':', missing)
                    # not a success; terminate and try next port
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                    proc.wait(timeout=2)
                    proc = None
                    continue
                print('OK: /metrics contains expected counters on port', port)
                success = True
                break
            except Exception as e:
                print('Port', port, 'did not respond with metrics yet:', e)
                try:
                    proc.terminate()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=2)
                except Exception:
                    pass
                proc = None
                continue

        if success:
            print('Smoke test succeeded; server logs at:', log_path)
            return 0
        else:
            print('Smoke test failed; no port returned expected metrics. Logs at:', log_path)
            return 2

    finally:
        if proc:
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        if lf and hasattr(lf, 'close'):
            try:
                lf.close()
            except Exception:
                pass


if __name__ == '__main__':
    rc = main()
    # leave logs around for debugging
    sys.exit(rc)
