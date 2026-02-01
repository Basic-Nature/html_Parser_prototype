#!/usr/bin/env python3
"""Smoke test: enable PROMETHEUS and verify /metrics returns 200 and includes counters.

This script creates a lightweight stub for `prometheus_client` so the webapp
registers its `/metrics` route on import, then uses Flask test client to GET it.
It emits a non-zero exit code on failure.
"""
import importlib
import json
import os
import sys
import types

# Ensure repository root is on sys.path so `import webapp...` works
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def install_stub_prometheus():
    # build a minimal prometheus_client stub that reads our telemetry_counters.json
    mod = types.ModuleType("prometheus_client")

    def generate_latest(registry=None):
        path = os.path.join("webapp", "parser", "log", "telemetry_counters.json")
        out_lines = []
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as fh:
                    counters = json.load(fh)
            else:
                counters = {}
        except Exception:
            counters = {}
        for k, v in (counters.items() if isinstance(counters, dict) else []):
            name = f"smart_{k}"
            try:
                val = float(v)
            except Exception:
                val = 0
            out_lines.append(f"{name} {val}")
        return ("\n".join(out_lines) + "\n").encode("utf-8")

    mod.generate_latest = generate_latest
    mod.CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"
    mod.REGISTRY = object()
    sys.modules["prometheus_client"] = mod


def main():
    os.environ["ENABLE_PROMETHEUS"] = "true"
    # install stub before importing the webapp so the route registers
    install_stub_prometheus()

    # Ensure a fresh import
    sys.modules.pop("webapp.Smart_Elections_Parser_Webapp", None)
    try:
        mod = importlib.import_module("webapp.Smart_Elections_Parser_Webapp")
    except Exception as e:
        print("ERROR: importing webapp.Smart_Elections_Parser_Webapp failed:", e)
        raise

    app = getattr(mod, "app", None)
    if app is None:
        print("ERROR: app not found on imported module")
        sys.exit(2)

    client = app.test_client()
    resp = client.get("/metrics")
    print(f"/metrics -> status {resp.status_code}")
    body = resp.data.decode("utf-8", "replace")
    print("--- body start ---")
    print(body)
    print("--- body end ---")

    ok = resp.status_code == 200
    # require at least one of our smart_* metrics to appear if counters file exists
    counters_path = os.path.join("webapp", "parser", "log", "telemetry_counters.json")
    if os.path.exists(counters_path):
        try:
            counters = json.load(open(counters_path, "r", encoding="utf-8"))
        except Exception:
            counters = {}
        for k in counters.keys():
            if f"smart_{k}" in body:
                break
        else:
            print("ERROR: expected counter keys not found in /metrics output")
            ok = False

    if not ok:
        sys.exit(3)
    print("PROMETHEUS smoke test: OK")


if __name__ == "__main__":
    main()
