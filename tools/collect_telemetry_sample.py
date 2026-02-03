#!/usr/bin/env python3
"""
Simple smoke test: emit a telemetry event and print recent telemetry lines.
"""
from __future__ import annotations

import json
import os
import uuid

try:
    from webapp.parser.utils.telemetry import TELEMETRY_PATH, emit_telemetry_event
except Exception:
    # Fallback: try package-relative imports if run from different cwd
    import sys
    sys.path.insert(0, os.getcwd())
    from webapp.parser.utils.telemetry import TELEMETRY_PATH, emit_telemetry_event


def main():
    run_id = f"smoke_{uuid.uuid4().hex[:8]}"
    payload = {
        "run_id": run_id,
        "note": "telemetry smoke test",
        "sample_metric": 123,
        "url": "https://example.com/smoke"
    }
    print("Emitting telemetry event:", payload)
    try:
        emit_telemetry_event("smoke_test", payload)
        print("emit_telemetry_event completed (no exception).")
    except Exception as e:
        print("emit_telemetry_event raised:", e)

    print("Reading last 10 lines from telemetry file:", TELEMETRY_PATH)
    try:
        if os.path.exists(TELEMETRY_PATH):
            with open(TELEMETRY_PATH, "rb") as f:
                lines = f.readlines()
            for line in lines[-10:]:
                try:
                    print(json.loads(line))
                except Exception:
                    print(line.decode("utf-8", errors="replace"))
        else:
            print("Telemetry file not found:", TELEMETRY_PATH)
    except Exception as e:
        print("Failed reading telemetry file:", e)


if __name__ == "__main__":
    main()
