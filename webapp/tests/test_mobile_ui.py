#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys

import pytest


def _extract_json_payload(stdout: str) -> dict | None:
    lines = (stdout or "").split("\n")
    json_start = next((i for i, line in enumerate(lines) if line.strip().startswith("{")), None)
    if json_start is None:
        return None
    try:
        return json.loads("\n".join(lines[json_start:]))
    except json.JSONDecodeError:
        return None


def test_mobile_ui_robust_check():
    """Run mobile UI headless check when local webapp is available."""
    result = subprocess.run(
        [
            sys.executable,
            "tools/ui_robust_check.py",
            "--url",
            "http://127.0.0.1:5000/ballot_lens",
            "--viewport",
            "mobile",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    output = result.stdout or ""
    if "ERR_CONNECTION_REFUSED" in output:
        pytest.skip("Local webapp is not running at http://127.0.0.1:5000; skipping mobile UI check.")

    payload = _extract_json_payload(output)
    if payload is None:
        pytest.fail(f"Could not parse mobile UI check output: {output[:500]}")

    tests = payload.get("tests", [])
    failed_tests = [t for t in tests if not t.get("passed")]
    if failed_tests:
        names = ", ".join(str(t.get("name", "unknown")) for t in failed_tests)
        pytest.fail(f"Mobile UI robust check failed for: {names}")
