#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys

import pytest


def _extract_json_payload(stdout: str) -> dict | None:
    """Extract the failure-only JSON block emitted by ui_robust_check.py."""
    output = stdout or ""
    marker = "Full results JSON:"
    marker_index = output.rfind(marker)

    if marker_index < 0:
        return None

    payload_text = output[
        marker_index + len(marker):
    ].strip()

    if not payload_text:
        return None

    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        return None

    return payload if isinstance(payload, dict) else None


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
    error_output = result.stderr or ""
    combined_output = "\n".join(
        part
        for part in (output, error_output)
        if part
    )

    if "ERR_CONNECTION_REFUSED" in combined_output:
        pytest.skip(
            "Local webapp is not running at "
            "http://127.0.0.1:5000; skipping mobile UI check."
        )

    # ui_robust_check.py defines exit code 0 as an all-passed result.
    # Successful runs intentionally use a human-readable summary rather than
    # the failure-only JSON diagnostic payload.
    if result.returncode == 0:
        return

    payload = _extract_json_payload(output)

    if payload is None:
        pytest.fail(
            "Mobile UI robust check exited "
            f"{result.returncode} without parseable JSON: "
            f"{combined_output[:500]}"
        )

    tests = payload.get("tests", [])
    failed_tests = [
        test
        for test in tests
        if not test.get("passed")
    ]

    if failed_tests:
        names = ", ".join(
            str(test.get("name", "unknown"))
            for test in failed_tests
        )
        pytest.fail(
            f"Mobile UI robust check failed for: {names}"
        )

    pytest.fail(
        "Mobile UI robust check exited "
        f"{result.returncode} without failed test details: "
        f"{combined_output[:500]}"
    )
