#!/usr/bin/env python3
"""Smoke test critical Smart Elections webapp API routes.

Usage examples:
  python tools/smoke_webapp_api.py
  python tools/smoke_webapp_api.py --base-url http://127.0.0.1:5000
  python tools/smoke_webapp_api.py --base-url https://<azure-app>.azurewebsites.net

This script is designed for concrete, fast verification of GET/POST behavior.
It treats auth-protected responses (401/403) as acceptable for protected routes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class EndpointCheck:
    name: str
    method: str
    path: str
    expected_statuses: tuple[int, ...]
    payload: dict | None = None


@dataclass
class CheckResult:
    name: str
    method: str
    url: str
    ok: bool
    status: int | None
    elapsed_ms: int
    detail: str


def _result_to_dict(result: CheckResult) -> dict:
    return {
        "name": result.name,
        "method": result.method,
        "url": result.url,
        "ok": result.ok,
        "status": result.status,
        "elapsed_ms": result.elapsed_ms,
        "detail": result.detail,
    }


def _normalize_base_url(base_url: str) -> str:
    cleaned = (base_url or "").strip()
    if not cleaned:
        cleaned = "http://127.0.0.1:5000"
    if not cleaned.startswith(("http://", "https://")):
        cleaned = "http://" + cleaned
    return cleaned.rstrip("/")


def _run_check(base_url: str, timeout: float, check: EndpointCheck) -> CheckResult:
    url = f"{base_url}{check.path}"
    headers = {"Accept": "application/json"}
    body_bytes = None
    if check.payload is not None:
        body_bytes = json.dumps(check.payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = Request(url=url, method=check.method.upper(), headers=headers, data=body_bytes)
    started = time.perf_counter()
    try:
        with urlopen(req, timeout=timeout) as response:
            status = int(getattr(response, "status", 0) or 0)
            data = response.read(220).decode("utf-8", errors="replace")
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            ok = status in check.expected_statuses
            detail = data.strip().replace("\n", " ")
            if len(detail) > 180:
                detail = detail[:180] + "..."
            return CheckResult(check.name, check.method, url, ok, status, elapsed_ms, detail)
    except HTTPError as exc:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        status = int(exc.code)
        sample = exc.read(220).decode("utf-8", errors="replace").strip().replace("\n", " ")
        if len(sample) > 180:
            sample = sample[:180] + "..."
        ok = status in check.expected_statuses
        return CheckResult(check.name, check.method, url, ok, status, elapsed_ms, sample)
    except URLError as exc:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return CheckResult(check.name, check.method, url, False, None, elapsed_ms, f"URLError: {exc.reason}")
    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return CheckResult(check.name, check.method, url, False, None, elapsed_ms, f"Error: {exc}")


def _default_checks() -> Iterable[EndpointCheck]:
    return (
        EndpointCheck(
            name="Health",
            method="GET",
            path="/health",
            expected_statuses=(200,),
        ),
        EndpointCheck(
            name="Heartbeat",
            method="GET",
            path="/heartbeat",
            expected_statuses=(200,),
        ),
        EndpointCheck(
            name="Certificate Info",
            method="GET",
            path="/api/auth/certificate_info",
            expected_statuses=(200, 401, 403),
        ),
        EndpointCheck(
            name="Data Framework Preview",
            method="GET",
            path="/api/data_framework/preview?mode=active",
            expected_statuses=(200, 401, 403),
        ),
        EndpointCheck(
            name="Integrity Trends",
            method="GET",
            path="/api/integrity_trends",
            expected_statuses=(200, 401, 403),
        ),
        EndpointCheck(
            name="Integrity Signal",
            method="POST",
            path="/api/integrity_signal",
            expected_statuses=(200, 400, 401, 403),
            payload={"signal_type": "smoke_check", "value": 0.5},
        ),
    )


def _print_report(results: list[CheckResult]) -> None:
    print("\n=== Webapp API Smoke Report ===")
    for r in results:
        status_str = str(r.status) if r.status is not None else "n/a"
        verdict = "PASS" if r.ok else "FAIL"
        print(f"[{verdict}] {r.method:4s} {r.url} ({status_str}, {r.elapsed_ms} ms) :: {r.name}")
        if r.detail:
            print(f"       {r.detail}")

    passed = sum(1 for r in results if r.ok)
    total = len(results)
    print(f"\nSummary: {passed}/{total} checks passed")


def run_smoke_suite(base_url: str, timeout: float, checks: Iterable[EndpointCheck] | None = None) -> list[CheckResult]:
    selected_checks = list(checks) if checks is not None else list(_default_checks())
    return [_run_check(base_url, timeout, check) for check in selected_checks]


def build_artifact(base_url: str, timeout: float, results: list[CheckResult]) -> dict:
    passed = sum(1 for r in results if r.ok)
    total = len(results)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "timeout_seconds": timeout,
        "summary": {
            "passed": passed,
            "total": total,
            "failed": total - passed,
        },
        "results": [_result_to_dict(r) for r in results],
    }


def write_artifact(path: str, artifact: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(artifact, indent=2), encoding="utf-8")


def _results_by_name(results: list[CheckResult]) -> dict[str, CheckResult]:
    return {f"{r.method} {r.name}": r for r in results}


def compare_results(primary: list[CheckResult], secondary: list[CheckResult]) -> list[str]:
    diffs: list[str] = []
    a = _results_by_name(primary)
    b = _results_by_name(secondary)
    keys = sorted(set(a.keys()) | set(b.keys()))
    for key in keys:
        left = a.get(key)
        right = b.get(key)
        if left is None or right is None:
            diffs.append(f"{key}: missing in {'primary' if left is None else 'secondary'} suite")
            continue
        if left.ok != right.ok or left.status != right.status:
            diffs.append(
                f"{key}: primary(ok={left.ok}, status={left.status}) vs secondary(ok={right.ok}, status={right.status})"
            )
    return diffs


def _print_compare_report(primary_name: str, secondary_name: str, diffs: list[str]) -> None:
    print("\n=== Suite Comparison ===")
    print(f"Primary:   {primary_name}")
    print(f"Secondary: {secondary_name}")
    if not diffs:
        print("No status differences detected across shared checks.")
        return
    print(f"Differences: {len(diffs)}")
    for item in diffs:
        print(f"- {item}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test Smart Elections webapp GET/POST APIs")
    parser.add_argument("--base-url", default="http://127.0.0.1:5000", help="Base URL for the webapp")
    parser.add_argument("--compare-base-url", default="", help="Optional second base URL to compare against primary")
    parser.add_argument("--timeout", type=float, default=8.0, help="Per-request timeout in seconds")
    parser.add_argument("--output-json", default="", help="Optional output path for JSON artifact")
    parser.add_argument("--compare-output-json", default="", help="Optional output path for comparison suite JSON artifact")
    parser.add_argument(
        "--strict-compare",
        action="store_true",
        help="Return non-zero when compare suite has status differences from primary",
    )
    args = parser.parse_args()

    base_url = _normalize_base_url(args.base_url)
    checks = list(_default_checks())
    results = run_smoke_suite(base_url, args.timeout, checks)
    _print_report(results)

    if args.output_json:
        artifact = build_artifact(base_url, args.timeout, results)
        write_artifact(args.output_json, artifact)
        print(f"Artifact written: {args.output_json}")

    compare_exit_code = 0
    if args.compare_base_url:
        compare_base_url = _normalize_base_url(args.compare_base_url)
        print(f"\nRunning comparison suite against: {compare_base_url}")
        compare_results_list = run_smoke_suite(compare_base_url, args.timeout, checks)
        _print_report(compare_results_list)

        if args.compare_output_json:
            compare_artifact = build_artifact(compare_base_url, args.timeout, compare_results_list)
            write_artifact(args.compare_output_json, compare_artifact)
            print(f"Comparison artifact written: {args.compare_output_json}")

        diffs = compare_results(results, compare_results_list)
        _print_compare_report(base_url, compare_base_url, diffs)
        if args.strict_compare and diffs:
            compare_exit_code = 2

    failed = [r for r in results if not r.ok]
    if failed:
        return 1
    return compare_exit_code


if __name__ == "__main__":
    sys.exit(main())
