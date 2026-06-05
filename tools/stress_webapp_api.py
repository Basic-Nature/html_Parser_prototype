#!/usr/bin/env python3
"""Concurrent API stress test for webapp structural integrity.

This test validates endpoint stability under load by checking:
- expected status conformance
- 5xx rate
- aggregate failure rate

Example:
  python tools/stress_webapp_api.py --base-url http://127.0.0.1:5000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class StressEndpoint:
    name: str
    method: str
    path: str
    expected_statuses: tuple[int, ...]
    payload: dict | None = None


@dataclass
class SampleResult:
    endpoint: str
    method: str
    path: str
    ok: bool
    status: int | None
    elapsed_ms: int
    detail: str


def _normalize_base_url(base_url: str) -> str:
    cleaned = (base_url or "").strip()
    if not cleaned:
        cleaned = "http://127.0.0.1:5000"
    if not cleaned.startswith(("http://", "https://")):
        cleaned = "http://" + cleaned
    return cleaned.rstrip("/")


def _default_endpoints() -> list[StressEndpoint]:
    return [
        StressEndpoint("Health", "GET", "/health", (200,)),
        StressEndpoint("Heartbeat", "GET", "/heartbeat", (200,)),
        StressEndpoint("Certificate Info", "GET", "/api/auth/certificate_info", (200, 401, 403)),
        StressEndpoint("Data Framework Preview", "GET", "/api/data_framework/preview?mode=active", (200, 401, 403)),
        StressEndpoint("Integrity Trends", "GET", "/api/integrity_trends", (200, 401, 403)),
        StressEndpoint(
            "Integrity Signal",
            "POST",
            "/api/integrity_signal",
            (200, 400, 401, 403),
            payload={"signal_type": "stress_check", "value": 0.5},
        ),
    ]


def _sample_once(base_url: str, timeout: float, endpoint: StressEndpoint) -> SampleResult:
    url = f"{base_url}{endpoint.path}"
    headers = {"Accept": "application/json"}
    body = None
    if endpoint.payload is not None:
        body = json.dumps(endpoint.payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = Request(url=url, method=endpoint.method.upper(), headers=headers, data=body)
    started = time.perf_counter()
    try:
        with urlopen(req, timeout=timeout) as response:
            status = int(getattr(response, "status", 0) or 0)
            sample = response.read(180).decode("utf-8", errors="replace").strip().replace("\n", " ")
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            return SampleResult(
                endpoint=endpoint.name,
                method=endpoint.method,
                path=endpoint.path,
                ok=status in endpoint.expected_statuses,
                status=status,
                elapsed_ms=elapsed_ms,
                detail=sample[:180],
            )
    except HTTPError as exc:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        status = int(exc.code)
        sample = exc.read(180).decode("utf-8", errors="replace").strip().replace("\n", " ")
        return SampleResult(
            endpoint=endpoint.name,
            method=endpoint.method,
            path=endpoint.path,
            ok=status in endpoint.expected_statuses,
            status=status,
            elapsed_ms=elapsed_ms,
            detail=sample[:180],
        )
    except URLError as exc:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return SampleResult(
            endpoint=endpoint.name,
            method=endpoint.method,
            path=endpoint.path,
            ok=False,
            status=None,
            elapsed_ms=elapsed_ms,
            detail=f"URLError: {exc.reason}",
        )
    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return SampleResult(
            endpoint=endpoint.name,
            method=endpoint.method,
            path=endpoint.path,
            ok=False,
            status=None,
            elapsed_ms=elapsed_ms,
            detail=f"Error: {exc}",
        )


def _percentile(sorted_values: list[int], percentile: float) -> int:
    if not sorted_values:
        return 0
    idx = int(round((len(sorted_values) - 1) * percentile))
    idx = max(0, min(idx, len(sorted_values) - 1))
    return sorted_values[idx]


def _build_jobs(endpoints: list[StressEndpoint], requests_per_endpoint: int) -> list[StressEndpoint]:
    jobs: list[StressEndpoint] = []
    for endpoint in endpoints:
        jobs.extend([endpoint] * max(1, requests_per_endpoint))
    random.shuffle(jobs)
    return jobs


def _summarize(results: list[SampleResult]) -> dict:
    endpoint_names = sorted({r.endpoint for r in results})
    by_endpoint = {}
    total_5xx = 0
    total_failures = 0
    all_latencies = []

    for name in endpoint_names:
        subset = [r for r in results if r.endpoint == name]
        statuses = Counter(str(r.status) if r.status is not None else "n/a" for r in subset)
        latencies = sorted(r.elapsed_ms for r in subset)
        failures = [r for r in subset if not r.ok]
        five_xx = [r for r in subset if r.status is not None and 500 <= r.status <= 599]

        total_5xx += len(five_xx)
        total_failures += len(failures)
        all_latencies.extend(latencies)

        by_endpoint[name] = {
            "total": len(subset),
            "passed": len(subset) - len(failures),
            "failed": len(failures),
            "five_xx": len(five_xx),
            "status_counts": dict(sorted(statuses.items())),
            "latency_ms": {
                "p50": _percentile(latencies, 0.50),
                "p95": _percentile(latencies, 0.95),
                "max": latencies[-1] if latencies else 0,
            },
        }

    total = len(results)
    failure_rate = (total_failures / total) if total else 1.0
    all_latencies_sorted = sorted(all_latencies)

    return {
        "summary": {
            "total_requests": total,
            "failures": total_failures,
            "failure_rate": failure_rate,
            "five_xx": total_5xx,
            "latency_ms": {
                "p50": _percentile(all_latencies_sorted, 0.50),
                "p95": _percentile(all_latencies_sorted, 0.95),
                "max": all_latencies_sorted[-1] if all_latencies_sorted else 0,
            },
        },
        "endpoints": by_endpoint,
    }


def _print_summary(report: dict, max_failure_rate: float, max_5xx: int) -> None:
    summary = report["summary"]
    print("\n=== API Stress Summary ===")
    print(
        f"total={summary['total_requests']} failures={summary['failures']} "
        f"failure_rate={summary['failure_rate']:.3f} 5xx={summary['five_xx']} "
        f"latency_ms(p50={summary['latency_ms']['p50']}, p95={summary['latency_ms']['p95']}, max={summary['latency_ms']['max']})"
    )
    print(f"Thresholds: max_failure_rate={max_failure_rate:.3f}, max_5xx={max_5xx}")

    for endpoint, data in report["endpoints"].items():
        print(
            f"- {endpoint}: total={data['total']} passed={data['passed']} failed={data['failed']} "
            f"5xx={data['five_xx']} p95={data['latency_ms']['p95']}ms statuses={data['status_counts']}"
        )


def _write_report(path: str, base_url: str, requests_per_endpoint: int, concurrency: int, timeout: float, report: dict) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "requests_per_endpoint": requests_per_endpoint,
        "concurrency": concurrency,
        "timeout_seconds": timeout,
        **report,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Stress test core webapp APIs for structural integrity")
    parser.add_argument("--base-url", default="http://127.0.0.1:5000", help="Base URL for the webapp")
    parser.add_argument("--requests-per-endpoint", type=int, default=30, help="Requests to run per endpoint")
    parser.add_argument("--concurrency", type=int, default=12, help="Concurrent workers")
    parser.add_argument("--timeout", type=float, default=8.0, help="Per-request timeout in seconds")
    parser.add_argument("--max-failure-rate", type=float, default=0.03, help="Maximum acceptable failure rate")
    parser.add_argument("--max-5xx", type=int, default=0, help="Maximum acceptable 5xx responses")
    parser.add_argument("--output-json", default="", help="Optional JSON report output path")
    args = parser.parse_args()

    base_url = _normalize_base_url(args.base_url)
    endpoints = _default_endpoints()
    jobs = _build_jobs(endpoints, args.requests_per_endpoint)

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        results = list(executor.map(lambda endpoint: _sample_once(base_url, args.timeout, endpoint), jobs))
    runtime_ms = int((time.perf_counter() - started) * 1000)

    report = _summarize(results)
    report["summary"]["runtime_ms"] = runtime_ms
    _print_summary(report, args.max_failure_rate, args.max_5xx)

    if args.output_json:
        _write_report(
            args.output_json,
            base_url,
            args.requests_per_endpoint,
            args.concurrency,
            args.timeout,
            report,
        )
        print(f"Report written: {args.output_json}")

    summary = report["summary"]
    if summary["five_xx"] > args.max_5xx:
        return 2
    if summary["failure_rate"] > args.max_failure_rate:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
