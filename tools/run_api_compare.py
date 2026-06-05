#!/usr/bin/env python3
"""Run local-vs-Azure smoke comparison with one command.

Example:
  python tools/run_api_compare.py --azure-base-url https://your-app.azurewebsites.net
"""

from __future__ import annotations

import argparse
import sys

import smoke_webapp_api as smoke


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare local and Azure API smoke suites")
    parser.add_argument("--azure-base-url", required=True, help="Azure webapp base URL")
    parser.add_argument("--local-base-url", default="http://127.0.0.1:5000", help="Local webapp base URL")
    parser.add_argument("--timeout", type=float, default=12.0, help="Per-request timeout in seconds")
    parser.add_argument("--local-output", default="tools/tmp/local_smoke.json", help="Local artifact output path")
    parser.add_argument("--azure-output", default="tools/tmp/azure_smoke.json", help="Azure artifact output path")
    parser.add_argument(
        "--allow-differences",
        action="store_true",
        help="Return success even if status differences are detected",
    )
    args = parser.parse_args()

    local_base = smoke._normalize_base_url(args.local_base_url)
    azure_base = smoke._normalize_base_url(args.azure_base_url)
    checks = list(smoke._default_checks())

    print(f"Running local smoke suite: {local_base}")
    local_results = smoke.run_smoke_suite(local_base, args.timeout, checks)
    smoke._print_report(local_results)
    smoke.write_artifact(args.local_output, smoke.build_artifact(local_base, args.timeout, local_results))
    print(f"Local artifact written: {args.local_output}")

    print(f"\nRunning Azure smoke suite: {azure_base}")
    azure_results = smoke.run_smoke_suite(azure_base, args.timeout, checks)
    smoke._print_report(azure_results)
    smoke.write_artifact(args.azure_output, smoke.build_artifact(azure_base, args.timeout, azure_results))
    print(f"Azure artifact written: {args.azure_output}")

    diffs = smoke.compare_results(local_results, azure_results)
    smoke._print_compare_report(local_base, azure_base, diffs)

    local_failures = sum(1 for r in local_results if not r.ok)
    azure_failures = sum(1 for r in azure_results if not r.ok)
    if local_failures or azure_failures:
        return 1
    if diffs and not args.allow_differences:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
