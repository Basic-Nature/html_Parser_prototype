#!/usr/bin/env python3
"""Smart Phase B validation runner.

This runner keeps validation focused and dynamic:
- Always runs the new parser URL policy orchestration test.
- Dynamically adds parser-adjacent tests/lint checks based on changed files.
- Supports a strict mode to include broader adjacency checks when desired.

Usage:
    python tools/smart_phase_b_validation.py
    python tools/smart_phase_b_validation.py --strict
    python tools/smart_phase_b_validation.py --changed webapp/parser/utils/output_utils.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str]) -> int:
    print(f"\n[RUN] {' '.join(command)}")
    completed = subprocess.run(command, cwd=str(REPO_ROOT), check=False)
    return completed.returncode


def _git_changed_files() -> list[str]:
    """Return changed files in working tree (staged + unstaged + untracked)."""
    files: set[str] = set()
    for diff_cmd in (
        ["git", "diff", "--name-only"],
        ["git", "diff", "--cached", "--name-only"],
        ["git", "ls-files", "--others", "--exclude-standard"],
    ):
        completed = subprocess.run(
            diff_cmd,
            cwd=str(REPO_ROOT),
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            continue
        for line in completed.stdout.splitlines():
            line = line.strip()
            if line:
                files.add(line.replace("\\", "/"))
    return sorted(files)


def _collect_targets(changed: list[str], strict: bool) -> tuple[list[str], list[str]]:
    """Build pytest and ruff targets from a changed-file set."""
    pytest_targets: set[str] = {
        "webapp/tests/test_html_parser_database_policy_flow.py",
    }
    ruff_targets: set[str] = {
        "webapp/tests/test_html_parser_database_policy_flow.py",
    }

    def has_prefix(prefix: str) -> bool:
        return any(path.startswith(prefix) for path in changed)

    def has_file(path: str) -> bool:
        return path in changed

    if has_file("webapp/parser/utils/database_comparison.py") or has_file("webapp/parser/html_election_parser.py"):
        pytest_targets.add("webapp/tests/test_database_comparison_centralized.py")
        ruff_targets.update(
            {
                "webapp/parser/utils/database_comparison.py",
                "webapp/parser/html_election_parser.py",
            }
        )

    if has_file("webapp/parser/utils/output_utils.py"):
        pytest_targets.add("webapp/tests/test_output_utils_database_crosscheck.py")
        ruff_targets.add("webapp/parser/utils/output_utils.py")

    if has_file("webapp/parser/health/fine_tune_bert_ner.py"):
        pytest_targets.add("webapp/tests/test_fine_tune_bert_ner.py")
        ruff_targets.add("webapp/parser/health/fine_tune_bert_ner.py")

    if has_prefix("webapp/parser/handlers/states/new_york/county/"):
        pytest_targets.add("webapp/tests/test_route_handler_contracts.py")

    if strict:
        pytest_targets.update(
            {
                "webapp/tests/test_phase_a_integration.py",
                "webapp/tests/test_output_utils_database_crosscheck.py",
            }
        )

    # Ensure only existing files are included.
    pytest_targets = {p for p in pytest_targets if (REPO_ROOT / p).exists()}
    ruff_targets = {p for p in ruff_targets if (REPO_ROOT / p).exists()}

    return sorted(pytest_targets), sorted(ruff_targets)


def _apply_change_scope(changed: list[str], scope: str) -> list[str]:
    """Filter changed files by scope before target selection."""
    if scope == "all":
        return changed

    # parser-only: ignore docs/UI/ops churn and keep parser-adjacent signals only.
    allowed_prefixes = (
        "webapp/parser/",
        "webapp/tests/",
        "tools/smart_phase_b_validation.py",
        "automate.py",
    )
    return [path for path in changed if path.startswith(allowed_prefixes)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run smart, dynamic Phase B validation.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Include broader parser-adjacent tests in addition to smart defaults.",
    )
    parser.add_argument(
        "--changed",
        nargs="*",
        default=None,
        help="Optional explicit changed-file list (workspace-relative paths).",
    )
    parser.add_argument(
        "--changed-scope",
        choices=["all", "parser-only"],
        default="parser-only",
        help="Scope used to filter changed files before selecting targets.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved targets only; do not execute pytest/ruff.",
    )
    parser.add_argument(
        "--summary-json",
        default="",
        help="Optional path to write a JSON run summary.",
    )
    args = parser.parse_args()

    if args.changed is not None and len(args.changed) > 0:
        changed = [p.replace("\\", "/") for p in args.changed]
    else:
        changed = _git_changed_files()

    scoped_changed = _apply_change_scope(changed, args.changed_scope)
    filtered_out = len(changed) - len(scoped_changed)

    print("[INFO] Smart Phase B Validation")
    print(f"[INFO] Changed files detected: {len(changed)}")
    for path in changed:
        print(f"  - {path}")
    print(f"[INFO] Changed scope: {args.changed_scope} ({len(scoped_changed)} retained, {filtered_out} filtered out)")

    pytest_targets, ruff_targets = _collect_targets(scoped_changed, strict=args.strict)

    print(f"[INFO] Pytest targets ({len(pytest_targets)}):")
    for path in pytest_targets:
        print(f"  - {path}")
    print(f"[INFO] Ruff targets ({len(ruff_targets)}):")
    for path in ruff_targets:
        print(f"  - {path}")

    summary = {
        "changed_count": len(changed),
        "changed": changed,
        "changed_scope": args.changed_scope,
        "scoped_changed_count": len(scoped_changed),
        "scoped_changed": scoped_changed,
        "filtered_out_count": filtered_out,
        "pytest_targets": pytest_targets,
        "ruff_targets": ruff_targets,
        "strict": bool(args.strict),
        "dry_run": bool(args.dry_run),
        "pytest_ok": None,
        "ruff_ok": None,
        "status": "pending",
    }

    if args.dry_run:
        print("[INFO] Dry-run requested; skipping execution.")
        summary["status"] = "dry-run"
        if args.summary_json:
            summary_path = Path(args.summary_json)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"[INFO] Summary written: {summary_path}")
        return 0

    if not pytest_targets:
        print("[WARN] No pytest targets resolved.")
        summary["status"] = "no-targets"
        if args.summary_json:
            summary_path = Path(args.summary_json)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"[INFO] Summary written: {summary_path}")
        return 1

    rc = _run([sys.executable, "-m", "pytest", "-q", *pytest_targets])
    if rc != 0:
        print("[FAIL] Pytest step failed.")
        summary["pytest_ok"] = False
        summary["status"] = "pytest-failed"
        if args.summary_json:
            summary_path = Path(args.summary_json)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"[INFO] Summary written: {summary_path}")
        return rc
    summary["pytest_ok"] = True

    if ruff_targets:
        rc = _run([sys.executable, "-m", "ruff", "check", *ruff_targets])
        if rc != 0:
            print("[FAIL] Ruff step failed.")
            summary["ruff_ok"] = False
            summary["status"] = "ruff-failed"
            if args.summary_json:
                summary_path = Path(args.summary_json)
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
                print(f"[INFO] Summary written: {summary_path}")
            return rc
        summary["ruff_ok"] = True
    else:
        summary["ruff_ok"] = True

    print("\n[PASS] Smart Phase B validation completed successfully.")
    summary["status"] = "passed"
    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[INFO] Summary written: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
