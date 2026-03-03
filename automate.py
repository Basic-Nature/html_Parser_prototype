#!/usr/bin/env python3
"""
Central automation script for Smart Elections Parser.

Runs all automated tasks:
- Generates comprehensive pipeline audit map
- Runs health bots and integrity checks
- Performs web asset linting and type checking
- Executes automated tests
- Validates webapp startup

Usage: python automate.py [--skip-web] [--skip-health] [--skip-tests]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))

# Detect localhost/development environment
POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
IS_LOCALHOST = POSTGRES_HOST in ("localhost", "127.0.0.1")
FLASK_ENV = os.environ.get("FLASK_ENV", "")
IS_DEVELOPMENT = FLASK_ENV.lower() in ("development", "dev")

# Silence warnings on localhost/development environments
if IS_LOCALHOST or IS_DEVELOPMENT:
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    # Suppress specific noisy warnings in development
    warnings.filterwarnings("ignore", message=".*eventlet.*")
    warnings.filterwarnings("ignore", message=".*socketio.*")
    os.environ.setdefault("PYTHONWARNINGS", "ignore::DeprecationWarning,ignore::FutureWarning")

from webapp.parser.health.health_router import BotPipeline
from webapp.parser.utils.logger_singleton import logger
from webapp.parser.utils.shared_logic import generate_docs_artifacts


REPORT_DIR = project_root / "output" / "reports"
REPORT_LOG_DIR = REPORT_DIR / "logs"
RUN_MANIFEST_PATH = REPORT_DIR / "automation_run_latest.json"
_STAGE_DETAILS: dict[str, dict[str, Any]] = {}


def _ensure_report_dirs() -> None:
    REPORT_LOG_DIR.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_existing_manifest(path: Path) -> dict[str, Any] | None:
    try:
        if not path.exists():
            return None
        raw = path.read_text(encoding="utf-8")
        loaded = json.loads(raw)
        return loaded if isinstance(loaded, dict) else None
    except Exception:
        return None


def _resolve_run_lineage(previous_manifest: dict[str, Any] | None) -> tuple[str, str | None]:
    explicit_run_id = str(os.environ.get("AUTOMATE_RUN_ID", "")).strip()
    explicit_parent_id = str(os.environ.get("AUTOMATE_PARENT_RUN_ID", "")).strip()

    if explicit_run_id:
        run_id = explicit_run_id
    else:
        gh_run_id = str(os.environ.get("GITHUB_RUN_ID", "")).strip()
        gh_attempt = str(os.environ.get("GITHUB_RUN_ATTEMPT", "")).strip() or "1"
        if gh_run_id:
            run_id = f"gh-{gh_run_id}-a{gh_attempt}"
        else:
            run_id = f"local-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-p{os.getpid()}"

    if explicit_parent_id:
        parent_run_id: str | None = explicit_parent_id
    else:
        parent = previous_manifest.get("run_id") if isinstance(previous_manifest, dict) else None
        parent_run_id = str(parent).strip() if parent else None

    if parent_run_id == run_id:
        parent_run_id = None
    return run_id, parent_run_id


def _compute_health_score(
    *,
    results: dict[str, Any],
    critical_failures: list[str],
    log_cleanup: dict[str, Any],
    report_retention: dict[str, Any],
) -> dict[str, Any]:
    critical_set = set(critical_failures)
    noncritical_failures = [
        stage for stage, value in results.items() if value is False and stage not in critical_set
    ]

    score = 100
    if results.get("embedding_cache_preflight") is False:
        score -= 30
    if results.get("pipeline_audit") is False:
        score -= 30
    score -= min(40, len(critical_set) * 15)
    score -= min(20, len(noncritical_failures) * 5)
    if isinstance(log_cleanup, dict) and log_cleanup.get("error"):
        score -= 5
    if isinstance(report_retention, dict) and report_retention.get("error"):
        score -= 5
    score = max(0, min(100, score))

    if score >= 90:
        state = "green"
    elif score >= 70:
        state = "yellow"
    else:
        state = "red"

    return {
        "score": score,
        "state": state,
        "critical_failures": sorted(critical_set),
        "noncritical_failures": sorted(noncritical_failures),
        "preflight_ok": results.get("embedding_cache_preflight") is True,
        "pipeline_audit_ok": results.get("pipeline_audit") is True,
    }


def _record_stage_detail(stage: str, detail: dict[str, Any]) -> None:
    _STAGE_DETAILS[stage] = detail


def _extract_context_lines(text: str, *, max_lines: int) -> list[str]:
    if not text:
        return []
    lines = [line for line in text.splitlines() if line.strip()]
    return lines[-max_lines:]


def _cleanup_ingested_stage_logs() -> dict[str, Any]:
    keep_all_logs = os.environ.get("AUTOMATE_KEEP_ALL_STAGE_LOGS", "").lower() in {
        "1",
        "true",
        "yes",
    }
    summary: dict[str, Any] = {
        "enabled": not keep_all_logs,
        "removed": [],
        "kept": [],
    }
    if keep_all_logs:
        return summary

    for stage_name, detail in _STAGE_DETAILS.items():
        stage_status = detail.get("status")
        for stream in ("stdout", "stderr"):
            path_key = f"{stream}_path"
            persisted_key = f"{stream}_persisted"
            path_value = detail.get(path_key)
            if not path_value:
                continue
            log_path = Path(path_value)
            if not log_path.exists():
                detail[persisted_key] = False
                continue
            should_keep = stage_status in {"failed", "timeout", "error"}
            if should_keep:
                detail[persisted_key] = True
                summary["kept"].append(str(log_path))
                continue
            try:
                log_path.unlink()
                detail[persisted_key] = False
                summary["removed"].append(str(log_path))
            except OSError:
                detail[persisted_key] = True
                summary["kept"].append(str(log_path))

        _STAGE_DETAILS[stage_name] = detail

    try:
        if REPORT_LOG_DIR.exists() and not any(REPORT_LOG_DIR.iterdir()):
            REPORT_LOG_DIR.rmdir()
    except OSError:
        pass

    summary["removed_count"] = len(summary["removed"])
    summary["kept_count"] = len(summary["kept"])
    return summary


def _run_report_retention(
    *,
    enabled: bool,
    pattern: str,
    max_age_days: int,
    max_files: int,
    max_total_bytes: int,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "enabled": enabled,
        "pattern": pattern,
        "max_age_days": max_age_days,
        "max_files": max_files,
        "max_total_bytes": max_total_bytes,
        "removed": [],
        "removed_count": 0,
        "freed_bytes": 0,
    }
    if not enabled:
        return summary

    protected_names = {RUN_MANIFEST_PATH.name}
    now_ts = time.time()
    cutoff_ts = now_ts - (max_age_days * 86400) if max_age_days >= 0 else None

    try:
        candidates = [
            p
            for p in REPORT_DIR.glob(pattern)
            if p.is_file() and p.name not in protected_names and not p.name.endswith("_latest.json")
        ]
    except OSError as exc:
        summary["error"] = str(exc)
        return summary

    def _file_mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    def _file_size(path: Path) -> int:
        try:
            return int(path.stat().st_size)
        except OSError:
            return 0

    removed_paths: set[str] = set()

    def _remove_file(path: Path, reason: str) -> None:
        if str(path) in removed_paths:
            return
        size_bytes = _file_size(path)
        try:
            path.unlink()
            removed_paths.add(str(path))
            summary["removed"].append(
                {
                    "path": str(path),
                    "reason": reason,
                    "size_bytes": size_bytes,
                }
            )
            summary["freed_bytes"] += size_bytes
        except OSError:
            pass

    ordered = sorted(candidates, key=_file_mtime)

    if cutoff_ts is not None:
        for path in ordered:
            if _file_mtime(path) < cutoff_ts:
                _remove_file(path, "age")

    remaining = [p for p in sorted(candidates, key=_file_mtime) if p.exists()]

    if max_files >= 0 and len(remaining) > max_files:
        overflow = len(remaining) - max_files
        for path in remaining[:overflow]:
            _remove_file(path, "max_files")

    remaining = [p for p in sorted(candidates, key=_file_mtime) if p.exists()]

    if max_total_bytes >= 0:
        current_size = sum(_file_size(p) for p in remaining)
        for path in remaining:
            if current_size <= max_total_bytes:
                break
            size_bytes = _file_size(path)
            _remove_file(path, "max_total_bytes")
            current_size -= size_bytes

    summary["removed_count"] = len(summary["removed"])
    return summary


def _run_subprocess_stage(
    stage: str,
    command: list[str],
    *,
    timeout: int,
    description: str,
) -> bool:
    _ensure_report_dirs()
    started = time.time()
    stdout_path = REPORT_LOG_DIR / f"{stage}.stdout.log"
    stderr_path = REPORT_LOG_DIR / f"{stage}.stderr.log"
    detail: dict[str, Any] = {
        "description": description,
        "command": command,
        "cwd": str(project_root),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }
    try:
        result = subprocess.run(
            command,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stdout_path.write_text(result.stdout or "", encoding="utf-8", errors="replace")
        stderr_path.write_text(result.stderr or "", encoding="utf-8", errors="replace")
        context_line_limit = 24 if result.returncode == 0 else 80
        detail.update(
            {
                "exit_code": result.returncode,
                "duration_ms": int((time.time() - started) * 1000),
                "status": "passed" if result.returncode == 0 else "failed",
                "stdout_tail": _extract_context_lines(result.stdout or "", max_lines=context_line_limit),
                "stderr_tail": _extract_context_lines(result.stderr or "", max_lines=context_line_limit),
            }
        )
        _record_stage_detail(stage, detail)
        if result.returncode == 0:
            logger.info(f"[AUTOMATE] {description} passed.")
            return True
        logger.error(f"[AUTOMATE] {description} failed with code {result.returncode}")
        print(result.stdout)
        print(result.stderr)
        return False
    except subprocess.TimeoutExpired:
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("Command timed out", encoding="utf-8")
        detail.update(
            {
                "exit_code": None,
                "duration_ms": int((time.time() - started) * 1000),
                "status": "timeout",
                "stdout_tail": [],
                "stderr_tail": ["Command timed out"],
            }
        )
        _record_stage_detail(stage, detail)
        logger.error(f"[AUTOMATE] {description} timed out.")
        return False
    except Exception as e:
        stderr_path.write_text(str(e), encoding="utf-8", errors="replace")
        detail.update(
            {
                "exit_code": None,
                "duration_ms": int((time.time() - started) * 1000),
                "status": "error",
                "error": str(e),
                "stdout_tail": [],
                "stderr_tail": _extract_context_lines(str(e), max_lines=20),
            }
        )
        _record_stage_detail(stage, detail)
        logger.error(f"[AUTOMATE] {description} failed: {e}")
        return False


def run_todo_index() -> bool:
    """Generate TODO indices (todos + high/medium/low)."""
    print("[AUTOMATE] Generating TODO indices...")
    logger.info("[AUTOMATE] Generating TODO indices...")
    try:
        result = subprocess.run(
            [
                sys.executable,
                "scripts/generate_todo_index.py",
                "--root",
                "webapp",
                "--root",
                "scripts",
                "--root",
                "docs",
            ],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            print("[AUTOMATE] TODO indices generated successfully.")
            logger.info("[AUTOMATE] TODO indices generated successfully.")
            logger.debug(f"[AUTOMATE] TODO index output: {result.stdout}")
            return True
        print(f"[AUTOMATE] TODO index failed with code {result.returncode}")
        logger.error(f"[AUTOMATE] TODO index failed with code {result.returncode}")
        logger.error(f"[AUTOMATE] STDERR: {result.stderr}")
        return False
    except subprocess.TimeoutExpired:
        print("[AUTOMATE] TODO index timed out.")
        logger.error("[AUTOMATE] TODO index timed out.")
        return False
    except Exception as e:
        print(f"[AUTOMATE] TODO index failed: {e}")
        logger.error(f"[AUTOMATE] TODO index failed: {e}")
        return False


def run_pipeline_audit():
    """Generate documentation artifacts (project audit + pipeline map + TODOs)."""
    print("[AUTOMATE] Generating documentation artifacts...")
    logger.info("[AUTOMATE] Generating documentation artifacts...")
    docs_ok = generate_docs_artifacts(project_root=str(project_root))
    todos_ok = run_todo_index()
    success = docs_ok and todos_ok
    if success:
        print("[AUTOMATE] Documentation artifacts generated successfully.")
        logger.info("[AUTOMATE] Documentation artifacts generated successfully.")
    else:
        print("[AUTOMATE] Failed to generate documentation artifacts.")
        logger.error("[AUTOMATE] Failed to generate documentation artifacts.")
    return success


def run_health_bots():
    """Run all health bots and integrity checks."""
    print("[AUTOMATE] Running health bots and integrity checks...")
    logger.info("[AUTOMATE] Running health bots and integrity checks...")
    try:
        pipeline = BotPipeline()
        pipeline.run()
        print("[AUTOMATE] Health bots completed successfully.")
        logger.info("[AUTOMATE] Health bots completed successfully.")
        return True
    except Exception as e:
        print(f"[AUTOMATE] Health bots failed: {e}")
        logger.error(f"[AUTOMATE] Health bots failed: {e}")
        return False


def run_web_checks():
    """Run linting and type checking for web assets (JS, CSS, HTML)."""
    print("[AUTOMATE] Running web asset checks (linting, type checking)...")
    logger.info("[AUTOMATE] Running web asset checks (linting, type checking)...")
    try:
        npm_cmd = shutil.which("npm.cmd") or shutil.which("npm")
        if not npm_cmd:
            raise FileNotFoundError("npm not found on PATH")
        return _run_subprocess_stage(
            "web_checks",
            [npm_cmd, "run", "verify:all"],
            timeout=300,
            description="Web checks (npm run verify:all)",
        )
    except subprocess.TimeoutExpired:
        print("[AUTOMATE] Web checks timed out.")
        logger.error("[AUTOMATE] Web checks timed out.")
        return False
    except FileNotFoundError:
        print("[AUTOMATE] npm not found. Install Node.js and npm to run web checks.")
        logger.error("[AUTOMATE] npm not found. Install Node.js and npm to run web checks.")
        return False
    except Exception as e:
        print(f"[AUTOMATE] Web checks failed: {e}")
        logger.error(f"[AUTOMATE] Web checks failed: {e}")
        return False


def run_automated_tests():
    """Run automated tests."""
    logger.info("[AUTOMATE] Running automated tests...")
    try:
        statement_test = project_root / "run_statement_test.py"
        if statement_test.exists():
            command = [sys.executable, "run_statement_test.py"]
            description = "Automated tests (run_statement_test.py)"
        else:
            command = [sys.executable, "-m", "pytest", "webapp/tests", "-q"]
            description = "Automated tests (pytest webapp/tests -q)"
        return _run_subprocess_stage("tests", command, timeout=300, description=description)
    except Exception as e:
        logger.error(f"[AUTOMATE] Tests failed: {e}")
        _record_stage_detail("tests", {"status": "error", "error": str(e)})
        return False


def validate_webapp_startup():
    """Quick validation that the webapp can start (doesn't run full server)."""
    logger.info("[AUTOMATE] Validating webapp startup...")
    try:
        import importlib

        importlib.invalidate_caches()
        importlib.import_module("webapp.Smart_Elections_Parser_Webapp")
        _record_stage_detail(
            "webapp_validation",
            {
                "status": "passed",
                "description": "Import webapp.Smart_Elections_Parser_Webapp",
                "started_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        logger.info("[AUTOMATE] Webapp import successful.")
        return True
    except Exception as e:
        _record_stage_detail(
            "webapp_validation",
            {
                "status": "failed",
                "description": "Import webapp.Smart_Elections_Parser_Webapp",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
            },
        )
        logger.error(f"[AUTOMATE] Webapp validation failed: {e}")
        return False


def run_self_check():
    """Run the consolidated UI robust check script and return True on success."""
    logger.info("[AUTOMATE] Running UI robust check (tools/ui_robust_check.py)...")
    return _run_subprocess_stage(
        "self_check",
        [sys.executable, "tools/ui_robust_check.py"],
        timeout=180,
        description="UI robust check",
    )


def run_ballot_lens_check():
    """Run the Ballot Lens UI verification script (tools/ui_robust_check.py)."""
    logger.info("[AUTOMATE] Running UI verification (tools/ui_robust_check.py)...")
    return _run_subprocess_stage(
        "ballot_lens_check",
        [sys.executable, "tools/ui_robust_check.py", "--viewport", "desktop"],
        timeout=180,
        description="Ballot Lens UI verification",
    )


def run_pipeline_check() -> bool:
    """Run the pipeline regression checker script."""
    logger.info("[AUTOMATE] Running pipeline regression check (scripts/pipeline_regression_check.py)...")
    return _run_subprocess_stage(
        "pipeline_check",
        [sys.executable, "scripts/pipeline_regression_check.py"],
        timeout=300,
        description="Pipeline regression check",
    )


def run_embedding_cache_preflight() -> bool:
    """Capture embedding cache health/status for every automation run manifest."""
    started = time.time()
    detail: dict[str, Any] = {
        "description": "Embedding cache lifecycle preflight",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    def _json_safe(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(k): _json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_json_safe(v) for v in value]
        return str(value)

    try:
        from webapp.parser.utils import embedding_cache

        cache_status = _json_safe(embedding_cache.get_embedding_cache_status())
        detail.update(
            {
                "status": "passed",
                "duration_ms": int((time.time() - started) * 1000),
                "cache_status": cache_status,
            }
        )
        _record_stage_detail("embedding_cache_preflight", detail)
        logger.info("[AUTOMATE] Embedding cache preflight captured.")
        return True
    except Exception as e:
        detail.update(
            {
                "status": "failed",
                "duration_ms": int((time.time() - started) * 1000),
                "error": str(e),
            }
        )
        _record_stage_detail("embedding_cache_preflight", detail)
        logger.error(f"[AUTOMATE] Embedding cache preflight failed: {e}")
        return False


def run_dl_compare_check(
    *,
    dl1_path: str,
    dl2_path: str,
    min_accuracy: float,
    max_mismatches: int,
    soft: bool,
) -> bool:
    """Run DL1 vs DL2 regression comparison report generation."""
    out_path = REPORT_DIR / "data_comparison_latest.json"
    command = [
        sys.executable,
        "scripts/data_comparison_report.py",
        "--dl1",
        dl1_path,
        "--dl2",
        dl2_path,
        "--out",
        str(out_path),
        "--min-accuracy",
        str(min_accuracy),
        "--max-mismatches",
        str(max_mismatches),
    ]
    if soft:
        command.append("--soft")
    ok = _run_subprocess_stage(
        "dl_compare",
        command,
        timeout=300,
        description="DL1 vs DL2 comparison report",
    )
    detail = _STAGE_DETAILS.get("dl_compare", {})
    detail["report_path"] = str(out_path)
    _record_stage_detail("dl_compare", detail)
    return ok


def main():
    parser = argparse.ArgumentParser(description="Run all automated scripts for Smart Elections Parser.")
    parser.add_argument("--skip-web", action="store_true", help="Skip web asset checks")
    parser.add_argument("--skip-health", action="store_true", help="Skip health bots")
    parser.add_argument("--skip-tests", action="store_true", help="Skip automated tests")
    parser.add_argument("--skip-webapp-check", action="store_true", help="Skip webapp startup validation")
    parser.add_argument("--self-check", action="store_true", help="Run UI robust check (tools/ui_robust_check.py) after other checks")
    parser.add_argument("--ballot-lens-check", action="store_true", help="Run UI verification (tools/ui_robust_check.py)")
    parser.add_argument("--pipeline-check", action="store_true", help="Run pipeline regression checker (scripts/pipeline_regression_check.py)")
    parser.add_argument("--compare-dl1-dl2", action="store_true", help="Run DL1 vs DL2 comparison report stage")
    parser.add_argument("--dl1-path", default="", help="Path to DL1 ground truth JSON for comparison stage")
    parser.add_argument("--dl2-path", default="", help="Path to DL2 parser output JSON for comparison stage")
    parser.add_argument("--compare-min-accuracy", type=float, default=0.95, help="Minimum accuracy threshold for comparison gate")
    parser.add_argument("--compare-max-mismatches", type=int, default=0, help="Maximum mismatch threshold for comparison gate")
    parser.add_argument("--compare-soft", action="store_true", help="Do not fail comparison stage on gate failure")
    parser.add_argument("--compare-strict", action="store_true", help="Treat comparison stage as critical when requested")
    parser.add_argument("--enforce-report-retention", action="store_true", help="Enable retention and size guards for report_* artifacts")
    parser.add_argument("--report-retention-pattern", default="report_*.json", help="Glob pattern under output/reports for retention candidate artifacts")
    parser.add_argument("--report-retention-days", type=int, default=30, help="Delete candidate report artifacts older than this many days (-1 disables age-based deletion)")
    parser.add_argument("--report-max-files", type=int, default=200, help="Maximum number of candidate report artifacts to keep (-1 disables file-count cap)")
    parser.add_argument("--report-max-bytes", type=int, default=268435456, help="Maximum combined bytes for candidate report artifacts (-1 disables size cap)")
    parser.add_argument("--strict-embedding-preflight", action="store_true", help="Treat embedding cache preflight failure as a critical failure")
    parser.add_argument("--simulate-unhandled-failure", action="store_true", help="Debug-only: raise a synthetic unhandled exception after preflight to validate failed-manifest handling")

    args = parser.parse_args()

    _ensure_report_dirs()
    run_started = datetime.now(timezone.utc)
    previous_manifest = _read_existing_manifest(RUN_MANIFEST_PATH)
    run_id, parent_run_id = _resolve_run_lineage(previous_manifest)
    ci_strict = os.environ.get("CI", "").lower() in {"1", "true", "yes"}
    env_embedding_strict = os.environ.get("EMBEDDING_PREFLIGHT_STRICT", "").lower() in {"1", "true", "yes"}
    strict_embedding_preflight = bool(args.strict_embedding_preflight or env_embedding_strict)

    bootstrap_manifest = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "started_at": run_started.isoformat(),
        "cwd": str(project_root),
        "run_id": run_id,
        "parent_run_id": parent_run_id,
        "status": "running",
        "results": {},
        "stage_details": {},
        "critical_failures": [],
        "strict_compare_mode": bool(args.compare_strict or ci_strict),
        "strict_embedding_preflight_mode": strict_embedding_preflight,
    }
    _write_json(RUN_MANIFEST_PATH, bootstrap_manifest)

    print("[AUTOMATE] Starting comprehensive automation run...")
    logger.info("[AUTOMATE] Starting comprehensive automation run...")

    results = {}
    critical_failures: list[str] = []

    try:
        # Always capture embedding cache preflight status
        results["embedding_cache_preflight"] = run_embedding_cache_preflight()

        if args.simulate_unhandled_failure:
            allow_sim = os.environ.get("AUTOMATE_ALLOW_SIMULATED_FAILURE", "").lower() in {"1", "true", "yes"}
            if not allow_sim:
                raise RuntimeError(
                    "--simulate-unhandled-failure requires AUTOMATE_ALLOW_SIMULATED_FAILURE=true"
                )
            raise RuntimeError("Synthetic unhandled failure (debug flag)")

        # Always run pipeline audit
        results["pipeline_audit"] = run_pipeline_audit()

        # Run health bots unless skipped
        if not args.skip_health:
            results["health_bots"] = run_health_bots()
        else:
            print("[AUTOMATE] Skipping health bots.")
            logger.info("[AUTOMATE] Skipping health bots.")
            results["health_bots"] = None

        # Run web checks unless skipped
        if not args.skip_web:
            results["web_checks"] = run_web_checks()
        else:
            print("[AUTOMATE] Skipping web checks.")
            logger.info("[AUTOMATE] Skipping web checks.")
            results["web_checks"] = None

        # Run tests unless skipped
        if not args.skip_tests:
            results["tests"] = run_automated_tests()
        else:
            print("[AUTOMATE] Skipping automated tests.")
            logger.info("[AUTOMATE] Skipping automated tests.")
            results["tests"] = None

        # Optional headless self-check
        if args.self_check:
            results['self_check'] = run_self_check()
        else:
            results['self_check'] = None

        # Optional Ballot Lens check
        if args.ballot_lens_check:
            results['ballot_lens_check'] = run_ballot_lens_check()
        else:
            results['ballot_lens_check'] = None

        # Optional pipeline regression check
        if args.pipeline_check:
            results["pipeline_check"] = run_pipeline_check()
        else:
            results["pipeline_check"] = None

        # Optional DL1 vs DL2 comparison stage
        if args.compare_dl1_dl2:
            if not args.dl1_path or not args.dl2_path:
                logger.error("[AUTOMATE] --compare-dl1-dl2 requires both --dl1-path and --dl2-path")
                _record_stage_detail(
                    "dl_compare",
                    {
                        "status": "failed",
                        "error": "Missing --dl1-path or --dl2-path",
                        "started_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
                results["dl_compare"] = False
            else:
                results["dl_compare"] = run_dl_compare_check(
                    dl1_path=args.dl1_path,
                    dl2_path=args.dl2_path,
                    min_accuracy=args.compare_min_accuracy,
                    max_mismatches=args.compare_max_mismatches,
                    soft=args.compare_soft,
                )
        else:
            results["dl_compare"] = None

        # Validate webapp unless skipped
        if not args.skip_webapp_check:
            results["webapp_validation"] = validate_webapp_startup()
        else:
            print("[AUTOMATE] Skipping webapp validation.")
            logger.info("[AUTOMATE] Skipping webapp validation.")
            results["webapp_validation"] = None

        # Summary
        print("[AUTOMATE] Automation run complete. Summary:")
        logger.info("[AUTOMATE] Automation run complete. Summary:")
        for task, success in results.items():
            status = "PASSED" if success else ("SKIPPED" if success is None else "FAILED")
            print(f"  {task:<20}: {status}")
            logger.info(f"  {task:<20}: {status}")

        # Exit with failure if any critical task failed
        critical_failures = [k for k, v in results.items() if v is False and k in ["pipeline_audit", "web_checks"]]
        # Optional self-check failure handling: if --self-check was requested, treat it as critical
        if args.self_check:
            sc = results.get('self_check')
            if sc is False:
                critical_failures.append('self_check')
        # Treat Ballot Lens check as critical only when requested
        if args.ballot_lens_check and results.get('ballot_lens_check') is False:
            critical_failures.append('ballot_lens_check')
        # Treat pipeline check as critical only when requested
        if args.pipeline_check and results.get('pipeline_check') is False:
            critical_failures.append('pipeline_check')
        # Hybrid policy for DL compare: strict in CI or when explicitly requested
        if args.compare_dl1_dl2 and results.get("dl_compare") is False and (args.compare_strict or ci_strict):
            critical_failures.append("dl_compare")
        # Optional strict policy for embedding preflight
        if results.get("embedding_cache_preflight") is False and strict_embedding_preflight:
            critical_failures.append("embedding_cache_preflight")

        log_cleanup = _cleanup_ingested_stage_logs()
        retention_enabled = bool(args.enforce_report_retention or ci_strict)
        report_retention = _run_report_retention(
            enabled=retention_enabled,
            pattern=args.report_retention_pattern,
            max_age_days=args.report_retention_days,
            max_files=args.report_max_files,
            max_total_bytes=args.report_max_bytes,
        )

        run_manifest = {
            "schema_version": "1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "started_at": run_started.isoformat(),
            "cwd": str(project_root),
            "run_id": run_id,
            "parent_run_id": parent_run_id,
            "status": "completed",
            "results": results,
            "stage_details": _STAGE_DETAILS,
            "log_cleanup": log_cleanup,
            "report_retention": report_retention,
            "critical_failures": critical_failures,
            "strict_compare_mode": bool(args.compare_strict or ci_strict),
            "strict_embedding_preflight_mode": strict_embedding_preflight,
            "health_score": _compute_health_score(
                results=results,
                critical_failures=critical_failures,
                log_cleanup=log_cleanup,
                report_retention=report_retention,
            ),
        }
        _write_json(RUN_MANIFEST_PATH, run_manifest)
        print(f"[AUTOMATE] Run manifest written: {RUN_MANIFEST_PATH}")
        logger.info(f"[AUTOMATE] Run manifest written: {RUN_MANIFEST_PATH}")
        if critical_failures:
            print(f"[AUTOMATE] Critical failures in: {', '.join(critical_failures)}")
            logger.error(f"[AUTOMATE] Critical failures in: {', '.join(critical_failures)}")
            sys.exit(1)
        else:
            print("[AUTOMATE] All critical tasks passed!")
            logger.info("[AUTOMATE] All critical tasks passed!")
    except Exception as exc:
        error_trace = traceback.format_exc()
        logger.error(f"[AUTOMATE] Unhandled exception: {exc}")
        logger.error(error_trace)

        try:
            log_cleanup = _cleanup_ingested_stage_logs()
        except Exception:
            log_cleanup = {
                "enabled": False,
                "error": "log_cleanup_failed_in_exception_handler",
            }

        try:
            retention_enabled = bool(args.enforce_report_retention or ci_strict)
            report_retention = _run_report_retention(
                enabled=retention_enabled,
                pattern=args.report_retention_pattern,
                max_age_days=args.report_retention_days,
                max_files=args.report_max_files,
                max_total_bytes=args.report_max_bytes,
            )
        except Exception:
            report_retention = {
                "enabled": False,
                "error": "report_retention_failed_in_exception_handler",
            }

        failure_manifest = {
            "schema_version": "1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "started_at": run_started.isoformat(),
            "cwd": str(project_root),
            "run_id": run_id,
            "parent_run_id": parent_run_id,
            "status": "failed",
            "results": results,
            "stage_details": _STAGE_DETAILS,
            "log_cleanup": log_cleanup,
            "report_retention": report_retention,
            "critical_failures": sorted(set(critical_failures + ["unhandled_exception"])),
            "strict_compare_mode": bool(args.compare_strict or ci_strict),
            "strict_embedding_preflight_mode": strict_embedding_preflight,
            "health_score": _compute_health_score(
                results=results,
                critical_failures=sorted(set(critical_failures + ["unhandled_exception"])),
                log_cleanup=log_cleanup,
                report_retention=report_retention,
            ),
            "unhandled_exception": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": error_trace,
            },
        }

        try:
            _write_json(RUN_MANIFEST_PATH, failure_manifest)
            print(f"[AUTOMATE] Failure manifest written: {RUN_MANIFEST_PATH}")
            logger.error(f"[AUTOMATE] Failure manifest written: {RUN_MANIFEST_PATH}")
        except Exception as manifest_exc:
            print(f"[AUTOMATE] Failed to write failure manifest: {manifest_exc}")
            logger.error(f"[AUTOMATE] Failed to write failure manifest: {manifest_exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()