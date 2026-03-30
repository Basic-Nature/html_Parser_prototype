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
import re
import shutil
import subprocess
import sys
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.automation_policy import (
    classify_intended_environment,
    cleanup_ingested_stage_logs,
    collect_critical_failures,
    compute_health_score,
    run_report_retention,
)
from scripts.automation_runtime import (
    build_bootstrap_manifest,
    build_completed_manifest,
    build_failure_manifest,
    execute_automation_stages,
)

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


def _record_stage_detail(stage: str, detail: dict[str, Any]) -> None:
    _STAGE_DETAILS[stage] = detail


def _extract_context_lines(text: str, *, max_lines: int) -> list[str]:
    if not text:
        return []
    lines = [line for line in text.splitlines() if line.strip()]
    return lines[-max_lines:]


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
    started = time.time()
    docs_ok = generate_docs_artifacts(project_root=str(project_root))
    todos_ok = run_todo_index()
    success = docs_ok and todos_ok
    _record_stage_detail(
        "pipeline_audit",
        {
            "description": "Generate docs artifacts and TODO index",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "duration_ms": int((time.time() - started) * 1000),
            "docs_artifacts_ok": bool(docs_ok),
            "todo_index_ok": bool(todos_ok),
            "status": "passed" if success else "failed",
        },
    )
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
    started = time.time()
    try:
        pipeline = BotPipeline()
        pipeline.run()
        _record_stage_detail(
            "health_bots",
            {
                "description": "Run BotPipeline health router orchestration",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "duration_ms": int((time.time() - started) * 1000),
                "status": "passed",
                "pipeline_last_run": _json_safe(getattr(pipeline, "last_run", None)),
                "pipeline_results": _json_safe(getattr(pipeline, "results", {})),
                "ai_suggestions_count": len(getattr(pipeline, "ai_suggestions", []) or []),
            },
        )
        print("[AUTOMATE] Health bots completed successfully.")
        logger.info("[AUTOMATE] Health bots completed successfully.")
        return True
    except Exception as e:
        _record_stage_detail(
            "health_bots",
            {
                "description": "Run BotPipeline health router orchestration",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "duration_ms": int((time.time() - started) * 1000),
                "status": "failed",
                "error": str(e),
            },
        )
        print(f"[AUTOMATE] Health bots failed: {e}")
        logger.error(f"[AUTOMATE] Health bots failed: {e}")
        return False


def _run_web_command(command: list[str], *, timeout: int) -> tuple[bool, dict[str, Any]]:
    started = time.time()
    try:
        result = subprocess.run(
            command,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0, {
            "command": command,
            "exit_code": result.returncode,
            "duration_ms": int((time.time() - started) * 1000),
            "stdout_tail": _extract_context_lines(result.stdout or "", max_lines=20),
            "stderr_tail": _extract_context_lines(result.stderr or "", max_lines=40),
        }
    except subprocess.TimeoutExpired:
        return False, {
            "command": command,
            "exit_code": None,
            "duration_ms": int((time.time() - started) * 1000),
            "timeout": True,
            "stdout_tail": [],
            "stderr_tail": ["Command timed out"],
        }
    except Exception as exc:
        return False, {
            "command": command,
            "exit_code": None,
            "duration_ms": int((time.time() - started) * 1000),
            "error": str(exc),
            "stdout_tail": [],
            "stderr_tail": _extract_context_lines(str(exc), max_lines=20),
        }


def run_web_checks(*, strict: bool = False) -> bool:
    """Run web checks with a resilient fallback path.

    Strict mode enforces `npm run verify:all` only. Non-strict mode falls back to
    a minimal but useful safety path when full verification isn't available.
    """
    print("[AUTOMATE] Running web asset checks (linting, type checking)...")
    logger.info("[AUTOMATE] Running web asset checks (linting, type checking)...")

    detail: dict[str, Any] = {
        "description": "Web checks with resilient fallback",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "strict": strict,
        "attempts": [],
    }

    npm_cmd = shutil.which("npm.cmd") or shutil.which("npm")
    node_cmd = shutil.which("node.exe") or shutil.which("node")

    if npm_cmd:
        primary_ok, primary_detail = _run_web_command(
            [npm_cmd, "run", "verify:all"],
            timeout=300,
        )
        primary_detail["label"] = "primary"
        detail["attempts"].append(primary_detail)
        if primary_ok:
            detail["status"] = "passed"
            detail["path"] = "npm verify:all"
            _record_stage_detail("web_checks", detail)
            logger.info("[AUTOMATE] Web checks passed via npm run verify:all.")
            return True

        if strict:
            detail["status"] = "failed"
            detail["path"] = "npm verify:all (strict)"
            _record_stage_detail("web_checks", detail)
            logger.error("[AUTOMATE] Web checks failed in strict mode (npm run verify:all).")
            return False

        logger.warning("[AUTOMATE] Full web verification failed; attempting safe fallback checks.")

        check_js_ok, check_js_detail = _run_web_command(
            [npm_cmd, "run", "check-js"],
            timeout=180,
        )
        check_js_detail["label"] = "fallback-check-js"
        detail["attempts"].append(check_js_detail)

        lint_web_ok, lint_web_detail = _run_web_command(
            [npm_cmd, "run", "lint:web"],
            timeout=180,
        )
        lint_web_detail["label"] = "fallback-lint-web"
        detail["attempts"].append(lint_web_detail)

        lint_md_ok, lint_md_detail = _run_web_command(
            [npm_cmd, "run", "lint:md"],
            timeout=180,
        )
        lint_md_detail["label"] = "fallback-lint-md"
        detail["attempts"].append(lint_md_detail)

        lint_python_ok, lint_python_detail = _run_web_command(
            [npm_cmd, "run", "lint:python"],
            timeout=300,
        )
        lint_python_detail["label"] = "fallback-lint-python"
        detail["attempts"].append(lint_python_detail)

        verify_python_ok, verify_python_detail = _run_web_command(
            [npm_cmd, "run", "verify:python"],
            timeout=300,
        )
        verify_python_detail["label"] = "fallback-verify-python"
        detail["attempts"].append(verify_python_detail)

        detail["fallback_results"] = {
            "check_js": check_js_ok,
            "lint_web": lint_web_ok,
            "lint_md": lint_md_ok,
            "lint_python": lint_python_ok,
            "verify_python": verify_python_ok,
        }

        if check_js_ok:
            detail["status"] = "passed"
            detail["path"] = "fallback"
            detail["degraded"] = not (lint_web_ok and lint_md_ok and lint_python_ok)
            detail["python_quality_ok"] = lint_python_ok
            detail["python_typecheck_ok"] = verify_python_ok
            _record_stage_detail("web_checks", detail)
            if lint_python_ok and verify_python_ok:
                logger.info("[AUTOMATE] Web checks passed via fallback path (check-js + Ruff + verify:python all connected).")
            elif lint_python_ok:
                logger.warning("[AUTOMATE] Web checks passed via fallback check-js + Ruff gates; verify:python (typecheck) reported issues.")
            else:
                logger.warning("[AUTOMATE] Web checks passed via fallback check-js gate; Ruff lint reported issues.")
            return True

        detail["status"] = "failed"
        detail["path"] = "fallback"
        _record_stage_detail("web_checks", detail)
        logger.error("[AUTOMATE] Web checks failed: fallback check-js gate did not pass.")
        return False

    if node_cmd and (project_root / "scripts" / "check_js_syntax.js").exists():
        logger.warning("[AUTOMATE] npm not found; running node + Ruff fallback checks.")
        node_ok, node_detail = _run_web_command(
            [node_cmd, "scripts/check_js_syntax.js"],
            timeout=180,
        )
        node_detail["label"] = "fallback-node-check-js"
        detail["attempts"].append(node_detail)

        ruff_ok, ruff_detail = _run_web_command(
            [sys.executable, "-m", "ruff", "check"],
            timeout=300,
        )
        ruff_detail["label"] = "fallback-python-ruff"
        detail["attempts"].append(ruff_detail)

        detail["path"] = "node fallback"
        detail["degraded"] = True
        detail["fallback_results"] = {
            "check_js": node_ok,
            "lint_python": ruff_ok,
        }
        detail["python_quality_ok"] = ruff_ok
        detail["status"] = "passed" if node_ok else "failed"
        detail["degraded"] = not ruff_ok
        _record_stage_detail("web_checks", detail)
        if node_ok and ruff_ok:
            logger.info("[AUTOMATE] Web checks passed via node fallback (Ruff connected).")
            return True
        if node_ok:
            logger.warning("[AUTOMATE] Web checks passed via node check-js fallback; Ruff reported issues.")
            return True
        logger.error("[AUTOMATE] Web checks failed via node fallback check-js gate.")
        return False

    detail["status"] = "failed"
    detail["path"] = "no-web-tooling"
    detail["error"] = "npm and node are unavailable"
    _record_stage_detail("web_checks", detail)
    logger.error("[AUTOMATE] npm and node not found; unable to run web checks or fallback.")
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


def run_smart_phase_b_check(*, strict: bool = False) -> bool:
    """Run focused parser-adjacent Phase B validation tooling."""
    logger.info("[AUTOMATE] Running smart Phase B validation (tools/smart_phase_b_validation.py)...")
    summary_path = REPORT_DIR / "smart_phase_b_summary.json"
    command = [
        sys.executable,
        "tools/smart_phase_b_validation.py",
        "--summary-json",
        str(summary_path),
    ]
    if strict:
        command.append("--strict")
    ok = _run_subprocess_stage(
        "smart_phase_b_check",
        command,
        timeout=600,
        description="Smart Phase B validation",
    )
    detail = _STAGE_DETAILS.get("smart_phase_b_check", {})
    detail["summary_path"] = str(summary_path)
    if summary_path.exists():
        try:
            loaded = json.loads(summary_path.read_text(encoding="utf-8"))
            detail["smart_phase_b_summary"] = _json_safe(loaded)
        except Exception as exc:
            detail["smart_phase_b_summary_error"] = str(exc)
    _record_stage_detail("smart_phase_b_check", detail)
    return ok


def run_smart_phase_b_check_with_scope(*, strict: bool = False, changed_scope: str = "parser-only") -> bool:
    """Run smart phase B check with a specific changed-file scope."""
    logger.info(
        f"[AUTOMATE] Running smart Phase B validation with scope='{changed_scope}' "
        f"(strict={bool(strict)})..."
    )
    summary_path = REPORT_DIR / "smart_phase_b_summary.json"
    command = [
        sys.executable,
        "tools/smart_phase_b_validation.py",
        "--changed-scope",
        changed_scope,
        "--summary-json",
        str(summary_path),
    ]
    if strict:
        command.append("--strict")

    ok = _run_subprocess_stage(
        "smart_phase_b_check",
        command,
        timeout=600,
        description="Smart Phase B validation",
    )
    detail = _STAGE_DETAILS.get("smart_phase_b_check", {})
    detail["summary_path"] = str(summary_path)
    detail["changed_scope"] = changed_scope
    if summary_path.exists():
        try:
            loaded = json.loads(summary_path.read_text(encoding="utf-8"))
            detail["smart_phase_b_summary"] = _json_safe(loaded)
        except Exception as exc:
            detail["smart_phase_b_summary_error"] = str(exc)
    _record_stage_detail("smart_phase_b_check", detail)
    return ok


def run_embedding_cache_preflight() -> bool:
    """Capture embedding cache health/status for every automation run manifest."""
    started = time.time()
    detail: dict[str, Any] = {
        "description": "Embedding cache lifecycle preflight",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

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


def run_docs_routing_gate() -> bool:
    """Validate docs navigation URLs resolve to source docs files."""
    started = time.time()
    nav_path = project_root / "docs" / "_data" / "navigation.yml"
    docs_root = project_root / "docs"
    detail: dict[str, Any] = {
        "description": "Docs routing gate (navigation URL target validation)",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "navigation_file": str(nav_path),
        "checked_targets": 0,
        "missing_targets": [],
    }

    def _url_to_candidates(url: str) -> list[Path]:
        normalized = (url or "").strip()
        if not normalized or normalized.startswith("http://") or normalized.startswith("https://"):
            return []
        if normalized.startswith("/html_Parser_prototype/"):
            rel = normalized[len("/html_Parser_prototype/") :]
        elif normalized.startswith("/"):
            rel = normalized[1:]
        else:
            rel = normalized

        rel = rel.split("?", 1)[0].split("#", 1)[0]
        if not rel:
            return [docs_root / "index.md"]

        if rel.endswith("/"):
            return [docs_root / rel / "index.md"]
        if rel.endswith(".html"):
            return [docs_root / f"{rel[:-5]}.md", docs_root / rel]
        if rel.endswith(".md"):
            return [docs_root / rel]
        return [docs_root / f"{rel}.md", docs_root / rel]

    try:
        if not nav_path.exists():
            detail.update(
                {
                    "status": "failed",
                    "duration_ms": int((time.time() - started) * 1000),
                    "error": "navigation.yml not found",
                }
            )
            _record_stage_detail("docs_routing_gate", detail)
            logger.error("[AUTOMATE] Docs routing gate failed: docs/_data/navigation.yml not found.")
            return False

        nav_text = nav_path.read_text(encoding="utf-8", errors="replace")
        urls = re.findall(r"^\s*url:\s*(\S+)\s*$", nav_text, flags=re.MULTILINE)
        missing: list[dict[str, Any]] = []
        checked = 0

        for url in urls:
            candidates = _url_to_candidates(url)
            if not candidates:
                continue
            checked += 1
            if not any(candidate.exists() for candidate in candidates):
                missing.append(
                    {
                        "url": url,
                        "candidates": [str(candidate) for candidate in candidates],
                    }
                )

        detail["checked_targets"] = checked
        detail["missing_count"] = len(missing)
        detail["missing_targets"] = missing[:25]
        detail["duration_ms"] = int((time.time() - started) * 1000)

        if missing:
            detail["status"] = "failed"
            _record_stage_detail("docs_routing_gate", detail)
            logger.error(f"[AUTOMATE] Docs routing gate failed: {len(missing)} unresolved navigation target(s).")
            return False

        detail["status"] = "passed"
        _record_stage_detail("docs_routing_gate", detail)
        logger.info(f"[AUTOMATE] Docs routing gate passed ({checked} target(s) checked).")
        return True
    except Exception as exc:
        detail.update(
            {
                "status": "failed",
                "duration_ms": int((time.time() - started) * 1000),
                "error": str(exc),
            }
        )
        _record_stage_detail("docs_routing_gate", detail)
        logger.error(f"[AUTOMATE] Docs routing gate failed: {exc}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run all automated scripts for Smart Elections Parser.")
    parser.add_argument(
        "--intended-env",
        choices=["auto", "local", "development", "ci", "staging", "production"],
        default="auto",
        help="Classify this automation run by intended environment for manifesting/policy decisions",
    )
    parser.add_argument("--skip-web", action="store_true", help="Skip web asset checks")
    parser.add_argument("--skip-health", action="store_true", help="Skip health bots")
    parser.add_argument("--skip-tests", action="store_true", help="Skip automated tests")
    parser.add_argument("--skip-webapp-check", action="store_true", help="Skip webapp startup validation")
    parser.add_argument("--strict-web-checks", action="store_true", help="Require npm run verify:all without fallback for web checks")
    parser.add_argument("--self-check", action="store_true", help="Run UI robust check (tools/ui_robust_check.py) after other checks")
    parser.add_argument("--ballot-lens-check", action="store_true", help="Run UI verification (tools/ui_robust_check.py)")
    parser.add_argument("--pipeline-check", action="store_true", help="Run pipeline regression checker (scripts/pipeline_regression_check.py)")
    parser.add_argument("--smart-phase-b-check", action="store_true", help="Run focused parser-adjacent smart validation stage")
    parser.add_argument("--smart-phase-b-strict", action="store_true", help="Use strict mode for smart phase B validation")
    parser.add_argument("--smart-phase-b-scope", choices=["all", "parser-only"], default="parser-only", help="Changed-file scope passed to smart phase B validation")
    parser.add_argument("--smart-phase-b-critical", action="store_true", help="Treat smart phase B stage as critical when enabled")
    parser.add_argument("--compare-dl1-dl2", action="store_true", help="Run DL1 vs DL2 comparison report stage")
    parser.add_argument("--dl1-path", default="", help="Path to DL1 ground truth JSON for comparison stage")
    parser.add_argument("--dl2-path", default="", help="Path to DL2 parser output JSON for comparison stage")
    parser.add_argument("--compare-min-accuracy", type=float, default=0.95, help="Minimum accuracy threshold for comparison gate")
    parser.add_argument("--compare-max-mismatches", type=int, default=0, help="Maximum mismatch threshold for comparison gate")
    parser.add_argument("--compare-soft", action="store_true", help="Do not fail comparison stage on gate failure")
    parser.add_argument("--compare-strict", action="store_true", help="Treat comparison stage as critical when requested")
    parser.add_argument("--docs-routing-gate", action="store_true", help="Run docs navigation routing gate as an extra plugin stage")
    parser.add_argument("--docs-routing-gate-strict", action="store_true", help="Treat docs routing gate failure as critical (requires --docs-routing-gate)")
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
    intended_environment = classify_intended_environment(
        args.intended_env,
        ci_strict=ci_strict,
        is_development=IS_DEVELOPMENT,
        is_localhost=IS_LOCALHOST,
    )
    strict_embedding_preflight = bool(args.strict_embedding_preflight or env_embedding_strict)
    strict_web_checks = bool(args.strict_web_checks or ci_strict)

    bootstrap_manifest = build_bootstrap_manifest(
        started_at=run_started.isoformat(),
        cwd=str(project_root),
        run_id=run_id,
        parent_run_id=parent_run_id,
        intended_environment=intended_environment,
        strict_compare_mode=bool(args.compare_strict or ci_strict),
        strict_embedding_preflight_mode=strict_embedding_preflight,
        strict_web_checks_mode=strict_web_checks,
    )
    _write_json(RUN_MANIFEST_PATH, bootstrap_manifest)

    print("[AUTOMATE] Starting comprehensive automation run...")
    logger.info("[AUTOMATE] Starting comprehensive automation run...")
    print(f"[AUTOMATE] Intended environment: {intended_environment}")
    logger.info(f"[AUTOMATE] Intended environment: {intended_environment}")

    results = {}
    critical_failures: list[str] = []

    try:
        extra_stage_runners: dict[str, Any] = {}
        if args.docs_routing_gate:
            extra_stage_runners["docs_routing_gate"] = run_docs_routing_gate
        if args.smart_phase_b_check:
            extra_stage_runners["smart_phase_b_check"] = lambda: run_smart_phase_b_check_with_scope(
                strict=bool(args.smart_phase_b_strict),
                changed_scope=str(args.smart_phase_b_scope or "parser-only"),
            )

        results = execute_automation_stages(
            args,
            strict_web_checks=strict_web_checks,
            logger=logger,
            record_stage_detail=_record_stage_detail,
            run_embedding_cache_preflight=run_embedding_cache_preflight,
            run_pipeline_audit=run_pipeline_audit,
            run_health_bots=run_health_bots,
            run_web_checks=run_web_checks,
            run_automated_tests=run_automated_tests,
            run_self_check=run_self_check,
            run_ballot_lens_check=run_ballot_lens_check,
            run_pipeline_check=run_pipeline_check,
            run_dl_compare_check=run_dl_compare_check,
            validate_webapp_startup=validate_webapp_startup,
            extra_stage_runners=extra_stage_runners or None,
        )

        # Summary
        print("[AUTOMATE] Automation run complete. Summary:")
        logger.info("[AUTOMATE] Automation run complete. Summary:")
        for task, success in results.items():
            status = "PASSED" if success else ("SKIPPED" if success is None else "FAILED")
            print(f"  {task:<20}: {status}")
            logger.info(f"  {task:<20}: {status}")

        # Exit with failure if any critical task failed
        always_critical_on_false: set[str] | None = None
        if args.docs_routing_gate and args.docs_routing_gate_strict:
            always_critical_on_false = {"docs_routing_gate"}
        if args.smart_phase_b_check and args.smart_phase_b_critical:
            if always_critical_on_false is None:
                always_critical_on_false = {"smart_phase_b_check"}
            else:
                always_critical_on_false.add("smart_phase_b_check")

        critical_failures = collect_critical_failures(
            results=results,
            args=args,
            ci_strict=ci_strict,
            strict_embedding_preflight=strict_embedding_preflight,
            always_critical_on_false=always_critical_on_false,
        )

        log_cleanup = cleanup_ingested_stage_logs(
            stage_details=_STAGE_DETAILS,
            report_log_dir=REPORT_LOG_DIR,
        )
        retention_enabled = bool(args.enforce_report_retention or ci_strict)
        report_retention = run_report_retention(
            enabled=retention_enabled,
            pattern=args.report_retention_pattern,
            max_age_days=args.report_retention_days,
            max_files=args.report_max_files,
            max_total_bytes=args.report_max_bytes,
            report_dir=REPORT_DIR,
            protected_names={RUN_MANIFEST_PATH.name},
        )

        run_manifest = build_completed_manifest(
            started_at=run_started.isoformat(),
            cwd=str(project_root),
            run_id=run_id,
            parent_run_id=parent_run_id,
            intended_environment=intended_environment,
            results=results,
            stage_details=_STAGE_DETAILS,
            log_cleanup=log_cleanup,
            report_retention=report_retention,
            critical_failures=critical_failures,
            strict_compare_mode=bool(args.compare_strict or ci_strict),
            strict_embedding_preflight_mode=strict_embedding_preflight,
            strict_web_checks_mode=strict_web_checks,
            health_score=compute_health_score(
                results=results,
                critical_failures=critical_failures,
                log_cleanup=log_cleanup,
                report_retention=report_retention,
            ),
        )
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
            log_cleanup = cleanup_ingested_stage_logs(
                stage_details=_STAGE_DETAILS,
                report_log_dir=REPORT_LOG_DIR,
            )
        except Exception:
            log_cleanup = {
                "enabled": False,
                "error": "log_cleanup_failed_in_exception_handler",
            }

        try:
            retention_enabled = bool(args.enforce_report_retention or ci_strict)
            report_retention = run_report_retention(
                enabled=retention_enabled,
                pattern=args.report_retention_pattern,
                max_age_days=args.report_retention_days,
                max_files=args.report_max_files,
                max_total_bytes=args.report_max_bytes,
                report_dir=REPORT_DIR,
                protected_names={RUN_MANIFEST_PATH.name},
            )
        except Exception:
            report_retention = {
                "enabled": False,
                "error": "report_retention_failed_in_exception_handler",
            }

        failure_critical_failures = sorted(set(critical_failures + ["unhandled_exception"]))
        failure_manifest = build_failure_manifest(
            started_at=run_started.isoformat(),
            cwd=str(project_root),
            run_id=run_id,
            parent_run_id=parent_run_id,
            intended_environment=intended_environment,
            results=results,
            stage_details=_STAGE_DETAILS,
            log_cleanup=log_cleanup,
            report_retention=report_retention,
            critical_failures=failure_critical_failures,
            strict_compare_mode=bool(args.compare_strict or ci_strict),
            strict_embedding_preflight_mode=strict_embedding_preflight,
            strict_web_checks_mode=strict_web_checks,
            health_score=compute_health_score(
                results=results,
                critical_failures=failure_critical_failures,
                log_cleanup=log_cleanup,
                report_retention=report_retention,
            ),
            exception_type=type(exc).__name__,
            exception_message=str(exc),
            traceback_text=error_trace,
        )

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