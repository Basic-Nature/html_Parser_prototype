from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from typing import Any, Callable


def execute_automation_stages(
    args: argparse.Namespace,
    *,
    strict_web_checks: bool,
    logger: Any,
    record_stage_detail: Callable[[str, dict[str, Any]], None],
    run_embedding_cache_preflight: Callable[[], bool],
    run_pipeline_audit: Callable[[], bool],
    run_health_bots: Callable[[], bool],
    run_web_checks: Callable[..., bool],
    run_automated_tests: Callable[[], bool],
    run_self_check: Callable[[], bool],
    run_ballot_lens_check: Callable[[], bool],
    run_pipeline_check: Callable[[], bool],
    run_dl_compare_check: Callable[..., bool],
    validate_webapp_startup: Callable[[], bool],
    extra_stage_runners: dict[str, Callable[[], Any]] | None = None,
) -> dict[str, Any]:
    results: dict[str, Any] = {}

    results["embedding_cache_preflight"] = run_embedding_cache_preflight()

    if args.simulate_unhandled_failure:
        allow_sim = os.environ.get("AUTOMATE_ALLOW_SIMULATED_FAILURE", "").lower() in {"1", "true", "yes"}
        if not allow_sim:
            raise RuntimeError("--simulate-unhandled-failure requires AUTOMATE_ALLOW_SIMULATED_FAILURE=true")
        raise RuntimeError("Synthetic unhandled failure (debug flag)")

    results["pipeline_audit"] = run_pipeline_audit()

    if not args.skip_health:
        results["health_bots"] = run_health_bots()
    else:
        print("[AUTOMATE] Skipping health bots.")
        logger.info("[AUTOMATE] Skipping health bots.")
        results["health_bots"] = None

    if not args.skip_web:
        results["web_checks"] = run_web_checks(strict=strict_web_checks)
    else:
        print("[AUTOMATE] Skipping web checks.")
        logger.info("[AUTOMATE] Skipping web checks.")
        results["web_checks"] = None

    if not args.skip_tests:
        results["tests"] = run_automated_tests()
    else:
        print("[AUTOMATE] Skipping automated tests.")
        logger.info("[AUTOMATE] Skipping automated tests.")
        results["tests"] = None

    if args.self_check:
        results["self_check"] = run_self_check()
    else:
        results["self_check"] = None

    if args.ballot_lens_check:
        results["ballot_lens_check"] = run_ballot_lens_check()
    else:
        results["ballot_lens_check"] = None

    if args.pipeline_check:
        results["pipeline_check"] = run_pipeline_check()
    else:
        results["pipeline_check"] = None

    if args.compare_dl1_dl2:
        if not args.dl1_path or not args.dl2_path:
            logger.error("[AUTOMATE] --compare-dl1-dl2 requires both --dl1-path and --dl2-path")
            record_stage_detail(
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

    if not args.skip_webapp_check:
        results["webapp_validation"] = validate_webapp_startup()
    else:
        print("[AUTOMATE] Skipping webapp validation.")
        logger.info("[AUTOMATE] Skipping webapp validation.")
        results["webapp_validation"] = None

    if extra_stage_runners:
        for stage_name, runner in extra_stage_runners.items():
            try:
                results[stage_name] = runner()
            except Exception as exc:
                logger.error(f"[AUTOMATE] Extra stage '{stage_name}' failed: {exc}")
                results[stage_name] = False

    return results


def build_bootstrap_manifest(
    *,
    started_at: str,
    cwd: str,
    run_id: str,
    parent_run_id: str | None,
    intended_environment: str,
    strict_compare_mode: bool,
    strict_embedding_preflight_mode: bool,
    strict_web_checks_mode: bool,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "started_at": started_at,
        "cwd": cwd,
        "run_id": run_id,
        "parent_run_id": parent_run_id,
        "intended_environment": intended_environment,
        "status": "running",
        "results": {},
        "stage_details": {},
        "critical_failures": [],
        "strict_compare_mode": strict_compare_mode,
        "strict_embedding_preflight_mode": strict_embedding_preflight_mode,
        "strict_web_checks_mode": strict_web_checks_mode,
    }


def build_completed_manifest(
    *,
    started_at: str,
    cwd: str,
    run_id: str,
    parent_run_id: str | None,
    intended_environment: str,
    results: dict[str, Any],
    stage_details: dict[str, Any],
    log_cleanup: dict[str, Any],
    report_retention: dict[str, Any],
    critical_failures: list[str],
    strict_compare_mode: bool,
    strict_embedding_preflight_mode: bool,
    strict_web_checks_mode: bool,
    health_score: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "started_at": started_at,
        "cwd": cwd,
        "run_id": run_id,
        "parent_run_id": parent_run_id,
        "intended_environment": intended_environment,
        "status": "completed",
        "results": results,
        "stage_details": stage_details,
        "log_cleanup": log_cleanup,
        "report_retention": report_retention,
        "critical_failures": critical_failures,
        "strict_compare_mode": strict_compare_mode,
        "strict_embedding_preflight_mode": strict_embedding_preflight_mode,
        "strict_web_checks_mode": strict_web_checks_mode,
        "health_score": health_score,
    }


def build_failure_manifest(
    *,
    started_at: str,
    cwd: str,
    run_id: str,
    parent_run_id: str | None,
    intended_environment: str,
    results: dict[str, Any],
    stage_details: dict[str, Any],
    log_cleanup: dict[str, Any],
    report_retention: dict[str, Any],
    critical_failures: list[str],
    strict_compare_mode: bool,
    strict_embedding_preflight_mode: bool,
    strict_web_checks_mode: bool,
    health_score: dict[str, Any],
    exception_type: str,
    exception_message: str,
    traceback_text: str,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "started_at": started_at,
        "cwd": cwd,
        "run_id": run_id,
        "parent_run_id": parent_run_id,
        "intended_environment": intended_environment,
        "status": "failed",
        "results": results,
        "stage_details": stage_details,
        "log_cleanup": log_cleanup,
        "report_retention": report_retention,
        "critical_failures": critical_failures,
        "strict_compare_mode": strict_compare_mode,
        "strict_embedding_preflight_mode": strict_embedding_preflight_mode,
        "strict_web_checks_mode": strict_web_checks_mode,
        "health_score": health_score,
        "unhandled_exception": {
            "type": exception_type,
            "message": exception_message,
            "traceback": traceback_text,
        },
    }