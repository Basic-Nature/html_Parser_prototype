from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any


def classify_intended_environment(
    requested: str,
    *,
    ci_strict: bool,
    is_development: bool,
    is_localhost: bool,
) -> str:
    normalized = (requested or "auto").strip().lower()
    if normalized and normalized != "auto":
        return normalized
    if ci_strict:
        return "ci"
    if is_development:
        return "development"
    if is_localhost:
        return "local"
    return "production"


def collect_critical_failures(
    *,
    results: dict[str, Any],
    args: argparse.Namespace,
    ci_strict: bool,
    strict_embedding_preflight: bool,
    base_critical_stages: set[str] | None = None,
    always_critical_on_false: set[str] | None = None,
) -> list[str]:
    critical_stage_set = base_critical_stages or {"pipeline_audit", "web_checks"}
    critical_failures = [
        stage
        for stage, value in results.items()
        if value is False and stage in critical_stage_set
    ]

    if always_critical_on_false:
        for stage in always_critical_on_false:
            if results.get(stage) is False:
                critical_failures.append(stage)

    if args.self_check and results.get("self_check") is False:
        critical_failures.append("self_check")
    if args.ballot_lens_check and results.get("ballot_lens_check") is False:
        critical_failures.append("ballot_lens_check")
    if args.pipeline_check and results.get("pipeline_check") is False:
        critical_failures.append("pipeline_check")
    if args.compare_dl1_dl2 and results.get("dl_compare") is False and (args.compare_strict or ci_strict):
        critical_failures.append("dl_compare")
    if results.get("embedding_cache_preflight") is False and strict_embedding_preflight:
        critical_failures.append("embedding_cache_preflight")

    return sorted(set(critical_failures))


def compute_health_score(
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


def cleanup_ingested_stage_logs(
    *,
    stage_details: dict[str, dict[str, Any]],
    report_log_dir: Path,
    keep_all_logs: bool | None = None,
) -> dict[str, Any]:
    if keep_all_logs is None:
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

    for stage_name, detail in stage_details.items():
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

        stage_details[stage_name] = detail

    try:
        if report_log_dir.exists() and not any(report_log_dir.iterdir()):
            report_log_dir.rmdir()
    except OSError:
        pass

    summary["removed_count"] = len(summary["removed"])
    summary["kept_count"] = len(summary["kept"])
    return summary


def run_report_retention(
    *,
    enabled: bool,
    pattern: str,
    max_age_days: int,
    max_files: int,
    max_total_bytes: int,
    report_dir: Path,
    protected_names: set[str] | None = None,
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

    protected = protected_names or set()
    now_ts = time.time()
    cutoff_ts = now_ts - (max_age_days * 86400) if max_age_days >= 0 else None

    try:
        candidates = [
            p
            for p in report_dir.glob(pattern)
            if p.is_file() and p.name not in protected and not p.name.endswith("_latest.json")
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