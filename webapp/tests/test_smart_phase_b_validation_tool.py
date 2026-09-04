"""Tests for tools/smart_phase_b_validation.py target resolution behavior."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[2]
    module_path = root / "tools" / "smart_phase_b_validation.py"
    spec = importlib.util.spec_from_file_location("smart_phase_b_validation", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_collect_targets_always_includes_core_orchestration_test():
    mod = _load_module()

    pytest_targets, ruff_targets = mod._collect_targets(changed=[], strict=False)

    assert "webapp/tests/test_html_parser_database_policy_flow.py" in pytest_targets
    assert "webapp/tests/test_html_parser_database_policy_flow.py" in ruff_targets


def test_collect_targets_includes_parser_adjacent_files_when_changed():
    mod = _load_module()
    changed = [
        "webapp/parser/utils/output_utils.py",
        "webapp/parser/utils/database_comparison.py",
        "webapp/parser/html_election_parser.py",
    ]

    pytest_targets, ruff_targets = mod._collect_targets(changed=changed, strict=False)

    assert "webapp/tests/test_output_utils_database_crosscheck.py" in pytest_targets
    assert "webapp/tests/test_database_comparison_centralized.py" in pytest_targets
    assert "webapp/parser/utils/output_utils.py" in ruff_targets
    assert "webapp/parser/utils/database_comparison.py" in ruff_targets
    assert "webapp/parser/html_election_parser.py" in ruff_targets


def test_collect_targets_strict_mode_adds_phase_a_slice():
    mod = _load_module()

    pytest_targets, _ruff_targets = mod._collect_targets(changed=[], strict=True)

    assert "webapp/tests/test_phase_a_integration.py" in pytest_targets


def test_apply_change_scope_parser_only_filters_docs_and_ui_files():
    mod = _load_module()

    changed = [
        "docs/FEATURES/DATABASE_COMPARISON.md",
        "webapp/frontend/ballot-lens/main.tsx",
        "webapp/parser/utils/output_utils.py",
        "webapp/tests/test_output_utils_database_crosscheck.py",
        "automate.py",
    ]

    filtered = mod._apply_change_scope(changed, "parser-only")

    assert "webapp/parser/utils/output_utils.py" in filtered
    assert "webapp/tests/test_output_utils_database_crosscheck.py" in filtered
    assert "automate.py" in filtered
    assert "docs/FEATURES/DATABASE_COMPARISON.md" not in filtered
    assert "webapp/frontend/ballot-lens/main.tsx" not in filtered


def test_apply_change_scope_all_keeps_everything():
    mod = _load_module()

    changed = [
        "docs/FEATURES/DATABASE_COMPARISON.md",
        "webapp/parser/utils/output_utils.py",
    ]

    filtered = mod._apply_change_scope(changed, "all")

    assert filtered == changed


# ---------------------------------------------------------------------------
# Manifest traceability contract
# ---------------------------------------------------------------------------


def _make_stage_detail(summary_payload: dict | None = None) -> dict:
    """Build a stage detail dict structured exactly as run_smart_phase_b_check_with_scope emits."""
    detail: dict = {
        "status": "passed",
        "summary_path": "output/reports/smart_phase_b_summary.json",
        "changed_scope": "parser-only",
    }
    if summary_payload is not None:
        detail["smart_phase_b_summary"] = summary_payload
    return detail


def test_smart_phase_b_stage_detail_keys_present_in_manifest():
    """Stage detail emitted by the smart phase B runner must survive into build_completed_manifest."""
    from scripts.automation_runtime import build_completed_manifest

    summary_payload = {
        "status": "passed",
        "changed_count": 57,
        "scoped_changed_count": 13,
        "filtered_out_count": 44,
        "changed_scope": "parser-only",
        "pytest_ok": True,
        "ruff_ok": True,
        "pytest_targets": ["webapp/tests/test_html_parser_database_policy_flow.py"],
        "ruff_targets": ["webapp/tests/test_html_parser_database_policy_flow.py"],
        "changed": [],
        "scoped_changed": [],
    }

    stage_details = {"smart_phase_b_check": _make_stage_detail(summary_payload)}

    manifest = build_completed_manifest(
        started_at="2026-03-30T00:00:00+00:00",
        cwd="/repo",
        run_id="test-run-id",
        parent_run_id=None,
        intended_environment="localhost",
        results={"smart_phase_b_check": True},
        stage_details=stage_details,
        log_cleanup={},
        report_retention={},
        critical_failures=[],
        strict_compare_mode=False,
        strict_embedding_preflight_mode=False,
        strict_web_checks_mode=False,
        health_score={},
    )

    detail = manifest["stage_details"]["smart_phase_b_check"]

    # Required top-level keys
    assert detail["status"] == "passed"
    assert detail["changed_scope"] == "parser-only"
    assert "summary_path" in detail

    # Nested summary must be present and structurally correct
    assert "smart_phase_b_summary" in detail
    summary = detail["smart_phase_b_summary"]
    assert summary["status"] == "passed"
    assert summary["changed_scope"] == "parser-only"
    assert isinstance(summary["changed_count"], int)
    assert isinstance(summary["scoped_changed_count"], int)
    assert isinstance(summary["filtered_out_count"], int)
    assert isinstance(summary["pytest_targets"], list)
    assert isinstance(summary["ruff_targets"], list)
    assert summary["pytest_ok"] is True
    assert summary["ruff_ok"] is True


def test_smart_phase_b_stage_detail_missing_summary_does_not_crash_manifest():
    """Manifest build must not raise when smart_phase_b_summary is absent (summary file write failure)."""
    from scripts.automation_runtime import build_completed_manifest

    stage_details = {"smart_phase_b_check": _make_stage_detail(summary_payload=None)}

    manifest = build_completed_manifest(
        started_at="2026-03-30T00:00:00+00:00",
        cwd="/repo",
        run_id="test-run-id-2",
        parent_run_id=None,
        intended_environment="localhost",
        results={"smart_phase_b_check": False},
        stage_details=stage_details,
        log_cleanup={},
        report_retention={},
        critical_failures=["smart_phase_b_check"],
        strict_compare_mode=False,
        strict_embedding_preflight_mode=False,
        strict_web_checks_mode=False,
        health_score={},
    )

    detail = manifest["stage_details"]["smart_phase_b_check"]
    assert detail["changed_scope"] == "parser-only"
    assert "summary_path" in detail
    assert "smart_phase_b_summary" not in detail
