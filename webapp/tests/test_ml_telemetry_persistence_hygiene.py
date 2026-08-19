"""Contracts for ML telemetry persistence hygiene."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from webapp.parser.utils import ml_pipeline_profile
from webapp.parser.utils import ml_telemetry


REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKED_TELEMETRY = (
    REPO_ROOT
    / "webapp/parser/log/ml_usage_telemetry.jsonl"
)


def _run_isolated(code: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _extract_sentinel(stdout: str, prefix: str) -> str:
    matches = [
        line[len(prefix):].strip()
        for line in stdout.splitlines()
        if line.startswith(prefix)
    ]

    assert matches, (
        f"Missing {prefix!r} sentinel in subprocess stdout:\n"
        f"{stdout}"
    )

    return matches[-1]


def test_explicit_log_path_override_is_preserved(tmp_path):
    target = tmp_path / "explicit" / "ml.jsonl"

    env = os.environ.copy()
    env["ML_TELEMETRY_PERSIST"] = "true"
    env["ML_TELEMETRY_LOG_PATH"] = str(target)

    code = """
from webapp.parser.utils.ml_telemetry import (
    get_ml_telemetry_log_path,
    record_ml_event,
)

record_ml_event("hygiene_test", "explicit_override")
print("TELEMETRY_PATH=" + str(get_ml_telemetry_log_path()))
"""

    result = _run_isolated(code, env)

    assert result.returncode == 0, result.stderr
    reported = _extract_sentinel(result.stdout, "TELEMETRY_PATH=")
    assert Path(reported).resolve() == target.resolve()
    assert target.exists()
    assert target.read_text(encoding="utf-8").count("\n") == 1


def test_default_log_path_uses_runtime_state_directory_not_repo(tmp_path):
    local_appdata = tmp_path / "LocalAppData"

    env = os.environ.copy()
    env["ML_TELEMETRY_PERSIST"] = "true"
    env.pop("ML_TELEMETRY_LOG_PATH", None)
    env["LOCALAPPDATA"] = str(local_appdata)

    code = """
from webapp.parser.utils.ml_telemetry import (
    get_ml_telemetry_log_path,
    record_ml_event,
)

record_ml_event("hygiene_test", "default_runtime_path")
print("TELEMETRY_PATH=" + str(get_ml_telemetry_log_path()))
"""

    result = _run_isolated(code, env)

    assert result.returncode == 0, result.stderr

    reported = _extract_sentinel(result.stdout, "TELEMETRY_PATH=")
    resolved = Path(reported).resolve()
    expected = (
        local_appdata
        / "ElectionPulse"
        / "logs"
        / "ml_usage_telemetry.jsonl"
    ).resolve()

    assert resolved == expected
    assert resolved.exists()
    assert REPO_ROOT.resolve() not in resolved.parents


def test_pipeline_profile_counts_resolved_telemetry_path(
    tmp_path,
    monkeypatch,
):
    runtime_log = tmp_path / "runtime" / "ml_usage_telemetry.jsonl"
    runtime_log.parent.mkdir(parents=True, exist_ok=True)
    runtime_log.write_text(
        '{"event": 1}\n{"event": 2}\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(
        ml_telemetry,
        "get_ml_telemetry_log_path",
        lambda: runtime_log,
    )
    monkeypatch.setattr(
        ml_telemetry,
        "get_ml_telemetry_snapshot",
        lambda **_kwargs: {
            "totals": {
                "events": 2,
                "components": 1,
                "actions": 1,
            }
        },
    )

    profile = ml_pipeline_profile.get_ml_pipeline_profile()

    assert (
        profile["training_inputs"]["ml_usage_telemetry_rows"]
        == 2
    )


def test_default_telemetry_path_is_not_tracked_repo_log():
    assert ml_telemetry.get_ml_telemetry_log_path().resolve() != (
        TRACKED_TELEMETRY.resolve()
    )
