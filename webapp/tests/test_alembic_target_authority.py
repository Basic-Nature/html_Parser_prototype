"""Fail-closed Alembic database-target authority contracts."""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]


def _base_env() -> dict[str, str]:
    env = os.environ.copy()

    for key in (
        "DATABASE_URL",
        "WEBSITE_SITE_NAME",
        "WEBSITE_INSTANCE_ID",
        "WEBSITE_HOSTNAME",
        "DEPLOY_ENV",
    ):
        env.pop(key, None)

    return env


def _run_alembic(*args: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _combined_output(result: subprocess.CompletedProcess[str]) -> str:
    return f"{result.stdout}\n{result.stderr}"


def test_runtime_requirements_include_alembic():
    requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")

    assert re.search(
        r"(?mi)^\s*alembic\s*>=\s*1\.18\.4\s*$",
        requirements,
    )


def test_development_without_database_url_keeps_explicit_sqlite_fallback():
    env = _base_env()
    env["DEPLOY_ENV"] = "development"

    result = _run_alembic("upgrade", "head", "--sql", env=env)

    assert result.returncode == 0, _combined_output(result)


def test_azure_app_service_requires_explicit_database_url():
    env = _base_env()
    env["WEBSITE_SITE_NAME"] = "BallotLens"

    result = _run_alembic("upgrade", "head", "--sql", env=env)

    assert result.returncode != 0
    assert (
        "production-like execution requires an explicit DATABASE_URL"
        in _combined_output(result)
    )


def test_production_deploy_env_requires_explicit_database_url():
    env = _base_env()
    env["DEPLOY_ENV"] = "production"

    result = _run_alembic("upgrade", "head", "--sql", env=env)

    assert result.returncode != 0
    assert (
        "production-like execution requires an explicit DATABASE_URL"
        in _combined_output(result)
    )


def test_azure_app_service_rejects_explicit_sqlite_database_url(tmp_path):
    env = _base_env()
    env["WEBSITE_SITE_NAME"] = "BallotLens"
    env["DATABASE_URL"] = f"sqlite:///{(tmp_path / 'forbidden.sqlite').as_posix()}"

    result = _run_alembic("upgrade", "head", "--sql", env=env)

    assert result.returncode != 0
    assert (
        "SQLite is not allowed for production-like execution"
        in _combined_output(result)
    )


def test_production_explicit_postgresql_url_is_allowed_for_offline_sql():
    env = _base_env()
    env["DEPLOY_ENV"] = "production"
    env["DATABASE_URL"] = (
        "postgresql://parser:parser@localhost:5432/smart_elections"
    )

    result = _run_alembic("upgrade", "head", "--sql", env=env)

    assert result.returncode == 0, _combined_output(result)
