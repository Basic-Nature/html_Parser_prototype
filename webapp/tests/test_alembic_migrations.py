"""Smoke tests for Alembic migration wiring."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from sqlalchemy import create_engine, inspect


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_alembic(*args: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )


class TestAlembicMigrations:
    def test_upgrade_and_downgrade_on_sqlite(self, tmp_path):
        db_path = tmp_path / "alembic_phase_a.sqlite"
        env = os.environ.copy()
        env["DATABASE_URL"] = f"sqlite:///{db_path.as_posix()}"

        _run_alembic("upgrade", "head", env=env)

        engine = create_engine(env["DATABASE_URL"])
        inspector = inspect(engine)
        tables_after_upgrade = set(inspector.get_table_names())
        assert "states" in tables_after_upgrade
        assert "contests" in tables_after_upgrade
        engine.dispose()

        _run_alembic("downgrade", "base", env=env)

        engine = create_engine(env["DATABASE_URL"])
        inspector = inspect(engine)
        tables_after_downgrade = set(inspector.get_table_names())
        assert "states" not in tables_after_downgrade
        assert "contests" not in tables_after_downgrade
        engine.dispose()

    def test_offline_postgresql_sql_generation(self):
        env = os.environ.copy()
        env["DATABASE_URL"] = "postgresql://parser:parser@localhost:5432/smart_elections"

        result = _run_alembic("upgrade", "head", "--sql", env=env)
        sql_text = result.stdout.upper()

        assert "CREATE TABLE STATES" in sql_text
        assert "CREATE TABLE CONTESTS" in sql_text
        assert "CREATE TABLE RESULTS" in sql_text
