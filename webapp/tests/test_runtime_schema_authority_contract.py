from __future__ import annotations

import ast
import inspect
from pathlib import Path

from webapp.parser.persistence import schema_bootstrap


REPO = Path(__file__).resolve().parents[2]

APP = REPO / "webapp" / "Smart_Elections_Parser_Webapp.py"
HEALTH = REPO / "webapp" / "parser" / "health" / "health_router.py"


def test_runtime_auto_init_defaults_false() -> None:
    source = APP.read_text(encoding="utf-8")

    assert source.count(
        'os.environ.get("AUTO_INIT_DB", "false")'
    ) == 2

    assert 'os.environ.get("AUTO_INIT_DB", "true")' not in source


def test_runtime_and_health_use_read_only_schema_verifier() -> None:
    app_source = APP.read_text(encoding="utf-8")
    health_source = HEALTH.read_text(encoding="utf-8")

    assert "verify_application_schema_compat" in app_source
    assert "ensure_application_schema_compat" not in app_source

    assert "verify_application_schema_compat" in health_source
    assert "ensure_application_schema_compat" not in health_source


def test_schema_verifier_contains_no_ddl() -> None:
    source = inspect.getsource(
        schema_bootstrap.verify_application_schema_compat
    )

    tree = ast.parse(source)
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
    }

    assert "has_table" in called_attributes
    assert "create_all" not in called_attributes
    assert "drop_all" not in called_attributes


def test_explicit_bootstrap_remains_deliberate_ddl_path() -> None:
    source = inspect.getsource(
        schema_bootstrap.ensure_application_schema_compat
    )

    assert "Base.metadata.create_all(engine)" in source


def test_core_runtime_files_have_no_create_all_call() -> None:
    app_source = APP.read_text(encoding="utf-8")
    health_source = HEALTH.read_text(encoding="utf-8")

    assert "create_all(" not in app_source
    assert "create_all(" not in health_source
