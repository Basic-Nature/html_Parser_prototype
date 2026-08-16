"""Contracts for Tranche 1 authority-context extraction."""

from __future__ import annotations

import ast
from pathlib import Path


TARGETS = ('get_request_principal', '_resolve_cert_session_id', '_derive_auth_context', '_apply_auth_context', '_session_has_principal', 'resolve_session_id')


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_extracted_context_module_owns_target_implementations():
    source = (
        _repo_root()
        / "webapp"
        / "parser"
        / "auth"
        / "context.py"
    ).read_text(encoding="utf-8")

    tree = ast.parse(source)
    names = {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }

    for target in TARGETS:
        assert target in names


def test_composition_root_keeps_compatibility_wrappers():
    source = (
        _repo_root()
        / "webapp"
        / "Smart_Elections_Parser_Webapp.py"
    ).read_text(encoding="utf-8")

    for target in TARGETS:
        assert f"_authority_context.{target}" in source

    assert "def _configure_authority_context_runtime" in source


def test_extracted_context_does_not_import_composition_root():
    source = (
        _repo_root()
        / "webapp"
        / "parser"
        / "auth"
        / "context.py"
    ).read_text(encoding="utf-8")

    tree = ast.parse(source)

    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)

    assert not any(
        module == "webapp.Smart_Elections_Parser_Webapp"
        or module.startswith("webapp.Smart_Elections_Parser_Webapp.")
        for module in imported_modules
    )
    assert "ContextVar" in source
