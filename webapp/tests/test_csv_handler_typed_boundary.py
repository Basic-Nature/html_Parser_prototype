"""C2G 1.3 contract for the first live typed format-handler seam."""

from __future__ import annotations

import ast
import inspect

from webapp.parser.contracts.table_pipeline import TableStage
from webapp.parser.handlers.formats import csv_handler


def _function_tree(function):
    source = inspect.getsource(function)
    return ast.parse(source), source


def _call_leaf(call: ast.Call) -> str:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return ""


def _call_count(function, name: str) -> int:
    tree, _ = _function_tree(function)
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _call_leaf(node) == name
    )


def test_csv_primary_path_uses_exactly_one_typed_builder_boundary() -> None:
    assert _call_count(
        csv_handler.parse_csv_election_results,
        "build_table_noninteractive_result",
    ) == 1
    assert _call_count(
        csv_handler.parse_csv_election_results,
        "build_table_noninteractive",
    ) == 0


def test_csv_primary_typed_call_declares_csv_source_type() -> None:
    tree, _ = _function_tree(csv_handler.parse_csv_election_results)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _call_leaf(node) == "build_table_noninteractive_result"
    ]

    assert len(calls) == 1

    source_type = [
        keyword.value.value
        for keyword in calls[0].keywords
        if keyword.arg == "source_type"
        and isinstance(keyword.value, ast.Constant)
    ]

    assert source_type == ["csv"]


def test_csv_fallback_parse_remains_legacy_control_path() -> None:
    assert _call_count(csv_handler.parse, "build_table_noninteractive") == 1
    assert _call_count(csv_handler.parse, "build_table_noninteractive_result") == 0


def test_csv_primary_finalization_boundary_remains_single() -> None:
    assert _call_count(
        csv_handler.parse_csv_election_results,
        "finalize_election_output",
    ) == 1


def test_typed_result_stage_contract_remains_interpreted_not_canonical() -> None:
    assert TableStage.INTERPRETED.value == "interpreted"
    assert "canonical" not in {stage.value for stage in TableStage}