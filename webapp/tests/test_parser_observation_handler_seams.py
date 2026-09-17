from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from webapp.parser.handlers.formats import csv_handler, xlsx_handler
from webapp.parser.services import parser_observation_callback as callback_module
from webapp.parser.services.parser_observation_callback import (
    emit_parser_observation_bundle_if_requested,
)


def _function_tree(function):
    source = inspect.getsource(function)
    return ast.parse(source), source


def _call_leaf(call: ast.Call) -> str:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return ""


def _first_call_line(function, name: str) -> int:
    tree, _ = _function_tree(function)
    lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _call_leaf(node) == name
    ]
    assert lines, name
    return min(lines)


def _parameter_names(function) -> set[str]:
    return set(inspect.signature(function).parameters)


def test_callback_none_is_a_true_noop(monkeypatch) -> None:
    called = []

    def _project(result):
        called.append(result)
        return {"unexpected": True}

    monkeypatch.setattr(
        callback_module,
        "project_parser_observation_bundle",
        _project,
    )

    assert emit_parser_observation_bundle_if_requested(
        object(),
        parser_observation_emit_func=None,
    ) is False
    assert called == []


def test_callback_non_callable_fails_closed_before_projection(monkeypatch) -> None:
    called = []

    def _project(result):
        called.append(result)
        return {"unexpected": True}

    monkeypatch.setattr(
        callback_module,
        "project_parser_observation_bundle",
        _project,
    )

    with pytest.raises(TypeError, match="must be callable"):
        emit_parser_observation_bundle_if_requested(
            object(),
            parser_observation_emit_func="not-callable",
        )
    assert called == []


def test_callback_delivers_exactly_one_projected_bundle(monkeypatch) -> None:
    result = object()
    projected = {
        "contract": "parser_observation_bundle_v1",
        "authority": {
            "inspection": "noncanonical_parser_evidence",
            "canonical": False,
        },
        "automatic_timestamp": False,
    }
    project_calls = []
    emitted = []

    def _project(value):
        project_calls.append(value)
        return projected

    monkeypatch.setattr(
        callback_module,
        "project_parser_observation_bundle",
        _project,
    )

    assert emit_parser_observation_bundle_if_requested(
        result,
        parser_observation_emit_func=emitted.append,
    ) is True
    assert project_calls == [result]
    assert emitted == [projected]
    assert emitted[0] is projected


def test_csv_primary_only_exposes_dormant_parser_observation_callback() -> None:
    assert "parser_observation_emit_func" in _parameter_names(
        csv_handler.parse_csv_election_results
    )
    assert "parser_observation_emit_func" not in _parameter_names(
        csv_handler.parse
    )

    build = _first_call_line(
        csv_handler.parse_csv_election_results,
        "build_table_noninteractive_result",
    )
    store = _first_call_line(
        csv_handler.parse_csv_election_results,
        "_store_pipeline_inspection_if_requested",
    )
    inspect_emit = _first_call_line(
        csv_handler.parse_csv_election_results,
        "_emit_pipeline_inspection_if_requested",
    )
    observation_emit = _first_call_line(
        csv_handler.parse_csv_election_results,
        "emit_parser_observation_bundle_if_requested",
    )
    finalize = _first_call_line(
        csv_handler.parse_csv_election_results,
        "finalize_election_output",
    )
    assert build < store < inspect_emit < observation_emit < finalize


def test_xlsx_primary_only_exposes_same_dormant_callback() -> None:
    assert "parser_observation_emit_func" in _parameter_names(
        xlsx_handler.parse_xlsx_election_results
    )
    assert "parser_observation_emit_func" not in _parameter_names(
        xlsx_handler.parse
    )

    for legacy_name in (
        "inspection_store",
        "inspection_principal",
        "inspection_emit_func",
    ):
        assert legacy_name not in _parameter_names(
            xlsx_handler.parse_xlsx_election_results
        )

    build = _first_call_line(
        xlsx_handler.parse_xlsx_election_results,
        "build_table_noninteractive_result",
    )
    observation_emit = _first_call_line(
        xlsx_handler.parse_xlsx_election_results,
        "emit_parser_observation_bundle_if_requested",
    )
    finalize = _first_call_line(
        xlsx_handler.parse_xlsx_election_results,
        "finalize_election_output",
    )
    assert build < observation_emit < finalize


def test_existing_pipeline_inspection_transport_is_not_reused() -> None:
    project_root = Path(callback_module.__file__).resolve().parents[3]

    callback_source = Path(callback_module.__file__).read_text(encoding="utf-8")
    socket_source = (
        project_root / "webapp/parser/socket_ballot_lens_orchestration.py"
    ).read_text(encoding="utf-8")
    store_source = (
        project_root / "webapp/parser/services/ephemeral_pipeline_inspection.py"
    ).read_text(encoding="utf-8")
    web_pipeline_source = (
        project_root / "webapp/parser/web_pipeline.py"
    ).read_text(encoding="utf-8")
    js_source = (
        project_root / "webapp/static/js/pipeline_inspection_consumer.js"
    ).read_text(encoding="utf-8")

    for forbidden in (
        "ProcessLocalInspectionStore",
        "pipeline_inspection_socket_v1",
        "inspection_emit_func",
        "socketio",
        "finalize_election_output",
    ):
        assert forbidden not in callback_source

    # W22 intentionally carries the observation emitter through the trusted
    # socket and web-pipeline layers. Keep the legacy inspection store and
    # browser consumer isolated from that private callback.
    for untouched in (
        store_source,
        js_source,
    ):
        assert "parser_observation_emit_func" not in untouched
        assert "parser_observation_callback" not in untouched

    # Transport layers may forward the emitter, but they must not import/reuse
    # the handler callback service directly.
    for transport_source in (
        socket_source,
        web_pipeline_source,
    ):
        assert "parser_observation_callback" not in transport_source


def test_callback_seam_is_not_metadata_or_return_value_wiring() -> None:
    _, csv_source = _function_tree(csv_handler.parse_csv_election_results)
    _, xlsx_source = _function_tree(xlsx_handler.parse_xlsx_election_results)

    assert '"parser_observation"' not in csv_source
    assert '"parser_observation"' not in xlsx_source
    assert "parser_observation_emit_func=" in csv_source
    assert "parser_observation_emit_func=" in xlsx_source
