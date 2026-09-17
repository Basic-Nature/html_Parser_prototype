from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

import webapp.parser.handlers.formats.json_handler as json_handler
from webapp.parser.contracts.artifact_identity import ArtifactIdentityHandoff
from webapp.parser.services.parser_observation_bundle import (
    project_parser_observation_bundle,
)
from webapp.parser.services.parser_result_observation_adapter import (
    adapt_final_parser_result_for_observation,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
JSON_PATH = REPO_ROOT / "webapp/parser/handlers/formats/json_handler.py"
BASE_INNER_NORMALIZED_SOURCE_SHA256 = (
    "4b619ce0479eab2658044c41b5204e0f975a7ecf66123a0ef5685042b310438e"
)


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _tree(path: Path) -> ast.Module:
    return ast.parse(_source(path), filename=str(path))


def _fn(tree: ast.AST, name: str) -> ast.FunctionDef:
    rows = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(rows) == 1
    return rows[0]


def test_json_rawjson_value_never_enters_observation_bundle():
    secret = "SECRET RAW JSON PAYLOAD 987654321"
    result = adapt_final_parser_result_for_observation(
        ["Contest", "RawJSON"],
        [{"Contest": "Contest A", "RawJSON": secret}],
        source_type="json",
        source_sha256="a" * 64,
    )
    payload = project_parser_observation_bundle(result)

    assert payload["contract"] == "parser_observation_bundle_v1"
    assert payload["authority"]["canonical"] is False
    assert payload["raw_rows_included"] is False
    assert payload["raw_headers_included"] is False
    assert payload["automatic_timestamp"] is False
    assert secret not in repr(payload)


def test_json_wrapper_emits_after_parse_without_changing_return(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    json_path = tmp_path / "fixture.json"
    json_path.write_text('{"results": []}', encoding="utf-8")

    expected = (
        ["Precinct", "Candidate A - Total Votes"],
        [{"Precinct": "P-1", "Candidate A - Total Votes": 7}],
        "Contest A",
        {"handler": "json_handler", "marker": "unchanged"},
    )
    parse_calls = []
    adapt_calls = []
    emit_calls = []
    typed_sentinel = object()
    callback = lambda _payload: None
    identity = ArtifactIdentityHandoff("b" * 64)

    def fake_parse(*args, **kwargs):
        parse_calls.append((args, kwargs))
        return expected

    def fake_adapt(headers, rows, **kwargs):
        adapt_calls.append((headers, rows, kwargs))
        return typed_sentinel

    def fake_emit(result, **kwargs):
        emit_calls.append((result, kwargs))
        return True

    monkeypatch.setattr(json_handler, "parse_json_election_results", fake_parse)
    monkeypatch.setattr(
        json_handler,
        "adapt_final_parser_result_for_observation",
        fake_adapt,
    )
    monkeypatch.setattr(
        json_handler,
        "emit_parser_observation_bundle_if_requested",
        fake_emit,
    )

    actual = json_handler.parse(
        manual_file=str(json_path),
        artifact_identity=identity,
        parser_observation_emit_func=callback,
    )

    assert actual is expected
    assert len(parse_calls) == 1
    assert parse_calls[0][0] == (str(json_path),)
    assert parse_calls[0][1] == {
        "session_id": None,
        "coordinator": None,
    }
    assert adapt_calls == [
        (
            expected[0],
            expected[1],
            {
                "source_type": "json",
                "source_sha256": "b" * 64,
            },
        )
    ]
    assert emit_calls == [
        (
            typed_sentinel,
            {"parser_observation_emit_func": callback},
        )
    ]


def test_json_wrapper_preserves_unknown_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    json_path = tmp_path / "fixture.json"
    json_path.write_text("{}", encoding="utf-8")
    expected = (
        ["Precinct"],
        [{"Precinct": "P-1"}],
        "Contest",
        {"handler": "json_handler"},
    )
    adapt_calls = []

    monkeypatch.setattr(
        json_handler,
        "parse_json_election_results",
        lambda *args, **kwargs: expected,
    )

    def fake_adapt(headers, rows, **kwargs):
        adapt_calls.append(kwargs)
        return object()

    monkeypatch.setattr(
        json_handler,
        "adapt_final_parser_result_for_observation",
        fake_adapt,
    )
    monkeypatch.setattr(
        json_handler,
        "emit_parser_observation_bundle_if_requested",
        lambda *args, **kwargs: True,
    )

    assert json_handler.parse(
        manual_file=str(json_path),
        parser_observation_emit_func=lambda _payload: None,
    ) is expected
    assert adapt_calls == [
        {
            "source_type": "json",
            "source_sha256": None,
        }
    ]


def test_json_wrapper_does_not_adapt_when_callback_absent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    json_path = tmp_path / "fixture.json"
    json_path.write_text("{}", encoding="utf-8")
    expected = (
        ["Precinct"],
        [{"Precinct": "P-1"}],
        "Contest",
        {"handler": "json_handler"},
    )
    monkeypatch.setattr(
        json_handler,
        "parse_json_election_results",
        lambda *args, **kwargs: expected,
    )

    def should_not_run(*args, **kwargs):
        raise AssertionError("observation adapter must remain dormant")

    monkeypatch.setattr(
        json_handler,
        "adapt_final_parser_result_for_observation",
        should_not_run,
    )
    monkeypatch.setattr(
        json_handler,
        "emit_parser_observation_bundle_if_requested",
        should_not_run,
    )

    assert json_handler.parse(manual_file=str(json_path)) is expected


def test_json_provided_tables_callback_fails_closed_before_extraction(
    monkeypatch: pytest.MonkeyPatch,
):
    def should_not_extract(*args, **kwargs):
        raise AssertionError("provided_tables extraction must not start")

    monkeypatch.setattr(
        json_handler,
        "robust_table_extraction",
        should_not_extract,
    )

    with pytest.raises(
        RuntimeError,
        match="parser observation callback is unavailable",
    ):
        json_handler.parse(
            html_context={
                "provided_tables": [
                    {
                        "headers": ["Precinct"],
                        "rows": [{"Precinct": "P-1"}],
                    }
                ]
            },
            parser_observation_emit_func=lambda _payload: None,
        )


def test_json_inner_parser_remains_observation_transport_free_and_exact():
    source = _source(JSON_PATH)
    tree = _tree(JSON_PATH)
    inner = _fn(tree, "parse_json_election_results")
    segment = ast.get_source_segment(source, inner) or ""

    assert hashlib.sha256(segment.encode("utf-8")).hexdigest() == (
        BASE_INNER_NORMALIZED_SOURCE_SHA256
    )
    for prohibited in (
        "parser_observation_emit_func",
        "ArtifactIdentityHandoff",
        "adapt_final_parser_result_for_observation",
        "emit_parser_observation_bundle_if_requested",
    ):
        assert prohibited not in segment


def test_json_wrapper_uses_existing_identity_only_and_source_type_json():
    source = _source(JSON_PATH)
    tree = _tree(JSON_PATH)
    wrapper = _fn(tree, "parse")
    segment = ast.get_source_segment(source, wrapper) or ""

    kwonly = [arg.arg for arg in wrapper.args.kwonlyargs]
    assert kwonly.count("artifact_identity") == 1
    assert "parser_observation_emit_func" in segment
    assert "artifact_identity.document_sha256" in segment
    assert 'source_type="json"' in segment
    assert "hashlib" not in segment
    assert "orjson.loads" not in segment
    assert "RawJSON" not in segment


def test_json_wrapper_observation_is_after_valid_four_tuple_contract():
    source = _source(JSON_PATH)
    tree = _tree(JSON_PATH)
    wrapper = _fn(tree, "parse")
    segment = ast.get_source_segment(source, wrapper) or ""

    validation = segment.index("len(result_any) == 4")
    adaptation = segment.index("adapt_final_parser_result_for_observation(")
    emission = segment.index("emit_parser_observation_bundle_if_requested(")
    final_return = segment.rindex("return cast(")

    assert validation < adaptation < emission < final_return


def test_json_callback_is_consumed_only_by_outer_wrapper():
    source = _source(JSON_PATH)
    tree = _tree(JSON_PATH)
    wrapper = _fn(tree, "parse")
    inner = _fn(tree, "parse_json_election_results")
    wrapper_segment = ast.get_source_segment(source, wrapper) or ""
    inner_segment = ast.get_source_segment(source, inner) or ""

    assert 'kwargs.pop(\n        "parser_observation_emit_func"' in wrapper_segment
    assert "parser_observation_emit_func" not in inner_segment
    assert "artifact_identity" not in inner_segment
