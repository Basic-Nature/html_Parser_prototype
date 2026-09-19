from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

import webapp.parser.handlers.formats.json_handler as json_handler
from webapp.parser.contracts.artifact_identity import ArtifactIdentityHandoff
from webapp.parser.services.json_source_derivative_evidence import (
    CONTRACT,
    FINGERPRINT_SCHEME,
    JsonSourceDerivativeObservationError,
    observe_json_source_derivative_if_requested,
)

ROOT = Path(__file__).resolve().parents[2]
JSON_HANDLER_PATH = ROOT / "webapp/parser/handlers/formats/json_handler.py"
FORMAT_ROUTER_PATH = ROOT / "webapp/parser/utils/format_router.py"
EXISTING_JSON_TEST_PATH = ROOT / "webapp/tests/test_json_parser_observation_handoff.py"
BASE_INNER_NORMALIZED_SOURCE_SHA256 = (
    "4b619ce0479eab2658044c41b5204e0f975a7ecf66123a0ef5685042b310438e"
)


class _ExplodesOnTouch:
    def __str__(self):
        raise AssertionError("dormant observer touched input")


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _function(path: Path, name: str) -> ast.FunctionDef:
    source = _source(path)
    tree = ast.parse(source, filename=str(path))
    rows = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(rows) == 1
    return rows[0]


def test_none_callback_is_dormant_before_input_validation():
    bomb = _ExplodesOnTouch()
    assert observe_json_source_derivative_if_requested(
        emit_func=None,
        decoded_json=bomb,
        producer=bomb,  # type: ignore[arg-type]
        source_document_sha256=bomb,  # type: ignore[arg-type]
    ) is None


def test_hash_count_only_observation_excludes_raw_values_and_keys():
    secret_key = "SECRET_KEY_81273"
    secret_value = "SECRET_VALUE_99127"
    decoded = {secret_key: [{"candidate": secret_value, "votes": 7}, None, True]}
    emitted = []
    observation = observe_json_source_derivative_if_requested(
        emit_func=emitted.append,
        decoded_json=decoded,
        producer="json_handler.parse",
        source_document_sha256="a" * 64,
    )
    assert observation is not None
    assert emitted == [observation]
    assert observation["contract"] == CONTRACT
    assert observation["fingerprint_scheme"] == FINGERPRINT_SCHEME
    assert observation["source_document_sha256"] == "a" * 64
    assert observation["object_count"] == 2
    assert observation["array_count"] == 1
    assert observation["key_count"] == 3
    assert observation["null_count"] == 1
    assert observation["boolean_count"] == 1
    assert observation["number_count"] == 1
    assert observation["string_count"] == 1
    assert observation["raw_json_values_included"] is False
    assert observation["raw_object_keys_included"] is False
    serialized = json.dumps(observation, sort_keys=True)
    assert secret_key not in serialized
    assert secret_value not in serialized


def test_canonical_key_order_is_stable():
    one = observe_json_source_derivative_if_requested(
        emit_func=lambda _payload: None,
        decoded_json={"b": 2, "a": {"y": 4, "x": 3}},
        producer="json_handler.parse",
        source_document_sha256=None,
    )
    two = observe_json_source_derivative_if_requested(
        emit_func=lambda _payload: None,
        decoded_json={"a": {"x": 3, "y": 4}, "b": 2},
        producer="json_handler.parse",
        source_document_sha256=None,
    )
    assert one is not None and two is not None
    assert one["derivative_sha256"] == two["derivative_sha256"]
    assert one["derivative_byte_count"] == two["derivative_byte_count"]


def test_unknown_identity_stays_unknown():
    observation = observe_json_source_derivative_if_requested(
        emit_func=lambda _payload: None,
        decoded_json={"results": []},
        producer="json_handler.parse",
        source_document_sha256=None,
    )
    assert observation is not None
    assert observation["source_document_sha256"] is None
    assert observation["source_identity_present"] is False


def test_invalid_identity_and_nonfinite_rejected():
    with pytest.raises(JsonSourceDerivativeObservationError):
        observe_json_source_derivative_if_requested(
            emit_func=lambda _payload: None,
            decoded_json={},
            producer="json_handler.parse",
            source_document_sha256="A" * 64,
        )
    with pytest.raises(JsonSourceDerivativeObservationError, match="non-finite"):
        observe_json_source_derivative_if_requested(
            emit_func=lambda _payload: None,
            decoded_json={"bad": float("nan")},
            producer="json_handler.parse",
            source_document_sha256=None,
        )


def test_callback_failure_is_wrapped():
    def explode(_payload):
        raise TypeError("SECRET CALLBACK DETAIL")
    with pytest.raises(
        JsonSourceDerivativeObservationError,
        match="observation callback failed",
    ) as exc_info:
        observe_json_source_derivative_if_requested(
            emit_func=explode,
            decoded_json={"ok": True},
            producer="json_handler.parse",
            source_document_sha256=None,
        )
    assert "SECRET CALLBACK DETAIL" not in str(exc_info.value)


def test_wrapper_emits_source_derivative_before_inner_parse(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    source = tmp_path / "fixture.json"
    source.write_text('{"results": []}', encoding="utf-8")
    expected = (["Precinct"], [{"Precinct": "P-1"}], "Contest", {"handler": "json_handler"})
    events = []
    callback = lambda _payload: None
    identity = ArtifactIdentityHandoff("b" * 64)

    def fake_observe(**kwargs):
        events.append(("observe", kwargs))
        return {"contract": CONTRACT}

    def fake_parse(*args, **kwargs):
        events.append(("parse", args, kwargs))
        return expected

    monkeypatch.setattr(
        json_handler,
        "_observe_json_source_derivative_path_if_requested",
        fake_observe,
    )
    monkeypatch.setattr(json_handler, "parse_json_election_results", fake_parse)

    actual = json_handler.parse(
        manual_file=str(source),
        artifact_identity=identity,
        json_source_observation_emit_func=callback,
    )
    assert actual is expected
    assert events[0][0] == "observe"
    assert events[0][1]["source_document_sha256"] == "b" * 64
    assert events[1][0] == "parse"


def test_source_callback_absent_remains_dormant(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    source = tmp_path / "fixture.json"
    source.write_text("{}", encoding="utf-8")
    expected = (["Precinct"], [], "Contest", {"handler": "json_handler"})
    def should_not_run(**kwargs):
        raise AssertionError("source derivative helper must remain dormant")
    monkeypatch.setattr(
        json_handler,
        "_observe_json_source_derivative_path_if_requested",
        should_not_run,
    )
    monkeypatch.setattr(
        json_handler,
        "parse_json_election_results",
        lambda *args, **kwargs: expected,
    )
    assert json_handler.parse(manual_file=str(source)) is expected


def test_provided_tables_source_callback_fails_closed(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        json_handler,
        "robust_table_extraction",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("provided_tables extraction must not start")
        ),
    )
    with pytest.raises(
        RuntimeError,
        match="json source derivative observation callback is unavailable",
    ):
        json_handler.parse(
            html_context={"provided_tables": [{"Precinct": "P-1"}]},
            json_source_observation_emit_func=lambda _payload: None,
        )


def test_inner_parser_and_generic_router_remain_unchanged_contractually():
    source = _source(JSON_HANDLER_PATH)
    inner = _function(JSON_HANDLER_PATH, "parse_json_election_results")
    segment = ast.get_source_segment(source, inner) or ""
    assert hashlib.sha256(segment.encode("utf-8")).hexdigest() == (
        BASE_INNER_NORMALIZED_SOURCE_SHA256
    )
    assert "json_source_observation_emit_func" not in segment

    wrapper = _function(JSON_HANDLER_PATH, "parse")
    wrapper_segment = ast.get_source_segment(source, wrapper) or ""
    assert "json_source_observation_emit_func" in wrapper_segment
    assert "_observe_json_source_derivative_path_if_requested" in wrapper_segment
    assert "orjson.loads" not in wrapper_segment

    router = _source(FORMAT_ROUTER_PATH)
    assert "json_source_observation_emit_func" not in router
    assert "**handler_kwargs" in router

    existing = _source(EXISTING_JSON_TEST_PATH)
    assert "json_source_observation_emit_func" not in existing
    assert "BASE_INNER_NORMALIZED_SOURCE_SHA256" in existing
