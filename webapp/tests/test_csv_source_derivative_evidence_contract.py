from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from webapp.parser.handlers.formats import csv_handler
from webapp.parser.handlers.states.pennsylvania import pennsylvania
from webapp.parser.services.csv_source_derivative_evidence import (
    CONTRACT,
    DERIVATIVE_FINGERPRINT_SCHEME,
    SOURCE_IDENTITY_SEMANTICS,
    SURFACE,
    CsvSourceDerivativeObservationError,
    observe_csv_source_derivative_if_requested,
)


ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "webapp/parser/handlers/formats/csv_handler.py"
PA_PATH = ROOT / "webapp/parser/handlers/states/pennsylvania/pennsylvania.py"
HTML_PATH = ROOT / "webapp/parser/html_election_parser.py"
WEB_PIPELINE_PATH = ROOT / "webapp/parser/web_pipeline.py"
SHARED_LOGIC_PATH = ROOT / "webapp/parser/utils/shared_logic.py"


def _capture():
    items = []
    return items, items.append


def _canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _tree(path):
    source = path.read_text(encoding="utf-8")
    return source, ast.parse(source, filename=str(path))


def _fn(tree, name):
    rows = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(rows) == 1
    return rows[0]


def _leaf(call):
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return ""


def _calls(fn, leaf):
    return [
        node for node in ast.walk(fn)
        if isinstance(node, ast.Call) and _leaf(node) == leaf
    ]


def test_contract_constants_are_exact():
    assert CONTRACT == "csv_source_derivative_observation_v1"
    assert SURFACE == "CSV_SOURCE_DERIVATIVE"
    assert (
        DERIVATIVE_FINGERPRINT_SCHEME
        == "SHA256_CANONICAL_JSON_OF_DECODED_CSV_SOURCE_V1"
    )
    assert SOURCE_IDENTITY_SEMANTICS == "SHA256_OF_IMMUTABLE_CONTENT_BYTES"


def test_none_callback_is_dormant_before_input_validation():
    assert observe_csv_source_derivative_if_requested(
        emit_func=None,
        decoded_headers=object(),
        decoded_rows=object(),
        encoding=object(),
        producer=object(),
        source_document_sha256=object(),
    ) is None


def test_hash_only_observation_and_source_linkage():
    items, emit = _capture()
    result = observe_csv_source_derivative_if_requested(
        emit_func=emit,
        decoded_headers=["Precinct", "Candidate", "Votes"],
        decoded_rows=[
            {"Precinct": "P1", "Candidate": "Alice Example", "Votes": "12"}
        ],
        encoding="utf-8",
        producer="csv_handler.parse_csv_election_results",
        source_document_sha256="a" * 64,
    )
    assert items == [result]
    assert result["source_document_sha256"] == "a" * 64
    assert result["header_count"] == 3
    assert result["row_count"] == 1
    assert result["raw_header_values_included"] is False
    assert result["raw_row_values_included"] is False
    assert result["canonical_authority"] is False
    assert result["new_persistence"] is False
    serialized = json.dumps(result, sort_keys=True)
    assert "Alice Example" not in serialized
    assert '"P1"' not in serialized
    assert '"Votes"' not in serialized


def test_derivative_fingerprint_is_canonical_decoded_projection():
    items, emit = _capture()
    result = observe_csv_source_derivative_if_requested(
        emit_func=emit,
        decoded_headers=["A", "B"],
        decoded_rows=[
            {"A": "1", "B": "2"},
            {"A": "", "B": None},
        ],
        encoding="utf-8",
        producer="pennsylvania.parse",
    )
    expected = {
        "headers": ["A", "B"],
        "rows": [
            [["A", "1"], ["B", "2"]],
            [["A", ""], ["B", None]],
        ],
    }
    encoded = _canonical(expected)
    assert result["derivative_sha256"] == hashlib.sha256(encoded).hexdigest()
    assert result["derivative_byte_count"] == len(encoded)
    assert result["nonempty_row_count"] == 1


def test_observation_hash_is_canonical_core_hash():
    items, emit = _capture()
    result = observe_csv_source_derivative_if_requested(
        emit_func=emit,
        decoded_headers=[],
        decoded_rows=[],
        encoding="utf-8",
        producer="csv_handler.parse_csv_election_results",
    )
    core = dict(result)
    observed = core.pop("observation_sha256")
    assert observed == hashlib.sha256(_canonical(core)).hexdigest()


def test_invalid_source_identity_rejected():
    items, emit = _capture()
    with pytest.raises(CsvSourceDerivativeObservationError, match="SHA-256"):
        observe_csv_source_derivative_if_requested(
            emit_func=emit,
            decoded_headers=[],
            decoded_rows=[],
            encoding="utf-8",
            producer="csv_handler.parse_csv_election_results",
            source_document_sha256="bad",
        )


def test_callback_failure_is_redacted():
    def explode(_payload):
        raise RuntimeError("SECRET CALLBACK DETAIL")

    with pytest.raises(
        CsvSourceDerivativeObservationError,
        match="observation callback failed",
    ) as excinfo:
        observe_csv_source_derivative_if_requested(
            emit_func=explode,
            decoded_headers=[],
            decoded_rows=[],
            encoding="utf-8",
            producer="csv_handler.parse_csv_election_results",
        )
    assert "SECRET CALLBACK DETAIL" not in str(excinfo.value)


def test_csv_primary_source_observer_is_before_normalization():
    _, tree = _tree(CSV_PATH)
    primary = _fn(tree, "parse_csv_election_results")
    readers = _calls(primary, "DictReader")
    observers = _calls(primary, "observe_csv_source_derivative_if_requested")
    normalizers = _calls(primary, "normalize_table_headers")
    assert len(readers) == len(observers) == len(normalizers) == 1
    assert readers[0].lineno < observers[0].lineno < normalizers[0].lineno


def test_csv_existing_typed_and_generic_result_observation_order_remains():
    _, tree = _tree(CSV_PATH)
    primary = _fn(tree, "parse_csv_election_results")
    typed = _calls(primary, "build_table_noninteractive_result")
    generic = _calls(primary, "emit_parser_observation_bundle_if_requested")
    final = _calls(primary, "finalize_election_output")
    assert len(typed) == len(generic) == len(final) == 1
    assert typed[0].lineno < generic[0].lineno < final[0].lineno


def test_csv_wrapper_consumes_identity_but_inner_only_receives_scalar():
    signature = inspect.signature(csv_handler.parse)
    inner_signature = inspect.signature(csv_handler.parse_csv_election_results)
    assert "artifact_identity" in signature.parameters
    assert signature.parameters["artifact_identity"].default is None
    assert "artifact_identity" not in inner_signature.parameters
    assert "csv_source_sha256" in inner_signature.parameters
    assert "csv_source_observation_emit_func" in inner_signature.parameters

    source, tree = _tree(CSV_PATH)
    wrapper = _fn(tree, "parse")
    handoff = _calls(wrapper, "parse_csv_election_results")
    assert len(handoff) == 1
    kw = {item.arg: item.value for item in handoff[0].keywords}
    rendered = ast.unparse(kw["csv_source_sha256"])
    assert "artifact_identity.document_sha256" in rendered
    assert "artifact_identity is not None" in rendered
    wrapper_source = ast.get_source_segment(source, wrapper) or ""
    assert "file_hash(" not in wrapper_source


def test_csv_provided_tables_source_observer_fails_closed():
    with pytest.raises(
        RuntimeError,
        match="CSV source derivative observation callback",
    ):
        csv_handler.parse(
            html_context={"provided_tables": [{"Candidate": "A"}]},
            csv_source_observation_emit_func=lambda payload: None,
        )


def test_pennsylvania_has_parallel_source_seam_before_totals():
    source, tree = _tree(PA_PATH)
    parse_fn = _fn(tree, "parse")
    readers = _calls(parse_fn, "DictReader")
    observers = _calls(parse_fn, "observe_csv_source_derivative_if_requested")
    assert len(readers) == len(observers) == 1
    assert readers[0].lineno < observers[0].lineno
    parse_source = ast.get_source_segment(source, parse_fn) or ""
    assert parse_source.index(
        "observe_csv_source_derivative_if_requested("
    ) < parse_source.index(
        "# Compute a grand total row for numeric columns"
    )
    assert "file_hash(" not in parse_source


def test_pennsylvania_identity_is_upstream_scalar_only():
    signature = inspect.signature(pennsylvania.parse)
    assert "artifact_identity" in signature.parameters
    assert signature.parameters["artifact_identity"].default is None
    source, tree = _tree(PA_PATH)
    parse_fn = _fn(tree, "parse")
    observer = _calls(parse_fn, "observe_csv_source_derivative_if_requested")[0]
    kw = {item.arg: item.value for item in observer.keywords}
    rendered = ast.unparse(kw["source_document_sha256"])
    assert "artifact_identity.document_sha256" in rendered
    assert "artifact_identity is not None" in rendered


def test_generic_router_layers_remain_unmodified_by_csv_specific_callback():
    for path in (HTML_PATH, WEB_PIPELINE_PATH, SHARED_LOGIC_PATH):
        text = path.read_text(encoding="utf-8")
        assert "csv_source_observation_emit_func" not in text
        assert "csv_source_derivative_evidence" not in text


def test_service_has_no_transport_file_io_or_persistence_dependency():
    service = (
        ROOT / "webapp/parser/services/csv_source_derivative_evidence.py"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "requests",
        "urllib",
        "socketio",
        "sqlalchemy",
        "psycopg",
        "file_hash(",
        "finalize_election_output",
    ):
        assert forbidden not in service
