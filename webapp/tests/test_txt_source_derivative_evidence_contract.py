from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from webapp.parser.handlers.formats import txt_handler
from webapp.parser.services.txt_source_derivative_evidence import (
    CONTRACT,
    DERIVATIVE_FINGERPRINT_SCHEME,
    SOURCE_IDENTITY_SEMANTICS,
    SURFACE,
    TxtSourceDerivativeObservationError,
    observe_txt_source_derivative_if_requested,
)


ROOT = Path(__file__).resolve().parents[2]
TXT_PATH = ROOT / "webapp/parser/handlers/formats/txt_handler.py"
ROUTER_PATH = ROOT / "webapp/parser/utils/format_router.py"
CSV_PATH = ROOT / "webapp/parser/handlers/formats/csv_handler.py"
FEC_PATH = ROOT / "webapp/parser/handlers/fec_handler.py"
WEBAPP_PATH = ROOT / "webapp/Smart_Elections_Parser_Webapp.py"
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
    assert CONTRACT == "txt_source_derivative_observation_v1"
    assert SURFACE == "TXT_DELIMITED_SOURCE_DERIVATIVE"
    assert (
        DERIVATIVE_FINGERPRINT_SCHEME
        == "SHA256_CANONICAL_JSON_OF_DECODED_TXT_DELIMITED_SOURCE_V1"
    )
    assert SOURCE_IDENTITY_SEMANTICS == "SHA256_OF_IMMUTABLE_CONTENT_BYTES"


def test_none_callback_is_dormant_before_input_validation():
    assert observe_txt_source_derivative_if_requested(
        emit_func=None,
        decoded_headers=object(),
        decoded_rows=object(),
        encoding=object(),
        delimiter=object(),
        producer=object(),
        source_document_sha256=object(),
    ) is None


def test_hash_only_observation_and_source_linkage():
    items, emit = _capture()
    result = observe_txt_source_derivative_if_requested(
        emit_func=emit,
        decoded_headers=["Precinct", "Candidate", "Votes"],
        decoded_rows=[
            {"Precinct": "P1", "Candidate": "Alice Example", "Votes": "12"}
        ],
        encoding="utf-8",
        delimiter="\t",
        producer="txt_handler.parse_txt_election_results",
        source_document_sha256="a" * 64,
    )
    assert items == [result]
    assert result["source_document_sha256"] == "a" * 64
    assert result["encoding"] == "utf-8"
    assert result["delimiter"] == "\t"
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


def test_derivative_fingerprint_includes_decode_metadata():
    items, emit = _capture()
    result = observe_txt_source_derivative_if_requested(
        emit_func=emit,
        decoded_headers=["A", "B"],
        decoded_rows=[
            {"A": "1", "B": "2"},
            {"A": "", "B": None},
        ],
        encoding="latin-1",
        delimiter="|",
        producer="txt_handler.parse_txt_election_results",
    )
    expected = {
        "encoding": "latin-1",
        "delimiter": "|",
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
    result = observe_txt_source_derivative_if_requested(
        emit_func=emit,
        decoded_headers=[],
        decoded_rows=[],
        encoding="utf-8",
        delimiter=",",
        producer="txt_handler.parse_txt_election_results",
    )
    core = dict(result)
    observed = core.pop("observation_sha256")
    assert observed == hashlib.sha256(_canonical(core)).hexdigest()


def test_invalid_source_identity_rejected():
    items, emit = _capture()
    with pytest.raises(TxtSourceDerivativeObservationError, match="SHA-256"):
        observe_txt_source_derivative_if_requested(
            emit_func=emit,
            decoded_headers=[],
            decoded_rows=[],
            encoding="utf-8",
            delimiter=",",
            producer="txt_handler.parse_txt_election_results",
            source_document_sha256="bad",
        )


def test_callback_failure_is_redacted():
    def explode(_payload):
        raise RuntimeError("SECRET CALLBACK DETAIL")

    with pytest.raises(
        TxtSourceDerivativeObservationError,
        match="observation callback failed",
    ) as excinfo:
        observe_txt_source_derivative_if_requested(
            emit_func=explode,
            decoded_headers=[],
            decoded_rows=[],
            encoding="utf-8",
            delimiter=",",
            producer="txt_handler.parse_txt_election_results",
        )
    assert "SECRET CALLBACK DETAIL" not in str(excinfo.value)


def test_reader_metadata_helper_preserves_existing_reader_behavior(tmp_path):
    source = tmp_path / "sample.txt"
    source.write_text(
        "Precinct\tCandidate\tVotes\nP1\tAlice\t12\n",
        encoding="utf-8",
    )
    legacy = txt_handler._read_delimited_file(str(source))
    headers, rows, encoding, delimiter = (
        txt_handler._read_delimited_file_with_metadata(str(source))
    )
    assert legacy == (headers, rows)
    assert headers == ["Precinct", "Candidate", "Votes"]
    assert rows == [{"Precinct": "P1", "Candidate": "Alice", "Votes": "12"}]
    assert encoding == "utf-8"
    assert delimiter == "\t"


def test_source_observer_is_after_decode_before_semantics():
    _, tree = _tree(TXT_PATH)
    primary = _fn(tree, "parse_txt_election_results")
    readers = _calls(primary, "_read_delimited_file_with_metadata")
    observers = _calls(primary, "observe_txt_source_derivative_if_requested")
    semantics = _calls(primary, "gather_lines_for_contest_detection")
    assert len(readers) == len(observers) == len(semantics) == 1
    assert readers[0].lineno < observers[0].lineno < semantics[0].lineno


def test_final_parser_observation_order_is_typed_result_then_emit_then_finalize():
    _, tree = _tree(TXT_PATH)
    primary = _fn(tree, "parse_txt_election_results")
    typed = _calls(primary, "build_table_noninteractive_result")
    generic = _calls(primary, "emit_parser_observation_bundle_if_requested")
    final = _calls(primary, "finalize_election_output")
    assert len(typed) == len(generic) == len(final) == 1
    assert typed[0].lineno < generic[0].lineno < final[0].lineno


def test_wrapper_consumes_identity_but_inner_only_receives_scalar():
    signature = inspect.signature(txt_handler.parse)
    inner_signature = inspect.signature(txt_handler.parse_txt_election_results)
    assert "artifact_identity" in signature.parameters
    assert signature.parameters["artifact_identity"].default is None
    assert "artifact_identity" not in inner_signature.parameters
    assert "txt_source_sha256" in inner_signature.parameters
    assert "txt_source_observation_emit_func" in inner_signature.parameters
    assert "parser_observation_emit_func" in inner_signature.parameters

    source, tree = _tree(TXT_PATH)
    wrapper = _fn(tree, "parse")
    handoff = _calls(wrapper, "parse_txt_election_results")
    assert len(handoff) == 1
    kw = {item.arg: item.value for item in handoff[0].keywords}
    rendered = ast.unparse(kw["txt_source_sha256"])
    assert "artifact_identity.document_sha256" in rendered
    assert "artifact_identity is not None" in rendered
    wrapper_source = ast.get_source_segment(source, wrapper) or ""
    assert "file_hash(" not in wrapper_source
    assert "hashlib" not in wrapper_source


def test_provided_tables_source_observer_fails_closed():
    with pytest.raises(
        RuntimeError,
        match="TXT source derivative observation callback",
    ):
        txt_handler.parse(
            html_context={"provided_tables": [{"Candidate": "A"}]},
            txt_source_observation_emit_func=lambda payload: None,
        )


def test_provided_tables_parser_observer_fails_closed():
    with pytest.raises(
        RuntimeError,
        match="parser observation callback",
    ):
        txt_handler.parse(
            html_context={"provided_tables": [{"Candidate": "A"}]},
            parser_observation_emit_func=lambda payload: None,
        )


def test_format_router_remains_generic_and_unmodified_for_txt_callbacks():
    router = ROUTER_PATH.read_text(encoding="utf-8")
    assert 'fmt in {"txt", "text"}' in router
    assert "return txt_handler" in router
    assert router.count("**handler_kwargs") >= 2
    assert "txt_source_observation_emit_func" not in router
    assert "txt_source_derivative_evidence" not in router


def test_neighbor_surfaces_remain_outside_txt_mutation_scope():
    for path in (
        CSV_PATH,
        FEC_PATH,
        WEBAPP_PATH,
        HTML_PATH,
        WEB_PIPELINE_PATH,
        SHARED_LOGIC_PATH,
    ):
        text = path.read_text(encoding="utf-8")
        assert "txt_source_observation_emit_func" not in text
        assert "txt_source_derivative_evidence" not in text


def test_service_has_no_transport_file_io_or_persistence_dependency():
    service = (
        ROOT / "webapp/parser/services/txt_source_derivative_evidence.py"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "requests",
        "urllib",
        "socketio",
        "sqlalchemy",
        "psycopg",
        "open(",
        "read_text(",
        "read_bytes(",
        "file_hash(",
        "finalize_election_output",
    ):
        assert forbidden not in service
