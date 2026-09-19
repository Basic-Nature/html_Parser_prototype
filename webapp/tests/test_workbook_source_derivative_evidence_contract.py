from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from webapp.parser.handlers.formats import xlsx_handler
from webapp.parser.services.workbook_source_derivative_evidence import (
    CONTRACT,
    DERIVATIVE_FINGERPRINT_SCHEME,
    SOURCE_IDENTITY_SEMANTICS,
    SUPPORTED_PRODUCERS,
    SURFACE,
    WorkbookSourceDerivativeObservationError,
    observe_workbook_source_derivative_if_requested,
)

ROOT = Path(__file__).resolve().parents[2]
FEC_PATH = ROOT / "webapp/parser/handlers/fec_handler.py"
XLSX_PATH = ROOT / "webapp/parser/handlers/formats/xlsx_handler.py"
ROUTER_PATH = ROOT / "webapp/parser/utils/format_router.py"
SERVICE_PATH = ROOT / "webapp/parser/services/workbook_source_derivative_evidence.py"

class _Frame:
    def __init__(self, columns, records):
        self.columns = list(columns)
        self._records = [dict(row) for row in records]
    def to_dict(self, *, orient):
        assert orient == "records"
        return [dict(row) for row in self._records]

def _capture():
    rows = []
    return rows, rows.append

def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")

def _tree(path):
    source = path.read_text(encoding="utf-8")
    return source, ast.parse(source, filename=str(path))

def _fn(tree, name):
    rows = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
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
    assert CONTRACT == "workbook_source_derivative_observation_v1"
    assert SURFACE == "WORKBOOK_SOURCE_DERIVATIVE"
    assert DERIVATIVE_FINGERPRINT_SCHEME == "SHA256_CANONICAL_JSON_OF_DECODED_WORKBOOK_SOURCE_V1"
    assert SOURCE_IDENTITY_SEMANTICS == "SHA256_OF_IMMUTABLE_CONTENT_BYTES"
    assert SUPPORTED_PRODUCERS == frozenset({
        "fec_handler.parse",
        "xlsx_handler.parse_xlsx_election_results",
        "format_router.prompt_and_handle_download",
    })

def test_none_callback_is_dormant_before_input_validation():
    assert observe_workbook_source_derivative_if_requested(
        emit_func=None,
        decoded_frame=object(),
        producer=object(),
        sheet_name=object(),
        source_document_sha256=object(),
    ) is None

def test_hash_only_observation_and_source_linkage():
    items, emit = _capture()
    result = observe_workbook_source_derivative_if_requested(
        emit_func=emit,
        decoded_frame=_Frame(
            ["Precinct", "Candidate", "Votes"],
            [{"Precinct": "P1", "Candidate": "Alice Example", "Votes": 12}],
        ),
        producer="xlsx_handler.parse_xlsx_election_results",
        sheet_name="Official Results",
        source_document_sha256="a" * 64,
    )
    assert items == [result]
    assert result["source_document_sha256"] == "a" * 64
    assert result["column_count"] == 3
    assert result["row_count"] == 1
    assert result["nonempty_row_count"] == 1
    assert result["canonical_authority"] is False
    assert result["new_persistence"] is False
    assert result["raw_sheet_name_included"] is False
    assert result["raw_column_values_included"] is False
    assert result["raw_cell_values_included"] is False
    serialized = json.dumps(result, sort_keys=True)
    assert "Official Results" not in serialized
    assert "Alice Example" not in serialized
    assert '"P1"' not in serialized
    assert '"Votes"' not in serialized

def test_derivative_fingerprint_is_deterministic_for_column_ordered_projection():
    frame_a = _Frame(["A", "B"], [{"A": "1", "B": "2"}, {"A": "", "B": None}])
    frame_b = _Frame(["A", "B"], [{"B": "2", "A": "1"}, {"B": None, "A": ""}])
    _, emit_a = _capture()
    _, emit_b = _capture()
    first = observe_workbook_source_derivative_if_requested(
        emit_func=emit_a, decoded_frame=frame_a,
        producer="fec_handler.parse", sheet_name=0,
    )
    second = observe_workbook_source_derivative_if_requested(
        emit_func=emit_b, decoded_frame=frame_b,
        producer="fec_handler.parse", sheet_name=0,
    )
    assert first["derivative_sha256"] == second["derivative_sha256"]
    assert first["derivative_byte_count"] == second["derivative_byte_count"]
    assert first["nonempty_row_count"] == 1

def test_observation_hash_is_canonical_core_hash():
    items, emit = _capture()
    result = observe_workbook_source_derivative_if_requested(
        emit_func=emit,
        decoded_frame=_Frame([], []),
        producer="format_router.prompt_and_handle_download",
        sheet_name=0,
    )
    core = dict(result)
    observed = core.pop("observation_sha256")
    assert observed == hashlib.sha256(_canonical(core)).hexdigest()
    assert items == [result]

def test_missing_source_identity_remains_unknown():
    _, emit = _capture()
    result = observe_workbook_source_derivative_if_requested(
        emit_func=emit,
        decoded_frame=_Frame(["A"], [{"A": "1"}]),
        producer="format_router.prompt_and_handle_download",
        sheet_name=0,
        source_document_sha256=None,
    )
    assert result["source_document_sha256"] is None

def test_invalid_source_identity_rejected():
    _, emit = _capture()
    with pytest.raises(WorkbookSourceDerivativeObservationError, match="SHA-256"):
        observe_workbook_source_derivative_if_requested(
            emit_func=emit,
            decoded_frame=_Frame([], []),
            producer="fec_handler.parse",
            source_document_sha256="bad",
        )

def test_callback_failure_is_redacted():
    def explode(_payload):
        raise RuntimeError("SECRET CALLBACK DETAIL")
    with pytest.raises(
        WorkbookSourceDerivativeObservationError,
        match="observation callback failed",
    ) as excinfo:
        observe_workbook_source_derivative_if_requested(
            emit_func=explode,
            decoded_frame=_Frame([], []),
            producer="xlsx_handler.parse_xlsx_election_results",
        )
    assert "SECRET CALLBACK DETAIL" not in str(excinfo.value)

def test_fec_direct_reader_observer_is_before_header_normalization():
    source, tree = _tree(FEC_PATH)
    parse_fn = _fn(tree, "parse")
    readers = _calls(parse_fn, "read_excel")
    observers = _calls(parse_fn, "observe_workbook_source_derivative_if_requested")
    canonicalizers = _calls(parse_fn, "canonicalize_headers")
    assert len(readers) == 1
    assert len(observers) == 1
    assert canonicalizers
    assert readers[0].lineno < observers[0].lineno < canonicalizers[0].lineno
    rendered = ast.get_source_segment(source, observers[0]) or ""
    assert "artifact_identity.document_sha256" in rendered
    assert 'producer="fec_handler.parse"' in rendered

def test_xlsx_primary_reader_observer_is_before_dataframe_normalization():
    source, tree = _tree(XLSX_PATH)
    primary = _fn(tree, "parse_xlsx_election_results")
    readers = _calls(primary, "read_excel")
    observers = _calls(primary, "observe_workbook_source_derivative_if_requested")
    normalizers = _calls(primary, "_dataframe_to_records")
    assert len(readers) == len(observers) == len(normalizers) == 1
    assert readers[0].lineno < observers[0].lineno < normalizers[0].lineno
    rendered = ast.get_source_segment(source, observers[0]) or ""
    assert "workbook_source_sha256" in rendered
    assert 'producer="xlsx_handler.parse_xlsx_election_results"' in rendered

def test_xlsx_wrapper_consumes_trusted_identity_as_scalar_only():
    signature = inspect.signature(xlsx_handler.parse)
    inner_signature = inspect.signature(xlsx_handler.parse_xlsx_election_results)
    assert "artifact_identity" in signature.parameters
    assert signature.parameters["artifact_identity"].default is None
    assert "artifact_identity" not in inner_signature.parameters
    assert "workbook_source_sha256" in inner_signature.parameters
    assert "workbook_source_observation_emit_func" in inner_signature.parameters
    source, tree = _tree(XLSX_PATH)
    wrapper = _fn(tree, "parse")
    handoff = _calls(wrapper, "parse_xlsx_election_results")
    assert len(handoff) == 1
    kw = {item.arg: item.value for item in handoff[0].keywords}
    rendered = ast.unparse(kw["workbook_source_sha256"])
    assert "artifact_identity.document_sha256" in rendered
    assert "artifact_identity is not None" in rendered
    wrapper_source = ast.get_source_segment(source, wrapper) or ""
    assert "file_hash(" not in wrapper_source

def test_xlsx_provided_tables_workbook_observer_fails_closed():
    with pytest.raises(
        RuntimeError,
        match="workbook source derivative observation callback",
    ):
        xlsx_handler.parse(
            html_context={"provided_tables": [{"Candidate": "A"}]},
            workbook_source_observation_emit_func=lambda payload: None,
        )

def test_router_header_probe_is_observed_without_fabricated_identity():
    source, tree = _tree(ROUTER_PATH)
    fn = _fn(tree, "prompt_and_handle_download")
    readers = _calls(fn, "read_excel")
    observers = _calls(fn, "observe_workbook_source_derivative_if_requested")
    assert len(readers) == len(observers) == 1
    assert readers[0].lineno < observers[0].lineno
    keywords = {item.arg: item.value for item in observers[0].keywords}
    assert isinstance(keywords["source_document_sha256"], ast.Constant)
    assert keywords["source_document_sha256"].value is None
    rendered = ast.get_source_segment(source, observers[0]) or ""
    assert 'producer="format_router.prompt_and_handle_download"' in rendered

def test_router_dispatch_and_url_registry_boundaries_remain_present():
    text = ROUTER_PATH.read_text(encoding="utf-8")
    assert "route_format_handler(" in text
    assert "_build_download_url(" in text
    assert "download_file(" in text
    assert "url_registry" not in SERVICE_PATH.read_text(encoding="utf-8")

def test_service_has_no_transport_file_io_or_persistence_dependency():
    service = SERVICE_PATH.read_text(encoding="utf-8")
    for forbidden in (
        "requests", "urllib", "socketio", "sqlalchemy", "psycopg",
        "file_hash(", "finalize_election_output", "Path(", "open(",
    ):
        assert forbidden not in service
