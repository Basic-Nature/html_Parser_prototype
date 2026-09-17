from __future__ import annotations

import ast
from pathlib import Path

import pytest

import webapp.parser.handlers.formats.pdf_handler as pdf_handler
from webapp.parser.contracts.artifact_identity import ArtifactIdentityHandoff
from webapp.parser.contracts.table_pipeline import TablePipelineResult, TableStage
from webapp.parser.services.parser_observation_bundle import (
    project_parser_observation_bundle,
)
from webapp.parser.services.parser_result_observation_adapter import (
    PARSER_RESULT_OBSERVATION_ADAPTER_CONTRACT,
    adapt_final_parser_result_for_observation,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PDF_PATH = REPO_ROOT / "webapp/parser/handlers/formats/pdf_handler.py"
ADAPTER_PATH = (
    REPO_ROOT
    / "webapp/parser/services/parser_result_observation_adapter.py"
)


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _tree(path: Path) -> ast.Module:
    return ast.parse(_source(path), filename=str(path))


def _fn(tree: ast.AST, name: str) -> ast.FunctionDef:
    rows = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == name
    ]
    assert len(rows) == 1
    return rows[0]


def test_adapter_is_container_only_interpreted_noncanonical_boundary():
    headers = ["Precinct", "Candidate A - Total Votes"]
    rows = [
        {"Precinct": "P-1", "Candidate A - Total Votes": 0},
        {"Precinct": "P-2", "Candidate A - Total Votes": -4},
        {"Precinct": "P-3", "Candidate A - Total Votes": None},
    ]

    result = adapt_final_parser_result_for_observation(
        headers,
        rows,
        source_type="pdf",
        source_sha256="a" * 64,
    )

    assert isinstance(result, TablePipelineResult)
    assert result.stage is TableStage.INTERPRETED
    assert list(result.headers) == headers
    assert [dict(row) for row in result.rows] == rows
    assert result.source_provenance.source_type == "pdf"
    assert result.source_provenance.source_sha256 == "a" * 64
    assert result.source_provenance.source_uri is None
    assert result.write_kind.value == "none"
    assert len(result.transformations) == 1
    record = result.transformations[0]
    assert record.operation == "final_parser_result_observation_adaptation"
    assert record.details["semantic_value_mutation"] is False


def test_adapter_does_not_hash_or_invent_missing_identity():
    result = adapt_final_parser_result_for_observation(
        ["Precinct"],
        [{"Precinct": "P-1"}],
        source_type="pdf",
    )
    assert result.source_provenance.source_sha256 is None

    source = _source(ADAPTER_PATH)
    for prohibited in (
        "hashlib",
        "sha256_file",
        "compute_sha256",
        "open(",
        "Path(",
        "datetime",
        "time.time",
        "finalize_election_output",
        "build_table_noninteractive",
        "socketio",
    ):
        assert prohibited not in source


def test_adapter_rejects_non_table_inputs():
    with pytest.raises(TypeError):
        adapt_final_parser_result_for_observation(
            "not-a-header-sequence",
            [],
            source_type="pdf",
        )

    with pytest.raises(TypeError):
        adapt_final_parser_result_for_observation(
            ["Precinct"],
            ["not-a-row-mapping"],
            source_type="pdf",
        )


def test_raw_pdf_text_row_never_enters_observation_bundle():
    secret = "SECRET RAW PDF PAGE TEXT 987654321"
    result = adapt_final_parser_result_for_observation(
        ["text"],
        [{"text": secret}],
        source_type="pdf",
        source_sha256="b" * 64,
    )
    payload = project_parser_observation_bundle(result)

    assert payload["contract"] == "parser_observation_bundle_v1"
    assert payload["authority"]["canonical"] is False
    assert payload["raw_rows_included"] is False
    assert payload["raw_headers_included"] is False
    assert payload["automatic_timestamp"] is False
    assert secret not in repr(payload)


def test_pdf_wrapper_emits_after_parse_without_changing_return(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    pdf_path = tmp_path / "fixture.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%fixture\n")

    expected = (
        ["Precinct", "Candidate A - Total Votes"],
        [{"Precinct": "P-1", "Candidate A - Total Votes": 7}],
        "Contest A",
        {"handler": "pdf_handler", "marker": "unchanged"},
    )
    parse_calls = []
    adapt_calls = []
    emit_calls = []
    typed_sentinel = object()
    callback = lambda _payload: None
    identity = ArtifactIdentityHandoff("c" * 64)

    def fake_parse(*args, **kwargs):
        parse_calls.append((args, kwargs))
        return expected

    def fake_adapt(headers, rows, **kwargs):
        adapt_calls.append((headers, rows, kwargs))
        return typed_sentinel

    def fake_emit(result, **kwargs):
        emit_calls.append((result, kwargs))
        return True

    monkeypatch.setattr(
        pdf_handler,
        "parse_pdf_election_results",
        fake_parse,
    )
    monkeypatch.setattr(
        pdf_handler,
        "adapt_final_parser_result_for_observation",
        fake_adapt,
    )
    monkeypatch.setattr(
        pdf_handler,
        "emit_parser_observation_bundle_if_requested",
        fake_emit,
    )

    actual = pdf_handler.parse(
        manual_file=str(pdf_path),
        artifact_identity=identity,
        parser_observation_emit_func=callback,
    )

    assert actual is expected
    assert len(parse_calls) == 1
    assert parse_calls[0][1]["artifact_identity"] is identity
    assert "parser_observation_emit_func" not in parse_calls[0][1]

    assert adapt_calls == [
        (
            expected[0],
            expected[1],
            {
                "source_type": "pdf",
                "source_sha256": "c" * 64,
            },
        )
    ]
    assert emit_calls == [
        (
            typed_sentinel,
            {"parser_observation_emit_func": callback},
        )
    ]


def test_pdf_wrapper_does_not_adapt_when_callback_absent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    pdf_path = tmp_path / "fixture.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%fixture\n")

    expected = (
        ["Precinct"],
        [{"Precinct": "P-1"}],
        "Contest A",
        {"handler": "pdf_handler"},
    )

    monkeypatch.setattr(
        pdf_handler,
        "parse_pdf_election_results",
        lambda *args, **kwargs: expected,
    )

    def should_not_run(*args, **kwargs):
        raise AssertionError("observation adapter must remain dormant")

    monkeypatch.setattr(
        pdf_handler,
        "adapt_final_parser_result_for_observation",
        should_not_run,
    )

    assert pdf_handler.parse(manual_file=str(pdf_path)) is expected


def test_pdf_provided_tables_callback_fails_closed_before_extraction():
    with pytest.raises(
        RuntimeError,
        match="parser observation callback is unavailable",
    ):
        pdf_handler.parse(
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


def test_pdf_inner_parser_remains_observation_transport_free():
    source = _source(PDF_PATH)
    tree = _tree(PDF_PATH)
    inner = _fn(tree, "parse_pdf_election_results")
    segment = ast.get_source_segment(source, inner) or ""

    assert "parser_observation_emit_func" not in segment
    assert "adapt_final_parser_result_for_observation" not in segment
    assert "emit_parser_observation_bundle_if_requested" not in segment


def test_pdf_wrapper_uses_existing_identity_only_and_no_raw_metadata():
    source = _source(PDF_PATH)
    tree = _tree(PDF_PATH)
    wrapper = _fn(tree, "parse")
    segment = ast.get_source_segment(source, wrapper) or ""

    assert "parser_observation_emit_func" in segment
    assert "artifact_identity.document_sha256" in segment
    assert 'source_type="pdf"' in segment
    assert "hashlib" not in segment
    assert "_record_parse_observation" not in segment
    assert "page_text_map" not in segment
    assert "all_text" not in segment
    assert "clean_text" not in segment


def test_adapter_contract_constant_is_exact():
    assert (
        PARSER_RESULT_OBSERVATION_ADAPTER_CONTRACT
        == "parser_result_observation_adapter_v1"
    )
