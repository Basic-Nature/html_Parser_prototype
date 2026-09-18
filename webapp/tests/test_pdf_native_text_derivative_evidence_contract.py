from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from webapp.parser.services.pdf_native_text_derivative_evidence import (
    CONTRACT,
    DERIVATIVE_FINGERPRINT_SCHEME,
    PAGE_FINGERPRINT_SCHEME,
    PdfNativeTextDerivativeObservationError,
    observe_pdf_native_text_derivative_if_requested,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PDF_PATH = REPO_ROOT / "webapp/parser/handlers/formats/pdf_handler.py"
SERVICE_PATH = (
    REPO_ROOT
    / "webapp/parser/services/pdf_native_text_derivative_evidence.py"
)


class Bomb:
    def __getattribute__(self, name):
        raise AssertionError(
            f"default-none path inspected dormant input unexpectedly: {name}"
        )


def _capture():
    items = []

    def emit(payload):
        items.append(payload)

    return items, emit


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


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


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        left = _call_name(node.value)
        return f"{left}.{node.attr}" if left else node.attr
    return ""


def test_contract_constant_exact():
    assert CONTRACT == "pdf_native_text_derivative_observation_v1"
    assert (
        DERIVATIVE_FINGERPRINT_SCHEME
        == "SHA256_UTF8_OF_SELECTED_NATIVE_TEXT_V1"
    )
    assert (
        PAGE_FINGERPRINT_SCHEME
        == "SHA256_UTF8_OF_NATIVE_PAGE_TEXT_V1"
    )



def test_default_none_returns_before_any_input_validation_hash_or_iteration():
    bomb = Bomb()
    assert observe_pdf_native_text_derivative_if_requested(
        emit_func=None,
        selected_text=bomb,
        page_text_map=bomb,
        native_text_mode=bomb,
        source_document_sha256=bomb,
    ) is None


def test_active_observer_hashes_exact_selected_text_and_pages_without_raw_text():
    items, emit = _capture()
    selected = "Precinct α\nVotes 42\n"
    page0 = "Precinct α\n"
    page1 = "Votes 42\n"

    result = observe_pdf_native_text_derivative_if_requested(
        emit_func=emit,
        selected_text=selected,
        page_text_map=[
            {"page": 0, "raw_text": page0, "char_count": len(page0)},
            {"page": 1, "raw_text": page1, "char_count": len(page1)},
        ],
        native_text_mode="text",
        source_document_sha256="a" * 64,
    )

    assert result == items[0]
    assert result["source_document_sha256"] == "a" * 64
    assert result["derivative_sha256"] == hashlib.sha256(
        selected.encode("utf-8")
    ).hexdigest()
    assert result["derivative_byte_count"] == len(selected.encode("utf-8"))
    assert result["derivative_char_count"] == len(selected)
    assert result["page_count"] == 2
    assert result["nonempty_page_count"] == 2
    assert result["page_fingerprints"][0]["sha256"] == hashlib.sha256(
        page0.encode("utf-8")
    ).hexdigest()
    assert result["page_fingerprints"][1]["sha256"] == hashlib.sha256(
        page1.encode("utf-8")
    ).hexdigest()

    rendered = json.dumps(result, sort_keys=True)
    assert "Precinct α" not in rendered
    assert "Votes 42" not in rendered
    for forbidden in (
        "pdf_path",
        "filename",
        "source_url",
        "requested_url",
        "session_id",
        "cookie",
        "headers",
    ):
        assert forbidden not in result


def test_missing_source_identity_stays_unknown_and_does_not_hash_source_pdf():
    items, emit = _capture()
    result = observe_pdf_native_text_derivative_if_requested(
        emit_func=emit,
        selected_text="x",
        page_text_map=[
            {"page": 0, "raw_text": "x", "char_count": 1},
        ],
        native_text_mode="raw",
        source_document_sha256=None,
    )
    assert result["source_document_sha256"] is None
    service = _source(SERVICE_PATH)
    for prohibited in (
        "Path(",
        "open(",
        "read_bytes",
        "read_text",
        "file_hash",
        "requests",
        "urllib",
        "socketio",
        "ArtifactIdentityHandoff",
    ):
        assert prohibited not in service

def test_invalid_source_identity_scalar_is_rejected_without_inference():
    items, emit = _capture()
    with pytest.raises(
        PdfNativeTextDerivativeObservationError,
        match="SHA-256",
    ):
        observe_pdf_native_text_derivative_if_requested(
            emit_func=emit,
            selected_text="x",
            page_text_map=[
                {"page": 0, "raw_text": "x", "char_count": 1},
            ],
            native_text_mode="text",
            source_document_sha256="not-a-sha",
        )


def test_observation_hash_is_canonical_core_hash():
    items, emit = _capture()
    result = observe_pdf_native_text_derivative_if_requested(
        emit_func=emit,
        selected_text="abc",
        page_text_map=[
            {"page": 0, "raw_text": "abc", "char_count": 3},
        ],
        native_text_mode="xhtml",
        source_document_sha256=None,
    )
    core = dict(result)
    observation_sha256 = core.pop("observation_sha256")
    canonical = json.dumps(
        core,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    assert observation_sha256 == hashlib.sha256(canonical).hexdigest()


def test_page_map_validation_rejects_conflicting_or_invented_counts():
    items, emit = _capture()
    with pytest.raises(
        PdfNativeTextDerivativeObservationError,
        match="char_count",
    ):
        observe_pdf_native_text_derivative_if_requested(
            emit_func=emit,
            selected_text="abc",
            page_text_map=[
                {"page": 0, "raw_text": "abc", "char_count": 99},
            ],
            native_text_mode="text",
        )

    with pytest.raises(
        PdfNativeTextDerivativeObservationError,
        match="unique",
    ):
        observe_pdf_native_text_derivative_if_requested(
            emit_func=emit,
            selected_text="abc",
            page_text_map=[
                {"page": 0, "raw_text": "a", "char_count": 1},
                {"page": 0, "raw_text": "b", "char_count": 1},
            ],
            native_text_mode="text",
        )


def test_callback_failure_is_sanitized_and_chained():
    def explode(_payload):
        raise RuntimeError("raw-secret-should-not-be-top-level-message")

    with pytest.raises(
        PdfNativeTextDerivativeObservationError,
        match="observation callback failed",
    ) as excinfo:
        observe_pdf_native_text_derivative_if_requested(
            emit_func=explode,
            selected_text="SECRET RAW TEXT",
            page_text_map=[
                {
                    "page": 0,
                    "raw_text": "SECRET RAW TEXT",
                    "char_count": len("SECRET RAW TEXT"),
                },
            ],
            native_text_mode="html",
        )
    assert "SECRET RAW TEXT" not in str(excinfo.value)



def test_pdf_handler_imports_service_and_inner_has_optional_scalar_and_callback():
    source = _source(PDF_PATH)
    tree = _tree(PDF_PATH)
    assert (
        "from ...services.pdf_native_text_derivative_evidence import "
        "observe_pdf_native_text_derivative_if_requested"
    ) in source

    inner = _fn(tree, "parse_pdf_election_results")
    kwonly = [arg.arg for arg in inner.args.kwonlyargs]
    assert kwonly.count("pdf_native_text_source_sha256") == 1
    assert kwonly.count("pdf_native_text_observation_emit_func") == 1

    calls = [
        node
        for node in ast.walk(inner)
        if isinstance(node, ast.Call)
        and _call_name(node.func).split(".")[-1]
        == "observe_pdf_native_text_derivative_if_requested"
    ]
    assert len(calls) == 1
    call = calls[0]
    kw = {item.arg: item.value for item in call.keywords}
    assert set(kw) == {
        "emit_func",
        "selected_text",
        "page_text_map",
        "native_text_mode",
        "source_document_sha256",
    }
    assert isinstance(kw["emit_func"], ast.Name)
    assert kw["emit_func"].id == "pdf_native_text_observation_emit_func"
    assert isinstance(kw["selected_text"], ast.Name)
    assert kw["selected_text"].id == "all_text"
    assert isinstance(kw["page_text_map"], ast.Name)
    assert kw["page_text_map"].id == "page_text_map"
    assert isinstance(kw["source_document_sha256"], ast.Name)
    assert kw["source_document_sha256"].id == "pdf_native_text_source_sha256"

def test_inner_still_does_not_consume_artifact_identity():
    tree = _tree(PDF_PATH)
    inner = _fn(tree, "parse_pdf_election_results")
    loads = [
        node
        for node in ast.walk(inner)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id == "artifact_identity"
    ]
    assert loads == []


def test_observation_seam_is_after_native_selection_before_markup_and_ocr_gate():
    source = _source(PDF_PATH)
    inner = _fn(_tree(PDF_PATH), "parse_pdf_election_results")
    segment = ast.get_source_segment(source, inner) or ""

    alt = segment.index(
        "alt_text, mode_used, alt_page_map = _extract_text_multi"
    )
    observer = segment.index(
        "observe_pdf_native_text_derivative_if_requested("
    )
    markup = segment.index("_is_mostly_markup(all_text)")
    ocr_gate = segment.index(
        'has_text = bool((all_text or "").strip())'
    )
    assert alt < observer < markup < ocr_gate



def test_wrapper_pops_callback_derives_scalar_forwards_both_and_guards_provided_tables():
    source = _source(PDF_PATH)
    wrapper = _fn(_tree(PDF_PATH), "parse")
    segment = ast.get_source_segment(source, wrapper) or ""

    assert (
        'pdf_native_text_observation_emit_func = kwargs.pop(\n'
        '        "pdf_native_text_observation_emit_func",'
    ) in segment
    assert (
        "native text derivative observation callback is unavailable "
        "for provided_tables wrapper path"
    ) in segment

    calls = [
        node
        for node in ast.walk(wrapper)
        if isinstance(node, ast.Call)
        and _call_name(node.func).split(".")[-1]
        == "parse_pdf_election_results"
    ]
    assert len(calls) == 1
    kw = {item.arg: item.value for item in calls[0].keywords}

    callback = kw["pdf_native_text_observation_emit_func"]
    assert isinstance(callback, ast.Name)
    assert callback.id == "pdf_native_text_observation_emit_func"

    assert "artifact_identity" in kw
    source_sha = kw["pdf_native_text_source_sha256"]
    assert isinstance(source_sha, ast.IfExp)
    rendered = ast.unparse(source_sha)
    assert "artifact_identity.document_sha256" in rendered
    assert "artifact_identity is not None" in rendered

def test_existing_structure_hook_and_parser_result_observation_boundaries_remain():
    source = _source(PDF_PATH)
    tree = _tree(PDF_PATH)
    structure = _fn(tree, "_record_page_text_structure_observation")
    structure_segment = ast.get_source_segment(source, structure) or ""

    for prohibited in (
        "all_text",
        "clean_text",
        "raw_text",
        "document_sha256",
        "artifact_identity",
        "hashlib",
        "sha256",
        "pdf_path",
    ):
        assert prohibited not in structure_segment

    wrapper = _fn(tree, "parse")
    wrapper_segment = ast.get_source_segment(source, wrapper) or ""
    assert "adapt_final_parser_result_for_observation(" in wrapper_segment
    assert "artifact_identity.document_sha256" in wrapper_segment
