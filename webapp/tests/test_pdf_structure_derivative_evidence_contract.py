from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from webapp.parser.services.pdf_structure_derivative_evidence import (
    CONTRACT,
    DERIVATIVE_FINGERPRINT_SCHEME,
    SUPPORTED_PHASES,
    PdfStructureDerivativeObservationError,
    observe_pdf_structure_derivative_if_requested,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PDF_PATH = REPO_ROOT / "webapp/parser/handlers/formats/pdf_handler.py"
SERVICE_PATH = (
    REPO_ROOT
    / "webapp/parser/services/pdf_structure_derivative_evidence.py"
)

EXPECTED_HELPER_SHA256 = {
    "_record_page_text_structure_observation": (
        "5f6e012246eacb5e0edad37eb5096c6a49fdb35f786a56c15e9084808e93ca7c"
    ),
    "_record_contest_hint_structure_observation": (
        "0c92239d5689baca42bbd31931a26f37037c6e220218db37633c5b675556dcca"
    ),
    "_record_columnar_structure_observation": (
        "4042432b01cf1e5affc4d0d1fd8aee37812b573f1c7159239853d20961e8c4e1"
    ),
}


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


def _canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def test_contract_constants_and_supported_phases_exact():
    assert CONTRACT == "pdf_structure_derivative_observation_v1"
    assert (
        DERIVATIVE_FINGERPRINT_SCHEME
        == "SHA256_CANONICAL_JSON_OF_BOUNDED_STRUCTURE_PHASE_V1"
    )
    assert SUPPORTED_PHASES == (
        "page_text_structure",
        "contest_hint_structure",
        "columnar_structure",
    )
    assert "geometry" not in SUPPORTED_PHASES


def test_default_none_returns_before_validation_hash_or_iteration():
    bomb = Bomb()
    assert observe_pdf_structure_derivative_if_requested(
        emit_func=None,
        phase=bomb,
        bounded_summary=bomb,
        source_document_sha256=bomb,
    ) is None


def test_page_text_phase_hashes_only_canonical_bounded_summary():
    items, emit = _capture()
    summary = {
        "page_count": 12,
        "page_line_total": 40,
        "page_line_pages": 12,
        "page_line_source": "page_map",
        "page_line_index_available": True,
        "page_lines_fallback": False,
        "page_text_map_entries": 12,
        "fitz_mode": "text",
    }
    result = observe_pdf_structure_derivative_if_requested(
        emit_func=emit,
        phase="page_text_structure",
        bounded_summary=summary,
        source_document_sha256="a" * 64,
    )
    assert result == items[0]
    assert result["source_document_sha256"] == "a" * 64
    assert result["phase"] == "page_text_structure"
    assert result["bounded_summary"] == summary
    expected = hashlib.sha256(
        _canonical(
            {
                "phase": "page_text_structure",
                "bounded_summary": summary,
            }
        )
    ).hexdigest()
    assert result["derivative_sha256"] == expected
    assert result["raw_content_included"] is False
    assert result["new_persistence"] is False


@pytest.mark.parametrize(
    ("phase", "summary"),
    [
        (
            "contest_hint_structure",
            {
                "contest_detection_available": True,
                "detected_title_count": 4,
                "selection_mode_if_already_present": "auto",
                "contest_segment_hint_count_if_already_present": 2,
            },
        ),
        (
            "columnar_structure",
            {
                "attempted": True,
                "attempt_count_if_already_present": 1,
                "failure_present": False,
                "result_present": True,
                "segment_count_if_already_present": 3,
            },
        ),
    ],
)
def test_other_repository_proven_phases_emit_hash_only(phase, summary):
    items, emit = _capture()
    result = observe_pdf_structure_derivative_if_requested(
        emit_func=emit,
        phase=phase,
        bounded_summary=summary,
        source_document_sha256=None,
    )
    assert result == items[0]
    assert result["source_document_sha256"] is None
    assert result["phase"] == phase
    assert result["bounded_summary"] == summary
    rendered = json.dumps(result, sort_keys=True)
    for forbidden in (
        "pdf_path",
        "filename",
        "source_url",
        "requested_url",
        "session_id",
        "cookie",
        "headers",
        "candidate",
        "votes",
        "raw_text",
    ):
        assert forbidden not in rendered.lower()


def test_geometry_is_rejected_without_repository_proven_runtime_producer():
    items, emit = _capture()
    with pytest.raises(
        PdfStructureDerivativeObservationError,
        match="unsupported",
    ):
        observe_pdf_structure_derivative_if_requested(
            emit_func=emit,
            phase="geometry",
            bounded_summary={},
            source_document_sha256=None,
        )


def test_summary_contract_rejects_extra_or_raw_fields():
    items, emit = _capture()
    with pytest.raises(
        PdfStructureDerivativeObservationError,
        match="fields",
    ):
        observe_pdf_structure_derivative_if_requested(
            emit_func=emit,
            phase="page_text_structure",
            bounded_summary={
                "page_count": 1,
                "page_line_total": 1,
                "page_line_pages": 1,
                "page_line_source": "page_map",
                "page_line_index_available": True,
                "page_lines_fallback": False,
                "page_text_map_entries": 1,
                "fitz_mode": "text",
                "raw_text": "SECRET",
            },
            source_document_sha256=None,
        )


def test_source_identity_is_scalar_only_and_source_file_is_never_rehashed():
    items, emit = _capture()
    with pytest.raises(
        PdfStructureDerivativeObservationError,
        match="SHA-256",
    ):
        observe_pdf_structure_derivative_if_requested(
            emit_func=emit,
            phase="columnar_structure",
            bounded_summary={
                "attempted": True,
                "attempt_count_if_already_present": None,
                "failure_present": False,
                "result_present": False,
                "segment_count_if_already_present": None,
            },
            source_document_sha256="not-a-sha",
        )

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


def test_observation_hash_is_canonical_core_hash():
    items, emit = _capture()
    result = observe_pdf_structure_derivative_if_requested(
        emit_func=emit,
        phase="contest_hint_structure",
        bounded_summary={
            "contest_detection_available": False,
            "detected_title_count": 0,
            "selection_mode_if_already_present": None,
            "contest_segment_hint_count_if_already_present": None,
        },
        source_document_sha256=None,
    )
    core = dict(result)
    observation_sha256 = core.pop("observation_sha256")
    assert observation_sha256 == hashlib.sha256(
        _canonical(core)
    ).hexdigest()


def test_callback_failure_is_sanitized_and_chained():
    def explode(_payload):
        raise RuntimeError("SECRET CALLBACK DETAIL")

    with pytest.raises(
        PdfStructureDerivativeObservationError,
        match="observation callback failed",
    ) as excinfo:
        observe_pdf_structure_derivative_if_requested(
            emit_func=explode,
            phase="columnar_structure",
            bounded_summary={
                "attempted": True,
                "attempt_count_if_already_present": 1,
                "failure_present": True,
                "result_present": False,
                "segment_count_if_already_present": None,
            },
            source_document_sha256=None,
        )
    assert "SECRET CALLBACK DETAIL" not in str(excinfo.value)


def test_existing_fail_open_diagnostic_helpers_are_byte_source_unchanged():
    source = _source(PDF_PATH)
    tree = _tree(PDF_PATH)
    for name, expected in EXPECTED_HELPER_SHA256.items():
        node = _fn(tree, name)
        segment = ast.get_source_segment(source, node) or ""
        assert hashlib.sha256(segment.encode("utf-8")).hexdigest() == expected
        assert "_record_parse_observation" in segment
        assert "except Exception" in segment
        assert "return False" in segment


def test_handler_imports_service_and_inner_has_optional_scalar_and_callback():
    source = _source(PDF_PATH)
    tree = _tree(PDF_PATH)
    assert (
        "from ...services.pdf_structure_derivative_evidence import "
        "observe_pdf_structure_derivative_if_requested"
    ) in source

    inner = _fn(tree, "parse_pdf_election_results")
    kwonly = [arg.arg for arg in inner.args.kwonlyargs]
    assert kwonly.count("pdf_structure_source_sha256") == 1
    assert kwonly.count("pdf_structure_observation_emit_func") == 1


def test_inner_still_does_not_consume_artifact_identity():
    inner = _fn(_tree(PDF_PATH), "parse_pdf_election_results")
    loads = [
        node
        for node in ast.walk(inner)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id == "artifact_identity"
    ]
    assert loads == []



def test_first_class_observer_call_count_and_phases_are_exact():
    inner = _fn(_tree(PDF_PATH), "parse_pdf_election_results")
    calls = [
        node
        for node in ast.walk(inner)
        if isinstance(node, ast.Call)
        and _call_name(node.func).split(".")[-1]
        == "observe_pdf_structure_derivative_if_requested"
    ]
    assert len(calls) == 6

    phases = []
    for call in calls:
        kw = {item.arg: item.value for item in call.keywords}
        assert set(kw) == {
            "emit_func",
            "phase",
            "bounded_summary",
            "source_document_sha256",
        }
        assert isinstance(kw["emit_func"], ast.Name)
        assert kw["emit_func"].id == "pdf_structure_observation_emit_func"
        assert isinstance(kw["source_document_sha256"], ast.Name)
        assert kw["source_document_sha256"].id == "pdf_structure_source_sha256"
        assert isinstance(kw["phase"], ast.Constant)
        phases.append(kw["phase"].value)
        assert isinstance(kw["bounded_summary"], ast.Dict)

    assert phases.count("page_text_structure") == 1
    assert phases.count("contest_hint_structure") == 1
    assert phases.count("columnar_structure") == 4
    assert "geometry" not in phases


def test_first_class_calls_are_adjacent_to_existing_repository_proven_seams():
    source = _source(PDF_PATH)

    page_old = source.index(
        "_record_page_text_structure_observation(",
        source.index('metadata["page_line_summary"] = page_summaries[:25]'),
    )
    page_new = source.index(
        "observe_pdf_structure_derivative_if_requested(",
        page_old,
    )
    ocr_evidence = source.index(
        'metadata["ocr_evidence"] = _build_ocr_evidence(',
        page_new,
    )
    assert page_old < page_new < ocr_evidence

    contest_old = source.index(
        "_record_contest_hint_structure_observation(",
        source.index('metadata["contest_detection"] = contest_detection_diag'),
    )
    contest_new = source.index(
        "observe_pdf_structure_derivative_if_requested(",
        contest_old,
    )
    probe_titles = source.index(
        'probe_titles = contest_probe_info.get',
        contest_new,
    )
    assert contest_old < contest_new < probe_titles

    source_lines = source.splitlines()
    inner = _fn(_tree(PDF_PATH), "parse_pdf_election_results")

    recon_assignments = []
    old_hooks = []
    result_ifs = []
    new_columnar = []

    for node in ast.walk(inner):
        if isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name) and target.id == "recon_result"
                for target in node.targets
            ):
                if (
                    isinstance(node.value, ast.Call)
                    and _call_name(node.value.func).split(".")[-1]
                    == "_try_columnar_reconstruction"
                ):
                    recon_assignments.append(node)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            if (
                _call_name(node.value.func).split(".")[-1]
                == "_record_columnar_structure_observation"
            ):
                old_hooks.append(node)
        elif isinstance(node, ast.If):
            if isinstance(node.test, ast.Name) and node.test.id == "recon_result":
                result_ifs.append(node)

        if (
            isinstance(node, ast.Call)
            and _call_name(node.func).split(".")[-1]
            == "observe_pdf_structure_derivative_if_requested"
            and any(
                kw.arg == "phase"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value == "columnar_structure"
                for kw in node.keywords
            )
        ):
            new_columnar.append(node)

    recon_assignments.sort(key=lambda node: node.lineno)
    old_hooks.sort(key=lambda node: node.lineno)
    result_ifs.sort(key=lambda node: node.lineno)
    new_columnar.sort(key=lambda node: node.lineno)

    assert len(recon_assignments) == 2
    assert len(old_hooks) == 2
    assert len(result_ifs) == 2
    assert len(new_columnar) == 4

    # Preserve the pre-existing columnar contract exactly:
    # reconstruction -> legacy diagnostic hook -> immediate result branch.
    for assignment, hook, result_if in zip(
        recon_assignments,
        old_hooks,
        result_ifs,
    ):
        assert hook.lineno > assignment.end_lineno
        between = "\n".join(
            source_lines[assignment.end_lineno : hook.lineno - 1]
        )
        assert between.strip() == ""

        following = "\n".join(
            source_lines[hook.end_lineno : hook.end_lineno + 4]
        )
        assert "if recon_result:" in following
        assert result_if.lineno > hook.end_lineno

        # Success path: first-class evidence is emitted before return.
        success_calls = [
            node
            for node in ast.walk(result_if)
            if isinstance(node, ast.Call)
            and _call_name(node.func).split(".")[-1]
            == "observe_pdf_structure_derivative_if_requested"
            and any(
                kw.arg == "phase"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value == "columnar_structure"
                for kw in node.keywords
            )
        ]
        assert len(success_calls) == 1
        success_kw = {
            item.arg: item.value
            for item in success_calls[0].keywords
        }
        summary = success_kw["bounded_summary"]
        assert isinstance(summary, ast.Dict)
        result_entry = {
            key.value: value
            for key, value in zip(summary.keys, summary.values)
            if isinstance(key, ast.Constant)
        }["result_present"]
        assert isinstance(result_entry, ast.Constant)
        assert result_entry.value is True

    # Failure paths are top-level siblings after each result branch.
    parent_map = {}
    for node in ast.walk(inner):
        for child in ast.iter_child_nodes(node):
            parent_map[id(child)] = node

    success_call_ids = {
        id(node)
        for result_if in result_ifs
        for node in ast.walk(result_if)
        if isinstance(node, ast.Call)
        and _call_name(node.func).split(".")[-1]
        == "observe_pdf_structure_derivative_if_requested"
        and any(
            kw.arg == "phase"
            and isinstance(kw.value, ast.Constant)
            and kw.value.value == "columnar_structure"
            for kw in node.keywords
        )
    }
    failure_calls = [
        node
        for node in new_columnar
        if id(node) not in success_call_ids
    ]
    assert len(failure_calls) == 2

    for call in failure_calls:
        kw = {item.arg: item.value for item in call.keywords}
        summary = kw["bounded_summary"]
        assert isinstance(summary, ast.Dict)
        result_entry = {
            key.value: value
            for key, value in zip(summary.keys, summary.values)
            if isinstance(key, ast.Constant)
        }["result_present"]
        assert isinstance(result_entry, ast.Constant)
        assert result_entry.value is False

def test_wrapper_pops_callback_derives_scalar_and_forwards_both():
    source = _source(PDF_PATH)
    tree = _tree(PDF_PATH)
    wrapper = _fn(tree, "parse")
    segment = ast.get_source_segment(source, wrapper) or ""

    pop_assignments = [
        node
        for node in ast.walk(wrapper)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "pdf_structure_observation_emit_func"
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == "kwargs"
        and node.value.func.attr == "pop"
    ]
    assert len(pop_assignments) == 1
    pop_call = pop_assignments[0].value
    assert len(pop_call.args) == 2
    assert (
        isinstance(pop_call.args[0], ast.Constant)
        and pop_call.args[0].value == "pdf_structure_observation_emit_func"
    )
    assert (
        isinstance(pop_call.args[1], ast.Constant)
        and pop_call.args[1].value is None
    )

    guard_message = (
        "structure derivative observation callback is unavailable "
        "for provided_tables wrapper path"
    )
    guard_ifs = []
    for node in ast.walk(wrapper):
        if not isinstance(node, ast.If):
            continue
        rendered_test = ast.unparse(node.test)
        if (
            "provided_tables" in rendered_test
            and "pdf_structure_observation_emit_func is not None"
            in rendered_test
        ):
            guard_ifs.append(node)
    assert len(guard_ifs) == 1
    raises = [
        node
        for node in ast.walk(guard_ifs[0])
        if isinstance(node, ast.Raise)
        and isinstance(node.exc, ast.Call)
        and _call_name(node.exc.func).split(".")[-1] == "RuntimeError"
        and len(node.exc.args) == 1
        and isinstance(node.exc.args[0], ast.Constant)
    ]
    assert len(raises) == 1
    assert raises[0].exc.args[0].value == guard_message

    calls = [
        node
        for node in ast.walk(wrapper)
        if isinstance(node, ast.Call)
        and _call_name(node.func).split(".")[-1]
        == "parse_pdf_election_results"
    ]
    assert len(calls) == 1
    kw = {item.arg: item.value for item in calls[0].keywords}

    callback = kw["pdf_structure_observation_emit_func"]
    assert isinstance(callback, ast.Name)
    assert callback.id == "pdf_structure_observation_emit_func"

    source_sha = kw["pdf_structure_source_sha256"]
    assert isinstance(source_sha, ast.IfExp)
    rendered = ast.unparse(source_sha)
    assert "artifact_identity.document_sha256" in rendered
    assert "artifact_identity is not None" in rendered

    assert "artifact_identity" in kw
    assert "pdf_native_text_source_sha256" in kw
    assert "pdf_native_text_observation_emit_func" in kw


def test_existing_native_text_and_parser_result_observation_boundaries_remain():
    source = _source(PDF_PATH)
    assert source.count(
        "observe_pdf_native_text_derivative_if_requested("
    ) == 1
    assert "adapt_final_parser_result_for_observation(" in source
    assert "emit_parser_observation_bundle_if_requested(" in source
    assert "PDFStructureProfileAccumulator" not in source
