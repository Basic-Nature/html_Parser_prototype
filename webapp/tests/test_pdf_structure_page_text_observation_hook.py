from __future__ import annotations

import ast
from enum import Enum
from pathlib import Path


PDF_PATH = (
    Path(__file__).resolve().parents[1]
    / "parser"
    / "handlers"
    / "formats"
    / "pdf_handler.py"
)


def _source() -> str:
    return PDF_PATH.read_text(encoding="utf-8")


def _tree() -> ast.Module:
    return ast.parse(_source(), filename=str(PDF_PATH))


def _function(name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in ast.walk(_tree())
        if isinstance(node, ast.FunctionDef)
        and node.name == name
    ]
    assert len(matches) == 1
    return matches[0]


def _helper_source() -> str:
    source = _source()
    lines = source.splitlines()
    node = _function("_record_page_text_structure_observation")
    return "\n".join(lines[node.lineno - 1 : node.end_lineno])


def _load_helper(record_callable, phase_enum):
    node = _function("_record_page_text_structure_observation")
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "_record_parse_observation": record_callable,
        "_StructureObservationPhase": phase_enum,
    }
    exec(compile(module, "<isolated-page-text-hook>", "exec"), namespace)
    return namespace["_record_page_text_structure_observation"]


class _Phase(str, Enum):
    PAGE_TEXT_STRUCTURE = "page_text_structure"


def test_hook_imports_are_guarded_observation_only():
    tree = _tree()
    source = _source()

    guarded = []
    for node in tree.body:
        if not isinstance(node, ast.Try):
            continue
        imported = []
        for child in node.body:
            if isinstance(child, ast.ImportFrom):
                imported.append((child.module, [a.name for a in child.names]))
        flat = repr(imported)
        if "parse_trace" in flat and "pdf_structure_profiler" in flat:
            guarded.append(node)

    assert len(guarded) == 1
    assert (
        "record_parse_observation as _record_parse_observation"
        in source
    )
    assert (
        "StructureObservationPhase as _StructureObservationPhase"
        in source
    )
    assert "_record_parse_observation = None" in source
    assert "_StructureObservationPhase = None" in source
    assert "PDFStructureProfileAccumulator" not in source


def test_hook_call_is_after_page_summary_and_before_ocr_evidence():
    source = _source()
    summary = source.index(
        'metadata["page_line_summary"] = page_summaries[:25]'
    )
    hook = source.index(
        "_record_page_text_structure_observation(",
        summary,
    )
    ocr_evidence = source.index(
        'metadata["ocr_evidence"] = _build_ocr_evidence(',
        hook,
    )
    assert summary < hook < ocr_evidence


def test_hook_call_return_is_ignored():
    parse_fn = _function("parse_pdf_election_results")
    matches = []
    for node in ast.walk(parse_fn):
        if not isinstance(node, ast.Expr):
            continue
        call = node.value
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id
            == "_record_page_text_structure_observation"
        ):
            matches.append(call)
    assert len(matches) == 1


def test_helper_emits_only_bounded_structure_summary():
    captured = {}

    def _record(**kwargs):
        captured.update(kwargs)
        return True

    helper = _load_helper(_record, _Phase)

    line_records = [
        {"page": 0, "text": "SECRET CANDIDATE 12345"},
        {"page": 0, "text": "SECRET VOTE 98765"},
    ]
    page_summaries = [
        {"page": 0, "sample": ["SECRET HEADER"]},
    ]
    page_text_map = [
        {"page": 0, "text": "SECRET RAW PAGE"},
    ]

    assert helper(
        pdf_page_total=10,
        line_records=line_records,
        page_summaries=page_summaries,
        page_lines_fallback=False,
        page_text_map=page_text_map,
        fitz_mode="text",
    ) is True

    assert captured["kind"] == "pdf_structure_phase_observed"
    assert captured["provenance"] == "OBSERVED"
    assert captured["source_location"].endswith(
        ":page_text_structure"
    )

    summary = captured["value_summary"]
    assert summary == {
        "phase": "page_text_structure",
        "page_count": 10,
        "page_line_total": 2,
        "page_line_pages": 1,
        "page_line_source": "page_map",
        "page_line_index_available": True,
        "page_lines_fallback": False,
        "page_text_map_entries": 1,
        "fitz_mode": "text",
    }

    rendered = repr(captured)
    assert "SECRET" not in rendered
    assert "12345" not in rendered
    assert "98765" not in rendered


def test_helper_is_fail_open_when_trace_observer_raises():
    def _boom(**_kwargs):
        raise RuntimeError("trace unavailable")

    helper = _load_helper(_boom, _Phase)

    assert helper(
        pdf_page_total=1,
        line_records=[],
        page_summaries=[],
        page_lines_fallback=True,
        page_text_map=[],
        fitz_mode=None,
    ) is False


def test_helper_is_fail_open_when_instrumentation_imports_unavailable():
    helper_without_trace = _load_helper(None, _Phase)
    assert helper_without_trace(
        pdf_page_total=1,
        line_records=[],
        page_summaries=[],
        page_lines_fallback=True,
        page_text_map=[],
        fitz_mode=None,
    ) is False

    helper_without_phase = _load_helper(lambda **_kwargs: True, None)
    assert helper_without_phase(
        pdf_page_total=1,
        line_records=[],
        page_summaries=[],
        page_lines_fallback=True,
        page_text_map=[],
        fitz_mode=None,
    ) is False


def test_helper_has_no_decision_or_raw_source_authority():
    helper_source = _helper_source()

    prohibited = (
        "adaptive_ocr_pipeline",
        "ocr_multi_pass",
        "OCR_CONFIDENCE_THRESHOLD",
        "select_contest",
        "return headers",
        "return data",
        "pdf_path",
        "all_text",
        "clean_text",
        "candidate",
        "party",
        "votes",
        "source_url",
    )
    for token in prohibited:
        assert token not in helper_source

    helper = _function("_record_page_text_structure_observation")
    metadata_assignments = []
    for node in ast.walk(helper):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
        )
        for target in targets:
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == "metadata"
            ):
                metadata_assignments.append(target)
    assert metadata_assignments == []
