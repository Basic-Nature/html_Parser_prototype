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
    node = _function("_record_contest_hint_structure_observation")
    return "\n".join(lines[node.lineno - 1 : node.end_lineno])


class _Phase(str, Enum):
    CONTEST_HINT_STRUCTURE = "contest_hint_structure"


def _load_helper(record_callable, phase_enum):
    node = _function("_record_contest_hint_structure_observation")
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "_record_parse_observation": record_callable,
        "_StructureObservationPhase": phase_enum,
    }
    exec(
        compile(module, "<isolated-contest-hint-hook>", "exec"),
        namespace,
    )
    return namespace["_record_contest_hint_structure_observation"]


def test_contest_hint_hook_reuses_guarded_instrumentation_imports():
    source = _source()

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


def test_contest_hint_hook_is_immediately_after_contest_detection_metadata():
    source = _source()
    assignment = source.index(
        'metadata["contest_detection"] = contest_detection_diag'
    )
    hook = source.index(
        "_record_contest_hint_structure_observation(",
        assignment,
    )
    probe_titles = source.index(
        "probe_titles = contest_probe_info.get",
        hook,
    )

    between = source[assignment:probe_titles]
    assert assignment < hook < probe_titles
    assert between.count(
        "_record_contest_hint_structure_observation("
    ) == 1


def test_contest_hint_hook_call_return_is_ignored():
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
            == "_record_contest_hint_structure_observation"
        ):
            matches.append(call)

    assert len(matches) == 1


def test_contest_hint_hook_passes_only_bounded_primitive_arguments():
    parse_fn = _function("parse_pdf_election_results")
    calls = [
        node
        for node in ast.walk(parse_fn)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id
            == "_record_contest_hint_structure_observation"
        )
    ]
    assert len(calls) == 1

    call = calls[0]
    keyword_names = {kw.arg for kw in call.keywords}
    assert keyword_names == {
        "contest_detection_available",
        "detected_title_count",
        "selection_mode_if_already_present",
        "contest_segment_hint_count_if_already_present",
    }

    rendered = ast.dump(call, include_attributes=False)
    assert "contest_detection_diag" not in rendered
    assert "lines" not in rendered
    assert "pdf_path" not in rendered
    assert "all_text" not in rendered


def test_contest_hint_helper_emits_counts_and_flags_without_title_text():
    captured = {}

    def _record(**kwargs):
        captured.update(kwargs)
        return True

    helper = _load_helper(_record, _Phase)

    assert helper(
        contest_detection_available=True,
        detected_title_count=31,
        selection_mode_if_already_present="auto",
        contest_segment_hint_count_if_already_present=4,
    ) is True

    assert captured["kind"] == "pdf_structure_phase_observed"
    assert captured["provenance"] == "OBSERVED"
    assert captured["source_location"].endswith(
        ":contest_hint_structure"
    )
    assert captured["value_summary"] == {
        "phase": "contest_hint_structure",
        "contest_detection_available": True,
        "detected_title_count": 31,
        "selection_mode_if_already_present": "auto",
        "contest_segment_hint_count_if_already_present": 4,
    }

    rendered = repr(captured)
    assert "SECRET CONTEST" not in rendered
    assert "candidate" not in rendered.lower()
    assert "vote" not in rendered.lower()


def test_contest_hint_helper_bounds_invalid_counts_and_selection_mode():
    captured = {}

    def _record(**kwargs):
        captured.update(kwargs)
        return True

    helper = _load_helper(_record, _Phase)

    long_mode = "x" * 200
    assert helper(
        contest_detection_available=False,
        detected_title_count=-7,
        selection_mode_if_already_present=long_mode,
        contest_segment_hint_count_if_already_present=-3,
    ) is True

    summary = captured["value_summary"]
    assert summary["contest_detection_available"] is False
    assert summary["detected_title_count"] == 0
    assert summary[
        "contest_segment_hint_count_if_already_present"
    ] is None
    assert summary["selection_mode_if_already_present"] == "x" * 80


def test_contest_hint_helper_is_fail_open_when_observer_raises():
    def _boom(**_kwargs):
        raise RuntimeError("trace unavailable")

    helper = _load_helper(_boom, _Phase)

    assert helper(
        contest_detection_available=True,
        detected_title_count=1,
        selection_mode_if_already_present=None,
        contest_segment_hint_count_if_already_present=None,
    ) is False


def test_contest_hint_helper_is_fail_open_when_instrumentation_unavailable():
    helper_without_trace = _load_helper(None, _Phase)

    assert helper_without_trace(
        contest_detection_available=True,
        detected_title_count=1,
        selection_mode_if_already_present=None,
        contest_segment_hint_count_if_already_present=None,
    ) is False

    helper_without_phase = _load_helper(lambda **_kwargs: True, None)

    assert helper_without_phase(
        contest_detection_available=True,
        detected_title_count=1,
        selection_mode_if_already_present=None,
        contest_segment_hint_count_if_already_present=None,
    ) is False


def test_contest_hint_helper_has_no_parser_decision_or_raw_content_authority():
    helper_source = _helper_source()

    prohibited = (
        "adaptive_ocr_pipeline",
        "ocr_multi_pass",
        "OCR_CONFIDENCE_THRESHOLD",
        "select_contest",
        "_try_columnar_reconstruction",
        "return headers",
        "return data",
        "pdf_path",
        "all_text",
        "clean_text",
        "detected_titles",
        "contest_detection_diag",
        "candidate",
        "party",
        "votes",
        "source_url",
    )
    for token in prohibited:
        assert token not in helper_source
