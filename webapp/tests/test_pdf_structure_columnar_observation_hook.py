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
    node = _function("_record_columnar_structure_observation")
    return "\n".join(lines[node.lineno - 1 : node.end_lineno])


class _Phase(str, Enum):
    COLUMNAR_STRUCTURE = "columnar_structure"


def _load_helper(record_callable, phase_enum):
    node = _function("_record_columnar_structure_observation")
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "_record_parse_observation": record_callable,
        "_StructureObservationPhase": phase_enum,
    }
    exec(
        compile(module, "<isolated-columnar-hook>", "exec"),
        namespace,
    )
    return namespace["_record_columnar_structure_observation"]


def _reconstruction_assignments(parse_fn: ast.FunctionDef):
    matches = []
    for node in ast.walk(parse_fn):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name)
            and target.id == "recon_result"
            for target in node.targets
        ):
            continue
        if not isinstance(node.value, ast.Call):
            continue
        if (
            isinstance(node.value.func, ast.Name)
            and node.value.func.id == "_try_columnar_reconstruction"
        ):
            matches.append(node)
    return sorted(matches, key=lambda node: node.lineno)


def _columnar_hook_exprs(parse_fn: ast.FunctionDef):
    matches = []
    for node in ast.walk(parse_fn):
        if not isinstance(node, ast.Expr):
            continue
        if not isinstance(node.value, ast.Call):
            continue
        if (
            isinstance(node.value.func, ast.Name)
            and node.value.func.id
            == "_record_columnar_structure_observation"
        ):
            matches.append(node)
    return sorted(matches, key=lambda node: node.lineno)


def test_columnar_hook_reuses_guarded_instrumentation_imports():
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


def test_columnar_hook_has_exactly_two_ignored_return_call_sites():
    parse_fn = _function("parse_pdf_election_results")
    assignments = _reconstruction_assignments(parse_fn)
    hooks = _columnar_hook_exprs(parse_fn)

    assert len(assignments) == 2
    assert len(hooks) == 2


def test_each_columnar_hook_is_after_existing_reconstruction_and_before_result_branch():
    source_lines = _source().splitlines()
    parse_fn = _function("parse_pdf_election_results")
    assignments = _reconstruction_assignments(parse_fn)
    hooks = _columnar_hook_exprs(parse_fn)

    for assignment, hook in zip(assignments, hooks):
        assert hook.lineno > assignment.end_lineno
        between = "\n".join(
            source_lines[assignment.end_lineno : hook.lineno - 1]
        )
        assert between.strip() == ""

        following = "\n".join(
            source_lines[hook.end_lineno : hook.end_lineno + 4]
        )
        assert "if recon_result:" in following


def test_columnar_reconstruction_call_sites_are_mutually_exclusive():
    parse_fn = _function("parse_pdf_election_results")
    assignments = _reconstruction_assignments(parse_fn)
    assert len(assignments) == 2

    parents = {}
    for node in ast.walk(parse_fn):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node

    first = assignments[0]
    second = assignments[1]

    cursor = first
    terminating_else = None

    while id(cursor) in parents:
        parent = parents[id(cursor)]
        if isinstance(parent, ast.If) and first in parent.orelse:
            if parent.orelse and isinstance(parent.orelse[-1], ast.Return):
                terminating_else = parent
                break
        cursor = parent

    assert terminating_else is not None

    second_inside_same_else = any(
        node is second
        for branch_node in terminating_else.orelse
        for node in ast.walk(branch_node)
    )
    assert second_inside_same_else is False


def test_columnar_hook_calls_pass_only_bounded_derived_arguments():
    parse_fn = _function("parse_pdf_election_results")
    hooks = _columnar_hook_exprs(parse_fn)

    expected = {
        "attempted",
        "attempt_count_if_already_present",
        "failure_present",
        "result_present",
        "segment_count_if_already_present",
    }

    for expr in hooks:
        call = expr.value
        assert {kw.arg for kw in call.keywords} == expected

        rendered = ast.dump(call, include_attributes=False)
        assert "selected_contest_title" not in rendered
        assert "pdf_path" not in rendered
        assert "lines" not in rendered
        assert "line_records" not in rendered
        assert "candidate" not in rendered.lower()
        assert "party" not in rendered.lower()
        assert "votes" not in rendered.lower()


def test_columnar_helper_emits_only_bounded_counts_and_flags():
    captured = {}

    def _record(**kwargs):
        captured.update(kwargs)
        return True

    helper = _load_helper(_record, _Phase)

    assert helper(
        attempted=True,
        attempt_count_if_already_present=2,
        failure_present=False,
        result_present=True,
        segment_count_if_already_present=3,
    ) is True

    assert captured["kind"] == "pdf_structure_phase_observed"
    assert captured["provenance"] == "OBSERVED"
    assert captured["source_location"].endswith(
        ":columnar_structure"
    )
    assert captured["value_summary"] == {
        "phase": "columnar_structure",
        "attempted": True,
        "attempt_count_if_already_present": 2,
        "failure_present": False,
        "result_present": True,
        "segment_count_if_already_present": 3,
    }


def test_columnar_helper_preserves_unknown_counts_as_none():
    captured = {}

    def _record(**kwargs):
        captured.update(kwargs)
        return True

    helper = _load_helper(_record, _Phase)

    assert helper(
        attempted=True,
        attempt_count_if_already_present=-1,
        failure_present=True,
        result_present=False,
        segment_count_if_already_present=-9,
    ) is True

    summary = captured["value_summary"]
    assert summary["attempt_count_if_already_present"] is None
    assert summary["segment_count_if_already_present"] is None
    assert summary["failure_present"] is True
    assert summary["result_present"] is False


def test_columnar_helper_is_fail_open_when_observer_raises():
    def _boom(**_kwargs):
        raise RuntimeError("trace unavailable")

    helper = _load_helper(_boom, _Phase)

    assert helper(
        attempted=True,
        attempt_count_if_already_present=None,
        failure_present=False,
        result_present=False,
        segment_count_if_already_present=None,
    ) is False


def test_columnar_helper_is_fail_open_when_instrumentation_unavailable():
    helper_without_trace = _load_helper(None, _Phase)

    assert helper_without_trace(
        attempted=True,
        attempt_count_if_already_present=None,
        failure_present=False,
        result_present=False,
        segment_count_if_already_present=None,
    ) is False

    helper_without_phase = _load_helper(lambda **_kwargs: True, None)

    assert helper_without_phase(
        attempted=True,
        attempt_count_if_already_present=None,
        failure_present=False,
        result_present=False,
        segment_count_if_already_present=None,
    ) is False


def test_columnar_helper_has_no_parser_decision_or_raw_content_authority():
    helper_source = _helper_source()

    prohibited = (
        "_try_columnar_reconstruction",
        "adaptive_ocr_pipeline",
        "ocr_multi_pass",
        "select_contest",
        "_get_page_orientation_map",
        "return recon_result",
        "metadata",
        "recon_result",
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
