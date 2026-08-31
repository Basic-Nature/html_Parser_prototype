from __future__ import annotations

import ast
from pathlib import Path


PDF_PATH = (
    Path(__file__).resolve().parents[1]
    / "parser"
    / "handlers"
    / "formats"
    / "pdf_handler.py"
)


def _tree():
    source = PDF_PATH.read_text(encoding="utf-8")
    return source, ast.parse(source, filename=str(PDF_PATH))


def _fn(tree, name):
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == name
    ]
    assert len(matches) == 1
    return matches[0]


def _call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        left = _call_name(node.value)
        return f"{left}.{node.attr}" if left else node.attr
    return ""


def _artifact_kwonly(fn):
    names = [arg.arg for arg in fn.args.kwonlyargs]
    assert names.count("artifact_identity") == 1
    idx = names.index("artifact_identity")
    return fn.args.kwonlyargs[idx], fn.args.kw_defaults[idx]


def _annotation_text(arg):
    return ast.unparse(arg.annotation)


def _wrapper_handoff_call(wrapper):
    calls = [
        node
        for node in ast.walk(wrapper)
        if isinstance(node, ast.Call)
        and _call_name(node.func).endswith(
            "parse_pdf_election_results"
        )
    ]
    assert len(calls) == 1
    return calls[0]


def test_wrapper_artifact_identity_is_keyword_only_default_none():
    _, tree = _tree()
    wrapper = _fn(tree, "parse")
    _arg, default = _artifact_kwonly(wrapper)
    assert isinstance(default, ast.Constant)
    assert default.value is None


def test_inner_artifact_identity_is_keyword_only_default_none():
    _, tree = _tree()
    inner = _fn(tree, "parse_pdf_election_results")
    _arg, default = _artifact_kwonly(inner)
    assert isinstance(default, ast.Constant)
    assert default.value is None


def test_artifact_identity_annotation_is_exact_on_both():
    _, tree = _tree()
    wrapper = _fn(tree, "parse")
    inner = _fn(tree, "parse_pdf_election_results")
    wrapper_arg, _ = _artifact_kwonly(wrapper)
    inner_arg, _ = _artifact_kwonly(inner)

    assert (
        _annotation_text(wrapper_arg)
        == "ArtifactIdentityHandoff | None"
    )
    assert (
        _annotation_text(inner_arg)
        == "ArtifactIdentityHandoff | None"
    )


def test_wrapper_forwards_exact_identity_keyword_once():
    _, tree = _tree()
    wrapper = _fn(tree, "parse")
    call = _wrapper_handoff_call(wrapper)

    keywords = [
        keyword
        for keyword in call.keywords
        if keyword.arg == "artifact_identity"
    ]
    assert len(keywords) == 1
    assert isinstance(keywords[0].value, ast.Name)
    assert keywords[0].value.id == "artifact_identity"


def test_wrapper_does_not_forward_star_kwargs_to_inner():
    _, tree = _tree()
    wrapper = _fn(tree, "parse")
    call = _wrapper_handoff_call(wrapper)

    assert all(
        keyword.arg is not None
        for keyword in call.keywords
    )


def test_inner_does_not_consume_artifact_identity():
    _, tree = _tree()
    inner = _fn(tree, "parse_pdf_election_results")

    loads = [
        node
        for node in ast.walk(inner)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id == "artifact_identity"
    ]
    assert loads == []


def test_contract_import_is_exactly_once():
    _, tree = _tree()

    imports = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "contracts.artifact_identity"
        and node.level == 3
        and any(
            alias.name == "ArtifactIdentityHandoff"
            for alias in node.names
        )
    ]
    assert len(imports) == 1


def test_no_hashing_in_wrapper_or_inner():
    _, tree = _tree()
    wrapper = _fn(tree, "parse")
    inner = _fn(tree, "parse_pdf_election_results")

    names = {
        _call_name(node.func).split(".")[-1]
        for fn in (wrapper, inner)
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
    }

    assert "sha256" not in names
    assert "sha256_file" not in names
    assert "compute_sha256" not in names


def test_no_geometry_acquisition_in_wrapper_or_inner():
    _, tree = _tree()
    wrapper = _fn(tree, "parse")
    inner = _fn(tree, "parse_pdf_election_results")

    names = {
        _call_name(node.func).split(".")[-1]
        for fn in (wrapper, inner)
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
    }

    assert "_get_page_orientation_map" not in names
    assert "_collect_page_orientation" not in names


def test_no_accumulator_construction_in_wrapper_or_inner():
    source, tree = _tree()
    wrapper = _fn(tree, "parse")
    inner = _fn(tree, "parse_pdf_election_results")

    for fn in (wrapper, inner):
        segment = ast.get_source_segment(source, fn) or ""
        assert "PDFStructureProfileAccumulator" not in segment


def test_no_trace_identity_egress():
    source, tree = _tree()
    wrapper = _fn(tree, "parse")
    inner = _fn(tree, "parse_pdf_election_results")

    for fn in (wrapper, inner):
        for node in ast.walk(fn):
            if (
                isinstance(node, ast.Call)
                and _call_name(node.func).endswith(
                    "_record_parse_observation"
                )
            ):
                rendered = ast.get_source_segment(
                    source,
                    node,
                ) or ""
                assert "artifact_identity" not in rendered
                assert "document_sha256" not in rendered


def test_existing_positional_parameters_are_preserved():
    _, tree = _tree()
    wrapper = _fn(tree, "parse")
    inner = _fn(tree, "parse_pdf_election_results")

    wrapper_positional = [
        arg.arg
        for arg in (
            list(wrapper.args.posonlyargs)
            + list(wrapper.args.args)
        )
    ]
    inner_positional = [
        arg.arg
        for arg in (
            list(inner.args.posonlyargs)
            + list(inner.args.args)
        )
    ]

    assert wrapper_positional == [
        "page",
        "coordinator",
        "html_context",
        "manual_file",
        "session_id",
    ]
    assert inner_positional == [
        "pdf_path",
        "session_id",
        "coordinator",
        "cancel_flag",
    ]


def test_no_artifact_identity_metadata_or_output_fields():
    source, tree = _tree()
    wrapper = _fn(tree, "parse")
    inner = _fn(tree, "parse_pdf_election_results")

    for fn in (wrapper, inner):
        for node in ast.walk(fn):
            if isinstance(node, ast.Constant) and isinstance(
                node.value,
                str,
            ):
                assert node.value not in {
                    "artifact_identity",
                    "document_sha256",
                }
