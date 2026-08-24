"""C2G 2.2 CSV wrapper explicit inspection ownership pass-through."""

from __future__ import annotations

import ast
import inspect

from webapp.parser.handlers.formats import csv_handler


def test_csv_wrapper_adds_keyword_only_inspection_inputs() -> None:
    signature = inspect.signature(csv_handler.parse)

    assert (
        signature.parameters["inspection_store"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )
    assert (
        signature.parameters["inspection_principal"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )


def test_csv_wrapper_has_exactly_one_primary_call() -> None:
    source = inspect.getsource(csv_handler.parse)
    tree = ast.parse(source)

    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "parse_csv_election_results"
    ]

    assert len(calls) == 1


def test_csv_wrapper_forwards_explicit_store_and_principal_names() -> None:
    source = inspect.getsource(csv_handler.parse)
    tree = ast.parse(source)

    call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "parse_csv_election_results"
    )

    values = {
        keyword.arg: keyword.value
        for keyword in call.keywords
        if keyword.arg is not None
    }

    assert isinstance(values["inspection_store"], ast.Name)
    assert values["inspection_store"].id == "inspection_store"

    assert isinstance(values["inspection_principal"], ast.Name)
    assert values["inspection_principal"].id == "inspection_principal"

    segment = ast.get_source_segment(source, call) or ""
    assert "context.get" not in segment
    assert "html_context.get" not in segment


def test_csv_primary_signature_remains_c2g21_contract() -> None:
    signature = inspect.signature(csv_handler.parse_csv_election_results)

    assert "session_id" in signature.parameters
    assert (
        signature.parameters["inspection_store"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )
    assert (
        signature.parameters["inspection_principal"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )