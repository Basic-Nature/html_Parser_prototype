"""C2G 2.1 CSV typed-result to ephemeral-store integration contract."""

from __future__ import annotations

import ast
import inspect

import pytest

from webapp.parser.Context_Integration.context_write_policy import ContextWriteKind
from webapp.parser.contracts.table_pipeline import (
    SourceProvenance,
    TablePipelineResult,
    TableStage,
    TransformationRecord,
)
from webapp.parser.handlers.formats import csv_handler
from webapp.parser.services.ephemeral_pipeline_inspection import (
    ProcessLocalInspectionStore,
    ProcessLocalTopologyAttestation,
)


def _topology() -> ProcessLocalTopologyAttestation:
    return ProcessLocalTopologyAttestation(
        app_service_instance_capacity=1,
        gunicorn_workers=1,
        evidence_ref="C2G_1_9:test",
    )


def _typed_result() -> TablePipelineResult:
    return TablePipelineResult.from_sequences(
        stage=TableStage.INTERPRETED,
        headers=("Precinct", "Election Day"),
        rows=(
            {"Precinct": "P-1", "Election Day": None},
            {"Precinct": "P-2", "Election Day": 0},
            {"Precinct": "P-3", "Election Day": -4},
        ),
        source_provenance=SourceProvenance(
            source_type="csv",
            source_sha256="a" * 64,
            evidence_ref="fixture://c2g21",
        ),
        transformations=(
            TransformationRecord(
                sequence=0,
                from_stage=TableStage.INTERPRETED,
                to_stage=TableStage.INTERPRETED,
                operation="vote_method_header_canonicalization",
                rule_source=(
                    "Context_Integration.Context_Library.constants."
                    "BALLOT_NAME_CANON_MAP"
                ),
                details={
                    "before_header": "election day",
                    "after_header": "Election Day",
                    "vote_value_mutation": False,
                    "unknown_example": None,
                    "confirmed_zero_example": 0,
                    "signed_example": -4,
                },
            ),
        ),
        write_kind=ContextWriteKind.NONE,
    )


def test_csv_reuses_session_id_and_adds_only_keyword_only_inspection_inputs() -> None:
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


def test_default_inspection_capture_is_noop() -> None:
    assert (
        csv_handler._store_pipeline_inspection_if_requested(
            _typed_result(),
            session_id="session-1",
        )
        is False
    )


@pytest.mark.parametrize(
    ("store", "session_id", "principal"),
    [
        (ProcessLocalInspectionStore(topology=_topology()), None, "principal-a"),
        (ProcessLocalInspectionStore(topology=_topology()), "session-1", None),
        (None, "session-1", "principal-a"),
    ],
)
def test_partial_ownership_fails_closed(store, session_id, principal) -> None:
    with pytest.raises(ValueError):
        csv_handler._store_pipeline_inspection_if_requested(
            _typed_result(),
            inspection_store=store,
            session_id=session_id,
            principal=principal,
        )


def test_typed_result_is_projected_and_stored_under_explicit_owner() -> None:
    store = ProcessLocalInspectionStore(topology=_topology())

    assert (
        csv_handler._store_pipeline_inspection_if_requested(
            _typed_result(),
            inspection_store=store,
            session_id="session-1",
            principal="principal-a",
        )
        is True
    )

    payload = store.get(
        session_id="session-1",
        principal="principal-a",
    )
    assert payload is not None
    assert payload["contract"] == "pipeline_inspection_v1"
    assert payload["authority"]["canonical"] is False
    assert payload["rows_included"] is False
    assert payload["headers_included"] is False
    assert "rows" not in payload
    assert "headers" not in payload

    details = payload["transformations"][0]["details"]
    assert details["unknown_example"] is None
    assert details["confirmed_zero_example"] == 0
    assert details["signed_example"] == -4

    assert store.get(
        session_id="session-1",
        principal="principal-b",
    ) is None


def test_real_csv_typed_seam_calls_helper_with_explicit_ownership_names() -> None:
    source = inspect.getsource(csv_handler.parse_csv_election_results)
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_store_pipeline_inspection_if_requested"
    ]

    assert len(calls) == 1
    call = calls[0]

    assert len(call.args) == 1
    assert isinstance(call.args[0], ast.Name)
    assert call.args[0].id == "_c2g_table_result"

    values = {
        keyword.arg: keyword.value
        for keyword in call.keywords
        if keyword.arg is not None
    }

    assert isinstance(values["inspection_store"], ast.Name)
    assert values["inspection_store"].id == "inspection_store"
    assert isinstance(values["session_id"], ast.Name)
    assert values["session_id"].id == "session_id"
    assert isinstance(values["principal"], ast.Name)
    assert values["principal"].id == "inspection_principal"

    segment = ast.get_source_segment(source, call) or ""
    assert "context" not in segment


def test_csv_wrapper_passes_ownership_without_direct_store_side_effect() -> None:
    source = inspect.getsource(csv_handler.parse)
    tree = ast.parse(source)

    primary_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "parse_csv_election_results"
    ]
    assert len(primary_calls) == 1

    call = primary_calls[0]
    values = {
        keyword.arg: keyword.value
        for keyword in call.keywords
        if keyword.arg is not None
    }

    assert isinstance(values["inspection_store"], ast.Name)
    assert values["inspection_store"].id == "inspection_store"
    assert isinstance(values["inspection_principal"], ast.Name)
    assert values["inspection_principal"].id == "inspection_principal"

    direct_storage_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (
                isinstance(node.func, ast.Name)
                and node.func.id
                in {
                    "_store_pipeline_inspection_if_requested",
                    "project_pipeline_inspection",
                    "ProcessLocalInspectionStore",
                }
            )
            or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "put"
            )
        )
    ]
    assert direct_storage_calls == []
