"""C2G 2.8.25 contracts for the off-runtime mutation-effect executor."""

from __future__ import annotations

import ast
import copy
from pathlib import Path

import pytest

from webapp.parser.contracts.table_structure_review import (
    TableStructureReviewAction,
    TableStructureReviewCommand,
    TableStructureReviewRequest,
)
from webapp.parser.contracts.table_structure_review_execution import (
    TableStructureReviewEffectExecutionResult,
)
from webapp.parser.contracts.table_structure_review_harmonization import (
    TableStructureHarmonizationRequest,
)
from webapp.parser.services.table_structure_review_effect_executor import (
    TableStructureReviewEffectExecutionError,
    execute_table_structure_review_mutation_effect,
)
from webapp.parser.services.table_structure_review_harmonization import (
    harmonize_table_structure_review,
)
from webapp.parser.services.table_structure_review_mutations import (
    apply_add_columns,
    apply_remove_columns,
    apply_rename_columns,
    apply_reorder_columns,
)
from webapp.parser.services.table_structure_review_state_machine import (
    TableStructureReviewEffect,
    TableStructureReviewEffectKind,
    initialize_review_state,
    transition_review_state,
)


ROOT = Path(__file__).resolve().parents[1]


def _request(*actions: TableStructureReviewAction) -> TableStructureReviewRequest:
    return TableStructureReviewRequest(
        review_id="review-c2g-2825",
        session_id=None,
        domain="table-structure",
        contest=None,
        candidate_headers=("District", "Candidate", "Value"),
        rows_preview=(
            {
                "District": "__LOCATION_A__",
                "Candidate": "__ENTITY_A__",
                "Value": "__VALUE_A__",
            },
        ),
        candidate_index=1,
        candidates_total=1,
        ml_avg_confidence=None,
        allowed_actions=actions,
    )


def _effect(
    action: TableStructureReviewAction,
    payload,
) -> TableStructureReviewEffect:
    request = _request(action)
    state = initialize_review_state(request)
    command = TableStructureReviewCommand(
        review_id=request.review_id,
        action=action,
        payload=payload,
    )
    transition = transition_review_state(state, command)
    return transition.effect


def _direct_expected(headers, rows, effect, context=None, source_location_label=None):
    if effect.kind is TableStructureReviewEffectKind.REQUEST_REMOVE_COLUMNS:
        mutation = apply_remove_columns(headers, rows, effect.indices)
    elif effect.kind is TableStructureReviewEffectKind.REQUEST_REORDER_COLUMNS:
        mutation = apply_reorder_columns(headers, rows, effect.indices)
    elif effect.kind is TableStructureReviewEffectKind.REQUEST_RENAME_COLUMNS:
        mutation = apply_rename_columns(headers, rows, dict(effect.renames))
    elif effect.kind is TableStructureReviewEffectKind.REQUEST_ADD_COLUMNS:
        mutation = apply_add_columns(headers, rows, effect.names)
    else:
        raise AssertionError(effect.kind)

    harmonized = harmonize_table_structure_review(
        TableStructureHarmonizationRequest(
            headers=tuple(mutation.headers),
            rows=tuple(copy.deepcopy(mutation.rows)),
            context=copy.deepcopy(context),
            source_location_label=source_location_label,
        )
    )
    return mutation, harmonized


@pytest.mark.parametrize(
    "action,payload,headers,rows",
    [
        (
            TableStructureReviewAction.REMOVE_COLUMNS,
            {"indices": (2,)},
            ["District", "Candidate", "Value"],
            [
                {
                    "District": "__LOCATION_A__",
                    "Candidate": "__ENTITY_A__",
                    "Value": "__VALUE_A__",
                }
            ],
        ),
        (
            TableStructureReviewAction.REORDER_COLUMNS,
            {"order": (1, 0, 1)},
            ["District", "Candidate", "Value"],
            [
                {
                    "District": "__LOCATION_A__",
                    "Candidate": "__ENTITY_A__",
                    "Value": "__VALUE_A__",
                }
            ],
        ),
        (
            TableStructureReviewAction.RENAME_COLUMNS,
            {"renames": {2: "Renamed Value"}},
            ["District", "Candidate", "Value"],
            [
                {
                    "District": "__LOCATION_A__",
                    "Candidate": "__ENTITY_A__",
                    "Value": "__VALUE_A__",
                }
            ],
        ),
        (
            TableStructureReviewAction.ADD_COLUMNS,
            {"names": ("Added Column",)},
            ["District", "Candidate", "Value"],
            [
                {
                    "District": "__LOCATION_A__",
                    "Candidate": "__ENTITY_A__",
                    "Value": "__VALUE_A__",
                }
            ],
        ),
    ],
)
def test_executor_composes_exact_state_machine_effect_mutation_and_harmonization(
    action,
    payload,
    headers,
    rows,
):
    effect = _effect(action, payload)
    headers_before = copy.deepcopy(headers)
    rows_before = copy.deepcopy(rows)

    expected_mutation, expected_harmonized = _direct_expected(
        headers,
        rows,
        effect,
        source_location_label="District",
    )

    result = execute_table_structure_review_mutation_effect(
        headers,
        rows,
        effect,
        source_location_label="District",
    )

    assert isinstance(result, TableStructureReviewEffectExecutionResult)
    assert result.effect_kind == effect.kind.value
    assert result.pre_harmonization_headers == tuple(expected_mutation.headers)
    assert [dict(row) for row in result.pre_harmonization_rows] == [
        dict(row) for row in expected_mutation.rows
    ]
    assert result.headers == tuple(expected_harmonized.headers)
    assert [dict(row) for row in result.rows] == [
        dict(row) for row in expected_harmonized.rows
    ]

    assert headers == headers_before
    assert rows == rows_before

    assert result.source_location_label == "District"
    assert result.pure_mutation_applied is True
    assert result.harmonization_applied is True
    assert result.legacy_output_semantics_preserved is True
    assert result.caller_input_mutation_preserved is False
    assert result.candidate_materialization_applied is False
    assert result.learning_side_effect_applied is False
    assert result.runtime_transport_wired is False
    assert result.canonical_authority is False


def test_reorder_effect_indices_are_used_as_order():
    effect = _effect(
        TableStructureReviewAction.REORDER_COLUMNS,
        {"order": (2, 0)},
    )

    assert effect.kind is TableStructureReviewEffectKind.REQUEST_REORDER_COLUMNS
    assert effect.indices == (2, 0)

    result = execute_table_structure_review_mutation_effect(
        ["District", "Candidate", "Value"],
        [
            {
                "District": "__LOCATION_A__",
                "Candidate": "__ENTITY_A__",
                "Value": "__VALUE_A__",
            }
        ],
        effect,
    )

    assert result.pre_harmonization_headers == ("Value", "District")


@pytest.mark.parametrize(
    "kind",
    [
        TableStructureReviewEffectKind.COMPLETE_ACCEPTED,
        TableStructureReviewEffectKind.REQUEST_RETRY_DECISION,
        TableStructureReviewEffectKind.RETURN_TO_PRIMARY_REVIEW,
        TableStructureReviewEffectKind.COMPLETE_ORIGINAL,
        TableStructureReviewEffectKind.REQUEST_CANDIDATE_MATERIALIZATION,
    ],
)
def test_executor_fails_closed_for_non_mutation_effects(kind):
    effect = TableStructureReviewEffect(
        kind=kind,
        candidate_materialization_required=(
            kind is TableStructureReviewEffectKind.REQUEST_CANDIDATE_MATERIALIZATION
        ),
    )

    with pytest.raises(
        TableStructureReviewEffectExecutionError,
        match="mutation effects only",
    ):
        execute_table_structure_review_mutation_effect(
            ["District", "Candidate"],
            [{"District": "__LOCATION_A__", "Candidate": "__ENTITY_A__"}],
            effect,
        )


def test_executor_preserves_known_percent_zero_harmonizer_hazard_for_parity():
    effect = _effect(
        TableStructureReviewAction.ADD_COLUMNS,
        {"names": ("Unused",)},
    )

    result = execute_table_structure_review_mutation_effect(
        ["District", "Candidate", "Percent Reported"],
        [
            {
                "District": "__LOCATION_A__",
                "Candidate": "__ENTITY_A__",
                "Percent Reported": 0,
            }
        ],
        effect,
        harmonization_context={"percent_reported": "100%"},
    )

    assert result.rows[0]["Percent Reported"] == "100%"


def test_executor_has_exactly_one_approved_non_test_off_runtime_session_consumer():
    service_path = (
        ROOT
        / "parser"
        / "services"
        / "table_structure_review_effect_executor.py"
    )
    state_path = (
        ROOT
        / "parser"
        / "services"
        / "table_structure_review_state_machine.py"
    )
    approved_session_service = (
        ROOT
        / "parser"
        / "services"
        / "table_structure_review_session.py"
    )
    tests_root = ROOT / "tests"

    assert "table_structure_review_effect_executor" not in state_path.read_text(
        encoding="utf-8"
    )
    assert approved_session_service.is_file()
    assert tests_root.is_dir()

    approved_hits = []
    unexpected_runtime_hits = []
    test_evidence_hits = []

    for path in (ROOT.parent / "webapp").rglob("*.py"):
        if path == service_path:
            continue

        source = path.read_text(encoding="utf-8")
        if not (
            "table_structure_review_effect_executor" in source
            or "execute_table_structure_review_mutation_effect" in source
        ):
            continue

        if tests_root in path.parents:
            test_evidence_hits.append(path)
            continue

        if path == approved_session_service:
            approved_hits.append(path)
        else:
            unexpected_runtime_hits.append(path)

    assert approved_hits == [approved_session_service]
    assert unexpected_runtime_hits == []

    # Tests may name the executor/session boundary as regression evidence.
    # They are intentionally not classified as runtime consumers.
    assert Path(__file__).resolve() in test_evidence_hits


def test_executor_has_no_database_transport_learning_or_canonical_calls():
    service_path = (
        ROOT
        / "parser"
        / "services"
        / "table_structure_review_effect_executor.py"
    )
    tree = ast.parse(service_path.read_text(encoding="utf-8"))

    called = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            called.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            called.add(node.func.attr)

    forbidden = {
        "commit",
        "flush",
        "add",
        "execute",
        "save_table_structure_to_db",
        "log_table_structure",
        "cache_table_structure",
        "finalize_election_output",
        "emit",
        "send",
        "publish",
    }
    assert called.isdisjoint(forbidden)
