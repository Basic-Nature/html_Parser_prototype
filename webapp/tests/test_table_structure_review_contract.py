from __future__ import annotations

from dataclasses import FrozenInstanceError
import math
from pathlib import Path

import pytest

from webapp.parser.contracts.table_structure_review import (
    CANONICAL_AUTHORITY,
    CLI_REPLACEMENT_AUTHORIZED,
    CONTRACT_VERSION,
    LEARNING_SIDE_EFFECT_AUTHORITY,
    PARSER_CONTROL_FLOW_AUTHORITY,
    REVIEW_ID_GENERATION_AUTHORITY,
    RUNTIME_TRANSPORT_WIRED,
    TIMESTAMP_GENERATION_AUTHORITY,
    TableStructureReviewAction,
    TableStructureReviewCommand,
    TableStructureReviewContractError,
    TableStructureReviewDecision,
    TableStructureReviewRequest,
    TableStructureReviewResult,
)


EXPECTED_ACTIONS = {
    "ACCEPT",
    "REJECT",
    "RETRY_DECISION",
    "REMOVE_COLUMNS",
    "REORDER_COLUMNS",
    "RENAME_COLUMNS",
    "ADD_COLUMNS",
    "NEXT_CANDIDATE",
    "PREVIOUS_CANDIDATE",
}


def _request(**overrides):
    values = {
        "review_id": "review-1",
        "session_id": None,
        "domain": "example.org",
        "contest": None,
        "candidate_headers": ("Precinct", "Candidate - Total"),
        "rows_preview": (
            {
                "Precinct": "P-1",
                "Candidate - Total": -4,
                "Unknown": None,
                "Confirmed Zero": 0,
            },
        ),
        "candidate_index": 1,
        "candidates_total": 2,
        "ml_avg_confidence": None,
        "allowed_actions": (
            TableStructureReviewAction.ACCEPT,
            TableStructureReviewAction.REJECT,
            TableStructureReviewAction.NEXT_CANDIDATE,
        ),
    }
    values.update(overrides)
    return TableStructureReviewRequest(**values)


def test_contract_authority_is_explicitly_inert_and_noncanonical():
    assert CONTRACT_VERSION == "table_structure_review_v1"
    assert CANONICAL_AUTHORITY is False
    assert RUNTIME_TRANSPORT_WIRED is False
    assert CLI_REPLACEMENT_AUTHORIZED is False
    assert PARSER_CONTROL_FLOW_AUTHORITY is False
    assert LEARNING_SIDE_EFFECT_AUTHORITY is False
    assert REVIEW_ID_GENERATION_AUTHORITY is False
    assert TIMESTAMP_GENERATION_AUTHORITY is False


def test_action_enum_exactly_models_typed_existing_actions_without_cancel_or_unknown():
    values = {action.value for action in TableStructureReviewAction}
    assert values == EXPECTED_ACTIONS
    assert "CANCEL" not in values
    assert "UNKNOWN_INPUT" not in values


def test_request_preserves_null_zero_and_signed_values_without_coercion():
    row = {
        "Precinct": "P-1",
        "Unknown": None,
        "Confirmed Zero": 0,
        "Signed Value": -4,
    }
    request = _request(
        rows_preview=(row,),
        ml_avg_confidence=None,
    )

    assert request.rows_preview[0]["Unknown"] is None
    assert request.rows_preview[0]["Confirmed Zero"] == 0
    assert request.rows_preview[0]["Signed Value"] == -4
    assert request.ml_avg_confidence is None

    # Validation must not mutate caller-owned evidence.
    assert row == {
        "Precinct": "P-1",
        "Unknown": None,
        "Confirmed Zero": 0,
        "Signed Value": -4,
    }


def test_request_is_frozen_at_the_dataclass_boundary():
    request = _request()
    with pytest.raises(FrozenInstanceError):
        request.domain = "changed.example"


@pytest.mark.parametrize(
    "bad_rows",
    [
        tuple({"x": index} for index in range(6)),
        ({"x": math.nan},),
        ({"x": math.inf},),
        ({"x": object()},),
        ({1: "non-string-key"},),
    ],
)
def test_request_rejects_unbounded_nonfinite_or_opaque_preview_evidence(
    bad_rows,
):
    with pytest.raises(TableStructureReviewContractError):
        _request(rows_preview=bad_rows)


@pytest.mark.parametrize(
    ("candidate_index", "candidates_total"),
    [
        (0, 1),
        (2, 1),
        (1, 0),
        (True, 1),
    ],
)
def test_request_candidate_position_fails_closed(
    candidate_index,
    candidates_total,
):
    with pytest.raises(TableStructureReviewContractError):
        _request(
            candidate_index=candidate_index,
            candidates_total=candidates_total,
        )


def test_request_does_not_invent_confidence_and_rejects_nonfinite_confidence():
    request = _request(ml_avg_confidence=None)
    assert request.ml_avg_confidence is None

    with pytest.raises(TableStructureReviewContractError):
        _request(ml_avg_confidence=math.nan)


def test_request_allowed_actions_are_typed_unique_and_explicit():
    with pytest.raises(TableStructureReviewContractError):
        _request(
            allowed_actions=(
                TableStructureReviewAction.ACCEPT,
                TableStructureReviewAction.ACCEPT,
            )
        )

    with pytest.raises(TableStructureReviewContractError):
        _request(allowed_actions=("ACCEPT",))


@pytest.mark.parametrize(
    "action",
    [
        TableStructureReviewAction.ACCEPT,
        TableStructureReviewAction.REJECT,
        TableStructureReviewAction.NEXT_CANDIDATE,
        TableStructureReviewAction.PREVIOUS_CANDIDATE,
    ],
)
def test_no_payload_actions_reject_payload(action):
    TableStructureReviewCommand(
        review_id="review-1",
        action=action,
        payload=None,
    )

    with pytest.raises(TableStructureReviewContractError):
        TableStructureReviewCommand(
            review_id="review-1",
            action=action,
            payload={},
        )


def test_retry_decision_payload_is_exact_and_boolean():
    command = TableStructureReviewCommand(
        review_id="review-1",
        action=TableStructureReviewAction.RETRY_DECISION,
        payload={"retry": False},
    )
    assert command.payload["retry"] is False

    with pytest.raises(TableStructureReviewContractError):
        TableStructureReviewCommand(
            review_id="review-1",
            action=TableStructureReviewAction.RETRY_DECISION,
            payload={"retry": "no"},
        )

    with pytest.raises(TableStructureReviewContractError):
        TableStructureReviewCommand(
            review_id="review-1",
            action=TableStructureReviewAction.RETRY_DECISION,
            payload={"retry": True, "extra": 1},
        )


def test_column_action_payloads_are_typed_without_reinterpreting_indices():
    remove = TableStructureReviewCommand(
        review_id="review-1",
        action=TableStructureReviewAction.REMOVE_COLUMNS,
        payload={"indices": (2, 0, -1)},
    )
    assert remove.payload["indices"] == (2, 0, -1)

    reorder = TableStructureReviewCommand(
        review_id="review-1",
        action=TableStructureReviewAction.REORDER_COLUMNS,
        payload={"order": [2, 0, 1]},
    )
    assert reorder.payload["order"] == [2, 0, 1]

    rename = TableStructureReviewCommand(
        review_id="review-1",
        action=TableStructureReviewAction.RENAME_COLUMNS,
        payload={"renames": {0: "", 2: "Total Votes"}},
    )
    assert rename.payload["renames"][0] == ""

    add = TableStructureReviewCommand(
        review_id="review-1",
        action=TableStructureReviewAction.ADD_COLUMNS,
        payload={"names": ("Absentee Mail", "Provisional")},
    )
    assert add.payload["names"] == ("Absentee Mail", "Provisional")


def test_command_rejects_string_action_and_unknown_payload_keys():
    with pytest.raises(TableStructureReviewContractError):
        TableStructureReviewCommand(
            review_id="review-1",
            action="ACCEPT",
            payload=None,
        )

    with pytest.raises(TableStructureReviewContractError):
        TableStructureReviewCommand(
            review_id="review-1",
            action=TableStructureReviewAction.ADD_COLUMNS,
            payload={"names": ("A",), "unexpected": True},
        )


def test_result_preserves_null_zero_signed_values_and_noncanonical_decision():
    result = TableStructureReviewResult(
        headers=("Precinct", "Total"),
        rows=(
            {
                "Precinct": "P-1",
                "Unknown": None,
                "Confirmed Zero": 0,
                "Signed Value": -4,
            },
        ),
        decision=TableStructureReviewDecision.ACCEPTED_REVIEW_STRUCTURE,
    )

    assert result.rows[0]["Unknown"] is None
    assert result.rows[0]["Confirmed Zero"] == 0
    assert result.rows[0]["Signed Value"] == -4
    assert result.canonical_authority is False
    assert result.runtime_transport_wired is False


def test_result_decisions_are_bounded_to_existing_return_outcomes():
    assert {item.value for item in TableStructureReviewDecision} == {
        "ACCEPTED_REVIEW_STRUCTURE",
        "ORIGINAL_STRUCTURE_RETAINED",
    }


def test_contract_source_has_no_runtime_transport_persistence_or_id_generation():
    import ast
    import webapp.parser.contracts.table_structure_review as contract_module

    source_path = Path(contract_module.__file__).resolve()
    source = source_path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(source_path))

    def dotted_name(node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            left = dotted_name(node.value)
            return f"{left}.{node.attr}" if left else node.attr
        return ""

    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])

    allowed_import_roots = {
        "__future__", "dataclasses", "enum", "math", "typing",
    }
    assert imported_roots <= allowed_import_roots

    forbidden_import_roots = {
        "flask", "flask_socketio", "socketio", "playwright",
        "selenium", "sqlalchemy", "psycopg", "requests",
        "uuid", "datetime", "time",
    }
    assert imported_roots.isdisjoint(forbidden_import_roots)

    call_names = {
        dotted_name(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    forbidden_call_suffixes = {
        "save_table_structure_to_db",
        "cache_table_structure",
        "prompt_user_to_confirm_table_structure",
        "uuid4", "now", "utcnow", "time",
    }
    forbidden_calls = {
        call_name
        for call_name in call_names
        if any(
            call_name == suffix or call_name.endswith('.' + suffix)
            for suffix in forbidden_call_suffixes
        )
    }
    assert forbidden_calls == set()

    # Documentation may name technologies while denying authority over them.
    # Runtime AST structure, not prose, is the isolation boundary.
    assert contract_module.RUNTIME_TRANSPORT_WIRED is False
    assert contract_module.CLI_REPLACEMENT_AUTHORIZED is False
    assert contract_module.PARSER_CONTROL_FLOW_AUTHORITY is False
    assert contract_module.LEARNING_SIDE_EFFECT_AUTHORITY is False
    assert contract_module.REVIEW_ID_GENERATION_AUTHORITY is False
    assert contract_module.TIMESTAMP_GENERATION_AUTHORITY is False
    assert contract_module.CANONICAL_AUTHORITY is False
