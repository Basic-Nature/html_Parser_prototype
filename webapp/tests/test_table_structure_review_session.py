"""C2G 2.8.27 tests for immutable-original off-runtime review sessions."""

from __future__ import annotations

import ast
import copy
from pathlib import Path

import pytest

from webapp.parser.contracts.table_structure_review import (
    TableStructureReviewAction,
    TableStructureReviewCommand,
    TableStructureReviewDecision,
    TableStructureReviewRequest,
)
from webapp.parser.services.table_structure_review_session import (
    TableStructureReviewSessionError,
    advance_table_structure_review_session,
    initialize_table_structure_review_session,
)
from webapp.parser.services.table_structure_review_state_machine import (
    TableStructureReviewPhase,
)


WEBAPP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WEBAPP_ROOT.parent


def _request(*actions: TableStructureReviewAction) -> TableStructureReviewRequest:
    return TableStructureReviewRequest(
        review_id="review-c2g-2827",
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
        candidates_total=2,
        ml_avg_confidence=None,
        allowed_actions=actions,
    )


def _command(
    action: TableStructureReviewAction,
    payload=None,
) -> TableStructureReviewCommand:
    return TableStructureReviewCommand(
        review_id="review-c2g-2827",
        action=action,
        payload=payload,
    )


def _base_table():
    return (
        ["District", "Candidate", "Value"],
        [
            {
                "District": "__LOCATION_A__",
                "Candidate": "__ENTITY_A__",
                "Value": "__VALUE_A__",
            }
        ],
    )


def _session(*actions: TableStructureReviewAction):
    headers, rows = _base_table()
    return initialize_table_structure_review_session(
        _request(*actions),
        headers,
        rows,
        source_location_label="District",
    )


def test_session_initialization_isolates_and_row_freezes_original_and_working():
    headers, rows = _base_table()
    session = initialize_table_structure_review_session(
        _request(
            TableStructureReviewAction.ACCEPT,
            TableStructureReviewAction.REJECT,
        ),
        headers,
        rows,
        source_location_label="District",
    )

    headers.append("CALLER_MUTATION")
    rows[0]["Value"] = "__CALLER_MUTATION__"

    assert session.original.headers == (
        "District",
        "Candidate",
        "Value",
    )
    assert session.working.headers == session.original.headers
    assert session.original.rows[0]["Value"] == "__VALUE_A__"
    assert session.working.rows[0]["Value"] == "__VALUE_A__"

    with pytest.raises(TypeError):
        session.original.rows[0]["Value"] = "__ILLEGAL__"

    with pytest.raises(TypeError):
        session.working.rows[0]["Value"] = "__ILLEGAL__"


def test_mutation_changes_working_only_and_preserves_original_baseline():
    session = _session(
        TableStructureReviewAction.RENAME_COLUMNS,
        TableStructureReviewAction.ACCEPT,
        TableStructureReviewAction.REJECT,
    )

    step = advance_table_structure_review_session(
        session,
        _command(
            TableStructureReviewAction.RENAME_COLUMNS,
            {"renames": {2: "Renamed Value"}},
        ),
    )

    assert step.effect_execution is not None
    assert step.completion is None

    assert step.session_after.original.headers == (
        "District",
        "Candidate",
        "Value",
    )
    assert step.session_after.original.rows[0]["Value"] == "__VALUE_A__"

    assert "Renamed Value" in step.session_after.working.headers
    assert step.session_after.working is not step.session_after.original


def test_accept_after_mutation_returns_current_working_snapshot():
    session = _session(
        TableStructureReviewAction.REMOVE_COLUMNS,
        TableStructureReviewAction.ACCEPT,
        TableStructureReviewAction.REJECT,
    )

    mutation_step = advance_table_structure_review_session(
        session,
        _command(
            TableStructureReviewAction.REMOVE_COLUMNS,
            {"indices": (2,)},
        ),
    )

    accepted = advance_table_structure_review_session(
        mutation_step.session_after,
        _command(TableStructureReviewAction.ACCEPT),
    )

    assert accepted.session_after.state.phase is TableStructureReviewPhase.COMPLETED
    assert (
        accepted.session_after.state.decision
        is TableStructureReviewDecision.ACCEPTED_REVIEW_STRUCTURE
    )
    assert accepted.completion is not None
    assert (
        accepted.completion.decision
        is TableStructureReviewDecision.ACCEPTED_REVIEW_STRUCTURE
    )
    assert accepted.completion.headers == accepted.session_before.working.headers
    assert accepted.completion.rows == accepted.session_before.working.rows
    assert accepted.completion.headers != accepted.session_before.original.headers


def test_reject_no_retry_returns_coherent_immutable_original_after_mutation():
    session = _session(
        TableStructureReviewAction.RENAME_COLUMNS,
        TableStructureReviewAction.REJECT,
        TableStructureReviewAction.ACCEPT,
    )

    mutation_step = advance_table_structure_review_session(
        session,
        _command(
            TableStructureReviewAction.RENAME_COLUMNS,
            {"renames": {2: "Renamed Value"}},
        ),
    )
    mutated_session = mutation_step.session_after

    rejected = advance_table_structure_review_session(
        mutated_session,
        _command(TableStructureReviewAction.REJECT),
    )

    assert rejected.completion is None
    assert rejected.session_after.state.phase is TableStructureReviewPhase.RETRY_DECISION
    assert rejected.session_after.working == mutated_session.working

    completed_original = advance_table_structure_review_session(
        rejected.session_after,
        _command(
            TableStructureReviewAction.RETRY_DECISION,
            {"retry": False},
        ),
    )

    assert completed_original.completion is not None
    assert (
        completed_original.completion.decision
        is TableStructureReviewDecision.ORIGINAL_STRUCTURE_RETAINED
    )
    assert (
        completed_original.completion.headers
        == completed_original.session_after.original.headers
    )
    assert (
        completed_original.completion.rows
        == completed_original.session_after.original.rows
    )

    # Prove we did NOT reproduce the raw-Git hybrid
    # "original headers + current/mutated data" behavior.
    assert (
        completed_original.completion.rows
        != completed_original.session_before.working.rows
    )


def test_retry_yes_preserves_current_working_snapshot():
    session = _session(
        TableStructureReviewAction.REMOVE_COLUMNS,
        TableStructureReviewAction.REJECT,
        TableStructureReviewAction.ACCEPT,
    )

    mutation_step = advance_table_structure_review_session(
        session,
        _command(
            TableStructureReviewAction.REMOVE_COLUMNS,
            {"indices": (2,)},
        ),
    )
    current_working = mutation_step.session_after.working

    rejected = advance_table_structure_review_session(
        mutation_step.session_after,
        _command(TableStructureReviewAction.REJECT),
    )

    retried = advance_table_structure_review_session(
        rejected.session_after,
        _command(
            TableStructureReviewAction.RETRY_DECISION,
            {"retry": True},
        ),
    )

    assert retried.completion is None
    assert retried.session_after.state.phase is TableStructureReviewPhase.PRIMARY_REVIEW
    assert retried.session_after.working == current_working
    assert retried.session_after.original == session.original


def test_candidate_navigation_fails_closed_without_materialization_authority():
    session = _session(
        TableStructureReviewAction.NEXT_CANDIDATE,
        TableStructureReviewAction.ACCEPT,
    )

    with pytest.raises(
        TableStructureReviewSessionError,
        match="candidate materialization authority",
    ):
        advance_table_structure_review_session(
            session,
            _command(TableStructureReviewAction.NEXT_CANDIDATE),
        )

    assert session.state.candidate_index == 1
    assert session.original == session.working


def test_session_preserves_null_zero_and_signed_values_in_original_baseline():
    request = _request(
        TableStructureReviewAction.ADD_COLUMNS,
        TableStructureReviewAction.REJECT,
    )
    rows = [
        {
            "District": "__LOCATION_A__",
            "Candidate": "__ENTITY_A__",
            "Null Evidence": None,
            "Zero Evidence": 0,
            "Signed Evidence": -4,
        }
    ]
    session = initialize_table_structure_review_session(
        request,
        [
            "District",
            "Candidate",
            "Null Evidence",
            "Zero Evidence",
            "Signed Evidence",
        ],
        rows,
    )

    assert session.original.rows[0]["Null Evidence"] is None
    assert session.original.rows[0]["Zero Evidence"] == 0
    assert session.original.rows[0]["Signed Evidence"] == -4


def test_session_service_has_no_database_transport_learning_or_canonical_calls():
    service_path = (
        WEBAPP_ROOT
        / "parser"
        / "services"
        / "table_structure_review_session.py"
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


def test_session_boundary_has_zero_non_test_runtime_callers():
    contract_path = (
        WEBAPP_ROOT
        / "parser"
        / "contracts"
        / "table_structure_review_session.py"
    )
    service_path = (
        WEBAPP_ROOT
        / "parser"
        / "services"
        / "table_structure_review_session.py"
    )
    this_test = Path(__file__).resolve()
    tests_root = WEBAPP_ROOT / "tests"

    assert contract_path.is_file()
    assert service_path.is_file()
    assert tests_root.is_dir()

    runtime_hits = []
    test_evidence_hits = []

    for path in WEBAPP_ROOT.rglob("*.py"):
        if path in {contract_path, service_path, this_test}:
            continue

        source = path.read_text(encoding="utf-8")
        if not (
            "table_structure_review_session" in source
            or "initialize_table_structure_review_session" in source
            or "advance_table_structure_review_session" in source
        ):
            continue

        if tests_root in path.parents:
            test_evidence_hits.append(path.relative_to(REPO_ROOT).as_posix())
            continue

        runtime_hits.append(path.relative_to(REPO_ROOT).as_posix())

    assert runtime_hits == []

    # The executor regression test is expected to reference the session service
    # because it proves that service is the executor's one approved off-runtime
    # consumer. Test-source references are evidence, not runtime wiring.
    assert (
        "webapp/tests/test_table_structure_review_effect_executor.py"
        in test_evidence_hits
    )


def test_session_contract_does_not_encode_historical_smart_wide_slots():
    contract_path = (
        WEBAPP_ROOT
        / "parser"
        / "contracts"
        / "table_structure_review_session.py"
    )
    source = contract_path.read_text(encoding="utf-8")

    for token in (
        "Election Day",
        "Mail-In",
        "Absentee Mail",
        "Early Voting",
        "Provisional Votes",
    ):
        assert token not in source
