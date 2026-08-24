from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from webapp.parser.contracts.table_structure_review import (
    TableStructureReviewAction,
    TableStructureReviewCommand,
    TableStructureReviewDecision,
    TableStructureReviewRequest,
)
from webapp.parser.services.table_structure_review_state_machine import (
    CANONICAL_AUTHORITY,
    CANDIDATE_MATERIALIZATION_IMPLEMENTED,
    CLI_REPLACEMENT_AUTHORIZED,
    LEARNING_SIDE_EFFECT_APPLICATION_IMPLEMENTED,
    PARSER_MUTATION_APPLICATION_IMPLEMENTED,
    RUNTIME_CALLER_WIRED,
    SERVICE_VERSION,
    TRANSPORT_WIRED,
    TableStructureReviewEffectKind,
    TableStructureReviewPhase,
    TableStructureReviewStateMachineError,
    initialize_review_state,
    transition_review_state,
)


PRIMARY_ACTIONS = (
    TableStructureReviewAction.ACCEPT,
    TableStructureReviewAction.REJECT,
    TableStructureReviewAction.REMOVE_COLUMNS,
    TableStructureReviewAction.REORDER_COLUMNS,
    TableStructureReviewAction.RENAME_COLUMNS,
    TableStructureReviewAction.ADD_COLUMNS,
    TableStructureReviewAction.NEXT_CANDIDATE,
    TableStructureReviewAction.PREVIOUS_CANDIDATE,
)


def _request(**overrides):
    values = {
        "review_id": "review-1",
        "session_id": "session-1",
        "domain": "example.org",
        "contest": None,
        "candidate_headers": ("A", "B", "C"),
        "rows_preview": (
            {"A": None, "B": 0, "C": -4},
        ),
        "candidate_index": 1,
        "candidates_total": 3,
        "ml_avg_confidence": None,
        "allowed_actions": PRIMARY_ACTIONS,
    }
    values.update(overrides)
    return TableStructureReviewRequest(**values)


def _command(action, payload=None, review_id="review-1"):
    return TableStructureReviewCommand(
        review_id=review_id,
        action=action,
        payload=payload,
    )


def test_service_authority_is_explicitly_scaffold_only():
    assert SERVICE_VERSION == "table_structure_review_state_machine_v1"
    assert RUNTIME_CALLER_WIRED is False
    assert TRANSPORT_WIRED is False
    assert PARSER_MUTATION_APPLICATION_IMPLEMENTED is False
    assert LEARNING_SIDE_EFFECT_APPLICATION_IMPLEMENTED is False
    assert CANDIDATE_MATERIALIZATION_IMPLEMENTED is False
    assert CLI_REPLACEMENT_AUTHORIZED is False
    assert CANONICAL_AUTHORITY is False


def test_initialize_state_preserves_request_coordination_facts_only():
    request = _request(
        candidate_index=2,
        candidates_total=4,
    )
    state = initialize_review_state(request)

    assert state.review_id == "review-1"
    assert state.phase is TableStructureReviewPhase.PRIMARY_REVIEW
    assert state.candidate_index == 2
    assert state.candidates_total == 4
    assert state.allowed_primary_actions == PRIMARY_ACTIONS
    assert state.decision is None


def test_state_is_frozen():
    state = initialize_review_state(_request())
    with pytest.raises(FrozenInstanceError):
        state.candidate_index = 2


def test_accept_completes_with_noncanonical_accepted_decision():
    state = initialize_review_state(_request())
    transition = transition_review_state(
        state,
        _command(TableStructureReviewAction.ACCEPT),
    )

    assert transition.state_after.phase is TableStructureReviewPhase.COMPLETED
    assert (
        transition.state_after.decision
        is TableStructureReviewDecision.ACCEPTED_REVIEW_STRUCTURE
    )
    assert (
        transition.effect.kind
        is TableStructureReviewEffectKind.COMPLETE_ACCEPTED
    )
    assert transition.effect.parser_mutation_applied is False
    assert transition.effect.learning_side_effect_applied is False


def test_reject_requires_retry_phase_before_original_return():
    state = initialize_review_state(_request())

    rejected = transition_review_state(
        state,
        _command(TableStructureReviewAction.REJECT),
    )

    assert (
        rejected.state_after.phase
        is TableStructureReviewPhase.RETRY_DECISION
    )
    assert (
        rejected.effect.kind
        is TableStructureReviewEffectKind.REQUEST_RETRY_DECISION
    )

    with pytest.raises(
        TableStructureReviewStateMachineError,
        match="retry phase accepts",
    ):
        transition_review_state(
            rejected.state_after,
            _command(TableStructureReviewAction.ACCEPT),
        )

    original = transition_review_state(
        rejected.state_after,
        _command(
            TableStructureReviewAction.RETRY_DECISION,
            {"retry": False},
        ),
    )

    assert original.state_after.phase is TableStructureReviewPhase.COMPLETED
    assert (
        original.state_after.decision
        is TableStructureReviewDecision.ORIGINAL_STRUCTURE_RETAINED
    )
    assert (
        original.effect.kind
        is TableStructureReviewEffectKind.COMPLETE_ORIGINAL
    )


def test_retry_true_returns_to_primary_without_claiming_side_effects():
    state = initialize_review_state(_request())
    rejected = transition_review_state(
        state,
        _command(TableStructureReviewAction.REJECT),
    )
    retry = transition_review_state(
        rejected.state_after,
        _command(
            TableStructureReviewAction.RETRY_DECISION,
            {"retry": True},
        ),
    )

    assert retry.state_after.phase is TableStructureReviewPhase.PRIMARY_REVIEW
    assert retry.state_after.decision is None
    assert (
        retry.effect.kind
        is TableStructureReviewEffectKind.RETURN_TO_PRIMARY_REVIEW
    )
    assert retry.effect.parser_mutation_applied is False
    assert retry.effect.learning_side_effect_applied is False


def test_retry_decision_is_invalid_in_primary_phase():
    state = initialize_review_state(_request())

    with pytest.raises(
        TableStructureReviewStateMachineError,
        match="valid only after REJECT",
    ):
        transition_review_state(
            state,
            _command(
                TableStructureReviewAction.RETRY_DECISION,
                {"retry": True},
            ),
        )


@pytest.mark.parametrize(
    ("action", "payload", "effect_kind", "indices", "names", "renames"),
    [
        (
            TableStructureReviewAction.REMOVE_COLUMNS,
            {"indices": (0, 2)},
            TableStructureReviewEffectKind.REQUEST_REMOVE_COLUMNS,
            (0, 2),
            (),
            (),
        ),
        (
            TableStructureReviewAction.REORDER_COLUMNS,
            {"order": [2, 0, 0]},
            TableStructureReviewEffectKind.REQUEST_REORDER_COLUMNS,
            (2, 0, 0),
            (),
            (),
        ),
        (
            TableStructureReviewAction.RENAME_COLUMNS,
            {"renames": {0: "Renamed", 2: ""}},
            TableStructureReviewEffectKind.REQUEST_RENAME_COLUMNS,
            (),
            (),
            ((0, "Renamed"), (2, "")),
        ),
        (
            TableStructureReviewAction.ADD_COLUMNS,
            {"names": ("Mail", "Provisional")},
            TableStructureReviewEffectKind.REQUEST_ADD_COLUMNS,
            (),
            ("Mail", "Provisional"),
            (),
        ),
    ],
)
def test_mutation_commands_create_effect_plans_without_applying_mutation(
    action,
    payload,
    effect_kind,
    indices,
    names,
    renames,
):
    state = initialize_review_state(_request())
    transition = transition_review_state(
        state,
        _command(action, payload),
    )

    assert transition.state_after is state
    assert transition.effect.kind is effect_kind
    assert transition.effect.indices == indices
    assert transition.effect.names == names
    assert transition.effect.renames == renames
    assert transition.effect.parser_mutation_applied is False
    assert transition.effect.learning_side_effect_applied is False


def test_candidate_navigation_wraps_index_but_requires_materialization():
    state = initialize_review_state(
        _request(candidate_index=3, candidates_total=3)
    )

    next_transition = transition_review_state(
        state,
        _command(TableStructureReviewAction.NEXT_CANDIDATE),
    )
    assert next_transition.state_after.candidate_index == 1
    assert next_transition.effect.navigation_delta == 1
    assert next_transition.effect.candidate_materialization_required is True
    assert (
        next_transition.effect.kind
        is TableStructureReviewEffectKind.REQUEST_CANDIDATE_MATERIALIZATION
    )

    previous_transition = transition_review_state(
        next_transition.state_after,
        _command(TableStructureReviewAction.PREVIOUS_CANDIDATE),
    )
    assert previous_transition.state_after.candidate_index == 3
    assert previous_transition.effect.navigation_delta == -1
    assert previous_transition.effect.candidate_materialization_required is True


def test_disallowed_primary_action_fails_closed():
    request = _request(
        allowed_actions=(
            TableStructureReviewAction.ACCEPT,
            TableStructureReviewAction.REJECT,
        )
    )
    state = initialize_review_state(request)

    with pytest.raises(
        TableStructureReviewStateMachineError,
        match="not allowed",
    ):
        transition_review_state(
            state,
            _command(
                TableStructureReviewAction.REMOVE_COLUMNS,
                {"indices": (0,)},
            ),
        )


def test_review_id_mismatch_fails_closed():
    state = initialize_review_state(_request())

    with pytest.raises(
        TableStructureReviewStateMachineError,
        match="review_id",
    ):
        transition_review_state(
            state,
            _command(
                TableStructureReviewAction.ACCEPT,
                review_id="other-review",
            ),
        )


def test_completed_state_rejects_further_commands():
    state = initialize_review_state(_request())
    completed = transition_review_state(
        state,
        _command(TableStructureReviewAction.ACCEPT),
    ).state_after

    with pytest.raises(
        TableStructureReviewStateMachineError,
        match="cannot accept commands",
    ):
        transition_review_state(
            completed,
            _command(TableStructureReviewAction.ACCEPT),
        )


def test_state_machine_source_has_no_runtime_transport_parser_or_side_effect_authority():
    import ast
    import webapp.parser.services.table_structure_review_state_machine as service

    source_path = Path(service.__file__).resolve()
    source = source_path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(source_path))

    imported_modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_modules.append(node.module or "")

    forbidden_import_fragments = {
        "table_builder",
        "legacy_adapter",
        "user_prompt",
        "logger_singleton",
        "flask",
        "socketio",
        "sqlalchemy",
        "psycopg",
        "requests",
        "playwright",
        "selenium",
    }

    assert not {
        item
        for item in imported_modules
        if any(
            fragment in item
            for fragment in forbidden_import_fragments
        )
    }

    def dotted_name(node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            left = dotted_name(node.value)
            return f"{left}.{node.attr}" if left else node.attr
        return ""

    calls = {
        dotted_name(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }

    forbidden_suffixes = {
        "prompt_user_to_confirm_table_structure",
        "harmonize_headers_and_data",
        "log_rejection_reason",
        "cache_table_structure",
        "save_table_structure_to_db",
        "build_review_request_from_legacy_preview",
        "input",
        "print",
        "open",
        "emit",
        "uuid4",
        "now",
        "utcnow",
        "time",
    }

    forbidden_calls = {
        call
        for call in calls
        if any(
            call == suffix or call.endswith("." + suffix)
            for suffix in forbidden_suffixes
        )
    }

    assert forbidden_calls == set()