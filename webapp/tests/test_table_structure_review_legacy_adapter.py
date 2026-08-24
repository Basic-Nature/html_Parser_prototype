from __future__ import annotations

from pathlib import Path

import pytest

from webapp.parser.contracts.table_structure_review import (
    TableStructureReviewAction,
    TableStructureReviewDecision,
)
from webapp.parser.services.table_structure_review_legacy_adapter import (
    ADAPTER_VERSION,
    CANONICAL_AUTHORITY,
    CLI_REPLACEMENT_AUTHORIZED,
    LEARNING_SIDE_EFFECT_AUTHORITY,
    PARSER_MUTATION_AUTHORITY,
    PRIMARY_ALLOWED_ACTIONS,
    REVIEW_ID_GENERATION_AUTHORITY,
    RUNTIME_CONTROL_AUTHORITY,
    TIMESTAMP_GENERATION_AUTHORITY,
    TRANSPORT_AUTHORITY,
    LegacyTableStructureReviewAdapterError,
    adapt_legacy_no_payload_primary_response,
    adapt_legacy_retry_response,
    build_add_columns_command,
    build_remove_columns_command,
    build_rename_columns_command,
    build_reorder_columns_command,
    build_review_request_from_legacy_preview,
    build_review_result_from_legacy_return,
    classify_legacy_primary_response,
)


def test_adapter_authority_is_explicitly_noncontrolling():
    assert ADAPTER_VERSION == "table_structure_review_legacy_adapter_v1"
    assert RUNTIME_CONTROL_AUTHORITY is False
    assert PARSER_MUTATION_AUTHORITY is False
    assert LEARNING_SIDE_EFFECT_AUTHORITY is False
    assert TRANSPORT_AUTHORITY is False
    assert CLI_REPLACEMENT_AUTHORIZED is False
    assert REVIEW_ID_GENERATION_AUTHORITY is False
    assert TIMESTAMP_GENERATION_AUTHORITY is False
    assert CANONICAL_AUTHORITY is False


def test_primary_allowed_actions_match_legacy_primary_prompt_not_retry_phase():
    assert PRIMARY_ALLOWED_ACTIONS == (
        TableStructureReviewAction.ACCEPT,
        TableStructureReviewAction.REJECT,
        TableStructureReviewAction.REMOVE_COLUMNS,
        TableStructureReviewAction.REORDER_COLUMNS,
        TableStructureReviewAction.RENAME_COLUMNS,
        TableStructureReviewAction.ADD_COLUMNS,
        TableStructureReviewAction.NEXT_CANDIDATE,
        TableStructureReviewAction.PREVIOUS_CANDIDATE,
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", TableStructureReviewAction.ACCEPT),
        (" y ", TableStructureReviewAction.ACCEPT),
        ("YES", TableStructureReviewAction.ACCEPT),
        ("n", TableStructureReviewAction.REJECT),
        (" no ", TableStructureReviewAction.REJECT),
        ("c", TableStructureReviewAction.REMOVE_COLUMNS),
        ("O", TableStructureReviewAction.REORDER_COLUMNS),
        ("r", TableStructureReviewAction.RENAME_COLUMNS),
        ("a", TableStructureReviewAction.ADD_COLUMNS),
        ("next", TableStructureReviewAction.NEXT_CANDIDATE),
        ("NXT", TableStructureReviewAction.NEXT_CANDIDATE),
        ("prev", TableStructureReviewAction.PREVIOUS_CANDIDATE),
        ("previous", TableStructureReviewAction.PREVIOUS_CANDIDATE),
        ("unknown", None),
    ],
)
def test_primary_token_classification_preserves_legacy_grammar(raw, expected):
    assert classify_legacy_primary_response(raw) is expected


def test_unknown_legacy_primary_input_is_not_typed_as_a_command():
    assert classify_legacy_primary_response("mystery") is None
    assert (
        adapt_legacy_no_payload_primary_response(
            review_id="review-1",
            raw_response="mystery",
        )
        is None
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", TableStructureReviewAction.ACCEPT),
        ("yes", TableStructureReviewAction.ACCEPT),
        ("no", TableStructureReviewAction.REJECT),
        ("next", TableStructureReviewAction.NEXT_CANDIDATE),
        ("previous", TableStructureReviewAction.PREVIOUS_CANDIDATE),
    ],
)
def test_no_payload_primary_tokens_build_typed_commands(raw, expected):
    command = adapt_legacy_no_payload_primary_response(
        review_id="review-1",
        raw_response=raw,
    )
    assert command.action is expected
    assert command.payload is None


def test_payload_primary_tokens_require_a_payload_builder():
    for raw in ("c", "o", "r", "a"):
        assert (
            adapt_legacy_no_payload_primary_response(
                review_id="review-1",
                raw_response=raw,
            )
            is None
        )


@pytest.mark.parametrize(
    ("raw", "retry"),
    [
        ("y", True),
        (" YES ", True),
        ("", False),
        ("n", False),
        ("no", False),
        ("garbage", False),
    ],
)
def test_retry_adapter_preserves_legacy_yes_else_false_semantics(raw, retry):
    command = adapt_legacy_retry_response(
        review_id="review-1",
        raw_response=raw,
    )
    assert command.action is TableStructureReviewAction.RETRY_DECISION
    assert command.payload == {"retry": retry}


def test_request_projection_preserves_legacy_preview_shape_and_values():
    rows = [
        {
            "Precinct": "P-1",
            "Unknown": None,
            "Confirmed Zero": 0,
            "Signed Value": -4,
            "Extra": "not projected",
        },
        {"Precinct": "P-2"},
    ]

    request = build_review_request_from_legacy_preview(
        review_id="caller-provided-review",
        session_id=None,
        domain="example.org",
        contest=None,
        candidate_headers=(
            "Precinct",
            "Unknown",
            "Confirmed Zero",
            "Signed Value",
        ),
        rows=rows,
        candidate_index_zero_based=0,
        candidates_total=2,
        ml_avg_confidence=None,
    )

    assert request.review_id == "caller-provided-review"
    assert request.candidate_index == 1
    assert request.candidates_total == 2
    assert request.ml_avg_confidence is None
    assert request.rows_preview[0]["Unknown"] is None
    assert request.rows_preview[0]["Confirmed Zero"] == 0
    assert request.rows_preview[0]["Signed Value"] == -4
    assert "Extra" not in request.rows_preview[0]
    assert request.rows_preview[1]["Unknown"] == ""

    assert rows[0]["Unknown"] is None
    assert rows[0]["Confirmed Zero"] == 0
    assert rows[0]["Signed Value"] == -4


def test_request_projection_is_bounded_to_first_five_rows():
    rows = [{"A": i} for i in range(10)]
    request = build_review_request_from_legacy_preview(
        review_id="review-1",
        session_id="session-1",
        domain="example.org",
        contest="Contest",
        candidate_headers=("A",),
        rows=rows,
        candidate_index_zero_based=0,
        candidates_total=1,
        ml_avg_confidence=0.5,
    )
    assert [row["A"] for row in request.rows_preview] == [0, 1, 2, 3, 4]


def test_remove_columns_adapter_preserves_legacy_digit_and_bounds_filtering():
    command = build_remove_columns_command(
        review_id="review-1",
        candidate_headers=("A", "B", "C"),
        raw_indices="1, 3, x, 9, 0",
    )
    assert command.action is TableStructureReviewAction.REMOVE_COLUMNS
    assert command.payload == {"indices": (0, 2)}

    assert (
        build_remove_columns_command(
            review_id="review-1",
            candidate_headers=("A", "B"),
            raw_indices="x, 9",
        )
        is None
    )


def test_reorder_adapter_preserves_legacy_subset_duplicate_and_bounds_behavior():
    command = build_reorder_columns_command(
        review_id="review-1",
        candidate_headers=("A", "B", "C"),
        raw_order="3,1 1 x 9",
    )
    assert command.action is TableStructureReviewAction.REORDER_COLUMNS
    assert command.payload == {"order": (2, 0, 0)}

    assert (
        build_reorder_columns_command(
            review_id="review-1",
            candidate_headers=("A", "B"),
            raw_order="x 9",
        )
        is None
    )


def test_rename_adapter_preserves_nonempty_names_and_fails_on_lossy_duplicate_indices():
    command = build_rename_columns_command(
        review_id="review-1",
        candidate_headers=("A", "B", "C"),
        raw_indices="1,3",
        raw_new_names=(" Renamed A ", ""),
    )
    assert command.action is TableStructureReviewAction.RENAME_COLUMNS
    assert command.payload == {"renames": {0: "Renamed A"}}

    with pytest.raises(
        LegacyTableStructureReviewAdapterError,
        match="duplicate legacy rename indices",
    ):
        build_rename_columns_command(
            review_id="review-1",
            candidate_headers=("A", "B"),
            raw_indices="1,1",
            raw_new_names=("First", "Second"),
        )


def test_rename_adapter_rejects_name_count_mismatch_instead_of_guessing():
    with pytest.raises(
        LegacyTableStructureReviewAdapterError,
        match="count must match",
    ):
        build_rename_columns_command(
            review_id="review-1",
            candidate_headers=("A", "B"),
            raw_indices="1,2",
            raw_new_names=("Only One",),
        )


def test_add_columns_adapter_preserves_trim_existing_and_same_input_duplicate_suppression():
    command = build_add_columns_command(
        review_id="review-1",
        candidate_headers=("A", "B"),
        raw_names=" C, A, C,  D , ",
    )
    assert command.action is TableStructureReviewAction.ADD_COLUMNS
    assert command.payload == {"names": ("C", "D")}

    assert (
        build_add_columns_command(
            review_id="review-1",
            candidate_headers=("A",),
            raw_names=" A, ,A ",
        )
        is None
    )


def test_result_projection_preserves_null_zero_signed_values_and_decision():
    rows = (
        {
            "Unknown": None,
            "Confirmed Zero": 0,
            "Signed Value": -4,
        },
    )

    accepted = build_review_result_from_legacy_return(
        headers=("Unknown", "Confirmed Zero", "Signed Value"),
        rows=rows,
        accepted_review_structure=True,
    )
    assert (
        accepted.decision
        is TableStructureReviewDecision.ACCEPTED_REVIEW_STRUCTURE
    )
    assert accepted.rows[0]["Unknown"] is None
    assert accepted.rows[0]["Confirmed Zero"] == 0
    assert accepted.rows[0]["Signed Value"] == -4

    original = build_review_result_from_legacy_return(
        headers=("Unknown", "Confirmed Zero", "Signed Value"),
        rows=rows,
        accepted_review_structure=False,
    )
    assert (
        original.decision
        is TableStructureReviewDecision.ORIGINAL_STRUCTURE_RETAINED
    )


def test_adapter_source_has_no_runtime_control_or_side_effect_imports_and_calls():
    import ast
    import webapp.parser.services.table_structure_review_legacy_adapter as adapter

    source_path = Path(adapter.__file__).resolve()
    source = source_path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(source_path))

    imported_roots = set()
    imported_modules = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".")[0])
                imported_modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_roots.add(node.module.split(".")[0])
                imported_modules.add(node.module)

    assert imported_roots <= {
        "__future__",
        "typing",
        "contracts",
    }

    forbidden_textual_import_targets = {
        "table_builder",
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
        target
        for target in forbidden_textual_import_targets
        if any(target in module for module in imported_modules)
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
        "open",
        "input",
        "print",
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

    assert adapter.RUNTIME_CONTROL_AUTHORITY is False
    assert adapter.PARSER_MUTATION_AUTHORITY is False
    assert adapter.LEARNING_SIDE_EFFECT_AUTHORITY is False
    assert adapter.TRANSPORT_AUTHORITY is False
    assert adapter.CANONICAL_AUTHORITY is False