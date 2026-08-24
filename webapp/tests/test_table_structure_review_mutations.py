from __future__ import annotations

import copy
from pathlib import Path

import pytest

from webapp.parser.services.table_structure_review_mutations import (
    CANONICAL_AUTHORITY,
    HARMONIZATION_INCLUDED,
    INPUT_OBJECT_MUTATION,
    LEARNING_SIDE_EFFECT_AUTHORITY,
    RUNTIME_CALLER_WIRED,
    SERVICE_VERSION,
    STATE_MACHINE_WIRED,
    TRANSPORT_AUTHORITY,
    TableStructureReviewMutationError,
    apply_add_columns,
    apply_remove_columns,
    apply_rename_columns,
    apply_reorder_columns,
)


def _legacy_safe_get(row, key, default=""):
    try:
        return row.get(key, default)
    except Exception:
        return default


def _legacy_remove_reference(headers, rows, wrong_idxs):
    candidate_headers = list(headers)
    data = copy.deepcopy(rows)

    candidate_headers = [
        h
        for i, h in enumerate(candidate_headers)
        if i not in wrong_idxs
    ]
    data = [
        {
            h: _legacy_safe_get(row, h, "")
            for h in candidate_headers
        }
        for row in data
    ]

    return candidate_headers, data


def _legacy_reorder_reference(headers, rows, order_indices):
    candidate_headers = list(headers)
    data = copy.deepcopy(rows)

    new_order = [
        candidate_headers[index]
        for index in order_indices
    ]

    if new_order:
        candidate_headers = new_order
        data = [
            {
                h: _legacy_safe_get(row, h, "")
                for h in candidate_headers
            }
            for row in data
        ]

    return candidate_headers, data


def _legacy_rename_reference(headers, rows, renames):
    candidate_headers = list(headers)
    data = copy.deepcopy(rows)

    for index, raw_name in renames.items():
        new_name = raw_name.strip()
        if new_name:
            candidate_headers[index] = new_name

    data = [
        {
            h: _legacy_safe_get(row, h, "")
            for h in candidate_headers
        }
        for row in data
    ]

    return candidate_headers, data


def _legacy_add_reference(headers, rows, names):
    candidate_headers = list(headers)
    data = copy.deepcopy(rows)

    for raw_name in names:
        col = raw_name.strip()
        if col and col not in candidate_headers:
            candidate_headers.append(col)
            for row in data:
                row[col] = _legacy_safe_get(row, col, "")

    for row in data:
        for col in candidate_headers:
            if col not in row:
                row[col] = ""

    return candidate_headers, data


def _assert_result_matches_reference(result, expected_headers, expected_rows):
    assert list(result.headers) == expected_headers
    assert [dict(row) for row in result.rows] == expected_rows


def test_service_authority_is_pure_and_unwired():
    assert SERVICE_VERSION == "table_structure_review_mutations_v1"
    assert RUNTIME_CALLER_WIRED is False
    assert STATE_MACHINE_WIRED is False
    assert HARMONIZATION_INCLUDED is False
    assert LEARNING_SIDE_EFFECT_AUTHORITY is False
    assert TRANSPORT_AUTHORITY is False
    assert INPUT_OBJECT_MUTATION is False
    assert CANONICAL_AUTHORITY is False


@pytest.mark.parametrize(
    "indices",
    [
        (0,),
        (1, 3),
        (1, 1),
        (-1,),
        (99,),
        (0, 2, 99, -1),
    ],
)
def test_remove_matches_frozen_legacy_pre_harmonization_semantics(indices):
    headers = ["A", "B", "C", "D"]
    rows = [
        {"A": None, "B": 0, "C": -4, "D": "x", "Extra": 17},
        {"A": "a", "C": 3},
    ]
    original_headers = copy.deepcopy(headers)
    original_rows = copy.deepcopy(rows)

    expected_headers, expected_rows = _legacy_remove_reference(
        headers,
        rows,
        list(indices),
    )
    result = apply_remove_columns(
        headers,
        rows,
        indices,
    )

    _assert_result_matches_reference(
        result,
        expected_headers,
        expected_rows,
    )
    assert headers == original_headers
    assert rows == original_rows


@pytest.mark.parametrize(
    "order",
    [
        (0, 1, 2),
        (2, 0),
        (2, 0, 0),
        (1,),
    ],
)
def test_reorder_matches_frozen_legacy_pre_harmonization_semantics(order):
    headers = ["A", "B", "C"]
    rows = [
        {"A": None, "B": 0, "C": -4, "Extra": "keep-only-if-add"},
        {"A": "a"},
    ]
    original_headers = copy.deepcopy(headers)
    original_rows = copy.deepcopy(rows)

    expected_headers, expected_rows = _legacy_reorder_reference(
        headers,
        rows,
        list(order),
    )
    result = apply_reorder_columns(
        headers,
        rows,
        order,
    )

    _assert_result_matches_reference(
        result,
        expected_headers,
        expected_rows,
    )
    assert headers == original_headers
    assert rows == original_rows


@pytest.mark.parametrize("bad_order", [(-1,), (3,), (0, 99)])
def test_reorder_fails_closed_outside_legacy_adapter_filtered_domain(bad_order):
    with pytest.raises(
        TableStructureReviewMutationError,
        match="outside current headers",
    ):
        apply_reorder_columns(
            ["A", "B", "C"],
            [{"A": 1}],
            bad_order,
        )


@pytest.mark.parametrize(
    "renames",
    [
        {0: "Renamed A"},
        {0: " Renamed A ", 2: ""},
        {1: "B"},
        {0: "ExistingNewKey"},
    ],
)
def test_rename_matches_frozen_legacy_pre_harmonization_semantics(renames):
    headers = ["A", "B", "C"]
    rows = [
        {
            "A": None,
            "B": 0,
            "C": -4,
            "ExistingNewKey": 777,
            "Extra": "not projected",
        },
        {"A": 5, "B": 6, "C": 7},
    ]
    original_headers = copy.deepcopy(headers)
    original_rows = copy.deepcopy(rows)

    expected_headers, expected_rows = _legacy_rename_reference(
        headers,
        rows,
        renames,
    )
    result = apply_rename_columns(
        headers,
        rows,
        renames,
    )

    _assert_result_matches_reference(
        result,
        expected_headers,
        expected_rows,
    )
    assert headers == original_headers
    assert rows == original_rows


def test_rename_explicitly_preserves_legacy_new_key_projection_behavior():
    result = apply_rename_columns(
        ["Old", "Zero", "Signed"],
        [
            {
                "Old": 42,
                "Zero": 0,
                "Signed": -4,
            }
        ],
        {0: "New"},
    )

    assert result.headers == ("New", "Zero", "Signed")
    assert result.rows[0]["New"] == ""
    assert result.rows[0]["Zero"] == 0
    assert result.rows[0]["Signed"] == -4


@pytest.mark.parametrize("bad_renames", [{-1: "X"}, {3: "X"}])
def test_rename_fails_closed_outside_legacy_adapter_filtered_domain(
    bad_renames,
):
    with pytest.raises(
        TableStructureReviewMutationError,
        match="outside current headers",
    ):
        apply_rename_columns(
            ["A", "B", "C"],
            [{"A": 1}],
            bad_renames,
        )


@pytest.mark.parametrize(
    "names",
    [
        ("C",),
        (" C ", "D"),
        ("A", "C", "C", "D", ""),
        ("ExistingExtra",),
    ],
)
def test_add_matches_frozen_legacy_pre_harmonization_semantics(names):
    headers = ["A", "B"]
    rows = [
        {
            "A": None,
            "B": 0,
            "Signed": -4,
            "ExistingExtra": 91,
        },
        {
            "A": 1,
            "Signed": -4,
        },
    ]
    original_headers = copy.deepcopy(headers)
    original_rows = copy.deepcopy(rows)

    expected_headers, expected_rows = _legacy_add_reference(
        headers,
        rows,
        names,
    )
    result = apply_add_columns(
        headers,
        rows,
        names,
    )

    _assert_result_matches_reference(
        result,
        expected_headers,
        expected_rows,
    )
    assert headers == original_headers
    assert rows == original_rows


def test_add_preserves_extra_row_keys_unlike_projection_mutations():
    result = apply_add_columns(
        ["A"],
        [{"A": 1, "Extra": -4}],
        ("B",),
    )

    assert result.headers == ("A", "B")
    assert dict(result.rows[0]) == {
        "A": 1,
        "Extra": -4,
        "B": "",
    }


def test_remove_reorder_rename_projection_preserve_none_zero_signed_if_key_survives():
    rows = [{"A": None, "B": 0, "C": -4}]

    removed = apply_remove_columns(
        ["A", "B", "C"],
        rows,
        (99,),
    )
    assert dict(removed.rows[0]) == {
        "A": None,
        "B": 0,
        "C": -4,
    }

    reordered = apply_reorder_columns(
        ["A", "B", "C"],
        rows,
        (2, 1, 0),
    )
    assert dict(reordered.rows[0]) == {
        "C": -4,
        "B": 0,
        "A": None,
    }

    renamed_same = apply_rename_columns(
        ["A", "B", "C"],
        rows,
        {0: "A"},
    )
    assert dict(renamed_same.rows[0]) == {
        "A": None,
        "B": 0,
        "C": -4,
    }


def test_pure_mutation_source_has_no_harmonization_runtime_or_side_effect_authority():
    import ast
    import webapp.parser.services.table_structure_review_mutations as service

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
        "state_machine",
        "legacy_adapter",
        "user_prompt",
        "logger_singleton",
        "shared_logic",
        "detect",
        "flask",
        "socketio",
        "sqlalchemy",
        "psycopg",
        "requests",
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
        "harmonize_headers_and_data",
        "safe_get",
        "safe_append",
        "log_rejection_reason",
        "cache_table_structure",
        "save_table_structure_to_db",
        "prompt_user_to_confirm_table_structure",
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