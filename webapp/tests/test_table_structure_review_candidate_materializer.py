"""C2G 2.8.30 tests for pure non-projecting candidate materialization."""

from __future__ import annotations

from pathlib import Path

import pytest

from webapp.parser.contracts.table_structure_review_candidates import (
    TableStructureReviewCandidateCatalog,
    TableStructureReviewCandidateHeaderProposal,
    TableStructureReviewCandidateRowBasis,
)
from webapp.parser.services.table_structure_review_candidate_materializer import (
    TableStructureReviewCandidateMaterializationError,
    materialize_table_structure_candidate,
)


WEBAPP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WEBAPP_ROOT.parent


def _proposal(index, *headers):
    return TableStructureReviewCandidateHeaderProposal(
        candidate_index=index,
        headers=tuple(headers),
    )


def _catalog(basis):
    return TableStructureReviewCandidateCatalog(
        review_id="review-c2g-2830",
        row_basis=basis,
        candidates=(
            _proposal(1, "District", "Candidate", "Value"),
            _proposal(
                2,
                "District",
                "Suggested Header",
                "Value",
                "Suggested Header",
            ),
        ),
    )


def test_materializer_uses_immutable_original_rows_when_explicitly_declared():
    catalog = _catalog(
        TableStructureReviewCandidateRowBasis.IMMUTABLE_ORIGINAL_ROWS
    )
    original = [
        {
            "District": "__ORIGINAL_DISTRICT__",
            "Candidate": "__ORIGINAL_ENTITY__",
            "Value": 0,
            "Original Only": -4,
        }
    ]
    working = [
        {
            "District": "__WORKING_DISTRICT__",
            "Candidate": "__WORKING_ENTITY__",
            "Value": 99,
        }
    ]

    result = materialize_table_structure_candidate(
        catalog,
        2,
        immutable_original_rows=original,
        current_working_rows=working,
    )

    assert result.row_basis is (
        TableStructureReviewCandidateRowBasis.IMMUTABLE_ORIGINAL_ROWS
    )
    assert result.rows[0]["District"] == "__ORIGINAL_DISTRICT__"
    assert result.rows[0]["Value"] == 0
    assert result.rows[0]["Original Only"] == -4


def test_materializer_uses_current_working_rows_when_explicitly_declared():
    catalog = _catalog(
        TableStructureReviewCandidateRowBasis.CURRENT_WORKING_ROWS
    )
    original = [
        {
            "District": "__ORIGINAL_DISTRICT__",
            "Value": 1,
        }
    ]
    working = [
        {
            "District": "__WORKING_DISTRICT__",
            "Value": -4,
            "Working Only": None,
        }
    ]

    result = materialize_table_structure_candidate(
        catalog,
        2,
        immutable_original_rows=original,
        current_working_rows=working,
    )

    assert result.row_basis is (
        TableStructureReviewCandidateRowBasis.CURRENT_WORKING_ROWS
    )
    assert result.rows[0]["District"] == "__WORKING_DISTRICT__"
    assert result.rows[0]["Value"] == -4
    assert result.rows[0]["Working Only"] is None


def test_materialization_preserves_candidate_header_order_duplicates_and_blanks():
    catalog = TableStructureReviewCandidateCatalog(
        review_id="review-c2g-2830",
        row_basis=(
            TableStructureReviewCandidateRowBasis.IMMUTABLE_ORIGINAL_ROWS
        ),
        candidates=(
            _proposal(
                1,
                "Value",
                "",
                "Value",
                "District",
            ),
        ),
    )

    result = materialize_table_structure_candidate(
        catalog,
        1,
        immutable_original_rows=[
            {
                "District": "__LOCATION__",
                "Value": 0,
            }
        ],
    )

    assert result.headers == (
        "Value",
        "",
        "Value",
        "District",
    )


def test_materializer_does_not_project_synthesize_or_drop_row_keys():
    catalog = _catalog(
        TableStructureReviewCandidateRowBasis.IMMUTABLE_ORIGINAL_ROWS
    )
    source_rows = [
        {
            "District": "__LOCATION__",
            "Candidate": "__ENTITY__",
            "Value": None,
            "Extra Evidence": "__PRESERVE_ME__",
        }
    ]

    result = materialize_table_structure_candidate(
        catalog,
        2,
        immutable_original_rows=source_rows,
    )

    # Candidate proposal contains "Suggested Header", but navigation/materialization
    # does not synthesize a missing key or blank.
    assert "Suggested Header" not in result.rows[0]

    # Existing evidence not present in the proposal is not dropped.
    assert result.rows[0]["Extra Evidence"] == "__PRESERVE_ME__"

    assert result.row_projection_applied is False
    assert result.missing_header_values_synthesized is False
    assert result.extra_row_keys_dropped is False
    assert result.harmonization_applied is False
    assert result.source_rows_mutated is False


def test_materializer_deep_copy_isolates_source_rows_and_freezes_result_mapping():
    catalog = _catalog(
        TableStructureReviewCandidateRowBasis.CURRENT_WORKING_ROWS
    )
    source_rows = [
        {
            "District": "__LOCATION__",
            "Value": {
                "nested": [1, 2],
            },
        }
    ]

    result = materialize_table_structure_candidate(
        catalog,
        1,
        current_working_rows=source_rows,
    )

    source_rows[0]["District"] = "__CALLER_MUTATION__"
    source_rows[0]["Value"]["nested"].append(3)

    assert result.rows[0]["District"] == "__LOCATION__"
    assert result.rows[0]["Value"]["nested"] == [1, 2]

    with pytest.raises(TypeError):
        result.rows[0]["District"] = "__ILLEGAL__"


@pytest.mark.parametrize(
    ("basis", "kwargs", "message"),
    [
        (
            TableStructureReviewCandidateRowBasis.IMMUTABLE_ORIGINAL_ROWS,
            {},
            "immutable_original_rows is required",
        ),
        (
            TableStructureReviewCandidateRowBasis.CURRENT_WORKING_ROWS,
            {},
            "current_working_rows is required",
        ),
    ],
)
def test_materializer_fails_closed_when_selected_row_basis_is_missing(
    basis,
    kwargs,
    message,
):
    catalog = _catalog(basis)

    with pytest.raises(
        TableStructureReviewCandidateMaterializationError,
        match=message,
    ):
        materialize_table_structure_candidate(
            catalog,
            1,
            **kwargs,
        )


@pytest.mark.parametrize(
    "bad_rows",
    [
        "not rows",
        b"not rows",
        123,
        [1, 2],
        [{"ok": 1}, 2],
    ],
)
def test_materializer_validates_selected_rows_before_normalization(bad_rows):
    catalog = _catalog(
        TableStructureReviewCandidateRowBasis.IMMUTABLE_ORIGINAL_ROWS
    )

    with pytest.raises(TableStructureReviewCandidateMaterializationError):
        materialize_table_structure_candidate(
            catalog,
            1,
            immutable_original_rows=bad_rows,
        )


def test_materializer_rejects_invalid_candidate_index_fail_closed():
    catalog = _catalog(
        TableStructureReviewCandidateRowBasis.IMMUTABLE_ORIGINAL_ROWS
    )

    for index in (0, 3, -1, True):
        with pytest.raises(
            TableStructureReviewCandidateMaterializationError
        ):
            materialize_table_structure_candidate(
                catalog,
                index,
                immutable_original_rows=[],
            )


def test_null_zero_signed_values_are_preserved_exactly():
    catalog = _catalog(
        TableStructureReviewCandidateRowBasis.CURRENT_WORKING_ROWS
    )

    result = materialize_table_structure_candidate(
        catalog,
        1,
        current_working_rows=[
            {
                "Null Evidence": None,
                "Zero Evidence": 0,
                "Signed Evidence": -4,
            }
        ],
    )

    assert result.rows[0]["Null Evidence"] is None
    assert result.rows[0]["Zero Evidence"] == 0
    assert result.rows[0]["Signed Evidence"] == -4


def test_materializer_has_zero_non_test_callers_and_no_session_integration():
    service_path = (
        WEBAPP_ROOT
        / "parser"
        / "services"
        / "table_structure_review_candidate_materializer.py"
    )
    tests_root = WEBAPP_ROOT / "tests"

    runtime_hits = []

    for path in WEBAPP_ROOT.rglob("*.py"):
        if path == service_path:
            continue

        if tests_root in path.parents:
            continue

        source = path.read_text(encoding="utf-8")

        if (
            "table_structure_review_candidate_materializer" in source
            or "materialize_table_structure_candidate" in source
        ):
            runtime_hits.append(
                path.relative_to(REPO_ROOT).as_posix()
            )

    assert runtime_hits == []


def test_materializer_source_has_no_projection_harmonization_or_side_effect_calls():
    service_path = (
        WEBAPP_ROOT
        / "parser"
        / "services"
        / "table_structure_review_candidate_materializer.py"
    )
    source = service_path.read_text(encoding="utf-8").lower()

    forbidden = (
        "harmonize_headers_and_data",
        "harmonize_table_structure_review",
        "finalize_election_output",
        "session.add",
        "commit(",
        "flush(",
        "execute(",
        "emit(",
        "publish(",
    )

    for token in forbidden:
        assert token not in source
