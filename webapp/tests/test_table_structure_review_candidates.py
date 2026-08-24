"""C2G 2.8.29 tests for table-structure candidate catalog semantics."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest

from webapp.parser.contracts.table_structure_review_candidates import (
    CATALOG_STORES_ROWS,
    MATERIALIZER_IMPLEMENTED,
    TableStructureReviewCandidateCatalog,
    TableStructureReviewCandidateContractError,
    TableStructureReviewCandidateHeaderProposal,
    TableStructureReviewCandidateRowBasis,
)


WEBAPP_ROOT = Path(__file__).resolve().parents[1]


def _proposal(index, *headers):
    return TableStructureReviewCandidateHeaderProposal(
        candidate_index=index,
        headers=tuple(headers),
    )


def test_catalog_requires_explicit_typed_row_basis():
    with pytest.raises(TypeError):
        TableStructureReviewCandidateCatalog(
            review_id="review-c2g-2829",
            candidates=(
                _proposal(1, "District", "Candidate", "Value"),
            ),
        )

    with pytest.raises(
        TableStructureReviewCandidateContractError,
        match="row_basis",
    ):
        TableStructureReviewCandidateCatalog(
            review_id="review-c2g-2829",
            row_basis="IMMUTABLE_ORIGINAL_ROWS",
            candidates=(
                _proposal(1, "District", "Candidate", "Value"),
            ),
        )


@pytest.mark.parametrize(
    "basis",
    [
        TableStructureReviewCandidateRowBasis.IMMUTABLE_ORIGINAL_ROWS,
        TableStructureReviewCandidateRowBasis.CURRENT_WORKING_ROWS,
    ],
)
def test_both_explicit_row_basis_vocabulary_values_are_valid(basis):
    catalog = TableStructureReviewCandidateCatalog(
        review_id="review-c2g-2829",
        row_basis=basis,
        candidates=(
            _proposal(1, "District", "Candidate", "Value"),
            _proposal(2, "District", "Suggested Candidate", "Value"),
        ),
    )

    assert catalog.row_basis is basis
    assert catalog.candidates_total == 2


def test_catalog_stores_header_proposals_only_and_no_row_fields():
    proposal_field_names = {
        item.name
        for item in fields(TableStructureReviewCandidateHeaderProposal)
    }
    catalog_field_names = {
        item.name
        for item in fields(TableStructureReviewCandidateCatalog)
    }

    assert proposal_field_names == {
        "candidate_index",
        "headers",
    }
    assert catalog_field_names == {
        "review_id",
        "row_basis",
        "candidates",
    }

    forbidden_row_fields = {
        "rows",
        "data",
        "candidate_rows",
        "candidate_tables",
        "working_rows",
        "original_rows",
    }

    assert proposal_field_names.isdisjoint(forbidden_row_fields)
    assert catalog_field_names.isdisjoint(forbidden_row_fields)
    assert CATALOG_STORES_ROWS is False
    assert MATERIALIZER_IMPLEMENTED is False


def test_candidate_indices_are_contiguous_ordered_and_one_based():
    with pytest.raises(
        TableStructureReviewCandidateContractError,
        match="contiguous",
    ):
        TableStructureReviewCandidateCatalog(
            review_id="review-c2g-2829",
            row_basis=(
                TableStructureReviewCandidateRowBasis
                .IMMUTABLE_ORIGINAL_ROWS
            ),
            candidates=(
                _proposal(1, "A"),
                _proposal(3, "B"),
            ),
        )

    with pytest.raises(
        TableStructureReviewCandidateContractError,
        match="contiguous",
    ):
        TableStructureReviewCandidateCatalog(
            review_id="review-c2g-2829",
            row_basis=(
                TableStructureReviewCandidateRowBasis
                .IMMUTABLE_ORIGINAL_ROWS
            ),
            candidates=(
                _proposal(2, "B"),
                _proposal(1, "A"),
            ),
        )


def test_header_order_duplicates_and_blank_strings_are_preserved():
    proposal = _proposal(
        1,
        "Value",
        "",
        "Value",
        "District",
    )

    assert proposal.headers == (
        "Value",
        "",
        "Value",
        "District",
    )


def test_candidate_lookup_is_explicitly_one_based_and_fail_closed():
    catalog = TableStructureReviewCandidateCatalog(
        review_id="review-c2g-2829",
        row_basis=(
            TableStructureReviewCandidateRowBasis
            .CURRENT_WORKING_ROWS
        ),
        candidates=(
            _proposal(1, "A"),
            _proposal(2, "B"),
        ),
    )

    assert catalog.candidate_at(1).headers == ("A",)
    assert catalog.candidate_at(2).headers == ("B",)

    for value in (0, 3, -1, True):
        with pytest.raises(TableStructureReviewCandidateContractError):
            catalog.candidate_at(value)


def test_catalog_terminology_does_not_encode_election_candidate_semantics():
    contract_path = (
        WEBAPP_ROOT
        / "parser"
        / "contracts"
        / "table_structure_review_candidates.py"
    )
    source = contract_path.read_text(encoding="utf-8")

    # These election-result concepts have no place in a structural-header
    # proposal catalog.
    for token in (
        "ballot_party",
        "party_line",
        "vote_method",
        "total_votes",
        "election_day_votes",
        "absentee_mail",
    ):
        assert token not in source.lower()


def test_catalog_contract_has_no_session_executor_or_transport_wiring():
    contract_path = (
        WEBAPP_ROOT
        / "parser"
        / "contracts"
        / "table_structure_review_candidates.py"
    )
    source = contract_path.read_text(encoding="utf-8")

    for token in (
        "table_structure_review_session",
        "table_structure_review_effect_executor",
        "socketio",
        "flask",
        "finalize_election_output",
        "commit(",
        "execute(",
    ):
        assert token not in source.lower()
