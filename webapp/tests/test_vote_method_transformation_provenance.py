"""C2G 1.6 first true semantic transformation provenance contract."""

from __future__ import annotations

from webapp.parser.Context_Integration.Context_Library.constants import (
    BALLOT_NAME_CANON_MAP,
)
from webapp.parser.contracts.table_pipeline import (
    TableStage,
    collect_transformations,
)
from webapp.parser.utils.salvage import (
    collapse_ballot_synonym_columns,
    normalize_ballot_column_name,
)


def _reviewed_alias_pair() -> tuple[str, str]:
    pairs = sorted(
        (str(raw), str(canon))
        for raw, canon in BALLOT_NAME_CANON_MAP.items()
        if raw and canon
        and str(raw) != str(canon)
        and normalize_ballot_column_name(str(raw)) == str(canon)
    )
    assert pairs, "Expected at least one reviewed ballot-name alias mapping"
    return pairs[0]


def test_legacy_normalization_without_collector_emits_no_observability_side_effect() -> None:
    raw, canon = _reviewed_alias_pair()

    headers, rows = collapse_ballot_synonym_columns(
        [raw],
        [{raw: -4}],
    )

    assert headers == [canon, "Total Vote"] or headers == [canon]
    assert rows[0][canon] == -4


def test_reviewed_vote_method_alias_emits_before_after_semantic_record() -> None:
    raw, canon = _reviewed_alias_pair()

    with collect_transformations() as records:
        headers, rows = collapse_ballot_synonym_columns(
            [raw],
            [{raw: -4}],
        )

    semantic = [
        record
        for record in records
        if record.operation == "vote_method_header_canonicalization"
    ]

    assert len(semantic) == 1
    record = semantic[0]

    assert record.sequence == 0
    assert record.from_stage is TableStage.INTERPRETED
    assert record.to_stage is TableStage.INTERPRETED
    assert record.confidence is None
    assert (
        record.rule_source
        == "Context_Integration.Context_Library.constants.BALLOT_NAME_CANON_MAP"
    )
    assert record.details["before_header"] == raw
    assert record.details["after_header"] == canon
    assert record.details["rule_kind"] == "reviewed_ballot_name_canon_map_direct_match"
    assert record.details["header_semantic_label_changed"] is True
    assert record.details["vote_value_mutation"] is False

    # The semantic label changes; the vote evidence does not.
    assert rows[0][canon] == -4
    assert raw not in rows[0]


def test_unknown_or_unmapped_header_does_not_claim_reviewed_mapping() -> None:
    raw = "Completely Unknown Election Method XYZ"
    assert raw.lower() not in BALLOT_NAME_CANON_MAP

    with collect_transformations() as records:
        collapse_ballot_synonym_columns(
            [raw],
            [{raw: None}],
        )

    semantic = [
        record
        for record in records
        if record.operation == "vote_method_header_canonicalization"
    ]
    assert semantic == []


def test_collector_preserves_null_zero_and_signed_vote_values() -> None:
    aliases = sorted(
        (str(raw), str(canon))
        for raw, canon in BALLOT_NAME_CANON_MAP.items()
        if raw and canon
        and str(raw) != str(canon)
        and normalize_ballot_column_name(str(raw)) == str(canon)
    )
    assert aliases

    raw, canon = aliases[0]

    for value in (None, 0, -4):
        with collect_transformations():
            _, rows = collapse_ballot_synonym_columns(
                [raw],
                [{raw: value}],
            )

        assert rows[0][canon] is value if value is None else rows[0][canon] == value