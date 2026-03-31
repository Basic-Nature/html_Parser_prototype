from __future__ import annotations

from webapp.parser.utils.data_comparator import DataComparator


def test_data_comparator_exact_near_missing_extra():
    dl1 = {
        "candidates": [
            {"name": "Alice A", "votes": 1000, "percent": 50.0, "party": "Dem"},
            {"name": "Bob B", "votes": 900, "percent": 45.0, "party": "Rep"},
        ]
    }
    dl2 = {
        "candidates": [
            {"name": "Alice A", "votes": 1000, "percent": 50.0, "party": "Dem"},
            {"name": "Bob B", "votes": 904, "percent": 45.1, "party": "Rep"},
            {"name": "Carol C", "votes": 25, "percent": 1.2, "party": "Ind"},
        ]
    }

    comparator = DataComparator()
    result = comparator.compare_datasets(dl1, dl2)

    assert result.exact_matches == 1
    assert result.near_matches == 1
    assert len(result.mismatches) == 0
    assert result.missing_candidates == []
    assert result.extra_candidates == ["Carol C"]
    assert result.accuracy == 1.0


def test_data_comparator_detects_mismatch_over_tolerance():
    dl1 = {"candidates": [{"name": "Alice A", "votes": 1000, "percent": 50.0}]}
    dl2 = {"candidates": [{"name": "Alice A", "votes": 960, "percent": 49.0}]}

    comparator = DataComparator()
    result = comparator.compare_datasets(dl1, dl2)

    assert result.exact_matches == 0
    assert result.near_matches == 0
    assert len(result.mismatches) >= 1
    assert result.accuracy == 0.0


def test_regression_report_contract_gate_fails():
    dl1 = {"candidates": [{"name": "Alice A", "votes": 1000}]}
    dl2 = {"candidates": [{"name": "Alice A", "votes": 700}]}

    comparator = DataComparator()
    result = comparator.compare_datasets(dl1, dl2)
    report = comparator.build_regression_report(
        result,
        context={"fixture": "sample"},
        min_accuracy=0.9,
        max_mismatches=0,
    )

    assert report["schema_version"] == "1.0"
    assert report["context"]["fixture"] == "sample"
    assert report["gate"]["status"] == "fail"
    assert report["summary"]["mismatch_count"] >= 1


def test_data_comparator_normalizes_candidate_keys_and_party_case():
    dl1 = {"candidates": [{"name": " Alice A ", "votes": "1,000", "percent": "50.0%", "party": "DEM"}]}
    dl2 = {"rows": [{"Candidate": "alice a", "vote_count": 1000, "percentage": 50.0, "Party": "dem"}]}

    comparator = DataComparator()
    result = comparator.compare_datasets(dl1, dl2)

    assert result.exact_matches == 1
    assert result.mismatches == []
    assert result.accuracy == 1.0


def test_data_comparator_reports_missing_candidates_for_empty_dl2():
    dl1 = {"candidates": [{"name": "Alice A", "votes": 1000}]}
    dl2 = {"candidates": []}

    comparator = DataComparator()
    result = comparator.compare_datasets(dl1, dl2)

    assert result.exact_matches == 0
    assert result.missing_candidates == ["Alice A"]
    assert result.extra_candidates == []
    assert result.accuracy == 0.0


def test_regression_report_contract_gate_passes_within_thresholds():
    dl1 = {"candidates": [{"name": "Alice A", "votes": 1000, "percent": 50.0}]}
    dl2 = {"candidates": [{"name": "Alice A", "votes": 1002, "percent": 50.1}]}

    comparator = DataComparator()
    result = comparator.compare_datasets(dl1, dl2)
    report = comparator.build_regression_report(
        result,
        context={"fixture": "near-match"},
        min_accuracy=0.9,
        max_mismatches=0,
    )

    assert report["gate"]["status"] == "pass"
    assert report["summary"]["near_matches"] == 1
    assert report["context"]["fixture"] == "near-match"
