# Permanent missing-value correctness tests for the FEC handler.
#
# This tranche intentionally does NOT unify CSV/XLSX fuzzy branch reachability.
# Identity-authority behavior remains for a later policy migration.

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pytest

from webapp.parser.handlers import fec_handler
from webapp.parser.utils import fec_utils


CANDIDATE = {
    "Cand_Name": "SMITH, JOHN",
    "Cand_Party_Affiliation": "DEM",
    "Cand_Office": "H",
    "Cand_Office_St": "TX",
}

MATCH_84 = {
    "cand_id": "CAND_A",
    "record": CANDIDATE,
    "score": 84,
    "method": "test",
}

MATCH_85 = {
    "cand_id": "CAND_A",
    "record": CANDIDATE,
    "score": 85,
    "method": "test",
}


def _write_source(
    tmp_path: Path,
    fmt: str,
    headers: list[str],
    row: dict[str, Any],
) -> Path:
    if fmt == "csv":
        path = tmp_path / "source.csv"
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=headers)
            writer.writeheader()
            writer.writerow(row)
        return path

    if fmt == "xlsx":
        pd = pytest.importorskip("pandas")
        path = tmp_path / "source.xlsx"
        frame = pd.DataFrame(
            [[row.get(header, None) for header in headers]],
            columns=headers,
        )
        frame.to_excel(path, index=False)
        return path

    raise AssertionError(fmt)


def _parse_one(
    monkeypatch,
    tmp_path: Path,
    *,
    fmt: str,
    headers: list[str],
    row: dict[str, Any],
    direct_record: dict[str, Any] | None,
    fuzzy_match: dict[str, Any] | None,
):
    direct_calls: list[str] = []
    fuzzy_calls: list[dict[str, Any]] = []

    def fake_direct(candidate_id):
        direct_calls.append(str(candidate_id))
        return direct_record

    def fake_fuzzy(name, state=None, party=None, cutoff=None, scorer=None, top_k=1):
        fuzzy_calls.append(
            {
                "name": name,
                "state": state,
                "party": party,
                "cutoff": cutoff,
                "top_k": top_k,
            }
        )
        return fuzzy_match

    monkeypatch.setattr(fec_handler, "get_candidate_by_id", fake_direct)
    monkeypatch.setattr(fec_handler, "find_candidate_by_name", fake_fuzzy)
    monkeypatch.setattr(fec_utils, "_append_ambiguous_log", lambda *a, **k: None)

    source = _write_source(tmp_path, fmt, headers, row)
    result = fec_handler.parse(
        None,
        None,
        context={"contest": "FEC Missing Values"},
        manual_file=str(source),
    )

    assert result is not None
    _headers, rows, _contest, _metadata = result
    assert len(rows) == 1
    return rows[0], direct_calls, fuzzy_calls


@pytest.mark.parametrize("fmt", ["csv", "xlsx"])
def test_resolved_id_can_fill_blank_party_and_name(
    monkeypatch,
    tmp_path,
    fmt,
):
    out, direct_calls, fuzzy_calls = _parse_one(
        monkeypatch,
        tmp_path,
        fmt=fmt,
        headers=["candidate_id", "candidate_name", "party", "state"],
        row={
            "candidate_id": "CAND_A",
            "candidate_name": "",
            "party": "",
            "state": "TX",
        },
        direct_record=CANDIDATE,
        fuzzy_match=None,
    )

    assert direct_calls == ["CAND_A"]
    assert fuzzy_calls == []
    assert out["candidate_name"] == "SMITH, JOHN"
    assert out["party"] == "DEM"
    assert out["_fec_candidate"] is CANDIDATE


def test_xlsx_blank_candidate_id_does_not_escape_as_nan(
    monkeypatch,
    tmp_path,
):
    out, direct_calls, fuzzy_calls = _parse_one(
        monkeypatch,
        tmp_path,
        fmt="xlsx",
        headers=["candidate_id", "candidate_name", "party", "state"],
        row={
            "candidate_id": "",
            "candidate_name": "John Smith",
            "party": "",
            "state": "TX",
        },
        direct_record=None,
        fuzzy_match=MATCH_84,
    )

    # Existing XLSX topology: no real ID -> fuzzy-name branch.
    assert direct_calls == []
    assert len(fuzzy_calls) == 1
    assert fuzzy_calls[0]["cutoff"] == 80
    assert out["candidate_id"] == ""
    assert str(out["candidate_id"]).lower() != "nan"
    assert out["party"] == "UNKNOWN"
    assert out["_fec_candidate_match"] is MATCH_84
    assert "_fec_candidate" not in out


def test_xlsx_blank_party_does_not_become_other(
    monkeypatch,
    tmp_path,
):
    out, _, _ = _parse_one(
        monkeypatch,
        tmp_path,
        fmt="xlsx",
        headers=["candidate_id", "candidate_name", "party", "state"],
        row={
            "candidate_id": "",
            "candidate_name": "John Smith",
            "party": "",
            "state": "TX",
        },
        direct_record=None,
        fuzzy_match=None,
    )

    assert out["party"] == "UNKNOWN"
    assert out["party"] != "OTHER"


def test_csv_no_id_branch_topology_is_preserved(
    monkeypatch,
    tmp_path,
):
    out, direct_calls, fuzzy_calls = _parse_one(
        monkeypatch,
        tmp_path,
        fmt="csv",
        headers=["candidate_id", "candidate_name", "party", "state"],
        row={
            "candidate_id": "",
            "candidate_name": "John Smith",
            "party": "",
            "state": "TX",
        },
        direct_record=None,
        fuzzy_match=MATCH_85,
    )

    assert direct_calls == []
    assert fuzzy_calls == []
    assert "_fec_candidate_match" not in out
    assert "_fec_candidate" not in out


def test_csv_unresolved_id_fuzzy_topology_is_preserved(
    monkeypatch,
    tmp_path,
):
    out, direct_calls, fuzzy_calls = _parse_one(
        monkeypatch,
        tmp_path,
        fmt="csv",
        headers=["candidate_id", "candidate_name", "party", "state"],
        row={
            "candidate_id": "UNKNOWN_ID",
            "candidate_name": "John Smith",
            "party": "",
            "state": "TX",
        },
        direct_record=None,
        fuzzy_match=MATCH_85,
    )

    assert direct_calls == ["UNKNOWN_ID"]
    assert len(fuzzy_calls) == 1
    assert fuzzy_calls[0]["cutoff"] == 80
    assert out["_fec_candidate_match"] is MATCH_85
    assert out["_fec_candidate"] is CANDIDATE
    assert out["party"] == "DEM"


def test_xlsx_unresolved_id_does_not_gain_new_fuzzy_fallback(
    monkeypatch,
    tmp_path,
):
    out, direct_calls, fuzzy_calls = _parse_one(
        monkeypatch,
        tmp_path,
        fmt="xlsx",
        headers=["candidate_id", "candidate_name", "party", "state"],
        row={
            "candidate_id": "UNKNOWN_ID",
            "candidate_name": "John Smith",
            "party": "",
            "state": "TX",
        },
        direct_record=None,
        fuzzy_match=MATCH_85,
    )

    assert direct_calls == ["UNKNOWN_ID"]
    assert fuzzy_calls == []
    assert "_fec_candidate_match" not in out
    assert "_fec_candidate" not in out
