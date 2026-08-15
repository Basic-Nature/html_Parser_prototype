# Permanent correctness tests for FEC candidate-name lookup mechanics.
#
# These tests intentionally do not redefine FEC confidence policy. The existing
# default/manual cutoffs and handler-level 80/85 policy remain separate
# migration work.

from __future__ import annotations

from types import SimpleNamespace
import sys

import pytest

from webapp.parser import fec_lookup


@pytest.fixture(autouse=True)
def isolated_fec_index(monkeypatch):
    data = {
        "CAND_A": {
            "Cand_Name": "SMITH, JOHN",
            "Cand_Office": "H",
            "Cand_Office_St": "TX",
            "Cand_Party_Affiliation": "DEM",
        },
        "CAND_B": {
            "Cand_Name": "DOE, JANE",
            "Cand_Office": "H",
            "Cand_Office_St": "TX",
            "Cand_Party_Affiliation": "REP",
        },
        "CAND_C": {
            "Cand_Name": "SMITH, JOHN",
            "Cand_Office": "S",
            "Cand_Office_St": "TX",
            "Cand_Party_Affiliation": "DEM",
        },
    }
    monkeypatch.setattr(fec_lookup, "_CACHE", data)
    monkeypatch.setattr(fec_lookup, "_NAME_INDEX", None)
    return data


def test_rapidfuzz_mapping_key_becomes_candidate_id(monkeypatch, isolated_fec_index):
    class FakeProcess:
        @staticmethod
        def extractOne(target, choices, scorer=None, score_cutoff=None):
            assert choices["CAND_A"] == "john smith"
            assert score_cutoff == 80
            return ("john smith", 100.0, "CAND_A")

        @staticmethod
        def extract(*args, **kwargs):
            raise AssertionError("top-k path not expected")

    fake_rapidfuzz = SimpleNamespace(
        fuzz=SimpleNamespace(token_sort_ratio=object()),
        process=FakeProcess,
    )
    monkeypatch.setitem(sys.modules, "rapidfuzz", fake_rapidfuzz)

    result = fec_lookup.find_candidate_by_name(
        "John Smith",
        cutoff=80,
        scorer="rapidfuzz",
    )

    assert result is not None
    assert result["cand_id"] == "CAND_A"
    assert result["record"] is isolated_fec_index["CAND_A"]
    assert result["score"] == 100
    assert result["method"] == "rapidfuzz"


def test_rapidfuzz_top_k_uses_mapping_keys_and_cutoff(monkeypatch, isolated_fec_index):
    class FakeProcess:
        @staticmethod
        def extractOne(*args, **kwargs):
            raise AssertionError("top-1 path not expected")

        @staticmethod
        def extract(target, choices, scorer=None, limit=None, score_cutoff=None):
            assert limit == 2
            assert score_cutoff == 75
            return [
                ("john smith", 100.0, "CAND_A"),
                ("john smith", 100.0, "CAND_C"),
            ]

    fake_rapidfuzz = SimpleNamespace(
        fuzz=SimpleNamespace(token_sort_ratio=object()),
        process=FakeProcess,
    )
    monkeypatch.setitem(sys.modules, "rapidfuzz", fake_rapidfuzz)

    result = fec_lookup.find_candidate_by_name(
        "John Smith",
        cutoff=75,
        scorer="rapidfuzz",
        top_k=2,
    )

    assert result is not None
    assert result["cand_id"] == "CAND_A"
    assert [item["cand_id"] for item in result["candidates"]] == [
        "CAND_A",
        "CAND_C",
    ]
    assert all(item["record"] is not None for item in result["candidates"])


def test_rapidfuzz_below_cutoff_returns_none(monkeypatch):
    class FakeProcess:
        @staticmethod
        def extractOne(target, choices, scorer=None, score_cutoff=None):
            assert score_cutoff == 90
            return None

        @staticmethod
        def extract(*args, **kwargs):
            raise AssertionError("top-k path not expected")

    fake_rapidfuzz = SimpleNamespace(
        fuzz=SimpleNamespace(token_sort_ratio=object()),
        process=FakeProcess,
    )
    monkeypatch.setitem(sys.modules, "rapidfuzz", fake_rapidfuzz)

    result = fec_lookup.find_candidate_by_name(
        "John Smith",
        cutoff=90,
        scorer="rapidfuzz",
    )

    assert result is None


def test_actual_rapidfuzz_mapping_contract_if_installed():
    rapidfuzz = pytest.importorskip("rapidfuzz")

    choices = {
        "CAND_A": "john smith",
        "CAND_B": "jane doe",
    }
    result = rapidfuzz.process.extractOne(
        "john smith",
        choices,
        scorer=rapidfuzz.fuzz.token_sort_ratio,
    )

    assert result is not None
    assert result[0] == "john smith"
    assert result[2] == "CAND_A"


def test_difflib_top_one_honors_cutoff(isolated_fec_index):
    result = fec_lookup.find_candidate_by_name(
        "John Smith",
        cutoff=100,
        scorer="difflib",
    )

    assert result is not None
    assert result["cand_id"] == "CAND_A"
    assert result["record"] is isolated_fec_index["CAND_A"]
    assert result["score"] == 100

    no_match = fec_lookup.find_candidate_by_name(
        "John Smythe",
        cutoff=100,
        scorer="difflib",
    )
    assert no_match is None


def test_difflib_cutoff_uses_unrounded_similarity(monkeypatch):
    class FakeMatcher:
        def __init__(self, _unused, target, norm):
            self.target = target
            self.norm = norm

        def ratio(self):
            # 0.899 -> 89.9. The cutoff decision should use 89.9,
            # while the public score remains the historical integer 89.
            return 0.899

    import difflib

    monkeypatch.setattr(difflib, "SequenceMatcher", FakeMatcher)

    result = fec_lookup.find_candidate_by_name(
        "John Smith",
        cutoff=89.5,
        scorer="difflib",
        top_k=1,
    )

    assert result is not None
    assert result["score"] == 89


def test_difflib_top_k_filters_every_result_by_cutoff():
    result = fec_lookup.find_candidate_by_name(
        "John Smith",
        cutoff=100,
        scorer="difflib",
        top_k=3,
    )

    assert result is not None
    assert [item["cand_id"] for item in result["candidates"]] == [
        "CAND_A",
        "CAND_C",
    ]
    assert all(item["score"] >= 100 for item in result["candidates"])


def test_state_and_party_arguments_remain_non_scoring_in_phase_2():
    result = fec_lookup.find_candidate_by_name(
        "John Smith",
        state="ZZ",
        party="OTHER",
        cutoff=100,
        scorer="difflib",
    )

    assert result is not None
    assert result["cand_id"] == "CAND_A"
