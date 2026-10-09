"""O4F record-instance public detail: only the smallest backend contract."""
from __future__ import annotations

from uuid import uuid4
from types import SimpleNamespace

import pytest

from webapp.parser.auth.capability_policy import (
    Capability, CapabilityPolicyError, assert_public_read_surface,
)
from webapp.parser.services import canonical_election_reader as reader


def _sample_record():
    item = {name: None for name in reader._CANONICAL_RECORD_PUBLIC_FIELDS}
    item.update({"id": str(uuid4()), "candidate": "Example", "total_votes": 0})
    item.update({
        "source_row_hash": "internal", "source_url": "not-public",
        "selected_dl_source": "DL1", "provenance": {"internal": True},
        "vote_components": [
            {"vote_method": "Election Day", "votes": 0, "source_column": "secret"},
            {"vote_method": "Adjustment", "votes": -2, "source_column": None},
        ],
    })
    return item


def test_projection_allowlist_and_numeric_semantics():
    public = reader._project_public_canonical_record(_sample_record())
    assert set(public) == set(reader._CANONICAL_RECORD_PUBLIC_FIELDS) | {"vote_components"}
    assert public["total_votes"] == 0
    assert public["election_date"] is None
    assert public["vote_components"] == [
        {"vote_method": "Election Day", "votes": 0},
        {"vote_method": "Adjustment", "votes": -2},
    ]
    assert not ({"source_url", "source_row_hash", "provenance", "selected_dl_source"} & set(public))


def test_empty_components_are_not_synthesized():
    item = _sample_record()
    item["vote_components"] = []
    assert reader._project_public_canonical_record(item)["vote_components"] == []


def test_exact_uuid_statement_has_no_collection_limit_or_ordering():
    record_id = uuid4()
    stmt = reader._build_result_statement(
        reader.CanonicalResultFilters(), exact_result_id=record_id,
    )
    sql = str(stmt).lower()
    assert "where canonical_election_results.id =" in sql
    assert " limit " not in sql
    assert " order by " not in sql


def test_existing_collection_builder_unchanged():
    sql = str(reader._build_result_statement(reader.CanonicalResultFilters())).lower()
    assert " order by " in sql
    assert " limit " in sql


def test_public_guard_is_read_only():
    assert assert_public_read_surface("ballotlens_canonical", "GET") == Capability.PUBLIC_READ
    with pytest.raises(CapabilityPolicyError):
        assert_public_read_surface("ballotlens_canonical", "POST")


class _Tx:
    def __init__(self):
        self.rolled_back = False
    def rollback(self):
        self.rolled_back = True


class _Conn:
    def __init__(self, row=None):
        self.dialect = SimpleNamespace(name="sqlite")
        self.row = row
        self.tx = _Tx()
        self.calls = 0
    def __enter__(self):
        return self
    def __exit__(self, *_):
        return False
    def begin(self):
        return self.tx
    def execute(self, stmt):
        self.calls += 1
        return SimpleNamespace(one_or_none=lambda: self.row)


class _Engine:
    def __init__(self, conn):
        self.conn = conn
    def connect(self):
        return self.conn


def test_missing_uuid_returns_none_and_rolls_back():
    conn = _Conn(row=None)
    assert reader.query_canonical_record_by_id(_Engine(conn), uuid4()) is None
    assert conn.calls == 1 and conn.tx.rolled_back


def test_existing_row_only_exposes_public_projection(monkeypatch):
    conn = _Conn(row=object())
    monkeypatch.setattr(reader, "_serialize_result", lambda unused: _sample_record())
    monkeypatch.setattr(reader, "_attach_components", lambda unused_conn, unused_items: None)
    public = reader.query_canonical_record_by_id(_Engine(conn), uuid4())
    assert public is not None
    assert public["total_votes"] == 0
    assert public["vote_components"][1]["votes"] == -2
    assert "source_url" not in public
    assert conn.tx.rolled_back
