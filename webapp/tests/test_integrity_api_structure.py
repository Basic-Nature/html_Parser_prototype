from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

try:
    from webapp.Smart_Elections_Parser_Webapp import app
except ImportError as exc:  # pragma: no cover
    pytest.skip(f"Cannot import webapp app: {exc}", allow_module_level=True)


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def _sample_trends() -> list[dict[str, Any]]:
    return [
        {
            "schema_version": "1.1",
            "session_id": "t1",
            "generated_at": "2026-02-25T10:00:00Z",
            "confidence": {"avg": 0.9, "median": 0.9, "count": 100, "buckets": {"low": 2, "medium": 8, "high": 90}},
            "unknown_ratio": 0.05,
            "segment_count": 100,
            "unknown_segment_count": 5,
            "review_signals": {"segments_needing_review": 2, "pattern_kb_matches": 3},
        },
        {
            "schema_version": "1.1",
            "session_id": "t2",
            "generated_at": "2026-02-25T10:05:00Z",
            "confidence": {"avg": 0.75, "median": 0.76, "count": 100, "buckets": {"low": 12, "medium": 28, "high": 60}},
            "unknown_ratio": 0.16,
            "segment_count": 100,
            "unknown_segment_count": 16,
            "review_signals": {"segments_needing_review": 11, "pattern_kb_matches": 1},
        },
    ]


def test_integrity_trends_response_shape(client):
    trends = _sample_trends()
    with patch("webapp.Smart_Elections_Parser_Webapp._load_integrity_trends", return_value=(trends, "mock-source", False)):
        resp = client.get("/api/integrity_trends")

    assert resp.status_code == 200
    payload = resp.get_json()

    assert payload["count"] == 2
    assert payload["source"] == "mock-source"
    assert payload["from_cache"] is False
    assert isinstance(payload["trends"], list)


def test_integrity_route_method_contracts():
    rules = {rule.rule: rule for rule in app.url_map.iter_rules()}

    trends_rule = rules.get("/api/integrity_trends")
    signal_rule = rules.get("/api/integrity_signal")

    assert trends_rule is not None
    assert signal_rule is not None
    assert "GET" in trends_rule.methods
    assert "POST" not in trends_rule.methods
    assert "POST" in signal_rule.methods
    assert "GET" not in signal_rule.methods


def test_integrity_signal_response_shape_and_status(client):
    trends = _sample_trends()
    with patch("webapp.Smart_Elections_Parser_Webapp._load_integrity_trends", return_value=(trends, "mock-source", False)):
        resp = client.post(
            "/api/integrity_signal",
            json={
                "confDropThreshold": 0.08,
                "unknownSpikeThreshold": 0.1,
                "reviewSpikeThreshold": 5.0,
                "baselineWindow": 30,
                "recentWindow": 5,
            },
        )

    assert resp.status_code == 200
    payload = resp.get_json()

    assert payload["source"] == "mock-source"
    assert payload["from_cache"] is False
    assert "signal" in payload
    assert payload["signal"]["status"] in {"ok", "alert", "insufficient_data", "error"}
    assert "alerts" in payload["signal"]
    assert "entry_count" in payload["signal"]


def test_integrity_signal_insufficient_data(client):
    with patch("webapp.Smart_Elections_Parser_Webapp._load_integrity_trends", return_value=([], "", False)):
        resp = client.post("/api/integrity_signal", json={})

    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["signal"]["status"] == "insufficient_data"
    assert payload["signal"]["entry_count"] == 0


def test_integrity_trends_recovers_from_malformed_primary_file(client):
    repo_root = Path(__file__).resolve().parents[2]
    primary_path = repo_root / "tools" / "debug_headless_output" / "context_digest_trends.json"
    cache_path = repo_root / "tools" / "tmp" / "integrity_trends_last.json"

    primary_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    primary_backup = primary_path.read_text(encoding="utf-8") if primary_path.exists() else None
    cache_backup = cache_path.read_text(encoding="utf-8") if cache_path.exists() else None

    cached_trends = [
        {
            "schema_version": "1.1",
            "session_id": "cache-1",
            "generated_at": "2026-02-25T10:00:00Z",
            "confidence": {"avg": 0.88, "median": 0.88, "count": 50, "buckets": {"low": 2, "medium": 8, "high": 40}},
            "unknown_ratio": 0.08,
            "segment_count": 50,
            "unknown_segment_count": 4,
            "review_signals": {"segments_needing_review": 2, "pattern_kb_matches": 1},
        }
    ]

    try:
        primary_path.write_text("{ malformed json", encoding="utf-8")
        cache_path.write_text(json.dumps(cached_trends), encoding="utf-8")

        resp = client.get("/api/integrity_trends")
        assert resp.status_code == 200

        payload = resp.get_json()
        assert payload["from_cache"] is True
        assert payload["count"] == 1
        assert payload["source"].endswith("integrity_trends_last.json")
        assert payload["trends"][0]["session_id"] == "cache-1"
    finally:
        if primary_backup is None:
            if primary_path.exists():
                primary_path.unlink()
        else:
            primary_path.write_text(primary_backup, encoding="utf-8")

        if cache_backup is None:
            if cache_path.exists():
                cache_path.unlink()
        else:
            cache_path.write_text(cache_backup, encoding="utf-8")


def test_integrity_endpoints_entry_count_consistency(client):
    trends = _sample_trends()
    with patch("webapp.Smart_Elections_Parser_Webapp._load_integrity_trends", return_value=(trends, "mock-source", False)):
        trends_resp = client.get("/api/integrity_trends")
        signal_resp = client.post(
            "/api/integrity_signal",
            json={
                "confDropThreshold": 0.08,
                "unknownSpikeThreshold": 0.1,
                "reviewSpikeThreshold": 5.0,
                "baselineWindow": 30,
                "recentWindow": 5,
            },
        )

    assert trends_resp.status_code == 200
    assert signal_resp.status_code == 200

    trends_payload = trends_resp.get_json()
    signal_payload = signal_resp.get_json()

    assert trends_payload["count"] == len(trends)
    assert signal_payload["signal"]["entry_count"] == trends_payload["count"]


def test_integrity_trends_normalizes_generated_at_into_timestamp(client):
    trends = [{
        "schema_version": "1.1",
        "session_id": "t1",
        "generated_at": "2026-02-25T10:00:00Z",
        "confidence": {"avg": 0.9, "median": 0.9, "count": 10, "buckets": {"low": 0, "medium": 1, "high": 9}},
        "unknown_ratio": 0.0,
        "segment_count": 10,
        "unknown_segment_count": 0,
        "review_signals": {"segments_needing_review": 0, "pattern_kb_matches": 1},
    }]
    with patch("webapp.Smart_Elections_Parser_Webapp._load_integrity_trends", return_value=(trends, "mock-source", False)):
        resp = client.get("/api/integrity_trends")

    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["trends"][0]["generated_at"] == "2026-02-25T10:00:00Z"


def test_integrity_signal_failure_returns_structured_500(client):
    trends = _sample_trends()
    with patch("webapp.Smart_Elections_Parser_Webapp._load_integrity_trends", return_value=(trends, "mock-source", False)):
        with patch("tools.analyze_context_digest_trends.compute_integrity_signal", side_effect=RuntimeError("signal failed")):
            resp = client.post("/api/integrity_signal", json={})

    assert resp.status_code == 500
    payload = resp.get_json()
    assert payload["error"] == "Failed to compute signal"
    assert payload["signal"]["status"] == "error"
