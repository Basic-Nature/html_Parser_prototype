from __future__ import annotations

from flask import Flask

from webapp.parser.quality_assurance import qa_endpoints as qae


class _StubIssue:
    def __init__(self, severity: str = "WARNING") -> None:
        self.severity = severity

    def to_dict(self):
        return {
            "issue_type": "stub_issue",
            "severity": self.severity,
            "description": "stub",
        }


class _StubResult:
    def __init__(self, confidence_score: float, issues: list[_StubIssue] | None = None):
        self.dataset_id = "stub-dataset"
        self.dl_status = "DL1"
        self.confidence_score = confidence_score
        self.issues = issues or []
        self.should_promote_to_dl2 = False
        self.summary = "stub summary"


def _build_app() -> Flask:
    app = Flask(__name__)
    app.register_blueprint(qae.qa_bp)
    return app


def _disable_auth(monkeypatch):
    monkeypatch.setattr(qae, "ENABLE_VERIFICATION_FRAMEWORK", True)
    monkeypatch.setattr(qae, "QA_REQUIRE_CERT_AUTH", False)
    monkeypatch.setattr(qae, "extract_client_principal", lambda _headers: (None, None, None))


def test_parse_and_classify_returns_auto_pass(monkeypatch):
    app = _build_app()
    _disable_auth(monkeypatch)
    monkeypatch.setattr(qae, "classify_as_dl1", lambda _metadata: _StubResult(96.0, []))

    payload = {
        "source_url": "https://example.gov/results",
        "handler_name": "csv_handler",
        "state_abbr": "CA",
        "county_name": "Los Angeles",
        "election_year": 2024,
        "contest_name": "Mayor",
        "contestant_count": 2,
        "data_row_count": 2,
        "extraction_confidence": 0.95,
        "trust_score": 95.0,
        "headers": ["Candidate", "Votes"],
        "data_rows": [{"candidate_name": "A", "vote_count": 10}],
        "database_cross_check": {
            "status": "match",
            "mismatches": [],
        },
    }

    with app.test_client() as client:
        resp = client.post("/api/data-assurance/parse-and-classify", json=payload)

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["qa_routing_state"] == "AUTO_PASS"
    assert body["review_priority"] == "low"


def test_parse_and_classify_returns_hard_fail_when_gate_failed(monkeypatch):
    app = _build_app()
    _disable_auth(monkeypatch)
    monkeypatch.setattr(qae, "classify_as_dl1", lambda _metadata: _StubResult(88.0, []))

    payload = {
        "source_url": "https://example.gov/results",
        "handler_name": "csv_handler",
        "state_abbr": "CA",
        "county_name": "Los Angeles",
        "election_year": 2024,
        "contest_name": "Mayor",
        "contestant_count": 2,
        "data_row_count": 2,
        "extraction_confidence": 0.90,
        "trust_score": 90.0,
        "headers": ["Candidate", "Votes"],
        "data_rows": [{"candidate_name": "A", "vote_count": 10}],
        "quality_gate": {"status": "failed", "reason": "database_cross_check_mismatch"},
    }

    with app.test_client() as client:
        resp = client.post("/api/data-assurance/parse-and-classify", json=payload)

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["qa_routing_state"] == "HARD_FAIL"
    assert body["review_priority"] == "high"


def test_pending_reviews_include_routing_fields(monkeypatch):
    app = _build_app()
    _disable_auth(monkeypatch)
    monkeypatch.setattr(
        qae,
        "get_pending_dl2_reviews",
        lambda: [
            {
                "dataset_id": "d1",
                "source_url": "https://a",
                "state_abbr": "CA",
                "contest_name": "Mayor",
                "extraction_confidence": 0.95,
                "trust_score": 92.0,
                "detected_issues_count": 0,
                "extracted_at": "2026-01-01T00:00:00Z",
            },
            {
                "dataset_id": "d2",
                "source_url": "https://b",
                "state_abbr": "CA",
                "contest_name": "Mayor",
                "extraction_confidence": 0.72,
                "trust_score": 62.0,
                "detected_issues_count": 2,
                "extracted_at": "2026-01-01T00:00:00Z",
            },
        ],
    )

    with app.test_client() as client:
        resp = client.get("/api/data-assurance/pending-dl2-reviews?limit=10")

    assert resp.status_code == 200
    body = resp.get_json()
    assert "entries" in body
    assert len(body["entries"]) == 2
    assert "qa_routing_state" in body["entries"][0]
    assert "review_priority" in body["entries"][0]
    assert "queue_action" in body["entries"][0]
    assert "next_run_guidance" in body["entries"][0]
    states = {entry["qa_routing_state"] for entry in body["entries"]}
    assert "WARN_REVIEW" in states or "AUTO_PASS" in states


def test_queue_actions_grouped_payload(monkeypatch):
    app = _build_app()
    _disable_auth(monkeypatch)
    monkeypatch.setattr(
        qae,
        "get_pending_dl2_reviews",
        lambda: [
            {
                "id": "q1",
                "url": "https://example.com/auto",
                "detected_issues_count": 0,
                "trust_score": 95,
                "extraction_confidence": 92,
            },
            {
                "id": "q2",
                "url": "https://example.com/review",
                "detected_issues_count": 3,
                "trust_score": 72,
                "extraction_confidence": 70,
            },
            {
                "id": "q3",
                "url": "https://example.com/fail",
                "detected_issues_count": 7,
                "trust_score": 40,
                "extraction_confidence": 45,
            },
        ],
    )

    with app.test_client() as client:
        response = client.get("/api/data-assurance/queue-actions?limit=10")

    assert response.status_code == 200
    payload = response.get_json()
    assert "groups" in payload
    assert set(payload["groups"].keys()) == {
        "auto_pass_candidates",
        "warn_review_queue",
        "hard_fail_retry_queue",
    }
    combined = (
        payload["groups"]["auto_pass_candidates"]
        + payload["groups"]["warn_review_queue"]
        + payload["groups"]["hard_fail_retry_queue"]
    )
    assert payload["total"] == len(combined)
    assert any(item["queue_action"]["action"] == "auto_pass_candidates" for item in combined)
    assert any(item["queue_action"]["action"] == "warn_review_queue" for item in combined)
    assert any(item["queue_action"]["action"] == "hard_fail_retry_queue" for item in combined)
    assert all("next_run_guidance" in item for item in combined)


def test_queue_actions_state_filter_and_invalid_limit(monkeypatch):
    app = _build_app()
    _disable_auth(monkeypatch)
    monkeypatch.setattr(
        qae,
        "get_pending_dl2_reviews",
        lambda: [
            {
                "id": "q1",
                "source_url": "https://example.com/auto",
                "detected_issues_count": 0,
                "trust_score": 95,
                "extraction_confidence": 0.95,
            },
            {
                "id": "q2",
                "source_url": "https://example.com/review",
                "detected_issues_count": 2,
                "trust_score": 70,
                "extraction_confidence": 0.78,
            },
        ],
    )

    with app.test_client() as client:
        response = client.get("/api/data-assurance/queue-actions?state=warn_review&limit=abc")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["state_filter"] == "WARN_REVIEW"
    assert payload["groups"]["auto_pass_candidates"] == []
    assert payload["groups"]["hard_fail_retry_queue"] == []
    assert len(payload["groups"]["warn_review_queue"]) == 1


def test_pending_reviews_invalid_limit_uses_default(monkeypatch):
    app = _build_app()
    _disable_auth(monkeypatch)
    monkeypatch.setattr(
        qae,
        "get_pending_dl2_reviews",
        lambda: [
            {
                "dataset_id": "d1",
                "source_url": "https://a",
                "state_abbr": "CA",
                "contest_name": "Mayor",
                "extraction_confidence": 0.95,
                "trust_score": 92.0,
                "detected_issues_count": 0,
            }
        ],
    )

    with app.test_client() as client:
        resp = client.get("/api/data-assurance/pending-dl2-reviews?limit=invalid")

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["pending_count"] == 1


def test_pipeline_audit_returns_catalog_and_summary(monkeypatch):
    app = _build_app()
    _disable_auth(monkeypatch)

    monkeypatch.setattr(qae, "_database_readiness", lambda: {"ok": True, "checked_tables": {}, "error": None})
    monkeypatch.setattr(qae, "get_pending_dl2_reviews", lambda: [{"trust_score": 55.0, "extraction_confidence": 0.7, "detected_issues_count": 2}])
    monkeypatch.setattr(qae, "get_dl2_inventory", lambda **_kwargs: [{"dataset_id": "dl2-1"}, {"dataset_id": "dl2-2"}])
    monkeypatch.setattr(qae, "get_rejected_count", lambda: 3)

    with app.test_client() as client:
        resp = client.get("/api/data-assurance/pipeline-audit")

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["audit_ok"] is True
    assert "endpoint_catalog" in body
    assert any(item["path"] == "/api/data-assurance/parse-and-classify" for item in body["endpoint_catalog"])
    assert "routing_summary" in body
    assert "queue_summary" in body
