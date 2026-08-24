"""
Data Assurance Endpoints: REST API for DL1/DL2 Classification & Review

Provides:
- POST /api/data-assurance/parse-and-classify - Submit URL for DL1 classification
- GET /api/data-assurance/pending-dl2-reviews - List DL1 pending review
- POST /api/data-assurance/verify-and-promote - Human review → promote to DL2
- GET /api/data-assurance/dl-inventory - Query all DL1/DL2 data
- GET /api/data-assurance/lineage/<id> - Get audit trail
- POST /api/data-assurance/export-dl2 - Export verified data
"""

from __future__ import annotations

import csv
import io
from functools import wraps
from io import StringIO
from typing import Any, Dict, List

from flask import Blueprint, jsonify, request, send_file

from ..config import ENABLE_VERIFICATION_FRAMEWORK, QA_REQUIRE_CERT_AUTH
from ..utils.cert_utils import extract_client_principal
from ..utils.privilege_tiers import PrivilegeTier, get_principal_tier
from ..auth.tiers import normalize_required_tier
from ..auth.authorization import tier_satisfies
from ..utils.shared_logic import safe_get, safe_strip
from .data_classifier import (
    DatasetMetadata,
    classify_as_dl1,
    get_dataset_lineage,
    get_dl2_inventory,
    get_pending_dl2_reviews,
    get_rejected_count,
    get_db_connection,
    promote_to_dl2,
)

qa_bp = Blueprint("data_assurance", __name__, url_prefix="/api/data-assurance")


def _derive_qa_routing_state(
    *,
    confidence_score: float,
    extraction_confidence: float,
    trust_score: float,
    issues: List[Dict[str, Any]],
    database_cross_check: Dict[str, Any] | None,
    quality_gate: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """Derive explicit review routing state for QA panel and review queue.

    States:
    - AUTO_PASS: minimal risk, can move forward with low-touch review
    - WARN_REVIEW: medium risk, queue for reviewer validation
    - HARD_FAIL: high risk, block promotion until corrected
    """
    reasons: List[str] = []

    issue_count = len(issues or [])
    severe_issues = [
        issue for issue in (issues or [])
        if str(issue.get("severity", "")).upper() in {"ERROR", "CRITICAL"}
    ]

    if severe_issues:
        reasons.append(f"severe_issues:{len(severe_issues)}")

    if quality_gate and str(quality_gate.get("status", "")).lower() == "failed":
        reasons.append("quality_gate_failed")

    cc = database_cross_check or {}
    cc_status = str(cc.get("status", "")).lower()
    if cc_status == "mismatch":
        mismatches = cc.get("mismatches") or []
        has_error_mismatch = any(
            str(m.get("severity", "")).lower() == "error"
            for m in mismatches
            if isinstance(m, dict)
        )
        reasons.append("database_cross_check_mismatch")
        if has_error_mismatch:
            reasons.append("database_cross_check_error_mismatch")

    if trust_score < 60:
        reasons.append("low_trust_score")
    if extraction_confidence < 0.70:
        reasons.append("low_extraction_confidence")
    if confidence_score < 60:
        reasons.append("low_classification_confidence")

    if (
        "quality_gate_failed" in reasons
        or "database_cross_check_error_mismatch" in reasons
        or issue_count >= 5
        or trust_score < 50
        or extraction_confidence < 0.60
        or confidence_score < 50
    ):
        return {
            "state": "HARD_FAIL",
            "priority": "high",
            "reasons": reasons or ["hard_fail_threshold"],
        }

    if (
        issue_count > 0
        or cc_status == "mismatch"
        or trust_score < 80
        or extraction_confidence < 0.85
        or confidence_score < 85
    ):
        return {
            "state": "WARN_REVIEW",
            "priority": "medium",
            "reasons": reasons or ["review_recommended"],
        }

    return {
        "state": "AUTO_PASS",
        "priority": "low",
        "reasons": ["high_confidence_clean_cross_check"],
    }


def _build_retry_guidance(*, routing_state: str, reasons: List[str], entry: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Build parser retry/improvement guidance for next run orchestration."""
    entry = entry or {}
    handler_name = str(entry.get("handler_name") or entry.get("source_handler") or "unknown")
    source_url = str(entry.get("source_url") or "")

    guidance: Dict[str, Any] = {
        "retry_required": routing_state == "HARD_FAIL",
        "routing_state": routing_state,
        "handler_name": handler_name,
        "source_url": source_url,
        "recommended_overrides": {},
        "recommended_steps": [],
    }

    # Base retry policy by routing state
    if routing_state == "AUTO_PASS":
        guidance["recommended_steps"] = [
            "Promote candidate after lightweight reviewer spot-check.",
            "Capture successful configuration for future handler priors.",
        ]
        guidance["recommended_overrides"] = {
            "skip_database_check": False,
            "disable_database_cross_check": False,
        }
        return guidance

    if routing_state == "WARN_REVIEW":
        guidance["recommended_steps"] = [
            "Route to manual review queue before promotion.",
            "Inspect headers and candidate mappings for ambiguous columns.",
            "Apply targeted parser rerun only if reviewer flags material defects.",
        ]
        guidance["recommended_overrides"] = {
            "skip_database_check": False,
            "disable_database_cross_check": False,
            "table_builder_debug": True,
        }

    if routing_state == "HARD_FAIL":
        guidance["recommended_steps"] = [
            "Block promotion and trigger parser rerun with stricter recovery path.",
            "Capture table-builder debug artifacts and context snapshots.",
            "Escalate to handler tuning queue if rerun still mismatches.",
        ]
        guidance["recommended_overrides"] = {
            "skip_database_check": True,
            "disable_database_cross_check": False,
            "table_builder_debug": True,
            "force_reparse": True,
            "enable_selenium_fallback": True,
        }

    # Reason-specific refinements
    if any("database_cross_check" in r for r in reasons):
        guidance["recommended_steps"].append("Run dual-source diff and inspect contest/county alignment.")
    if any("low_extraction_confidence" in r for r in reasons):
        guidance["recommended_overrides"]["skip_pivot"] = False
        guidance["recommended_steps"].append("Increase table extraction retries and verify location/candidate columns.")
    if any("low_trust_score" in r for r in reasons):
        guidance["recommended_steps"].append("Validate source URL trust and routing decision before rerun.")
    if any("severe_issues" in r for r in reasons):
        guidance["recommended_overrides"]["require_manual_column_review"] = True

    return guidance


def _build_queue_action(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Derive orchestration queue action payload from an annotated pending-review entry."""
    state = str(entry.get("qa_routing_state") or "WARN_REVIEW")
    reasons = entry.get("routing_reasons") or []
    priority = str(entry.get("review_priority") or "medium")

    action_map = {
        "AUTO_PASS": "auto_pass_candidates",
        "WARN_REVIEW": "warn_review_queue",
        "HARD_FAIL": "hard_fail_retry_queue",
    }
    action = action_map.get(state, "warn_review_queue")

    return {
        "action": action,
        "priority": priority,
        "state": state,
        "retry_guidance": _build_retry_guidance(routing_state=state, reasons=reasons, entry=entry),
    }


def _endpoint_catalog() -> List[Dict[str, str]]:
    """Catalog QA + extraction-continuation endpoints relevant to pipeline auditing."""
    return [
        {"method": "POST", "path": "/api/data-assurance/parse-and-classify", "purpose": "DL1 classification + issue detection"},
        {"method": "GET", "path": "/api/data-assurance/pending-dl2-reviews", "purpose": "Review queue with routing states"},
        {"method": "GET", "path": "/api/data-assurance/queue-actions", "purpose": "Queue orchestration actions + retry guidance"},
        {"method": "POST", "path": "/api/data-assurance/verify-and-promote", "purpose": "Manual promotion to DL2"},
        {"method": "GET", "path": "/api/data-assurance/dl-inventory", "purpose": "Verified inventory query"},
        {"method": "GET", "path": "/api/data-assurance/stats", "purpose": "Queue and quality metrics"},
        {"method": "GET", "path": "/api/data-assurance/pipeline-audit", "purpose": "Endpoint and DB readiness audit"},
        {"method": "GET", "path": "/api/data_framework/preview", "purpose": "Preview extracted/curated records"},
        {"method": "GET", "path": "/api/data_framework/scaffold", "purpose": "Scaffold gaps for continued extraction"},
        {"method": "GET", "path": "/api/data_framework/curated", "purpose": "Curated review feed"},
        {"method": "GET", "path": "/api/data_framework/warehouse_status", "purpose": "Warehouse deficit and priority status"},
    ]


def _database_readiness() -> Dict[str, Any]:
    """Best-effort DB readiness check for verification pipeline tables."""
    readiness = {
        "ok": False,
        "checked_tables": {
            "verified_data.verified_datasets": False,
            "verified_data.quality_issues": False,
            "verified_data.verification_lineage": False,
        },
        "error": None,
    }
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            readiness["error"] = "db_connection_unavailable"
            return readiness
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT table_schema || '.' || table_name AS full_name
            FROM information_schema.tables
            WHERE table_schema = 'verified_data'
              AND table_name IN ('verified_datasets', 'quality_issues', 'verification_lineage')
            """
        )
        found = {row[0] for row in cursor.fetchall()}
        for table_name in list(readiness["checked_tables"].keys()):
            readiness["checked_tables"][table_name] = table_name in found
        readiness["ok"] = all(readiness["checked_tables"].values())
        cursor.close()
    except Exception as exc:
        readiness["error"] = str(exc)
    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass
    return readiness


def _bounded_int_query_arg(name: str, *, default: int, minimum: int, maximum: int) -> int:
    """Return a bounded integer query argument with safe fallback on invalid input."""
    value = request.args.get(name, default=default, type=int)
    if value is None:
        value = default
    return max(minimum, min(maximum, int(value)))


def _normalize_routing_state_filter(raw: str) -> str:
    state = safe_strip(raw or "").upper()
    return state if state in {"AUTO_PASS", "WARN_REVIEW", "HARD_FAIL"} else ""


def _annotate_review_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Attach routing, queue action, and next-run guidance to a review entry."""
    detected_count = int(entry.get("detected_issues_count") or 0)
    synth_issues = [{"severity": "WARNING"}] * detected_count if detected_count > 0 else []

    routing = _derive_qa_routing_state(
        confidence_score=90.0 - min(60.0, detected_count * 5.0),
        extraction_confidence=float(entry.get("extraction_confidence") or 0.0),
        trust_score=float(entry.get("trust_score") or 0.0),
        issues=synth_issues,
        database_cross_check=None,
        quality_gate=None,
    )
    entry["qa_routing_state"] = routing["state"]
    entry["review_priority"] = routing["priority"]
    entry["routing_reasons"] = routing["reasons"]

    action_payload = _build_queue_action(entry)
    entry["queue_action"] = action_payload
    entry["next_run_guidance"] = action_payload.get("retry_guidance", {})
    return entry


def _require_qa_enabled(f):
    """Decorator: Verify QA framework is enabled."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not ENABLE_VERIFICATION_FRAMEWORK:
            return jsonify({"error": "Data assurance framework disabled"}), 403
        return f(*args, **kwargs)

    return decorated_function


def _get_reviewer_principal() -> str | None:
    """Extract principal from request (must be authenticated)."""
    principal, _, _ = extract_client_principal(request.headers)
    return principal


def _get_reviewer_identity() -> tuple[str | None, str]:
    """Extract reviewer principal + source from request headers."""
    principal, source, _ = extract_client_principal(request.headers)
    return principal, source or ""


def _normalize_required_tier(tier: str) -> PrivilegeTier:
    # Compatibility wrapper for the canonical tier vocabulary.
    return normalize_required_tier(tier)


def _require_reviewer(f):
    """Decorator: Require authenticated reviewer (or allow fallback if cert auth disabled)."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        principal, principal_source = _get_reviewer_identity()
        
        # If certificate auth is required and no principal found, reject
        if QA_REQUIRE_CERT_AUTH and not principal:
            return jsonify({
                "error": "Unauthorized: Certificate authentication required",
                "help": "Set QA_REQUIRE_CERT_AUTH=false in environment to disable cert requirement"
            }), 401
        
        # If cert auth is optional and no principal, use fallback principal
        if not principal:
            # Use a fallback principal for development/testing
            # This should only happen when QA_REQUIRE_CERT_AUTH=false
            from flask import g
            principal = "system:development"
            principal_source = "development_fallback"
            g.reviewer_principal = principal
            g.reviewer_source = principal_source
            g.reviewer_tier = PrivilegeTier.STANDARD_USER.name
        else:
            from flask import g
            g.reviewer_principal = principal
            g.reviewer_source = principal_source
            g.reviewer_tier = get_principal_tier(principal, principal_source).name
        
        return f(*args, **kwargs)

    return decorated_function


def _require_reviewer_tier(required_tier: str):
    """Decorator: Require a minimum privilege tier for QA endpoints."""
    required = _normalize_required_tier(required_tier)

    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            from flask import g

            principal = getattr(g, "reviewer_principal", None)
            principal_source = getattr(g, "reviewer_source", "")
            if not principal:
                principal, principal_source = _get_reviewer_identity()
                if not principal:
                    return jsonify({"error": "Unauthorized"}), 401

            actual = get_principal_tier(principal, principal_source)
            if not tier_satisfies(actual, required):
                return jsonify(
                    {
                        "error": "Forbidden",
                        "required_tier": required.name,
                        "actual_tier": actual.name,
                    }
                ), 403

            g.reviewer_tier = actual.name
            return f(*args, **kwargs)

        return decorated_function

    return decorator


# ===== ENDPOINTS =====


@qa_bp.route("/parse-and-classify", methods=["POST"])
@_require_qa_enabled
@_require_reviewer
def parse_and_classify():
    """
    Submit parsed election data for DL1 classification.

    This is typically called from Ballot Lens after parsing a URL.
    The parser returns (headers, data_rows, contest, metadata),
    and this endpoint classifies it and stores in PostgreSQL.

    Request body:
    {
        "source_url": "https://elections.example.gov/results",
        "handler_name": "html_handler",
        "state_abbr": "CA",
        "county_name": "Los Angeles",
        "election_year": 2024,
        "contest_name": "President",
        "contestant_count": 5,
        "data_row_count": 5,
        "extraction_confidence": 0.92,
        "trust_score": 85.5,
        "headers": ["Candidate", "Votes", "%"],
        "data_rows": [
            {"Candidate": "Alice", "Votes": "150000", "%": "45.2%"},
            ...
        ]
    }

    Returns:
        {
            "dataset_id": "uuid",
            "dl_status": "DL1",
            "confidence_score": 87.5,
            "issues": [...],
            "should_promote_to_dl2": false,
            "summary": "DL1 unverified. 0 issues detected. Trust score: 85.5/100"
        }
    """
    data = request.get_json(force=True) or {}

    metadata_payload = safe_get(data, "metadata", {}) or {}
    # Extract required fields
    source_url = safe_strip(safe_get(data, "source_url", ""))
    handler_name = safe_strip(safe_get(data, "handler_name", ""))
    state_abbr = safe_strip(safe_get(data, "state_abbr", "")).upper()
    county_name = safe_get(data, "county_name")
    election_year = safe_get(data, "election_year", 0)
    contest_name = safe_strip(safe_get(data, "contest_name", ""))
    contestant_count = safe_get(data, "contestant_count", 0)
    data_row_count = safe_get(data, "data_row_count", 0)
    extraction_confidence = float(safe_get(data, "extraction_confidence", 0.0))
    trust_score = float(safe_get(data, "trust_score", 0.0))
    headers = safe_get(data, "headers", [])
    data_rows = safe_get(data, "data_rows", [])

    # Optional cross-check and gate payloads (can be sent either top-level or nested under metadata)
    database_cross_check = safe_get(data, "database_cross_check", None) or safe_get(metadata_payload, "database_cross_check", None)
    quality_gate = safe_get(data, "quality_gate", None) or safe_get(metadata_payload, "quality_gate", None)

    # Validate required fields
    if not all([source_url, handler_name, state_abbr, election_year, contest_name]):
        return jsonify({"error": "Missing required fields"}), 400

    try:
        # Create metadata object
        metadata = DatasetMetadata(
            source_url=source_url,
            handler_name=handler_name,
            state_abbr=state_abbr,
            county_name=county_name,
            election_year=int(election_year),
            contest_name=contest_name,
            contestant_count=int(contestant_count),
            data_row_count=int(data_row_count),
            extraction_confidence=extraction_confidence,
            trust_score=trust_score,
            headers=headers,
            data_rows=data_rows,
        )

        # Classify as DL1
        result = classify_as_dl1(metadata)

        issues_dict = [issue.to_dict() for issue in result.issues]
        routing = _derive_qa_routing_state(
            confidence_score=float(result.confidence_score or 0.0),
            extraction_confidence=extraction_confidence,
            trust_score=trust_score,
            issues=issues_dict,
            database_cross_check=database_cross_check if isinstance(database_cross_check, dict) else None,
            quality_gate=quality_gate if isinstance(quality_gate, dict) else None,
        )
        retry_guidance = _build_retry_guidance(
            routing_state=routing["state"],
            reasons=list(routing.get("reasons") or []),
            entry={
                "handler_name": handler_name,
                "source_url": source_url,
            },
        )

        return jsonify({
            "dataset_id": result.dataset_id,
            "dl_status": result.dl_status,
            "confidence_score": result.confidence_score,
            "issues": issues_dict,
            "should_promote_to_dl2": result.should_promote_to_dl2,
            "summary": result.summary,
            "qa_routing_state": routing["state"],
            "review_priority": routing["priority"],
            "routing_reasons": routing["reasons"],
            "next_run_guidance": retry_guidance,
        })

    except Exception as e:
        readiness = _database_readiness()
        if not bool(readiness.get("ok", False)):
            return jsonify({
                "error": "QA classification database unavailable",
                "code": "qa_database_unavailable",
                "available": False,
                "retryable": True,
                "reason": readiness.get("error") or "required_qa_tables_unavailable",
            }), 503

        return jsonify({"error": f"Classification failed: {e}"}), 500


@qa_bp.route("/pending-dl2-reviews", methods=["GET"])
@_require_qa_enabled
@_require_reviewer
def get_pending_reviews():
    """
    Get all DL1 datasets pending manual review for promotion to DL2.

    Query parameters:
    - limit: Max results (default: 50, max: 200)

    Returns:
        {
            "pending_count": 12,
            "entries": [
                {
                    "dataset_id": "uuid",
                    "source_url": "...",
                    "state_abbr": "CA",
                    "contest_name": "President",
                    "extraction_confidence": 0.92,
                    "trust_score": 85.5,
                    "detected_issues_count": 0,
                    "extracted_at": "2024-02-05T10:00:00Z"
                }
            ]
        }
    """
    limit = _bounded_int_query_arg("limit", default=50, minimum=1, maximum=200)

    try:
        entries = get_pending_dl2_reviews()
        entries = entries[:limit]  # Client-side pagination (simple)

        for entry in entries:
            _annotate_review_entry(entry)

        return jsonify({
            "pending_count": len(entries),
            "entries": entries,
        })

    except Exception as e:
        return jsonify({"error": f"Failed to fetch pending reviews: {e}"}), 500


@qa_bp.route("/queue-actions", methods=["GET"])
@_require_qa_enabled
@_require_reviewer
def get_queue_actions():
    """Return grouped queue actions that directly drive reviewer/rerun orchestration."""
    limit = _bounded_int_query_arg("limit", default=200, minimum=1, maximum=500)
    state_filter = _normalize_routing_state_filter(safe_get(request.args, "state", ""))

    try:
        entries = get_pending_dl2_reviews()[:limit]
        grouped = {
            "auto_pass_candidates": [],
            "warn_review_queue": [],
            "hard_fail_retry_queue": [],
        }

        for entry in entries:
            _annotate_review_entry(entry)

            if state_filter and entry.get("qa_routing_state") != state_filter:
                continue

            bucket = safe_get(entry, "queue_action", {}).get("action")
            if bucket not in grouped:
                bucket = "warn_review_queue"
            grouped[bucket].append(entry)

        return jsonify(
            {
                "total": sum(len(v) for v in grouped.values()),
                "state_filter": state_filter or None,
                "groups": grouped,
            }
        )
    except Exception as exc:
        return jsonify({"error": f"Failed to build queue actions: {exc}"}), 500


@qa_bp.route("/verify-and-promote", methods=["POST"])
@_require_qa_enabled
@_require_reviewer
@_require_reviewer_tier("admin_reviewer")
def verify_and_promote():
    """
    Promote a DL1 dataset to DL2 after human review.

    This is the primary workflow for approving unverified data.

    Request body:
    {
        "dataset_id": "uuid",
        "certification_reason": "Manual review completed. Data verified against official source.",
        "resolve_issues": {
            "issue_uuid_1": "Confirmed duplicate, removed in source",
            "issue_uuid_2": "Percentage rounding, acceptable"
        }
    }

    Returns:
        {
            "success": true,
            "dataset_id": "uuid",
            "new_status": "DL2",
            "verified_by": "reviewer@elections.gov",
            "timestamp": "2024-02-05T10:15:00Z"
        }
    """
    from flask import g
    principal = g.reviewer_principal

    data = request.get_json(force=True) or {}
    dataset_id = safe_strip(safe_get(data, "dataset_id", ""))
    certification_reason = safe_strip(safe_get(data, "certification_reason", ""))
    resolve_issues = safe_get(data, "resolve_issues", {})

    if not dataset_id or not certification_reason:
        return jsonify({"error": "dataset_id and certification_reason required"}), 400

    try:
        success = promote_to_dl2(
            dataset_id=dataset_id,
            reviewer_principal=principal,
            certification_reason=certification_reason,
            resolve_issues=resolve_issues or None,
        )

        if not success:
            return jsonify({"error": "Failed to promote dataset"}), 400

        from datetime import datetime, timezone

        return jsonify({
            "success": True,
            "dataset_id": dataset_id,
            "new_status": "DL2",
            "verified_by": principal,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    except Exception as e:
        return jsonify({"error": f"Promotion failed: {e}"}), 500


@qa_bp.route("/dl-inventory", methods=["GET"])
@_require_qa_enabled
@_require_reviewer
def get_inventory():
    """
    Query all verified DL2 datasets with optional filtering.

    Query parameters:
    - state: Filter by state abbreviation (e.g., 'CA')
    - county: Filter by county name (e.g., 'Los Angeles')
    - year: Filter by election year (e.g., 2024)
    - limit: Max results (default: 100, max: 1000)

    Returns:
        {
            "total": 342,
            "filtered": 12,
            "entries": [
                {
                    "dataset_id": "uuid",
                    "source_url": "...",
                    "state_abbr": "CA",
                    "county_name": "Los Angeles",
                    "contest_name": "President",
                    "extraction_confidence": 0.92,
                    "trust_score": 85.5,
                    "extracted_at": "2024-02-05T10:00:00Z"
                }
            ]
        }
    """
    state = safe_strip(safe_get(request.args, "state", "")).upper()
    county = safe_strip(safe_get(request.args, "county", ""))
    year = request.args.get("year", type=int)
    limit = max(1, min(1000, int(request.args.get("limit", 100))))

    try:
        entries = get_dl2_inventory(
            state_abbr=state if state else None,
            county_name=county if county else None,
            year=year,
        )
        entries = entries[:limit]

        return jsonify({
            "total": len(get_dl2_inventory()),  # Total DL2 count (expensive, cache this)
            "filtered": len(entries),
            "entries": entries,
        })

    except Exception as e:
        return jsonify({"error": f"Failed to fetch inventory: {e}"}), 500


@qa_bp.route("/lineage/<dataset_id>", methods=["GET"])
@_require_qa_enabled
@_require_reviewer
def get_lineage(dataset_id: str):
    """
    Get complete audit trail (lineage) for a dataset.

    Shows every decision made on this data:
    - When it was classified
    - What issues were detected
    - Who reviewed it
    - When it was promoted to DL2

    Returns:
        {
            "dataset_id": "uuid",
            "events": [
                {
                    "action_type": "classification",
                    "action_status": "completed",
                    "action_timestamp": "2024-02-05T10:00:00Z",
                    "confidence_score": 87.5,
                    "details": {...}
                },
                {
                    "action_type": "promoted_to_dl2",
                    "reviewer_principal": "john@elections.gov",
                    "certification_reason": "Manual review completed",
                    "action_timestamp": "2024-02-05T10:15:00Z"
                }
            ]
        }
    """
    dataset_id = safe_strip(dataset_id)
    if not dataset_id:
        return jsonify({"error": "Invalid dataset_id"}), 400

    try:
        lineage = get_dataset_lineage(dataset_id)

        return jsonify({
            "dataset_id": dataset_id,
            "events": lineage,
        })

    except Exception as e:
        return jsonify({"error": f"Failed to fetch lineage: {e}"}), 500


@qa_bp.route("/export-dl2", methods=["POST"])
@_require_qa_enabled
@_require_reviewer
def export_dl2_data():
    """
    Export verified DL2 data as CSV.

    Request body:
    {
        "state": "CA",  // Optional filter
        "county": "Los Angeles",  // Optional filter
        "year": 2024,  // Optional filter
        "format": "csv"  // or "json"
    }

    Returns:
        CSV file with columns:
        dataset_id, state, county, contest, candidates, trust_score, extracted_at
    """
    from flask import g
    principal = g.reviewer_principal
    data = request.get_json(force=True) or {}

    state = safe_strip(safe_get(data, "state", "")).upper()
    county = safe_strip(safe_get(data, "county", ""))
    year = safe_get(data, "year", type=int)
    format_type = safe_strip(safe_get(data, "format", "csv")).lower()

    if format_type not in ["csv", "json"]:
        return jsonify({"error": "format must be 'csv' or 'json'"}), 400

    try:
        entries = get_dl2_inventory(
            state_abbr=state if state else None,
            county_name=county if county else None,
            year=year,
        )

        if format_type == "json":
            return jsonify({
                "exported_by": principal,
                "count": len(entries),
                "data": entries,
            })

        # CSV format
        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=[
            "dataset_id", "state_abbr", "county_name", "contest_name",
            "contestant_count", "extraction_confidence", "trust_score",
            "extracted_at", "source_url"
        ])
        writer.writeheader()
        for entry in entries:
            writer.writerow(entry)

        output.seek(0)
        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype="text/csv",
            as_attachment=True,
            download_name=f"dl2_verified_{state or 'all'}_{year or 'all'}.csv"
        )

    except Exception as e:
        return jsonify({"error": f"Export failed: {e}"}), 500


@qa_bp.route("/stats", methods=["GET"])
@_require_qa_enabled
@_require_reviewer
def get_stats():
    """
    Get data assurance statistics.

    Returns:
        {
            "dl1_pending_count": 12,
            "dl2_verified_count": 342,
            "rejected_count": 5,
            "avg_trust_score": 82.5,
            "avg_extraction_confidence": 0.89
        }
    """
    try:
        pending = get_pending_dl2_reviews()
        verified = get_dl2_inventory()

        avg_trust = sum(e.get("trust_score", 0) for e in verified) / len(verified) if verified else 0
        avg_confidence = sum(e.get("extraction_confidence", 0) for e in verified) / len(verified) if verified else 0

        return jsonify({
            "dl1_pending_count": len(pending),
            "dl2_verified_count": len(verified),
            "rejected_count": int(get_rejected_count()),
            "avg_trust_score": round(avg_trust, 2),
            "avg_extraction_confidence": round(avg_confidence, 2),
        })

    except Exception as e:
        return jsonify({"error": f"Failed to fetch stats: {e}"}), 500


@qa_bp.route("/pipeline-audit", methods=["GET"])
@_require_qa_enabled
@_require_reviewer
def pipeline_audit():
    """Audit endpoint coverage + pipeline readiness for continued extraction loops."""
    try:
        pending = get_pending_dl2_reviews()
        verified = get_dl2_inventory()
        db = _database_readiness()
        catalog = _endpoint_catalog()

        queue_summary = {
            "pending_dl1": len(pending),
            "verified_dl2": len(verified),
            "rejected": int(get_rejected_count()),
        }

        # Count current routing distribution from pending queue snapshots
        routing_summary = {"AUTO_PASS": 0, "WARN_REVIEW": 0, "HARD_FAIL": 0}
        for entry in pending:
            detected_count = int(entry.get("detected_issues_count") or 0)
            synth_issues = [{"severity": "WARNING"}] * detected_count if detected_count > 0 else []
            routing = _derive_qa_routing_state(
                confidence_score=90.0 - min(60.0, detected_count * 5.0),
                extraction_confidence=float(entry.get("extraction_confidence") or 0.0),
                trust_score=float(entry.get("trust_score") or 0.0),
                issues=synth_issues,
                database_cross_check=None,
                quality_gate=None,
            )
            routing_summary[routing["state"]] = routing_summary.get(routing["state"], 0) + 1

        return jsonify(
            {
                "audit_ok": bool(db.get("ok", False)),
                "database": db,
                "queue_summary": queue_summary,
                "routing_summary": routing_summary,
                "endpoint_catalog": catalog,
                "recommendations": [
                    "Prioritize HARD_FAIL queue for parser rerun or manual correction.",
                    "Use WARN_REVIEW queue to refine dynamic table mappings and context prompts.",
                    "Continuously reconcile warehouse_status deficits with pending DL1 sources.",
                ],
            }
        )
    except Exception as exc:
        return jsonify({"error": f"Pipeline audit failed: {exc}"}), 500
