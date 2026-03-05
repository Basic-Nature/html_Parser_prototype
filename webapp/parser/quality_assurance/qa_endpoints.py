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

from flask import Blueprint, jsonify, request, send_file

from ..config import ENABLE_VERIFICATION_FRAMEWORK, QA_REQUIRE_CERT_AUTH
from ..utils.cert_utils import extract_client_principal
from ..utils.privilege_tiers import PrivilegeTier, get_principal_tier
from ..utils.shared_logic import safe_get, safe_strip
from .data_classifier import (
    DatasetMetadata,
    classify_as_dl1,
    get_dataset_lineage,
    get_dl2_inventory,
    get_pending_dl2_reviews,
    get_rejected_count,
    promote_to_dl2,
)

qa_bp = Blueprint("data_assurance", __name__, url_prefix="/api/data-assurance")


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
    normalized = str(tier or "").strip().lower().replace("-", "_")
    tier_map = {
        "reviewer": PrivilegeTier.STANDARD_USER,
        "standard_user": PrivilegeTier.STANDARD_USER,
        "admin_reviewer": PrivilegeTier.ADMIN_REVIEWER,
        "admin_full_trust": PrivilegeTier.ADMIN_FULL_TRUST,
        "root_admin": PrivilegeTier.ROOT_ADMIN,
    }
    return tier_map.get(normalized, PrivilegeTier.ROOT_ADMIN)


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
            if int(actual) < int(required):
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

        return jsonify({
            "dataset_id": result.dataset_id,
            "dl_status": result.dl_status,
            "confidence_score": result.confidence_score,
            "issues": [issue.to_dict() for issue in result.issues],
            "should_promote_to_dl2": result.should_promote_to_dl2,
            "summary": result.summary,
        })

    except Exception as e:
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
    limit = max(1, min(200, int(request.args.get("limit", 50))))

    try:
        entries = get_pending_dl2_reviews()
        entries = entries[:limit]  # Client-side pagination (simple)

        return jsonify({
            "pending_count": len(entries),
            "entries": entries,
        })

    except Exception as e:
        return jsonify({"error": f"Failed to fetch pending reviews: {e}"}), 500


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
