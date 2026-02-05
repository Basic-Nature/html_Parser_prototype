"""
Quarantine Review Endpoints: Transparent UI for URL quarantine review.

Provides:
- GET /api/quarantine/pending - List quarantined URLs with explanations
- GET /api/quarantine/<id> - Full details of quarantine entry
- POST /api/quarantine/<id>/review - Record review decision with certification
- GET /api/quarantine/stats - Quarantine queue statistics
"""

from __future__ import annotations

from functools import wraps

from flask import Blueprint, jsonify, request

from .config import ENABLE_VERIFICATION_FRAMEWORK
from .health.quarantine_queue import (
    ReviewStatus,
    get_quarantine_queue,
)
from .utils.cert_utils import extract_client_principal
from .utils.shared_logic import safe_get, safe_strip

quarantine_bp = Blueprint("quarantine", __name__, url_prefix="/api/quarantine")


def _require_quarantine_enabled(f):
    """Decorator: Verify quarantine framework is enabled."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not ENABLE_VERIFICATION_FRAMEWORK:
            return jsonify({"error": "Quarantine framework disabled"}), 403
        return f(*args, **kwargs)
    return decorated_function


def _get_reviewer_principal() -> str | None:
    """Extract principal from request (must be authenticated)."""
    principal, _ = extract_client_principal(request.headers)
    return principal


def _require_reviewer(f):
    """Decorator: Require authenticated reviewer."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        principal = _get_reviewer_principal()
        if not principal:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated_function


# ===== QUARANTINE ENDPOINTS =====

@quarantine_bp.route("/pending", methods=["GET"])
@_require_quarantine_enabled
@_require_reviewer
def get_pending_quarantines():
    """
    Get pending quarantined URLs with full explanations.
    
    Each entry includes:
    - Why it was quarantined (with human-readable explanation)
    - What data was collected (with retention/usage info)
    - What impact this has on the user
    - Full audit trail
    
    Returns:
        JSON list of quarantine entries
    """
    limit = max(1, min(100, int(request.args.get("limit", 20))))
    
    queue = get_quarantine_queue()
    entries = queue.get_pending(limit=limit)
    
    result = []
    for entry in entries:
        result.append({
            "id": entry.quarantine_id,
            "url": entry.url,
            "timestamp": entry.timestamp,
            "session_id": entry.session_id,
            "principal": entry.principal,
            "reason": entry.reason,
            "reason_explanation": entry.reason_explanation,
            "reason_impact": entry.reason_impact,
            "trust_score": entry.trust_score,
            "trust_factors": entry.trust_factors,
            "data_collected": [
                {
                    "type": dc.data_type,
                    "description": dc.description,
                    "usage": dc.usage,
                    "retention_days": dc.retention_days,
                }
                for dc in entry.data_collected
            ],
            "error_messages": entry.error_messages,
            "extraction_attempts": entry.extraction_attempts,
            "review_status": entry.review_status,
        })
    
    return jsonify({
        "pending_count": len(result),
        "entries": result,
    })


@quarantine_bp.route("/<quarantine_id>", methods=["GET"])
@_require_quarantine_enabled
@_require_reviewer
def get_quarantine_detail(quarantine_id: str):
    """
    Get detailed information about a specific quarantine entry.
    
    Includes full transparency into:
    - Why URL was quarantined
    - What data was collected and for what purpose
    - Retention policy
    - Full review history
    
    Returns:
        JSON with full quarantine details
    """
    queue = get_quarantine_queue()
    entries = queue.get_pending(limit=10000)
    
    entry = next((e for e in entries if e.quarantine_id == quarantine_id), None)
    if not entry:
        return jsonify({"error": "Quarantine entry not found"}), 404
    
    return jsonify({
        "id": entry.quarantine_id,
        "url": entry.url,
        "timestamp": entry.timestamp,
        "session_id": entry.session_id,
        "principal": entry.principal,
        "reason": entry.reason,
        "reason_explanation": entry.reason_explanation,
        "reason_impact": entry.reason_impact,
        "trust_score": entry.trust_score,
        "trust_factors": entry.trust_factors,
        "data_collected": [
            {
                "type": dc.data_type,
                "description": dc.description,
                "usage": dc.usage,
                "retention_days": dc.retention_days,
            }
            for dc in entry.data_collected
        ],
        "error_messages": entry.error_messages,
        "extraction_attempts": entry.extraction_attempts,
        "review_status": entry.review_status,
        "review_history": entry.review_history,
    })


@quarantine_bp.route("/<quarantine_id>/review", methods=["POST"])
@_require_quarantine_enabled
@_require_reviewer
def submit_quarantine_review(quarantine_id: str):
    """
    Submit a review decision for a quarantine entry.
    
    This decision is certified by the reviewer's principal and permanently
    logged with full audit trail showing:
    - Who reviewed it
    - When it was reviewed
    - Why the decision was made (certification_reason)
    - What action was taken
    
    Request body:
    {
        "status": "approved" | "rejected" | "needs_more_info" | "appealed",
        "notes": "human-readable review notes",
        "certification_reason": "Why this decision was made (for transparency)"
    }
    
    Returns:
        JSON confirmation with updated entry status
    """
    principal = _get_reviewer_principal()
    if not principal:
        return jsonify({"error": "Unauthorized"}), 401
    
    data = request.get_json(force=True) or {}
    status_str = safe_strip(safe_get(data, "status", "")).lower()
    notes = safe_strip(safe_get(data, "notes", ""))
    certification_reason = safe_strip(safe_get(data, "certification_reason", ""))
    
    if not status_str:
        return jsonify({"error": "status required"}), 400
    
    try:
        status = ReviewStatus(status_str)
    except ValueError:
        return jsonify({
            "error": f"Invalid status. Must be one of: {', '.join([s.value for s in ReviewStatus])}"
        }), 400
    
    if not certification_reason:
        return jsonify({"error": "certification_reason required (explain your decision for transparency)"}), 400
    
    queue = get_quarantine_queue()
    success = queue.record_review(
        quarantine_id=quarantine_id,
        status=status,
        reviewer_principal=principal,
        notes=notes,
        certification_reason=certification_reason,
    )
    
    if not success:
        return jsonify({"error": "Quarantine entry not found"}), 404
    
    return jsonify({
        "success": True,
        "quarantine_id": quarantine_id,
        "status": status.value,
        "reviewed_by": principal,
        "certification_reason": certification_reason,
        "notes": notes,
    })


@quarantine_bp.route("/stats", methods=["GET"])
@_require_quarantine_enabled
@_require_reviewer
def get_quarantine_stats():
    """
    Get quarantine queue statistics.
    
    Helps reviewers prioritize and understand current quarantine workload.
    
    Returns:
        JSON with queue statistics
    """
    queue = get_quarantine_queue()
    stats = queue.get_stats()
    
    return jsonify({
        "total_pending": stats.get("total_pending", 0),
        "pending_by_reason": stats.get("pending_by_reason", {}),
        "oldest_entry": stats.get("oldest_entry"),
    })
