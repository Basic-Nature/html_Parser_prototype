"""Verification Framework API Endpoints

Implements REST endpoints for the DL2 → DL1 verification workflow:
- Fetch unverified rows (DL2 samples)
- Submit verification decisions
- View verification history
- Export verified data (DL1)
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from functools import wraps
from typing import Optional

from flask import Blueprint, Response, jsonify, request
from webapp.parser.config import (
    ENABLE_VERIFICATION_FRAMEWORK,
    SYSTEM_AUTHOR,
    SYSTEM_MISSION,
    VERIFICATION_LOG_FILE,
)
from webapp.parser.utils.logger_singleton import logger
from webapp.parser.utils.shared_logic import safe_get, safe_strip
from webapp.parser.utils.verification_framework import (
    VerificationConfidence,
    VerificationLineageEntry,
    VerificationLog,
    VerificationStatus,
    classify_anomaly,
)

try:
    from webapp.parser.verification.local_dl_sync import LocalStorageSync
    SYNC_AVAILABLE = True
except ImportError:
    SYNC_AVAILABLE = False
    LocalStorageSync = None


verification_bp = Blueprint("verification", __name__, url_prefix="/api/verification")


# ============================================================================
# Authentication & Authorization Decorators
# ============================================================================

def _require_verification_enabled(f):
    """Decorator: Verify that verification framework is enabled."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not ENABLE_VERIFICATION_FRAMEWORK:
            return jsonify({"error": "Verification framework disabled"}), 403
        return f(*args, **kwargs)
    return decorated_function


def _get_verifier_principal() -> Optional[str]:
    """Extract principal from request (must be authenticated)."""
    from webapp.parser.utils.cert_utils import extract_client_principal
    principal, source = extract_client_principal(request.headers)
    return principal


def _require_verifier_tier(tier: str):
    """Decorator: Require minimum privilege tier for verification endpoints.
    
    Args:
        tier: One of "reviewer", "admin_reviewer", "admin_full_trust", "root_admin"
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            principal = _get_verifier_principal()
            if not principal:
                return jsonify({"error": "Unauthorized"}), 401
            
            # TODO: Check principal's tier from privilege_tiers module
            # For now, accept any authenticated principal
            return f(*args, **kwargs)
        return decorated_function
    return decorator


def _require_principal(tier: str = "REVIEWER"):
    """Alias for _require_verifier_tier for backwards compatibility."""
    return _require_verifier_tier(tier.lower())


# ============================================================================
# Verification Endpoints
# ============================================================================

@verification_bp.route("/system/mission", methods=["GET"])
@_require_verification_enabled
def get_system_mission():
    """Retrieve system mission and governance info.
    
    Returns:
        JSON with system authorship, mission, and local storage info
    """
    return jsonify({
        "author": SYSTEM_AUTHOR,
        "mission": SYSTEM_MISSION,
        "storage_type": "local_filesystem",
        "governance_url": "/SYSTEM_GOVERNANCE.md",
        "sync_api": "/api/verification/sync",
    })


@verification_bp.route("/log/stats", methods=["GET"])
@_require_verification_enabled
@_require_verifier_tier("reviewer")
def get_verification_stats():
    """Get verification audit trail statistics.
    
    Returns:
        JSON with counts by status, confidence, anomaly type
    """
    principal = _get_verifier_principal()
    try:
        vlog = VerificationLog(VERIFICATION_LOG_FILE)
        stats = vlog.get_stats()
        stats["retrieved_at"] = datetime.now(timezone.utc).isoformat()
        stats["retrieved_by"] = principal
        
        logger.info({
            "level": "INFO",
            "type": "verification",
            "message": "Verification stats retrieved",
            "session_id": None,
            "principal": principal,
            "total_entries": stats.get("total", 0),
        })
        
        return jsonify(stats)
    except Exception as e:
        logger.error({
            "level": "ERROR",
            "type": "verification",
            "message": f"Failed to retrieve verification stats: {e}",
            "session_id": None,
            "principal": principal,
        })
        return jsonify({"error": str(e)}), 500


@verification_bp.route("/log/entries", methods=["GET"])
@_require_verification_enabled
@_require_verifier_tier("reviewer")
def get_verification_entries():
    """Retrieve verification log entries (paginated).
    
    Query Parameters:
        limit: Max entries to return (default 100, max 1000)
        dl2_id: Filter by DL2 row ID (optional)
        status: Filter by status (approved|rejected|flagged|pending)
    
    Returns:
        JSON array of verification entries with pagination info
    """
    principal = _get_verifier_principal()
    try:
        limit = int(request.args.get("limit", 100))
        limit = max(1, min(1000, limit))
    except Exception:
        limit = 100
    
    dl2_id_filter = request.args.get("dl2_id", "").strip()
    status_filter = request.args.get("status", "").strip().lower()
    
    try:
        vlog = VerificationLog(VERIFICATION_LOG_FILE)
        all_entries = vlog.read_all()
        
        filtered = []
        for entry in all_entries:
            if dl2_id_filter and entry.dl2_id != dl2_id_filter:
                continue
            if status_filter and entry.status.value != status_filter:
                continue
            filtered.append(entry.to_dict())
            if len(filtered) >= limit:
                break
        
        logger.info({
            "level": "INFO",
            "type": "verification",
            "message": f"Verification entries retrieved (limit={limit})",
            "session_id": None,
            "principal": principal,
            "count": len(filtered),
        })
        
        return jsonify({
            "entries": filtered,
            "count": len(filtered),
            "limit": limit,
            "total_available": len(all_entries),
            "filters": {
                "dl2_id": dl2_id_filter or None,
                "status": status_filter or None,
            }
        })
    except Exception as e:
        logger.error({
            "level": "ERROR",
            "type": "verification",
            "message": f"Failed to retrieve verification entries: {e}",
            "session_id": None,
            "principal": principal,
        })
        return jsonify({"error": str(e)}), 500


@verification_bp.route("/submission", methods=["POST"])
@_require_verification_enabled
@_require_verifier_tier("admin_reviewer")
def submit_verification():
    """Submit a verification decision for a DL2 row.
    
    Request Body:
        {
            "dl2_id": "row_abc123",
            "dl2_data": {"candidate": "John Smith", "votes": "12345"},
            "dl1_id": "verified_row_abc123" (optional, if matching DL1 row),
            "status": "approved|rejected|flagged",
            "confidence": "high|medium|low|unsure",
            "notes": "Human explanation...",
            "anomalies": [
                {"type": "data_formatting", "field": "candidate", "description": "..."}
            ],
            "correction_data": {"candidate": "John Smith"} (if approved with corrections)
        }
    
    Returns:
        JSON with verification entry and audit trail confirmation
    """
    principal = _get_verifier_principal()
    if not principal:
        return jsonify({"error": "Unauthorized"}), 401
    
    try:
        data = request.get_json(force=True) or {}
    except Exception as e:
        logger.error({
            "level": "ERROR",
            "type": "verification",
            "message": f"Failed to parse verification submission: {e}",
            "session_id": None,
            "principal": principal,
        })
        return jsonify({"error": "Invalid JSON"}), 400
    
    # Validate required fields
    dl2_id = safe_strip(safe_get(data, "dl2_id"))
    dl2_data = safe_get(data, "dl2_data")
    dl1_id = safe_strip(safe_get(data, "dl1_id", None))
    status_str = safe_strip(safe_get(data, "status", "")).lower()
    confidence_str = safe_strip(safe_get(data, "confidence", "unsure")).lower()
    notes = safe_strip(safe_get(data, "notes", ""))
    anomalies = safe_get(data, "anomalies", [])
    correction_data = safe_get(data, "correction_data", {})
    
    if not dl2_id or not isinstance(dl2_data, dict):
        return jsonify({"error": "dl2_id and dl2_data required"}), 400
    
    try:
        status = VerificationStatus(status_str)
    except ValueError:
        return jsonify({"error": f"Invalid status: {status_str}"}), 400
    
    try:
        confidence = VerificationConfidence(confidence_str)
    except ValueError:
        return jsonify({"error": f"Invalid confidence: {confidence_str}"}), 400
    
    try:
        # Create lineage entry
        entry = VerificationLineageEntry(
            dl2_id=dl2_id,
            dl2_data=dl2_data,
            dl1_id=dl1_id,
            verifier_principal=principal,
            status=status,
            confidence=confidence,
            notes=notes,
            anomalies=anomalies if isinstance(anomalies, list) else [],
            correction_data=correction_data if isinstance(correction_data, dict) else {},
        )
        
        # Append to immutable log
        vlog = VerificationLog(VERIFICATION_LOG_FILE)
        success = vlog.append(entry)
        
        if not success:
            return jsonify({"error": "Failed to write verification log"}), 500
        
        logger.info({
            "level": "INFO",
            "type": "verification",
            "message": f"Verification submitted: {dl2_id} → {status.value}",
            "session_id": None,
            "principal": principal,
            "dl2_id": dl2_id,
            "status": status.value,
            "confidence": confidence.value,
            "entry_hash": entry.entry_hash,
        })
        
        return jsonify({
            "success": True,
            "entry": entry.to_dict(),
            "audit_trail_confirmed": True,
        }), 201
    
    except Exception as e:
        logger.error({
            "level": "ERROR",
            "type": "verification",
            "message": f"Failed to submit verification: {e}",
            "session_id": None,
            "principal": principal,
            "dl2_id": dl2_id,
        })
        return jsonify({"error": str(e)}), 500


@verification_bp.route("/comparison", methods=["POST"])
@_require_verification_enabled
@_require_verifier_tier("reviewer")
def compare_dl1_dl2():
    """Compare DL2 row against DL1 row and classify anomalies.
    
    Request Body:
        {
            "dl2_row": {"candidate": "John Smith", "votes": "12345"},
            "dl1_row": {"candidate": "JOHN SMITH", "votes": "12345"},
            "field_mapping": {"candidate": "candidate", "votes": "votes"} (optional)
        }
    
    Returns:
        JSON with anomaly classifications (by field)
    """
    principal = _get_verifier_principal()
    
    try:
        data = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400
    
    dl2_row = safe_get(data, "dl2_row", {})
    dl1_row = safe_get(data, "dl1_row", {})
    field_mapping = safe_get(data, "field_mapping") or {}
    
    if not isinstance(dl2_row, dict) or not isinstance(dl1_row, dict):
        return jsonify({"error": "dl2_row and dl1_row must be dicts"}), 400
    
    # Use field_mapping if provided; otherwise assume same field names
    if not field_mapping:
        field_mapping = {k: k for k in dl2_row.keys()}
    
    try:
        comparison = {
            "dl2_row": dl2_row,
            "dl1_row": dl1_row,
            "field_anomalies": {},
            "has_anomalies": False,
            "anomaly_count": 0,
        }
        
        for dl2_field, dl1_field in field_mapping.items():
            if dl2_field not in dl2_row or dl1_field not in dl1_row:
                continue
            
            dl2_val = dl2_row[dl2_field]
            dl1_val = dl1_row[dl1_field]
            
            is_anom, anom_type, description = classify_anomaly(dl2_val, dl1_val, dl2_field)
            
            comparison["field_anomalies"][dl2_field] = {
                "is_anomaly": is_anom,
                "anomaly_type": anom_type.value if anom_type else None,
                "description": description,
                "dl2_value": str(dl2_val),
                "dl1_value": str(dl1_val),
            }
            
            if is_anom:
                comparison["has_anomalies"] = True
                comparison["anomaly_count"] += 1
        
        logger.info({
            "level": "INFO",
            "type": "verification",
            "message": "DL1/DL2 comparison performed",
            "session_id": None,
            "principal": principal,
            "anomaly_count": comparison["anomaly_count"],
        })
        
        return jsonify(comparison)
    
    except Exception as e:
        logger.error({
            "level": "ERROR",
            "type": "verification",
            "message": f"Comparison failed: {e}",
            "session_id": None,
            "principal": principal,
        })
        return jsonify({"error": str(e)}), 500


@verification_bp.route("/export/dl1", methods=["GET"])
@_require_verification_enabled
@_require_verifier_tier("admin_full_trust")
def export_dl1_verified():
    """Export verified (DL1) rows in CSV format.
    
    Query Parameters:
        state: Filter by state (optional)
        county: Filter by county (optional)
        contest: Filter by contest (optional)
        limit: Max rows (default 1000)
    
    Returns:
        CSV file with verified election data
    """
    principal = _get_verifier_principal()
    
    try:
        vlog = VerificationLog(VERIFICATION_LOG_FILE)
        entries = vlog.read_all()
        
        # Filter to approved entries
        approved = [e for e in entries if e.status == VerificationStatus.APPROVED]
        
        if not approved:
            return Response("no_verified_data", status=204, mimetype="text/plain")
        
        # Build CSV
        import csv
        from io import StringIO
        
        output = StringIO()
        writer = csv.writer(output)
        
        # Headers from first entry
        headers = list(approved[0].dl2_data.keys())
        headers.extend(["verified_at", "verified_by", "dl2_id", "verification_confidence"])
        writer.writerow(headers)
        
        # Rows
        for entry in approved:
            row = [entry.dl2_data.get(h, "") for h in approved[0].dl2_data.keys()]
            row.extend([
                entry.timestamp,
                entry.verifier_principal,
                entry.dl2_id,
                entry.confidence.value,
            ])
            writer.writerow(row)
        
        csv_content = output.getvalue()
        
        logger.info({
            "level": "INFO",
            "type": "verification",
            "message": f"DL1 verified export ({len(approved)} rows)",
            "session_id": None,
            "principal": principal,
        })
        
        return Response(
            csv_content,
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=dl1_verified.csv"}
        )
    
    except Exception as e:
        logger.error({
            "level": "ERROR",
            "type": "verification",
            "message": f"Failed to export DL1 verified: {e}",
            "session_id": None,
            "principal": principal,
        })
        return jsonify({"error": str(e)}), 500


# ============================================================================
# DL1/DL2 Local File System Sync Endpoints (Phase 2)
# ============================================================================

@verification_bp.route("/sync/status", methods=["GET"])
@_require_verification_enabled
@_require_principal("REVIEWER")
def sync_status():
    """
    Get DL1/DL2 local storage synchronization status.
    
    Returns:
    {
        "available": bool,
        "storage_path": str,
        "stats": {
            "dl2": {"file_count": int, "total_size_bytes": int},
            "dl1": {"file_count": int, "total_size_bytes": int},
            "total_promoted": int,
            "dedup_groups": int
        }
    }
    """
    if not SYNC_AVAILABLE:
        return jsonify({"available": False, "reason": "LocalStorageSync not available"}), 503
    
    try:
        verification_dir = os.path.join(
            os.environ.get('CONTEXT_LIBRARY_DIR', 'context_library'),
            'verification'
        )
        sync = LocalStorageSync(verification_dir)
        available = sync.is_available()
        
        payload = {
            "available": available,
            "storage_path": str(verification_dir),
            "stats": sync.get_storage_stats() if available else None
        }
        return jsonify(payload)
    except Exception as e:
        logger.error({
            "level": "ERROR",
            "type": "sync",
            "message": f"Failed to get sync status: {e}"
        })
        return jsonify({"error": str(e)}), 500


@verification_bp.route("/sync/dl2/list", methods=["GET"])
@_require_verification_enabled
@_require_principal("REVIEWER")
def sync_list_dl2():
    """
    List unverified samples in DL2 (local storage).
    
    Query params:
    - limit: Max files to return (default: 50)
    
    Returns:
    {
        "success": bool,
        "files": [{
            "file_id": str,
            "filename": str,
            "size_bytes": int,
            "hash": str,
            "created_at": str,
            "promoted": bool
        }],
        "timestamp": str
    }
    """
    if not SYNC_AVAILABLE:
        return jsonify({"error": "LocalStorageSync not available"}), 503
    
    try:
        limit = request.args.get("limit", type=int) or 50
        
        verification_dir = os.path.join(
            os.environ.get('CONTEXT_LIBRARY_DIR', 'context_library'),
            'verification'
        )
        sync = LocalStorageSync(verification_dir)
        if not sync.is_available():
            return jsonify({"error": "Storage not available"}), 503
        
        files = sync.list_dl2_samples(limit=limit)
        
        return jsonify({
            "success": True,
            "files": files,
            "count": len(files),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        logger.error({
            "level": "ERROR",
            "type": "sync",
            "message": f"Failed to list DL2: {e}"
        })
        return jsonify({"error": str(e)}), 500


@verification_bp.route("/sync/dl1/list", methods=["GET"])
@_require_verification_enabled
@_require_principal("REVIEWER")
def sync_list_dl1():
    """
    List verified/approved samples in DL1 (local storage).
    
    Query params:
    - limit: Max files to return (default: 50)
    
    Returns:
    {
        "success": bool,
        "files": [{
            "file_id": str,
            "filename": str,
            "size_bytes": int,
            "hash": str,
            "approved_at": str
        }],
        "timestamp": str
    }
    """
    if not SYNC_AVAILABLE:
        return jsonify({"error": "LocalStorageSync not available"}), 503
    
    try:
        limit = request.args.get("limit", type=int) or 50
        
        verification_dir = os.path.join(
            os.environ.get('CONTEXT_LIBRARY_DIR', 'context_library'),
            'verification'
        )
        sync = LocalStorageSync(verification_dir)
        if not sync.is_available():
            return jsonify({"error": "Storage not available"}), 503
        
        files = sync.list_dl1_approved(limit=limit)
        
        return jsonify({
            "success": True,
            "files": files,
            "count": len(files),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        logger.error({
            "level": "ERROR",
            "type": "sync",
            "message": f"Failed to list DL1: {e}"
        })
        return jsonify({"error": str(e)}), 500


@verification_bp.route("/sync/dl2/stage", methods=["POST"])
@_require_verification_enabled
@_require_principal("ADMIN_REVIEWER")
def sync_stage_dl2():
    """
    Stage a new extracted file into DL2 (unverified dataset).
    
    Body:
    {
        "source_file": str,           # Path to extracted CSV file
        "file_id": str|null,          # Optional: custom file ID
        "metadata": dict|null         # Optional: extraction metadata
    }
    
    Returns:
    {
        "success": bool,
        "file_id": str,
        "storage_path": str,
        "timestamp": str
    }
    """
    if not SYNC_AVAILABLE:
        return jsonify({"error": "LocalStorageSync not available"}), 503
    
    try:
        data = request.get_json(force=True) or {}
        source_file = safe_strip(data.get("source_file", ""))
        file_id = safe_strip(data.get("file_id", ""))
        metadata = data.get("metadata")
        
        if not source_file:
            return jsonify({"error": "source_file required"}), 400
        
        if not os.path.exists(source_file):
            return jsonify({"error": f"Source file not found: {source_file}"}), 404
        
        verification_dir = os.path.join(
            os.environ.get('CONTEXT_LIBRARY_DIR', 'context_library'),
            'verification'
        )
        sync = LocalStorageSync(verification_dir)
        
        staged_id = sync.stage_dl2_file(source_file, file_id=file_id or None, metadata=metadata)
        
        return jsonify({
            "success": True,
            "file_id": staged_id,
            "storage_path": str(sync.dl2_dir / f"{staged_id}.csv"),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        logger.error({
            "level": "ERROR",
            "type": "sync",
            "message": f"Failed to stage DL2: {e}"
        })
        return jsonify({"error": str(e)}), 500


@verification_bp.route("/sync/promote", methods=["POST"])
@_require_verification_enabled
@_require_principal("ADMIN_FULL_TRUST")
def sync_promote():
    """
    Promote an approved DL2 file to verified DL1 dataset.
    
    Body:
    {
        "file_id": str,                # File ID in DL2
        "verifier_principal": str,     # Principal approving
        "verification_notes": str      # Approval notes (optional)
    }
    
    Returns:
    {
        "success": bool,
        "file_id": str,
        "promotion_record": {...}
    }
    """
    if not SYNC_AVAILABLE:
        return jsonify({"error": "LocalStorageSync not available"}), 503
    
    try:
        data = request.get_json(force=True) or {}
        file_id = safe_strip(data.get("file_id", ""))
        verifier_principal = safe_strip(data.get("verifier_principal", ""))
        verification_notes = safe_strip(data.get("verification_notes", ""))
        
        if not file_id:
            return jsonify({"error": "file_id required"}), 400
        
        verification_dir = os.path.join(
            os.environ.get('CONTEXT_LIBRARY_DIR', 'context_library'),
            'verification'
        )
        sync = LocalStorageSync(verification_dir)
        
        promotion_record = sync.promote_to_dl1(
            file_id,
            verifier_principal=verifier_principal,
            verification_notes=verification_notes
        )
        
        return jsonify({
            "success": True,
            "file_id": file_id,
            "promotion_record": promotion_record
        })
    except FileNotFoundError as e:
        logger.warning({
            "level": "WARNING",
            "type": "sync",
            "message": f"File not found during promotion: {e}"
        })
        return jsonify({"error": str(e)}), 404
    except FileExistsError as e:
        logger.warning({
            "level": "WARNING",
            "type": "sync",
            "message": f"File already exists in DL1: {e}"
        })
        return jsonify({"error": str(e)}), 409
    except Exception as e:
        logger.error({
            "level": "ERROR",
            "type": "sync",
            "message": f"Failed to promote DL2→DL1: {e}"
        })
        return jsonify({"error": str(e)}), 500


__all__ = ["verification_bp"]
