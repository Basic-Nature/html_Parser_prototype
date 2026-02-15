"""
Quarantine Queue: Transparent URL quarantine workflow with audit trails.

Provides:
- QuarantineEntry: Immutable record of why a URL was quarantined
- QuarantineQueue: Persistent FIFO queue with audit logging
- QuarantineReviewSession: Interactive review with certification of decisions
- Audit trails: Every decision logged with principal, timestamp, rationale

Design:
- All quarantine entries stored in JSON with human-readable rationale
- Each entry explains: what data was collected, why, and what it means
- Review process: approve/reject/request-more-info with certification
- Principal-auditable: All reviews signed with principal + timestamp
- Transparent: Users can see exactly what triggered quarantine + why
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import LOG_DIR
from ..utils.logger_singleton import logger

# ===== ENUMS =====

class QuarantineReason(str, Enum):
    """Explicit reasons for quarantine with human-readable descriptions."""
    
    LOW_TRUST_SCORE = "low_trust_score"
    SUSPICIOUS_HOST = "suspicious_host"
    INVALID_SCHEME = "invalid_scheme"
    CLOUDFLARE_CHALLENGE = "cloudflare_challenge"
    EXTRACTION_FAILURE = "extraction_failure"
    ANOMALY_DETECTED = "anomaly_detected"
    MANUAL_REVIEW_REQUESTED = "manual_review_requested"
    
    @property
    def explanation(self) -> str:
        """Human-readable explanation of what this reason means."""
        explanations = {
            "low_trust_score": "URL trust score below threshold. Requires human review before processing.",
            "suspicious_host": "Hostname matches known suspicious pattern or CDN blocklist.",
            "invalid_scheme": "URL scheme is not http/https or missing required components.",
            "cloudflare_challenge": "Cloudflare protection detected. Requires JS execution or manual verification.",
            "extraction_failure": "Automated parsing failed. Needs manual inspection before retry.",
            "anomaly_detected": "AI anomaly detection flagged unusual patterns in extracted data.",
            "manual_review_requested": "User or system explicitly requested manual review.",
        }
        return explanations.get(self.value, "Unknown reason - requires review.")
    
    @property
    def impact(self) -> str:
        """What impact does this have on the user's session?"""
        impacts = {
            "low_trust_score": "Processing paused until URL is certified by reviewer.",
            "suspicious_host": "URL blocked for this session. Admin approval required to whitelist.",
            "invalid_scheme": "URL rejected outright. Verify URL format and try again.",
            "cloudflare_challenge": "Cannot bypass automatically. Try opening URL in browser first.",
            "extraction_failure": "No data extracted. Manual correction may be needed.",
            "anomaly_detected": "Data integrity concern. Review recommended before use.",
            "manual_review_requested": "Awaiting human review before continuing.",
        }
        return impacts.get(self.value, "Awaiting review.")


class ReviewStatus(str, Enum):
    """Status of quarantine review."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_INFO = "needs_more_info"
    APPEALED = "appealed"


# ===== DATA CLASSES =====

@dataclass
class DataCollectionNotice:
    """Explain what data was collected and why."""
    data_type: str  # "trust_score_factors", "extraction_metadata", etc.
    description: str  # Human-readable explanation
    retention_days: int = 30  # How long we keep it
    usage: str = ""  # What it's used for (anomaly detection, statistics, etc.)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QuarantineEntry:
    """Immutable record of URL quarantine with full transparency."""
    
    # Identity (required fields first)
    quarantine_id: str
    url: str
    timestamp: str  # ISO 8601
    reason: str  # QuarantineReason.value
    
    # Identity (optional fields)
    session_id: Optional[str] = None
    principal: Optional[str] = None
    
    # Why it was quarantined (optional analysis)
    trust_score: Optional[float] = None
    trust_factors: Optional[Dict[str, Any]] = None
    
    # What data was collected (with explanations)
    data_collected: List[DataCollectionNotice] = field(default_factory=list)
    
    # Audit trail
    extraction_attempts: int = 0
    error_messages: List[str] = field(default_factory=list)
    
    # Review state
    review_status: str = ReviewStatus.PENDING.value
    review_history: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def reason_explanation(self) -> str:
        """Get human-readable explanation of quarantine reason."""
        try:
            return QuarantineReason(self.reason).explanation
        except ValueError:
            return f"Quarantine reason: {self.reason}"
    
    @property
    def reason_impact(self) -> str:
        """Get explanation of what this quarantine means for the user."""
        try:
            return QuarantineReason(self.reason).impact
        except ValueError:
            return "Awaiting review."
    
    def add_review(
        self,
        status: ReviewStatus,
        reviewer_principal: str,
        notes: str,
        certification_reason: str = ""
    ) -> None:
        """Record a review decision with full audit trail."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reviewer": reviewer_principal,
            "status": status.value,
            "notes": notes,
            "certification_reason": certification_reason,  # Why this decision was made
        }
        self.review_history.append(entry)
        self.review_status = status.value
    
    def to_json(self) -> str:
        """Serialize to JSON for storage."""
        data = asdict(self)
        return json.dumps(data, indent=2, default=str)
    
    @staticmethod
    def from_json(json_str: str) -> QuarantineEntry:
        """Deserialize from JSON."""
        data = json.loads(json_str)
        data["data_collected"] = [
            DataCollectionNotice(**dc) if isinstance(dc, dict) else dc
            for dc in data.get("data_collected", [])
        ]
        return QuarantineEntry(**data)


# ===== QUARANTINE QUEUE =====

class QuarantineQueue:
    """Persistent quarantine queue with transparent audit trails."""
    
    def __init__(self, queue_dir: Optional[Path] = None):
        """
        Initialize quarantine queue.
        
        Args:
            queue_dir: Directory for quarantine records (default: LOG_DIR/quarantine)
        """
        self.queue_dir = Path(queue_dir or LOG_DIR) / "quarantine"
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        
        self.pending_file = self.queue_dir / "pending.jsonl"
        self.reviewed_file = self.queue_dir / "reviewed.jsonl"
        self._lock = threading.RLock()
    
    def enqueue(
        self,
        url: str,
        reason: QuarantineReason,
        session_id: Optional[str] = None,
        principal: Optional[str] = None,
        trust_score: Optional[float] = None,
        trust_factors: Optional[Dict[str, Any]] = None,
        data_notices: Optional[List[DataCollectionNotice]] = None,
        error_messages: Optional[List[str]] = None,
    ) -> QuarantineEntry:
        """
        Enqueue a URL for quarantine with full explanation.
        
        Args:
            url: The URL being quarantined
            reason: QuarantineReason enum
            session_id: Session that triggered quarantine
            principal: User/principal that triggered quarantine
            trust_score: Trust score if reason is LOW_TRUST_SCORE
            trust_factors: Breakdown of trust factors
            data_notices: Explain what data was collected and why
            error_messages: Any error messages encountered
        
        Returns:
            QuarantineEntry record
        """
        quarantine_id = hashlib.sha256(
            f"{url}:{int(time.time()*1000)}".encode()
        ).hexdigest()[:16]
        
        # Default data collection notices if not provided
        if data_notices is None:
            data_notices = [
                DataCollectionNotice(
                    data_type="trust_score",
                    description="Automated trust assessment of URL. Used to prevent malicious extraction attempts.",
                    usage="Security filtering; statistical analysis of source reliability",
                ),
                DataCollectionNotice(
                    data_type="url_metadata",
                    description="URL hostname, scheme, and structure. Used to identify suspicious patterns.",
                    usage="Blocklist matching; domain reputation analysis",
                ),
            ]
        
        entry = QuarantineEntry(
            quarantine_id=quarantine_id,
            url=url,
            timestamp=datetime.now(timezone.utc).isoformat(),
            session_id=session_id,
            principal=principal,
            reason=reason.value,
            trust_score=trust_score,
            trust_factors=trust_factors,
            data_collected=data_notices,
            error_messages=error_messages or [],
        )
        
        # Persist to pending queue
        with self._lock:
            with open(self.pending_file, "a", encoding="utf-8") as f:
                f.write(entry.to_json() + "\n")
        
        logger.info({
            "level": "INFO",
            "type": "quarantine",
            "message": f"[Quarantine] URL enqueued: {quarantine_id}",
            "quarantine_id": quarantine_id,
            "url": url,
            "reason": reason.value,
            "reason_explanation": entry.reason_explanation,
            "session_id": session_id,
            "principal": principal,
        })
        
        return entry
    
    def get_pending(self, limit: int = 100) -> List[QuarantineEntry]:
        """Get pending quarantine entries."""
        if not self.pending_file.exists():
            return []
        
        entries = []
        try:
            with open(self.pending_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = QuarantineEntry.from_json(line)
                        if entry.review_status == ReviewStatus.PENDING.value:
                            entries.append(entry)
                            if len(entries) >= limit:
                                break
                    except Exception as e:
                        logger.warning({
                            "level": "WARNING",
                            "type": "quarantine",
                            "message": f"Failed to parse quarantine entry: {e}"
                        })
        except Exception as e:
            logger.error({
                "level": "ERROR",
                "type": "quarantine",
                "message": f"Failed to read quarantine queue: {e}"
            })
        
        return entries

    def has_pending_url(self, url: str) -> bool:
        """Return True if the URL is already pending review."""
        if not url or not self.pending_file.exists():
            return False
        try:
            with open(self.pending_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = QuarantineEntry.from_json(line)
                        if entry.url == url and entry.review_status == ReviewStatus.PENDING.value:
                            return True
                    except Exception:
                        continue
        except Exception:
            return False
        return False

    def get_latest_review_status_for_url(self, url: str) -> str | None:
        """Return the latest review status for a URL from reviewed history."""
        if not url or not self.reviewed_file.exists():
            return None
        latest_status = None
        latest_ts = None
        try:
            with open(self.reviewed_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = QuarantineEntry.from_json(line)
                    except Exception:
                        continue
                    if entry.url != url:
                        continue
                    ts = entry.timestamp
                    if latest_ts is None or (isinstance(ts, str) and ts > latest_ts):
                        latest_ts = ts
                        latest_status = entry.review_status
        except Exception:
            return None
        return latest_status
    
    def record_review(
        self,
        quarantine_id: str,
        status: ReviewStatus,
        reviewer_principal: str,
        notes: str,
        certification_reason: str = "",
    ) -> bool:
        """
        Record a review decision.
        
        Args:
            quarantine_id: ID of quarantine entry
            status: Review status (APPROVED, REJECTED, etc.)
            reviewer_principal: Principal performing review
            notes: Review notes
            certification_reason: Why this decision was made (transparency)
        
        Returns:
            True if recorded, False if not found
        """
        with self._lock:
            entries = []
            found = False
            
            if self.pending_file.exists():
                with open(self.pending_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        entry = QuarantineEntry.from_json(line)
                        if entry.quarantine_id == quarantine_id:
                            entry.add_review(
                                status,
                                reviewer_principal,
                                notes,
                                certification_reason
                            )
                            found = True
                        entries.append(entry)
            
            if found:
                # Write reviewed entry to reviewed file
                approved_entry = next(
                    (e for e in entries if e.quarantine_id == quarantine_id),
                    None
                )
                if approved_entry:
                    with open(self.reviewed_file, "a", encoding="utf-8") as f:
                        f.write(approved_entry.to_json() + "\n")
                
                # Rewrite pending file without reviewed entry
                with open(self.pending_file, "w", encoding="utf-8") as f:
                    for entry in entries:
                        if entry.quarantine_id != quarantine_id:
                            f.write(entry.to_json() + "\n")
                
                logger.info({
                    "level": "INFO",
                    "type": "quarantine",
                    "message": f"[Quarantine] Review recorded: {quarantine_id}",
                    "quarantine_id": quarantine_id,
                    "status": status.value,
                    "reviewer": reviewer_principal,
                    "certification_reason": certification_reason,
                })
        
        return found
    
    def get_stats(self) -> Dict[str, Any]:
        """Get quarantine queue statistics."""
        pending = self.get_pending(limit=10000)
        pending_by_reason = {}
        for entry in pending:
            reason = entry.reason
            pending_by_reason[reason] = pending_by_reason.get(reason, 0) + 1
        
        return {
            "total_pending": len(pending),
            "pending_by_reason": pending_by_reason,
            "oldest_entry": pending[-1].timestamp if pending else None,
        }


# Singleton
_default_queue: Optional[QuarantineQueue] = None


def get_quarantine_queue() -> QuarantineQueue:
    """Get singleton quarantine queue."""
    global _default_queue
    if _default_queue is None:
        _default_queue = QuarantineQueue()
    return _default_queue
