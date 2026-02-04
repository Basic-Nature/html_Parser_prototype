"""
Audit Trail Router - Multi-Tier Compliance Logging

Purpose:
- Route decisions to tier-specific audit logs (5 separate JSONL files)
- Assign event_chain_ids for breaking-chain forensic investigation
- Maintain compliance grade audit trails per tier
- Track every decision with principal, confidence, timestamp

Log Files:
1. admin_full_trust_decisions.jsonl - Root and FULL_TRUST tier decisions only
2. admin_reviewer_decisions.jsonl - REVIEWER tier decisions only
3. trust_history.jsonl - STANDARD_USER tier decisions (existing)
4. factor_chain_anomalies.jsonl - Breaking chains (all tiers, separate file)
5. admin_trust_decisions.jsonl - Compliance summary metadata
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import LOG_DIR
from ..utils.logger_singleton import logger

# ============================================================================
# AUDIT LOG PATHS
# ============================================================================

ADMIN_FULL_TRUST_LOG = Path(LOG_DIR) / "admin_full_trust_decisions.jsonl"
ADMIN_REVIEWER_LOG = Path(LOG_DIR) / "admin_reviewer_decisions.jsonl"
TRUST_HISTORY_LOG = Path(LOG_DIR) / "trust_history.jsonl"
FACTOR_CHAIN_ANOMALIES_LOG = Path(LOG_DIR) / "factor_chain_anomalies.jsonl"
ADMIN_TRUST_DECISIONS_LOG = Path(LOG_DIR) / "admin_trust_decisions.jsonl"

# Lock for atomic writes
_AUDIT_LOCK = threading.Lock()


def _ensure_audit_logs():
    """Create all audit log files if they don't exist."""
    for log_file in [
        ADMIN_FULL_TRUST_LOG,
        ADMIN_REVIEWER_LOG,
        TRUST_HISTORY_LOG,
        FACTOR_CHAIN_ANOMALIES_LOG,
        ADMIN_TRUST_DECISIONS_LOG
    ]:
        try:
            log_file.touch(exist_ok=True)
        except Exception:
            pass


_ensure_audit_logs()


# ============================================================================
# AUDIT ENTRY STRUCTURE
# ============================================================================

@dataclass
class AuditEntry:
    """Single audit trail entry (JSONL line, text-serializable)."""
    event_chain_id: str                        # UUID for this decision + related events
    timestamp: str                             # ISO 8601
    principal: str                             # Who made the decision
    principal_tier: int                        # 0=standard, 1=reviewer, 2=full, 3=root
    
    # Decision context
    decision_type: str                         # "trust_score_computed" | "snapshot_decided" | "quarantine_flagged" | "reject_flagged"
    url: str                                   # Target URL
    decision_reason: str                       # Plain text reason
    
    # Confidence assessment
    confidence_score: float                    # 0.0-1.0
    trust_score: Optional[int] = None          # URL trust score (0-100)
    
    # Factor chain integration
    factor_chain_id: str = ""                  # Linked factor_chain_tracker.py chain_id
    breaking_chain_detected: bool = False      # Factor anomalies found
    anomaly_severity: str = "none"             # "none" | "low" | "medium" | "high" | "critical"
    
    # Metadata
    session_id: str = ""                       # Session context
    output_file: str = ""                      # CSV output path (if applicable)
    
    def to_json_line(self) -> str:
        """Serialize to JSONL format (single line)."""
        data = {
            "event_chain_id": self.event_chain_id,
            "timestamp": self.timestamp,
            "principal": self.principal,
            "principal_tier": self.principal_tier,
            "decision_type": self.decision_type,
            "url": self.url,
            "decision_reason": self.decision_reason,
            "confidence_score": self.confidence_score,
            "trust_score": self.trust_score,
            "factor_chain_id": self.factor_chain_id,
            "breaking_chain_detected": self.breaking_chain_detected,
            "anomaly_severity": self.anomaly_severity,
            "session_id": self.session_id,
            "output_file": self.output_file
        }
        return json.dumps(data, separators=(",", ":"))


@dataclass
class ComplianceMetadata:
    """Aggregate metadata for admin_trust_decisions.jsonl."""
    day: str                                   # YYYY-MM-DD
    root_admin_decisions: int = 0
    full_trust_decisions: int = 0
    reviewer_decisions: int = 0
    standard_decisions: int = 0
    breaking_chains_detected: int = 0
    malicious_acts_removed: int = 0
    quarantined_urls_count: int = 0
    rejected_urls_count: int = 0
    
    def to_json_line(self) -> str:
        """Serialize to JSONL format."""
        data = {
            "day": self.day,
            "root_admin_decisions": self.root_admin_decisions,
            "full_trust_decisions": self.full_trust_decisions,
            "reviewer_decisions": self.reviewer_decisions,
            "standard_decisions": self.standard_decisions,
            "breaking_chains_detected": self.breaking_chains_detected,
            "malicious_acts_removed": self.malicious_acts_removed,
            "quarantined_urls_count": self.quarantined_urls_count,
            "rejected_urls_count": self.rejected_urls_count
        }
        return json.dumps(data, separators=(",", ":"))


# ============================================================================
# AUDIT ROUTING & LOGGING
# ============================================================================

def log_decision_with_tier(
    event: AuditEntry,
    privilege_tier: int
) -> bool:
    """
    Route audit entry to tier-specific log.
    Returns True if successful.
    """
    try:
        json_line = event.to_json_line()
        
        # Determine log file(s) to write to
        logs_to_write = []
        
        # Root admin decisions
        if privilege_tier == 3:
            logs_to_write.append(ADMIN_FULL_TRUST_LOG)
        # Full trust decisions
        elif privilege_tier == 2:
            logs_to_write.append(ADMIN_FULL_TRUST_LOG)
        # Reviewer decisions
        elif privilege_tier == 1:
            logs_to_write.append(ADMIN_REVIEWER_LOG)
        # Standard user decisions
        else:
            logs_to_write.append(TRUST_HISTORY_LOG)
        
        # Always log anomalies to factor_chain_anomalies if breaking chain detected
        if event.breaking_chain_detected:
            logs_to_write.append(FACTOR_CHAIN_ANOMALIES_LOG)
        
        # Atomic write to all applicable logs
        with _AUDIT_LOCK:
            for log_file in logs_to_write:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(json_line + "\n")
                    f.flush()
                    os.fsync(f.fileno())
        
        logger.info({
            "level": "INFO",
            "type": "audit",
            "message": f"Decision logged: {event.decision_type}",
            "event_chain_id": event.event_chain_id,
            "principal": event.principal,
            "principal_tier": privilege_tier,
            "url": event.url,
            "confidence": event.confidence_score,
            "session_id": event.session_id
        })
        
        return True
    except Exception as exc:
        logger.error({
            "level": "ERROR",
            "type": "audit",
            "message": f"Failed to log decision: {exc}",
            "event_chain_id": event.event_chain_id,
            "session_id": event.session_id
        })
        return False


def add_event_chain_id() -> str:
    """
    Generate and return a new event_chain_id (UUID).
    Used to correlate related events across decision points.
    """
    return str(uuid.uuid4())[:16]


# ============================================================================
# COMPLIANCE SUMMARIZATION
# ============================================================================

def summarize_daily_compliance(date_str: Optional[str] = None) -> ComplianceMetadata:
    """
    Summarize compliance metrics for a given day.
    
    Args:
        date_str: YYYY-MM-DD format (defaults to today)
    
    Returns:
        ComplianceMetadata with aggregated counts
    """
    if not date_str:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    metadata = ComplianceMetadata(day=date_str)
    
    try:
        # Scan all audit logs for entries from this day
        for log_file in [
            ADMIN_FULL_TRUST_LOG,
            ADMIN_REVIEWER_LOG,
            TRUST_HISTORY_LOG,
            FACTOR_CHAIN_ANOMALIES_LOG
        ]:
            if not log_file.exists():
                continue
            
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        entry = json.loads(line)
                    except Exception:
                        continue
                    
                    # Check timestamp
                    ts = entry.get("timestamp", "")
                    if not ts.startswith(date_str):
                        continue
                    
                    # Increment counters by tier
                    tier = entry.get("principal_tier", 0)
                    if tier == 3:
                        metadata.root_admin_decisions += 1
                    elif tier == 2:
                        metadata.full_trust_decisions += 1
                    elif tier == 1:
                        metadata.reviewer_decisions += 1
                    else:
                        metadata.standard_decisions += 1
                    
                    # Check for breaking chains
                    if entry.get("breaking_chain_detected"):
                        metadata.breaking_chains_detected += 1
                    
                    # Check decision type
                    decision = entry.get("decision_type", "")
                    if "quarantine" in decision.lower():
                        metadata.quarantined_urls_count += 1
                    elif "reject" in decision.lower():
                        metadata.rejected_urls_count += 1
        
        # Query malicious acts from confidence scorer log
        malicious_log = Path(LOG_DIR) / "malicious_acts_removed.txt"
        if malicious_log.exists():
            with open(malicious_log, "r", encoding="utf-8") as f:
                for line in f:
                    if date_str in line:
                        metadata.malicious_acts_removed += 1
        
        return metadata
    except Exception as exc:
        logger.error({
            "level": "ERROR",
            "type": "audit",
            "message": f"Failed to summarize compliance: {exc}",
            "session_id": None
        })
        return metadata


def write_compliance_summary(date_str: Optional[str] = None) -> bool:
    """
    Write daily compliance summary to admin_trust_decisions.jsonl.
    Returns True if successful.
    """
    try:
        metadata = summarize_daily_compliance(date_str)
        json_line = metadata.to_json_line()
        
        with _AUDIT_LOCK:
            with open(ADMIN_TRUST_DECISIONS_LOG, "a", encoding="utf-8") as f:
                f.write(json_line + "\n")
                f.flush()
                os.fsync(f.fileno())
        
        logger.info({
            "level": "INFO",
            "type": "audit",
            "message": f"Compliance summary written for {metadata.day}",
            "day": metadata.day,
            "root_decisions": metadata.root_admin_decisions,
            "full_trust_decisions": metadata.full_trust_decisions,
            "reviewer_decisions": metadata.reviewer_decisions,
            "session_id": None
        })
        
        return True
    except Exception as exc:
        logger.error({
            "level": "ERROR",
            "type": "audit",
            "message": f"Failed to write compliance summary: {exc}",
            "session_id": None
        })
        return False


# ============================================================================
# FORENSIC QUERY HELPERS
# ============================================================================

def get_audit_entries_for_chain(event_chain_id: str) -> list[Dict[str, Any]]:
    """Retrieve all audit entries for a given event_chain_id."""
    entries = []
    
    for log_file in [
        ADMIN_FULL_TRUST_LOG,
        ADMIN_REVIEWER_LOG,
        TRUST_HISTORY_LOG,
        FACTOR_CHAIN_ANOMALIES_LOG
    ]:
        if not log_file.exists():
            continue
        
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                
                if entry.get("event_chain_id") == event_chain_id:
                    entries.append(entry)
    
    return entries


def get_principal_decisions(
    principal: str,
    limit: int = 100,
    days_back: int = 7
) -> list[Dict[str, Any]]:
    """Retrieve recent decisions by a principal (for audit review)."""
    entries = []
    
    for log_file in [
        ADMIN_FULL_TRUST_LOG,
        ADMIN_REVIEWER_LOG,
        TRUST_HISTORY_LOG
    ]:
        if not log_file.exists():
            continue
        
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or len(entries) >= limit:
                    continue
                
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                
                if entry.get("principal") == principal:
                    entries.append(entry)
    
    return sorted(entries, key=lambda e: e.get("timestamp", ""), reverse=True)[:limit]
