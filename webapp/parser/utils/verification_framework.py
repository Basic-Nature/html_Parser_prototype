"""Dual-Truth Verification Framework for Smart Elections Parser

Implements the verification workflow that validates AI-extracted data (DL2) 
against human-verified ground truth (DL1).

System Mission:
  - Protect the voice of the people by preserving legitimate votes
  - Detect unintentional data errors at acceptable thresholds
  - NOT designed to detect criminal fraud or malicious interference
  - Collaborative intelligence: mechanical efficiency + biological wisdom

Original Author & Conception:
  Juancarlos Barragan
  DOB: March 18, 1996
  Location: 6858 S 12th Ave, Tucson, AZ
  Date: February 2026

Architecture:
  DL2 (AI-Extracted) → Human Review → DL1 (Verified Ground Truth)
  
  Verification Log tracks:
  - Source data (DL2 row, metadata)
  - Verifier identity (principal, timestamp)
  - Verification decision (approved, rejected, flagged)
  - Confidence level (high, medium, low)
  - Lineage (traceability from extraction → verification → promotion)
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .logger_singleton import logger


class VerificationStatus(str, Enum):
    """Verification decision states for DL2 rows."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    FLAGGED = "flagged"  # Requires secondary review


class VerificationConfidence(str, Enum):
    """Confidence level for human verification decisions."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNSURE = "unsure"


class AnomalyType(str, Enum):
    """Classification of detected anomalies (unintentional mistakes only)."""
    DATA_FORMATTING = "data_formatting"  # E.g., extra spaces, case mismatch
    NUMERIC_PRECISION = "numeric_precision"  # Rounding, decimal places
    MISSING_FIELD = "missing_field"  # Blank or null value
    DUPLICATE_RECORD = "duplicate_record"  # Same data reported twice
    ENCODING_ISSUE = "encoding_issue"  # UTF-8 or special character problem
    EXTRACTION_ERROR = "extraction_error"  # Parser did not extract field correctly
    CONTEXT_MISMATCH = "context_mismatch"  # Data doesn't match geographic/temporal context
    OTHER = "other"  # Unclassified


class VerificationLineageEntry:
    """Single row in the verification audit trail.
    
    Immutable record of: DL2 row extraction → human review → DL1 promotion
    """
    
    def __init__(
        self,
        dl2_id: str,
        dl2_data: Dict[str, Any],
        dl1_id: Optional[str],
        verifier_principal: str,
        status: VerificationStatus,
        confidence: VerificationConfidence,
        notes: str = "",
        anomalies: Optional[List[Dict[str, Any]]] = None,
        correction_data: Optional[Dict[str, Any]] = None,
    ):
        """Initialize verification lineage record.
        
        Args:
            dl2_id: Unique ID of extracted (DL2) row
            dl2_data: Full extracted data payload
            dl1_id: ID of matched DL1 row (if any)
            verifier_principal: Human reviewer's principal ID
            status: Verification decision (approved, rejected, flagged)
            confidence: Reviewer's confidence in decision
            notes: Human-written explanation
            anomalies: List of detected unintentional mistakes (by type)
            correction_data: Corrected values (if approved with fixes)
        """
        self.dl2_id = dl2_id
        self.dl2_data = dl2_data
        self.dl1_id = dl1_id
        self.verifier_principal = verifier_principal
        self.status = status
        self.confidence = confidence
        self.notes = notes
        self.anomalies = anomalies or []
        self.correction_data = correction_data or {}
        
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.entry_hash = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """Compute SHA256 hash of entry for immutability verification.
        
        Returns:
            40-char hex hash
        """
        payload = {
            "dl2_id": self.dl2_id,
            "dl2_data_keys": sorted(self.dl2_data.keys()),
            "dl1_id": self.dl1_id,
            "verifier": self.verifier_principal,
            "status": self.status.value,
            "timestamp": self.timestamp,
        }
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:40]
    
    def to_dict(self) -> Dict[str, Any]:
        """Export to JSON-serializable dict."""
        return {
            "dl2_id": self.dl2_id,
            "dl2_data": self.dl2_data,
            "dl1_id": self.dl1_id,
            "verifier_principal": self.verifier_principal,
            "status": self.status.value,
            "confidence": self.confidence.value,
            "notes": self.notes,
            "anomalies": self.anomalies,
            "correction_data": self.correction_data,
            "timestamp": self.timestamp,
            "entry_hash": self.entry_hash,
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> VerificationLineageEntry:
        """Reconstruct from JSON dict."""
        return VerificationLineageEntry(
            dl2_id=data.get("dl2_id"),
            dl2_data=data.get("dl2_data", {}),
            dl1_id=data.get("dl1_id"),
            verifier_principal=data.get("verifier_principal"),
            status=VerificationStatus(data.get("status", "pending")),
            confidence=VerificationConfidence(data.get("confidence", "unsure")),
            notes=data.get("notes", ""),
            anomalies=data.get("anomalies", []),
            correction_data=data.get("correction_data", {}),
        )


class VerificationLog:
    """Manages verification audit trail (JSONL format).
    
    Immutable append-only log stored in Google Drive or local filesystem.
    Each entry represents a human verification decision.
    """
    
    def __init__(self, log_path: Path | str):
        """Initialize verification log.
        
        Args:
            log_path: Path to JSONL file (e.g., gs://bucket/verification_log.jsonl)
        """
        self.log_path = Path(log_path) if isinstance(log_path, str) else log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def append(self, entry: VerificationLineageEntry) -> bool:
        """Append verification entry to log (atomic append-only).
        
        Args:
            entry: VerificationLineageEntry to record
        
        Returns:
            True if successful, False otherwise
        """
        try:
            import orjson
            with open(self.log_path, "ab") as f:
                f.write(orjson.dumps(entry.to_dict()) + b"\n")
                f.flush()
            logger.info({
                "level": "INFO",
                "type": "verification",
                "message": f"Verification logged: {entry.dl2_id} → {entry.status.value}",
                "session_id": None,
                "dl2_id": entry.dl2_id,
                "status": entry.status.value,
                "verifier": entry.verifier_principal,
            })
            return True
        except Exception as e:
            logger.error({
                "level": "ERROR",
                "type": "verification",
                "message": f"Failed to append verification log: {e}",
                "session_id": None,
            })
            return False
    
    def read_all(self, limit: Optional[int] = None) -> List[VerificationLineageEntry]:
        """Read all verification entries from log.
        
        Args:
            limit: Max entries to read (None = all)
        
        Returns:
            List of VerificationLineageEntry objects
        """
        if not self.log_path.exists():
            return []
        
        entries = []
        try:
            import orjson
            with open(self.log_path, "rb") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        data = orjson.loads(line)
                        entries.append(VerificationLineageEntry.from_dict(data))
                        if limit and len(entries) >= limit:
                            break
                    except Exception:
                        continue
        except Exception as e:
            logger.error({
                "level": "ERROR",
                "type": "verification",
                "message": f"Failed to read verification log: {e}",
                "session_id": None,
            })
        
        return entries
    
    def get_by_dl2_id(self, dl2_id: str) -> Optional[VerificationLineageEntry]:
        """Lookup verification decision for a specific DL2 row.
        
        Args:
            dl2_id: DL2 row identifier
        
        Returns:
            VerificationLineageEntry if found, None otherwise
        """
        for entry in self.read_all():
            if entry.dl2_id == dl2_id:
                return entry
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Compute verification statistics.
        
        Returns:
            Dict with counts by status/confidence
        """
        entries = self.read_all()
        stats = {
            "total": len(entries),
            "by_status": {
                "approved": 0,
                "rejected": 0,
                "flagged": 0,
                "pending": 0,
            },
            "by_confidence": {
                "high": 0,
                "medium": 0,
                "low": 0,
                "unsure": 0,
            },
            "by_anomaly_type": {},
        }
        
        for entry in entries:
            stats["by_status"][entry.status.value] += 1
            stats["by_confidence"][entry.confidence.value] += 1
            for anom in entry.anomalies:
                atype = anom.get("type", "other")
                stats["by_anomaly_type"][atype] = stats["by_anomaly_type"].get(atype, 0) + 1
        
        return stats


def classify_anomaly(
    dl2_value: Any,
    dl1_value: Any,
    field_name: str = "unknown",
) -> Tuple[bool, Optional[AnomalyType], str]:
    """Classify difference between DL2 and DL1 as unintentional mistake.
    
    Args:
        dl2_value: Extracted (AI) value
        dl1_value: Verified (human) value
        field_name: Name of field for context
    
    Returns:
        (is_anomaly, anomaly_type, description)
    """
    if dl2_value == dl1_value:
        return False, None, "No difference"
    
    str_dl2 = str(dl2_value).strip()
    str_dl1 = str(dl1_value).strip()
    
    # Case-insensitive match (formatting)
    if str_dl2.lower() == str_dl1.lower():
        return True, AnomalyType.DATA_FORMATTING, f"Case mismatch: '{dl2_value}' vs '{dl1_value}'"
    
    # Missing field in DL2
    if not str_dl2 or str_dl2 in ("", "None", "null"):
        return True, AnomalyType.MISSING_FIELD, f"DL2 missing value for {field_name}"
    
    # Try numeric comparison
    try:
        f_dl2 = float(dl2_value)
        f_dl1 = float(dl1_value)
        if abs(f_dl2 - f_dl1) < 0.01:  # Small rounding difference
            return True, AnomalyType.NUMERIC_PRECISION, f"Rounding: {dl2_value} vs {dl1_value}"
    except (ValueError, TypeError):
        pass
    
    # Encoding/special character issue
    if len(str_dl2) == len(str_dl1):
        # Same length but different content → likely encoding
        return True, AnomalyType.ENCODING_ISSUE, f"Possible encoding issue: '{dl2_value}' vs '{dl1_value}'"
    
    # Generic extraction error
    return True, AnomalyType.EXTRACTION_ERROR, f"Mismatch for {field_name}: '{dl2_value}' vs '{dl1_value}'"


__all__ = [
    "VerificationStatus",
    "VerificationConfidence",
    "AnomalyType",
    "VerificationLineageEntry",
    "VerificationLog",
    "classify_anomaly",
]
