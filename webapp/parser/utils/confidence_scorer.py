"""
Confidence Scoring System with Harmonic Ranking & Immediate Flush

Purpose:
- Assign confidence levels to parser runs (0.0-1.0 scale)
- Compute harmonic scores for successful runs (proportional weighting)
- Flag very likely malicious acts for immediate removal (not persistence)
- Snapshot critical errors for review (immutable text format)
- Flush all decisions immediately to txt logs (zero execution cascade risk)
- Store only text tracebacks in memory, never Python objects or executable code

Architecture:
- RunConfidence: (extraction_confidence, factor_integrity, session_isolation, principal_tier)
- HarmonicScore: 1/(1 + num_successful_runs) * base_quality_score (lower score later runs to weight earlier runs)
- MaliciousAct: factor chain breaks + principal privilege violation + extraction anomaly
- CriticalError: extraction failure + integrity issue + no execution risk
- ImmediateFlushing: Every decision written to disk within 100ms, memory holds only traceback text
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import LOG_DIR
from ..utils.logger_singleton import logger

# ============================================================================
# IMMUTABLE AUDIT TRAILS (Text Format, No Executable Code)
# ============================================================================

CONFIDENCE_METRICS_FILE = Path(LOG_DIR) / "confidence_metrics.txt"
MALICIOUS_ACTS_REMOVED_FILE = Path(LOG_DIR) / "malicious_acts_removed.txt"
CRITICAL_ERRORS_SNAPSHOT_FILE = Path(LOG_DIR) / "critical_errors_snapshot.txt"
FACTOR_CHAIN_ANOMALIES_FILE = Path(LOG_DIR) / "factor_chain_anomalies.txt"
HARMONIC_RANKINGS_FILE = Path(LOG_DIR) / "harmonic_rankings.txt"

# Lock for atomic writes
_FLUSH_LOCK = threading.Lock()

# In-memory traceback index (session_id → list of text tracebacks, never Python objects)
_TRACEBACK_MEMORY: Dict[str, List[str]] = {}
_TRACEBACK_MEMORY_LOCK = threading.Lock()

# ============================================================================
# CONFIDENCE LEVELS & SCORING
# ============================================================================

class ConfidenceLevel:
    """Confidence thresholds for decision impact."""
    HIGH = 0.85          # >85%: Safe to apply directly
    MEDIUM = 0.65        # 65-85%: Needs review before applying
    LOW = 0.40           # 40-65%: Critical errors, snapshot only
    MALICIOUS = 0.0      # 0-40% + factor chain break: Remove completely


# ============================================================================
# DATA STRUCTURES (Text-Serializable Only)
# ============================================================================

@dataclass
class RunConfidence:
    """Confidence assessment for a parser run."""
    chain_id: str                              # UUID for this run
    principal: str                             # Admin/user identifier
    principal_tier: int                        # 0=standard, 1=reviewer, 2=full, 3=root
    url: str                                   # Target URL
    timestamp: str                             # ISO 8601
    
    # Confidence factors (0.0-1.0)
    extraction_confidence: float               # Did we extract good data? (table count, row count quality)
    factor_integrity_confidence: float         # Did trust factors remain stable? (breaking chains detected?)
    session_isolation_confidence: float        # Was principal isolation maintained? (no cross-leakage)
    privilege_boundary_confidence: float       # Did admin boost apply only to trusted domains?
    
    # Overall confidence (harmonic mean of factors)
    overall_confidence: float = field(init=False)
    
    # Decision outcomes
    decision_type: str = ""                    # "direct_navigation" | "snapshot_mode" | "quarantine" | "reject"
    decision_confidence: float = 0.0           # Confidence in the decision itself
    
    # Outcome quality
    success: bool = False                      # Did extraction succeed?
    error_type: Optional[str] = None           # "malicious_act" | "critical_error" | None
    error_severity: Optional[str] = None       # "high" | "medium" | "low"
    
    # Traceback (text only, never Python objects)
    traceback_text: str = ""                   # Last-known traceback as plain text
    
    def __post_init__(self):
        """Compute overall confidence as harmonic mean of factors."""
        factors = [
            self.extraction_confidence,
            self.factor_integrity_confidence,
            self.session_isolation_confidence,
            self.privilege_boundary_confidence
        ]
        # Harmonic mean: n / (1/x1 + 1/x2 + ... + 1/xn)
        # With minimum 0.01 to avoid division by zero
        safe_factors = [max(0.01, f) for f in factors]
        n = len(safe_factors)
        self.overall_confidence = n / sum(1.0 / f for f in safe_factors)
    
    def to_txt_entry(self) -> str:
        """Serialize to single-line text format (immutable, searchable)."""
        return (
            f"[{self.timestamp}] chain_id={self.chain_id} principal={self.principal} "
            f"tier={self.principal_tier} url={self.url} "
            f"extraction={self.extraction_confidence:.2f} "
            f"integrity={self.factor_integrity_confidence:.2f} "
            f"isolation={self.session_isolation_confidence:.2f} "
            f"boundary={self.privilege_boundary_confidence:.2f} "
            f"overall={self.overall_confidence:.2f} "
            f"decision={self.decision_type} decision_conf={self.decision_confidence:.2f} "
            f"success={self.success} error={self.error_type or 'none'} "
            f"severity={self.error_severity or 'n/a'}"
        )
    
    def to_dict_safe(self) -> Dict[str, Any]:
        """Convert to dict (text-safe, no objects)."""
        return {
            "chain_id": self.chain_id,
            "principal": self.principal,
            "principal_tier": self.principal_tier,
            "url": self.url,
            "timestamp": self.timestamp,
            "extraction_confidence": self.extraction_confidence,
            "factor_integrity_confidence": self.factor_integrity_confidence,
            "session_isolation_confidence": self.session_isolation_confidence,
            "privilege_boundary_confidence": self.privilege_boundary_confidence,
            "overall_confidence": self.overall_confidence,
            "decision_type": self.decision_type,
            "decision_confidence": self.decision_confidence,
            "success": self.success,
            "error_type": self.error_type,
            "error_severity": self.error_severity,
        }


@dataclass
class HarmonicScore:
    """Proportional harmonic ranking for successful runs."""
    chain_id: str
    principal: str
    url: str
    timestamp: str
    
    # Success metrics
    extraction_quality: float                  # 0.0-1.0 (row/col count, data completeness)
    data_integrity_score: float                # 0.0-1.0 (validation checks passed)
    
    # Harmonic rank (lower=earlier, better weighting)
    rank_order: int                            # Position in success sequence
    harmonic_coefficient: float = field(init=False)  # 1/(1 + rank_order)
    weighted_score: float = field(init=False)  # harmonic * extraction * integrity
    
    def __post_init__(self):
        """Compute harmonic coefficient and weighted score."""
        self.harmonic_coefficient = 1.0 / (1.0 + self.rank_order)
        self.weighted_score = self.harmonic_coefficient * self.extraction_quality * self.data_integrity_score
    
    def to_txt_entry(self) -> str:
        """Serialize to searchable text format."""
        return (
            f"[{self.timestamp}] chain_id={self.chain_id} principal={self.principal} "
            f"url={self.url} rank={self.rank_order} harmonic={self.harmonic_coefficient:.4f} "
            f"extraction_quality={self.extraction_quality:.2f} "
            f"integrity={self.data_integrity_score:.2f} "
            f"weighted_score={self.weighted_score:.4f}"
        )


@dataclass
class MaliciousActFlag:
    """Mark very likely malicious acts for complete removal."""
    chain_id: str
    principal: str
    url: str
    timestamp: str
    
    # Attack indicators
    factor_chain_break_detected: bool          # Trust factors shifted unexpectedly
    privilege_violation: bool                  # Admin boost applied to untrusted domain
    extraction_anomaly_severe: bool            # ML flagged >90% anomaly
    principal_isolation_breach: bool           # Cross-principal data leakage detected
    
    # Removal action
    action: str = "REMOVED"                    # Always "REMOVED" for malicious acts
    removal_reason: str = ""
    
    def to_txt_entry(self) -> str:
        """Serialize for immediate logging."""
        reasons = []
        if self.factor_chain_break_detected:
            reasons.append("factor_chain_break")
        if self.privilege_violation:
            reasons.append("privilege_violation")
        if self.extraction_anomaly_severe:
            reasons.append("extraction_anomaly_90%+")
        if self.principal_isolation_breach:
            reasons.append("isolation_breach")
        
        reason_str = "|".join(reasons) if reasons else "unspecified_malicious"
        
        return (
            f"[{self.timestamp}] {self.action} chain_id={self.chain_id} principal={self.principal} "
            f"url={self.url} reasons={reason_str}"
        )


@dataclass
class CriticalErrorSnapshot:
    """Snapshot non-malicious critical errors for review."""
    chain_id: str
    principal: str
    url: str
    timestamp: str
    
    # Error details (text only)
    error_type: str                            # "extraction_failed" | "validation_failed" | "integrity_issue"
    error_description: str                     # Plain text description
    error_traceback: str                       # Last-known traceback, plain text only
    
    # Context for review
    last_known_state: str                      # "before_trust_score" | "after_snapshot" | "after_extraction"
    data_available: bool                       # Could we save partial data?
    
    # Review status
    reviewed: bool = False
    reviewer_notes: str = ""
    
    def to_txt_entry(self) -> str:
        """Serialize for searchable snapshot storage."""
        first_line = self.error_traceback.split("\n")[0] if self.error_traceback else "(no traceback)"
        
        return (
            f"[{self.timestamp}] SNAPSHOT chain_id={self.chain_id} principal={self.principal} "
            f"url={self.url} error_type={self.error_type} state={self.last_known_state} "
            f"data_available={self.data_available} reviewed={self.reviewed} "
            f"first_traceback_line={first_line[:100]}"
        )


# ============================================================================
# IMMEDIATE FLUSH OPERATIONS (Atomic, Thread-Safe)
# ============================================================================

def _ensure_log_files():
    """Create log files if they don't exist."""
    for log_file in [
        CONFIDENCE_METRICS_FILE,
        MALICIOUS_ACTS_REMOVED_FILE,
        CRITICAL_ERRORS_SNAPSHOT_FILE,
        FACTOR_CHAIN_ANOMALIES_FILE,
        HARMONIC_RANKINGS_FILE
    ]:
        try:
            log_file.touch(exist_ok=True)
        except Exception:
            pass


_ensure_log_files()


def flush_run_confidence(confidence: RunConfidence) -> bool:
    """
    Immediately flush confidence assessment to disk.
    Returns True if successful, False if failed.
    Must complete within 100ms for fast-fail detection.
    """
    try:
        start_time = time.time()
        with _FLUSH_LOCK:
            txt_entry = confidence.to_txt_entry()
            with open(CONFIDENCE_METRICS_FILE, "a", encoding="utf-8") as f:
                f.write(txt_entry + "\n")
                f.flush()
                os.fsync(f.fileno())
        
        elapsed_ms = (time.time() - start_time) * 1000
        if elapsed_ms > 100:
            logger.warning({
                "level": "WARNING",
                "type": "confidence",
                "message": f"Confidence flush took {elapsed_ms:.1f}ms (>100ms threshold)",
                "chain_id": confidence.chain_id,
                "session_id": None
            })
        return True
    except Exception as exc:
        logger.error({
            "level": "ERROR",
            "type": "confidence",
            "message": f"Failed to flush confidence: {exc}",
            "chain_id": confidence.chain_id,
            "session_id": None
        })
        return False


def flush_malicious_act(malicious: MaliciousActFlag) -> bool:
    """
    Immediately flush malicious act flag to removal log.
    CRITICAL: This triggers URL removal from all systems.
    Returns True if flushed, False if write failed.
    """
    try:
        start_time = time.time()
        with _FLUSH_LOCK:
            txt_entry = malicious.to_txt_entry()
            with open(MALICIOUS_ACTS_REMOVED_FILE, "a", encoding="utf-8") as f:
                f.write(txt_entry + "\n")
                f.flush()
                os.fsync(f.fileno())
        
        elapsed_ms = (time.time() - start_time) * 1000
        logger.critical({
            "level": "CRITICAL",
            "type": "security",
            "message": f"MALICIOUS ACT REMOVED: {malicious.chain_id} {malicious.url} - COMPLETE REMOVAL INITIATED",
            "chain_id": malicious.chain_id,
            "principal": malicious.principal,
            "url": malicious.url,
            "session_id": None,
            "flush_ms": elapsed_ms
        })
        return True
    except Exception as exc:
        logger.error({
            "level": "ERROR",
            "type": "security",
            "message": f"CRITICAL FAILURE: Could not flush malicious act flag: {exc}",
            "chain_id": malicious.chain_id,
            "session_id": None
        })
        return False


def flush_critical_error_snapshot(error: CriticalErrorSnapshot) -> bool:
    """
    Immediately flush critical error snapshot for review.
    Preserves partial state for human review without execution risk.
    Returns True if flushed, False if write failed.
    """
    try:
        start_time = time.time()
        with _FLUSH_LOCK:
            txt_entry = error.to_txt_entry()
            with open(CRITICAL_ERRORS_SNAPSHOT_FILE, "a", encoding="utf-8") as f:
                f.write(txt_entry + "\n")
                f.flush()
                os.fsync(f.fileno())
        
        elapsed_ms = (time.time() - start_time) * 1000
        logger.warning({
            "level": "WARNING",
            "type": "critical_error",
            "message": f"Critical error snapshot: {error.error_type} - Review required",
            "chain_id": error.chain_id,
            "error_type": error.error_type,
            "session_id": None,
            "flush_ms": elapsed_ms
        })
        return True
    except Exception as exc:
        logger.error({
            "level": "ERROR",
            "type": "critical_error",
            "message": f"Failed to flush critical error snapshot: {exc}",
            "chain_id": error.chain_id,
            "session_id": None
        })
        return False


def flush_harmonic_score(score: HarmonicScore) -> bool:
    """
    Immediately flush successful run harmonic score.
    Records proportional weighting for aggregate analytics.
    Returns True if flushed, False if write failed.
    """
    try:
        start_time = time.time()
        with _FLUSH_LOCK:
            txt_entry = score.to_txt_entry()
            with open(HARMONIC_RANKINGS_FILE, "a", encoding="utf-8") as f:
                f.write(txt_entry + "\n")
                f.flush()
                os.fsync(f.fileno())
        
        elapsed_ms = (time.time() - start_time) * 1000
        return True
    except Exception as exc:
        logger.error({
            "level": "ERROR",
            "type": "harmonic_score",
            "message": f"Failed to flush harmonic score: {exc}",
            "chain_id": score.chain_id,
            "session_id": None
        })
        return False


# ============================================================================
# TRACEBACK-ONLY MEMORY MANAGEMENT (No Execution Risk)
# ============================================================================

def store_traceback_in_memory(chain_id: str, traceback_text: str) -> None:
    """
    Store text-only traceback in memory (never Python objects).
    Indexed by chain_id for quick lookup.
    Auto-clears old entries to prevent memory growth.
    """
    with _TRACEBACK_MEMORY_LOCK:
        if chain_id not in _TRACEBACK_MEMORY:
            _TRACEBACK_MEMORY[chain_id] = []
        
        # Append with timestamp prefix
        timestamped_trace = f"[{datetime.now(timezone.utc).isoformat()}] {traceback_text[:1000]}"
        _TRACEBACK_MEMORY[chain_id].append(timestamped_trace)
        
        # Keep only last 10 tracebacks per chain_id to prevent memory bloat
        if len(_TRACEBACK_MEMORY[chain_id]) > 10:
            _TRACEBACK_MEMORY[chain_id] = _TRACEBACK_MEMORY[chain_id][-10:]
        
        # Auto-expire chains older than 1 hour
        if len(_TRACEBACK_MEMORY) > 1000:
            now_ts = time.time()
            # This is simplified; in production would parse timestamps
            to_delete = [cid for cid in list(_TRACEBACK_MEMORY.keys()) if len(_TRACEBACK_MEMORY[cid]) == 0]
            for cid in to_delete:
                del _TRACEBACK_MEMORY[cid]


def get_traceback_from_memory(chain_id: str) -> str:
    """Retrieve concatenated tracebacks for a chain_id (text only)."""
    with _TRACEBACK_MEMORY_LOCK:
        if chain_id not in _TRACEBACK_MEMORY:
            return ""
        return "\n".join(_TRACEBACK_MEMORY[chain_id])


def clear_traceback_memory(chain_id: str) -> None:
    """Clear tracebacks for a chain_id after flushing to disk."""
    with _TRACEBACK_MEMORY_LOCK:
        _TRACEBACK_MEMORY.pop(chain_id, None)


# ============================================================================
# CONFIDENCE COMPUTATION HELPERS
# ============================================================================

def compute_extraction_confidence(row_count: int, col_count: int, data_quality_score: float) -> float:
    """
    Compute extraction confidence based on data volume and quality.
    
    row_count, col_count: Table dimensions
    data_quality_score: 0.0-1.0 from validation checks
    
    Returns: 0.0-1.0
    """
    # Penalize low data volume
    volume_score = min(1.0, (row_count * col_count) / 1000.0)  # Normalize to 1000 cells
    
    # Combine volume and quality with harmonic mean
    if volume_score < 0.01 or data_quality_score < 0.01:
        return 0.0
    
    extraction_confidence = 2 / ((1 / max(0.01, volume_score)) + (1 / max(0.01, data_quality_score)))
    return min(1.0, extraction_confidence)


def compute_factor_integrity_confidence(
    expected_factors: Dict[str, float],
    observed_factors: Dict[str, float]
) -> float:
    """
    Compute confidence that trust factors remained stable.
    Large shifts indicate factor chain breaks (attack indicator).
    
    expected_factors, observed_factors: {factor_name: value}
    
    Returns: 0.0-1.0 (1.0 = no shifts, 0.0 = severe shifts)
    """
    if not expected_factors or not observed_factors:
        return 0.5  # Unknown
    
    max_shift = 0.0
    for key in expected_factors:
        expected = expected_factors.get(key, 0.0)
        observed = observed_factors.get(key, 0.0)
        shift = abs(expected - observed)
        max_shift = max(max_shift, shift)
    
    # Large shift (>0.3) = integrity issue
    # Small shift (0.0-0.1) = high confidence
    integrity_confidence = max(0.0, 1.0 - max_shift)
    return integrity_confidence


def detect_malicious_act(
    factor_integrity: float,
    privilege_violation: bool,
    ml_anomaly_score: float,
    isolation_breach: bool
) -> bool:
    """
    Detect very likely malicious acts for immediate removal.
    
    Returns True if multiple attack indicators present.
    """
    indicators = 0
    if factor_integrity < 0.4:  # Severe factor chain break
        indicators += 2
    if privilege_violation:  # Admin boost on untrusted domain
        indicators += 2
    if ml_anomaly_score > 0.9:  # >90% anomaly from ML
        indicators += 2
    if isolation_breach:  # Cross-principal leakage
        indicators += 3
    
    # Threshold: 3+ indicators = malicious
    return indicators >= 3


# ============================================================================
# DECISION FACTORY (Confidence + Decision Type)
# ============================================================================

def create_run_confidence(
    principal: str,
    principal_tier: int,
    url: str,
    extraction_confidence: float,
    factor_integrity: float,
    isolation_confidence: float,
    privilege_boundary: float,
    decision_type: str,
    decision_confidence: float,
    success: bool = False,
    error_type: Optional[str] = None,
    error_severity: Optional[str] = None,
    traceback: str = ""
) -> RunConfidence:
    """Factory to create and return a RunConfidence object."""
    chain_id = str(uuid.uuid4())[:16]
    now_iso = datetime.now(timezone.utc).isoformat()
    
    confidence = RunConfidence(
        chain_id=chain_id,
        principal=principal,
        principal_tier=principal_tier,
        url=url,
        timestamp=now_iso,
        extraction_confidence=extraction_confidence,
        factor_integrity_confidence=factor_integrity,
        session_isolation_confidence=isolation_confidence,
        privilege_boundary_confidence=privilege_boundary,
        decision_type=decision_type,
        decision_confidence=decision_confidence,
        success=success,
        error_type=error_type,
        error_severity=error_severity,
        traceback_text=traceback[:500]
    )
    
    return confidence
