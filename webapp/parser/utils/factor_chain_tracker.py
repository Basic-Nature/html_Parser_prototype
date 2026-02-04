"""
Deterministic Factor Chain Tracker with Breaking-Chain Detection

Purpose:
- Monitor trust factor evolution during analysis (start → decision → finalize)
- Detect breaking chains (trust factors shifting unexpectedly = attack indicator)
- Track factor monotonicity, SSL validity toggles, dependency preservation
- Validate factor dependencies (e.g., gov_domain can't imply verified_domain alone)
- Output immutable text traceback for forensic investigation
- Integrate with confidence scorer to flag malicious acts

Breaking Chain Indicators (Attack Detection):
1. Factor monotonicity violation: verified_domain drops True→False mid-analysis
2. SSL toggle: ssl_valid changes mid-analysis (impossible under normal conditions)
3. Dependency breach: gov_domain=False but verified_domain=True (logically invalid)
4. Phishing growth only: phishing_indicators can increase (discovery) but never decrease
5. Historical success bounds: historical_success stays in 0.0-1.0 range
6. Allowlist consistency: allowlist_match stable unless URL explicitly removed from list

All detected anomalies logged to factor_chain_anomalies.txt (text only, no code).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from ..config import LOG_DIR
from ..utils.logger_singleton import logger

# ============================================================================
# IMMUTABLE ANOMALY LOG (Text Format, Forensic Grade)
# ============================================================================

FACTOR_CHAIN_ANOMALIES_FILE = Path(LOG_DIR) / "factor_chain_anomalies.txt"


def _ensure_anomaly_log():
    """Create anomaly log if it doesn't exist."""
    try:
        FACTOR_CHAIN_ANOMALIES_FILE.touch(exist_ok=True)
    except Exception:
        pass


_ensure_anomaly_log()


# ============================================================================
# FACTOR CHAIN STRUCTURE (Text-Serializable Only)
# ============================================================================

@dataclass
class FactorSnapshot:
    """Immutable snapshot of trust factors at a decision point."""
    decision_index: int                        # 0=start, 1=snapshot, 2=finalize
    decision_name: str                         # "trust_computed" | "snapshot_decided" | "finalize"
    timestamp: str                             # ISO 8601
    factors: Dict[str, float]                  # {factor_name: value}
    
    def to_txt_line(self) -> str:
        """Serialize to single text line."""
        factors_str = "|".join(f"{k}={v:.2f}" for k, v in self.factors.items())
        return f"  @{self.decision_index}[{self.decision_name}] {factors_str}"


@dataclass
class FactorChain:
    """Deterministic chain of trust factors for a URL analysis."""
    chain_id: str                              # UUID for this analysis
    principal: str                             # Who initiated
    principal_tier: int                        # 0=standard, 1=reviewer, 2=full, 3=root
    url: str                                   # Target URL
    session_id: str                            # Session context
    start_timestamp: str                       # ISO 8601
    
    # Factor evolution (locked in, append-only)
    snapshots: List[FactorSnapshot] = field(default_factory=list)  # Decision points
    
    # Anomalies detected (immutable list)
    anomalies: List[Dict[str, Any]] = field(default_factory=list)
    
    # Final assessment
    has_breaking_chain: bool = False           # One or more anomalies detected
    anomaly_severity: str = "none"             # "none" | "low" | "medium" | "high" | "critical"
    anomaly_detail: str = ""                   # Human-readable summary
    
    # Decision context
    final_decision: str = ""                   # "direct" | "snapshot" | "quarantine" | "reject"
    final_confidence: float = 0.0              # 0.0-1.0
    
    def add_snapshot(self, decision_name: str, factors: Dict[str, float]) -> None:
        """Append a factor snapshot (thread-safe immutability through append-only)."""
        index = len(self.snapshots)
        now_iso = datetime.now(timezone.utc).isoformat()
        snapshot = FactorSnapshot(
            decision_index=index,
            decision_name=decision_name,
            timestamp=now_iso,
            factors=factors.copy()  # Immutable copy
        )
        self.snapshots.append(snapshot)
    
    def add_anomaly(self, anomaly_type: str, detail: str, severity: str) -> None:
        """Record detected anomaly."""
        anomaly = {
            "type": anomaly_type,
            "detail": detail,
            "severity": severity,
            "snapshot_index": len(self.snapshots) - 1,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        self.anomalies.append(anomaly)
        self.has_breaking_chain = True
        
        # Escalate severity
        severity_order = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
        current_level = severity_order.get(self.anomaly_severity, 0)
        new_level = severity_order.get(severity, 0)
        if new_level > current_level:
            self.anomaly_severity = severity
    
    def to_txt_entry(self) -> str:
        """Serialize entire chain to single searchable text line."""
        snapshots_str = "\n".join(s.to_txt_line() for s in self.snapshots)
        anomalies_str = (
            "\n".join(f"    ! {a['type']}: {a['detail']} ({a['severity']})" for a in self.anomalies)
            if self.anomalies
            else "    (no anomalies)"
        )
        
        return (
            f"[{self.start_timestamp}] CHAIN={self.chain_id} principal={self.principal} "
            f"tier={self.principal_tier} url={self.url} breaking={self.has_breaking_chain} "
            f"severity={self.anomaly_severity}\n"
            f"SNAPSHOTS:\n{snapshots_str}\n"
            f"ANOMALIES:\n{anomalies_str}\n"
            f"FINAL: decision={self.final_decision} confidence={self.final_confidence:.2f}"
        )


# ============================================================================
# BREAKING CHAIN DETECTION
# ============================================================================

# Trust factors that should never decrease (monotonic)
MONOTONIC_FACTORS = {
    "verified_domain",              # Once established, stays true
    "phishing_indicators_count",    # Only increases (discovery)
}

# Trust factors that should never toggle (binary stable)
BINARY_STABLE_FACTORS = {
    "ssl_valid",                    # HTTPS certificate: can't toggle
    "gov_domain",                   # Domain classification: can't toggle
}

# Valid factor ranges
FACTOR_BOUNDS = {
    "historical_success": (0.0, 1.0),
    "ssl_valid_years": (0.0, 30.0),
    "domain_age_days": (0.0, 100000.0),
}

# Factor dependencies (antecedent → consequent constraint)
FACTOR_DEPENDENCIES = {
    # If gov_domain=False, verified_domain MUST also be False
    "gov_domain=False": ["verified_domain=False"],
}


def detect_breaking_chains(chain: FactorChain) -> List[Dict[str, Any]]:
    """
    Analyze factor chain for breaking chain attacks.
    Returns list of detected anomalies.
    """
    anomalies = []
    
    if len(chain.snapshots) < 2:
        return anomalies  # Need at least 2 snapshots to compare
    
    start_factors = chain.snapshots[0].factors
    
    for i in range(1, len(chain.snapshots)):
        current_factors = chain.snapshots[i].factors
        decision_name = chain.snapshots[i].decision_name
        
        # ===== CHECK 1: Monotonic Factors =====
        for factor_name in MONOTONIC_FACTORS:
            if factor_name not in start_factors or factor_name not in current_factors:
                continue
            
            start_val = start_factors.get(factor_name, 0.0)
            current_val = current_factors.get(factor_name, 0.0)
            
            if isinstance(start_val, bool):
                # Boolean monotonic: True can't become False
                if start_val and not current_val:
                    detail = f"{factor_name} dropped from True to False at decision {decision_name}"
                    anomalies.append({
                        "type": "monotonicity_violation",
                        "factor": factor_name,
                        "detail": detail,
                        "severity": "critical",
                        "evidence": f"{factor_name}: {start_val}→{current_val}"
                    })
            else:
                # Numeric monotonic: should not decrease
                if current_val < start_val:
                    detail = f"{factor_name} decreased from {start_val} to {current_val} at {decision_name}"
                    anomalies.append({
                        "type": "monotonicity_violation",
                        "factor": factor_name,
                        "detail": detail,
                        "severity": "high",
                        "evidence": f"{factor_name}: {start_val}→{current_val}"
                    })
        
        # ===== CHECK 2: Binary Stable Factors =====
        for factor_name in BINARY_STABLE_FACTORS:
            if factor_name not in start_factors or factor_name not in current_factors:
                continue
            
            start_val = start_factors.get(factor_name)
            current_val = current_factors.get(factor_name)
            
            if isinstance(start_val, bool) and isinstance(current_val, bool):
                if start_val != current_val:
                    detail = f"{factor_name} toggled from {start_val} to {current_val} at {decision_name}"
                    anomalies.append({
                        "type": "integrity_violation",
                        "factor": factor_name,
                        "detail": detail,
                        "severity": "critical",
                        "evidence": f"{factor_name}: {start_val}→{current_val}"
                    })
        
        # ===== CHECK 3: Factor Bounds =====
        for factor_name, (min_val, max_val) in FACTOR_BOUNDS.items():
            if factor_name not in current_factors:
                continue
            
            current_val = current_factors.get(factor_name, 0.0)
            if not isinstance(current_val, (int, float)):
                continue
            
            if current_val < min_val or current_val > max_val:
                detail = f"{factor_name} out of bounds [{min_val}, {max_val}]: {current_val} at {decision_name}"
                anomalies.append({
                    "type": "bounds_violation",
                    "factor": factor_name,
                    "detail": detail,
                    "severity": "high",
                    "evidence": f"{factor_name}: {current_val} (out of range)"
                })
        
        # ===== CHECK 4: Dependency Violations =====
        for antecedent, consequents in FACTOR_DEPENDENCIES.items():
            # Parse antecedent (e.g., "gov_domain=False")
            if "=" not in antecedent:
                continue
            
            factor_name, expected_str = antecedent.split("=", 1)
            expected_val = expected_str.lower() == "true" if expected_str.lower() in ("true", "false") else expected_str
            
            if factor_name not in current_factors:
                continue
            
            current_val = current_factors.get(factor_name)
            
            # If antecedent is true, check consequents
            if current_val == expected_val:
                for consequent in consequents:
                    if "=" not in consequent:
                        continue
                    cons_factor, cons_expected_str = consequent.split("=", 1)
                    cons_expected = cons_expected_str.lower() == "true" if cons_expected_str.lower() in ("true", "false") else cons_expected_str
                    
                    if cons_factor not in current_factors:
                        continue
                    
                    cons_actual = current_factors.get(cons_factor)
                    if cons_actual != cons_expected:
                        detail = (
                            f"Dependency violation: {antecedent} → {cons_factor}={cons_expected}, "
                            f"but got {cons_factor}={cons_actual} at {decision_name}"
                        )
                        anomalies.append({
                            "type": "dependency_violation",
                            "factor": f"{factor_name}→{cons_factor}",
                            "detail": detail,
                            "severity": "medium",
                            "evidence": f"{antecedent} but {cons_factor}={cons_actual}"
                        })
    
    return anomalies


# ============================================================================
# FACTOR CHAIN TRACKING & FLUSHING
# ============================================================================

def flush_factor_chain_analysis(chain: FactorChain) -> bool:
    """
    Flush factor chain analysis to forensic log.
    Called after finalization regardless of outcome.
    Returns True if successful.
    """
    try:
        txt_entry = chain.to_txt_entry()
        with open(FACTOR_CHAIN_ANOMALIES_FILE, "a", encoding="utf-8") as f:
            f.write(txt_entry + "\n\n")
            f.flush()
            import os
            os.fsync(f.fileno())
        
        if chain.has_breaking_chain:
            logger.error({
                "level": "ERROR",
                "type": "factor_chain",
                "message": f"Breaking chain detected: {chain.anomaly_detail}",
                "chain_id": chain.chain_id,
                "principal": chain.principal,
                "url": chain.url,
                "anomaly_severity": chain.anomaly_severity,
                "anomaly_count": len(chain.anomalies),
                "session_id": chain.session_id
            })
        
        return True
    except Exception as exc:
        logger.error({
            "level": "ERROR",
            "type": "factor_chain",
            "message": f"Failed to flush factor chain: {exc}",
            "chain_id": chain.chain_id,
            "session_id": chain.session_id
        })
        return False


# ============================================================================
# CHAIN CREATION & INITIALIZATION
# ============================================================================

def create_factor_chain(
    principal: str,
    principal_tier: int,
    url: str,
    session_id: str
) -> FactorChain:
    """Factory to create a new factor chain."""
    chain_id = str(uuid.uuid4())[:16]
    now_iso = datetime.now(timezone.utc).isoformat()
    
    return FactorChain(
        chain_id=chain_id,
        principal=principal,
        principal_tier=principal_tier,
        url=url,
        session_id=session_id,
        start_timestamp=now_iso
    )


def finalize_factor_chain(
    chain: FactorChain,
    final_decision: str,
    final_confidence: float
) -> FactorChain:
    """Finalize chain with decision and confidence."""
    chain.final_decision = final_decision
    chain.final_confidence = final_confidence
    
    # Generate anomaly detail if any anomalies
    if chain.anomalies:
        anomaly_types = [a.get("type", "unknown") for a in chain.anomalies]
        count = len(chain.anomalies)
        chain.anomaly_detail = f"{count} anomaly(ies): {', '.join(set(anomaly_types))}"
    
    return chain
