"""
risk_gates.py

Three-dimensional risk assessment with multi-gate suspicion scoring.
Replaces single-score thresholds with proportional tri-partitioned model:
  - Confidence Gate (extraction conviction)
  - Verification Gate (ground truth alignment)
  - Anomaly Gate (statistical suspension)

Combined via weighted vector to produce composite_suspicion ∈ [0, 1],
then classified into block/warn/log tiers (⅓-proportioned boundaries).

Architecture:
  Dimension 1: confidence_gate = extraction_confidence (0→1 scale, 1=certain)
  Dimension 2: verification_gate = ground_truth_match_ratio (0→1 scale, 1=perfect match)
  Dimension 3: anomaly_gate = suspicious_score (0→1 scale, 0=clean, 1=highly suspicious)

Composite Suspicion (inverse reasoning):
  suspicion = w₁(1 - confidence) + w₂(1 - verification) + w₃(anomaly)
  where w₁ + w₂ + w₃ = 1.0 (default: 0.33 each)

Risk Tier Classification (⅓-based boundaries):
  BLOCK:  suspicion >= 0.72  (upper third → refuse/escalate)
  WARN:   0.45 ≤ suspicion < 0.72  (middle third → confirm/verify)
  LOG:    suspicion < 0.45  (lower third → automatic/audit-only)

The ⅓ partitioning ensures:
  - Clear separation between tiers
  - Proportional "weight" to each dimension's contribution
  - Data clusters that emerge naturally from multi-gate interactions
"""

from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class RiskGateScores:
    """Container for three-dimensional risk assessment."""
    confidence_gate: float  # ∈ [0, 1], 1=certain
    verification_gate: float  # ∈ [0, 1], 1=perfect match
    anomaly_gate: float  # ∈ [0, 1], 0=clean, 1=highly suspicious
    composite_suspicion: float  # ∈ [0, 1], final score
    risk_tier: str  # "block", "warn", or "log"
    tier_confidence: float  # How close to nearest boundary


@dataclass
class RiskGateConfig:
    """Configuration for three-dimensional risk model."""
    
    # Weights for the three gates (must sum to 1.0)
    weight_confidence: float = 0.33
    weight_verification: float = 0.33
    weight_anomaly: float = 0.34
    
    # Tier boundaries (⅓-partitioned)
    tier_boundary_warn_log: float = 0.45  # suspicion < this → log
    tier_boundary_block_warn: float = 0.72  # suspicion >= this → block
    
    # Sub-component thresholds for gate computation
    verification_match_threshold: float = 0.8  # 80% match = full verification
    anomaly_pattern_weight: float = 0.4  # How much suspicious patterns affect anomaly gate
    anomaly_outlier_weight: float = 0.6  # How much statistical outliers affect anomaly gate


class RiskGateEvaluator:
    """Multi-dimensional risk assessment engine."""
    
    def __init__(self, config: Optional[RiskGateConfig] = None):
        """Initialize with optional custom configuration."""
        self.config = config or RiskGateConfig()
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Ensure config weights sum to 1.0 and boundaries are valid."""
        weight_sum = (
            self.config.weight_confidence +
            self.config.weight_verification +
            self.config.weight_anomaly
        )
        if not abs(weight_sum - 1.0) < 0.001:
            raise ValueError(
                f"Weights must sum to 1.0, got {weight_sum:.3f}"
            )
        if not (0.0 <= self.config.tier_boundary_warn_log <= 1.0):
            raise ValueError("tier_boundary_warn_log must be ∈ [0, 1]")
        if not (0.0 <= self.config.tier_boundary_block_warn <= 1.0):
            raise ValueError("tier_boundary_block_warn must be ∈ [0, 1]")
        if self.config.tier_boundary_wan_log >= self.config.tier_boundary_block_wan:
            raise ValueError(
                "tier_boundary_warn_log must be < tier_boundary_block_warn"
            )
    
    # =========================================================================
    # GATE COMPUTATION: Three independent risk dimensions
    # =========================================================================
    
    def compute_confidence_gate(
        self,
        extraction_confidence: float
    ) -> float:
        """
        Confidence Gate: How certain is parser extraction?
        
        Input: extraction_confidence ∈ [0, 1]
          1.0 = absolute certainty
          0.5 = 50/50 guess
          0.0 = complete garbage
        
        Output: confidence_gate ∈ [0, 1]
          Used in suspicion = w₁(1 - confidence_gate) + ...
          So low confidence → high suspicion contribution
        
        Args:
            extraction_confidence: Raw parser confidence score
        
        Returns:
            Normalized confidence gate ∈ [0, 1]
        """
        # Clamp to valid range
        conf = max(0.0, min(1.0, extraction_confidence))
        
        # Apply gentle sigmoid smoothing to avoid cliff at boundaries
        # f(x) = x (no transformation; linear pass-through)
        # Could be enhanced with logistic sigmoid if desired
        return conf
    
    def compute_verification_gate(
        self,
        ground_truth_matches: int,
        total_records: int,
        fallback_verification_score: Optional[float] = None
    ) -> float:
        """
        Verification Gate: How well does data align with ground truth (DL1)?
        
        Input: Counts of matched records vs. total
          ground_truth_matches = number of rows matching DL1
          total_records = total rows extracted/imported
        
        Output: verification_gate ∈ [0, 1]
          1.0 = perfect match with DL1
          0.5 = 50% of data verified
          0.0 = no verified matches
        
        If DL1 data unavailable, use fallback_verification_score for estimation.
        
        Args:
            ground_truth_matches: Count of rows matching verified DL1 data
            total_records: Total count of records imported/extracted
            fallback_verification_score: Optional estimation if DL1 unavailable (0–1)
        
        Returns:
            Verification gate ∈ [0, 1]
        """
        if total_records <= 0 and fallback_verification_score is not None:
            # No data available; use fallback
            return max(0.0, min(1.0, fallback_verification_score))
        
        if total_records <= 0:
            # No data and no fallback; assume unverified
            return 0.0
        
        # Match ratio
        match_ratio = ground_truth_matches / total_records
        
        # Clamp and return
        return max(0.0, min(1.0, match_ratio))
    
    def compute_anomaly_gate(
        self,
        suspicious_pattern_count: int,
        outlier_record_count: int,
        total_records: int,
        integrity_flags: Optional[List[str]] = None
    ) -> float:
        """
        Anomaly Gate: Statistical suspicion from patterns and outliers.
        
        Input: Counts of suspicious patterns/outliers relative to data volume
        Output: anomaly_gate ∈ [0, 1]
          0.0 = completely clean data
          0.5 = moderate anomalies (10% of records are outliers)
          1.0 = highly suspicious (>30% of records flagged)
        
        Args:
            suspicious_pattern_count: Count of rows matching patterns (test, demo, fake, etc.)
            outlier_record_count: Count of statistical outliers (e.g., votes > population)
            total_records: Total records for context
            integrity_flags: Optional list of custom integrity violation strings
        
        Returns:
            Anomaly gate ∈ [0, 1]
        """
        if total_records <= 0:
            # No data; assume default low anomaly
            return 0.0
        
        # Compute pattern anomaly ratio
        pattern_ratio = (
            suspicious_pattern_count / total_records
            if total_records > 0
            else 0.0
        )
        
        # Compute outlier anomaly ratio
        outlier_ratio = (
            outlier_record_count / total_records
            if total_records > 0
            else 0.0
        )
        
        # Blend: patterns weight 40%, outliers 60% (configurable)
        blended_anomaly = (
            self.config.anomaly_pattern_weight * pattern_ratio +
            self.config.anomaly_outlier_weight * outlier_ratio
        )
        
        # Boost for integrity flags (each flag adds 0.1 up to max)
        if integrity_flags:
            flag_boost = min(0.3, len(integrity_flags) * 0.1)
            blended_anomaly += flag_boost
        
        # Clamp and return
        return max(0.0, min(1.0, blended_anomaly))
    
    # =========================================================================
    # COMPOSITE SCORE: Weighted vector of three gates
    # =========================================================================
    
    def compute_composite_suspicion(
        self,
        confidence_gate: float,
        verification_gate: float,
        anomaly_gate: float
    ) -> float:
        """
        Composite Suspicion: Weighted combination of three risk dimensions.
        
        Formula:
          suspicion = w₁(1 - confidence) + w₂(1 - verification) + w₃(anomaly)
          where w₁ + w₂ + w₃ = 1.0
        
        Reasoning:
          - Low confidence → high suspicion
          - Low verification → high suspicion
          - High anomaly → high suspicion
        
        Args:
            confidence_gate: ∈ [0, 1], 1=certain → use as (1 - confidence)
            verification_gate: ∈ [0, 1], 1=perfect → use as (1 - verification)
            anomaly_gate: ∈ [0, 1], 0=clean → use as-is
        
        Returns:
            Composite suspicion score ∈ [0, 1]
        """
        suspicion = (
            self.config.weight_confidence * (1.0 - confidence_gate) +
            self.config.weight_verification * (1.0 - verification_gate) +
            self.config.weight_anomaly * anomaly_gate
        )
        return max(0.0, min(1.0, suspicion))
    
    def classify_risk_tier(
        self,
        composite_suspicion: float
    ) -> Tuple[str, float]:
        """
        Classify suspicion into risk tier using ⅓-proportioned boundaries.
        
        Tier Distribution:
          BLOCK:  suspicion >= 0.72  (top ⅓, 72–100%)
          WARN:   0.45 ≤ suspicion < 0.72  (middle ⅓, 45–72%)
          LOG:    suspicion < 0.45  (bottom ⅓, 0–45%)
        
        The "third more/less" principle ensures:
          - WARN tier (~27% width) is middle third
          - Each dimension can be independently tuned to move data between tiers
          - Clusters emerge naturally from interactions
        
        Args:
            composite_suspicion: ∈ [0, 1]
        
        Returns:
            Tuple[tier: str, tier_confidence: float]
            tier: "block" | "warn" | "log"
            tier_confidence: Distance from nearest boundary (0=on boundary, 1=deep in tier)
        """
        boundary_warn_log = self.config.tier_boundary_warn_log
        boundary_block_warn = self.config.tier_boundary_block_warn
        
        if composite_suspicion >= boundary_block_warn:
            # BLOCK tier
            # Confidence: how far above BLOCK threshold (0–1 scale)
            blocks_width = 1.0 - boundary_block_warn
            if blocks_width > 0:
                confidence = (
                    (composite_suspicion - boundary_block_warn) / blocks_width
                )
            else:
                confidence = 1.0
            return ("block", confidence)
        
        elif composite_suspicion >= boundary_warn_log:
            # WARN tier
            # Confidence: position within tier (0=near log boundary, 1=near block boundary)
            warn_width = boundary_block_warn - boundary_warn_log
            if warn_width > 0:
                confidence = (
                    (composite_suspicion - boundary_warn_log) / warn_width
                )
            else:
                confidence = 0.5
            return ("warn", confidence)
        
        else:
            # LOG tier
            # Confidence: how far below WARN threshold
            if boundary_warn_log > 0:
                confidence = 1.0 - (composite_suspicion / boundary_warn_log)
            else:
                confidence = 1.0
            return ("log", confidence)
    
    # =========================================================================
    # UNIFIED EVALUATION: All gates + classification in one call
    # =========================================================================
    
    def evaluate(
        self,
        extraction_confidence: float,
        ground_truth_matches: int,
        total_records: int,
        suspicious_pattern_count: int = 0,
        outlier_record_count: int = 0,
        integrity_flags: Optional[List[str]] = None,
        fallback_verification_score: Optional[float] = None
    ) -> RiskGateScores:
        """
        Complete risk evaluation: compute three gates + composite + tier.
        
        Args:
            extraction_confidence: Parser confidence (0–1)
            ground_truth_matches: Count of DL1-verified matches
            total_records: Total records imported/extracted
            suspicious_pattern_count: Count of suspicious pattern matches
            outlier_record_count: Count of statistical outliers
            integrity_flags: Optional list of custom integrity violations
            fallback_verification_score: Fallback if DL1 unavailable (0–1)
        
        Returns:
            RiskGateScores with gates, composite suspicion, and risk tier
        """
        # Compute three gates
        conf_gate = self.compute_confidence_gate(extraction_confidence)
        verif_gate = self.compute_verification_gate(
            ground_truth_matches,
            total_records,
            fallback_verification_score
        )
        anom_gate = self.compute_anomaly_gate(
            suspicious_pattern_count,
            outlier_record_count,
            total_records,
            integrity_flags
        )
        
        # Composite suspicion
        composite = self.compute_composite_suspicion(conf_gate, verif_gate, anom_gate)
        
        # Tier classification
        tier, tier_confidence = self.classify_risk_tier(composite)
        
        return RiskGateScores(
            confidence_gate=conf_gate,
            verification_gate=verif_gate,
            anomaly_gate=anom_gate,
            composite_suspicion=composite,
            risk_tier=tier,
            tier_confidence=tier_confidence
        )


# === Module-level convenience functions ===

_default_evaluator = RiskGateEvaluator()


def evaluate_risk(
    extraction_confidence: float,
    ground_truth_matches: int,
    total_records: int,
    **kwargs
) -> RiskGateScores:
    """
    Quick risk evaluation using default configuration.
    
    Args:
        extraction_confidence: Parser confidence (0–1)
        ground_truth_matches: Count of DL1-verified matches
        total_records: Total records imported/extracted
        **kwargs: Additional arguments passed to evaluator.evaluate()
    
    Returns:
        RiskGateScores
    """
    return _default_evaluator.evaluate(
        extraction_confidence,
        ground_truth_matches,
        total_records,
        **kwargs
    )
