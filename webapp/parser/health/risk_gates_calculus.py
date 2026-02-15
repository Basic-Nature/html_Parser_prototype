"""
risk_gates_calculus.py

HIGHER-DIMENSIONAL RISK ASSESSMENT WITH DERIVATIVE GATES
Smart Elections Parser – Nine-Dimensional Vector Model (3 gates + 6 derivatives)

Mathematical Foundation:
  Original: 3 gates (confidence, verification, anomaly) → composite suspicion → tier
  Enhanced: 3 gates + 6 derivative variables → composite + rate-of-change → sub-tier

The six derivative dimensions:
  1. ∂(confidence)/∂t: Rate of change in parser confidence over time
  2. ∂(verification)/∂t: Rate of change in DL1 alignment over time
  3. ∂(anomaly)/∂t: Rate of change in anomaly detection over time
  4. Slope at LOG→WARN boundary (approaching 0.45)
  5. Slope at WARN→BLOCK boundary (approaching 0.72)
  6. Convergence term: lim (1/3^n) as n→∞ (asymptotic stability)

This creates sub-tiers within each main tier:
  - PASS: Deep in tier, stable derivatives, moving away from boundaries
  - SLOW: Near boundary, unstable derivatives, approaching threshold
  - STOP: At boundary, requires intervention before crossing

Physical interpretation:
  - PASS = green light, auto-process
  - SLOW = yellow light, monitor closely
  - STOP = red light, require confirmation before crossing tier boundary

Calculus analogy: Integrals & derivatives in n-dimensional space
  - Composite suspicion = integral of gate vectors over time
  - Sub-tier classification = derivative at boundary (slope of approach)
  - Convergence = limit as precision → ∞ (1/3 = 0.333... → 1/∞)
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from webapp.parser.health.risk_gates import RiskGateConfig, RiskGateEvaluator, RiskGateScores


@dataclass
class DerivativeGates:
    """Six derivative dimensions for calculus-based risk assessment."""
    
    # Rate of change for each primary gate (∂/∂t)
    d_confidence_dt: float  # ∈ [-1, 1], negative = degrading, positive = improving
    d_verification_dt: float  # ∈ [-1, 1]
    d_anomaly_dt: float  # ∈ [-1, 1]
    
    # Boundary approach slopes (how fast approaching tipping points)
    slope_toward_warn: float  # ∈ [0, 1], 0 = stationary, 1 = rapid approach to 0.45
    slope_toward_block: float  # ∈ [0, 1], 0 = stationary, 1 = rapid approach to 0.72
    
    # Convergence to infinity (1/3^n as n→∞)
    convergence_stability: float  # ∈ [0, 1], 0 = diverging, 1 = converged


@dataclass
class SubTierClassification:
    """Sub-tier within main risk tier (PASS/SLOW/STOP)."""
    
    main_tier: str  # "log", "warn", or "block"
    sub_tier: str  # "pass", "slow", or "stop"
    composite_suspicion: float  # Main suspicion score
    boundary_distance: float  # Distance to nearest tier boundary
    approach_velocity: float  # Speed of approach toward boundary (from derivatives)
    action: str  # Recommended action (AUTO_PROCEED, MONITOR_CLOSELY, REQUIRE_CONFIRMATION)


class CalculusRiskEvaluator:
    """
    Nine-dimensional risk evaluator with derivative gates.
    
    Extends base RiskGateEvaluator with calculus-based boundary analysis:
      - Primary dimension: 3 gates (confidence, verification, anomaly)
      - Derivative dimension: 6 rate-of-change & slope variables
      - Sub-tier classification: PASS/SLOW/STOP within LOG/WARN/BLOCK
    
    Mathematical model:
      Let S(t) = composite suspicion at time t
      Let G = [confidence, verification, anomaly] be gate vector
      Let dG/dt = [d_confidence_dt, d_verification_dt, d_anomaly_dt]
      
      Approach velocity V(t) at boundary b:
        V(t) = |dS/dt| = |∂S/∂G · dG/dt|
              = |w₁·d_conf + w₂·d_verif + w₃·d_anom|
      
      Sub-tier classification at suspicion S with boundary b:
        distance = |S - b|
        if V(t) > threshold AND distance < ε:
          → STOP (about to cross boundary)
        elif V(t) > 0 AND distance < 2ε:
          → SLOW (approaching boundary)
        else:
          → PASS (stable in tier)
    """
    
    def __init__(self, config: Optional[RiskGateConfig] = None):
        """Initialize with optional custom configuration."""
        self.base_evaluator = RiskGateEvaluator(config=config)
        self.config = config or RiskGateConfig()
        
        # Sub-tier thresholds (epsilon distances from boundaries)
        self.epsilon_stop = 0.05  # Within 5% of boundary → STOP
        self.epsilon_slow = 0.15  # Within 15% of boundary → SLOW
        self.velocity_threshold = 0.1  # Approach velocity > 0.1 → unstable
    
    # =========================================================================
    # DERIVATIVE COMPUTATION: Rate of change for gates
    # =========================================================================
    
    def compute_derivative_gates(
        self,
        current_scores: RiskGateScores,
        previous_scores: Optional[RiskGateScores] = None,
        time_delta: float = 1.0
    ) -> DerivativeGates:
        """
        Compute derivative gates (∂/∂t) for each dimension.
        
        If previous_scores provided: compute actual derivatives
        If previous_scores absent: estimate from current position & trends
        
        Args:
            current_scores: Current RiskGateScores
            previous_scores: Optional previous RiskGateScores for time series
            time_delta: Time elapsed between measurements (default 1.0)
        
        Returns:
            DerivativeGates with 6 derivative dimensions
        """
        if previous_scores is not None:
            # Actual derivatives: ΔG / Δt
            d_conf = (
                (current_scores.confidence_gate - previous_scores.confidence_gate) 
                / time_delta
            )
            d_verif = (
                (current_scores.verification_gate - previous_scores.verification_gate) 
                / time_delta
            )
            d_anom = (
                (current_scores.anomaly_gate - previous_scores.anomaly_gate) 
                / time_delta
            )
        else:
            # Estimate: assume zero change if no history
            d_conf = 0.0
            d_verif = 0.0
            d_anom = 0.0
        
        # Boundary slopes: how fast are we approaching tipping points?
        suspicion = current_scores.composite_suspicion
        
        # Slope toward WARN boundary (0.45)
        slope_warn = self._compute_boundary_slope(
            suspicion,
            self.config.tier_boundary_warn_log,
            d_conf, d_verif, d_anom
        )
        
        # Slope toward BLOCK boundary (0.72)
        slope_block = self._compute_boundary_slope(
            suspicion,
            self.config.tier_boundary_block_warn,
            d_conf, d_verif, d_anom
        )
        
        # Convergence stability: how stable is the current state?
        # Using infinite series: Σ(1/3^n) as n→∞ converges to 0.5
        # We measure stability as inverse of derivative magnitude
        derivative_magnitude = math.sqrt(d_conf**2 + d_verif**2 + d_anom**2)
        convergence = 1.0 / (1.0 + derivative_magnitude)  # ∈ [0, 1]
        
        return DerivativeGates(
            d_confidence_dt=max(-1.0, min(1.0, d_conf)),
            d_verification_dt=max(-1.0, min(1.0, d_verif)),
            d_anomaly_dt=max(-1.0, min(1.0, d_anom)),
            slope_toward_warn=max(0.0, min(1.0, slope_warn)),
            slope_toward_block=max(0.0, min(1.0, slope_block)),
            convergence_stability=convergence
        )
    
    def _compute_boundary_slope(
        self,
        current_suspicion: float,
        boundary: float,
        d_conf: float,
        d_verif: float,
        d_anom: float
    ) -> float:
        """
        Compute slope of approach toward a boundary.
        
        Using chain rule: dS/dt = ∂S/∂G · dG/dt
        Where S = composite suspicion, G = gate vector
        
        ∂S/∂conf = -w₁ (negative because S = w₁(1-conf) + ...)
        ∂S/∂verif = -w₂
        ∂S/∂anom = +w₃
        
        Args:
            current_suspicion: Current suspicion score
            boundary: Boundary value (0.45 or 0.72)
            d_conf, d_verif, d_anom: Gate derivatives
        
        Returns:
            Slope ∈ [0, 1], normalized to boundary distance
        """
        # Compute dS/dt using chain rule
        dS_dt = (
            -self.config.weight_confidence * d_conf +
            -self.config.weight_verification * d_verif +
            self.config.weight_anomaly * d_anom
        )
        
        # Only count if moving TOWARD boundary (not away)
        distance = boundary - current_suspicion
        if distance > 0 and dS_dt > 0:
            # Moving up toward boundary from below
            approach_rate = dS_dt / max(0.01, distance)
        elif distance < 0 and dS_dt < 0:
            # Moving down toward boundary from above
            approach_rate = abs(dS_dt) / max(0.01, abs(distance))
        else:
            # Moving away or parallel; no approach
            approach_rate = 0.0
        
        # Normalize to [0, 1]
        return min(1.0, approach_rate)
    
    # =========================================================================
    # SUB-TIER CLASSIFICATION: PASS / SLOW / STOP
    # =========================================================================
    
    def classify_sub_tier(
        self,
        scores: RiskGateScores,
        derivatives: DerivativeGates
    ) -> SubTierClassification:
        """
        Classify into sub-tier (PASS/SLOW/STOP) within main tier.
        
        Decision logic:
          1. Identify main tier (LOG/WARN/BLOCK) from base classifier
          2. Compute distance to nearest boundary
          3. Compute approach velocity from derivatives
          4. Apply PASS/SLOW/STOP thresholds:
             - STOP: Within epsilon_stop AND velocity > threshold
                     (about to cross boundary; require intervention)
             - SLOW: Within epsilon_slow OR velocity > threshold
                     (approaching boundary or unstable; monitor closely)
             - PASS: Otherwise (stable, safe to auto-proceed)
        
        Args:
            scores: RiskGateScores from base evaluator
            derivatives: DerivativeGates from derivative computation
        
        Returns:
            SubTierClassification with sub-tier and action recommendation
        """
        suspicion = scores.composite_suspicion
        main_tier = scores.risk_tier
        
        # Identify nearest boundary and compute distance
        if main_tier == "log":
            nearest_boundary = self.config.tier_boundary_warn_log
            boundary_distance = nearest_boundary - suspicion
        elif main_tier == "warn":
            # WARN tier has two boundaries; pick nearest
            dist_to_log = suspicion - self.config.tier_boundary_warn_log
            dist_to_block = self.config.tier_boundary_block_warn - suspicion
            if dist_to_log < dist_to_block:
                nearest_boundary = self.config.tier_boundary_warn_log
                boundary_distance = -dist_to_log  # Negative = below boundary
            else:
                nearest_boundary = self.config.tier_boundary_block_warn
                boundary_distance = dist_to_block
        else:  # block
            nearest_boundary = self.config.tier_boundary_block_warn
            boundary_distance = -(suspicion - nearest_boundary)  # Negative = above boundary
        
        # Compute approach velocity (magnitude of dS/dt)
        approach_velocity = abs(
            -self.config.weight_confidence * derivatives.d_confidence_dt +
            -self.config.weight_verification * derivatives.d_verification_dt +
            self.config.weight_anomaly * derivatives.d_anomaly_dt
        )
        
        # Classify sub-tier
        abs_distance = abs(boundary_distance)
        
        if abs_distance <= self.epsilon_stop and approach_velocity > self.velocity_threshold:
            sub_tier = "stop"
            action = "REQUIRE_CONFIRMATION"
        elif abs_distance <= self.epsilon_slow or approach_velocity > self.velocity_threshold:
            sub_tier = "slow"
            action = "MONITOR_CLOSELY"
        else:
            sub_tier = "pass"
            action = "AUTO_PROCEED"
        
        return SubTierClassification(
            main_tier=main_tier,
            sub_tier=sub_tier,
            composite_suspicion=suspicion,
            boundary_distance=boundary_distance,
            approach_velocity=approach_velocity,
            action=action
        )
    
    # =========================================================================
    # UNIFIED EVALUATION: Base + Derivatives + Sub-Tier
    # =========================================================================
    
    def evaluate_with_derivatives(
        self,
        extraction_confidence: float,
        ground_truth_matches: int,
        total_records: int,
        suspicious_pattern_count: int = 0,
        outlier_record_count: int = 0,
        integrity_flags: Optional[List[str]] = None,
        previous_scores: Optional[RiskGateScores] = None,
        time_delta: float = 1.0
    ) -> Tuple[RiskGateScores, DerivativeGates, SubTierClassification]:
        """
        Complete nine-dimensional risk evaluation.
        
        Pipeline:
          1. Compute base 3-gate risk (confidence, verification, anomaly)
          2. Compute 6 derivative gates (∂/∂t, slopes, convergence)
          3. Classify into sub-tier (PASS/SLOW/STOP) within main tier
        
        Returns:
            Tuple[RiskGateScores, DerivativeGates, SubTierClassification]
        """
        # Step 1: Base evaluation (3 gates → suspicion → tier)
        scores = self.base_evaluator.evaluate(
            extraction_confidence=extraction_confidence,
            ground_truth_matches=ground_truth_matches,
            total_records=total_records,
            suspicious_pattern_count=suspicious_pattern_count,
            outlier_record_count=outlier_record_count,
            integrity_flags=integrity_flags
        )
        
        # Step 2: Derivative gates (6 dimensions)
        derivatives = self.compute_derivative_gates(scores, previous_scores, time_delta)
        
        # Step 3: Sub-tier classification (PASS/SLOW/STOP)
        sub_tier = self.classify_sub_tier(scores, derivatives)
        
        return scores, derivatives, sub_tier


# === Module-level convenience function ===

_default_calculus_evaluator = CalculusRiskEvaluator()


def evaluate_risk_with_calculus(
    extraction_confidence: float,
    ground_truth_matches: int,
    total_records: int,
    **kwargs
) -> Tuple[RiskGateScores, DerivativeGates, SubTierClassification]:
    """
    Quick nine-dimensional risk evaluation using default config.
    
    Args:
        extraction_confidence: Parser confidence (0–1)
        ground_truth_matches: Count of DL1-verified matches
        total_records: Total records imported/extracted
        **kwargs: Additional arguments (previous_scores, time_delta, etc.)
    
    Returns:
        Tuple[RiskGateScores, DerivativeGates, SubTierClassification]
    """
    return _default_calculus_evaluator.evaluate_with_derivatives(
        extraction_confidence,
        ground_truth_matches,
        total_records,
        **kwargs
    )


# =============================================================================
# CALCULUS-BASED TIER VISUALIZATION
# =============================================================================

TIER_VISUAL_MAP = """
NINE-DIMENSIONAL RISK TIER MAP (with sub-tiers)
═══════════════════════════════════════════════════════════════════════════

                                    SUSPICION SCALE
        ┌─────────────────────────────────────────────────────────────────┐
        0.00                  0.45                  0.72                1.00
        └───────────────────────┼──────────────────────┼──────────────────┘
                               ▼                      ▼
                          LOG→WARN                WARN→BLOCK
                          boundary                 boundary

LOG TIER (0.00 – 0.45)
├─ PASS (0.00 – 0.30):  🟢 Deep in tier, stable, auto-proceed
├─ SLOW (0.30 – 0.40):  🟡 Approaching boundary, monitor closely
└─ STOP (0.40 – 0.45):  🔴 At boundary, require confirmation before crossing

WARN TIER (0.45 – 0.72)
├─ PASS (0.50 – 0.67):  🟢 Stable in tier, monitor standard
├─ SLOW (0.45 – 0.50 OR 0.67 – 0.72):  🟡 Near boundaries, watch closely
└─ STOP (boundary ± 0.05):  🔴 About to cross, require intervention

BLOCK TIER (0.72 – 1.00)
├─ PASS (0.85 – 1.00):  🟢 Deep escalation, admin review
├─ SLOW (0.72 – 0.85):  🟡 High risk but not critical
└─ STOP (boundary ± 0.05):  🔴 Critical threshold, immediate action

DERIVATIVE INFLUENCE (6 dimensions):
  ∂(confidence)/∂t → affects approach velocity
  ∂(verification)/∂t → affects stability
  ∂(anomaly)/∂t → affects trend direction
  slope_toward_warn → influences SLOW/STOP at 0.45
  slope_toward_block → influences SLOW/STOP at 0.72
  convergence_stability → overall system stability (lim 1/3^n → 0)

ACTIONS BY SUB-TIER:
  PASS:  AUTO_PROCEED (green light, no intervention needed)
  SLOW:  MONITOR_CLOSELY (yellow light, watch trends)
  STOP:  REQUIRE_CONFIRMATION (red light, human approval before tier change)
"""


def visualize_sub_tier_classification(sub_tier: SubTierClassification) -> str:
    """
    Generate human-readable visualization of sub-tier classification.
    
    Args:
        sub_tier: SubTierClassification result
    
    Returns:
        Formatted string with tier position and action
    """
    emoji_map = {
        "pass": "🟢",
        "slow": "🟡",
        "stop": "🔴"
    }
    
    emoji = emoji_map.get(sub_tier.sub_tier, "⚪")
    
    return f"""
    {emoji} {sub_tier.main_tier.upper()} tier → {sub_tier.sub_tier.upper()} sub-tier
    
    Composite Suspicion: {sub_tier.composite_suspicion:.3f}
    Distance to Boundary: {abs(sub_tier.boundary_distance):.3f}
    Approach Velocity: {sub_tier.approach_velocity:.3f}
    
    Recommended Action: {sub_tier.action}
    """
