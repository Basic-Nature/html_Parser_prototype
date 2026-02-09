"""
CALCULUS RISK GATES: PRACTICAL USAGE EXAMPLES
Smart Elections Parser – 9-Dimensional Model Demonstrations

This demonstrates the enhanced calculus-based risk assessment with:
  - 3 primary gates (confidence, verification, anomaly)
  - 6 derivative gates (∂/∂t, slopes, convergence)
  - PASS/SLOW/STOP sub-tier classification
"""

from webapp.parser.health.risk_gates_calculus import (
    evaluate_risk_with_calculus,
    CalculusRiskEvaluator,
    visualize_sub_tier_classification
)
from webapp.parser.health.risk_gates import evaluate_risk


# =============================================================================
# EXAMPLE 1: Basic 3-gate model (simple use case)
# =============================================================================
def example_basic_three_gate():
    """
    Simple use case: Just need tier classification (LOG/WARN/BLOCK).
    No derivatives, no sub-tiers—quick assessment.
    """
    print("=" * 70)
    print("EXAMPLE 1: Basic 3-Gate Model (Confidence, Verification, Anomaly)")
    print("=" * 70)
    
    scores = evaluate_risk(
        extraction_confidence=0.87,
        ground_truth_matches=42,
        total_records=48,
        suspicious_pattern_count=0,
        outlier_record_count=1
    )
    
    print(f"\nConfidence Gate: {scores.confidence_gate:.3f}")
    print(f"Verification Gate: {scores.verification_gate:.3f}")
    print(f"Anomaly Gate: {scores.anomaly_gate:.3f}")
    print(f"Composite Suspicion: {scores.composite_suspicion:.3f}")
    print(f"Risk Tier: {scores.risk_tier.upper()}")
    print(f"Tier Confidence: {scores.tier_confidence:.2%}")
    
    print("\nAction: AUTO-IMPORT (LOG tier; no confirmation needed)")


# =============================================================================
# EXAMPLE 2: Enhanced 9-dimensional model (with derivatives)
# =============================================================================
def example_calculus_with_derivatives():
    """
    Enhanced use case: Track time-series data, detect trends.
    Includes 6 derivative dimensions + sub-tier classification.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: 9-Dimensional Calculus Model (with Derivatives)")
    print("=" * 70)
    
    # First assessment (no history yet)
    scores_t0, derivatives_t0, sub_tier_t0 = evaluate_risk_with_calculus(
        extraction_confidence=0.82,
        ground_truth_matches=150,
        total_records=200,
        suspicious_pattern_count=5,
        outlier_record_count=8
    )
    
    print("\n--- Time T=0 (Initial Assessment) ---")
    print(f"Main Tier: {scores_t0.risk_tier.upper()}")
    print(f"Sub-Tier: {sub_tier_t0.sub_tier.upper()}")
    print(f"Action: {sub_tier_t0.action}")
    print(f"Composite Suspicion: {scores_t0.composite_suspicion:.3f}")
    print(f"Boundary Distance: {abs(sub_tier_t0.boundary_distance):.3f}")
    
    # Second assessment (1 minute later, confidence degrading)
    scores_t1, derivatives_t1, sub_tier_t1 = evaluate_risk_with_calculus(
        extraction_confidence=0.75,  # Dropped from 0.82
        ground_truth_matches=140,  # Dropped from 150
        total_records=200,
        suspicious_pattern_count=8,  # Increased from 5
        outlier_record_count=12,  # Increased from 8
        previous_scores=scores_t0,  # Pass previous for ∂/∂t calculation
        time_delta=1.0  # 1 minute elapsed
    )
    
    print("\n--- Time T=1 (1 minute later, quality degrading) ---")
    print(f"Main Tier: {scores_t1.risk_tier.upper()}")
    print(f"Sub-Tier: {sub_tier_t1.sub_tier.upper()}")
    print(f"Action: {sub_tier_t1.action}")
    print(f"Composite Suspicion: {scores_t1.composite_suspicion:.3f}")
    print(f"Boundary Distance: {abs(sub_tier_t1.boundary_distance):.3f}")
    print(f"Approach Velocity: {sub_tier_t1.approach_velocity:.3f}")
    
    print("\n--- Derivative Analysis (Rate of Change) ---")
    print(f"∂(confidence)/∂t: {derivatives_t1.d_confidence_dt:.3f}")
    print(f"∂(verification)/∂t: {derivatives_t1.d_verification_dt:.3f}")
    print(f"∂(anomaly)/∂t: {derivatives_t1.d_anomaly_dt:.3f}")
    print(f"Slope toward WARN: {derivatives_t1.slope_toward_warn:.3f}")
    print(f"Slope toward BLOCK: {derivatives_t1.slope_toward_block:.3f}")
    print(f"Convergence Stability: {derivatives_t1.convergence_stability:.3f}")
    
    print("\nInterpretation:")
    print("  → Confidence is DEGRADING (negative ∂/∂t)")
    print("  → Verification is DEGRADING (negative ∂/∂t)")
    print("  → Anomalies are INCREASING (positive ∂/∂t)")
    print("  → System is approaching WARN boundary (slope_toward_warn > 0)")
    print(f"  → Recommendation: {sub_tier_t1.action}")


# =============================================================================
# EXAMPLE 3: Sub-tier transitions (PASS → SLOW → STOP)
# =============================================================================
def example_subtier_transitions():
    """
    Demonstrate how sub-tiers change as data quality degrades.
    Shows PASS → SLOW → STOP transitions within LOG tier.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Sub-Tier Transitions (PASS → SLOW → STOP)")
    print("=" * 70)
    
    evaluator = CalculusRiskEvaluator()
    
    # Scenario: Parser confidence slowly degrading from 0.90 to 0.50
    confidence_values = [0.90, 0.85, 0.75, 0.65, 0.55, 0.50]
    previous = None
    
    for idx, conf in enumerate(confidence_values):
        scores, derivatives, sub_tier = evaluator.evaluate_with_derivatives(
            extraction_confidence=conf,
            ground_truth_matches=int(100 * conf),  # Proportional to confidence
            total_records=100,
            suspicious_pattern_count=int(5 * (1 - conf)),  # Inversely proportional
            outlier_record_count=int(8 * (1 - conf)),
            previous_scores=previous,
            time_delta=1.0
        )
        
        emoji = {"pass": "🟢", "slow": "🟡", "stop": "🔴"}.get(sub_tier.sub_tier, "⚪")
        
        print(f"\nStep {idx+1}: Confidence = {conf:.2f}")
        print(f"  {emoji} {scores.risk_tier.upper()} / {sub_tier.sub_tier.upper()}")
        print(f"  Suspicion: {scores.composite_suspicion:.3f}")
        print(f"  Action: {sub_tier.action}")
        
        if sub_tier.sub_tier == "slow":
            print(f"  ⚠️  Approaching boundary! Distance: {abs(sub_tier.boundary_distance):.3f}")
        elif sub_tier.sub_tier == "stop":
            print(f"  🚨 AT BOUNDARY! Confirm before crossing into {scores.risk_tier.upper()} tier.")
        
        previous = scores


# =============================================================================
# EXAMPLE 4: Infinity convergence (1/3 = 0.333... → 1/∞)
# =============================================================================
def example_infinity_convergence():
    """
    Demonstrate convergence stability as system approaches equilibrium.
    Shows how the 6th derivative dimension captures asymptotic behavior.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Infinity Convergence (1/3 → 1/∞)")
    print("=" * 70)
    
    print("\nScenario: System starts unstable, gradually converges to stable state")
    print("Watch convergence_stability approach 1.0 as system stabilizes.\n")
    
    evaluator = CalculusRiskEvaluator()
    
    # Simulate system stabilizing over 5 time steps
    stability_trend = [
        (0.75, 0.70, 0.15),  # Initial: moderate quality
        (0.78, 0.72, 0.12),  # Step 1: slight improvement
        (0.80, 0.75, 0.10),  # Step 2: continued improvement
        (0.82, 0.78, 0.08),  # Step 3: approaching stable
        (0.83, 0.80, 0.07),  # Step 4: nearly stable
        (0.83, 0.80, 0.07),  # Step 5: fully converged (no change)
    ]
    
    previous = None
    
    for idx, (conf, verif, anom) in enumerate(stability_trend):
        scores, derivatives, sub_tier = evaluator.evaluate_with_derivatives(
            extraction_confidence=conf,
            ground_truth_matches=int(100 * verif),
            total_records=100,
            suspicious_pattern_count=int(10 * anom),
            outlier_record_count=int(5 * anom),
            previous_scores=previous,
            time_delta=1.0
        )
        
        print(f"Step {idx+1}:")
        print(f"  Gates: conf={conf:.2f}, verif={verif:.2f}, anom={anom:.2f}")
        print(f"  Convergence Stability: {derivatives.convergence_stability:.4f}")
        print(f"  Derivative Magnitude: {abs(derivatives.d_confidence_dt) + abs(derivatives.d_verification_dt) + abs(derivatives.d_anomaly_dt):.4f}")
        
        if derivatives.convergence_stability > 0.95:
            print(f"  ✅ CONVERGED (stability → 1.0, derivatives → 0)")
        elif derivatives.convergence_stability > 0.80:
            print(f"  ⏳ APPROACHING CONVERGENCE...")
        else:
            print(f"  📈 UNSTABLE (high rate of change)")
        
        previous = scores
    
    print("\nInterpretation:")
    print("  As system stabilizes (∂G/∂t → 0), convergence_stability → 1.0")
    print("  This captures the asymptotic behavior: lim(1/3^n) as n→∞ = 0")
    print("  Stable systems converge; unstable systems diverge.")


# =============================================================================
# EXAMPLE 5: Comparison: 3-gate vs 9-dimensional model
# =============================================================================
def example_comparison_3d_vs_9d():
    """
    Side-by-side comparison: same data, different assessments.
    Shows how derivatives provide additional insight.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Comparison (3-Gate vs 9-Dimensional)")
    print("=" * 70)
    
    # Same input data
    extraction_confidence = 0.68
    ground_truth_matches = 60
    total_records = 100
    suspicious_patterns = 12
    outliers = 8
    
    print("\nInput Data:")
    print(f"  Extraction Confidence: {extraction_confidence:.2f}")
    print(f"  Ground Truth Matches: {ground_truth_matches}/{total_records}")
    print(f"  Suspicious Patterns: {suspicious_patterns}")
    print(f"  Outliers: {outliers}")
    
    # 3-gate assessment
    print("\n--- 3-Gate Model (Base) ---")
    scores_3d = evaluate_risk(
        extraction_confidence=extraction_confidence,
        ground_truth_matches=ground_truth_matches,
        total_records=total_records,
        suspicious_pattern_count=suspicious_patterns,
        outlier_record_count=outliers
    )
    print(f"Risk Tier: {scores_3d.risk_tier.upper()}")
    print(f"Composite Suspicion: {scores_3d.composite_suspicion:.3f}")
    print(f"Action: {'AUTO-IMPORT' if scores_3d.risk_tier == 'log' else 'CONFIRM' if scores_3d.risk_tier == 'warn' else 'ESCALATE'}")
    
    # 9-dimensional assessment
    print("\n--- 9-Dimensional Model (with Derivatives) ---")
    scores_9d, derivatives_9d, sub_tier_9d = evaluate_risk_with_calculus(
        extraction_confidence=extraction_confidence,
        ground_truth_matches=ground_truth_matches,
        total_records=total_records,
        suspicious_pattern_count=suspicious_patterns,
        outlier_record_count=outliers
    )
    print(f"Risk Tier: {scores_9d.risk_tier.upper()} / {sub_tier_9d.sub_tier.upper()}")
    print(f"Composite Suspicion: {scores_9d.composite_suspicion:.3f}")
    print(f"Action: {sub_tier_9d.action}")
    print(f"Boundary Distance: {abs(sub_tier_9d.boundary_distance):.3f}")
    print(f"Approach Velocity: {sub_tier_9d.approach_velocity:.3f}")
    
    print("\nKey Difference:")
    print("  3-Gate Model: Provides tier classification only")
    print("  9-Dimensional: Adds sub-tier, boundary distance, approach velocity")
    print("                 → More granular control for edge cases")


# =============================================================================
# RUN ALL EXAMPLES
# =============================================================================
if __name__ == "__main__":
    example_basic_three_gate()
    example_calculus_with_derivatives()
    example_subtier_transitions()
    example_infinity_convergence()
    example_comparison_3d_vs_9d()
    
    print("\n" + "=" * 70)
    print("All examples complete!")
    print("=" * 70)
