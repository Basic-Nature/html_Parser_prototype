from __future__ import annotations

from webapp.parser.health.risk_gates import RiskGateEvaluator
from webapp.parser.health.risk_gates_calculus import CalculusRiskEvaluator
from webapp.parser.html_election_parser import _apply_risk_assessment


def test_risk_gate_evaluator_initializes_with_valid_boundaries():
    evaluator = RiskGateEvaluator()
    scores = evaluator.evaluate(
        extraction_confidence=0.9,
        ground_truth_matches=9,
        total_records=10,
        suspicious_pattern_count=0,
        outlier_record_count=0,
    )
    assert scores.risk_tier == "log"
    assert 0.0 <= scores.composite_suspicion <= 1.0


def test_calculus_evaluator_supports_fallback_verification_score():
    evaluator = CalculusRiskEvaluator()
    scores, derivatives, sub_tier = evaluator.evaluate_with_derivatives(
        extraction_confidence=0.95,
        ground_truth_matches=0,
        total_records=0,
        suspicious_pattern_count=0,
        outlier_record_count=0,
        fallback_verification_score=1.0,
    )
    assert scores.risk_tier == "log"
    assert sub_tier.action in {"AUTO_PROCEED", "MONITOR_CLOSELY", "REQUIRE_CONFIRMATION"}
    assert 0.0 <= derivatives.convergence_stability <= 1.0


def test_calculus_evaluator_blocks_high_suspicion_data():
    evaluator = CalculusRiskEvaluator()
    scores, _, _ = evaluator.evaluate_with_derivatives(
        extraction_confidence=0.2,
        ground_truth_matches=1,
        total_records=10,
        suspicious_pattern_count=4,
        outlier_record_count=6,
    )
    assert scores.risk_tier == "block"
    assert scores.composite_suspicion >= 0.72


def test_apply_risk_assessment_enriches_metadata():
    headers = ["candidate", "votes"]
    data = [{"candidate": "A", "votes": 1} for _ in range(20)]
    metadata = {
        "row_count": 20,
        "quality_metrics": {"extraction_confidence": 0.92},
        "audit_signals": {"anomaly_count": 1, "semantic_mismatch_count": 0},
    }

    enriched = _apply_risk_assessment(
        headers,
        data,
        metadata,
        session_id="test-risk-session",
        trust_score=99.99,
    )

    assert isinstance(enriched.get("risk_assessment"), dict)
    assert enriched.get("risk_tier") in {"log", "warn", "block"}
    assert enriched.get("risk_sub_tier") in {"pass", "slow", "stop"}
    assert enriched.get("risk_action") in {
        "AUTO_PROCEED",
        "MONITOR_CLOSELY",
        "REQUIRE_CONFIRMATION",
    }
