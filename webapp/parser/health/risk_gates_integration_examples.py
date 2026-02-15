"""
risk_gates_integration_examples.py

Practical examples showing how the three-dimensional risk gate model
applies to parsing workflows, data validation, and access control.

Use cases:
  1. Parser ingestion: Decide whether to block/warn/log extraction
  2. Data Framework upload: Gate sensitive data imports via risk tier
  3. BallotLens browsing: Filter/highlight results by suspicion level
  4. Guarded actions: Require confirmation for high-risk operations
"""

from typing import Any, Dict, Optional

from webapp.parser.health.risk_gates import RiskGateEvaluator


# =============================================================================
# EXAMPLE 1: Parser Ingestion Workflow
# =============================================================================
def evaluate_parser_extraction(
    contest_id: str,
    state: str,
    year: int,
    extraction_confidence: float,
    extracted_row_count: int,
    dl1_verified_matches: int = 0,  # Rows matching ground truth
    suspicious_keywords_found: int = 0,
    statistical_outliers: int = 0
) -> Dict[str, Any]:
    """
    Evaluate parser extraction risk and determine ingestion action.
    
    Scenario:
      Parser finished extracting an election result PDF.
      - extraction_confidence: 0.88 (parser is confident)
      - extracted_row_count: 256 candidates
      - dl1_verified_matches: 215 (84% match DL1 ground truth)
      - suspicious_keywords_found: 3 (e.g., "test data" in notes)
      - statistical_outliers: 2 (votes > population, one data entry issue)
    
    Action logic:
      - BLOCK (tier >= 0.72): Refuse to import; escalate to human review
      - WARN (0.45 ≤ tier < 0.72): Import with explicit confirmation prompt
      - LOG (tier < 0.45): Auto-import; log decision
    
    Returns:
        Dict with risk assessment, recommended action, and audit context.
    """
    evaluator = RiskGateEvaluator()
    
    # Evaluate risk
    scores = evaluator.evaluate(
        extraction_confidence=extraction_confidence,
        ground_truth_matches=dl1_verified_matches,
        total_records=extracted_row_count,
        suspicious_pattern_count=suspicious_keywords_found,
        outlier_record_count=statistical_outliers,
        integrity_flags=["test_keywords"] if suspicious_keywords_found > 0 else None
    )
    
    # Determine action
    action = "IMPORT_AUTO" if scores.risk_tier == "log" else None
    action = "IMPORT_CONFIRM" if scores.risk_tier == "warn" else action
    action = "IMPORT_ESCALATE" if scores.risk_tier == "block" else action
    
    return {
        "contest_id": contest_id,
        "state": state,
        "year": year,
        "risk_assessment": {
            "confidence_gate": round(scores.confidence_gate, 3),
            "verification_gate": round(scores.verification_gate, 3),
            "anomaly_gate": round(scores.anomaly_gate, 3),
            "composite_suspicion": round(scores.composite_suspicion, 3),
            "risk_tier": scores.risk_tier,
            "tier_confidence": round(scores.tier_confidence, 2),
        },
        "recommended_action": action,
        "audit_note": f"Extracted {extracted_row_count} candidates with {extraction_confidence:.0%} parser confidence. "
                     f"DL1 verification: {dl1_verified_matches}/{extracted_row_count} ({100*scores.verification_gate:.0f}%). "
                     f"Anomaly flags: {suspicious_keywords_found} patterns + {statistical_outliers} outliers."
    }


# =============================================================================
# EXAMPLE 2: Data Framework Upload Gating
# =============================================================================
def evaluate_data_framework_upload(
    source_dataset_name: str,
    row_count: int,
    target_table: str,
    parser_version: str,
    extraction_confidence_avg: float,
    dl1_match_percentage: float,
    anomaly_ratio: float
) -> Dict[str, Any]:
    """
    Evaluate whether to allow direct upload to Data Framework or require review.
    
    Scenario:
      User wants to upload parsed election results directly to the
      SQL-backed Data Framework for browsing and comparison.
      Risk: If data quality is poor, it pollutes the public viewport.
    
    Returns:
        Dict with gating decision and required confirmation level.
    """
    evaluator = RiskGateEvaluator()
    
    scores = evaluator.evaluate(
        extraction_confidence=extraction_confidence_avg,
        ground_truth_matches=int(row_count * dl1_match_percentage),
        total_records=row_count,
        outlier_record_count=int(row_count * anomaly_ratio),
        integrity_flags=[]
    )
    
    gates_description = {
        "confidence_gate": scores.confidence_gate,
        "verification_gate": scores.verification_gate,
        "anomaly_gate": scores.anomaly_gate
    }
    
    # Gating logic
    if scores.risk_tier == "log":
        gate_decision = "ALLOW_AUTO"
        required_approval = None
        note = "Data quality acceptable; direct upload approved."
    elif scores.risk_tier == "warn":
        gate_decision = "ALLOW_WITH_CONFIRMATION"
        required_approval = "USER_PROMPT"
        note = "Medium-risk data; requires user confirmation before upload."
    else:  # block
        gate_decision = "DENY"
        required_approval = "ADMIN_REVIEW"
        note = "High-risk data; escalated to admin review. Do not expose to Data Framework."
    
    return {
        "source": source_dataset_name,
        "target_table": target_table,
        "parser_version": parser_version,
        "row_count": row_count,
        "risk_gates": gates_description,
        "composite_suspicion": round(scores.composite_suspicion, 3),
        "gate_decision": gate_decision,
        "required_approval": required_approval,
        "audit_note": note
    }


# =============================================================================
# EXAMPLE 3: BallotLens Result Filtering & Highlighting
# =============================================================================
def evaluate_ballot_lens_display(
    contest_id: str,
    candidate_name: str,
    votes_received: int,
    extraction_confidence: float,
    dl1_verified: bool = False,
    anomaly_flags: Optional[list] = None
) -> Dict[str, Any]:
    """
    Determine how to display a candidate result in BallotLens UI.
    
    Scenario:
      BallotLens shows election results to voters with confidence indicators.
      Each row can be filtered/highlighted based on risk tier.
    
    Display hints:
      - LOG tier: Green badge, normal display (can filter out if user wants "high confidence only")
      - WARN tier: Yellow badge, normal display with "⚠️ verify" tooltip
      - BLOCK tier: Red badge, hidden by default (requires explicit click to show)
    
    Returns:
        Dict with UI display hints and filter categorization.
    """
    evaluator = RiskGateEvaluator()
    
    scores = evaluator.evaluate(
        extraction_confidence=extraction_confidence,
        ground_truth_matches=1 if dl1_verified else 0,
        total_records=1,
        integrity_flags=anomaly_flags or []
    )
    
    # UI styling
    badge_color = {
        "log": "#28a745",    # Green
        "warn": "#ffc107",   # Yellow
        "block": "#dc3545"   # Red
    }[scores.risk_tier]
    
    visibility = {
        "log": "visible",
        "warn": "visible",
        "block": "hidden_by_default"  # Click to reveal
    }[scores.risk_tier]
    
    return {
        "contest_id": contest_id,
        "candidate_name": candidate_name,
        "votes": votes_received,
        "risk_tier": scores.risk_tier,
        "badge_color": badge_color,
        "visibility": visibility,
        "tooltip": (
            "High confidence extraction, verified against ground truth (DL1)."
            if scores.risk_tier == "log"
            else (
                "⚠️ Medium-risk extraction. Verify against official results."
                if scores.risk_tier == "warn"
                else "🚫 Low confidence extraction. Verify with official sources before using."
            )
        ),
        "filter_categories": {
            "high_confidence_only": scores.risk_tier == "log",
            "with_warnings": scores.risk_tier in ("log", "warn"),
            "all_results": True
        }
    }


# =============================================================================
# EXAMPLE 4: Guarded Action Confirmation (Risk-Based Gate)
# =============================================================================
def evaluate_guarded_action(
    action_type: str,
    action_context: Dict[str, Any],
    user_has_guarded_key: bool = False
) -> Dict[str, Any]:
    """
    Determine whether a sensitive action requires guarded key confirmation.
    
    Sensitive actions:
      - Adding a new URL to the parser's source library
      - Uploading data to Data Framework (high-risk tier)
      - Running parser on unverified source
    
    Guarded key: GitHub secret env var + local terminal prompt
      - User must either: (a) have CI/Azure env var, or (b) enter local prompt
      - Prevents accidental or unauthorized sensitive operations
    
    Returns:
        Dict with confirmation requirement and enforcement level.
    """
    
    # Context-specific risk (examples)
    risk_profiles = {
        "add_source_url": {
            "suspicion": 0.6,  # Medium risk
            "reason": "New source URLs increase parser scope; verify legitimacy"
        },
        "upload_to_data_framework_high_risk": {
            "suspicion": 0.85,  # High risk
            "reason": "High-risk data would pollute public electoral viewport"
        },
        "parse_unverified_pdf": {
            "suspicion": 0.72,  # Borderline block
            "reason": "PDF source not in verified allowlist; potential for injection"
        }
    }
    
    profile = risk_profiles.get(action_type, {"suspicion": 0.5, "reason": "Unknown action"})
    
    # Determine enforcement
    if profile["suspicion"] >= 0.72:
        enforcement = "BLOCK" if not user_has_guarded_key else "ALLOW_WITH_CONFIRMATION"
        gate_status = "LOCKED"
    elif profile["suspicion"] >= 0.45:
        enforcement = "WARN_WITH_CONFIRMATION"
        gate_status = "CAUTION"
    else:
        enforcement = "ALLOW_AUTO"
        gate_status = "OPEN"
    
    return {
        "action_type": action_type,
        "estimated_risk_tier": (
            "block" if profile["suspicion"] >= 0.72
            else ("warn" if profile["suspicion"] >= 0.45 else "log")
        ),
        "gate_status": gate_status,
        "enforcement": enforcement,
        "required_guarded_key": profile["suspicion"] >= 0.45,
        "user_has_key": user_has_guarded_key,
        "reason": profile["reason"],
        "suggested_fallback": (
            "Local terminal prompt for guarded key" if not user_has_guarded_key
            else "Environment variable (CI/Azure) detected; key required"
        )
    }


# =============================================================================
# EXAMPLE 5: Tier-Based Data Access Control (Admin Dashboard)
# =============================================================================
def summarize_risk_distribution(
    parsed_contests: list
) -> Dict[str, Any]:
    """
    Summary view for admin dashboard: risk tier distribution across datasets.
    
    Useful for:
      - Monitoring data quality trends
      - Identifying systematic issues (e.g., all CA parsing low confidence)
      - Planning review workload (number of WARN/BLOCK items)
    
    Sample input:
      parsed_contests = [
        {"state": "CA", "year": 2024, "confidence": 0.92, "dl1_match": 0.88, ...},
        {"state": "TX", "year": 2024, "confidence": 0.65, "dl1_match": 0.42, ...},
        ...
      ]
    
    Returns:
        Aggregated risk metrics for dashboard.
    """
    evaluator = RiskGateEvaluator()
    
    tier_counts = {"log": 0, "warn": 0, "block": 0}
    state_risk_profile = {}  # Aggregate by state
    
    for contest in parsed_contests:
        scores = evaluator.evaluate(
            extraction_confidence=contest.get("confidence", 0.5),
            ground_truth_matches=int(contest.get("rows", 0) * contest.get("dl1_match", 0.0)),
            total_records=contest.get("rows", 0),
            outlier_record_count=int(contest.get("rows", 0) * contest.get("anomaly_ratio", 0.0))
        )
        
        tier_counts[scores.risk_tier] += 1
        
        state = contest.get("state", "UNKNOWN")
        if state not in state_risk_profile:
            state_risk_profile[state] = {"log": 0, "warn": 0, "block": 0}
        state_risk_profile[state][scores.risk_tier] += 1
    
    total = sum(tier_counts.values())
    
    return {
        "total_contests": total,
        "tier_distribution": tier_counts,
        "tier_percentages": {
            "log": round(100 * tier_counts["log"] / total, 1) if total > 0 else 0,
            "warn": round(100 * tier_counts["warn"] / total, 1) if total > 0 else 0,
            "block": round(100 * tier_counts["block"] / total, 1) if total > 0 else 0
        },
        "state_profiles": state_risk_profile,
        "immediate_action_items": (
            f"{tier_counts['block']} items awaiting admin review (BLOCK tier)"
        ),
        "user_confirmation_items": (
            f"{tier_counts['warn']} items awaiting user confirmation (WARN tier)"
        ),
        "auto_processing": (
            f"{tier_counts['log']} items auto-processed (LOG tier)"
        )
    }


# =============================================================================
# REFERENCE: Three-Gate Model Explained
# =============================================================================
"""
THREE-DIMENSIONAL RISK ASSESSMENT MODEL
========================================

Core Dimensions:
  1. CONFIDENCE GATE (Extraction Conviction)
     - Measures: How certain is the parser in its extraction?
     - Range: 0.0 (garbage) → 1.0 (absolute certainty)
     - Impact: Low confidence raises suspicion
     
  2. VERIFICATION GATE (Ground Truth Alignment)
     - Measures: What % of extracted data matches verified DL1 ground truth?
     - Range: 0.0 (no matches) → 1.0 (perfect match)
     - Impact: Low verification raises suspicion
     
  3. ANOMALY GATE (Statistical Suspension)
     - Measures: Are there patterns/outliers suggesting error or fraud?
     - Range: 0.0 (clean) → 1.0 (highly suspicious)
     - Impact: High anomaly directly raises suspicion

Composite Suspicion Formula:
  suspicion = w₁(1 - confidence) + w₂(1 - verification) + w₃(anomaly)
  where w₁ + w₂ + w₃ = 1.0 (default: 0.33 each)

Reasoning (Inverse for first two):
  - Parser claims 90% certainty (conf=0.9) → suspicion contribution: 0.33×(1-0.9) = 0.033
  - Data matches DL1 at 50% (verif=0.5) → suspicion contribution: 0.33×(1-0.5) = 0.165
  - Found 15% anomalies (anom=0.15) → suspicion contribution: 0.34×0.15 = 0.051
  - Total suspicion = 0.249 → TIER: LOG (< 0.45)

Risk Tier Boundaries (⅓-Proportioned):
  
  SUSPICION < 0.45 (Lower Third)
    └─ ACTION: LOG_ONLY
       Auto-import/process; audit trail only
       Confidence: 0–100%; how deep into LOG tier
    
  0.45 ≤ SUSPICION < 0.72 (Middle Third)
    └─ ACTION: WARN_CONFIRM
       User/system must confirm before proceeding
       Confidence: 0–100%; position within WARN tier (0=near LOG, 1=near BLOCK)
    
  SUSPICION >= 0.72 (Upper Third)
    └─ ACTION: BLOCK_ESCALATE
       Refuse; escalate to human admin review
       Confidence: 0–100%; how deep into BLOCK tier

The "Third More/Less" Principle:
  ⅓ width per tier ensures proportional distribution:
    - Lower third: 0.00–0.45 (45% of suspicion space)
    - Middle third: 0.45–0.72 (27% of suspicion space)
    - Upper third: 0.72–1.00 (28% of suspicion space)
  
  Allows independent tuning of each dimension to shift data between tiers:
    Example: Improve confidence from 0.5 to 0.8 (add DL1 context)
      → (1 - 0.8) instead of (1 - 0.5)
      → Reduces suspicion by 0.33×0.3 = 0.099
      → Can shift from WARN to LOG if other gates favorable

Data Clusters Emerge From Interactions:
  1. High-Confidence, Verified, Clean (conf≈1, verif≈1, anom≈0) → LOG tier
  2. High-Confidence, Unverified, Moderate Anomalies → WARN tier
  3. Low-Confidence, Unverified, Many Anomalies → BLOCK tier
  4. High-Confidence, Clean, but Low Verification (new source) → WARN tier
  
  The 3D geometry naturally creates clusters of similar risk profiles,
  improving interpretability vs. single-score systems.

Integration Points:
  • Parser Ingestion: Determine IMPORT behavior
  • Data Framework: Gate access to public electoral viewport
  • BallotLens: Filter/highlight results by confidence
  • Guarded Actions: Require confirmation for sensitive operations
  • Admin Dashboard: Monitor risk distribution across datasets
"""
