"""
TECHNICAL SPECIFICATION: Three-Dimensional Risk Assessment Model

Smart Elections Parser – Risk Gate Architecture
Version 1.0
Date: 2026-02-05

==============================================================================
EXECUTIVE SUMMARY
==============================================================================

The Smart Elections Parser now uses a three-dimensional risk vector to assess
data integrity, replacing single-score thresholds. This model combines:

  1. CONFIDENCE GATE: Parser's extraction certainty (0.0 – 1.0)
  2. VERIFICATION GATE: Ground truth alignment with DL1 (0.0 – 1.0)
  3. ANOMALY GATE: Statistical suspension from patterns/outliers (0.0 – 1.0)

These three independent dimensions are weighted and combined into a single
COMPOSITE SUSPICION score (0.0 – 1.0), which is then classified into one
of three RISK TIERS using ⅓-proportioned boundaries:

  • LOG (0.00 – 0.45): Auto-process, audit-only logging
  • WARN (0.45 – 0.72): User confirmation required
  • BLOCK (0.72 – 1.00): Escalate to admin review

The ⅓-partition ensures data clusters naturally separate by risk profile,
improving interpretability and enabling proportional enforcement of access
controls (guarded key gates, UI visibility, approval workflows).

==============================================================================
MATHEMATICAL FOUNDATION
==============================================================================

1. GATE NORMALIZATION
─────────────────────

Each dimension is normalized to [0, 1] with contextual interpretation:

  confidence_gate = CLAMP(extraction_confidence, [0, 1])
    Interpretation: 1.0 = parser is certain, 0.0 = parser is guessing
    Usage in suspicion: (1 - confidence_gate)  [inverted reasoning]
  
  verification_gate = CLAMP(ground_truth_matches / total_records, [0, 1])
    Interpretation: 1.0 = 100% verified, 0.0 = 0% verified
    Usage in suspicion: (1 - verification_gate)  [inverted reasoning]
  
  anomaly_gate = blend(pattern_ratio, outlier_ratio, integrity_flags)
    Interpretation: 0.0 = clean data, 1.0 = highly suspicious
    Usage in suspicion: anomaly_gate  [direct reasoning]
    Formula: anomaly_gate = w_pattern × (patterns/total) + w_outlier × (outliers/total)
             where w_pattern = 0.40, w_outlier = 0.60, plus 0.1 per integrity flag (cap 0.3)

2. SUSPICION COMPUTATION (WEIGHTED VECTOR)
───────────────────────────────────────────

Composite Suspicion combines the three gates via weighted sum:

  SUSPICION = w₁(1 - confidence) + w₂(1 - verification) + w₃(anomaly)
  
  where:
    w₁ = weight_confidence    = 0.33
    w₂ = weight_verification = 0.33
    w₃ = weight_anomaly       = 0.34
    ────────────────────────────────
      Σ = 1.00
  
Intuition:
  • Parser low on confidence (conf≈0.5) → (1-0.5) = 0.5 → contributes 0.33×0.5 ≈ 0.17 to suspicion
  • Data not verified (verif≈0.3) → (1-0.3) = 0.7 → contributes 0.33×0.7 ≈ 0.23 to suspicion
  • Many anomalies (anom≈0.6) → 0.6 → contributes 0.34×0.6 ≈ 0.20 to suspicion
  • TOTAL SUSPICION ≈ 0.60 → TIER: WARN

3. TIER CLASSIFICATION (⅓-PARTITIONED BOUNDARIES)
──────────────────────────────────────────────────

Suspicion score is mapped to discrete risk tier using proportional thirds:

  Tier Width Analysis:
    Lower Third:  0.00 – 0.45  (450 basis points, ~45% of space)
    Middle Third: 0.45 – 0.72  (270 basis points, ~27% of space)
    Upper Third:  0.72 – 1.00  (280 basis points, ~28% of space)
  
  Asymmetry Rationale:
    - Lower third (LOG) is broader: most data should auto-process
    - Middle third (WARN) is narrower: user confirmation gates
    - Upper third (BLOCK) is narrower: escalation is uncommon
    
  Tier Confidence (Distance to Boundary):
    Within each tier, compute % distance to nearest boundary.
    Examples:
      - suspicion = 0.20 in LOG tier
        → tier_confidence = 1 - (0.20 / 0.45) ≈ 0.56  (56% deep into LOG)
      - suspicion = 0.60 in WARN tier
        → tier_confidence = (0.60 - 0.45) / (0.72 - 0.45) ≈ 0.56  (56% toward BLOCK boundary)
      - suspicion = 0.85 in BLOCK tier
        → tier_confidence = (0.85 - 0.72) / (1 - 0.72) ≈ 0.46  (46% deep into BLOCK)

==============================================================================
DATA CLUSTER ANALYSIS
==============================================================================

The three-gate model naturally produces clusters of similar risk profiles,
improving interpretability. Common clusters:

Cluster 1: HIGH-CONFIDENCE, VERIFIED, CLEAN
  conf=0.95, verif=0.92, anom=0.05
  → suspicion = 0.33(0.05) + 0.33(0.08) + 0.34(0.05) ≈ 0.058
  → TIER: LOG (deep, tier_conf ≈ 0.87)
  Interpretation: Ground truth matches, parser certain, no anomalies
  Action: Auto-import, audit entry only

Cluster 2: HIGH-CONFIDENCE, UNVERIFIED, CLEAN
  conf=0.90, verif=0.40, anom=0.10
  → suspicion = 0.33(0.10) + 0.33(0.60) + 0.34(0.10) ≈ 0.290
  → TIER: LOG (at top boundary, tier_conf ≈ 0.98)
  Interpretation: Parser is certain, but lacking DL1 context (new source)
  Action: Auto-import, but monitor for future mismatches

Cluster 3: MEDIUM-CONFIDENCE, PARTIALLY-VERIFIED, MODERATE-ANOMALIES
  conf=0.72, verif=0.65, anom=0.25
  → suspicion = 0.33(0.28) + 0.33(0.35) + 0.34(0.25) ≈ 0.293
  → TIER: LOG (near boundary)
  Interpretation: Mixed signals; data quality acceptable but not ideal
  Action: Auto-import, but flag for review

Cluster 4: MEDIUM-CONFIDENCE, LOW-VERIFICATION, HIGH-ANOMALIES
  conf=0.65, verif=0.30, anom=0.45
  → suspicion = 0.33(0.35) + 0.33(0.70) + 0.34(0.45) ≈ 0.504
  → TIER: WARN (center of warn, tier_conf ≈ 0.76)
  Interpretation: Multiple risk factors present; require confirmation
  Action: Prompt user for explicit confirmation before import

Cluster 5: LOW-CONFIDENCE, LOW-VERIFICATION, MANY-ANOMALIES
  conf=0.45, verif=0.15, anom=0.65
  → suspicion = 0.33(0.55) + 0.33(0.85) + 0.34(0.65) ≈ 0.689
  → TIER: WARN (near BLOCK boundary, tier_conf ≈ 0.99)
  Interpretation: Multiple severe issues; escalation recommended
  Action: Require confirmation; log as high-risk

Cluster 6: CONSISTENTLY-POOR ACROSS ALL DIMENSIONS
  conf=0.30, verif=0.05, anom=0.80
  → suspicion = 0.33(0.70) + 0.33(0.95) + 0.34(0.80) ≈ 0.816
  → TIER: BLOCK (deep, tier_conf ≈ 0.83)
  Interpretation: Systematic quality failure; likely data or source error
  Action: Refuse import; escalate to admin review

==============================================================================
INTEGRATION ARCHITECTURE
==============================================================================

1. PARSER WORKFLOW
─────────────────

Parse HTML/PDF Election Document
    ↓
[Extract candidates, votes, metadata]
    ↓
Confidence Assessment (parser confidence → confidence_gate)
    ↓
Compare Against DL1 Ground Truth (match % → verification_gate)
    ↓
Anomaly Detection (patterns, outliers, flags → anomaly_gate)
    ↓
Evaluate Risk (combine three gates → composite_suspicion → tier)
    ↓
ACTION DISPATCH:
  • LOG tier → IMPORT_AUTO
    - Insert into dl2.election_results
    - Log decision to audit trail
    - No user interaction needed
  
  • WARN tier → IMPORT_CONFIRM
    - Prompt user: "High-risk data. Review and confirm?"
    - On confirm: insert with "user_confirmed" flag in metadata
    - On reject: skip import, log reason
  
  • BLOCK tier → IMPORT_ESCALATE
    - Refuse to import
    - Escalate to admin review queue
    - Require guarded key + manual approval for override

2. DATA FRAMEWORK GATING
────────────────────────

User attempts to upload parsed results to public Data Framework
    ↓
Query previous risk_tier from parser metadata
    ↓
If tier == LOG:
    → Allow direct upload
    → Publish to Data Framework viewport
  
If tier == WARN:
    → Show confirmation dialog: "Medium-risk data. Publish to web view?"
    → On confirm: upload with confidence badge (yellow)
    → On reject: save to user's private workspace only
  
If tier == BLOCK:
    → Refuse publication
    → Offer: "Save privately for review" or "Escalate for admin approval"
    → Only admins can force-publish via guarded key

3. BALLOTLENS VISIBILITY & FILTERING
─────────────────────────────────────

Render election results in BallotLens UI
    ↓
For each result row in viewport:
    ↓
Fetch risk_tier from metadata
    ↓
Apply display rule:
  
  tier == LOG:
    • Display normally (green badge)
    • Include in all filter modes
    • Tooltip: "High confidence, verified"
  
  tier == WARN:
    • Display normally (yellow badge)
    • Include in "Show all" and "Medium & above" filters
    • Exclude from "High confidence only" filter
    • Tooltip: "⚠️ Verify against official results"
  
  tier == BLOCK:
    • Hidden by default (red badge, collapsed)
    • Require explicit click to expand
    • Include in "Show all" filter only
    • Tooltip: "🚫 Do not use; requires verification"

4. GUARDED ACTION GATES
──────────────────────

Sensitive operations (add URL, upload data, run parser on unverified source)
    ↓
Assess operation's inherent risk_tier (context-specific)
    ↓
If tier >= WARN:
    → Check environment variable (GUARDED_INGESTION_KEY)
    → If present: allowed with confirmation
    → If absent: prompt for local terminal password
    → Guarded key gates: GitHub secret (CI) or local hash (development)
  
If tier == BLOCK:
    → Always require guarded key (no fallback to auto-allow)
    → Log all confirmations to audit trail
    → Admins review guarded_key usage periodically

==============================================================================
CONFIGURATION & CUSTOMIZATION
==============================================================================

1. ADJUSTING GATE WEIGHTS
──────────────────────────

Edit in health_config.py:

  RISK_GATES_CONFIG = {
      "weight_confidence": 0.33,     # Increase if parser confidence is unreliable
      "weight_verification": 0.33,   # Increase if ground truth is critical
      "weight_anomaly": 0.34,        # Increase if anomaly detection is strong
      ...
  }

Example: If DL1 verification is most important:
  weight_confidence: 0.20
  weight_verification: 0.50  # Boosted
  weight_anomaly: 0.30
  
  Effect: Low verification (e.g., 0.5) now contributes 0.5×(1-0.5) = 0.25 to suspicion,
          vs. 0.33×0.5 = 0.165 in default config. Tightens gating.

2. ADJUSTING TIER BOUNDARIES
──────────────────────────────

Edit in health_config.py:

  RISK_GATES_CONFIG = {
      "tier_boundary_warn_log": 0.45,      # Move up to make WARN tier narrower
      "tier_boundary_block_warn": 0.72,    # Move down to make BLOCK tier broader
      ...
  }

Example: If blocking is too aggressive, soften boundaries:
  tier_boundary_warn_log: 0.55   # Make LOG tier broader (0–0.55)
  tier_boundary_block_warn: 0.85 # Make WARN tier broader (0.55–0.85), BLOCK narrower (0.85–1.0)
  
  Effect: More data falls into LOG tier; WARN tier is broader and more forgiving.

3. ANOMALY GATE SUB-WEIGHTS
────────────────────────────

Edit in health_config.py:

  RISK_GATES_CONFIG = {
      "anomaly_pattern_weight": 0.4,     # Suspicious keywords matter less
      "anomaly_outlier_weight": 0.6,     # Statistical outliers matter more
      ...
  }

Default: Patterns (40%) + Outliers (60%)
  → If domain knowledge suggests keywords are false positives, reduce to: 0.2 + 0.8
  → If outliers are expected in some contexts, adjust to: 0.6 + 0.4

==============================================================================
DEPLOYMENT & AUDITING
==============================================================================

1. AUDIT TRAIL
──────────────

Every risk evaluation is logged to integrity_monitor.jsonl with:
  - timestamp
  - contest_id / source_dataset
  - confidence_gate, verification_gate, anomaly_gate (raw scores)
  - composite_suspicion (final score)
  - risk_tier (classification)
  - action_taken (IMPORT_AUTO, IMPORT_CONFIRM, IMPORT_ESCALATE, etc.)
  - user_id (if human confirmation) or "SYSTEM" (if auto)
  - audit_notes (context for decision)

2. ADMIN DASHBOARD
───────────────────

Displays risk distribution:
  - % of contests in each tier (LOG, WARN, BLOCK)
  - Tier breakdown by state/year/race
  - Trend analysis (improving or degrading quality)
  - Top anomaly patterns requiring human attention
  - Guarded key usage history (when sensitive actions confirmed)

3. CONTINUOUS IMPROVEMENT
──────────────────────────

Monitor audit trail to identify:
  - Dead zones: Data clusters where tier classification is inaccurate
  - Weight imbalances: One gate consistently dominates suspicion
  - Threshold drift: Tiers shift over time (suggest retuning)
  - Operator feedback: Admins flag false positives/negatives for analysis

Adjust RISK_GATES_CONFIG periodically based on audit insights.

==============================================================================
APPENDIX: ALGORITHM PSEUDOCODE
==============================================================================

Algorithm: evaluate_risk(extraction_confidence, ground_truth_matches, total_records, ...)
Input:
  - extraction_confidence: float ∈ [0, 1]
  - ground_truth_matches: int (count of rows matching DL1)
  - total_records: int (total extracted rows)
  - suspicious_pattern_count: int
  - outlier_record_count: int
  - integrity_flags: list[str]
  
Output:
  - RiskGateScores object with gates, suspicion, tier, tier_confidence

Steps:
  1. Normalize confidence gate
     conf_gate ← CLAMP(extraction_confidence, 0, 1)
  
  2. Compute verification gate
     IF total_records == 0 AND fallback_score EXISTS THEN
       verif_gate ← fallback_score
     ELSE IF total_records == 0 THEN
       verif_gate ← 0.0
     ELSE
       verif_gate ← CLAMP(ground_truth_matches / total_records, 0, 1)
     END IF
  
  3. Compute anomaly gate
     pattern_ratio ← CLAMP(suspicious_pattern_count / total_records, 0, 1)
     outlier_ratio ← CLAMP(outlier_record_count / total_records, 0, 1)
     blended ← 0.40 × pattern_ratio + 0.60 × outlier_ratio
     IF LENGTH(integrity_flags) > 0 THEN
       flag_boost ← MIN(0.3, LENGTH(integrity_flags) × 0.1)
       blended ← blended + flag_boost
     END IF
     anom_gate ← CLAMP(blended, 0, 1)
  
  4. Compute composite suspicion
     suspicion ← 0.33 × (1 - conf_gate) +
                 0.33 × (1 - verif_gate) +
                 0.34 × anom_gate
     suspicion ← CLAMP(suspicion, 0, 1)
  
  5. Classify tier
     IF suspicion >= 0.72 THEN
       tier ← "BLOCK"
       tier_confidence ← (suspicion - 0.72) / (1.0 - 0.72)
     ELSE IF suspicion >= 0.45 THEN
       tier ← "WARN"
       tier_confidence ← (suspicion - 0.45) / (0.72 - 0.45)
     ELSE
       tier ← "LOG"
       tier_confidence ← 1 - (suspicion / 0.45)
     END IF
  
  6. Return RiskGateScores(
       confidence_gate=conf_gate,
       verification_gate=verif_gate,
       anomaly_gate=anom_gate,
       composite_suspicion=suspicion,
       risk_tier=tier,
       tier_confidence=tier_confidence
     )

==============================================================================
REFERENCES
==============================================================================

Code Location:
  • risk_gates.py: Core RiskGateEvaluator class + helper functions
  • health_config.py: RISK_GATES_CONFIG constants
  • risk_gates_integration_examples.py: 5 practical integration examples

Configuration:
  • health_config.py: RISK_GATES_CONFIG dict with all tunable parameters

Documentation:
  • This file (risk_gates_spec.md): Complete technical specification
  
Integration Points (TBD):
  • webapp/parser/html_election_parser.py: Call evaluate_risk() after parsing
  • webapp/templates/data_framework.html: Gate uploads by risk tier
  • webapp/static/js/ballot_lens_modern.js: Filter/highlight by tier
  • webapp/Smart_Elections_Parser_Webapp.py: Guarded key confirmation flow

==============================================================================
END OF SPECIFICATION
==============================================================================
"""
