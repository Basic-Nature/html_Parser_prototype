"""
THREE-DIMENSIONAL RISK GATES – QUICK START GUIDE

Smart Elections Parser Implementation
Date: 2026-02-05

═════════════════════════════════════════════════════════════════════════════
WHAT IS THIS?
═════════════════════════════════════════════════════════════════════════════

A new risk assessment system that replaces single-score thresholds with
three independent risk dimensions:

  1. CONFIDENCE GATE (0–1): Parser certainty in extraction
  2. VERIFICATION GATE (0–1): Ground truth (DL1) alignment
  3. ANOMALY GATE (0–1): Statistical suspicion from patterns/outliers

Combined → COMPOSITE SUSPICION (0–1) → Classified into three RISK TIERS:

  🟢 LOG (0.00–0.45): Auto-process, no confirmation needed
  🟡 WARN (0.45–0.72): Require user/admin confirmation
  🔴 BLOCK (0.72–1.00): Escalate to human review

═════════════════════════════════════════════════════════════════════════════
FILES & STRUCTURE
═════════════════════════════════════════════════════════════════════════════

📁 webapp/parser/health/
├── risk_gates.py                           [Core Implementation]
│   ├── RiskGateConfig (dataclass)
│   ├── RiskGateScores (dataclass)
│   ├── RiskGateEvaluator (class)
│   │   ├── compute_confidence_gate()
│   │   ├── compute_verification_gate()
│   │   ├── compute_anomaly_gate()
│   │   ├── compute_composite_suspicion()
│   │   ├── classify_risk_tier()
│   │   └── evaluate()  [main entry point]
│   └── evaluate_risk()  [quick function]
│
├── health_config.py                        [Configuration]
│   ├── RISK_GATES_CONFIG (dict)
│   │   ├── weight_confidence: 0.33
│   │   ├── weight_verification: 0.33
│   │   ├── weight_anomaly: 0.34
│   │   ├── tier_boundary_warn_log: 0.45
│   │   └── tier_boundary_block_warn: 0.72
│   └── Legacy thresholds (marked deprecated)
│
├── risk_gates_integration_examples.py       [Usage Examples]
│   ├── evaluate_parser_extraction()
│   ├── evaluate_data_framework_upload()
│   ├── evaluate_ballot_lens_display()
│   ├── evaluate_guarded_action()
│   └── summarize_risk_distribution()
│
└── risk_gates_spec.py                      [Technical Documentation]
    └── Full specification with theory, clusters, integration points

📄 ALGORITHMIC_APPROACH_SUMMARY.md          [Your Vision → Implementation]
    └── How the three-gate model answers your algorithmic request

═════════════════════════════════════════════════════════════════════════════
BASIC USAGE
═════════════════════════════════════════════════════════════════════════════

1. QUICKEST: Use the convenience function

   from webapp.parser.health.risk_gates import evaluate_risk

   scores = evaluate_risk(
       extraction_confidence=0.87,          # Parser reported 87% confidence
       ground_truth_matches=42,             # 42 rows match DL1
       total_records=48,                    # Total rows extracted
       suspicious_pattern_count=0,          # No suspicious keywords
       outlier_record_count=1               # 1 statistical outlier
   )

   print(f"Risk tier: {scores.risk_tier}")           # Output: "log"
   print(f"Suspicion: {scores.composite_suspicion:.3f}")  # Output: 0.091
   print(f"Action: AUTO-IMPORT")

2. NORMAL: Customize evaluator with different config

   from webapp.parser.health.health_config import RISK_GATES_CONFIG
   from webapp.parser.health.risk_gates import RiskGateEvaluator, RiskGateConfig

   # Use default config from health_config.py

   evaluator = RiskGateEvaluator()

   # Or customize weights (e.g., emphasize verification more)

   custom_config = RiskGateConfig(
       weight_confidence=0.25,
       weight_verification=0.50,  # Boosted for critical ground truth
       weight_anomaly=0.25,
       tier_boundary_warn_log=0.45,
       tier_boundary_block_warn=0.72
   )
   strict_evaluator = RiskGateEvaluator(config=custom_config)

   scores = evaluator.evaluate(
       extraction_confidence=0.87,
       ground_truth_matches=42,
       total_records=48,
       suspicious_pattern_count=0,
       outlier_record_count=1
   )

3. ADVANCED: Manual gate computation

   from webapp.parser.health.risk_gates import RiskGateEvaluator

   evaluator = RiskGateEvaluator()

   # Compute each gate independently

   conf_gate = evaluator.compute_confidence_gate(0.87)
   verif_gate = evaluator.compute_verification_gate(42, 48)
   anom_gate = evaluator.compute_anomaly_gate(0, 1, 48)

   # Combine into suspicion

   suspicion = evaluator.compute_composite_suspicion(conf_gate, verif_gate, anom_gate)

   # Classify tier

   tier, tier_confidence = evaluator.classify_risk_tier(suspicion)

   print(f"Tier: {tier}, Confidence: {tier_confidence:.2f}")

═════════════════════════════════════════════════════════════════════════════
INTEGRATION CHECKLIST
═════════════════════════════════════════════════════════════════════════════

[ ] 1. Parser Ingestion (webapp/parser/html_election_parser.py)
       • After parsing, collect: extraction_confidence, DL1 match counts, anomalies
       • Call evaluate_risk()
       • Store RiskGateScores in metadata
       • Dispatch action:
         - LOG: Insert into database, log audit entry
         - WARN: Prompt user "Confirm this data?" → if yes, insert with flag
         - BLOCK: Refuse import, escalate to admin review queue

[ ] 2. Data Framework Upload (webapp/Smart_Elections_Parser_Webapp.py)
       • Gate upload by risk_tier
       • Allow "log" tier immediately
       • Require confirmation for "warn" tier
       • Return 403 Forbidden for "block" tier (unless guarded key)

[ ] 3. BallotLens Visibility (webapp/static/js/ballot_lens_modern.js)
       • Add risk_tier to result metadata
       • Filter based on confidence level:
         - "High confidence only" → show "log" tier only
         - "All results" → show all tiers
       • Style badges: green (log), yellow (warn), red/hidden (block)

[ ] 4. Guarded Action Gates (webapp/Smart_Elections_Parser_Webapp.py)
       • Identify sensitive operations (add URL, upload high-risk data, etc.)
       • Check GUARDED_INGESTION_KEY env var
       • If not present and tier >= WARN: prompt for local password
       • Log all confirmations to audit trail

[ ] 5. Health Monitoring (webapp/parser/health/health_router.py)
       • Export risk distribution to admin dashboard
       • Track trends: % of LOG/WARN/BLOCK over time
       • Alert if BLOCK tier count spikes
       • Quarterly review: suggest threshold tuning

═════════════════════════════════════════════════════════════════════════════
EXAMPLE SCENARIOS
═════════════════════════════════════════════════════════════════════════════

SCENARIO A: High-Quality Extraction
  Inputs:
    - extraction_confidence: 0.92
    - ground_truth_matches: 198 / 200
    - outliers: 0
    - suspicious_patterns: 0

  Gate Values:
    - confidence_gate: 0.92 → (1 - 0.92) = 0.08 contribution
    - verification_gate: 0.99 → (1 - 0.99) = 0.01 contribution
    - anomaly_gate: 0.00 → 0.00 contribution

  Composite Suspicion:
    = 0.33 × 0.08 + 0.33 × 0.01 + 0.34 × 0.00 ≈ 0.030

  Result: TIER LOG (deep, tier_confidence ≈ 0.93)
  Action: AUTO-IMPORT (no confirmation needed)

SCENARIO B: Medium-Risk Extraction
  Inputs:
    - extraction_confidence: 0.72
    - ground_truth_matches: 75 / 120
    - outliers: 8
    - suspicious_patterns: 2

  Gate Values:
    - confidence_gate: 0.72 → 0.28 contribution
    - verification_gate: 0.625 → 0.375 contribution
    - anomaly_gate: 0.087 → 0.087 contribution (8% outliers + 2% patterns)

  Composite Suspicion:
    = 0.33 × 0.28 + 0.33 × 0.375 + 0.34 × 0.087 ≈ 0.253

  Result: TIER LOG (at upper boundary, tier_confidence ≈ 0.94)
  Action: AUTO-IMPORT but monitor

SCENARIO C: High-Risk Extraction
  Inputs:
    - extraction_confidence: 0.45
    - ground_truth_matches: 20 / 100
    - outliers: 25
    - suspicious_patterns: 10
    - integrity_flags: ["test_data", "duplicate_entries"]

  Gate Values:
    - confidence_gate: 0.45 → 0.55 contribution
    - verification_gate: 0.20 → 0.80 contribution
    - anomaly_gate: 0.35 (25% outliers + 10% patterns + 0.2 flag boost) → 0.35 contribution

  Composite Suspicion:
    = 0.33 × 0.55 + 0.33 × 0.80 + 0.34 × 0.35 ≈ 0.553

  Result: TIER WARN (center of tier, tier_confidence ≈ 0.73)
  Action: REQUIRE CONFIRMATION (user must review and approve)

SCENARIO D: Unacceptable Extraction
  Inputs:
    - extraction_confidence: 0.35
    - ground_truth_matches: 5 / 100
    - outliers: 50
    - suspicious_patterns: 20
    - integrity_flags: ["massive_anomalies", "source_unverified"]

  Gate Values:
    - confidence_gate: 0.35 → 0.65 contribution
    - verification_gate: 0.05 → 0.95 contribution
    - anomaly_gate: 0.65 (50% outliers + 20% patterns capped + flag boost) → 0.65 contribution

  Composite Suspicion:
    = 0.33 × 0.65 + 0.33 × 0.95 + 0.34 × 0.65 ≈ 0.756

  Result: TIER BLOCK (deep, tier_confidence ≈ 0.76)
  Action: ESCALATE (refuse import, escalate to admin review)
       Requires guarded key + admin approval to override

═════════════════════════════════════════════════════════════════════════════
CONFIGURATION TUNING
═════════════════════════════════════════════════════════════════════════════

Located in: webapp/parser/health/health_config.py (lines 113–130)

Default Config:
  weight_confidence: 0.33     # Parser certainty (1/3 weight)
  weight_verification: 0.33   # DL1 alignment (1/3 weight)
  weight_anomaly: 0.34        # Pattern/outlier suspension (1/3 weight)

  tier_boundary_warn_log: 0.45    # Threshold: LOG < 0.45 ≤ WARN
  tier_boundary_block_warn: 0.72  # Threshold: WARN < 0.72 ≤ BLOCK

Tuning Strategy:

  • More permissive (auto-import more data):
    Increase tier_boundary_block_warn from 0.72 → 0.85
    Effect: WARN tier becomes 0.45–0.85 (broader), BLOCK tier 0.85–1.0 (narrower)
    Use when: You trust parser confidence and want fewer escalations

  • More strict (require more confirmations):
    Decrease tier_boundary_warn_log from 0.45 → 0.35
    Effect: LOG tier becomes 0–0.35 (narrower), WARN tier 0.35–0.72 (broader)
    Use when: DL1 verification is critical, or you're testing new sources

  • Emphasize verification over parser confidence:
    weight_confidence: 0.25
    weight_verification: 0.50
    weight_anomaly: 0.25
    Effect: Low DL1 alignment adds more suspicion
    Use when: Ground truth is most important

  • Emphasize anomaly detection:
    weight_confidence: 0.25
    weight_verification: 0.25
    weight_anomaly: 0.50
    Effect: High outlier/pattern counts add more suspicion
    Use when: Statistical integrity is critical

═════════════════════════════════════════════════════════════════════════════
AUDIT & MONITORING
═════════════════════════════════════════════════════════════════════════════

Every evaluation is logged to: webapp/parser/log/integrity_monitor.jsonl

Sample log entry:
  {
    "timestamp": "2026-02-05T14:32:10Z",
    "contest_id": "CA_2024_SENATE",
    "state": "CA",
    "year": 2024,
    "confidence_gate": 0.87,
    "verification_gate": 0.875,
    "anomaly_gate": 0.02,
    "composite_suspicion": 0.091,
    "risk_tier": "log",
    "tier_confidence": 0.80,
    "action_taken": "IMPORT_AUTO",
    "user_id": "SYSTEM",
    "audit_notes": "California 2024 Senate: 48 candidates extracted with 87.5% DL1 verification."
  }

Admin Dashboard (TBD):
  • View distribution of LOG/WARN/BLOCK tiers
  • Track trends over time (improving or degrading quality?)
  • Identify datasets with systematic issues
  • Review guarded key confirmations
  • Quarterly threshold tuning recommendations

═════════════════════════════════════════════════════════════════════════════
TROUBLESHOOTING
═════════════════════════════════════════════════════════════════════════════

Q: Data keeps falling into WARN tier; how do I lower it?
A: You have several options:

   1. Reduce tier_boundary_block_warn (e.g., 0.72 → 0.65) → more WARN, less LOG
   2. Boost verification by adding more DL1 data (improves verification_gate)
   3. Reduce anomaly detection sensitivity (fewer outliers flagged)
   4. Lower weight_verification if DL1 matching is giving false negatives

Q: Some high-quality data is incorrectly classified as BLOCK
A: Check these:

   1. Is anomaly_gate inflated? (Check suspicious_pattern_count and outliers)
   2. Is extraction_confidence actually low? (Verify parser output)
   3. Is ground_truth_matches count correct? (Ensure DL1 joining logic is right)
   4. Consider lowering tier_boundary_block_warn (0.72 → 0.80) to be less strict

Q: How do I know if my tuning is working?
A: Monitor the audit trail:
   • Count results in each tier (target: 70% LOG, 20% WARN, 10% BLOCK)
   • Track false positives: How many WARN/BLOCK actually contained good data?
   • Survey operators: Are confirmations helping catch real issues?
   • Review escalations: Are BLOCK items actually problematic, or false alarms?

Q: Can I have different thresholds for different states or contests?
A: Yes! You can subclass RiskGateConfig:

   from webapp.parser.health.risk_gates import RiskGateConfig, RiskGateEvaluator

# Stricter for CA (complex source)

   ca_config = RiskGateConfig(
       weight_verification=0.50,  # Emphasize DL1
       tier_boundary_warn_log=0.40  # Tighter LOG tier
   )
   ca_evaluator = RiskGateEvaluator(config=ca_config)

# More permissive for smaller states

   small_state_config = RiskGateConfig(
       tier_boundary_block_warn=0.80  # Looser BLOCK tier
   )
   small_state_evaluator = RiskGateEvaluator(config=small_state_config)

═════════════════════════════════════════════════════════════════════════════
NEXT STEPS IN ROADMAP
═════════════════════════════════════════════════════════════════════════════

1. IMMEDIATE (This week)
   ☐ Review risk_gates.py implementation
   ☐ Verify cluster analysis matches your expectations
   ☐ Customize tier boundaries for your risk tolerance

2. SHORT-TERM (Next 1-2 weeks)
   ☐ Integrate evaluate_risk() into html_election_parser.py
   ☐ Implement guarded_key enforcement in Smart_Elections_Parser_Webapp.py
   ☐ Add risk tier metadata to Data Framework uploads

3. MEDIUM-TERM (Next month)
   ☐ Wire Data Framework to display risk badges
   ☐ Integrate BallotLens filtering by risk tier
   ☐ Set up admin dashboard with trend monitoring
   ☐ Finalize guarded key rotation and Azure secrets setup

4. LONG-TERM (Future)
   ☐ Live-cycle parsing with auto-refresh
   ☐ State/county dynamic browsing with risk aggregation
   ☐ ML-based anomaly gate tuning
   ☐ Quarterly threshold optimization based on audit data

═════════════════════════════════════════════════════════════════════════════
REFERENCES & DOCUMENTATION
═════════════════════════════════════════════════════════════════════════════

📖 Full Technical Spec:
   webapp/parser/health/risk_gates_spec.py (540 lines)
   → Theory, clusters, integration architecture, pseudocode

📖 Algorithmic Vision Explained:
   ALGORITHMIC_APPROACH_SUMMARY.md (top-level directory)
   → How your "third more/less" idea became the 3-gate model

📖 Code Examples:
   webapp/parser/health/risk_gates_integration_examples.py (360 lines)
   → 5 practical scenarios with complete code

📖 Configuration:
   webapp/parser/health/health_config.py (lines 113–130)
   → All tunable parameters in one place

═════════════════════════════════════════════════════════════════════════════
"""
