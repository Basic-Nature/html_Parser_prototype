# Confidence/Caution Entity Mapping Framework

**Date Created**: 2026-02-03  
**Status**: Design Phase (Ready for Implementation)  
**Motivation**: Nonpartisan, unbiased election integrity decision-making via transparent, weighted signal aggregation and principled override semantics.

---

## Executive Summary

This framework embeds a mathematical confidence/caution model into entity mapping and decision gates throughout the parser pipeline. Rather than binary pass/fail validation, entities are scored on a continuous scale with three decision outcomes:

- **PROCEED**: High confidence, low caution, low override → direct action
- **CAUTION**: Mixed signals → guarded action, user interaction, logging
- **STOP**: Low confidence, high caution, or high override → quarantine, manual review

The model is *nonpartisan* by design: signals are weighted based on data quality, source verification, and consistency (not political affiliation), and overrides are tracked transparently for audit.

---

## 1. Mathematical Model

### Decision Calculation

For any entity (office, party, jurisdiction, contest type, URL, data source, candidate):

```txt
confidence_score = Σ(signal_i × weight_i) / Σ(weight_i)
                   where signal_i ∈ [0, 1], weight_i ∈ [0, 1]

caution_score = Σ(anomaly_j × weight_j) / Σ(weight_j)
                where anomaly_j ∈ [0, 1], weight_j ∈ [0, 1]

override_score = Σ(override_trigger_k)
                 where override_trigger_k ∈ {+0.3, +0.2, +0.15}
```

### Decision Gates

| Condition | Decision | Action | Logging |
| ----------- | ---------- | -------- | --------- |
| `confidence ≥ 2/3 AND caution ≤ 1/3 AND override ≤ 1/3` | **PROCEED** | Full trust, direct action | INFO + decision_code |
| `confidence ∈ [1/3, 2/3) OR caution ∈ (1/3, 2/3]` | **CAUTION** | Guarded action, user prompt, extra logging | WARNING + decision_code + signals |
| `confidence < 1/3 OR caution > 2/3 OR override > 1/3` | **STOP** | Quarantine, manual review, escalation | ERROR + decision_code + override_reasons |

### Numerical Examples

***Example 1: Verified Government Office***

- Signal: exact match to verified registry (confidence = 0.99, weight = 1.0)
- Anomaly: none (caution = 0.0, weight = 0.0)
- Override: none (override = 0.0)
- **Result**: PROCEED ✅

***Example 2: Partial Match + One Anomaly***

- Signals: fuzzy match to office name (0.75), partial header match (0.80); weights = 0.6, 0.4
  - confidence = (0.75 × 0.6 + 0.80 × 0.4) / 1.0 = 0.77
- Anomaly: suspicious column count (caution = 0.40, weight = 0.5); other signals neutral (weight = 0.5)
  - caution = (0.40 × 0.5 + 0.0 × 0.5) / 1.0 = 0.20
- Override: none (override = 0.0)
- **Result**: PROCEED (confidence 0.77 ≥ 2/3, caution 0.20 ≤ 1/3) ✅

***Example 3: Low Match + High Anomaly + Admin Override***

- Signal: weak domain pattern match (confidence = 0.55, weight = 1.0)
- Anomaly: multiple inconsistencies flagged (caution = 0.65, weight = 1.0)
- Override: admin flag (+0.3), corrections from verified source (+0.2) → override = 0.5
- **Result**: STOP (override 0.5 > 1/3, caution 0.65 > 2/3) 🛑
  - But: Admin flagged as actionable → escalate to manual correction workflow with audit trail.

---

## 2. Signal Weighting Strategy

### Principle: Quality-Driven, Source-Aware Weighting

Signals are weighted by:

1. **Source Authority**: Verified > curated > inferred > heuristic
2. **Recency**: Fresh > aged > stale (refreshed weekly vs. quarterly)
3. **Coverage**: Full match > partial match > fuzzy match
4. **Consistency**: Multiple sources agree > single source > conflicting signals

### Signal Coefficient Catalog

#### Jurisdiction Signals (State/County/Precinct)

| Signal | Weight | Confidence | Notes |
| --------- | ------- | ----------- | ------------ |
| Exact match in FIPS registry | 1.0 | 0.99 | US Census Bureau official |
| Match in SoS county list | 0.9 | 0.98 | Secretary of State official |
| Match in alias mapping (curated) | 0.7 | 0.85 | Community-maintained aliases |
| Fuzzy match (Levenshtein ≥ 0.90) | 0.5 | 0.75 | Heuristic, typo-tolerant |
| Fuzzy match (Levenshtein 0.75–0.89) | 0.3 | 0.60 | Weak heuristic |
| Fuzzy match (Levenshtein < 0.75) | 0.1 | 0.30 | Not reliable alone |

#### Office Signals

| Signal | Weight | Confidence | Notes |
| --------- | ------- | ----------- | ------------ |
| Exact match to state election code | 1.0 | 0.99 | Official state statute |
| Match to common alias (Pres→President) | 0.8 | 0.90 | Well-established convention |
| Contextual match (ballot measure type) | 0.6 | 0.70 | Inferred from context |
| Header pattern match | 0.4 | 0.55 | HTML parsing heuristic |

#### Party Signals

| Signal | Weight | Confidence | Notes |
| --------- | ------- | ----------- | ------------ |
| Match to FEC official party list | 1.0 | 0.99 | Federal Election Commission |
| Match to state party list | 0.95 | 0.98 | State-level official |
| Match to common alias (Dem→Democratic) | 0.85 | 0.90 | Social convention |
| Match to write-in/independent pattern | 0.5 | 0.70 | Inferred from text pattern |
| Non-major party abbreviation | 0.3 | 0.50 | Uncertain domain |

#### Source/URL Signals

| Signal | Weight | Confidence | Notes |
| --------- | ------- | ----------- | ------------ |
| Verified government domain (SoS) | 1.0 | 0.99 | Whitelisted, trust bonus |
| Government domain pattern (.gov) | 0.9 | 0.95 | Pattern-based, verified via SSL |
| Known third-party aggregator | 0.7 | 0.85 | Curated source list |
| Suspicious TLD (.xyz, .loan) | 0.1 | 0.20 | High-risk indicator |

---

## 3. Override Variable Semantics

Override score increases when special flags are set. Each trigger contributes additively:

```txt
override_score = Σ(trigger_value)
```

### Override Triggers

| Trigger | Value | Condition | Audit Trail |
| --------- | ------- | ----------- | ------------ |
| Admin Correction Flag | +0.3 | An admin has manually marked this entity as "trusted" or "reject" | logged with admin ID, timestamp, reason |
| Verified Source Correction | +0.2 | Data matches a correction from verified/official source (e.g., county clerk) | logged with source, link, date |
| Anomaly Count Threshold | +0.15/item | Each anomaly beyond first (mismatched_totals, missing_candidate, header_mismatch) | logged per anomaly, row number |
| ML Model Flagged (Confidence < 0.5) | +0.15 | Sentence-Transformers or anomaly detection model returns low confidence | logged with model version, score |
| Contextual Mismatch (State/County/Contest) | +0.10 | Entity claimed for one state but data suggests another | logged with both states/counties |
| Multiple Corrections Same Entity | +0.10 | Entity has been corrected > 2 times in past 30 days | logged with correction count |

### Decision Rules with Overrides

1. **If override_score ≤ 1/3**: Ignore override; use standard gates.
2. **If 1/3 < override_score ≤ 2/3**: Escalate to manual review; user can force PROCEED with acknowledgment.
3. **If override_score > 2/3**: Force STOP; require admin unlock to proceed.

### Audit Requirements

Every override trigger must log:

- Entity ID + type (office, party, jurisdiction, etc.)
- Trigger type + value
- Context (handler, URL, session_id)
- User/source that initiated trigger (if applicable)
- Timestamp
- Reason (optional but encouraged)

---

## 4. Pipeline Integration Points

### A. URL Trust Scorer ([webapp/parser/utils/url_trust_scorer.py](../webapp/parser/utils/url_trust_scorer.py))

**Current State**: Binary thresholds (90–100 = direct, 70–89 = direct, 50–69 = snapshot, 30–49 = quarantine, 0–29 = reject).

**New State**: Replace thresholds with confidence/caution model.

- Signals: domain verification, pattern match, phishing detection, allowlist status
- Caution triggers: typosquat patterns, suspicious TLD, certificate age, SSL anomalies
- Override: verified source bonus (+0.2), user exemption flag (+0.3)
- **Decision gate**: PROCEED → direct navigation; CAUTION → DOM snapshot mode; STOP → reject

### B. Handler Selection ([webapp/parser/state_router.py](../webapp/parser/state_router.py))

**Current State**: Fuzzy match on state; pick highest-confidence handler.

**New State**: Weight handlers by confidence score derived from state/county match.

- Signals: exact state match (0.99), fuzzy state match (0.75), handler success history (0.8)
- Caution triggers: handler timeout history, missing dependencies, parse failures on similar data
- Override: user override via UI (CAUTION gate), fallback handler selection
- **Decision gate**: PROCEED → use recommended handler; CAUTION → offer alternatives; STOP → force manual

### C. Contest Selection Modal ([static/js/run_parser.js](../static/js/run_parser.js) + [webapp/parser/web_pipeline.py](../webapp/parser/web_pipeline.py))

**Current State**: User picks from list of contests.

**New State**: Annotate each contest with decision badge.

- Signals: match to expected contests for state/county, handler success rate
- Caution triggers: similar contest name but different type, missing verification
- Override: manual user selection (treated as admin flag)
- **Decision display**:
  - ✅ PROCEED: full name, confidence % (green)
  - ⚠️ CAUTION: name + warning (yellow), clickable for details
  - 🛑 STOP: grayed out, clickable to show reason (red)

### D. Anomaly Quarantine ([webapp/parser/Context_Integration/Integrity_check.py](../webapp/parser/Context_Integration/Integrity_check.py))

**Current State**: Binary flag (pass/fail).

**New State**: Decision gates with override handling.

- Signals: row count match to expected, header alignment, value consistency
- Caution triggers: minor discrepancies, partial matches
- Override: verified source correction, admin override
- **Decision gate**:
  - PROCEED → emit data without flags
  - CAUTION → emit data with quality warning
  - STOP → quarantine, log for manual review, optionally escalate to health task

### E. Data Ingestion ([webapp/parser/health/manual_correction_bot.py](../webapp/parser/health/manual_correction_bot.py))

**Current State**: Accept/reject corrections interactively.

**New State**: Gate corrections by override score.

- Signals: correction source authority, consistency with existing data
- Caution triggers: correction contradicts other sources
- Override: admin approval required if override_score > 1/3
- **Decision gate**:
  - PROCEED (override ≤ 1/3) → auto-apply correction
  - CAUTION (1/3 < override ≤ 2/3) → require user acknowledgment
  - STOP (override > 2/3) → require admin approval + audit entry

---

## 5. Implementation Phases

### Phase A: Scaffolding (Week 1)

1. Create `entity_confidence_map.py` with signal catalog + weighting functions.
2. Extend `shared_logic.py` with `safe_decide()` helper + decision tuple types.
3. Create `CONFIDENCE_CAUTION_FRAMEWORK.md` (this document).
4. Setup logging infrastructure for decision events (JSONL format).

### Phase B: Soft Launch (Week 2)

1. Implement decision logic in safe_* validators (log decisions, don't enforce gates).
2. Add decision badges to contest selection modal (UI enhancement).
3. Enable Prometheus metrics for decision counts + distribution.
4. Run for 1 week: measure impact, monitor anomalies, collect user feedback.

### Phase C: Gate Enforcement (Week 3+)

1. Enable decision gates in URL trust scorer (start with CAUTION prompts).
2. Enable gates in handler selection + anomaly quarantine (with override options).
3. Rollout per-handler based on confidence in handler-specific decision logic.
4. Monitor false positive rate; refine coefficients as needed.

---

## 6. Backward Compatibility Strategy

### Parallel API Approach

Create new `safe_decide_*` functions alongside existing validators:

```python
# Old API (binary)
result = safe_match_office(label, state)  # returns bool or str

# New API (decision-aware)
value, confidence, decision_code = safe_decide_office(label, state)
# decision_code ∈ {PROCEED, CAUTION, STOP}
```

### Gradual Migration

1. **Week 1–2**: Introduce new API; old API still in use; log warnings for deprecated calls.
2. **Week 3–4**: Migrate one handler at a time (start with highest-traffic handler).
3. **Week 5+**: Remove old API; all handlers use new API.

### Fallback Behavior

If new API not available, old API defaults to:

- `confidence ≥ 2/3` → return True (PROCEED)
- Otherwise → return False (CAUTION/STOP treated as fail)

---

## 7. Nonpartisan Design Principles

To ensure unbiased, nonpartisan decision-making:

1. **No political signal weighting**: Never weight signals based on party affiliation, geography, or demographic factors.
2. **Data-driven coefficients**: Weights derived from:
   - Source authority (e.g., FIPS registry vs. Wikipedia)
   - Verification consistency (e.g., multiple sources agree)
   - Historical accuracy (e.g., handler success rate on similar data)
3. **Transparent override triggers**: All overrides logged with reason + audit trail.
4. **Symmetric treatment**: Same rules apply to all states, parties, counties regardless of size or politics.
5. **Public coefficient review**: Signal weights documented + subject to community review (GitHub issues).
6. **Regular audits**: Monthly audit reports on decision distribution by state/county/party to detect bias.

---

## 8. Monitoring & Metrics

### Prometheus Counters

```txt
decision_proceed_total{entity_type, handler, state}
decision_caution_total{entity_type, handler, state, caution_reason}
decision_stop_total{entity_type, handler, state, override_trigger}

confidence_score_histogram{entity_type, bucket}
caution_score_histogram{entity_type, bucket}
override_score_histogram{trigger_type, bucket}
```

### Audit Dashboard (Grafana)

- **Decision Flow**: % PROCEED vs. CAUTION vs. STOP over time
- **Signal Distribution**: Which signals most commonly trigger CAUTION/STOP?
- **Override Trends**: Which overrides most frequently used?
- **Bias Detection**: Decision rate by state/county/party (should be uniform)

### Health Task Integration

Weekly task: `integrity_check_runner` generates report on decision anomalies (e.g., "Arizona contests have 2× STOP rate; investigate").

---

## 9. Questions for Review

1. **Signal Coefficients**: Are the proposed weights reasonable for your use case? Should we adjust jurisdiction weights (e.g., favor FIPS more)?
2. **Override Thresholds**: Is 1/3 the right threshold for escalation? Should override > 2/3 require multi-level approval?
3. **Anomaly Weighting**: Should multiple anomalies on same entity compound (product rule) or sum (linear)?
4. **Rollout Timeline**: Can we commit to 1-week soft-launch period before full gate enforcement?
5. **Audit Access**: Who should have access to override audit logs? (suggest: election officials + internal team only)

---

**Next Steps**: Review this document with election integrity stakeholders; finalize signal coefficients; begin Phase A implementation.
