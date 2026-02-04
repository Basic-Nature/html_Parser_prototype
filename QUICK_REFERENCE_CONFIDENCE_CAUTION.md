# Quick Reference: Confidence/Caution Framework

## TL;DR

**What:** Decision gates for election entities (offices, parties, jurisdictions, sources)  
**Why:** Nonpartisan, data-driven trust scoring; audit all decisions  
**How:** Call `safe_decide_*()` → get `DecisionTuple` → act on decision_code ∈ {proceed, caution, stop}

---

## Decision Codes

| Code | Meaning | Action | Frequency (Week 2 Soft-Launch) |
| --- | --- | --- | --- |
| **proceed** ✅ | High confidence, low caution, low override | Accept & use canonical | ~70% |
| **caution** ⚠️ | Mixed signals | Accept with warning badge | ~20% |
| **stop** 🛑 | Low confidence OR high caution OR high override | Reject & quarantine | ~10% |

---

## The Four Safe Decide Functions

### 1. safe_decide_jurisdiction()

```python
from webapp.parser.utils.safe_decide import safe_decide_jurisdiction
from webapp.parser.Context_Integration.library.entity_confidence_map import SignalType

decision = safe_decide_jurisdiction(
    entity_id="Los Angeles County",      # The raw text from source
    state="CA",                           # State abbreviation
    signals=[
        (SignalType.EXACT_MATCH_VERIFIED, True),   # Matched FIPS registry?
        (SignalType.FUZZY_MATCH_HIGH, False),      # Fuzzy match ≥0.90?
    ],
    session_id="sess_xyz"
)

# decision["decision_code"] in {"proceed", "caution", "stop"}
# decision["confidence_score"] ∈ [0, 1]
# decision["reasoning"] = human-readable explanation
```

### 2. safe_decide_office()

```python
decision = safe_decide_office(
    entity_id="Pres",                              # Raw text
    state="CA",
    signals=[
        (SignalType.EXACT_MATCH_VERIFIED, False),
        (SignalType.FUZZY_MATCH_HIGH, True),      # "Pres" ≈ "President"
        (SignalType.CONTEXTUAL_MATCH, True),      # Ballot context suggests presidential
    ],
    session_id="sess_xyz"
)
```

### 3. safe_decide_party()

```python
decision = safe_decide_party(
    entity_id="Dem",                               # Raw text
    signals=[
        (SignalType.EXACT_MATCH_VERIFIED, False),
        (SignalType.ALIAS_MATCH, True),            # "Dem" -> "Democratic"
    ],
    session_id="sess_xyz"
)
```

### 4. safe_decide_source()

```python
decision = safe_decide_source(
    url="https://sos.ca.gov/results/2024",        # Full URL
    signals=[
        (SignalType.EXACT_MATCH_VERIFIED, True),   # In whitelist?
        (SignalType.CONTEXTUAL_MATCH, False),
    ],
    anomalies=[
        (AnomalyType.TYPOSQUAT_PATTERN, False),    # Looks like typo?
        (AnomalyType.SSL_CERTIFICATE_AGE, False),
    ],
    session_id="sess_xyz"
)
```

---

## Helper Predicates

```python
from webapp.parser.utils.safe_decide import should_proceed, should_caution, should_stop

if should_proceed(decision):
    contest["office"] = decision["value"]  # Use canonical
elif should_caution(decision):
    logger.warning(f"Caution: {decision['reasoning']}")
    emit_ui_badge(contest, "caution", decision["confidence_score"])
else:  # should_stop()
    logger.error(f"Rejected: {decision['reasoning']}")
    mark_url_processed(url, status="quarantined")
    return  # Skip this entity
```

---

## DecisionTuple Structure

```python
{
    "value": "President",                          # Canonical resolved entity
    "decision_code": "proceed",                    # proceed | caution | stop
    "confidence_score": 0.82,                      # ∈ [0, 1]
    "caution_score": 0.10,                         # ∈ [0, 1]
    "override_score": 0.0,                         # ≥ 0, unbounded
    "signals_observed": [                          # Which signals fired
        "exact_match_verified",
        "contextual_match"
    ],
    "anomalies_observed": [],                      # Which anomalies detected
    "reasoning": "[OFFICE] President | Decision: proceed",  # Human-readable
    "timestamp": "2026-02-10T14:32:15.123Z",       # ISO8601
    "session_id": "sess_xyz"                       # Audit linkage
}
```

---

## Signal Types (All 10)

**Jurisdiction/Office/Party/Source signals:**

- `EXACT_MATCH_VERIFIED` — Exact match in official registry (FIPS, FEC, statute)
- `EXACT_MATCH_CURATED` — Exact match in curated list (SoS, community)
- `FUZZY_MATCH_HIGH` — Fuzzy match Levenshtein ≥ 0.90
- `FUZZY_MATCH_MEDIUM` — Fuzzy match Levenshtein 0.75–0.89
- `FUZZY_MATCH_LOW` — Fuzzy match Levenshtein < 0.75
- `CONTEXTUAL_MATCH` — Inferred from surrounding data
- `PATTERN_MATCH` — HTML/CSV heuristic
- `ALIAS_MATCH` — Common abbreviation (Pres → President)
- `HEADER_ALIGNMENT` — Column count/names match
- `HANDLER_SUCCESS` — Historical parse success rate

---

## Anomaly Types (All 8)

- `MISMATCHED_TOTALS` — Row/column count unexpected
- `MISSING_CANDIDATE` — Expected candidate absent
- `SUSPICIOUS_HEADER` — Unexpected column names
- `VALUE_INCONSISTENCY` — Data type mismatch, NaN
- `TYPOSQUAT_PATTERN` — Domain/name looks like typo
- `SSL_CERTIFICATE_AGE` — Old/expired cert
- `CONFLICTING_SOURCES` — Multiple sources disagree
- `CONTEXTUAL_MISMATCH` — State/county/contest mismatch

---

## Override Triggers (All 6)

- `ADMIN_FLAG` — Admin manually trusted/rejected (+0.3)
- `VERIFIED_SOURCE_CORRECTION` — Matches official correction (+0.2)
- `ANOMALY_COUNT` — Per anomaly beyond first (+0.15 each)
- `ML_MODEL_LOW_CONFIDENCE` — Sentence-Transformers < 0.5 (+0.15)
- `CONTEXTUAL_MISMATCH` — State/county/contest mismatch (+0.10)
- `MULTIPLE_CORRECTIONS` — Entity corrected > 2× in 30 days (+0.10)

**Rule:** If override_score > 1/3, escalate; if > 2/3, force STOP.

---

## Confidence Gate Math (Reference)

$$\text{confidence\_score} = \frac{\sum (\text{signal\_weight} \times \text{baseline\_conf})}{\sum \text{signal\_weight}}$$

$$\text{caution\_score} = \frac{\sum (\text{anomaly\_weight} \times \text{baseline\_caution})}{\sum \text{anomaly\_weight}}$$

$$\text{override\_score} = \sum \text{trigger\_values}$$

**Decision Gates:**

- **PROCEED:** confidence ≥ 2/3 AND caution ≤ 1/3 AND override ≤ 1/3
- **CAUTION:** (confidence < 2/3 OR caution > 1/3 OR override > 1/3) AND NOT STOP
- **STOP:** confidence < 1/3 OR caution > 2/3 OR override > 1/3

---

## Usage Patterns

### Pattern 1: Simple URL Verification

```python
from webapp.parser.utils.safe_decide import safe_decide_source
from webapp.parser.Context_Integration.library.entity_confidence_map import SignalType

decision = safe_decide_source(
    url=target_url,
    signals=[(SignalType.EXACT_MATCH_VERIFIED, url_in_whitelist)],
    session_id=session_id
)

if not should_proceed(decision):
    return "URL rejected"
```

### Pattern 2: Contest Modal with Decision Badges

```python
decisions = []
for office in candidate_offices:
    d = safe_decide_office(
        entity_id=office["raw"],
        state=state,
        signals=[(SignalType.FUZZY_MATCH_HIGH, office["fuzzy_score"] >= 0.90)],
        session_id=session_id
    )
    decisions.append(d)
    
# Emit to UI for badge rendering
for d in decisions:
    badge_class = "badge-success" if should_proceed(d) else "badge-warning" if should_caution(d) else "badge-danger"
    emit("contest_option", {
        "label": d["value"],
        "decision": d["decision_code"],
        "confidence": d["confidence_score"],
        "badge_class": badge_class
    })
```

### Pattern 3: Anomaly Quarantine

```python
from webapp.parser.Context_Integration.library.entity_confidence_map import AnomalyType

decision = safe_decide_jurisdiction(
    entity_id=county,
    state=state,
    signals=[...],
    anomalies=[
        (AnomalyType.MISMATCHED_TOTALS, row_count != expected_count),
        (AnomalyType.MISSING_CANDIDATE, candidate not in results),
    ],
    session_id=session_id
)

if should_stop(decision):
    logger.error(f"Contest quarantined: {decision['reasoning']}")
    contest["quarantine_reason"] = decision["anomalies_observed"]
    # Prompt manual review
```

---

## Testing (What to Verify)

```python
# Test 1: Simple exact match
decision = safe_decide_jurisdiction("Los Angeles County", "CA", 
                                    [(EXACT_MATCH_VERIFIED, True)])
assert should_proceed(decision)

# Test 2: Fuzzy match + anomaly
decision = safe_decide_office("Pres", "CA",
                              [(FUZZY_MATCH_HIGH, True)],
                              [(VALUE_INCONSISTENCY, True)])
assert should_caution(decision)  # Mixed signals

# Test 3: Typosquat domain
decision = safe_decide_source("https://electionpulse-phishing.tld",
                              [(CONTEXTUAL_MATCH, True)],
                              [(TYPOSQUAT_PATTERN, True)])
assert should_stop(decision)  # High caution

# Test 4: Override escalation
decision = safe_decide_jurisdiction("Unknown County", "CA",
                                    [(FUZZY_MATCH_LOW, True)],
                                    overrides=[ADMIN_FLAG, VERIFIED_SOURCE_CORRECTION])
# override_score = 0.3 + 0.2 = 0.5 > 1/3 → escalate
assert should_caution(decision) or should_proceed(decision)
```

---

## Common Mistakes to Avoid

❌ **Don't:** Hardcode decision logic; always use `should_*()` helpers

```python
# Bad:
if decision["decision_code"] == "proceed":
    use_entity()

# Good:
if should_proceed(decision):
    use_entity()
```

❌ **Don't:** Ignore session_id; it's essential for audit trails

```python
# Bad:
decision = safe_decide_jurisdiction("LA County", "CA", signals)

# Good:
decision = safe_decide_jurisdiction("LA County", "CA", signals, 
                                    session_id=request.sid)
```

❌ **Don't:** Forget to handle CAUTION (not just PROCEED/STOP)

```python
# Bad:
if should_proceed(decision):
    use_entity()
else:
    reject_entity()

# Good:
if should_proceed(decision):
    use_entity()
elif should_caution(decision):
    use_entity_with_warning()  # Badge in UI
else:
    reject_entity()
```

❌ **Don't:** Mix old binary trust scoring with new gates; use new API consistently

---

## Where to Find Things

| What | Where |
| --- | --- |
| Calculation engine | `webapp/parser/Context_Integration/library/entity_confidence_map.py` |
| Safe decide functions | `webapp/parser/utils/safe_decide.py` |
| Decision tuple type | `webapp/parser/utils/shared_logic.py` (DecisionTuple) |
| Signal definitions | `entity_confidence_map.py` (SignalType enum + JURISDICTION_SIGNALS, etc.) |
| Anomaly definitions | `entity_confidence_map.py` (AnomalyType enum + ANOMALY_CATALOG) |
| Override definitions | `entity_confidence_map.py` (OverrideTrigger enum) |
| Tests | `webapp/tests/test_entity_confidence_map.py` (when created) |
| Vocab files | `webapp/parser/Context_Integration/vocab/` (when created) |
| Documentation | `docs/CONFIDENCE_CAUTION_FRAMEWORK.md` (design), `PHASE_A_IMPLEMENTATION_ROADMAP.md` (tasks) |

---

## Phase Timeline

| Phase | Duration | Goal | Status |
| --- | --- | --- | --- |
| **A** | Week 1 | Scaffolding (entity_confidence_map, safe_decide, tests) | ✅ **IN PROGRESS** |
| **B** | Week 2 | Soft Launch (logging only, no gates, UI badges) | ⏳ Next |
| **C** | Week 3+ | Gate Enforcement (URL, handler, anomaly quarantine) | ⏳ After B |
| **D** | Week 4+ | ML Integration (sentence-transformers) | ⏳ After C |
| **E** | Ongoing | Monitoring & Coefficient Tuning | ⏳ After C |

---

## Key Design Principles

1. **Nonpartisan:** No political weighting; only data quality factors
2. **Transparent:** All weights documented; reasoning included in every decision
3. **Auditable:** Every decision logged with session_id + signals + anomalies
4. **Backward Compatible:** Parallel API coexists with legacy code
5. **Safe by Default:** PROCEED requires high confidence + low caution (conservative)

---

**Status:** Phase A scaffolding COMPLETE ✅  
**Next:** Logger integration (Task 1 of 5 remaining)  
**Confidence:** High. Ready for testing + Phase B soft launch.

For detailed implementation tasks, see: `PHASE_A_IMPLEMENTATION_ROADMAP.md`  
For full design spec, see: `docs/CONFIDENCE_CAUTION_FRAMEWORK.md`
