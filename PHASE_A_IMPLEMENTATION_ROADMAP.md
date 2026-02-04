# Phase A Implementation Roadmap: Scaffolding (Week 1)

## Overview

Phase A establishes the foundational infrastructure for embedding confidence/caution decision gates throughout the Smart Elections Parser pipeline. This week focuses on scaffolding and integration points—not enforcement.

**Objectives:**

- ✅ Create entity_confidence_map.py (DONE)
- ✅ Extend shared_logic.py with DecisionTuple type (DONE)
- ✅ Create safe_decide.py module with helper functions (DONE)
- ⏳ Integrate decision logging into logger_singleton.py
- ⏳ Create vocab/ folder structure and initial entity files
- ⏳ Setup Prometheus decision metrics
- ⏳ Integration tests for confidence/caution calculation

## Architecture

### Key Files

**Created:**

- `webapp/parser/Context_Integration/library/entity_confidence_map.py` (500+ lines)
  - SignalType, AnomalyType, OverrideTrigger enums
  - SignalCoefficient, AnomalyCoefficient, ConfidenceCautionResult dataclasses
  - EntityConfidenceMap class with calculation engine
  - Signal catalogs: JURISDICTION_SIGNALS, OFFICE_SIGNALS, PARTY_SIGNALS, SOURCE_SIGNALS
  - Anomaly catalog: ANOMALY_CATALOG (8 anomaly types)
  - Singleton accessor: get_confidence_map()

- `webapp/parser/utils/safe_decide.py` (250+ lines)
  - safe_decide_jurisdiction(), safe_decide_office(), safe_decide_party(), safe_decide_source()
  - Helper functions: should_proceed(), should_caution(), should_stop()
  - Decision event logging: _emit_decision_log()
  - All functions return DecisionTuple with audit metadata

- Extended `webapp/parser/utils/shared_logic.py`
  - Added DecisionTuple TypedDict (value, decision_code, scores, signals, anomalies, reasoning, timestamp, session_id)

**To Create:**

- `webapp/parser/Context_Integration/vocab/entities/offices.txt` — canonical office names with confidence
- `webapp/parser/Context_Integration/vocab/entities/parties.txt` — FEC party list with confidence
- `webapp/parser/Context_Integration/vocab/entities/jurisdictions.txt` — FIPS counties with confidence
- `webapp/parser/Context_Integration/vocab/validators/office_aliases.txt` — common abbreviations
- `webapp/parser/Context_Integration/vocab/validators/party_aliases.txt` — Democratic/GOP/Dem mappings
- `webapp/parser/Context_Integration/vocab/sources/verified_sources.txt` — whitelisted domains
- `webapp/parser/Context_Integration/vocab/scoring/signal_coefficients.json` — override default weights (optional)

**To Extend:**

- `webapp/parser/utils/logger_singleton.py` — add decision event filtering + deduplication
- `webapp/parser/utils/metrics_prom.py` — add decision_proceed_total, decision_caution_total, decision_stop_total counters
- `webapp/parser/utils/db_utils.py` — optional: create decision_log table for long-term audit

## Implementation Checklist

### Task 1: Extend Logger for Decision Events (1–2 hours)

**File:** `webapp/parser/utils/logger_singleton.py`

**Changes:**

1. Add decision event type to valid log types
2. Create `_filter_decision_noise()` to deduplicate repeated decisions (same entity, same decision_code within 5 minutes)
3. Add "decision" to DEFAULT_CONSOLE_LEVELS if ENABLE_DECISION_LOGGING=true
4. Ensure decision logs include session_id for audit linking

**Code Snippet:**

```python
# In logger_singleton.py, after normalize_log_obj()

DECISION_DEDUPE_WINDOW_SEC = 300  # 5 minutes
_DECISION_DEDUP_CACHE: Dict[str, float] = {}  # key=f"{entity_id}|{decision_code}", value=timestamp

def _filter_decision_noise(obj: dict) -> bool:
    """Skip duplicate decisions for same entity within 5-min window."""
    if obj.get("type") != "decision":
        return True  # Keep non-decision logs
    
    entity_id = obj.get("entity_id")
    decision_code = obj.get("decision_code")
    if not (entity_id and decision_code):
        return True  # Keep logs we can't dedupe
    
    key = f"{entity_id}|{decision_code}"
    now = time.time()
    last_ts = _DECISION_DEDUP_CACHE.get(key)
    
    if last_ts and (now - last_ts) < DECISION_DEDUPE_WINDOW_SEC:
        return False  # Skip duplicate
    
    _DECISION_DEDUP_CACHE[key] = now
    return True
```

**Test:**

- Emit same decision twice; verify second is deduplicated
- Emit after 6 minutes; verify both logged

---

### Task 2: Setup Prometheus Metrics (1–2 hours)

**File:** `webapp/parser/utils/metrics_prom.py` (create if missing)

**Changes:**

1. Import prometheus_client Counter
2. Register three counters:
   - `decision_proceed_total` with labels [entity_type, handler, state, confidence_bucket]
   - `decision_caution_total` with labels [entity_type, caution_reason, state]
   - `decision_stop_total` with labels [entity_type, override_trigger, state]
3. Create increment functions: `increment_decision_proceed()`, etc.

**Code Snippet:**

```python
from prometheus_client import Counter

decision_proceed_total = Counter(
    'decision_proceed_total',
    'Count of PROCEED decisions',
    ['entity_type', 'handler', 'state', 'confidence_bucket']
)

decision_caution_total = Counter(
    'decision_caution_total',
    'Count of CAUTION decisions',
    ['entity_type', 'caution_reason', 'state']
)

decision_stop_total = Counter(
    'decision_stop_total',
    'Count of STOP decisions',
    ['entity_type', 'override_trigger', 'state']
)

def increment_decision_proceed(entity_type, handler, state, confidence_score):
    bucket = "high" if confidence_score >= 0.90 else "medium" if confidence_score >= 0.70 else "low"
    decision_proceed_total.labels(
        entity_type=entity_type or "unknown",
        handler=handler or "unknown",
        state=state or "unknown",
        confidence_bucket=bucket
    ).inc()

# Similar for caution, stop...
```

**Test:**

- Call increment_decision_proceed() and verify metric increments
- Query /metrics endpoint and verify labels

---

### Task 3: Create Vocab Files (1–2 hours)

**Directory:** `webapp/parser/Context_Integration/vocab/`

**Structure:**

```txt
vocab/
  entities/
    offices.txt         — canonical office names, one per line
    parties.txt         — FEC party list
    jurisdictions.txt   — FIPS counties
  validators/
    office_aliases.txt  — "Pres -> President"
    party_aliases.txt   — "GOP -> Republican"
  sources/
    verified_sources.txt — whitelisted .gov domains
  scoring/
    signal_coefficients.json — (optional) override default weights
```

**offices.txt format:**

```txt
President
U.S. Senator
U.S. Representative
Governor
State Senator
State Representative
...
```

**office_aliases.txt format:**

```txt
Pres -> President
Sen -> U.S. Senator
Rep -> U.S. Representative
Gov -> Governor
...
```

**parties.txt format (FEC):**

```txt
Democratic Party
Republican Party
Independent
Green Party
Libertarian Party
...
```

**party_aliases.txt format:**

```txt
Dem -> Democratic Party
GOP -> Republican Party
Indep -> Independent
Green -> Green Party
Lib -> Libertarian Party
...
```

**verified_sources.txt format:**

```txt
.gov
sos.ca.gov
elections.ca.gov
secretary.state.co.us
...
```

**signal_coefficients.json format (optional override):**

```json
{
  "jurisdiction": {
    "EXACT_MATCH_VERIFIED": 1.0,
    "EXACT_MATCH_CURATED": 0.9,
    "ALIAS_MATCH": 0.7,
    ...
  },
  "office": {...},
  ...
}
```

**Test:**

- Load vocab files and verify entries count
- Lookup "Pres" in office_aliases and verify maps to "President"

---

### Task 4: Integration Tests (2–3 hours)

**File:** `webapp/tests/test_entity_confidence_map.py` (create)

**Tests to Write:**

1. **test_jurisdiction_exact_match_high_confidence**
   - Signals: [EXACT_MATCH_VERIFIED=True, ALIAS_MATCH=False]
   - Expected: decision_code="proceed", confidence_score ≥ 0.95

2. **test_office_fuzzy_match_medium_confidence**
   - Signals: [FUZZY_MATCH_MEDIUM=True, CONTEXTUAL_MATCH=False]
   - Expected: decision_code="caution", confidence_score 0.50–0.70

3. **test_source_with_anomaly_low_confidence**
   - Signals: [EXACT_MATCH_CURATED=True]
   - Anomalies: [TYPOSQUAT_PATTERN=True, SUSPICIOUS_HEADER=True]
   - Expected: decision_code="stop", caution_score ≥ 0.80

4. **test_override_escalation**
   - Signals: [FUZZY_MATCH_LOW=True]
   - Overrides: [ADMIN_FLAG, VERIFIED_SOURCE_CORRECTION, ANOMALY_COUNT]
   - Expected: decision_code="proceed" or "caution" (override > 1/3)

5. **test_safe_decide_jurisdiction_logging**
   - Call safe_decide_jurisdiction() with test data
   - Verify decision log emitted
   - Check log contains session_id, reasoning, scores

6. **test_safe_decide_party_deduplication**
   - Call safe_decide_party() twice with same inputs
   - Verify second call deduplicated in logs (within 5-min window)

**Code Skeleton:**

```python
import pytest
from webapp.parser.Context_Integration.library.entity_confidence_map import (
    DecisionCode, SignalType, AnomalyType, OverrideTrigger, get_confidence_map
)
from webapp.parser.utils.safe_decide import safe_decide_jurisdiction, should_proceed

def test_jurisdiction_exact_match_high_confidence():
    confidence_map = get_confidence_map()
    result = confidence_map.calculate_confidence_caution(
        entity_id="Los Angeles County",
        entity_type="jurisdiction",
        signals=[(SignalType.EXACT_MATCH_VERIFIED, True)],
        anomalies=[],
        override_triggers=[]
    )
    assert result.decision_code == DecisionCode.PROCEED
    assert result.confidence_score >= 0.95

def test_safe_decide_jurisdiction_logging(caplog):
    decision = safe_decide_jurisdiction(
        entity_id="Los Angeles County",
        state="CA",
        signals=[(SignalType.EXACT_MATCH_VERIFIED, True)],
        session_id="test_sess_123"
    )
    assert should_proceed(decision)
    assert decision["session_id"] == "test_sess_123"
    # Verify log record emitted
    assert "Los Angeles County" in caplog.text or decision["decision_code"] in caplog.text
```

**Run Tests:**

```bash
pytest webapp/tests/test_entity_confidence_map.py -v
```

---

### Task 5: Update INFRASTRUCTURE_PLAN.md (30 min)

**File:** `docs/INFRASTRUCTURE_PLAN.md`

**Add Section:** Phase 2 Observability → Decision Gates (after "Monitoring & Telemetry")

**Content:**

```markdown
### Decision Gate Integration (Phase 2b: Week 1)

Embed confidence/caution gates into observable pipeline:

1. **Decision Event Logging** (logger_singleton.py)
   - Log decision_code, confidence/caution scores, signals, anomalies
   - Deduplicate repeated decisions (5-min window)
   - Filter by ENABLE_DECISION_LOGGING env var

2. **Prometheus Metrics** (metrics_prom.py)
   - decision_proceed_total{entity_type, handler, state, confidence_bucket}
   - decision_caution_total{entity_type, caution_reason, state}
   - decision_stop_total{entity_type, override_trigger, state}

3. **Vocabulary Files** (vocab/ folder)
   - entities/: offices, parties, jurisdictions
   - validators/: aliases for fuzzy matching
   - sources/: verified domain whitelist
   - scoring/: signal coefficient overrides (optional)

4. **Decision Audit Trail**
   - All decisions logged with session_id for linkage
   - Accessible via /health_tasks or dedicated admin dashboard
   - No enforcement (yet); soft logging only

**Risk:** False positives if signal weights miscalibrated. Week 1 is observation-only; Phase 2c enforces gates.
```

**Also Update:** Phase 3 Deployment section to note:
> "Phase 3 gates will consume decision metrics from Phase 2b to auto-scale resources and trigger quarantine workflows."

---

## Acceptance Criteria (Phase A Complete)

- [x] entity_confidence_map.py created with 12+ signal types, 8 anomaly types, override triggers
- [x] safe_decide.py created with 4 safe_decide_* functions + helper predicates
- [x] DecisionTuple type added to shared_logic.py
- [ ] logger_singleton.py extended with decision event filtering
- [ ] Prometheus metrics registered (decision_proceed_total, caution_total, stop_total)
- [ ] Vocab files created (offices, parties, jurisdictions, aliases, sources)
- [ ] Integration tests written and passing (≥6 test cases)
- [ ] INFRASTRUCTURE_PLAN.md updated with Phase 2b section
- [ ] Code review: All new functions have docstrings + type hints

---

## Next Steps (Phase B: Week 2—Soft Launch)

Once Phase A complete:

1. **Enable Decision Badges in UI** (`static/js/run_parser.js`)
   - Add ✅ PROCEED / ⚠️ CAUTION / 🛑 STOP badge to contest options
   - Show confidence/caution scores on hover

2. **Run Soft-Launch Week** (logging only, no gates)
   - Monitor decision distribution: how many PROCEED vs CAUTION vs STOP?
   - Collect false positive rate
   - Measure signal coefficient calibration accuracy

3. **Phase C Enforcement** (Week 3+)
   - Enable gates in URL trust scorer, handler selection, anomaly quarantine
   - Gradual per-handler rollout with monitoring

---

## Files Summary

**Created (Phase A):**

- webapp/parser/Context_Integration/library/entity_confidence_map.py
- webapp/parser/utils/safe_decide.py
- webapp/parser/Context_Integration/vocab/entities/{offices,parties,jurisdictions}.txt
- webapp/parser/Context_Integration/vocab/validators/{office,party}_aliases.txt
- webapp/parser/Context_Integration/vocab/sources/verified_sources.txt
- webapp/tests/test_entity_confidence_map.py

**Modified (Phase A):**

- webapp/parser/utils/shared_logic.py (added DecisionTuple)
- webapp/parser/utils/logger_singleton.py (decision filtering)
- webapp/parser/utils/metrics_prom.py (decision counters)
- docs/INFRASTRUCTURE_PLAN.md (Phase 2b section)

**To Create (Phase B+):**

- Grafana dashboard JSON (decision flow visualization)
- Admin audit UI for override approvals
- Health task for weekly decision anomaly detection

---

## Time Estimate

- Entity confidence map: 1.5 hours (DONE)
- Safe decide helpers: 1 hour (DONE)
- Logger extension: 1–2 hours
- Prometheus metrics: 1–2 hours
- Vocab files: 1–2 hours
- Integration tests: 2–3 hours
- Documentation + review: 1 hour

***Total: 8–10 hours for Phase A (1 week)***

---

**Status:** Scaffolding created (entity_confidence_map.py + safe_decide.py). Ready for logger integration.

**Next Immediate Action:** Extend logger_singleton.py + setup Prometheus metrics (Tasks 1–2).
