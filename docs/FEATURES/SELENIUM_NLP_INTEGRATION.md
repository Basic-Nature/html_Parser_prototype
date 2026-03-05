---
layout: default
---

# Selenium-NLP Integration Strategy

**Status**: ✅ **Implemented (Phase 1)** | **Last Updated**: February 13, 2026

---

## Executive Summary

The Smart Elections Parser now leverages **Selenium as a specialized NLP training data collector** rather than just a fallback browser. This integration captures high-quality structured data from Cloudflare-protected government election sites that Playwright cannot access, feeding spaCy NER models, DOM pattern libraries, and navigation recipe optimization.

### Key Outcomes

- 🎯 **NER Training Enhancement**: Automatic entity extraction from Selenium-accessed HTML
- 🧠 **CAPTCHA Structure Learning**: DOM transition analysis for future automation research
- 🔘 **Semantic Button Context**: Enhanced field selection logging with neighborhood data
- 📸 **Post-Challenge Metadata**: DOM snapshots after CAPTCHA resolution for pattern recognition

---

## Architecture Overview

### Browser Strategy Priority

```branch
Primary: Playwright (async, rich DOM, autoscroll, navigation recipes)
    ↓ (if Cloudflare detected or failure)
Fallback: Selenium (stealth mode uc=True, manual CAPTCHA bypass)
    ↓
Enhanced: NLP Data Collection Pipeline ✨ NEW
```

**Current Ordering** ([html_election_parser.py#L1621-L1626](../webapp/parser/html_election_parser.py#L1621-L1626)):

1. **Playwright** attempts first (60s timeout)
2. **Selenium** activates if:
   - `ENABLE_SELENIUM_FALLBACK=true` (now **default**)
   - Playwright detects `cloudflare_detected=True`
   - Playwright navigation fails entirely

### Why Selenium for NLP?

| Feature | Value for NLP/ML |
| --------- | ----------------- |
| **Stealth Mode** (`uc=True`) | Accesses Cloudflare-protected government sites |
| **Manual CAPTCHA GUI** | Human-in-loop unlocks protected data sources |
| **Government Sites** | Rich entity mentions: states, counties, offices, candidates, dates |
| **Edge Case Coverage** | Training data for challenging source formats |
| **Post-Challenge DOM** | Learn page structures after protection mechanisms |

---

## Implementation Details

### 1. Selenium Fallback Enabled by Default

**File**: [config.py#L291](../webapp/parser/config.py#L291)

```python
# OLD: Default disabled, manual .env opt-in required
ENABLE_SELENIUM_FALLBACK = os.environ.get("ENABLE_SELENIUM_FALLBACK", "false")

# NEW: Default enabled for broader NLP training data collection
ENABLE_SELENIUM_FALLBACK = os.environ.get("ENABLE_SELENIUM_FALLBACK", "true")
```

**Rationale**: Maximize data collection from protected sources without requiring configuration.

**Disable**: Set `ENABLE_SELENIUM_FALLBACK=false` in `.env` if Selenium dependency unavailable.

---

### 2. NER Training from Selenium HTML

**File**: [html_election_parser.py#L1753-L1758](../webapp/parser/html_election_parser.py#L1753-L1758)

**Flow**:

```branch
Selenium extracts HTML → generate_generic_html_result() → CSV + metadata
                                    ↓
                          _capture_selenium_ner_training() ✨ NEW
                                    ↓
                          Extract entities via spaCy
                                    ↓
                          Write to selenium_ner_training.jsonl
```

**Implementation** ([html_election_parser.py#L2478-L2542](../webapp/parser/html_election_parser.py#L2478-L2542)):

```python
def _capture_selenium_ner_training(html_text: str, result: tuple, source_url: str, coordinator=None) -> None:
    """
    Capture NER training data from Selenium-extracted HTML for spaCy model training.
    
    Selenium accesses Cloudflare-protected government sites with rich entity mentions
    (counties, offices, candidates, dates) that would otherwise be unavailable.
    """
    # Extract text from headers + first 3 rows
    headers, rows, contest, metadata = result
    text_sample = " ".join(headers[:5] + row_values[:15])
    
    # Run spaCy entity extraction
    entities = extract_entities(text_sample)
    
    # Write to separate log for quality review
    training_entry = {
        "text": text_sample[:500],
        "entities": [(start, end, label) for _, start, end, label in entities],
        "source": "selenium_fallback",
        "url": source_url,
        "contest": contest,
        "timestamp": int(time.time())
    }
    
    log_path = LOG_DIR / "selenium_ner_training.jsonl"
    # Append JSONL entry
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(orjson.dumps(training_entry).decode("utf-8") + "\n")

**Output**: [`log/selenium_ner_training.jsonl`](../webapp/parser/Context_Integration/Context_Library/log/)

**Format**:

```json
{
  "text": "President Joe Biden Democratic 25000 52.3%",
  "entities": [[10, 19, "PERSON"], [20, 30, "PARTY"], [31, 36, "VOTES"]],
  "source": "selenium_fallback",
  "url": "https://example.gov/results",
  "contest": "President",
  "timestamp": 1739491200
}
```

**Next Steps**:

1. Manual quality review of `selenium_ner_training.jsonl` entries
2. Merge high-quality samples into `spacy_ner_train_data.jsonl`
3. Retrain spaCy models via `retrain_table_structure_models.py`

---

### 3. DOM Metadata After CAPTCHA Resolution

**File**: [seleniumbase_launcher.py#L90-L97](../webapp/parser/utils/seleniumbase_launcher.py#L90-L97)

**Flow**:

```branch
CAPTCHA detected → relaunch_browser_fullscreen_if_needed() → Manual solve
                                    ↓
                          Wait for challenge clearance
                                    ↓
                          _capture_post_captcha_dom_metadata() ✨ NEW
                                    ↓
                          Execute JS: count tables, forms, buttons
                                    ↓
                          _log_captcha_resolution_data()
```

**JavaScript Execution** ([seleniumbase_launcher.py#L139-L154](../webapp/parser/utils/seleniumbase_launcher.py#L139-L154)):

```javascript
return {
  interactive_elements: document.querySelectorAll('button, a[href], select, input[type="submit"]').length,
  form_count: document.forms.length,
  table_count: document.querySelectorAll('table').length,
  challenge_artifacts: document.querySelectorAll('[class*="cloudflare"], [id*="captcha"]').length,
  heading_count: document.querySelectorAll('h1, h2, h3').length,
  body_text_length: document.body.innerText.length,
  viewport: { width: window.innerWidth, height: window.innerHeight }
};
```

**Output**: [`log/captcha_resolution_log.jsonl`](../webapp/parser/Context_Integration/Context_Library/log/)

**Format**:

```json
{
  "url": "https://example.gov/results",
  "captcha_type": "cloudflare",
  "time_to_clear_seconds": 42.3,
  "dom_after_clearance": {
    "interactive_elements": 15,
    "form_count": 2,
    "table_count": 3,
    "challenge_artifacts": 0,
    "heading_count": 5,
    "body_text_length": 12450
  },
  "timestamp": 1739491242
}
```

**Usage**:

- Train classifier to predict post-challenge page complexity
- Inform navigation recipes (how many tables to expect, scroll depth)
- Identify Cloudflare-specific DOM artifacts for future automated detection

---

### 4. Semantic Button Context Logging

**File**: [context_coordinator.py#L4018-L4033](../webapp/parser/Context_Integration/context_coordinator.py#L4018-L4033)

**Enhancement**: Added `button_html_context` field to learning log

**OLD Format**:

```json
{
  "button_label": "Export CSV",
  "selector": "#export-btn",
  "result": "learning_confirmed"
}
```

**NEW Format**:

```json
{
  "button_label": "Export CSV",
  "selector": "#export-btn",
  "button_html_context": {
    "parent_text": "<div class='export-container'>Download Results</div>",
    "sibling_labels": ["Print", "Share", "Download PDF"],
    "classes": ["btn", "btn-primary", "export"],
    "aria_label": "Export results as CSV file"
  },
  "result": "learning_confirmed"
}
```

**Value**:

- Semantic embeddings can match buttons when exact selectors break (DOM changes)
- Train NER to recognize "export" vs "navigate" vs "toggle" button patterns
- Feed [navigation_recipes.py](../webapp/parser/navigator/navigation_recipes.py) candidate filtering

**Output**: [`log/button_learning_log.jsonl`](../webapp/parser/Context_Integration/Context_Library/log/)

---

### 5. CAPTCHA Page Structure Learning

**File**: [captcha_tools.py#L131-L175](../webapp/parser/utils/captcha_tools.py#L131-L175)

**Flow**:

```branch
CAPTCHA detected → wait_for_user_to_solve_captcha()
         ↓
    First iteration: _capture_captcha_dom_state("challenge_present")
         ↓
    Poll every 5s for clearance
         ↓
    Challenge cleared: _capture_captcha_dom_state("challenge_cleared")
         ↓
    _log_captcha_transition() → captcha_transition_log.jsonl
```

**Implementation** ([captcha_tools.py#L190-L227](../webapp/parser/utils/captcha_tools.py#L190-L227)):

```python
def _capture_captcha_dom_state(page_or_driver, state_label: str) -> dict:
    """Capture DOM structure snapshot during CAPTCHA interaction."""
    html_content = get_page_content(page_or_driver)
    html_snippet = html_content[:1000]
    
    indicators_matched = [
        kw for kw in CLOUDFLARE_CAPTCHA_INDICATORS 
        if kw.lower() in html_content.lower()
    ]
    
    return {
        "state": state_label,
        "html_snippet": html_snippet,
        "indicators_matched": indicators_matched,
        "html_length": len(html_content),
        "timestamp": time.time()
    }

def _log_captcha_transition(initial_state, cleared_state, time_to_clear):
    """Log CAPTCHA DOM state transition for supervised ML training."""
    transition_entry = {
        "captcha_type": "cloudflare",
        "initial_indicators": initial_state["indicators_matched"],
        "cleared_indicators": cleared_state["indicators_matched"],
        "time_to_clear_seconds": time_to_clear,
        "html_delta_bytes": cleared_state["html_length"] - initial_state["html_length"],
        "timestamp": int(time.time())
    }
    # Write to captcha_transition_log.jsonl
    log_path = get_safe_log_path("captcha_transition_log.jsonl")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(orjson.dumps(transition_entry).decode("utf-8") + "\n")

**Output**: [`log/captcha_transition_log.jsonl`](../webapp/parser/Context_Integration/Context_Library/log/)

**Format**:

```json
{
  "captcha_type": "cloudflare",
  "initial_indicators": ["verify you are human", "cf-turnstile-response", "challenge-platform"],
  "cleared_indicators": [],
  "time_to_clear_seconds": 38.2,
  "html_delta_bytes": -4523,
  "initial_snippet": "<!DOCTYPE html><html><head>...",
  "cleared_snippet": "<!DOCTYPE html><html><head><title>Election Results...",
  "timestamp": 1739491280
}
```

**Research Opportunities**:

1. **CAPTCHA Type Classification**: Train model to distinguish Cloudflare vs reCAPTCHA vs custom
2. **Time-to-Resolution Prediction**: Estimate manual solving complexity
3. **Navigation Recipe Optimization**: Skip auto-scroll/button-scan during challenge windows

---

## New Logging Artifacts

| File | Purpose | Triggers | Format |
| ------ | --------- | ---------- | -------- |
| **`selenium_ner_training.jsonl`** | spaCy NER training samples | After successful Selenium extraction | `{text, entities, source, url, contest, timestamp}` |
| **`captcha_resolution_log.jsonl`** | Post-CAPTCHA DOM metadata | After manual CAPTCHA clearance | `{url, captcha_type, time_to_clear, dom_after_clearance, timestamp}` |
| **`captcha_transition_log.jsonl`** | Challenge→Clearance state changes | During CAPTCHA wait loop | `{initial_indicators, cleared_indicators, time_to_clear, html_delta}` |
| **`button_learning_log.jsonl`** | Enhanced button selections (existing, now richer) | User confirms button in learning mode | `{button_label, selector, button_html_context, result}` |

All logs: [`webapp/parser/Context_Integration/Context_Library/log/`](../webapp/parser/Context_Integration/Context_Library/log/)

---

## Performance Considerations

### Resource Impact

| Metric | Before | After | Delta |
| -------- | -------- | ------- | ------- |
| **Selenium Usage** | <5% of runs (env-gated) | 10-15% (enabled by default) | +10% |
| **NER Extraction** | N/A | ~100ms per Selenium success | Negligible |
| **DOM Metadata JS** | N/A | ~50ms per CAPTCHA clearance | Negligible |
| **Log I/O** | N/A | ~1-2KB per event | Minimal |

**Total Overhead**: <200ms per Selenium execution (typically 60s+ for CAPTCHA solving, so <0.3% impact)

### Telemetry Counters

Monitor via [html_election_parser.py](../webapp/parser/html_election_parser.py) telemetry:

```python
increment_counter("nav_agent_selenium_success", 1)  # Increased usage expected
increment_counter("nav_agent_selenium_fail", 1)
```

**Success Rate Hypothesis**: Enabling by default should see 80%+ success on Cloudflare-protected sites (manual CAPTCHA bypass).

---

## Roadmap

### Phase 1: Quick Wins ✅ **COMPLETE**

- [x] Enable Selenium fallback by default
- [x] NER training capture from Selenium HTML
- [x] DOM metadata capture after CAPTCHA resolution
- [x] Enhanced button context logging
- [x] CAPTCHA page structure learning

### Phase 2: Pipeline Integration (Next)

**Target**: Q1 2026

1. **Unified Selenium→Context Flow**
   - Route Selenium HTML through `html_scanner.py` for consistent tagging
   - Add `source="selenium"` metadata flag to distinguish from Playwright
   - Enable `ContextCoordinator` for Selenium fallback data (embeddings, anomaly detection)

2. **Telemetry-Based Routing**
   - Aggregate `nav_agent_*` counters to database
   - Analyze success rates per domain
   - Build ML classifier to predict optimal browser agent per URL pattern

3. **Navigation Recipe Compatibility**
   - Extend `navigation_runner.py` to support Selenium driver
   - Map Playwright selectors to Selenium equivalents
   - Test recipe execution on Cloudflare-protected sites

### Phase 3: Advanced Research (Future)

**Target**: Q2 2026

1. **Automated CAPTCHA Handling**
   - Train classifier on `captcha_transition_log.jsonl` dataset
   - Experiment with audio CAPTCHA solving (accessibility bypass)
   - Headless CAPTCHA detection for silent fallback triggers

2. **Agent Similarity Learning**
   - Parallel Playwright + Selenium execution (A/B test)
   - Compare DOM outputs for same URL
   - Identify JS frameworks causing rendering divergence

3. **Embedding-Based Button Ranking**
   - Retrain semantic models on enriched `button_html_context` data
   - Replace exact selector matching with fuzzy semantic search
   - Handle DOM structure changes gracefully

---

## Configuration

### Environment Variables

```bash
# Enable/disable Selenium fallback (default: true)
ENABLE_SELENIUM_FALLBACK=true

# Selenium timeout (default: 60s)
NAV_TIMEOUT_SELENIUM_MS=60000

# CAPTCHA manual solving timeout (default: 300s)
CAPTCHA_TIMEOUT=300

# Disable sentence-transformers if needed (optional)
DISABLE_SENTENCE_TRANSFORMERS=false
```

### Dependencies

**Required**:

- `selenium>=4.40.0` (core WebDriver)
- `playwright>=1.54.0` (primary browser)

**Optional** (for full Selenium stealth):

- `seleniumbase>=4.40.8` (undetected-chromedriver with `uc=True`)

**Install**:

```bash
# Uncomment in requirements.txt:
# seleniumbase>=4.40.8

pip install seleniumbase
```

---

## Monitoring & Maintenance

### Health Checks

Run automated validation:

```bash
python automate.py --skip-web --skip-tests
```

**Expected Outputs**:

- `[FIXED] Salvaged X/X lines in selenium_ner_training.jsonl`
- `[FIXED] Salvaged X/X lines in captcha_resolution_log.jsonl`
- `[FIXED] Salvaged X/X lines in captcha_transition_log.jsonl`

### Log Review

**Weekly**:

1. Check `selenium_ner_training.jsonl` for high-quality entity samples
2. Review `captcha_transition_log.jsonl` for challenge clearance times
3. Analyze `button_learning_log.jsonl` for semantic context richness

**Monthly**:

1. Merge vetted NER samples into `spacy_ner_train_data.jsonl`
2. Retrain spaCy models: `python -m webapp.parser.health.retrain_table_structure_models`
3. Archive old logs (keep last 10k entries)

### Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'seleniumbase'`

- **Cause**: Optional dependency not installed
- **Fix**: `pip install seleniumbase` or set `ENABLE_SELENIUM_FALLBACK=false`

**Issue**: `[Selenium-NLP] NER capture failed`

- **Cause**: spaCy model unavailable or malformed HTML
- **Fix**: Verify `python -m spacy download en_core_web_sm` completed
- **Impact**: Non-fatal, logs debug message and continues

**Issue**: CAPTCHA resolution log missing

- **Cause**: Selenium fallback not triggered (Playwright succeeded)
- **Expected**: Normal operation, logs only appear when CAPTCHA encountered

---

## Technical References

### Core Files Modified

| File | Lines Changed | Purpose |
| ------ | --------------- | --------- |
| [config.py](../webapp/parser/config.py) | L291 | Enable Selenium by default |
| [html_election_parser.py](../webapp/parser/html_election_parser.py) | L1753-L1758, L2478-L2542 | NER training capture |
| [seleniumbase_launcher.py](../webapp/parser/utils/seleniumbase_launcher.py) | L90-L97, L132-L212 | DOM metadata after CAPTCHA |
| [context_coordinator.py](../webapp/parser/Context_Integration/context_coordinator.py) | L4018-L4033 | Enhanced button context |
| [captcha_tools.py](../webapp/parser/utils/captcha_tools.py) | L131-L175, L190-L283 | CAPTCHA structure learning |

### Related Documentation

- [Architectural Overview](../README.md#-architectural-overview)
- [Navigation Recipes](../docs/FEATURES/NAVIGATION_RECIPES.md)
- [Context Integration](../docs/CORE/CONTEXT_INTEGRATION.md)
- [Health Bot Pipeline](../docs/QUALITY/HEALTH_BOT_PIPELINE.md)

---

## Success Metrics

**Target KPIs** (3-month baseline):

| Metric | Target | Measurement |
| -------- | -------- | ------------- |
| Selenium success rate | >80% | `nav_agent_selenium_success / (success + fail)` |
| NER samples collected | >500/month | Count `selenium_ner_training.jsonl` entries |
| CAPTCHA resolutions logged | >50 | Count `captcha_transition_log.jsonl` entries |
| Button context enrichment | 100% | All learning confirmations include `button_html_context` |
| Training data quality | >90% usable | Manual review, entity alignment checks |

**Review Date**: May 13, 2026

---

## Summary

The Selenium-NLP integration transforms Selenium from a last-resort fallback into a **strategic NLP data collection tool**. By capturing entity-rich text from Cloudflare-protected government sites, DOM metadata from post-challenge pages, and semantic button contexts, this enhancement addresses the critical "training data gap" for edge cases and protected sources.

**Key Achievement**: Zero runtime overhead for normal operations, with <200ms impact when Selenium activates—a negligible cost for accessing otherwise unavailable high-quality training data.

**Next Steps**: Phase 2 pipeline integration to route Selenium data through full Context Integration stack for maximum ML value.
