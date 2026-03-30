# Integrity Monitoring & Drift Detection

**Status**: ✅ Production (Schema v1.1 + Full UI Integration)  
**Last Updated**: 2026-01-26  
**Owner**: Core Parser Pipeline

---

## Overview

The Smart Elections Parser includes **comprehensive integrity monitoring** with:

- **Backend drift detection**: ML/NLP performance tracking with configurable thresholds
- **Real-time alerts**: SocketIO event emission during parse sessions
- **Ballot Lens integration**: Live diagnostics panel with metrics and alerts
- **Quality Dashboard**: Historical trend analysis with sparklines, alert management, threshold configuration, session comparison, and export capabilities

This system detects model drift, DOM changes, and data quality degradation, enabling proactive intervention before issues affect downstream systems.

---

## Architecture

### 1. Context Digest Collection

Every parse session generates a **context digest** with:

- **Segment counts**: Total, unknown, labeled, empty
- **Panel coverage**: Extracted vs. total panels
- **Model signals**: Label distribution, confidence statistics (min/max/avg/median), confidence buckets (low/medium/high)
- **Review signals**: Segments needing review, Pattern KB matches
- **Entity extraction**: States, counties, election types detected
- **Schema version**: `1.1` (with segment/unknown metrics)
- **Timestamp**: ISO 8601 generated_at

**Schema v1.1 additions** (2026-01-26):

- `segment_count`
- `unknown_segment_count`
- `unknown_ratio`
- `labeled_segment_count`

**Output locations**:

- Per-session digest: `tools/debug_headless_output/context_digest_{session_id}.json`
- Rolling trend file: `tools/debug_headless_output/context_digest_trends.json` (max 120 entries)

---

### 2. Rolling Trend Storage

The `context_digest_trends.json` file maintains a **rolling window** of the most recent 120 digests with compact drift signals:

- Label distribution
- Confidence ranges (min/max/avg/median)
- Confidence buckets (low/medium/high counts)
- Review signals (segments needing review, Pattern KB matches)
- Unknown ratio
- Segment counts

This enables **historical comparison** without retaining all full digests.

---

### 3. Drift Detection

The `tools/analyze_context_digest_trends.py` script computes **trend deltas** using a sliding window:

**Windows**:

- **Baseline**: Previous 30 digests (configurable)
- **Recent**: Last 5 digests (configurable)

**Metrics tracked**:

- `confidence_avg_delta`: Average confidence change
- `unknown_ratio_delta`: Unknown label ratio change
- `segments_review_delta`: Segments needing review change
- `pattern_kb_matches_delta`: Pattern KB match rate change

**Alert thresholds** (configurable):

- **Confidence drop**: ≤ -0.08 → Warning
- **Unknown spike**: ≥ +0.10 → Warning
- **Review spike**: ≥ +5.0 segments → Warning

---

### 4. Real-Time Emission

After each parse session, the pipeline:

1. Writes context digest to file
2. Updates rolling trend file
3. ✅ **NEW**: Computes integrity signal using trend analyzer
4. ✅ **NEW**: Emits `integrity_signal` event via SocketIO with:
   - `session_id`: Session identifier for routing
   - `signal`: Computed deltas, alerts, baseline/recent metrics
   - `timestamp`: Emission time
   - `status`: `ok`, `alert`, `insufficient_data`, or `error`

**UI Integration**: ✅ **COMPLETE** (2026-02-22)

The Ballot Lens UI now includes full integrity monitoring with:

1. **Real-time Alert Toasts**: `integrity_signal` events trigger warning toasts with drift messages
2. **Diagnostics Panel**: Live metrics panel showing confidence, unknown ratio, segments review, and Pattern KB matches
3. **Delta Indicators**: Color-coded deltas (red for negative trends, green for positive)
4. **Alert Details**: Expandable list of specific warnings (confidence drop, unknown spike, review spike)
5. **Manual Review Routing**: High-priority review flag when segments_review spikes ≥5.0

**Implementation Files**:

- JavaScript: `webapp/static/js/ballot_lens_modern.js` (socket handler + `updateIntegrityPanel()` function)
- CSS: `webapp/static/css/ballot_lens_modern.css` (`.integrity-panel` styles with purple gradient theme)

**UI Features**:

- Panel dynamically created on first signal, inserted near overview cards
- Metrics display baseline → recent comparison with delta arrows
- Toast notifications for all alerts (8-10 second duration)
- Stores review flag in `window.__integrityReviewNeeded` for downstream routing

---

## Usage

### CLI Analysis (Standalone)

```bash
# Default analysis (30 baseline, 5 recent, standard thresholds)
python tools/analyze_context_digest_trends.py

# Custom windows and thresholds
python tools/analyze_context_digest_trends.py \
  --window 20 \
  --recent 3 \
  --conf-drop-threshold 0.10 \
  --unknown-spike-threshold 0.15 \
  --review-spike-threshold 7.0

# Export analysis JSON for downstream processing
python tools/analyze_context_digest_trends.py \
  --json-out tools/debug_headless_output/trend_alerts.json
```

**Output**:

```txt
[TREND] baseline: {'confidence_avg': 0.78, 'unknown_ratio': 0.12, ...}
[TREND] recent: {'confidence_avg': 0.68, 'unknown_ratio': 0.24, ...}
[TREND] deltas: {'confidence_avg_delta': -0.10, 'unknown_ratio_delta': 0.12, ...}
[ALERT] confidence_drop: Confidence avg dropped by 0.100
[ALERT] unknown_spike: Unknown ratio increased by 0.120
```

---

### Programmatic (Pipeline Integration)

The `compute_integrity_signal()` helper is automatically called after context digest emission:

```python
from tools.analyze_context_digest_trends import compute_integrity_signal

signal = compute_integrity_signal(
    trend_file="tools/debug_headless_output/context_digest_trends.json",
    window=30,
    recent=5,
    conf_drop_threshold=0.08,
    unknown_spike_threshold=0.10,
    review_spike_threshold=5.0,
)

# signal contains:
# {
#   "status": "ok" | "alert" | "insufficient_data" | "error",
#   "entry_count": 42,
#   "baseline_window": 30,
#   "recent_window": 5,
#   "baseline": {...},
#   "recent": {...},
#   "deltas": {...},
#   "alerts": [{"type": "confidence_drop", "severity": "warning", "message": "..."}],
# }
```

---

## Alert Types

### 1. Confidence Drop

**Trigger**: Baseline vs. recent confidence avg drops by ≥ threshold

**Possible Causes**:

- New DOM structures not in training data
- Vendor UI redesign
- Increased page complexity
- Model drift

**Response**:

- Review recent sessions for parsing errors
- Check for new vendor formats
- Consider model retraining or Pattern KB expansion

---

### 2. Unknown Spike

**Trigger**: Unknown label ratio increases by ≥ threshold

**Possible Causes**:

- Novel HTML structures
- Missing segment labeling rules
- Entity extraction failures

**Response**:

- Inspect unlabeled segments in recent digests
- Add canonical labels to Pattern KB
- Expand KNOWN_* constants in `context_library/constants.py`

---

### 3. Review Spike

**Trigger**: Segments needing review increases by ≥ threshold

**Possible Causes**:

- Ambiguous page structures (e.g., mixed tables/lists)
- Low ML confidence scores
- Pattern KB mismatches

**Response**:

- Use manual correction workflow (see `health/manual_correction_bot.py`)
- Enrich Pattern KB with validated examples
- Adjust segment labeling thresholds if appropriate

---

## Implementation Details

### File Modifications

**webapp/parser/utils/html_scanner.py**:

- Added `_build_model_signals()` to compute confidence stats and label distribution
- Added `_build_context_digest()` to assemble full digest with schema v1.1
- Added `_update_digest_trends()` to maintain rolling trend file
- Added integrity_signal emission after context_digest write
- Lazy import of `compute_integrity_signal()` to avoid startup cost

**tools/analyze_context_digest_trends.py**:

- New script with CLI and programmatic interfaces
- `compute_integrity_signal()` helper for inline pipeline use
- Configurable windows and thresholds
- JSON export option

---

## UI Integration

### Ballot Lens (Real-time Monitoring)

**Location**: Ballot Lens Modern UI during active parse sessions

**Features**:

- **SocketIO handler**: Listens for `integrity_signal` events emitted by pipeline
- **Live diagnostics panel**: Shows current metrics, deltas, and active alerts in purple gradient panel
- **Toast notifications**: Visual alerts for confidence drops, unknown spikes, review spikes
- **Manual review routing**: Sets `window.__integrityReviewNeeded` flag when segments_review spikes significantly

**Files**:

- `webapp/static/js/ballot_lens_modern.js` - SocketIO handler + updateIntegrityPanel()
- `webapp/static/css/ballot_lens_modern.css` - .integrity-panel styles

### Quality Dashboard (Historical Analysis)

**Location**: `/quality_dashboard` route

**Features**:

- **Trend sparklines**: Chart.js visualizations for 4 key metrics over time
  - Confidence average (purple line)
  - Unknown ratio (orange line)
  - Segments needing review (red line)
  - Pattern KB matches (green line)
- **Alert management**: Dismissible alert cards with type/severity/message
- **Threshold configuration**: Modal UI to adjust sensitivity thresholds
  - Confidence drop threshold
  - Unknown spike threshold
  - Review spike threshold
  - Baseline/recent window sizes
- **Historical comparison**: Select two sessions and view side-by-side metrics diff
- **Export reports**: Download integrity analysis as JSON or CSV
- **Live reload**: Refresh button to fetch latest trends and recompute signal

**API Endpoints**:

- `GET /api/integrity_trends` - Returns rolling trend file data
- `POST /api/integrity_signal` - Computes integrity signal with custom thresholds
- `GET /api/integrity_export` - Exports full integrity report as JSON

**Files**:

- `webapp/templates/quality_dashboard.html` - Dashboard template with integrity section
- `webapp/static/css/quality_dashboard.css` - Integrity section styles
- `webapp/static/js/quality_dashboard.js` - Data loading, sparklines, alerts, comparison
- `webapp/Smart_Elections_Parser_Webapp.py` - Flask API routes (lines 5764-5860)

---

## Future Enhancements (Not in Current Scope)

### Phase 2 (PLANNED - Future)

- **Vendor-specific baselines**: Track drift per state/county handler
- **Alert priority ranking**: High/medium/low severity tiers
- **Auto-correction suggestions**: Propose Pattern KB entries for unknown segments
- **Historical trend visualization**: Web UI dashboard with charts
- **Threshold auto-tuning**: Learn optimal thresholds from feedback

### Phase 3 (RESEARCH - Future)

- **Contextual drift detection**: Compare DOM similarity before/after alerts
- **Segment-level attribution**: Which specific segments cause drift?
- **Multi-model ensemble**: Track drift across multiple NLP models
- **Predictive alerting**: Forecast drift before confidence drops significantly

---

## Related Documentation

- [Context Integration Architecture](../CORE/CONTEXT_INTEGRATION.md)
- [Database Comparison](DATABASE_COMPARISON.md) - URL deduplication before parsing
- [Selenium-NLP Integration](SELENIUM_NLP_INTEGRATION.md) - CAPTCHA fallback + training data
- [Project Audit](../GOVERNANCE/project_audit.md) - Manual review workflow
- [Pattern KB](../DEVELOPMENT/PATTERN_KB.md) - Feedback-driven learning

---

## Quick Reference

**Trend file location**: `tools/debug_headless_output/context_digest_trends.json`
**Per-session digests**: `tools/debug_headless_output/context_digest_{session_id}.json`
**Analyzer script**: `tools/analyze_context_digest_trends.py`
**Schema version**: `1.1` (includes segment/unknown metrics)
**Rolling window size**: 120 digests (configurable via `_update_digest_trends()`)
**Default thresholds**: Confidence: -0.08, Unknown: +0.10, Review: +5.0

---

**Monitoring Status**: Active ✅  
**Production Ready**: Yes  
**Breaking Changes**: None (backward compatible with v1.0 digests)
