# Integrity Monitoring UI Integration - Complete

**Date**: February 22, 2026
**Status**: ✅ Production Ready

---

## What Was Implemented

### 1. SocketIO Event Handler

**File**: `webapp/static/js/ballot_lens_modern.js`

Added `socket.on('integrity_signal')` handler that:

- Receives integrity signal events from backend
- Displays alert toasts for warnings
- Updates live diagnostics panel
- Routes to manual review for high-priority issues

### 2. Diagnostics Panel

**Function**: `updateIntegrityPanel(signal, sessionId)`

Creates/updates a dynamic panel showing:

- **Status Badge**: ok (green), alert (orange), insufficient_data (blue), error (red)
- **Live Metrics**:
  - Confidence Average (with delta)
  - Unknown Ratio (with delta)
  - Segments Review (with delta)
  - Pattern KB Matches (with delta)
- **Delta Indicators**: Color-coded arrows (🔴 red for negative, 🟢 green for positive)
- **Alert Details**: Expandable list with icons (📉 confidence drop, ❓ unknown spike, 📋 review spike)
- **Footer**: Displays baseline/recent window sizes and total trend entries

### 3. Visual Styling

**File**: `webapp/static/css/ballot_lens_modern.css`

Added `.integrity-panel` styles with:

- **Purple gradient theme** (matches monitoring/diagnostics concept)
- **Glassmorphism effect** (semi-transparent background with blurred overlay)
- **Hover animations** (subtle lift + shadow enhancement)
- **Metric rows** with grid layout for label/value/delta alignment
- **Delta badges** with distinct colors and backgrounds
- **Responsive design** following existing artifact-card patterns

### 4. Alert Flow

**Toast Notifications**: 8-10 second duration

1. `confidence_drop` → "ML/NLP Alert: Confidence avg dropped by X.XXX"
2. `unknown_spike` → "ML/NLP Alert: Unknown ratio increased by X.XXX"
3. `review_spike` → "High-priority review needed - segments flagged for manual correction"

### 5. Manual Review Routing

**Trigger**: `segments_review_delta ≥ 5.0`

Sets window flags for downstream routing:

- `window.__integrityReviewNeeded = true`
- `window.__integritySessionId = <session_id>`

Future integration can check these flags to auto-route to correction workflow.

---

## How It Works

### Real-Time Flow

```branch
Parse Session Start
       ↓
scan_html_for_context() executes
       ↓
Context digest written to file
       ↓
Rolling trend file updated
       ↓
compute_integrity_signal() called
       ↓
Backend emits 'integrity_signal' event
       ↓
Frontend receives event
       ↓
┌──────────────────────────────┐
│  IF status === 'alert'       │
│  → Display toast warnings    │
│  → Update diagnostics panel  │
│  → Set review flag if needed │
└──────────────────────────────┘
       ↓
User sees live metrics + alerts
```

### Signal Structure

```json
{
  "type": "integrity_signal",
  "session_id": "abc123",
  "timestamp": 1708617234.567,
  "signal": {
    "status": "alert",
    "entry_count": 42,
    "baseline_window": 30,
    "recent_window": 5,
    "baseline": {
      "confidence_avg": 0.78,
      "unknown_ratio": 0.12,
      "segments_review": 3.2,
      "pattern_kb_matches": 45.7
    },
    "recent": {
      "confidence_avg": 0.68,
      "unknown_ratio": 0.24,
      "segments_review": 8.5,
      "pattern_kb_matches": 42.1
    },
    "deltas": {
      "confidence_avg_delta": -0.10,
      "unknown_ratio_delta": 0.12,
      "segments_review_delta": 5.3,
      "pattern_kb_matches_delta": -3.6
    },
    "alerts": [
      {
        "type": "confidence_drop",
        "severity": "warning",
        "message": "Confidence avg dropped by 0.100"
      },
      {
        "type": "unknown_spike",
        "severity": "warning",
        "message": "Unknown ratio increased by 0.120"
      },
      {
        "type": "review_spike",
        "severity": "warning",
        "message": "Segments needing review increased by 5.30"
      }
    ]
  }
}
```

---

## Visual Design

### Panel Appearance

```txt
┌────────────────────────────────────────────────┐
│ 🔍 ML/NLP Integrity Monitor       [alert ⚠️]  │
├────────────────────────────────────────────────┤
│  Confidence Avg:         0.680    -0.100 🔴   │
│  Unknown Ratio:          0.240    +0.120 🔴   │
│  Segments Review:        8.5      +5.3   🔴   │
│  Pattern KB Matches:     42.1     -3.6   🔴   │
│                                                │
│  ▼ Alerts (3)                                  │
│    📉 Confidence avg dropped by 0.100          │
│    ❓ Unknown ratio increased by 0.120         │
│    📋 Segments needing review increased by 5.3 │
│                                                │
│  Baseline: 30 sessions | Recent: 5 | Total: 42│
└────────────────────────────────────────────────┘
```

### Color Scheme

- **Panel Background**: Purple gradient (`rgba(147, 51, 234, 0.08)` → `rgba(17, 24, 39, 0.6)`)
- **Border**: Purple glow (`rgba(147, 51, 234, 0.55)`)
- **Positive Deltas**: Green (`var(--accent-success)` with 15% opacity background)
- **Negative Deltas**: Red (`var(--accent-danger)` with 15% opacity background)
- **Status Badges**: Green (ok), Orange (alert), Blue (insufficient_data), Red (error)

---

## Testing

### Manual Test

1. Start webapp: `python -m webapp.Smart_Elections_Parser_Webapp`
2. Open Ballot Lens UI
3. Run a parse session (any URL)
4. After 2+ sessions, integrity_signal will emit
5. Watch for:
   - Toast notifications (if alerts present)
   - Diagnostics panel appearing below overview cards
   - Live metric updates with delta indicators

### Expected Behavior

- **First session**: Panel shows "Accumulating baseline data..." (insufficient_data)
- **Subsequent sessions**: Panel shows metrics with deltas
- **If drift detected**: Toasts appear + alerts expandable in panel
- **High review spike**: Extra toast about manual correction needed

---

## Files Modified

### JavaScript

- `webapp/static/js/ballot_lens_modern.js` (+175 lines)
  - Added IntegritySignalPayload, IntegritySignal, IntegrityMetrics, IntegrityDeltas, IntegrityAlert typedefs
  - Added `socket.on('integrity_signal')` handler
  - Added `updateIntegrityPanel()` function
  - Added window flags for manual review routing

### CSS

- `webapp/static/css/ballot_lens_modern.css` (+185 lines)
  - Added `.integrity-panel` and related classes
  - Added `.metric-row`, `.metric-label`, `.metric-value`, `.metric-delta` styles
  - Added `.integrity-alerts`, `.alert-list`, `.alert-item` styles
  - Added badge variants (`.badge-success`, `.badge-warning`, `.badge-info`)

### Documentation

- `docs/FEATURES/INTEGRITY_MONITORING.md`
  - Updated "UI Integration" section with completion status
  - Added implementation file references
  - Added UI feature list

### Test

- `webapp/tests/test_integrity_signal.py`
  - Validates end-to-end flow (digest write → trend update → signal emission)

---

## Future Enhancements

1. **Trend Charts**: Add sparkline visualizations for metric history
2. **Threshold Configuration**: Allow UI-based adjustment of alert thresholds
3. **Export Reports**: Download integrity report as JSON/CSV
4. **Historical Comparison**: Compare current session vs. specific past session
5. **Alert Dismissal**: Allow users to acknowledge/dismiss alerts
6. **Manual Review Integration**: Auto-launch correction workflow when review flag set

---

## Production Checklist

- [x] SocketIO event handler implemented
- [x] Diagnostics panel UI complete
- [x] CSS styling with responsive design
- [x] TypeScript type definitions added
- [x] Toast notifications for alerts
- [x] Manual review routing flags
- [x] Documentation updated
- [x] No compile/lint errors
- [x] Matching design language with existing UI

**Status**: Ready for production deployment 🚀
