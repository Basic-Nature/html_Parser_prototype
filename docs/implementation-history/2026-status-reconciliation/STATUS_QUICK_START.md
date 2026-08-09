# Status Reconciliation System - Quick Start Guide

## What This Solves

**The Problem**: Dashboard was showing incomplete information by displaying only what the parser had done, ignoring what the Google Sheets workflow showed.

**Example**:

- URL: Arizona 2024 Presidential Election
- Parser Status: `pending` (never attempted)
- Dashboard showed: ⏳ Pending, [🔄 Parse] button
- Reality: Google Sheets shows "PROD Loaded" (already finalized)
- Result: Confusing, contradictory information

**The Solution**: Reconcile two status systems to show the TRUE status

---

## How It Works

### Two Sources of Truth

1. **Parser Status** (from `.processed_urls` file)
   - What the parser actually attempted
   - Values: success, fail, error, partial, pending, etc.
   - Authority: Highest when present (evidence of execution)

2. **Worklist Status** (from Google Sheets)
   - What the workflow shows
   - Values: PROD Loaded, QC Loaded, Pre-QC Fail/Fix, Download Needed, etc.
   - Authority: Used when parser hasn't touched the URL

3. **Default** (when both missing)
   - Status: pending
   - Means: No tracking data exists

### Authority Hierarchy

```txt
Parser Status (if exists) > Worklist Status (if exists) > Default (pending)
```

**Example Reconciliation**:

| Parser | Worklist | Result | Reason |
| -------- | ---------- | -------- | -------- |
| `success` | `QC Loaded` | ✅ Success | Parser wins (execution evidence) |
| `pending` | `PROD Loaded` | 📦 Production | Parser didn't run; use worklist |
| `fail` | `Pre-QC Fail/Fix` | ❌ Failed | Parser wins (more recent action) |
| null | null | ⏳ Pending | No data; default to pending |

---

## Using the Dashboard

### Dashboard URL

```txt
http://localhost:5000/url_status_dashboard
```

### What You'll See

**Status Column** (now shows correct information):

- ✅ Success (if parser succeeded)
- ❌ Failed (if parser failed)
- 📦 Production (if in workflow as finalized)
- ⏳ Pending (if never processed)
- 🔍 QC Complete, ⚠️ QC Failed, etc. (workflow statuses)

**Parse Button** (intelligently hidden):

- Shows ✗ for URLs already in production
- Shows ✓ for URLs that need parsing
- Prevented unnecessary re-work

**Tooltip on Status Badge**:

- Hover over badge to see: "Authority: parser" or "Authority: worklist"
- See both statuses for debugging

### Filtering

**By Canonical Status**:

```txt
?status=production          # Show only finalized URLs
?status=pending             # Show unprocessed URLs
?status=fail                # Show failed URLs
```

**By Parser Status Only** (debugging):

```txt
?parser_status=success      # What parser actually succeeded on
```

---

## API Reference

### Request

```bash
GET /api/url_status?status=production&state=arizona&limit=50&hide_pii=true
```

### Response

```json
{
  "success": true,
  "total": 213,
  "filtered": 42,
  "entries": [
    {
      "url": "https://elections.az.gov/...",
      "label": "Arizona President 2024",

      "parser_status": "pending",           // What parser tried
      "worklist_status": "PROD Loaded",     // What workflow shows
      "canonical_status": "production",     // ← USE THIS

      "status_info": {
        "icon": "📦",
        "label": "Production",
        "badge_class": "success",
        "authority": "worklist"             // Which won
      },

      "in_production": true,
      "production_source": "google_sheets",
      "last_processed": null,
      "state": "Arizona"
    }
  ],
  "status_breakdown": {
    "production": 100,
    "pending": 50,
    "fail": 6
  }
}
```

### Key Fields

- **`canonical_status`** - Use this for display (already reconciled)
- **`status_info`** - Badge data (icon, label, color)
- **`parser_status`** - Raw parser status (for debugging)
- **`worklist_status`** - Raw worklist status (for debugging)
- **`authority`** - Which system "won" (parser | worklist)

---

## Data Privacy

### What's Hidden from Public View

- Employee names (DL1 assignee, DL2 assignee)
- Email addresses
- Phone numbers

### Configuration

**Always use this for public dashboards**:

```bash
GET /api/url_status?hide_pii=true
```

**Only for internal/authenticated endpoints**:

```bash
GET /api/url_status?hide_pii=false
```

Dashboard automatically uses `hide_pii=true`.

---

## Common Scenarios

### Scenario 1: "Why is this still shown as pending?"

**Check**:

1. Look at `canonical_status` in API response
2. If it says `production`, the dashboard is correct (it's finalized)
3. If it says `pending`, no parser or workflow data exists for this URL

**Solution**:

- Add to Google Sheets if it should be tracked
- Or run parser explicitly on the URL

### Scenario 2: "Why can't I re-parse this URL?"

**Reason**: Status is `production` or similar workflow status

**Check**: Hover over status badge to see authority

**Solution**:

- If it's truly finalized, update the workflow
- If it needs re-parsing, manually change status in Google Sheets

### Scenario 3: "Parser succeeded but status shows failed in QC"

**This is correct behavior**:

- `canonical_status` = `success` (shows parser worked)
- Worklist shows `Pre-QC Fail/Fix` (QC found issues)
- Both are important pieces of information
- Hover over badge to see both

---

## Integration Points

### For Developers

**Import the reconciliation system**:

```python
from webapp.parser.utils.status_reconciliation import StatusReconciliation

# Reconcile a single URL
canonical, info = StatusReconciliation.reconcile(
    url="https://example.com/data",
    parser_status="success",
    worklist_status="QC Loaded",
    production_source=None,
    last_processed="2026-02-19 12:00:00"
)

# canonical = "success" (parser wins)
# info = {
#   "icon": "✅",
#   "label": "Success",
#   "badge_class": "success",
#   "authority": "parser",
#   "parsed": True,
#   ...
# }
```

**Filter URLs needing action**:

```python
if StatusReconciliation.status_requires_action(canonical):
    # Status needs manual intervention (fail, error, qc_failed, etc.)
    pass

if StatusReconciliation.status_is_complete(canonical):
    # Status indicates processing finished
    pass
```

### For Dashboard

**When displaying an entry**:

```javascript
const entry = apiResponse.entries[0];

// Show reconciled status
badge.textContent = `${entry.status_info.icon} ${entry.status_info.label}`;
badge.className = `status-badge ${entry.status_info.badge_class}`;

// Show parse button only if appropriate
const needsParsing = [
  'pending', 'fail', 'error', 'download_needed'
].includes(entry.canonical_status);

parseButton.hidden = !needsParsing;
```

---

## Status Values Reference

### Parser Statuses

| Value | Icon | Meaning |
| ------- | ------ | --------- |
| `success` | ✅ | URL parsed successfully |
| `fail` | ❌ | Parsing failed |
| `error` | ⚠️ | Execution error |
| `partial` | 🔸 | Some data extracted |
| `cancelled` | ⏹️ | Parsing cancelled |
| `rejected` | 🚫 | Output rejected |
| `quarantined` | ⚠️ | In quarantine hold |
| `skipped_data_exists` | ⏭️ | Already in production |
| `pending` | ⏳ | Not processed |

### Worklist Statuses

| Value | Icon | Meaning |
| ------- | ------ | --------- |
| `PROD Loaded` | 📦 | In production |
| `QC Loaded` | ✓ | QC complete |
| `QC1 Fail/Fix` | ❌ | QC 1 failed |
| `QC2 Fail/Fix` | ❌ | QC 2 failed |
| `Pre-QC Fail/Fix` | ❌ | Pre-QC failed |
| `Download Needed` | 📥 | Needs download |
| `DL1 Processing` | ⚙️ | DL1 working |
| `DL2 Processing` | ⚙️ | DL2 working |
| `Cand Check DL1` | 🔍 | Candidate review |

---

## Testing

**Run tests**:

```bash
cd c:\Users\olivi\html_Parser_prototype
python -m pytest tests/test_status_reconciliation.py -v
```

**All tests should pass** ✓

---

## Files Modified

1. **Created**:
   - `webapp/parser/utils/status_reconciliation.py` - Main system
   - `tests/test_status_reconciliation.py` - Tests
   - `docs/temp/STATUS_RECONCILIATION_GUIDE.md` - Detailed guide

2. **Updated**:
   - `webapp/Smart_Elections_Parser_Webapp.py` - API endpoint
   - `webapp/templates/url_status_dashboard.html` - Dashboard UI

---

## Troubleshooting

### Issue: Status still shows "pending" for URLs I know are in production

**Check**:

1. Verify URL matches exactly (case-sensitive)
2. Confirm Google Sheets has the correct status
3. Check API response: is `canonical_status` showing as `production`?

**Fix**:

- Ensure URL in `.processed_urls` matches URL in sheets
- Run parser again to update status

### Issue: Parse button won't show for failed URLs

**Check**:

1. Confirm `canonical_status` is actually `fail` or `error`
2. Check that dashboard is not filtering them out

**Fix**:

- Remove filters and search for `?status=fail`
- Or use `?parser_status=fail` to see raw parser status

### Issue: Personal names still visible in exports

**Check**:

1. Verify `hide_pii=true` is in request

**Fix**:

- Add `?hide_pii=true` to API request
- Dashboard does this automatically

---

## Next Steps

See [STATUS_RECONCILIATION_GUIDE.md](./STATUS_RECONCILIATION_GUIDE.md) for deeper technical details.
