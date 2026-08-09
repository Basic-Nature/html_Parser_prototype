# Status Reconciliation System

## Overview

The Smart Elections Parser tracks URLs through **two separate systems** that must be reconciled to show accurate status:

1. **Parser Status** — What the parser attempted
2. **Worklist Status** — What the Google Sheets workflow shows

This document explains how these two systems work together.

---

## The Problem

Before reconciliation, the dashboard would show:

- **URL**: Arizona 2024 President
- **Parser Status**: `pending` (never attempted)
- **Dashboard Shows**: ⏳ Pending — "🔄 Parse" button
- **Reality**: Google Sheets shows "PROD Loaded" (already in production)

**Result**: Confusing, contradictory status information.

---

## The Solution

Use `StatusReconciliation` to merge both systems and determine the **canonical truth**.

### Status Authority Hierarchy

When reconciling, use this priority:

1. **Parser Status** (if URL was processed)
   - Authority: Direct evidence that parser ran
   - Examples: `success`, `fail`, `error`, `partial`
   - Reason: Most accurate representation of what happened

2. **Worklist Status** (if parser never touched the URL)
   - Authority: Google Sheets workflow tracking
   - Examples: `PROD Loaded`, `QC Loaded`, `Download Needed`
   - Reason: Fallback when parser hasn't processed yet

3. **Default** (if both are missing)
   - Status: `pending`
   - Reason: URL has no tracking data

### Examples

#### Example 1: Parser succeeded, in production

```python
parser_status = "success"
worklist_status = "PROD Loaded"

# Result: canonical_status = "success"
# Authority: "parser" (actual execution takes priority)
# Display: ✅ Success (parser status is authoritative)
```

#### Example 2: Parser never ran, worklist shows production

```python
parser_status = None
worklist_status = "PROD Loaded"

# Result: canonical_status = "production"
# Authority: "worklist"
# Display: 📦 Production (worklist is authority since parser didn't run)
```

#### Example 3: Parser failed, worklist shows QC

```python
parser_status = "fail"
worklist_status = "QC Loaded"

# Result: canonical_status = "fail"
# Authority: "parser" (parser status overrides worklist)
# Display: ❌ Failed (parser failure is more recent/accurate)
```

#### Example 4: No tracking anywhere

```python
parser_status = None
worklist_status = None

# Result: canonical_status = "pending"
# Authority: "default"
# Display: ⏳ Pending (no evidence of processing)
```

---

## API Response

### `/api/url_status` Response

```json
{
  "success": true,
  "total": 213,
  "filtered": 42,
  "entries": [
    {
      "url": "https://apps.azsos.gov/election/2024/ge/...",
      "label": "AZ President 2024",

      // Raw statuses (for debugging)
      "parser_status": "pending",
      "worklist_status": "PROD Loaded",

      // Reconciled status (use THIS for display)
      "canonical_status": "production",
      "status_info": {
        "icon": "📦",
        "label": "Production",
        "badge_class": "success",
        "authority": "worklist",
        "source": "google_sheets",
        "last_processed": null,
        "parsed": false,
        "in_worklist": true
      },

      // Additional metadata
      "in_production": true,
      "production_source": "google_sheets",
      "last_processed": null,
      "state": "Arizona",
      "county": null
    }
  ],
  "status_breakdown": {
    "production": 100,
    "qc_complete": 20,
    "pending": 50,
    "fail": 6,
    "error": 2,
    "dl1_processing": 3,
    "preqc_failed": 2
  },
  "canonical_statuses": ["production", "qc_complete", "pending", "fail", "error", "dl1_processing", "preqc_failed"]
}
```

### Key Fields

- **`canonical_status`**: Use this for display (already reconciled)
- **`status_info`**: Badge info (icon, label, color, authority)
- **`parser_status`**: Raw parser status (for debugging)
- **`worklist_status`**: Raw worklist status (for debugging)
- **`authority`**: Which system is authoritative for this URL

---

## Parser Status Values

Status values from `.processed_urls`:

| Status | Icon | Meaning | Notes |
| -------- | ------ | --------- | ------- |
| `success` | ✅ | URL parsed successfully | Data extracted and ready |
| `fail` | ❌ | Parsing failed | Extraction attempt failed; check logs |
| `error` | ⚠️ | Execution error | Script error, not parsing error |
| `partial` | 🔸 | Partial success | Some data extracted, some missing |
| `cancelled` | ⏹️ | Processing cancelled | User or system stopped parsing |
| `rejected` | 🚫 | Output rejected | Data quality check failed |
| `quarantined` | ⚠️ | Quarantine hold | Suspicious data, manual review pending |
| `skipped_data_exists` | ⏭️ | Skipped (in production) | URL already in database, skipped parsing |
| `pending` | ⏳ | Not processed | Parser has not attempted this URL yet |

---

## Worklist Status Values

Status values from Google Sheets:

| Status | Icon | Meaning | Authority |
| -------- | ------ | --------- | ----------- |
| `PROD Loaded` | 📦 | In production | Final/official status |
| `QC Loaded` | ✓ | QC complete | Quality check passed |
| `QC2 Fail/Fix` | ❌ | QC2 failed | Round 2 quality check failed |
| `QC1 Fail/Fix` | ❌ | QC1 failed | Round 1 quality check failed |
| `Pre-QC Fail/Fix` | ❌ | Pre-QC failed | Initial quality check failed |
| `Cand Check DL1` | 🔍 | Candidate check | DL1 reviewing candidates |
| `Download Needed` | 📥 | Download needed | URL needs to be downloaded/parsed |
| `DL1 Processing` | ⚙️ | DL1 processing | DL1 currently working on this |
| `DL2 Processing` | ⚙️ | DL2 processing | DL2 currently working on this |
| `Draft` | 📝 | Draft | Initial/incomplete status |
| (blank) | ○ | Not tracked | No workflow status yet |

---

## Data Privacy (PII Filtering)

The following columns are **automatically hidden** from all public API responses:

- `Work in Progress - DL1` (contains personal names)
- `Work in Progress - DL2` (contains personal names)
- `Assigned To`
- `Email`
- `Phone`

**API Behavior**:

- Query param: `?hide_pii=true` (default)
- Set `?hide_pii=false` only for internal, authenticated endpoints
- Dashboard always requests with `hide_pii=true`

---

## Usage in Dashboard

### Display Status Badge

```javascript
// Use 'canonical_status' for display
const entry = apiResponse.entries[0];
const status = entry.status_info;

// Render badge
badge.innerHTML = `${status.icon} ${status.label}`;
badge.className = `badge badge-${status.badge_class}`;

// Show sub-text if parsed
if (entry.status_info.parsed) {
  subtext.textContent = `Parser: ${entry.parser_status} (${entry.last_processed})`;
}
```

### Hide "Parse" Button if Appropriate

```javascript
// Don't show parse button if:
// 1. Status is already in production
// 2. Status requires manual action
// 3. Status is processing

const shouldShowParseButton = ![
  'production', 'qc_complete', 'dl1_processing',
  'dl2_processing', 'qc2_failed', 'qc1_failed',
  'preqc_failed'
].includes(entry.canonical_status);

parseButton.hidden = !shouldShowParseButton;
```

### Filter by Canonical Status

```javascript
// Get only production URLs
fetch('/api/url_status?status=production&limit=50')
  .then(r => r.json())
  .then(data => {
    console.log(`${data.status_breakdown.production} URLs in production`);
  });

// Get only failed URLs
fetch('/api/url_status?status=fail&limit=50')
  .then(r => r.json())
  .then(data => {
    console.log(`${data.status_breakdown.fail} URLs failed`);
  });
```

---

## Testing Status Reconciliation

### Test Case 1: Parser Success Overrides Worklist

```python
from webapp.parser.utils.status_reconciliation import StatusReconciliation

canonical, info = StatusReconciliation.reconcile(
    url="https://example.com/data",
    parser_status="success",
    worklist_status="QC Loaded",
    production_source=None,
    last_processed="2026-02-19 12:34:56"
)

assert canonical == "success"
assert info['authority'] == "parser"
assert info['parsed'] == True
```

### Test Case 2: Worklist Used When Parser Missing

```python
canonical, info = StatusReconciliation.reconcile(
    url="https://example.com/data",
    parser_status=None,
    worklist_status="PROD Loaded",
    production_source="google_sheets",
    last_processed=None
)

assert canonical == "production"
assert info['authority'] == "worklist"
assert info['parsed'] == False
```

### Test Case 3: Default to Pending

```python
canonical, info = StatusReconciliation.reconcile(
    url="https://example.com/data",
    parser_status=None,
    worklist_status=None,
    production_source=None,
    last_processed=None
)

assert canonical == "pending"
assert info['authority'] == "default"
```

---

## Implementation Details

### Module: `webapp/parser/utils/status_reconciliation.py`

**Key Classes**:

1. **`StatusReconciliation`**
   - `reconcile()` — Main reconciliation logic
   - `get_status_priority()` — Sort order (for UI)
   - `status_requires_action()` — Needs manual intervention?
   - `status_is_complete()` — Processing finished?

2. **`WorklistParser`**
   - `sanitize_row()` — Remove PII columns
   - `extract_contest_key()` — Create URL-to-contest matcher
   - `get_public_columns()` — Safe columns for public display

### Example: Manual Reconciliation

```python
from webapp.parser.utils.status_reconciliation import StatusReconciliation

# URL data from database
url = "https://apps.azsos.gov/election/2024/ge/..."
parser_status = processed_map[url].get('status')  # "pending"
worklist_status = sheets_data[contest_key].get('status')  # "PROD Loaded"

# Reconcile
canonical_status, status_info = StatusReconciliation.reconcile(
    url=url,
    parser_status=parser_status,
    worklist_status=worklist_status,
    production_source="google_sheets",
    last_processed=None
)

# canonical_status = "production"
# status_info.authority = "worklist"
# status_info.icon = "📦"

# Use in UI
print(f"{status_info['icon']} {status_info['label']}")  # 📦 Production
```

---

## FAQ

**Q: Why does my URL show "Pending" in the dashboard but "PROD Loaded" in Google Sheets?**

A: The parser has never run on this URL (parser_status = None), so the worklist status shows as the canonical authority. The dashboard now displays "📦 Production" instead of "⏳ Pending".

**Q: How do I see the raw parser status in the dashboard?**

A: Hover over the badge, or look at the API response field `parser_status`. The dashboard shows both:

- **Primary**: `canonical_status` (what you should care about)
- **Secondary**: `parser_status` (for debugging)

**Q: What if parser says "success" but Google Sheets says "Pre-QC Fail/Fix"?**

A: Parser status wins. The system shows `canonical_status="success"` and `authority="parser"`. This indicates the parser ran successfully, but QC found issues. You should investigate why QC failed on a successful parse.

**Q: Are personal names visible in the API?**

A: No. By default, `?hide_pii=true` removes all columns containing personal names. This is required for public-facing dashboards.

---

## Related Documentation

- [DATABASE_COMPARISON.md](../FEATURES/DATABASE_COMPARISON.md) — How production data is checked
- [SELENIUM_NLP_INTEGRATION.md](../FEATURES/SELENIUM_NLP_INTEGRATION.md) — Parser fallback strategies
- [URL_STATUS_SYSTEM_REFERENCE.md](./URL_STATUS_SYSTEM_REFERENCE.md) — Original report baseline
