# URL Status Dashboard & Report System

## What We Built

### 1. Quick Report Script (`tools/url_status_report.py`)

**Purpose**: Baseline analysis tool to see what data we have

**Features**:

- Reads all URLs from `urls.txt` (213 URLs found)
- Checks `.processed_urls` for processing history
- Queries production databases (Google Sheets + warehouse)
- Generates Markdown + CSV reports

**Usage**:

```bash
python tools/url_status_report.py [--format md|csv|both] [--output-dir output]
```

**Output Example**:

```txt
Total URLs:           213
Parsed (Success):     1 (0.5%)
Failed/Error:         6 (2.8%)
In Production:        0 (0.0%)
Pending:              206 (96.7%)
```

Generated Reports:

- `output/url_status_report_YYYYMMDD_HHMMSS.md`
- `output/url_status_report_YYYYMMDD_HHMMSS.csv`

---

### 2. URL Status Dashboard (`/url_status_dashboard`)

**Purpose**: Interactive web dashboard for URL tracking and management

**Features**:
✅ **Real-time Status Overview**

- Total URLs, Parsed, Failed, Pending, In Production
- Visual pie chart showing status breakdown
- Bar chart showing top states by pending URLs

✅ **Advanced Filtering**

- Filter by status (success/fail/error/pending/etc.)
- Filter by state, county
- Date range filtering
- Pagination support

✅ **Detailed URL Table**

- URL, Label, Parser Status, Production Status
- Last Processed date
- Quick actions: Re-parse button for failed/pending URLs
- Direct links to source URLs

✅ **Export Capabilities**

- Export filtered results to CSV
- All filters applied to export

**Access**:

1. Start Flask: `python -m webapp.Smart_Elections_Parser_Webapp`
2. Navigate to: `http://localhost:5000/url_status_dashboard`
3. Or click "URL Status" in navbar

---

### 3. Enhanced Worklist Import Modal

**Purpose**: Import URLs from Google Sheets worklist with status preview

**Features**:

- Import button in ballot_lens URL form
- Modal shows worklist data with checkboxes
- **Status Validation**: Each URL checked against:
  - ✓ Finalized (green) - Already in Google Sheets/warehouse
  - ⚠ Duplicate (yellow) - Duplicate entry
  - ○ New (gray) - Ready to parse
- Bulk select/deselect
- Merges all 3 URL columns (Download 1, Download 2, Source Link)
- Automatic deduplication when adding to URL input

**Usage**:

1. Go to ballot_lens
2. Click "📋 Import from Worklist" button
3. Select desired rows
4. Click "Import Selected URLs"
5. URLs added to batch input field

---

### 4. API Endpoints

#### `/api/url_status` (GET)

Query processed URLs with filtering

**Parameters**:

- `status` - Filter by status (success|fail|error|pending|cancelled|skipped_data_exists)
- `state` - Filter by state name
- `county` - Filter by county name
- `from_date` - Filter by date (YYYY-MM-DD)
- `to_date` - Filter by date (YYYY-MM-DD)
- `limit` - Max results (default: 100, max: 1000)
- `offset` - Pagination offset

**Response**:

```json
{
  "success": true,
  "total": 213,
  "filtered": 42,
  "limit": 100,
  "offset": 0,
  "entries": [
    {
      "url": "https://...",
      "label": "2024 President Arizona",
      "parser_status": "fail",
      "in_production": false,
      "production_source": null,
      "last_processed": "2026-01-30 17:13:16",
      "state": "Arizona",
      "county": null
    }
  ],
  "status_breakdown": {
    "success": 1,
    "fail": 6,
    "pending": 206
  }
}
```

#### `/api/validate_urls` (POST)

Validate URLs against existing finalized data (used by worklist modal)

**Request**:

```json
{
  "urls": ["https://...", "https://..."]
}
```

**Response**:

```json
{
  "success": true,
  "results": [
    {
      "url": "https://...",
      "exists": true,
      "source": "google_sheets",
      "metadata": {
        "state": "Arizona",
        "county": "Maricopa",
        "contest": "President"
      }
    }
  ]
}
```

---

## Status Values Tracked

| Status | Meaning | Badge |
| -------- | --------- | ------- |
| `success` | Parsed successfully, output generated | ✅ Success |
| `fail` | Handler failed or download error | ❌ Failed |
| `error` | Exception during processing | ⚠️ Error |
| `pending` | Not yet processed | ⏳ Pending |
| `partial` | Incomplete result structure | 🔸 Partial |
| `cancelled` | User cancelled parsing | ⏹️ Cancelled |
| `skipped_data_exists` | Already in production (Google Sheets/warehouse) | ⏭️ Skipped |
| `rejected` | Low trust score (security) | 🚫 Rejected |
| `quarantined` | Awaiting manual review | ⚠️ Quarantine |

---

## File Locations

**Data Storage**:

- `.processed_urls` - JSON array tracking all processed URLs
  - Location: `webapp/parser/Context_Integration/Context_Library/.processed_urls`
- `urls.txt` - Master URL list (213 URLs)
  - Location: `webapp/parser/urls.txt`

**Reports**:

- Markdown reports: `output/url_status_report_*.md`
- CSV reports: `output/url_status_report_*.csv`

**Templates**:

- Dashboard: `webapp/templates/url_status_dashboard.html`
- Worklist: `webapp/templates/worklist.html`
- Ballot Lens: `webapp/templates/ballot_lens.html`

**Scripts**:

- Report generator: `tools/url_status_report.py`

---

## Current Baseline (February 19, 2026)

```txt
📊 Status Summary:
   Total URLs:           213
   Parsed (Success):     1 (0.5%)
   Failed/Error:         6 (2.8%)
   In Production:        0 (0.0%)
   Pending:              206 (96.7%)

⚠️ Gap Analysis:
   212 URLs need attention
   - 206 pending (not yet processed)
   - 6 failed (need retry/investigation)

📋 URL List Details:
   - Well-organized with tab-delimited schema
   - Columns: year, contest, state, county, format, notes, URL
   - Covers 2024 General Election (President, Senate, House)
   - Multiple formats: PDF, CSV, HTML, XLSX, TXT, JSON
   - 50+ states/territories represented

💡 Next Steps:
   1. Review failed URLs in dashboard (6 need attention)
   2. Start batch processing pending URLs (206 waiting)
   3. Monitor production sync status
   4. Set up automated worklist imports
```

---

## Workflow Integration

### Before Parsing

1. **Check Production Status**
   - Run: `python tools/url_status_report.py`
   - Or: Open `/url_status_dashboard`
   - Identify URLs already in production (skip unnecessary work)

2. **Import from Worklist**
   - Open ballot_lens
   - Click "Import from Worklist"
   - Select URLs with ○ New badge
   - Import to batch

3. **Verify Status**
   - Dashboard shows which URLs are pending
   - Failed URLs highlighted for retry

### During Parsing

- Real-time progress events show status updates
- `.processed_urls` updated automatically
- Status visible in dashboard immediately

### After Parsing

1. **Review Results**
   - Dashboard auto-refreshes
   - Check success/fail counts
   - Investigate failures

2. **Export Reports**
   - Click "Export CSV" for spreadsheet analysis
   - Or run report script for Markdown docs

3. **Sync to Production**
   - Verified data moves to Google Sheets
   - Warehouse sync happens automatically
   - Next run will show "⏭️ Skipped" for finalized URLs

---

## Tips & Best Practices

✅ **Run baseline report before big batches**

```bash
python tools/url_status_report.py --format both
```

✅ **Use filters to focus on problem areas**

- Dashboard: Filter by status="fail" to review errors
- Dashboard: Filter by state to process region by region

✅ **Monitor pending vs production ratio**

- High pending count = need more parsing
- High production count = avoid duplicate work

✅ **Re-parse failed URLs strategically**

- Check error patterns first (dashboard shows last processed date)
- Retry after fixing known issues
- Use "Re-parse" button in dashboard

✅ **Export before major changes**

- Backup current state as CSV
- Track progress over time
- Share status with team

---

## Future Enhancements (Suggested)

1. **Auto-Retry Logic**
   - Automatically retry failed URLs with exponential backoff
   - Track retry count in `.processed_urls`

2. **Bi-directional Worklist Sync**
   - Update Google Sheets with parser status
   - Show "Parser Status" column in worklist

3. **Alert System**
   - Email/Slack notifications for high failure rates
   - Daily status summary reports

4. **State-by-State Progress**
   - Visual map showing completion by state
   - Priority-based queue management

5. **Historical Trending**
   - Track success rate over time
   - Identify degrading URL sources

6. **Batch Operations**
   - Bulk re-parse selection from dashboard
   - Batch mark as reviewed/ignored

---

## Testing the System

### Quick Test

```bash
# 1. Generate baseline report
python tools/url_status_report.py

# 2. Start Flask server
python -m webapp.Smart_Elections_Parser_Webapp

# 3. Open dashboard
# Navigate to: http://localhost:5000/url_status_dashboard

# 4. Test worklist import
# Navigate to: http://localhost:5000/ballot_lens
# Click "Import from Worklist" button
```

### Full Integration Test

1. Open ballot_lens
2. Import URLs from worklist
3. Parse a few URLs
4. Refresh dashboard - see updated counts
5. Export CSV report
6. Verify status badges show correctly

---

## Documentation Links

- Main parser README: `README.md`
- Database comparison: `docs/FEATURES/DATABASE_COMPARISON.md`
- Worklist integration: (this doc)
- Quality dashboard: `webapp/templates/quality_dashboard.html`

---

**Built**: February 19, 2026
**Last Updated**: February 19, 2026
**Status**: ✅ Production Ready
