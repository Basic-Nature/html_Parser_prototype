# Data Import Test Runbook — Pre-Commit Verification

This runbook guides you through a complete integration test of the data pipeline before committing the infrastructure changes.

**Timeline**: ~15-30 minutes depending on data size
**Risk Level**: Low (all steps are dry-run or local DB first)
**Goal**: Verify that data flows cleanly from Google Sheets → Local DB → Warehouse → Webapp renders correctly

---

## Phase 1: Verify Credentials & Data Source (5 min)

### Step 1.1: Confirm Google Sheets credentials

```bash
# Check that your .env or GitHub secrets have these set:
$env:GOOGLE_SHEETS_DB_LITE_ID        # Should be a long string (sheet ID)
$env:GOOGLE_SHEETS_SA_PRIVATE_KEY    # Should start with -----BEGIN PRIVATE KEY-----
$env:GOOGLE_SHEETS_SA_CLIENT_EMAIL   # Should be something@*.iam.gserviceaccount.com
```

If not set, load from `.env`:

```bash
# PowerShell:
Get-Content .env | ForEach-Object {
  if ($_ -match "^([^=]+)=(.*)$") {
    $name, $value = $matches[1], $matches[2]
    [Environment]::SetEnvironmentVariable($name, $value)
  }
}
```

### Step 1.2: Inspect the Google Sheet

1. Go to Google Sheets: `https://docs.google.com/spreadsheets/d/${{ env.GOOGLE_SHEETS_DB_LITE_ID }}`
2. Check the sheet has these columns:
   - `county`, `state`, `year`, `candidate`, `party`, `votes` (or similar)
   - Verify **at least 1000+ rows** of data (ideally 169k for Finalized Data)
3. Note the **worksheet tab name** (default is first sheet, or specify with `--worksheet "Sheet Name"`)

---

## Phase 2: Dry-Run the Import (5 min)

### Step 2.1: Run import in dry-run mode

```bash
# Set Python path
$env:PYTHONPATH = (Get-Location).Path

# Run dry-run with verbose output
python scripts/import_database_lite.py `
  --sheet-id $env:GOOGLE_SHEETS_DB_LITE_ID `
  --dry-run `
  --limit 100  # Only process first 100 rows for speed
```

**Expected output:**

```txt
Sheet: Finalized Data (1a2b3c...)
Rows processed: 100
Records ready: 85-100 (some rows may be skipped if missing candidate/votes)
Skipped: X
Sample record:
{
  "state": "Arizona",
  "county": "Apache",
  "year": 2012,
  "candidate": "Barack Obama",
  "party": "DEM",
  "votes": "17147",
  ...
}
```

**Checklist:**

- [ ] No ModuleNotFoundError (Python path is correct)
- [ ] No credential errors (Google Sheets SA key is valid)
- [ ] Sheet title matches what you expected
- [ ] Row count is reasonable (>50, <200 with --limit 100)
- [ ] Sample record has all expected fields (state, county, year, candidate, party, votes)
- [ ] Sample votes are numeric and non-zero

### Step 2.2: Check for data quality issues

If you see warnings like:

- "missing candidate" — some rows have empty candidate names, OK to skip these
- "missing votes" — some rows have 0 or empty votes, these are skipped
- "skipped: X" — normal if X is <5% of total rows

**If >20% of rows are skipped:** ⚠️ Stop, investigate the sheet structure. Run with `--worksheet "Different Sheet"` if needed.

---

## Phase 3: Run Full Import (5 min)

**BEFORE THIS STEP:** Back up your local database:

```bash
# PowerShell:
Copy-Item election_data.db election_data.db.backup
```

### Step 3.1: Run the actual import

```bash
python scripts/import_database_lite.py `
  --sheet-id $env:GOOGLE_SHEETS_DB_LITE_ID `
  --batch-size 500
```

**Expected output:**

```txt
Sheet: Finalized Data (...)
Rows processed: 169540
Records ready: 165432
Skipped: 4108
Inserted 165432 rows into warehouse_election_results
```

**Checklist:**

- [ ] No database errors
- [ ] Insertion completed without rollback
- [ ] "Inserted X rows" message appears
- [ ] X > 1000 (at least thousands of rows)

### Step 3.2: Verify data was written

```bash
python -c "
from webapp.parser.utils.db_utils import SessionLocal
from webapp.parser.models import WarehouseElectionResult
session = SessionLocal()
count = session.query(WarehouseElectionResult).count()
sample = session.query(WarehouseElectionResult).first()
print(f'Total rows in DB: {count}')
print(f'Sample: {sample.__dict__}')
session.close()
"
```

**Expected:**

```txt
Total rows in DB: 165432
Sample: {'id': 1, 'state': 'Arizona', 'county': 'Apache', 'year': 2012, ...}
```

---

## Phase 4: Start the Webapp & Test Visualizations (10 min)

### Step 4.1: Start the Flask app

```bash
# Terminal 1: Start the webapp
$env:FLASK_ENV = "development"
$env:LOG_LEVEL = "DEBUG"
python -m flask --app webapp.Smart_Elections_Parser_Webapp run --host=127.0.0.1 --port=5000
```

**Expected:**

```txt
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```

### Step 4.2: Test health endpoint

```bash
# Terminal 2: Test health check
Invoke-WebRequest http://127.0.0.1:5000/health -UseBasicParsing | Select-Object -ExpandProperty Content | ConvertFrom-Json
```

**Expected:**

```json
{
  "status": "ok",
  "timestamp": "2026-03-30T15:30:45Z",
  "uptime_seconds": 2.5
}
```

### Step 4.3: Test data API endpoint

```bash
# Get URL status summary (should show some data if import worked)
Invoke-WebRequest "http://127.0.0.1:5000/api/url_status?limit=5" -UseBasicParsing | ConvertFrom-Json
```

**Expected:**

```json
{
  "success": true,
  "total": 12345,
  "filtered": 120,
  "entries": [...],
  "status_breakdown": {...}
}
```

### Step 4.4: Test data framework visualization endpoints

```bash
# Check warehouse data availability
$response = Invoke-WebRequest "http://127.0.0.1:5000/api/data_framework/warehouse_status" -UseBasicParsing
$response.Content | ConvertFrom-Json
```

**Expected responses:**

- HTTP 200 with `"available": true` — data is loaded and queryable
- HTTP 403 — data not loaded (expected if import skipped)
- HTTP 500 — database error (investigate)

---

## Phase 5: Visualization Integrity Checks (5 min)

### Step 5.1: Open browser to webapp

```txt
http://127.0.0.1:5000/
```

In the browser console (F12), check for errors:

```javascript
// Check if data framework initialized
console.log(window.dataFramework)

// Should output: { available: true, sources: [...], schemas: {...} }
```

### Step 5.2: Navigate to visualizations

**If your app has these pages, check them:**

- [ ] `/ballot_lens/` — should render state/county dropdowns with actual data
- [ ] `/dashboard/` — should show charts/tables with election data
- [ ] `/results/` — should list counties and contests from database
- [ ] Any custom viz page — verify it queries and renders data

**Common issues & fixes:**

| Issue | Fix |
| ------- | ----- |
| "No data available" | Data import didn't complete; go back to Phase 3 |
| Dropdown shows "Loading..." forever | Database connection issue; check POSTGRES_* env vars |
| Chart renders but shows 0 data | Schema mismatch; inspect `/api/data_framework/warehouse_status` response |
| Console error: "Cannot read property 'available'" | Data framework not initialized; check if Flask app returned it in response |

---

## Phase 6: Commit Checklist

Before pushing to main and triggering Azure seed:

- [ ] **Data import**: ✓ Dry-run showed reasonable row counts
- [ ] **Database writes**: ✓ Full import completed without errors
- [ ] **Sample query**: ✓ Found rows in `warehouse_election_results`
- [ ] **Health endpoint**: ✓ Returns 200 + "status": "ok"
- [ ] **Visualizations**: ✓ Webapp renders pages without JS errors
- [ ] **No new warnings**: ✓ No new alerts in Flask debug logs

If all boxes checked: **Ready to commit!**

---

## Phase 7: Deploy to Azure (Manual, ~10 min)

Once committed and pushed to `main`:

1. Go to **GitHub → Actions → Seed Warehouse → Run workflow**
2. Set inputs:
   - `import_finalized_data`: ✓ true
   - `import_voting_equipment`: leave as false (until equipment import is tested)
   - `dry_run`: leave as false
3. Press **Run workflow**
4. Monitor the logs — should take ~5 min to:
   - Open firewall
   - Import data
   - Close firewall
   - Return success

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'webapp'"

```bash
$env:PYTHONPATH = (Get-Location).Path
# Then retry the command
```

### "GOOGLE_SERVICE_ACCOUNT_PATH is not set"

Check that either:

- `.env` file exists and has `GOOGLE_SERVICE_ACCOUNT_PATH=/path/to/json`, OR
- Environment has all `GOOGLE_SHEETS_SA_*` variables set

### "Missing database connection"

```bash
# Check DATABASE_URL or POSTGRES_* env vars:
$env:DATABASE_URL; $env:POSTGRES_HOST; $env:POSTGRES_USER
```

### "Skipped >20% of rows"

The sheet structure might not match expected headers. Run:

```bash
python scripts/import_database_lite.py `
  --sheet-id $env:GOOGLE_SHEETS_DB_LITE_ID `
  --limit 1 `
  --dry-run
```

And inspect the sample record. If it's missing key fields, the header hints in `import_database_lite.py` (lines 37-59) might need updating.

---

## Next Steps After Successful Test

1. **Commit changes:**

   ```bash
   git add .github/workflows/seed-warehouse.yml scripts/import_database_lite.py
   git commit -m "feat: add warehouse seed workflow and env-var credentials support"
   git push
   ```

2. **Trigger Azure seed** (see Phase 7 above)

3. **Backfill midterm data** (once voting equipment is tested):

   ```bash
   python scripts/backfill_midterm_elections.py --backfill-all --src-dir path/to/medsl_csvs/
   ```

4. **Monitor live app:** <https://www.electionpulse.org/health>

---

**Questions?** Check the logs in `output/reports/` or `tools/debug_headless_output/`
