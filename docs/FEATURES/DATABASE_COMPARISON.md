# Database Comparison Feature

## Overview

The Database Comparison feature prevents re-parsing URLs that already have finalized data available in the database or Google Sheets. This saves resources and avoids duplicate processing.

## How It Works

Before launching the parser for each URL, the system checks three data sources in order:

1. **Google Sheets "Finalized Data" tab**
   - Checks for exact URL matches
   - Also checks normalized URLs (ignoring query params, trailing slashes)
   - Returns state, county, contest, and candidate count metadata

2. **Warehouse Database (`warehouse_election_results` table)**
   - Queries for rows with matching `source_url`
   - Returns aggregated data: row count, candidate count, contest info
   - Only matches URLs with existing parsed data

3. **Verified Datasets (`verified_datasets` table)**
   - Checks for URLs with approved QA status
   - Looks for statuses: `approved`, `verified`, or `finalized`
   - Returns verification timestamp and QA metadata

If data is found in any source, the URL is **skipped** and marked in `.processed_urls` as:

```json
{
  "url": "https://...",
  "status": "skipped_data_exists",
  "data_source": "google_sheets|warehouse|verified_datasets",
  "retrieved_from_database": true,
  "state": "Georgia",
  "county": "Fulton",
  "contest": "President",
  "timestamp": "2026-01-23 15:30:00"
}
```

## Usage

### Enable/Disable Database Checks

Database checks are **enabled by default**. To disable:

```python
# In Python code
main(urls=urls, skip_database_check=True)
```

```bash
# Via CLI (future enhancement)
python webapp/parser/html_election_parser.py --skip-database-check
```

### Web UI Integration

The web UI automatically performs database checks before processing URLs. Users see:

- **Skipped URLs** with reason: "Finalized data exists in [source]"
- **Processing only new URLs** without existing data
- **Summary statistics** showing skipped vs. processed counts

### Force Re-Parse

To force re-parsing of URLs despite existing data:

```python
# Python
main(urls=urls, skip_database_check=True)
```

```bash
# CLI (future)
python webapp/parser/html_election_parser.py --force-reparse
```

## API Integration

The database comparison logic is available via:

```python
from webapp.parser.utils.database_comparison import check_existing_finalized_data

# Check a single URL
data_exists, data_source, metadata = check_existing_finalized_data(
    url="https://results.enr.clarityelections.com/GA/Fulton/...",
    session_id="optional_session_id",
    state="Georgia",  # Optional hint
    county="Fulton"   # Optional hint
)

if data_exists:
    print(f"Data exists in {data_source}")
    print(f"Metadata: {metadata}")
else:
    print("No existing data - proceed with parsing")
```

## Benefits

1. **Reduced Processing Time**
   - Skip URLs with existing finalized data
   - Focus resources on new/updated URLs

2. **Accurate Tracking**
   - `.processed_urls` reflects true data source
   - Distinguish between "parsed" vs. "retrieved from database"

3. **Cost Savings**
   - Avoid redundant API calls
   - Reduce browser automation overhead
   - Minimize database writes

4. **Better Reporting**
   - Know which URLs have authoritative data
   - Identify gaps in finalized data coverage

## Configuration

Environment variables (optional):

```env
# Google Sheets credentials (required for Sheets checks)
GOOGLE_SERVICE_ACCOUNT_JSON=path/to/credentials.json
GOOGLE_SHEETS_ELECTION_DATA_ID=your_sheet_id

# Database connection (required for warehouse checks)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=election_data
POSTGRES_USER=parser
POSTGRES_PASSWORD=secure_password

# Feature toggles (optional)
SKIP_DATABASE_CHECK=false  # Default: false (checks enabled)
```

## Logging

Database comparison events are logged with type `"database"`:

```json
{
  "level": "INFO",
  "type": "database",
  "message": "[DatabaseComparison] Checking for existing finalized data: https://...",
  "session_id": "sess_abc123",
  "url": "https://..."
}
```

Successful matches:

```json
{
  "level": "INFO",
  "type": "database",
  "message": "[DatabaseComparison] Found existing data in google_sheets for https://...",
  "session_id": "sess_abc123",
  "url": "https://...",
  "data_source": "google_sheets",
  "metadata": {"state": "Georgia", "county": "Fulton", ...}
}
```

Skipped URLs:

```json
{
  "level": "INFO",
  "type": "database",
  "message": "[DatabaseComparison] Skipping URL - finalized data exists in warehouse",
  "session_id": "sess_abc123",
  "url": "https://...",
  "data_source": "warehouse"
}
```

## Testing

Run the test suite:

```bash
python test_database_comparison.py
```

Expected output:

```txt
🔬 Database Comparison Feature Tests

================================================================================
Database Comparison Feature Test
================================================================================

[Test 1] Fulton County, GA - Known to have finalized data
URL: https://results.enr.clarityelections.com/GA/Fulton/115229/web.308426/#/summary
Expected data exists: True
--------------------------------------------------------------------------------
✓ Data exists: True
✓ Data source: google_sheets
✓ Metadata: {'state': 'Georgia', 'county': 'Fulton', ...}
✅ Test PASSED - Result matches expectation

[Test 2] Fake URL - Should NOT have finalized data
URL: https://example.com/fake-election-results
Expected data exists: False
--------------------------------------------------------------------------------
✓ Data exists: False
✓ Data source: None
✅ Test PASSED - Result matches expectation

================================================================================
Test Summary
================================================================================
Total tests: 2
Passed: 2 ✅
Failed: 0 ❌
Errors: 0 ⚠️

🎉 All tests passed!
```

## Troubleshooting

### Database check always returns False

**Causes:**

- Google Sheets credentials not configured
- Database connection unavailable
- Table schemas don't match expected structure

**Solutions:**

1. Verify `.env` has correct credentials
2. Check database connectivity: `python -c "from webapp.parser.utils.db_utils import get_engine; get_engine().connect()"`
3. Ensure tables exist: `warehouse_election_results`, `verified_datasets`

### URLs marked as "skipped" but should be parsed

**Cause:** Data exists in source but may be outdated

**Solution:** Use `--skip-database-check` or `--force-reparse` flag

### .processed_urls not updating with database metadata

**Cause:** `mark_url_processed()` called without database metadata

**Solution:** Ensure integration passes metadata:

```python
mark_url_processed(
    url,
    status="skipped_data_exists",
    data_source=data_source,
    retrieved_from_database=True,
    **(metadata or {})
)
```

## Future Enhancements

- [ ] Add CLI flags: `--skip-database-check`, `--force-reparse`
- [ ] Cache database check results per session
- [ ] Add metrics: skipped vs. processed ratio
- [ ] Support partial re-parsing (update only changed contests)
- [ ] Add stale data detection (check last_updated timestamps)

## Related Files

- `webapp/parser/utils/database_comparison.py` - Core comparison logic
- `webapp/parser/html_election_parser.py` - Integration into main() function
- `test_database_comparison.py` - Test suite
- `.processed_urls` - Tracking file with database metadata
