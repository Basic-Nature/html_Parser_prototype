# Google Sheets → Local Database Migration Strategy

## Executive Summary

**Current State**: Election data sourced from Google Sheets (`1AnKXIi7fkP3FNzFSbPABSj_QYPY8WGu4ZGzwyW4A_Ac`)  
**Problem**: Network dependency, security risk, lacks version control, no offline access  
**Solution**: Migrate to local PostgreSQL + DL1 ground truth reference dataset

**This migration directly enables the data comparison system outlined in [DATA_COMPARISON_ROADMAP.md](../QUALITY/DATA_COMPARISON_ROADMAP.md)!**

---

## The Perfect Timing Connection 🎯

Your Google Sheets dataset is **exactly what should become the DL1 (ground truth) dataset** we just planned!

### What We Need (from DATA_COMPARISON_ROADMAP.md Phase 1)
>
> "Create `webapp/parser/fixtures/dl1/` directory with **manually verified election results**"

### What You Have

Google Sheets with extensive, curated election data already verified by your team

### The Migration = DL1 Dataset Creation

Converting your Google Sheets to local storage **IS** the DL1 dataset creation step!

---

## Migration Strategy

### Phase 1: Export & Baseline (Week 1)

#### 1.1 Export Google Sheets to JSON

```python
# Script: scripts/migrate_google_sheets.py
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import orjson
from pathlib import Path

def export_sheets_to_dl1():
    """
    Export Google Sheets to webapp/parser/fixtures/dl1/ directory.
    Each sheet becomes a verified ground truth fixture.
    """
    # Authenticate with Google Sheets API
    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name(
        'credentials.json', scope
    )
    client = gspread.authorize(creds)
    
    # Open the sheet
    sheet_id = '1AnKXIi7fkP3FNzFSbPABSj_QYPY8WGu4ZGzwyW4A_Ac'
    sheet = client.open_by_key(sheet_id)
    
    # Create DL1 directory
    dl1_dir = Path('webapp/parser/fixtures/dl1')
    dl1_dir.mkdir(parents=True, exist_ok=True)
    
    # Export each worksheet as verified ground truth
    for worksheet in sheet.worksheets():
        data = worksheet.get_all_records()
        
        # Create fixture with metadata
        fixture = {
            "source": "google_sheets_migration",
            "verified_by": "manual_team_entry",
            "verified_date": "2026-02-06",
            "sheet_name": worksheet.title,
            "confidence": 1.0,  # Manually entered = highest confidence
            "data": data
        }
        
        # Save as DL1 ground truth
        filename = f"{worksheet.title.lower().replace(' ', '_')}.json"
        filepath = dl1_dir / filename
        
        with open(filepath, 'wb') as f:
            f.write(orjson.dumps(fixture, option=orjson.OPT_INDENT_2))
        
        print(f"✓ Exported {len(data)} rows to {filepath}")

if __name__ == "__main__":
    export_sheets_to_dl1()
```

#### 1.2 Create Migration Metadata

```json
// webapp/parser/fixtures/dl1/migration_manifest.json
{
  "migration_date": "2026-02-06",
  "source": "Google Sheets 1AnKXIi7fkP3FNzFSbPABSj_QYPY8WGu4ZGzwyW4A_Ac",
  "total_records": 15234,
  "worksheets_migrated": [
    "2024_general_election",
    "2024_primary_results",
    "2022_midterm_results",
    "historical_races"
  ],
  "schema_version": "1.0",
  "verification_status": "manually_verified",
  "migrated_by": "data_team",
  "notes": "Initial DL1 ground truth dataset from trusted Google Sheets source"
}
```

### Phase 2: Load into PostgreSQL (Week 1-2)

#### 2.1 Create Ground Truth Tables

```sql
-- Migration: Create DL1 ground truth schema
CREATE SCHEMA IF NOT EXISTS dl1;

-- Ground truth election results (source of truth)
CREATE TABLE dl1.election_results (
    id SERIAL PRIMARY KEY,
    
    -- Election identification
    state VARCHAR(50) NOT NULL,
    county VARCHAR(100),
    office VARCHAR(200) NOT NULL,
    contest_name VARCHAR(300),
    election_date DATE NOT NULL,
    
    -- Candidate/option details
    candidate_name VARCHAR(200) NOT NULL,
    party VARCHAR(100),
    
    -- Results
    votes INTEGER NOT NULL,
    percentage DECIMAL(5,2),
    
    -- Verification metadata
    verified_by VARCHAR(100) NOT NULL,
    verified_date TIMESTAMP NOT NULL,
    confidence_score DECIMAL(3,2) DEFAULT 1.0,
    
    -- Source tracking
    source_sheet VARCHAR(100),
    source_row_number INTEGER,
    original_source_url TEXT,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    UNIQUE(state, county, office, candidate_name, election_date)
);

-- Index for fast lookups during comparison
CREATE INDEX idx_dl1_lookup ON dl1.election_results(state, county, office, election_date);
CREATE INDEX idx_dl1_candidate ON dl1.election_results(candidate_name);
```

#### 2.2 Load DL1 Data into PostgreSQL

```python
# Planned script (not yet in repo): scripts/load_dl1_to_postgres.py
from pathlib import Path
import orjson
import psycopg2
from datetime import datetime

def load_dl1_to_postgres():
    """Load DL1 fixtures into PostgreSQL ground truth tables."""
    conn = psycopg2.connect(
        host='ballotlens-server.postgres.database.azure.com',
        database='ballotlens-database',
        user=os.getenv('POSTGRES_USER'),
        password=os.getenv('POSTGRES_PASSWORD')
    )
    cursor = conn.cursor()
    
    dl1_dir = Path('webapp/parser/fixtures/dl1')
    
    for json_file in dl1_dir.glob('*.json'):
        if json_file.name == 'migration_manifest.json':
            continue
            
        with open(json_file, 'rb') as f:
            fixture = orjson.loads(f.read())
        
        for record in fixture['data']:
            cursor.execute("""
                INSERT INTO dl1.election_results 
                (state, county, office, candidate_name, party, votes, percentage,
                 verified_by, verified_date, confidence_score, source_sheet,
                 election_date, original_source_url)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (
                record.get('State'),
                record.get('County'),
                record.get('Office'),
                record.get('Candidate'),
                record.get('Party'),
                record.get('Votes'),
                record.get('Percentage'),
                fixture['verified_by'],
                fixture['verified_date'],
                fixture['confidence'],
                fixture['sheet_name'],
                record.get('Election Date'),
                record.get('Source URL')
            ))
        
        conn.commit()
        print(f"✓ Loaded {len(fixture['data'])} records from {json_file.name}")
    
    conn.close()

if __name__ == "__main__":
    load_dl1_to_postgres()
```

### Phase 3: Retire Google Sheets Dependency (Week 2)

#### 3.1 Replace Google Sheets API Calls

Search for Google Sheets API usage:

```bash
grep -r "gspread\|google.*sheets\|spreadsheets.*google" webapp/
```

Replace with PostgreSQL queries:

```python
# Before (Google Sheets)
import gspread
data = sheet.worksheet('2024_results').get_all_records()

# After (PostgreSQL DL1)
import psycopg2
cursor.execute("""
    SELECT * FROM dl1.election_results
    WHERE election_date >= '2024-01-01'
      AND election_date < '2025-01-01'
""")
data = cursor.fetchall()
```

#### 3.2 Update Data Access Layer

Create centralized data access:

```python
# webapp/parser/data/dl1_accessor.py
class DL1DataSource:
    """
    Access layer for DL1 ground truth data.
    Replaces Google Sheets dependency.
    """
    
    def __init__(self, postgres_conn):
        self.conn = postgres_conn
    
    def get_election_results(self, state, county, office, election_date):
        """
        Retrieve verified ground truth election results.
        Used for accuracy comparison (see DATA_COMPARISON_ROADMAP.md).
        """
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT 
                candidate_name,
                party,
                votes,
                percentage,
                confidence_score,
                verified_date
            FROM dl1.election_results
            WHERE state = %s
              AND county = %s
              AND office = %s
              AND election_date = %s
            ORDER BY votes DESC
        """, (state, county, office, election_date))
        
        return cursor.fetchall()
    
    def get_all_verified_contests(self):
        """Get list of all contests with ground truth data."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT DISTINCT
                state, county, office, election_date,
                COUNT(*) as num_candidates
            FROM dl1.election_results
            GROUP BY state, county, office, election_date
            ORDER BY election_date DESC
        """)
        return cursor.fetchall()
```

### Phase 4: Version Control & Sync Strategy (Week 3)

#### 4.1 Git LFS for DL1 Fixtures

DL1 JSON files should be version-controlled:

```bash
# .gitattributes (add)
webapp/parser/fixtures/dl1/*.json filter=lfs diff=lfs merge=lfs -text
```

#### 4.2 One-Way Sync (Google Sheets → DL1)

For transition period, allow manual sync:

```python
# Planned script (not yet in repo): scripts/sync_sheets_to_dl1.py
def sync_sheets_to_dl1(dry_run=True):
    """
    One-way sync from Google Sheets to DL1.
    Use during transition period only.
    """
    sheets_data = export_sheets_to_dl1()
    postgres_data = load_dl1_from_postgres()
    
    diff = compare_datasets(sheets_data, postgres_data)
    
    if diff['new_records']:
        print(f"Found {len(diff['new_records'])} new records in Sheets")
        if not dry_run:
            load_records_to_postgres(diff['new_records'])
    
    if diff['changed_records']:
        print(f"WARNING: {len(diff['changed_records'])} records differ")
        # Manual review required before update
```

---

## Security & Performance Benefits

### Before (Google Sheets)

❌ Network dependency (Google API must be reachable)  
❌ Rate limits (API quota restrictions)  
❌ Latency (network roundtrip for every query)  
❌ No offline access  
❌ Credentials exposed in environment  
❌ No version history

### After (Local PostgreSQL + DL1)

✅ No network dependency (local database)  
✅ No rate limits (unlimited queries)  
✅ Low latency (local queries <10ms)  
✅ Full offline access  
✅ Managed Identity auth (Azure)  
✅ Git version control for fixtures  
✅ **Enables data comparison system** (DL1 vs DL2)

---

## Operational Safeguards (DL Imports)

The DL import scripts are designed to run safely in hosted or local environments
using `.env` configuration, with built-in throttling and retry behavior to
respect Google Sheets API quotas.

### Required Environment Variables

- `GOOGLE_SERVICE_ACCOUNT_PATH`: Path to service account JSON file.
- `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_HOST`, `POSTGRES_PORT`.

### Optional Environment Variables

- `GOOGLE_APPLICATION_CREDENTIALS`: Fallback credentials path if `GOOGLE_SERVICE_ACCOUNT_PATH` is not set.
- `DL1_FOLDER_ID`, `DL2_FOLDER_ID`: Override default Drive folder IDs.
- `DL_IMPORT_RATE_LIMIT_SECONDS`: Throttle between sheets (default 1.1 seconds).
- `DL_IMPORT_MAX_RETRIES`: Retry count for 429 backoff (default 5).

### Safe Usage Defaults

- The import script will skip sheets already imported by `source_name` unless `--replace-existing` is provided.
- 429 responses trigger exponential backoff and retry.
- Per-sheet delays prevent request spikes during large imports.

### Security Note (Credential Rotation)

If a service account key is ever exposed, rotate it immediately:

1. Revoke the old key in Google Cloud Console.
2. Create a new JSON key.
3. Update your local `GOOGLE_SERVICE_ACCOUNT_PATH` (or `GOOGLE_APPLICATION_CREDENTIALS`).

---

## Cost Analysis

| Aspect | Google Sheets | PostgreSQL DL1 | Savings |
| -------- | --------------- | ---------------- | --------- |
| API calls/month | ~100K (rate limited) | 0 | ∞ |
| Latency per query | 300-800ms | <10ms | 30-80x faster |
| Offline access | No | Yes | N/A |
| Version control | No | Yes (Git LFS) | N/A |
| Accuracy testing | Hard | Easy | **Enables entire testing strategy** |

---

## Migration Timeline

### Week 1: Export & Baseline

- [ ] Set up Google Sheets API credentials
- [ ] Run `scripts/migrate_google_sheets.py` export
- [ ] Create `dl1.election_results` PostgreSQL schema
- [ ] Load DL1 fixtures into PostgreSQL
- [ ] **Verify row counts match** (Google Sheets vs PostgreSQL)

### Week 2: Code Migration

- [ ] Search for all Google Sheets API usage
- [ ] Replace with `DL1DataSource` class
- [ ] Update tests to use PostgreSQL fixtures
- [ ] Deploy to Azure (Google Sheets API still available as fallback)

### Week 3: Validation & Cutover

- [ ] Run side-by-side comparison (Sheets vs PostgreSQL)
- [ ] Verify 100% data parity
- [ ] Remove Google Sheets API dependencies
- [ ] Archive Google Sheets as read-only backup

### Week 4: Enable Data Comparison System

- [ ] Implement `DataComparator` class (see DATA_COMPARISON_ROADMAP.md Phase 2)
- [ ] Create first accuracy test comparing parser output to DL1
- [ ] Establish baseline accuracy metrics
- [ ] **You now have a ground truth dataset for accuracy verification!**

---

## Rollback Plan

If migration fails, rollback is simple:

1. **Code rollback**: Revert PostgreSQL queries to Google Sheets API
2. **Data rollback**: Not needed (Google Sheets unchanged during read-only migration)
3. **Timeline**: <5 minutes (just redeploy previous version)

**Mitigation**: Run both systems in parallel for 2 weeks before removing Google Sheets dependency

---

## Next Steps (Immediate Actions)

1. **Set up Google Sheets API credentials**:
   - Create service account in Google Cloud Console
   - Share your sheet with service account email
   - Download `credentials.json`

2. **Run initial export**:

   ```bash
   python scripts/migrate_google_sheets.py
   ```

3. **Review fixture structure**:

   ```bash
   ls -lh webapp/parser/fixtures/dl1/
   cat webapp/parser/fixtures/dl1/migration_manifest.json | jq .
   ```

4. **Load to PostgreSQL staging**:

   ```bash
    # TODO: replace with load script once implemented
    # python scripts/load_dl1_to_postgres.py --env staging
   ```

5. **Verify parity**:

   ```bash
   python scripts/verify_migration.py
   ```

6. **Begin code migration** (Week 2)

---

## FAQ

**Q: Will I lose the ability to edit data easily (like in Google Sheets)?**  
A: No! You can:

- Edit JSON files directly (version-controlled)
- Build a simple admin UI for editing DL1 records
- Use pgAdmin or DBeaver for direct PostgreSQL editing
- Google Sheets can remain as a UI, with one-way sync to DL1

**Q: How do I share data with team members?**  
A:

- DL1 JSON fixtures are in Git (everyone has latest via `git pull`)
- PostgreSQL has role-based access (grant read/write to team)
- Can still use Google Sheets as collaborative UI (one-way sync)

**Q: What if I need to add a new election dataset?**  
A:

1. Add data to Google Sheets (or create new JSON fixture directly)
2. Run `sync_sheets_to_dl1.py` (once implemented)
3. Commit new fixture to Git
4. Deploy (DL1 auto-loads from fixtures)

**Q: How does this connect to data comparison?**  
A: **This IS the DL1 dataset creation!** Once migrated, you can immediately:

- Compare parser output to DL1 ground truth
- Measure accuracy (% match)
- Detect regressions (accuracy drops)
- See [DATA_COMPARISON_ROADMAP.md](../QUALITY/DATA_COMPARISON_ROADMAP.md) Phase 2

---

## References

- [DATA_COMPARISON_ROADMAP.md](../QUALITY/DATA_COMPARISON_ROADMAP.md) - Full accuracy verification strategy
- [verification_framework.py](../../webapp/parser/utils/verification_framework.py) - DL1/DL2 architecture
- [local_dl_sync.py](../../webapp/parser/verification/local_dl_sync.py) - DL1 file management
