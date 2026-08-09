# Ballot Lens Multi-Pathway Integration Testing Guide

**Purpose**: Verify that ballot lens produces consistent, valid election data across all user navigation pathways and can be compared with database for data correctness.

## Overview

Users interact with ballot lens through multiple pathways:

1. **CLI Pathway** - Direct command-line parser invocation with URLs
2. **Webapp UI Pathway** - Form submission via browser (Ballot Lens web interface)
3. **API Pathway** - Direct Flask endpoint calls or programmatic integration
4. **Database Comparison** - Validating output against existing election data warehouse

## Test Suite Location

- **Test File**: `webapp/tests/test_ballot_lens_pathways.py`
- **Run Tests**: `pytest webapp/tests/test_ballot_lens_pathways.py -v`

## Key Features

### 1. CSV Output Validation

Every pathway must produce valid CSV election data with:

- **Required Headers**: Office, Candidate, Party, Votes (at minimum)
- **Data Integrity**: No null/empty critical fields
- **Structure**: Valid CSV format (no malformed rows)
- **Content**: Election result data (not errors or debug output)

**Validation States**:

- `VALID` - CSV passes all checks, ready for database comparison
- `EMPTY` - File exists but contains no data rows
- `INCOMPLETE` - Missing required fields or has empty critical cells
- `MALFORMED` - CSV structure broken or unreadable
- `ERROR` - Execution error or other failure

**Tests:**

```bash
# Run CSV validation tests
pytest webapp/tests/test_ballot_lens_pathways.py::TestCSVValidation -v

# Test results:
# - test_empty_csv_detected      (PASS)
# - test_valid_csv_detected      (PASS)
# - test_malformed_csv_detected  (PASS)
```

### 2. Pathway Consistency

All pathways processing the same election URL should produce:

- **Identical Headers** - Same column names/order
- **Identical Data Rows** - Same candidate/party/vote records
- **Same Data Hash** - Content hash matches across runs
- **Consistent Format** - CSV structure doesn't vary

**Key Test**:

```python
def test_pathway_csv_headers_consistent(self, temp_output_dir, sample_html_fixture):
    # Verify all pathways use same CSV schema
```

### 3. Edge Case Handling

Tests verify graceful handling of:

- **Invalid URLs** - Non-existent domains or malformed URLs
- **No Election Data** - HTML without any results (empty table, missing tables)
- **Timeout/Network** - Connection failures, slow responses
- **Malformed HTML** - Broken tables, missing fields

**Tests:**

```bash
# Run edge case tests
pytest webapp/tests/test_ballot_lens_pathways.py::TestEdgeCases -v

# Covers:
# - test_invalid_url_handling
# - test_html_without_election_data
```

## Usage Scenarios

### Scenario 1: Validate New Election URL

**Goal**: Confirm that a new election result URL produces valid data.

```bash
# 1. Run CSV validation on downloaded output
pytest webapp/tests/test_ballot_lens_pathways.py::TestCSVValidation -v

# 2. Check that headers match known schema
# Expected headers should include: Office, Candidate, Party, Votes

# 3. Validate data sample
# First row should have all required fields populated
```

### Scenario 2: Compare Two Pathways

**Goal**: Confirm that CLI and webapp produce identical results for same URL.

```bash
# 1. Run ballot lens via CLI
python webapp/parser/html_election_parser.py --url "https://example.com/election" --output ./cli_output

# 2. Run via webapp form submission (manual or via Selenium test)
# Navigate to http://localhost:5000/ballot_lens
# Submit URL form
# Download CSV from results

# 3. Compare CSVs
# Place both in temp directories
# Run consistency test:
pytest webapp/tests/test_ballot_lens_pathways.py::TestPathwayConsistency::test_all_pathways_produce_valid_csv -v

# 4. Verify data hashes match
pytest webapp/tests/test_ballot_lens_pathways.py::TestDataComparison::test_csv_content_hash_consistency -v
```

### Scenario 3: Database Validation

**Goal**: Ensure parsed election data matches what's already in database.

```bash
# 1. Parse election URL -> generate CSV
# Run any of the three pathways (CLI, webapp, API)

# 2. Load CSV into comparison tool
from webapp.tests.test_ballot_lens_pathways import validate_csv, hash_csv_content
from pathlib import Path

csv_path = Path("output/election_results.csv")
validation = validate_csv(csv_path)

if validation.is_valid:
    print(f"Rows: {validation.rows_count}")
    print(f"Headers: {validation.headers}")
    print(f"Sample:", validation.sample_row)

    # 3. Query database for same election
    # SELECT * FROM election_results WHERE state='CA' AND county='Alameda' AND election_date='2024-11-05'

    # 4. Compare:
    # - Row count should match (±tombstoned records if applicable)
    # - Candidate names should match exactly
    # - Party codes should match database codec
    # - Vote totals should match (or be reconciled for data corrections)
else:
    print(f"Validation failed: {validation.errors}")
```

## Test Data Files

### Sample HTML Fixture

A minimal valid election results HTML fixture is provided:

```html
<!DOCTYPE html>
<html>
<body>
    <h1>Sample County General Election Results - November 5, 2024</h1>
    <table>
        <tr><th>Race</th><th>Office</th><th>Candidate</th><th>Party</th><th>Votes</th></tr>
        <tr><td>President</td><td>President</td><td>Alice Johnson</td><td>Democratic</td><td>45,230</td></tr>
        <tr><td>President</td><td>President</td><td>Bob Smith</td><td>Republican</td><td>38,920</td></tr>
    </table>
</body>
</html>
```

## Pathway Execution Details

### CLI Execution

```bash
python webapp/parser/html_election_parser.py \
  --url "https://example.com/results/state/county/year/general" \
  --output ./output
```

**Expects:**

- CSV file(s) generated in output directory
- Exit code 0 on success
- Headers matching standard schema

### Webapp API Execution

```bash
# Flask test endpoint (or HTTP request)
POST /api/parse
Content-Type: application/json
{
  "url": "https://example.com/results/...",
  "format": "csv"
}

# Response
{
  "status": "success",
  "csv": "Office,Candidate,Party,Votes\n..."
}
```

**Expects:**

- HTTP 200 response
- JSON with csv field
- Valid CSV content

### Direct API Execution

```python
from webapp.parser.html_election_parser import orchestrate_url

result = orchestrate_url(
    target_url="https://example.com/...",
    processed_info={},
    session_id="manual_test",
    output_bypass=True,
    trust_bypass=True
)

# result = {
#   'headers': ['Office', 'Candidate', 'Party', 'Votes'],
#   'data': [['President', 'Alice Johnson', 'Democratic', '45230'], ...],
#   'contest': {...},
#   'metadata': {...}
# }
```

## Data Comparison Workflow

### 1. Extract Data

```python
from pathlib import Path
from webapp.tests.test_ballot_lens_pathways import validate_csv, read_csv_content, hash_csv_content
import csv

csv_path = Path("output/results.csv")
validation = validate_csv(csv_path)

if validation.is_valid:
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Parsed {len(rows)} rows")
    print(f"Data hash: {hash_csv_content(read_csv_content(csv_path))}")
```

### 2. Query Database

```python
# Pseudo-code for database comparison
def compare_with_database(csv_rows, election_key):
    """
    Args:
        csv_rows: List of dict rows from parsed CSV
        election_key: Dict with state, county, date, type
    """
    db_rows = query_election_database(election_key)

    # Compare row counts
    if len(csv_rows) != len(db_rows):
        print(f"WARNING: Row count mismatch: {len(csv_rows)} vs {len(db_rows)}")

    # Compare candidate names (normalized)
    db_candidates = {normalize_name(r['candidate']) for r in db_rows}
    csv_candidates = {normalize_name(r['Candidate']) for r in csv_rows}

    missing_in_csv = db_candidates - csv_candidates
    extra_in_csv = csv_candidates - db_candidates

    if missing_in_csv:
        print(f"Missing in parsed CSV: {missing_in_csv}")
    if extra_in_csv:
        print(f"Extra in parsed CSV: {extra_in_csv}")

    # Compare vote totals
    for csv_row in csv_rows:
        db_row = find_matching_db_row(csv_row, db_rows)
        if db_row:
            csv_votes = int(csv_row['Votes'].replace(',', ''))
            db_votes = int(db_row['votes'])
            if csv_votes != db_votes:
                print(f"Vote mismatch for {csv_row['Candidate']}: {csv_votes} vs {db_votes}")
```

### 3. Log Discrepancies

Create a validation report:

```python
@dataclass
class DiscrepancyReport:
    """Report of differences between CSV and database."""
    url: str
    state: str
    county: str

    # Counts
    csv_row_count: int
    db_row_count: int

    # Candidates
    missing_candidates: List[str]
    extra_candidates: List[str]

    # Votes
    vote_mismatches: Dict[str, Tuple[int, int]]  # {candidate: (csv_votes, db_votes)}

    # Status
    is_consistent: bool
    notes: str
```

## Running the Tests

### Quick Start

```bash
cd /path/to/html_Parser_prototype

# Run all pathway tests
pytest webapp/tests/test_ballot_lens_pathways.py -v

# Run specific test class
pytest webapp/tests/test_ballot_lens_pathways.py::TestCSVValidation -v
pytest webapp/tests/test_ballot_lens_pathways.py::TestEdgeCases -v
pytest webapp/tests/test_ballot_lens_pathways.py::TestDataComparison -v

# Run with detailed output
pytest webapp/tests/test_ballot_lens_pathways.py -vv --tb=short
```

### Filtering Tests

```bash
# Run only validation tests
pytest webapp/tests/test_ballot_lens_pathways.py -k "csv" -v

# Run only consistency tests
pytest webapp/tests/test_ballot_lens_pathways.py -k "consistency" -v

# Run only edge case tests
pytest webapp/tests/test_ballot_lens_pathways.py -k "edge" -v
```

### With Coverage

```bash
pytest webapp/tests/test_ballot_lens_pathways.py \
  --cov=webapp.parser \
  --cov-report=html \
  --cov-report=term-missing
```

## Expected Results

### CSV Validation Tests

```txt
TestCSVValidation::test_empty_csv_detected ............ PASS
TestCSVValidation::test_valid_csv_detected ............ PASS
TestCSVValidation::test_malformed_csv_detected ........ PASS
```

### Edge Case Tests

```txt
TestEdgeCases::test_invalid_url_handling ............. PASS
TestEdgeCases::test_html_without_election_data ....... PASS
```

### Pathway Tests

```txt
TestPathwayConsistency::test_all_pathways_produce_valid_csv .. PASS
TestPathwayConsistency::test_pathway_csv_headers_consistent .. PASS
```

### Data Comparison Tests

```txt
TestDataComparison::test_csv_content_hash_consistency ........ PASS
TestDataComparison::test_required_fields_present ............ PASS
```

## Troubleshooting

### Issue: Tests timeout

**Cause**: Network URL fetch taking too long, or large HTML parsing

**Solution**:

```bash
# Increase pytest timeout
pytest webapp/tests/test_ballot_lens_pathways.py --timeout=120 -v

# Or use fixture with mock HTML instead of live URLs
pytest webapp/tests/test_ballot_lens_pathways.py::TestCSVValidation -v
```

### Issue: "No CSV output generated"

**Cause**: Parser didn't extract election data from HTML

**Solution**:

1. Verify HTML contains election result tables
2. Check HTML structure matches known formats
3. Review parser logs for format detection issues
4. Try manually running with `--format` override

### Issue: "Row count mismatch"

**Cause**: CSV has different number of records than database

**Possible reasons**:

- Database has tombstoned/deleted records
- Parser included/excluded certain race types
- Database filters by election type but CSV didn't
- Parser incorrectly de-duplicated or split records

**Resolution**:

- Debug the data differences row-by-row
- Identify if it's a parser issue or data quality issue
- Create a targeted fix or update validation rules

## Integration with CI/CD

Add to CI pipeline to validate all election data parses correctly:

```yaml
# Example GitHub Actions
- name: Run ballot lens pathway tests
  run: |
    pytest webapp/tests/test_ballot_lens_pathways.py \
      --junit-xml=test-results.xml \
      --cov=webapp.parser \
      -v

- name: Check test results
  if: failure()
  run: |
    echo "CSV validation failed - review parser output"
```

## Further Reading

- **Parser Architecture**: See `docs/CORE/parser_architecture.md`
- **Risk Assessment**: See `docs/FEATURES/risk_gates_integration.md`
- **Database Schema**: See `docs/DEPLOYMENT/database_schema.md`
- **Data Comparison Utilities**: See `webapp/tests/test_ballot_lens_pathways.py`
