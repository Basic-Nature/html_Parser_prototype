# Multi-Pathway Ballot Lens Testing - Quick Reference

## What Was Built

A complete framework for testing ballot lens across multiple user navigation pathways and comparing results with database election data.

## Key Files

|File|Purpose|Type|
|---|---|---|
|`webapp/tests/test_ballot_lens_pathways.py`|Integration test suite (9 tests, all passing)|Test|
|`tools/compare_ballot_lens_output.py`|CSV↔Database comparison utility|Tool|
|`docs/FEATURES/BALLOT_LENS_PATHWAYS.md`|Complete testing guide (8,500 words)|Docs|
|`docs/FEATURES/MULTI_PATHWAY_SUMMARY.md`|Executive summary|Docs|
|`demo_election_results.csv`|Sample election data for testing|Demo|
|`demo_comparison.json`|Sample comparison report|Demo|

## Quick Start

### Run Tests

```bash
# All tests
pytest webapp/tests/test_ballot_lens_pathways.py -v

# Just CSV validation
pytest webapp/tests/test_ballot_lens_pathways.py::TestCSVValidation -v

# Expected: 9/9 PASSING
```

### Compare Election Data

```bash
# Run ballot lens (any pathway) → produces CSV
# Then compare:
python tools/compare_ballot_lens_output.py \
  --csv output/results.csv \
  --state CA \
  --county "Alameda" \
  --election-date 2024-11-05 \
  --output-json report.json
```

## Test Coverage

### Pathways Tested

- ✅ CLI (command-line execution)
- ✅ Webapp API (HTTP endpoint)
- ✅ Direct API (orchestrate_url function call)

### Validations

- ✅ CSV structure (headers, format)
- ✅ Data integrity (no null critical fields)
- ✅ Pathway consistency (identical output)
- ✅ Database comparison (row/vote matching)
- ✅ Edge cases (invalid URLs, no data, etc.)

### Test Results

```txt
TestCSVValidation .................. 3/3 PASS
TestPathwayConsistency ............ 2/2 PASS
TestEdgeCases ..................... 2/2 PASS
TestDataComparison ................ 2/2 PASS
────────────────────────────────────────────
TOTAL ............................ 9/9 PASS
```

## Data Validation States

- **VALID**: CSV passes all checks, ready for comparison
- **EMPTY**: File has headers but no data rows
- **INCOMPLETE**: Missing required fields or empty critical cells
- **MALFORMED**: CSV structure broken or unreadable
- **ERROR**: Execution error or other failure

## Comparison Discrepancy Types

- **Missing**: Candidate in DB but not in CSV
- **Extra**: Candidate in CSV but not in DB
- **Row Count**: CSV has different number of records
- **Vote Mismatch**: Vote totals don't match

## Example Usage Scenarios

### 1. Validate New Election URL

```bash
pytest webapp/tests/test_ballot_lens_pathways.py::TestCSVValidation -v
# Ensures downloaded CSV has valid structure
```

### 2. Compare Two Pathways

```python
# Run via CLI and webapp
# Save both CSVs
# Run comparison test:
pytest webapp/tests/test_ballot_lens_pathways.py::TestPathwayConsistency -v
```

### 3. Check Against Database

```bash
python tools/compare_ballot_lens_output.py \
  --csv output/results.csv \
  --state CA \
  --county "Alameda" \
  --election-date 2024-11-05

# Output: CONSISTENT / INCONSISTENT + discrepancy list
```

## Required CSV Format

Minimum required columns:

```txt
Office,Candidate,Party,Votes
President,Alice Johnson,Democratic,45230
President,Bob Smith,Republican,38920
```

All rows must have these fields populated (no nulls).

## Documentation Map

```branch
docs/FEATURES/
├── BALLOT_LENS_PATHWAYS.md (8,500 words)
│   └─ Complete guide with usage scenarios
├── MULTI_PATHWAY_SUMMARY.md (this file summary)
│   └─ Technical overview and architecture
└── QUICK_REFERENCE.md (this file)
    └─ Quick lookup and examples
```

## Common Commands

```bash
# Run all tests
pytest webapp/tests/test_ballot_lens_pathways.py -v

# Run with coverage
pytest webapp/tests/test_ballot_lens_pathways.py --cov=webapp.parser -v

# Run specific test class
pytest webapp/tests/test_ballot_lens_pathways.py::TestDataComparison -v

# Compare CSV against database
python tools/compare_ballot_lens_output.py \
  --csv output/results.csv \
  --state CA \
  --county "Alameda" \
  --election-date 2024-11-05 \
  --output-json report.json \
  --output-report report.html

# Show comparison utility help
python tools/compare_ballot_lens_output.py --help
```

## Output Formats

### Comparison Tool JSON

```json
{
  "summary": {
    "is_consistent": true,
    "csv_rows": 4,
    "db_rows": 4,
    "csv_total_votes": 166050,
    "db_total_votes": 166050,
    "discrepancies_count": 0
  },
  "discrepancies": []
}
```

### Comparison Tool Console Output

```txt
Status: CONSISTENT
CSV Rows:      4
Database Rows: 4
Match:         True
CSV Votes:     166,050
DB Votes:      166,050
Match:         True
Discrepancies: 0
```

## Troubleshooting

|Issue|Cause|Solution|
|---|---|---|
|"No CSV output generated"|Parser didn't extract data|Check HTML has election tables; try manual parsing|
|"Row count mismatch"|Database has tombstoned records|Review discrepancies; may be expected|
|Test timeout|Large HTML file|Increase pytest timeout or use mock data|
|Import error|Missing dependencies|Ensure all ballot lens modules present|

## Next Steps

1. Read `BALLOT_LENS_PATHWAYS.md` for detailed guide
2. Run tests: `pytest webapp/tests/test_ballot_lens_pathways.py -v`
3. Generate CSV via ballot lens (any pathway)
4. Compare with database: `python tools/compare_ballot_lens_output.py --csv output.csv ...`
5. Review JSON/HTML reports for discrepancies

## Support

For detailed information:

- **Testing**: See `docs/FEATURES/BALLOT_LENS_PATHWAYS.md`
- **Architecture**: See `docs/FEATURES/MULTI_PATHWAY_SUMMARY.md`
- **Tool Usage**: Run `python tools/compare_ballot_lens_output.py --help`
- **Code**: `webapp/tests/test_ballot_lens_pathways.py` (well-commented)

---

**Status**: ✅ Complete and tested
**Last Updated**: February 20, 2026
**All Tests**: 9/9 PASSING
