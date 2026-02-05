---
layout: default
title: Verification & QA Framework
---

# Verification & QA Framework

Comprehensive quality assurance framework for testing, validation, and verification of parsed election data.

> **Note**: This document consolidates content from:
> - [VERIFICATION_FRAMEWORK.md](../VERIFICATION_FRAMEWORK.md) - QA framework
> - [VERIFICATION_TESTING_GUIDE.md](../VERIFICATION_TESTING_GUIDE.md) - Testing procedures
> - [VERIFICATION_SYNC_IMPLEMENTATION.md](../VERIFICATION_SYNC_IMPLEMENTATION.md) - Sync verification
>
> For complete details, consult the individual source documents linked above.

## 🎯 Overview

The verification framework ensures election data accuracy through:
- Automated validation tests
- Data integrity checks
- Cross-validation with source documents
- Confidence scoring and flagging
- Manual QA workflows
- Historical comparison

## ✅ Validation Levels

### Level 1: Schema Validation
- Data structure conforms to expected format
- Required fields present
- Field types correct

### Level 2: Format Validation
- Names within acceptable ranges
- Vote counts non-negative integers
- Percentages in valid range (0-100 or 0-1)
- Dates in valid format

### Level 3: Semantic Validation
- Party names match canonical list
- Candidate names reasonable (not obviously corrupted)
- Vote totals consistent across records
- Percentages sum appropriately

### Level 4: Cross-Validation
- Data matches source document (manual spot-check)
- Multi-source consistency checks
- Historical pattern matching
- Anomaly detection

## 🧪 Testing Framework

### Unit Tests
Location: `webapp/tests/test_*.py`

```bash
# Run unit tests
python -m pytest webapp/tests/ -v

# Run specific test
python -m pytest webapp/tests/test_table_builder.py::test_normalize_headers

# Coverage report
python -m pytest --cov=webapp webapp/tests/
```

### Integration Tests
```bash
# Test full parsing pipeline
python -m pytest webapp/tests/integration/ -v

# Test specific format
python -m pytest webapp/tests/integration/test_html_parsing.py
```

### Smoke Tests
Quick validation of core functionality:
```bash
# Run smoke test
python run_statement_test.py

# Expected result: "PASS" for all critical functions
```

## 🎯 Manual QA Workflow

1. **Parse Document**
   - Load sample election document
   - Parse with parser
   - Review extracted data

2. **Visual Inspection**
   - Open source document
   - Compare extracted values to source
   - Flag any discrepancies

3. **Spot Checks**
   - Verify 5–10 key races
   - Validate vote totals sum correctly
   - Check percentages add to ~100%

4. **Edge Case Testing**
   - Test write-in candidates
   - Verify tied/close races
   - Check multi-party races

5. **Document Findings**
   - Notes on errors found
   - Confidence assessment
   - Recommended corrections

## 📊 Confidence Scoring

Each extracted value receives a confidence score (0.0–1.0):

```python
score = (
    0.3 * header_accuracy +       # How well column identified
    0.3 * value_validation +      # How well value conforms to rules
    0.2 * context_consistency +   # How consistent with surrounding data
    0.2 * source_reliability      # How trustworthy the source
)
```

**Interpretation**:
- **0.9–1.0**: Highly confident, minimal review needed
- **0.7–0.89**: Confident, spot-check recommended
- **0.5–0.69**: Moderate, review recommended
- **0.3–0.49**: Low, significant review needed
- **0.0–0.29**: Very low, expert review required

## 🚨 Data Quality Issues

### Common Issues & Handling

| Issue | Cause | Detection | Fix |
|-------|-------|-----------|-----|
| Missing candidates | Extraction failed on section | Lower confidence | Re-extract or manual entry |
| Duplicate candidates | Name normalization failed | Duplicate key detection | Merge duplicates |
| Vote sum mismatch | Incomplete extraction or rounding | Sum validation | Flag for review |
| Parse failures | Unsupported format or corruption | Exception logging | Try alt method |
| Candidate name corruption | OCR errors (PDF) | Name validation | Manual correction |

### Anomaly Detection

Automated alerts for unusual patterns:

```python
# Unusually high/low percentages
if vote_pct > 95 or vote_pct < 5:
    flag("Unusual vote share", candidate, pct)

# Vote totals don't match
if sum(votes) != reported_total:
    flag("Vote total mismatch", actual, expected)

# Duplicate candidate names
if len(candidates) != len(set(c.name for c in candidates)):
    flag("Duplicate candidates detected")

# Missing races
if races_found < races_expected:
    flag("Missing races", found, expected)
```

## 🔍 Verification Workflows

### Complete Document Verification

```
Parse Document
    ↓
[Auto-Validation]
├─ Schema check: PASS/FAIL
├─ Format check: PASS/FAIL
├─ Semantic check: PASS/FAIL
└─ Confidence scores assigned
    ↓
[Anomaly Check]
├─ Flag unusual values
├─ Look for patterns
└─ Compare to historical data
    ↓
[Manual QA]
├─ Spot-check 10% of data
├─ Verify source document match
└─ Assessment: PASS/FAIL/NEEDS_REVIEW
    ↓
[Report Generation]
├─ Summary of findings
├─ Recommended actions
└─ Sign-off for use
```

## 📈 Metrics & Reporting

### Key Metrics

```
- Total records extracted
- Validation pass rate
- Average confidence score
- False positive rate
- Time to manual verification
```

### Daily Report

```bash
# Generate verification report
python health/generate_verification_report.py \
  --date 2024-01-15 \
  --format html \
  --output verification_report.html
```

## ✨ Best Practices

### For Developers
- [ ] Write tests for new parsing logic
- [ ] Include validation checks in handlers
- [ ] Document known limitations
- [ ] Flag uncertain extractions

### For QA
- [ ] Document test cases
- [ ] Record issues in issue tracker
- [ ] Create regression tests for bugs found
- [ ] Regular knowledge sharing with team

---

**Related Documents**:
- [Data Models & Schema](../CORE/DATA_MODELS.md) - Data structure details
- [Quarantine System](./QUARANTINE_SYSTEM.md) - Isolation & review process
- [ML Framework](./ML_FRAMEWORK.md) - ML-based validation

**Sources**:
- [VERIFICATION_FRAMEWORK.md](../VERIFICATION_FRAMEWORK.md)
- [VERIFICATION_TESTING_GUIDE.md](../VERIFICATION_TESTING_GUIDE.md)

**Last Updated**: Consolidated QA framework
