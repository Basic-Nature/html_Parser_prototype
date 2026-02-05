---
layout: default
title: Quarantine & Review System
---

## Quarantine & Review System

Systematic approach to isolate, review, and remediate problematic parsed election data.

> **Note**: This document references:
>
> - [QUARANTINE_SYSTEM_GUIDE.md](../QUARANTINE_SYSTEM_GUIDE.md) - Full quarantine procedures
> - [QUARANTINE_INDEX.md](../QUARANTINE_INDEX.md) - Quarantine inventory
>
> For complete details, consult the source documents linked above.

## 🎯 Overview

The quarantine system prevents low-quality data from being used while enabling expert review and correction:

- **Isolation**: Separate problematic results from clean results
- **Analysis**: Root cause identification
- **Review**: Expert assessment and decision
- **Correction**: Fix data or reject
- **Reprocessing**: Re-extract with improved parameters
- **Approval**: Re-queue for use

## 📊 Quarantine Triggers

Results are quarantined when any of these occur:

| Trigger | Threshold | Action |
| --------- | ----------- | -------- |
| Overall confidence score | < 0.50 | Auto-quarantine |
| Field confidence score | < 0.40 | Flag for review |
| Validation failures | > 5% of fields | Auto-quarantine |
| Vote total mismatch | > 5% variance | Flag for review |
| Missing key races | > 10% of expected | Auto-quarantine |
| Duplicate candidates | Any detected | Flag for review |
| Parse errors | Any unhandled | Auto-quarantine |

## 🔄 Quarantine Workflow

```tree
Parsed Result
    ↓
[CHECK TRIGGERS]
├─ Validation pass? ──┐
├─ Confidence OK?  ──┤──→ APPROVED (use immediately)
└─ No errors? ───────┘
                      OR
                      ↓
                  [QUARANTINE]
                  ├─ Store in isolation
                  ├─ Log reason/category
                  └─ Create review ticket
                      ↓
                  [EXPERT REVIEW]
                  ├─ Assess data quality
                  ├─ Identify root cause
                  └─ Decision:
                     ├→ APPROVE (use as-is)
                     ├→ CORRECT (fix & reuse)
                     ├→ REPROCESS (re-extract)
                     └→ REJECT (discard)
```

## 📁 Quarantine Categories

### Critical Issues (Immediate Review)

- Complete parse failures
- No data extracted
- Unhandled exceptions
- Corrupt data detected

### High-Confidence Issues (Prompt Review)

- Missing major races
- Invalid vote totals
- Duplicate candidates
- Validation failures

### Low-Confidence Issues (Batch Review)

- Low confidence scores
- Minor anomalies
- Partial extraction
- Uncertain headers

## 🔍 Review Process

### Step 1: Categorize Issue

```tree
Q1: What type of problem?
├─ Extraction failure (no data)
├─ Data corruption (invalid values)
├─ Incomplete extraction (missing sections)
├─ Format mismatch (structure error)
└─ Confidence issue (uncertain values)
```

### Step 2: Root Cause Analysis

```tree
Q2: What caused the problem?
├─ Source document quality (OCR, format)
├─ Parser limitation (unsupported layout)
├─ Configuration issue (wrong settings)
├─ Bug in code (logic error)
└─ External issue (network, service)
```

### Step 3: Decision

```tree
Q3: How to proceed?
├─ APPROVE if data usable despite issues
├─ CORRECT if fixable errors present
├─ REPROCESS if better method available
└─ REJECT if irrecoverable
```

## ✏️ Manual Correction Workflow

When QA team chooses "CORRECT":

1. **Open Correction Interface**
   - Load quarantined result
   - Display source document alongside
   - Mark problematic fields for editing

2. **Edit Data**
   - Correct invalid values
   - Fill missing candidates
   - Fix validation errors
   - Update confidence scores

3. **Validate Changes**
   - Re-run validation checks
   - Verify vote totals
   - Check for new issues

4. **Document Changes**
   - Note what was corrected
   - Record reason for changes
   - Assign QA reviewer

5. **Approve & Release**
   - Mark as verified
   - Move from quarantine to approved
   - Ready for use

## 🔄 Reprocessing

When QA team chooses "REPROCESS":

1. **Identify Better Method**
   - Try alternative extraction strategy
   - Different format (PDF → HTML if available)
   - Different parser version
   - Manual input parameters

2. **Reparse Document**

   ```bash
   python -c "
   from webapp.parser import html_election_parser
   result = html_election_parser.parse(
       url='file://original_source.pdf',
       strategy='ml_extraction',  # Try different strategy
       force_reparse=True
   )
   "
   ```

3. **Compare Results**
   - New result vs original
   - Check if improved
   - Assess confidence differences

4. **Select Better Result**
   - Use reprocessed result if better
   - Keep original if no improvement
   - Document decision

## 📊 Quarantine Statistics

### Daily Report

```bash
# Generate quarantine report
python health/quarantine_report.py --date 2024-01-15

# Output:
# Total processed:     150
# Approved:            135 (90%)
# Quarantined:         15 (10%)
#   - Critical:        3
#   - High:            7
#   - Low:             5
```

### Trends & Patterns

```bash
# Identify recurring issues
python health/quarantine_analysis.py --lookback 30

# Output:
# Top 5 Quarantine Reasons:
# 1. Low confidence (35%)
# 2. Vote total mismatch (20%)
# 3. Missing races (18%)
# 4. Parse failures (15%)
# 5. Duplicates (12%)
```

## 📋 Quarantine Queue Management

### Priority Ordering

```txt
Critical issues → High issues → Low issues → Archive old reviews
```

### SLA (Service Level Agreement)

```list
- Critical: Review within 2 hours
- High: Review within 8 hours
- Low: Review within 24 hours
- Archive: Move to archive after 30 days if unresolved
```

### Escalation

```tree
If unresolved after SLA:
├─ 1st escalation: Notify team lead
├─ 2nd escalation: Notify manager
└─ 3rd escalation: Consider rejecting / manual approach
```

## 🎯 Performance Targets

```txt
- 95%+ approval rate for new parser versions
- < 5% quarantine rate
- < 1% final rejection rate
- < 2 hours average review time (critical)
- < 4 hours average reprocessing time
```

## 🛠️ Tools & Utilities

### Command-Line Tools

```bash
# List all quarantined items
python -m health.quarantine list

# Move item from quarantine to approved
python -m health.quarantine approve --id 12345

# Reject and archive
python -m health.quarantine reject --id 12345 --reason "Unrecoverable corruption"

# Force reprocess
python -m health.quarantine reprocess --id 12345 --strategy ml_extraction
```

### Web UI

- Quarantine Queue View: `/qa/quarantine`
- Review Interface: `/qa/quarantine/{id}`
- Statistics Dashboard: `/qa/statistics`

## 📈 Metrics & Reporting

### Key Metrics

```txt
- Quarantine rate (% of total)
- Average resolution time
- Approval rate
- Correction rate
- Rejection rate
- Re-extraction success rate
```

### Monthly Report

```bash
python health/generate_monthly_report.py \
  --month "2024-01" \
  --format pdf \
  --output monthly_report.pdf
```

---

**Related Documents**:

- [Verification Framework](./VERIFICATION.md) - QA and validation
- [Operations Runbook](../DEPLOYMENT/OPERATIONS.md) - Operational procedures
- [ML Framework](./ML_FRAMEWORK.md) - ML-based quality improvements

**Source**:

- [QUARANTINE_SYSTEM_GUIDE.md](../QUARANTINE_SYSTEM_GUIDE.md)

**Last Updated**: Consolidated quarantine system guide
