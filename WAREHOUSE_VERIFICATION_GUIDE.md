# Smart Elections Parser: Gated Warehouse Verification System

## Overview

This document describes the comprehensive gated verification system implemented to ensure only verified, high-quality election data reaches the warehouse database.

**Core Policy:** "We should not send any unconfirmed data when parsing to the database. This is highly restricted to verified data."

## Architecture

### Three-Layer Verification System

```txt
┌─────────────────────────────────────────────────────┐
│ Layer 1: URL Trust Scoring (at parse time)          │
│ • Score URLs on 0-100 scale                         │
│ • Tiers: trusted (85+), pending (60-84), blocked(<60)│
│ • Block suspicious domains, quarantine uncertain    │
└─────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│ Layer 2: Header Confidence Validation (pre-insert)  │
│ • Score CSV headers on 0.0-1.0 scale                │
│ • Critical columns: candidate, party, votes         │
│ • Flag low-confidence headers for review            │
└─────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│ Layer 3: Exact Duplicate Detection (at insert)      │
│ • Query existing verified/pending records           │
│ • Block insertion if exact match found              │
│ • Allow re-processing of rejected/unverified data   │
└─────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│ Layer 4: Warehouse Storage with Verification Tracking │
│ • Store verification_status: unverified/pending/    │
│    verified/rejected                                │
│ • Track source_url and source_principal             │
│ • Enable audit trail and manual review              │
└─────────────────────────────────────────────────────┘
```

## Implementation Status

### Phase 1: Database Model & Gating Logic ✅ COMPLETE

**Database Fields Added:**

```python
verification_status: String(16)  # unverified/pending/verified/rejected
source_url: String(2048)         # URL that generated this data
source_principal: String(256)    # User/principal who initiated parsing
verification_notes: Text         # Manual review notes
verified_at: DateTime            # When verification occurred
verified_by: String(256)         # Who verified the data
```

**Duplicate Detection:**

- Exact match across: state, county, contest, candidate, party, votes, precinct, election_date
- Only checks against verified/pending records
- Allows re-processing of rejected/unverified data

**URL Verification Tiers:**

- **Trusted (85+):** Direct navigation allowed, auto-verified upon insertion
- **Pending (60-84):** DOM snapshot mode, manual review required before warehouse
- **Blocked (<60):** Rejected entirely, logged for security audit

### Phase 2: Header Confidence Scoring ✅ COMPLETE

**Module:** `webapp/parser/utils/header_confidence.py`

**Confidence Scoring:**

- 1.0 = Exact match (case-insensitive)
- 0.85+ = Strong fuzzy match (contains exact alias)
- 0.70+ = Plausible match (category-level)
- <0.70 = No reliable match → FLAG FOR REVIEW

**Critical Columns:** candidate, party, votes, precinct

**Audit Tool:** `scripts/audit_headers_before_promotion.py`

```bash
python scripts/audit_headers_before_promotion.py [--threshold 0.85] [--limit 500]
```

**Output:**

- `log/header_audit_report.json` - Pass rate, pass/fail count
- `log/flagged_headers.jsonl` - Individual flagged headers with confidence scores

### Phase 3: Gated URL Execution ⏳ PENDING

**Implementation Location:** `webapp/parser/html_election_parser.py`

**Logic Flow:**

```python
def orchestrate_url(target_url, ...):
    # Step 1: Compute trust score
    trust_score, trust_factors = compute_trust_score(
        target_url,
        context={'state': state, 'county': county},
        principal=principal,
        principal_source=principal_source
    )
    
    # Step 2: Check if should reject
    if should_reject(trust_score, target_url, privilege_tier):
        logger.error(f"URL rejected: trust_score={trust_score}")
        mark_url_processed(target_url, status="rejected")
        return
    
    # Step 3: Check if should quarantine
    if should_quarantine(trust_score, target_url, privilege_tier):
        logger.warning(f"URL quarantined: trust_score={trust_score}")
        socketio.emit('url_quarantine_required', {
            'url': target_url,
            'trust_score': trust_score,
            'trust_factors': trust_factors
        })
        mark_url_processed(target_url, status="quarantined")
        return
    
    # Step 4: Proceed with parsing (trusted URL)
    result = parse_url_normally(target_url)
```

**Session Isolation:**

- Each URL parsed in isolated session with cancellation flag
- User cannot re-execute quarantined URL without explicit approval
- Audit trail logs all approval/rejection events

### Phase 4: URL Approval System ⏳ PENDING

**New Endpoints:**

```txt
GET /api/urls/status
  Returns: {
    "urls": [
      {
        "url": "https://elections.maryland.gov/...",
        "trust_score": 95,
        "verification_status": "trusted",  // verified|pending|rejected
        "verified_rows_in_warehouse": 450,
        "last_attempted": "2026-02-02T10:00:00Z",
        "last_status": "success",
        "trust_factors": { ... }
      }
    ]
  }

POST /api/urls/approve
  Payload: {
    "url": "https://...",
    "session_id": "sess_...",
    "approval_reason": "Admin verified domain"
  }
  Returns: { "success": true, "message": "URL approved, queued for re-processing" }
```

**SocketIO Event:** `@socketio.on('approve_url')`

- Handles manual URL approval
- Logs approval event with principal and timestamp
- Re-queues URL for parsing with approval flag

### Phase 5: Execute Migration ⏳ PENDING

**Pre-Migration Validation:**

```bash
# 1. Audit headers
python scripts/audit_headers_before_promotion.py

# 2. Check pass rate >= 85%
cat log/header_audit_report.json | grep "pass_rate"

# 3. Review flagged headers
cat log/flagged_headers.jsonl | jq .

# 4. Execute promotion
python -m webapp.parser.health.dataset_promotion
```

**Expected Results:**

- Verified/pending data inserted with verification_status set correctly
- Blocked URLs skipped entirely
- Source URL and principal tracked on all records
- Duplicate records skipped with count in batch metadata

### Phase 6: UI Data Status Display ⏳ PENDING

**New Features:**

- URL status table showing: URL, Trust Score, Status, Verified Rows
- Approval button for pending URLs
- Real-time updates via SocketIO
- Audit trail of all approvals

---

## Usage Guide

### For Administrators

**Pre-Migration Audit:**

```bash
# Run header confidence audit
python scripts/audit_headers_before_promotion.py --threshold 0.85

# Review report
cat log/header_audit_report.json

# Check flagged headers
cat log/flagged_headers.jsonl | head -5
```

**Execute Warehouse Migration:**

```bash
# Via API
curl -X POST http://localhost:5000/api/health_tasks \
  -H "Authorization: Bearer $HEALTH_TASK_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"task": "dataset_promotion_latest"}'

# Check task status
curl http://localhost:5000/api/health_tasks \
  -H "Authorization: Bearer $HEALTH_TASK_TOKEN"
```

**Review Ingested Data:**

```sql
-- Count by verification status
SELECT verification_status, COUNT(*) 
FROM warehouse_election_results 
GROUP BY verification_status;

-- Find duplicate warnings
SELECT state, county, contest, candidate, party, COUNT(*) 
FROM warehouse_election_results 
WHERE verification_status IN ('verified', 'pending')
GROUP BY state, county, contest, candidate, party 
HAVING COUNT(*) > 1;

-- Check source tracking
SELECT DISTINCT source_url 
FROM warehouse_election_results 
ORDER BY source_url;
```

### For Users (UI)

1. **Upload or link election data**
   - Provide URL or CSV file
   - System scores URL trustworthiness
   - Low-trust URLs quarantined automatically

2. **Review verification status**
   - View URL trust scores in ballot_lens interface
   - See which URLs have verified rows in warehouse
   - Approve pending URLs if confident

3. **Track data provenance**
   - Click on data rows to see source URL and user
   - Verify audit trail of approvals/rejections
   - Export data with full provenance metadata

---

## Key Functions Reference

### URL Trust Scoring

```python
from webapp.parser.utils.url_trust_scorer import compute_trust_score

score, factors = compute_trust_score(
    url="https://elections.maryland.gov/...",
    context={"state": "MD", "county": "Baltimore"},
    principal="user@example.com",
    principal_source="sso_oid"
)
# score: 0-100
# factors: {
#   "verified_domain": True,
#   "gov_domain": True,
#   "allowlist_match": True,
#   "historical_success": 0.95,
#   ...
# }
```

### Header Confidence Scoring

```python
from webapp.parser.utils.header_confidence import (
    get_header_confidence,
    validate_row_headers,
    should_insert_row
)

# Score individual header
score = get_header_confidence("ballot_candidate", "candidate")
# Returns: 1.0 (exact match after normalization)

# Validate row headers
passed, scores, flagged = validate_row_headers(
    headers=["candidate", "party", "votes", "precinct"],
    critical_columns=["candidate", "party", "votes"],
    confidence_threshold=0.85
)
# passed: True
# scores: {"candidate": 1.0, "party": 1.0, "votes": 1.0}
# flagged: []
```

### Duplicate Detection

```python
from webapp.parser.health.promotion_helpers import check_exact_duplicate
from sqlalchemy.orm import Session

session = Session(engine)
is_duplicate = check_exact_duplicate(
    session=session,
    state="MD",
    county="Baltimore",
    contest="President",
    candidate="John Doe",
    party="Democratic",
    votes=12345,
    precinct="Precinct 1",
    election_date=datetime(2024, 11, 5)
)
# Returns: True if exact match found in verified/pending records
```

---

## Audit Trail & Logging

**Log Locations:**

- `log/header_audit_report.json` - Header confidence audit summary
- `log/flagged_headers.jsonl` - Individual flagged headers (NDJSON)
- `log/trust_history.jsonl` - Historical URL trust scores (NDJSON)
- `log/processed_urls.json` - URL processing status and timestamps
- `log/db_monitor.jsonl` - Database operations log

**Audit Events Include:**

- Timestamp (ISO 8601)
- Event type (audit, trust_scorer, duplicate_check, etc.)
- Session ID for tracing
- Trust score and factors
- Duplicate detection results
- Verification status decisions

---

## Security & Data Governance

**Implementation Guarantees:**

1. ✅ No unverified data enters warehouse by default (all new data flagged unverified)
2. ✅ Source URL tracking on every record (enables audit/provenance)
3. ✅ Exact duplicate prevention (no duplicate records except approved re-runs)
4. ✅ URL verification tiers (trusted/pending/blocked)
5. ✅ Gated execution (low-trust URLs quarantined for approval)
6. ✅ Session isolation (each parse in separate session)
7. ✅ Audit trail (all approvals/rejections logged)

**Privilege Tiers:**

- **Admin/Government Official:** Can approve low-trust URLs with boost to trust score
- **Standard User:** Can only execute trusted URLs; quarantined URLs require admin approval

---

## Testing Checklist

- [ ] Header audit produces report with correct pass_rate
- [ ] Duplicate detection catches exact matches
- [ ] URL tier function returns correct tier for test URLs
- [ ] Dataset promotion respects verification gates
- [ ] Warehouse records have verification_status populated
- [ ] Source URL and principal tracked on all records
- [ ] UI displays URL status and verification info
- [ ] Approval workflow executes correctly

---

## Configuration

**Environment Variables:**

```txt
# Header confidence scoring
HEADER_CONFIDENCE_THRESHOLD=0.85  # Default: 0.85

# URL trust scoring
URL_TRUST_THRESHOLD_HIGH=80        # Score >= 80: trusted
URL_TRUST_THRESHOLD_MEDIUM=50      # Score >= 50: snapshot mode
URL_TRUST_THRESHOLD_LOW=30         # Score >= 30: quarantine

# Health task token for warehouse promotion
HEALTH_TASK_TOKEN=<token>

# Privilege tier boost for admins
ENABLE_PRIVILEGE_TIER_BOOST=true
```

---

## Next Steps

1. **Phase 3:** Implement URL gating in html_election_parser.py
2. **Phase 4:** Create approval endpoints in Smart_Elections_Parser_Webapp.py
3. **Phase 5:** Execute dataset_promotion with verification gates enabled
4. **Phase 6:** Build UI components for URL status and approval workflow

---

## Contact & Support

For questions or issues with the verification system, refer to:

- `PHASE_12_COMPLETION_REPORT.md` - Implementation summary
- Code comments in header_confidence.py and promotion_helpers.py
- Inline docstrings in url_trust_scorer.py
