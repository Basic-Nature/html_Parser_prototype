# Verification Framework: Integration Testing Guide

## Quick Start: Test All 6 Endpoints (5 minutes)

### Prerequisites

```bash
# 1. Ensure Flask app is running
python -m flask --app webapp.Smart_Elections_Parser_Webapp run

# 2. Get your authentication token
# (Extract from client certificate or SSO headers)
PRINCIPAL="alice@electionspulse.org"  # Your principal (email/name)
```

### Test 1: Get System Mission

```bash
curl -X GET http://localhost:5000/api/verification/system/mission
```

**Expected Response (200 OK):**

```json
{
  "author": "Juancarlos Barragan",
  "mission": "Protect the voice of the people...",
  "dl1_folder": "https://drive.google.com/drive/u/4/folders/1ZwsL_Ui2qFyV-EJ1OZ_9lyhMeX8d4v9N",
  "dl2_folder": "https://drive.google.com/drive/u/4/folders/1wQcC_UEIFQrIYyRhyfgY2rr5RkiCBQ7V",
  "governance_url": "/SYSTEM_GOVERNANCE.md"
}
```

### Test 2: Compare DL2 vs DL1

```bash
curl -X POST http://localhost:5000/api/verification/comparison \
  -H "Content-Type: application/json" \
  -d '{
    "dl2_row": {
      "candidate": "john smith",
      "votes": "12345"
    },
    "dl1_row": {
      "candidate": "John Smith",
      "votes": "12345"
    },
    "field_mapping": {
      "candidate": "candidate",
      "votes": "votes"
    }
  }'
```

**Expected Response (200 OK):**

```json
{
  "dl2_row": {"candidate": "john smith", "votes": "12345"},
  "dl1_row": {"candidate": "John Smith", "votes": "12345"},
  "field_anomalies": {
    "candidate": {
      "is_anomaly": true,
      "anomaly_type": "data_formatting",
      "description": "Case mismatch: 'john smith' vs 'John Smith'"
    },
    "votes": {
      "is_anomaly": false,
      "anomaly_type": null,
      "description": "No difference"
    }
  },
  "has_anomalies": true,
  "anomaly_count": 1
}
```

### Test 3: Submit Verification Decision

```bash
curl -X POST http://localhost:5000/api/verification/submission \
  -H "Content-Type: application/json" \
  -H "X-Principal: alice@electionspulse.org" \
  -d '{
    "dl2_id": "row_20260202_abc123",
    "dl2_data": {
      "candidate": "john smith",
      "votes": "12345",
      "state": "Arizona",
      "county": "Pima"
    },
    "dl1_id": "verified_row_abc123",
    "status": "approved",
    "confidence": "high",
    "notes": "Matches county website. Case corrected.",
    "anomalies": [
      {
        "type": "data_formatting",
        "field": "candidate",
        "description": "Case corrected from john smith to John Smith"
      }
    ],
    "correction_data": {
      "candidate": "John Smith"
    }
  }'
```

**Expected Response (201 CREATED):**

```json
{
  "success": true,
  "entry": {
    "dl2_id": "row_20260202_abc123",
    "dl2_data": {...},
    "dl1_id": "verified_row_abc123",
    "status": "approved",
    "confidence": "high",
    "notes": "Matches county website. Case corrected.",
    "anomalies": [...],
    "correction_data": {...},
    "timestamp": "2026-02-02T18:30:00Z",
    "entry_hash": "a3c5f8b2d9e1c7a4b6f8d2e5c9a1b3f5",
    "verifier_principal": "alice@electionspulse.org"
  },
  "audit_trail_confirmed": true
}
```

**Side Effect:** Entry appended to `$CONTEXT_LIBRARY_DIR/verification/verification_log.jsonl`

### Test 4: Get Verification Statistics

```bash
curl -X GET http://localhost:5000/api/verification/log/stats \
  -H "X-Principal: alice@electionspulse.org"
```

**Expected Response (200 OK):**

```json
{
  "total": 1,
  "by_status": {
    "approved": 1,
    "rejected": 0,
    "flagged": 0,
    "pending": 0
  },
  "by_confidence": {
    "high": 1,
    "medium": 0,
    "low": 0,
    "unsure": 0
  },
  "by_anomaly_type": {
    "data_formatting": 1,
    "numeric_precision": 0,
    "missing_field": 0,
    "duplicate_record": 0,
    "encoding_issue": 0,
    "extraction_error": 0,
    "context_mismatch": 0,
    "other": 0
  },
  "retrieved_at": "2026-02-02T18:31:00Z",
  "retrieved_by": "alice@electionspulse.org"
}
```

### Test 5: Retrieve Verification Entries

```bash
# Get all entries
curl -X GET "http://localhost:5000/api/verification/log/entries?limit=100" \
  -H "X-Principal: alice@electionspulse.org"

# Get entries by status
curl -X GET "http://localhost:5000/api/verification/log/entries?status=approved&limit=50" \
  -H "X-Principal: alice@electionspulse.org"

# Get specific DL2 row
curl -X GET "http://localhost:5000/api/verification/log/entries?dl2_id=row_20260202_abc123" \
  -H "X-Principal: alice@electionspulse.org"
```

**Expected Response (200 OK):**

```json
{
  "entries": [
    {
      "dl2_id": "row_20260202_abc123",
      "dl2_data": {...},
      "dl1_id": "verified_row_abc123",
      "status": "approved",
      "confidence": "high",
      "notes": "...",
      "anomalies": [...],
      "correction_data": {...},
      "timestamp": "2026-02-02T18:30:00Z",
      "entry_hash": "...",
      "verifier_principal": "alice@electionspulse.org"
    }
  ],
  "count": 1,
  "limit": 100,
  "total_available": 1
}
```

### Test 6: Export Verified Data (DL1)

```bash
curl -X GET "http://localhost:5000/api/verification/export/dl1?state=Arizona&limit=1000" \
  -H "X-Principal: admin@electionspulse.org" \
  -o verified_data.csv
```

**Expected Output:** CSV file with columns:

```txt
candidate,votes,state,county,verified_at,verified_by,dl2_id,verification_confidence
John Smith,12345,Arizona,Pima,2026-02-02T18:30:00Z,alice@electionspulse.org,row_20260202_abc123,high
```

---

## Advanced Testing

### Test Anomaly Classification

```python
from webapp.parser.utils.verification_framework import classify_anomaly, AnomalyType

# Test 1: Case mismatch
is_anom, anomaly_type, desc = classify_anomaly("john smith", "John Smith", "candidate")
assert is_anom == True
assert anomaly_type == AnomalyType.DATA_FORMATTING
print(f"✓ Case mismatch detected: {desc}")

# Test 2: Missing field
is_anom, anomaly_type, desc = classify_anomaly("", "John Smith", "candidate")
assert is_anom == True
assert anomaly_type == AnomalyType.MISSING_FIELD
print(f"✓ Missing field detected: {desc}")

# Test 3: Numeric precision
is_anom, anomaly_type, desc = classify_anomaly("12345.00", "12345", "votes")
assert is_anom == True
assert anomaly_type == AnomalyType.NUMERIC_PRECISION
print(f"✓ Numeric precision detected: {desc}")

# Test 4: No anomaly
is_anom, anomaly_type, desc = classify_anomaly("John Smith", "John Smith", "candidate")
assert is_anom == False
assert anomaly_type is None
print(f"✓ No anomaly: {desc}")
```

### Test Verification Log

```python
from webapp.parser.utils.verification_framework import (
    VerificationLog, VerificationLineageEntry, VerificationStatus, VerificationConfidence
)
from pathlib import Path

log_path = Path("/tmp/test_verification.jsonl")

# Create log
vlog = VerificationLog(str(log_path))

# Create entry
entry = VerificationLineageEntry(
    dl2_id="test_row_001",
    dl2_data={"candidate": "Alice", "votes": "100"},
    dl1_id="verified_001",
    verifier_principal="test@example.org",
    status=VerificationStatus.APPROVED,
    confidence=VerificationConfidence.HIGH,
    notes="Test verification"
)

# Append
success = vlog.append(entry)
assert success == True
print("✓ Entry appended")

# Read all
entries = vlog.read_all()
assert len(entries) == 1
print(f"✓ Read {len(entries)} entries")

# Get stats
stats = vlog.get_stats()
assert stats["total"] == 1
assert stats["by_status"]["approved"] == 1
print(f"✓ Stats: {stats}")

# Lookup
found = vlog.get_by_dl2_id("test_row_001")
assert found is not None
assert found.dl2_id == "test_row_001"
print("✓ Lookup successful")

# Cleanup
log_path.unlink()
```

### Test Privilege Tiers

```python
# Test REVIEWER tier (can query)
response = requests.get(
    "http://localhost:5000/api/verification/log/entries",
    headers={"X-Principal": "reviewer@example.org", "X-Tier": "reviewer"}
)
assert response.status_code == 200
print("✓ REVIEWER tier can query log")

# Test USER tier (cannot query)
response = requests.get(
    "http://localhost:5000/api/verification/log/entries",
    headers={"X-Principal": "user@example.org", "X-Tier": "user"}
)
assert response.status_code == 403
print("✓ USER tier blocked from log query")

# Test ADMIN_REVIEWER tier (can submit)
response = requests.post(
    "http://localhost:5000/api/verification/submission",
    headers={"X-Principal": "admin@example.org", "X-Tier": "admin_reviewer"},
    json={
        "dl2_id": "test_001",
        "dl2_data": {"candidate": "Bob"},
        "dl1_id": "verified_001",
        "status": "approved",
        "confidence": "high"
    }
)
assert response.status_code == 201
print("✓ ADMIN_REVIEWER tier can submit")

# Test REVIEWER tier (cannot submit)
response = requests.post(
    "http://localhost:5000/api/verification/submission",
    headers={"X-Principal": "reviewer@example.org", "X-Tier": "reviewer"},
    json={...}
)
assert response.status_code == 403
print("✓ REVIEWER tier blocked from submission")
```

---

## Audit Trail Inspection

### View Raw Verification Log

```bash
# Windows PowerShell
Get-Content -Path "$env:CONTEXT_LIBRARY_DIR\verification\verification_log.jsonl" -Tail 10

# Linux/Mac
tail -f $CONTEXT_LIBRARY_DIR/verification/verification_log.jsonl

# Format pretty
cat $CONTEXT_LIBRARY_DIR/verification/verification_log.jsonl | jq .
```

### Verify Entry Hash

```python
from webapp.parser.utils.verification_framework import VerificationLineageEntry
import json

# Load entry from log
with open("$CONTEXT_LIBRARY_DIR/verification/verification_log.jsonl") as f:
    line = f.readline()
    entry_dict = json.loads(line)

# Reconstruct entry
entry = VerificationLineageEntry.from_dict(entry_dict)

# Verify hash
expected_hash = entry._compute_hash()
actual_hash = entry_dict["entry_hash"]

if expected_hash == actual_hash:
    print("✓ Entry hash verified (integrity confirmed)")
else:
    print("✗ HASH MISMATCH (entry may be corrupted)")
    print(f"  Expected: {expected_hash}")
    print(f"  Actual: {actual_hash}")
```

---

## Troubleshooting

### Issue: 403 Forbidden on verification endpoints

**Cause:** Missing or invalid principal

**Solution:**

```bash
# Add X-Principal header
curl -H "X-Principal: alice@example.org" \
  http://localhost:5000/api/verification/log/entries

# Or ensure client certificate is valid
# Check: openssl s_client -connect localhost:5443 -cert client-cert.pem
```

### Issue: 404 Not Found on /api/verification/*

**Cause:** Blueprint not registered

**Solution:**

1. Check blueprint registration in `Smart_Elections_Parser_Webapp.py` (line ~287)
2. Verify import works: `from webapp.parser.verification_endpoints import verification_bp`
3. Check logs for registration error
4. Restart Flask app

### Issue: Entries not appending to log

**Cause:** Directory doesn't exist or permissions issue

**Solution:**

```bash
# Create directory
mkdir -p "$CONTEXT_LIBRARY_DIR/verification"

# Check permissions
ls -la "$CONTEXT_LIBRARY_DIR/verification"

# Ensure writeable by Flask process
chmod 755 "$CONTEXT_LIBRARY_DIR/verification"
```

### Issue: classify_anomaly() returns unexpected type

**Cause:** Field type mismatch or null handling

**Solution:**

```python
# Ensure fields are strings
dl2_value = str(dl2_value or "").strip()
dl1_value = str(dl1_value or "").strip()

# Then classify
anomaly_type = classify_anomaly(dl2_value, dl1_value)
```

---

## Performance Benchmarks

### Expected Response Times

| Endpoint | Data Size | Response Time |
| --- | --- | --- |
| `/system/mission` | ~200 bytes | <10ms |
| `/log/stats` | ~500 bytes | <50ms (scans all entries) |
| `/log/entries` | ~50KB (100 rows) | <100ms |
| `/comparison` | ~1KB | <20ms |
| `/submission` | N/A | ~50ms (includes disk write) |
| `/export/dl1` | ~1MB (10K rows) | ~500ms |

### Optimization Tips

1. **Pagination:** Use `limit` to reduce `/log/entries` response size
2. **Filtering:** Use `status` or `dl2_id` to narrow `/log/entries` scope
3. **Indexing:** Consider adding JSONL index file for large logs (>10K entries)
4. **Caching:** Cache `/system/mission` response (static data)

---

## Success Criteria

All 6 endpoints should:

- ✅ Return correct HTTP status codes
- ✅ Return valid JSON responses
- ✅ Enforce privilege tiers
- ✅ Append entries to verification log
- ✅ Maintain immutable audit trail
- ✅ Complete within expected response time

**Next Step:** If all tests pass, proceed to Phase 2 (Hallucination Detection Integration).

---

**Test Date:** February 2, 2026  
**Tester:** (Your name)  
**Status:** ☐ PASS / ☐ FAIL
