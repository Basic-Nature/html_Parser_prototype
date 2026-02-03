# Verification Framework: DL1 ↔ DL2 Dual-Truth System

## Overview

The Smart Elections Parser implements a **dual-truth verification system** that maintains strict separation between:

- **DL1**: Human-verified ground truth (authoritative source of truth)
- **DL2**: AI-extracted working dataset (subject to hallucination)

This architecture ensures that the database preserves "the voice of the people" by requiring human verification before data promotion from DL2 → DL1.

## Architecture

```txt
┌─────────────────────────────────────────────────────────────┐
│ System Mission (Immutable)                                   │
│ "Protect the voice of the people by preserving the accurate │
│  count of legitimate votes. Detect unintentional data errors │
│  at acceptable thresholds."                                  │
│ Author: Juancarlos Barragan | DOB: 1996-03-18              │
└─────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ DL2 (AI-Extracted Dataset)                                   │
│ ├─ Source: Automatic parsing from election websites         │
│ ├─ Authority: Subject to AI hallucination/errors            │
│ ├─ Mutability: Corrected through human verification         │
│ └─ Location: Google Drive (DL2_DRIVE_FOLDER_URL)            │
└──────────────────────────────────────────────────────────────┘
                           ↓
                    Human Review Flow
                           ↓
┌──────────────────────────────────────────────────────────────┐
│ Verification Decision (Immutable Audit Trail)                │
│ ├─ Status: approved | rejected | flagged                    │
│ ├─ Confidence: high | medium | low | unsure                 │
│ ├─ Anomalies: classified by type (unintentional mistakes)    │
│ └─ Logged to: VERIFICATION_LOG_FILE (append-only)           │
└──────────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────┐
│ DL1 (Verified Ground Truth)                                  │
│ ├─ Source: Human-verified, row-by-row validated data       │
│ ├─ Authority: Canonical "voice of the people"               │
│ ├─ Immutability: Append-only; corrections tracked            │
│ └─ Location: Google Drive (DL1_DRIVE_FOLDER_URL)            │
└──────────────────────────────────────────────────────────────┘
```

## Verification Workflow

### 1. DL2 Extraction (Automated)

```bash
Parser extracts election data from website
→ Result stored in DL2 (Google Drive folder)
→ Each row gets unique dl2_id
```

### 2. Human Review (Manual)

Authorized reviewer uses verification UI to:

- View DL2 row side-by-side with potential DL1 match
- Classify any anomalies (data formatting, missing fields, etc.)
- Submit verification decision

### 3. Verification Decision

```json
{
  "dl2_id": "row_abc123",
  "status": "approved|rejected|flagged",
  "confidence": "high|medium|low|unsure",
  "anomalies": [
    {
      "type": "data_formatting|numeric_precision|missing_field|encoding_issue|extraction_error",
      "field": "candidate_name",
      "description": "..."
    }
  ],
  "correction_data": {"field": "corrected_value"},
  "notes": "Explanation of verification decision"
}
```

### 4. DL1 Promotion (Automatic or Manual)

**Automatic (if confidence >= MIN_VERIFICATION_CONFIDENCE):**

- Approved rows promoted to DL1 automatically
- ADMIN_FULL_TRUST gets notification

**Manual (if confidence < threshold):**

- ADMIN_FULL_TRUST must explicitly approve
- ROOT_ADMIN can override

### 5. Immutable Audit Trail

All decisions logged to `VERIFICATION_LOG_FILE`:

```jsonl
{"dl2_id": "row_abc123", "status": "approved", "confidence": "high", "timestamp": "...", "entry_hash": "..."}
```

## Anomaly Classification

The system detects **unintentional mistakes only** (not fraud):

| Anomaly Type | Example | Unintentional? | Action |
| --- | --- | --- | --- |
| **Data Formatting** | "John Smith" vs "john smith" | ✅ Yes | Correct case |
| **Numeric Precision** | "12345" vs "12345.00" | ✅ Yes | Normalize decimals |
| **Missing Field** | Empty cell | ✅ Yes | Fill or flag |
| **Encoding Issue** | UTF-8 corruption | ✅ Yes | Re-encode |
| **Extraction Error** | Parser missed row | ✅ Yes | Re-extract |
| **Vote Suppression** | Candidate votes deleted | ❌ No | **Escalate to authorities** |
| **Vote Inflation** | Artificial vote increase | ❌ No | **Escalate to authorities** |

**Critical:** Anomalies suggesting criminal intent (vote suppression, ballot stuffing, etc.) are **immediately escalated to election officials and law enforcement**. The parser does not make fraud determinations.

## API Endpoints

### 1. System Mission & Governance

```bash
GET /api/verification/system/mission
```

Returns system authorship, mission, and framework URLs.

```json
{
  "author": "Juancarlos Barragan",
  "mission": "Protect the voice of the people...",
  "dl1_folder": "https://drive.google.com/drive/u/4/folders/...",
  "dl2_folder": "https://drive.google.com/drive/u/4/folders/...",
  "governance_url": "/SYSTEM_GOVERNANCE.md"
}
```

### 2. Verification Statistics

```bash
GET /api/verification/log/stats
```

**Requires:** Authenticated principal (REVIEWER tier or above)

Returns verification audit trail statistics:

```json
{
  "total": 1250,
  "by_status": {
    "approved": 1000,
    "rejected": 200,
    "flagged": 50,
    "pending": 0
  },
  "by_confidence": {
    "high": 950,
    "medium": 250,
    "low": 30,
    "unsure": 20
  },
  "by_anomaly_type": {
    "data_formatting": 120,
    "encoding_issue": 45,
    "missing_field": 15
  }
}
```

### 3. Retrieve Verification Entries

```bash
GET /api/verification/log/entries?limit=100&status=flagged&dl2_id=row_abc123
```

**Query Parameters:**

- `limit`: Max entries (1-1000, default 100)
- `status`: Filter by status (approved|rejected|flagged|pending)
- `dl2_id`: Filter by DL2 row ID

**Response:**

```json
{
  "entries": [
    {
      "dl2_id": "row_abc123",
      "dl2_data": {"candidate": "John Smith", "votes": "12345"},
      "dl1_id": "verified_row_abc123",
      "status": "approved",
      "confidence": "high",
      "anomalies": [{"type": "data_formatting", "field": "candidate"}],
      "correction_data": {},
      "timestamp": "2026-02-02T18:30:00Z",
      "entry_hash": "a3c5f8b2d9e1c7a4b6f8d2e5c9a1b3f5",
      "verifier_principal": "alice@electionspulse.org"
    }
  ],
  "count": 10,
  "total_available": 1250
}
```

### 4. Submit Verification Decision

```bash
POST /api/verification/submission
```

**Requires:** Authenticated principal (ADMIN_REVIEWER tier or above)

**Request Body:**

```json
{
  "dl2_id": "row_abc123",
  "dl2_data": {
    "candidate": "John Smith",
    "votes": "12345",
    "state": "Arizona",
    "county": "Pima"
  },
  "dl1_id": "verified_row_abc123",
  "status": "approved",
  "confidence": "high",
  "notes": "Matches official county website. Data formatting corrected.",
  "anomalies": [
    {
      "type": "data_formatting",
      "field": "candidate",
      "description": "Case corrected from 'john smith' to 'John Smith'"
    }
  ],
  "correction_data": {
    "candidate": "John Smith"
  }
}
```

**Response:**

```json
{
  "success": true,
  "entry": { /* verification entry */ },
  "audit_trail_confirmed": true
}
```

### 5. Compare DL1 ↔ DL2

```bash
POST /api/verification/comparison
```

Compares DL2 row against DL1 row and classifies anomalies automatically.

**Request Body:**

```json
{
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
}
```

**Response:**

```json
{
  "dl2_row": {...},
  "dl1_row": {...},
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

### 6. Export Verified Data (DL1)

```bash
GET /api/verification/export/dl1?state=Arizona&limit=1000
```

**Requires:** Authenticated principal (ADMIN_FULL_TRUST tier or above)

**Response:** CSV file with approved rows

```csv
candidate,votes,state,county,verified_at,verified_by,dl2_id,verification_confidence
John Smith,12345,Arizona,Pima,2026-02-02T18:30:00Z,alice@electionspulse.org,row_abc123,high
```

## Configuration

### Environment Variables

```bash
# Verification Framework Toggle
ENABLE_VERIFICATION_FRAMEWORK=true

# DL1/DL2 Folder URLs (Google Drive)
DL1_DRIVE_FOLDER_URL=https://drive.google.com/drive/u/4/folders/1ZwsL_Ui2qFyV-EJ1OZ_9lyhMeX8d4v9N
DL2_DRIVE_FOLDER_URL=https://drive.google.com/drive/u/4/folders/1wQcC_UEIFQrIYyRhyfgY2rr5RkiCBQ7V

# Verification Confidence Threshold (0.0-1.0)
# Approved rows with confidence >= this value auto-promote to DL1
MIN_VERIFICATION_CONFIDENCE=0.85

# DL2 Retention Period
# How many days to keep extracted (DL2) rows before archival
DL2_RETENTION_DAYS=90

# Allow unverified exports (not recommended)
ALLOW_UNVERIFIED_EXPORTS=false
```

### File Paths

```python
# Verification audit trail
VERIFICATION_LOG_FILE: $CONTEXT_LIBRARY_DIR/verification/verification_log.jsonl

# System governance document
SYSTEM_GOVERNANCE_FILE: /SYSTEM_GOVERNANCE.md
```

## Security & Audit Trail

### Privilege Tiers

| Tier | Can Verify | Can Promote | Can Override | Can Export |
| --- | --- | --- | --- | --- |
| **ROOT_ADMIN** | Yes | Yes | Yes (with audit) | Yes |
| **ADMIN_FULL_TRUST** | Yes | Yes (auto) | Yes | Yes |
| **ADMIN_REVIEWER** | Yes | No | No | No |
| **REVIEWER** | Advisory | No | No | No |
| **USER** | No | No | No | No |

### Immutability Guarantees

✅ **Append-Only Log:** Verification entries cannot be modified (only appended)  
✅ **Entry Hashing:** SHA256 hash of each entry for integrity verification  
✅ **Timestamp Ordering:** Entries must be chronologically ordered  
✅ **Principal Attribution:** Every decision attributed to verifier principal  
✅ **Cryptographic Signing:** ROOT_ADMIN actions signed

### Audit Trail Inspection

```python
from webapp.parser.utils.verification_framework import VerificationLog

vlog = VerificationLog(VERIFICATION_LOG_FILE)

# Get all entries
entries = vlog.read_all()

# Get stats
stats = vlog.get_stats()

# Lookup specific row
entry = vlog.get_by_dl2_id("row_abc123")
print(entry.to_dict())
```

## Integration with Existing Systems

### Parser Integration

When extraction completes, store dl2_id and metadata:

```python
from webapp.parser.utils.verification_framework import VerificationLog

# After extraction...
dl2_id = f"row_{timestamp}_{content_hash}"
dl2_data = {"candidate": "John Smith", "votes": "12345", ...}

# Emit event for verification queue
socketio.emit('dl2_extracted', {
    "dl2_id": dl2_id,
    "dl2_data": dl2_data,
    "source_url": url,
    "extraction_confidence": 0.95
})
```

### ML Integration

Confidence scoring can inform verification priority:

```python
from webapp.parser.Context_Integration.Integrity_check import analyze_contests

# Get AI confidence
results = analyze_contests(contests)
anomaly_rate = results.get("anomaly_rate", 0)

# High anomaly rate → Flag for secondary review
if anomaly_rate > 0.15:
    status = VerificationStatus.FLAGGED
else:
    status = VerificationStatus.PENDING
```

## Next Steps

### Phase 2: Verification Dashboard UI

- [ ] Build verification review interface
- [ ] Side-by-side DL1 ↔ DL2 comparison UI
- [ ] Anomaly classification UI
- [ ] Approval/rejection workflow

### Phase 3: DL Drive Sync

- [ ] Automated sync from DL2 folder (Google Drive)
- [ ] Automated promotion to DL1 folder
- [ ] Folder-based access control

### Phase 4: ML Feedback Loop

- [ ] Use verification decisions to retrain models
- [ ] Track model accuracy by verifier feedback
- [ ] Adaptive confidence thresholds

## References

- [SYSTEM_GOVERNANCE.md](../SYSTEM_GOVERNANCE.md) - System mission & ethics
- [verification_framework.py](./utils/verification_framework.py) - Implementation
- [verification_endpoints.py](./verification_endpoints.py) - API endpoints

---

**Original Conception:** Juancarlos Barragan  
**DOB:** March 18, 1996  
**Date:** February 2, 2026
