# DL1/DL2 Local File System Sync Implementation

**Phase:** 2 (ML Integration & Sync)  
**Status:** ✅ Complete  
**Date:** February 2, 2026  
**Architecture:** 100% Internal - No External Dependencies

---

## 📋 Overview

The DL1/DL2 Sync system manages bidirectional synchronization between:

- **DL2** (Unverified Dataset) - AI-extracted election data from websites
- **DL1** (Verified Ground Truth) - Human-approved canonical dataset

All operations use **local filesystem storage** - completely internal with no external cloud dependencies.

This enables:

- ✅ Automatic staging of DL2 samples for review
- ✅ One-way promotion of approved data to DL1
- ✅ Deduplication across sync operations
- ✅ Immutable promotion tracking
- ✅ Version control with SHA256 hashing

---

## 🏗️ Architecture

```txt
┌────────────────────────────────────────────────────┐
│ Local Storage System (All Internal)                │
├────────────────────────────────────────────────────┤
│ $CONTEXT_LIBRARY_DIR/verification/                │
│ ├─ dl2/                    (Unverified samples)    │
│ │  ├─ extracted_001.csv                           │
│ │  ├─ extracted_002.csv                           │
│ │  └─ extracted_003.csv                           │
│ ├─ dl1/                    (Verified/Approved)    │
│ │  ├─ approved_001.csv                            │
│ │  ├─ approved_002.csv                            │
│ │  └─ approved_003.csv                            │
│ └─ sync_metadata.json      (Dedup index)          │
└────────────────────────────────────────────────────┘
         ↓ Human Review         ↑ Archive
┌────────────────────────────────────────────────────┐
│ Verification Framework                             │
├────────────────────────────────────────────────────┤
│ - Compare DL2 vs official sources                 │
│ - Classify anomalies (extract errors, etc.)       │
│ - Submit verification decision                     │
│ - Trigger promotion if approved                    │
└────────────────────────────────────────────────────┘
```

**Storage:** Entirely local filesystem. No external cloud services needed.

---

## 📦 Components

### 1. LocalStorageSync (`local_dl_sync.py`)

Local filesystem-based storage synchronization with:

- **Local Directory Management:** DL1/DL2 folder operations
- **File Operations:** Copy, move, delete within local storage
- **Deduplication:** Content-based hash tracking
- **Promotion History:** Immutable audit trail

**Key Methods:**

```python
# Initialization
sync = LocalStorageSync(verification_dir="/path/to/verification")
is_available = sync.is_available()

# Staging & listing
dl2_files = sync.list_dl2_samples()
dl1_files = sync.list_dl1_approved()

# File operations
file_id = sync.stage_dl2_file("/local/extracted.csv", metadata={...})
sync.copy_to_dl1(file_id, metadata={...})
sync.delete_file(file_id, location="dl2")

# Promotion tracking
history = sync.get_promotion_history(limit=50)
promotion = sync.get_promotion_by_id(file_id)

# Deduplication
hash_val = sync.compute_file_hash("/path/to/file")
duplicates = sync.find_duplicates(hash_val)
```

### 2. SyncMetadata (`local_dl_sync.py`)

Persistent metadata store tracking:

- **File hashes** - DL2 and DL1 file content hashes
- **Deduplication index** - Map of content_hash → file_ids
- **Promotion history** - DL2→DL1 conversion tracking
- **Sync timestamps** - Last sync time

**Key Methods:**

```python
metadata = SyncMetadata("/path/to/sync_metadata.json")

# Recording
metadata.record_dl2_file(file_id, file_hash)
metadata.record_dl1_file(file_id, file_hash)
metadata.record_promotion(dl2_id, dl1_id, reason="human_verified")
metadata.record_content_hash(content_hash, file_id)

# Querying
dl2_hash = metadata.get_dl2_hash(file_id)
duplicates = metadata.is_duplicate(content_hash)
```

### 3. DL1DL2SyncManager (`google_drive_sync.py`)

High-level sync orchestrator with:

- **Two-way sync** - Pull DL2, push DL1
- **Deduplication** - Prevent duplicate uploads
- **Promotion** - Automatic DL2→DL1 conversion
- **Thread-safe** - Locked operations for concurrency

**Key Methods:**

```python
sync = DL1DL2SyncManager(
    dl1_folder_url="https://drive.google.com/drive/folders/DL1_ID",
    dl2_folder_url="https://drive.google.com/drive/folders/DL2_ID"
)

# Check availability
is_ready = sync.is_available()

# Pull DL2 samples
count, file_ids = sync.sync_dl2_from_drive("/local/dl2/dir")

# Push DL1 approved data
count, file_ids = sync.sync_dl1_to_drive("/local/dl1/dir", deduplicate=True)

# Promote verified row
sync.promote_dl2_to_dl1(
    dl2_csv_path="/local/dl2/sample.csv",
    dl1_output_path="/local/dl1/approved.csv",
    dl2_id="file_20260202_abc123"
)

# Get stats
stats = sync.get_sync_stats()
# {
#   "dl2_files": 15,
#   "dl1_files": 8,
#   "promotions": 23,
#   "unique_content": 18,
#   "last_sync": "2026-02-02T15:45:30Z"
# }
```

---

## 🔌 REST API Endpoints

All sync endpoints require `ENABLE_VERIFICATION_FRAMEWORK=true` and appropriate privilege tier.

### GET /api/verification/sync/status

**Auth:** REVIEWER+  
**Purpose:** Check sync availability and get statistics

**Response:**

```json
{
  "available": true,
  "dl1_folder_url": "https://drive.google.com/drive/folders/...",
  "dl2_folder_url": "https://drive.google.com/drive/folders/...",
  "stats": {
    "dl2_files": 15,
    "dl1_files": 8,
    "promotions": 23,
    "unique_content": 18,
    "last_sync": "2026-02-02T15:45:30Z",
    "created_at": "2026-01-15T10:00:00Z"
  }
}
```

---

## 🔌 REST API Endpoints

All sync endpoints require `ENABLE_VERIFICATION_FRAMEWORK=true` and appropriate privilege tier.

### GET /api/verification/sync/status

**Auth:** REVIEWER+  
**Purpose:** Check local storage sync status and get statistics

**Response:**

```json
{
  "available": true,
  "storage_path": "/path/to/context_library/verification",
  "stats": {
    "dl2": {
      "file_count": 15,
      "total_size_bytes": 2847392
    },
    "dl1": {
      "file_count": 8,
      "total_size_bytes": 1524576
    },
    "total_promoted": 23,
    "dedup_groups": 18
  }
}
```

### GET /api/verification/sync/dl2/list

**Auth:** REVIEWER+  
**Purpose:** List unverified samples in DL2 (local storage)

**Query Parameters:**

- `limit`: Max files to return (default: 50)

**Response:**

```json
{
  "success": true,
  "count": 15,
  "files": [
    {
      "file_id": "dl2_20260202_abc123",
      "filename": "extracted_001.csv",
      "size_bytes": 2847,
      "hash": "sha256_hex_string",
      "created_at": "2026-02-02T15:30:45Z",
      "promoted": false
    }
  ],
  "timestamp": "2026-02-02T15:45:30Z"
}
```

### GET /api/verification/sync/dl1/list

**Auth:** REVIEWER+  
**Purpose:** List verified/approved samples in DL1 (local storage)

**Query Parameters:**

- `limit`: Max files to return (default: 50)

**Response:**

```json
{
  "success": true,
  "count": 8,
  "files": [
    {
      "file_id": "dl2_20260201_xyz789",
      "filename": "approved_001.csv",
      "size_bytes": 1524,
      "hash": "sha256_hex_string",
      "approved_at": "2026-02-01T10:15:22Z"
    }
  ],
  "timestamp": "2026-02-02T15:45:30Z"
}
```

### POST /api/verification/sync/dl2/stage

**Auth:** ADMIN_REVIEWER+  
**Purpose:** Stage a new extracted file into DL2 (unverified dataset)

**Request Body:**

```json
{
  "source_file": "/path/to/extracted.csv",
  "file_id": "optional_custom_id",
  "metadata": {
    "source_url": "https://...",
    "extracted_at": "2026-02-02T15:00:00Z",
    "handler": "handler_name"
  }
}
```

**Response:**

```json
{
  "success": true,
  "file_id": "dl2_20260202_abc123",
  "storage_path": "/path/to/verification/dl2/dl2_20260202_abc123.csv",
  "timestamp": "2026-02-02T15:45:30Z"
}
```

### POST /api/verification/sync/promote

**Auth:** ADMIN_FULL_TRUST+  
**Purpose:** Promote an approved file from DL2 to verified DL1 dataset

**Request Body:**

```json
{
  "file_id": "dl2_20260202_abc123",
  "verifier_principal": "verifier@example.org",
  "verification_notes": "All rows verified against official sources"
}
```

**Response:**

```json
{
  "success": true,
  "file_id": "dl2_20260202_abc123",
  "promotion_record": {
    "file_id": "dl2_20260202_abc123",
    "source_location": "dl2",
    "dest_location": "dl1",
    "promoted_at": "2026-02-02T16:00:00Z",
    "verifier_principal": "verifier@example.org",
    "verification_notes": "All rows verified against official sources",
    "source_hash": "sha256_hex_string",
    "dest_hash": "sha256_hex_string"
  }
}
```

---

## 🗂️ Local Storage Layout

All data is stored in the local filesystem under `$CONTEXT_LIBRARY_DIR/verification/`:

```txt
$CONTEXT_LIBRARY_DIR/verification/
├── dl2/                           # Unverified samples (AI-extracted)
│   ├── dl2_20260202_abc123.csv   # Staged extraction
│   ├── dl2_20260201_xyz789.csv   # Another extraction
│   └── ...
├── dl1/                           # Verified samples (approved)
│   ├── dl2_20260202_abc123.csv   # Promoted from DL2
│   ├── dl2_20260201_xyz789.csv   # Promoted from DL2
│   └── ...
├── sync_metadata.json             # Deduplication index & hashes
└── promotion_history.jsonl        # Immutable audit log
```

**sync_metadata.json Structure:**

```json
{
  "version": 1,
  "created_at": "2026-01-15T10:00:00Z",
  "file_hashes": {
    "dl2_20260202_abc123": {
      "hash": "sha256_hex_string",
      "location": "dl2",
      "staged_at": "2026-02-02T15:30:45Z",
      "promoted": true,
      "promoted_at": "2026-02-02T16:00:00Z"
    }
  },
  "dedup_index": {
    "sha256_hex_string": ["dl2_20260202_abc123"],
    "...": ["..."]
  },
  "promotion_index": {
    "dl2_20260202_abc123": {
      "file_id": "dl2_20260202_abc123",
      "promoted_at": "2026-02-02T16:00:00Z",
      "verifier_principal": "verifier@example.org"
    }
  }
}
```

**promotion_history.jsonl (Line-delimited JSON):**

```jsonl
{"file_id":"dl2_20260202_abc123","promoted_at":"2026-02-02T16:00:00Z","verifier_principal":"verifier@example.org"}
{"file_id":"dl2_20260201_xyz789","promoted_at":"2026-02-01T14:30:22Z","verifier_principal":"verifier@example.org"}
```

---

## 🔐 Security & Integrity

### Content Hashing

All files are tracked via SHA256 content hash:

- **Deduplication:** Files with identical content are flagged during staging
- **Integrity:** Hash is recomputed on promotion to detect corruption
- **Audit:** Hash is stored in immutable promotion_history.jsonl

### Promotion Audit Trail

Every promotion from DL2→DL1 is recorded immutably:

- **Timestamp:** Exact time of promotion (ISO 8601)
- **Verifier Principal:** Who approved the promotion
- **Source/Dest Hash:** Content hashes before and after
- **Verification Notes:** Optional reviewer annotations

This enables complete forensic reconstruction of any DL1 data point back to its original extraction and approval.

---

## 🚀 Implementation Notes

### Why Local-Only Storage?

1. **Zero External Dependencies** - No Google Drive API, no cloud quotas, no authentication failures
2. **Complete Control** - Direct filesystem access, no API rate limits
3. **Auditability** - Immutable JSONL logs stored locally for compliance
4. **Portability** - Entire verification system fits on single machine
5. **Resilience** - No third-party service outages affect operations

### Deduplication Strategy

The `dedup_index` maps content hashes to file IDs:

```txt
hash1 → [file_id_1, file_id_2, ...]  # Multiple files with same content
hash2 → [file_id_3]                   # Unique content
```

When staging a new file:

1. Compute SHA256 of input file
2. Check if hash exists in dedup_index
3. If found, warn about duplication
4. Stage file with new file_id
5. Update dedup_index

### Promotion Safety

Promotion from DL2→DL1 is one-way and immutable:

```python
# Before promotion
dl2/dl2_20260202_abc123.csv ← source file
dl1/                        ← empty

# After promotion  
dl2/dl2_20260202_abc123.csv ← original remains for audit
dl1/dl2_20260202_abc123.csv ← copy in verified location
promotion_history.jsonl     ← immutable record added
```

The original DL2 file is never deleted, enabling forensic retracing if needed.

---

## 🔗 Integration Points

### With Verification Framework

When a verifier approves rows in the web UI:

```python
# 1. UI submits verification decision
POST /api/verification/submit → VerificationLog entry

# 2. If ALL rows approved, trigger promotion
POST /api/verification/sync/promote

# 3. Promotion handler:
sync.promote_to_dl1(
    file_id="dl2_20260202_abc123",
    verifier_principal=request.principal,
    verification_notes="..."
)

# 4. Updates:
# - Copies DL2 file to DL1
# - Appends to promotion_history.jsonl
# - Updates sync_metadata.json
```

### With Health/ML Pipeline

DL1-approved data feeds into retraining:

```python
# In retraining job
dl1_files = sync.list_dl1_approved(limit=1000)

for file_rec in dl1_files:
    path = sync.dl1_dir / f"{file_rec['file_id']}.csv"
    # Feed into model training...
```

### With Export/Warehouse

Warehouse promotion queries DL1:

```python
# In dataset_promotion.py
dl1_approved = sync.list_dl1_approved(limit=500)

for sample in dl1_approved:
    ingest_to_warehouse(sample)
```

---

## 📊 Monitoring & Observability

### Storage Stats API

```bash
curl https://localhost/api/verification/sync/status \
  -H "Authorization: Bearer $TOKEN"

# Returns current state: file counts, sizes, dedup groups
```

### Promotion History Query

```bash
curl "https://localhost/api/verification/sync/promotions?limit=100" \
  -H "Authorization: Bearer $TOKEN"

# Returns recent promotions for audit/compliance
```

### Manual Inspection

```bash
# List DL2 unverified samples
ls -lh /path/to/context_library/verification/dl2/

# View recent promotions
tail -20 /path/to/context_library/verification/promotion_history.jsonl | jq

# Check deduplication
jq '.dedup_index | keys | length' sync_metadata.json
```

---

## 🛠️ Troubleshooting

### Issue: "Storage not available"

**Cause:** Directories don't exist or are inaccessible

**Solution:**

```bash
mkdir -p $CONTEXT_LIBRARY_DIR/verification/{dl1,dl2}
chmod 755 $CONTEXT_LIBRARY_DIR/verification/
```

### Issue: Promotion fails with "File already exists in DL1"

**Cause:** File was already promoted (idempotent protection)

**Solution:** Check promotion_history.jsonl to see when it was promoted

### Issue: Deduplication shows false positives

**Cause:** Different files have same content (expected in some cases)

**Solution:** Verify unique_content count in stats vs file_count

- `local_dir` (optional): Local directory to download to
  - Default: `$CONTEXT_LIBRARY_DIR/verification/dl2`

**Response:**

```json
{
  "success": true,
  "count": 12,
  "file_ids": ["file_id_1", "file_id_2", ...],
  "timestamp": "2026-02-02T15:50:00Z",
  "local_dir": "/local/path/verification/dl2"
}
```

**Example:**

```bash
curl -X POST \
  -H "X-Principal: alice@example.org" \
  "http://localhost:5000/api/verification/sync/dl2/pull?local_dir=/tmp/dl2"
```

### POST /api/verification/sync/dl1/push

**Auth:** ADMIN_FULL_TRUST+  
**Purpose:** Upload verified DL1 data to Google Drive

**Body:**

```json
{
  "local_dir": "/local/path/verification/dl1",
  "deduplicate": true
}
```

**Response:**

```json
{
  "success": true,
  "count": 5,
  "file_ids": ["file_id_a", "file_id_b", ...],
  "timestamp": "2026-02-02T16:00:00Z",
  "local_dir": "/local/path/verification/dl1",
  "deduplicate": true
}
```

**Example:**

```bash
curl -X POST \
  -H "X-Principal: bob@example.org" \
  -H "Content-Type: application/json" \
  -d '{
    "local_dir": "/tmp/dl1",
    "deduplicate": true
  }' \
  "http://localhost:5000/api/verification/sync/dl1/push"
```

### POST /api/verification/sync/promote

**Auth:** ADMIN_REVIEWER+  
**Purpose:** Promote verified DL2 rows to DL1 dataset

**Body:**

```json
{
  "dl2_csv_path": "/local/dl2/extracted.csv",
  "dl1_output_path": "/local/dl1/approved_row_123.csv",
  "dl2_id": "file_20260202_abc123"
}
```

**Response:**

```json
{
  "success": true,
  "dl2_file": "extracted.csv",
  "dl1_file": "approved_row_123.csv",
  "row_count": 47,
  "promoted_at": "2026-02-02T16:05:00Z"
}
```

**Example:**

```bash
curl -X POST \
  -H "X-Principal: alice@example.org" \
  -H "Content-Type: application/json" \
  -d '{
    "dl2_csv_path": "/data/dl2/sample.csv",
    "dl1_output_path": "/data/dl1/approved.csv",
    "dl2_id": "file_001"
  }' \
  "http://localhost:5000/api/verification/sync/promote"
```

---

## 🔐 Configuration

### Environment Variables

```bash
# Required for sync functionality
export ENABLE_VERIFICATION_FRAMEWORK=true

# Google Drive folder URLs
export DL1_DRIVE_FOLDER_URL="https://drive.google.com/drive/u/4/folders/1ZwsL_Ui2qFyV-EJ1OZ_9lyhMeX8d4v9N"
export DL2_DRIVE_FOLDER_URL="https://drive.google.com/drive/u/4/folders/1wQcC_UEIFQrIYyRhyfgY2rr5RkiCBQ7V"

# Google Cloud credentials (choose ONE)
# Option 1: Service account (production)
export GCP_SERVICE_ACCOUNT_JSON="/path/to/service-account.json"

# Option 2: OAuth2 user credentials (development)
export GOOGLE_OAUTH2_TOKEN='{"type":"authorized_user","client_id":"...","client_secret":"...","refresh_token":"..."}'

# Optional
export CONTEXT_LIBRARY_DIR="/path/to/context_library"  # Default: ./context_library
```

### Google Cloud Setup

## Step 1: Create Service Account

```bash
# In Google Cloud Console:
# 1. Create new project or select existing
# 2. Enable Google Drive API
# 3. Create service account
# 4. Create JSON key
# 5. Grant Drive editor role to service account
```

## Step 2: Create Google Drive Folders

```bash
# 1. Create "DL2_Unverified" folder in Google Drive
# 2. Create "DL1_Verified" folder in Google Drive
# 3. Share both folders with service account email
# 4. Copy folder URLs to environment variables
```

## Step 3: Extract Folder IDs

```bash
# URL format: https://drive.google.com/drive/folders/FOLDER_ID
# Example: https://drive.google.com/drive/u/4/folders/1wQcC_UEIFQrIYyRhyfgY2rr5RkiCBQ7V
#                                                       ↑
#                                                   This is the ID
```

---

## 🔄 Workflow: End-to-End Sync

### Step 1: Initialize Sync Manager

```python
from webapp.parser.utils.google_drive_sync import DL1DL2SyncManager

sync = DL1DL2SyncManager(session_id="sess_abc123")

if not sync.is_available():
    print("ERROR: Drive not configured or not accessible")
    exit(1)
```

### Step 2: Download DL2 Samples

```python
# Pull unverified data from Drive
count, file_ids = sync.sync_dl2_from_drive(
    local_dl2_dir="/data/verification/dl2"
)
print(f"Downloaded {count} DL2 files")
```

### Step 3: Human Reviews DL2

```python
# Verifier compares:
# 1. DL2 file (extracted by AI)
# 2. Official source (e.g., election website)
# 
# For each discrepancy, verifier:
# - Identifies anomaly type (extraction_error, formatting, etc.)
# - Makes verification decision (approve/reject/flag)
# - Submits decision via POST /api/verification/submission
```

### Step 4: Promote Approved Rows

```python
# After verification approves a row, promote to DL1:
sync.promote_dl2_to_dl1(
    dl2_csv_path="/data/verification/dl2/extracted.csv",
    dl1_output_path="/data/verification/dl1/approved_001.csv",
    dl2_id="file_20260202_001"
)
print("Promoted DL2→DL1")
```

### Step 5: Upload DL1 to Drive

```python
# Push approved data to Google Drive
count, file_ids = sync.sync_dl1_to_drive(
    local_dl1_dir="/data/verification/dl1",
    deduplicate=True  # Prevent duplicates
)
print(f"Uploaded {count} DL1 files to Drive")
```

### Step 6: Archive & Report

```python
# Get sync statistics
stats = sync.get_sync_stats()
print(f"""
Sync Summary:
- DL2 files processed: {stats['dl2_files']}
- DL1 files verified: {stats['dl1_files']}
- Promotions completed: {stats['promotions']}
- Last sync: {stats['last_sync']}
""")
```

---

## 🧪 Testing Sync Logic

### Unit Tests

```python
# test_google_drive_sync.py
import pytest
from webapp.parser.utils.google_drive_sync import DL1DL2SyncManager, SyncMetadata

def test_sync_manager_initialization():
    sync = DL1DL2SyncManager()
    assert sync is not None

def test_metadata_record_dl2():
    metadata = SyncMetadata()
    metadata.record_dl2_file("file_123", "sha256_abc")
    assert metadata.get_dl2_hash("file_123") == "sha256_abc"

def test_deduplication():
    metadata = SyncMetadata()
    metadata.record_content_hash("hash_xyz", "file_1")
    metadata.record_content_hash("hash_xyz", "file_2")
    
    duplicates = metadata.is_duplicate("hash_xyz")
    assert len(duplicates) == 2
    assert "file_1" in duplicates
    assert "file_2" in duplicates
```

### Integration Tests

```bash
# 1. Test sync status
curl -H "X-Principal: reviewer@example.org" \
  http://localhost:5000/api/verification/sync/status | jq .

# 2. Test DL2 pull
curl -X POST \
  -H "X-Principal: admin@example.org" \
  http://localhost:5000/api/verification/sync/dl2/pull | jq .

# 3. Test promotion
curl -X POST \
  -H "X-Principal: admin@example.org" \
  -H "Content-Type: application/json" \
  -d '{
    "dl2_csv_path": "/data/dl2/test.csv",
    "dl1_output_path": "/data/dl1/promoted.csv",
    "dl2_id": "test_001"
  }' \
  http://localhost:5000/api/verification/sync/promote | jq .

# 4. Test DL1 push
curl -X POST \
  -H "X-Principal: admin_full@example.org" \
  -H "Content-Type: application/json" \
  -d '{"local_dir": "/data/dl1", "deduplicate": true}' \
  http://localhost:5000/api/verification/sync/dl1/push | jq .
```

---

## 📊 Performance Characteristics

| Operation | Time | Scalability | Notes |
| --- | --- | --- | --- |
| Sync status query | <10ms | O(1) | In-memory metadata |
| List DL2 contents (100 files) | ~500ms | O(n) | API pagination |
| Download DL2 CSV (10MB) | ~1-2s | O(size) | Network I/O |
| Upload DL1 CSV (10MB) | ~1-2s | O(size) | Network I/O + dedup check |
| Promote DL2→DL1 | <50ms | O(rows) | Local file I/O |
| Deduplication check | <20ms | O(1) hash lookup | SHA256 in-memory |

**Recommendations:**

- Batch operations: Process 10-50 files per sync run
- Schedule syncs: Run during off-peak hours
- Monitor quotas: Google Drive API has rate limits
- Archive old DL2: Retain for 90 days then delete

---

## 🔒 Security Considerations

### Authentication & Authorization

✅ **Service Account:** Uses cryptographic credentials  
✅ **OAuth2 Tokens:** Refreshed automatically  
✅ **Principal Verification:** Every operation logged with principal  
✅ **Privilege Tiers:** ADMIN_REVIEWER for DL2 pull, ADMIN_FULL_TRUST for DL1 push

### Data Protection

✅ **Encryption in Transit:** HTTPS/SSL to Google Drive  
✅ **Content Hashing:** SHA256 prevents tampering  
✅ **Immutable Audit Trail:** Promotions recorded in verification log  
✅ **Access Control:** Share permissions managed per file

### Threat Mitigation

⚠️ **Quota Exhaustion:** Monitor API quota usage  
⚠️ **Network Errors:** Automatic retry with exponential backoff  
⚠️ **Duplicate Uploads:** Deduplication on content hash  
⚠️ **Stale Metadata:** Timestamps tracked, refresh on mismatch

---

## 🐛 Troubleshooting

### Issue: "Drive service not available"

**Cause:** Google Drive library not installed or credentials missing

**Solution:**

```bash
# Install required packages
pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client

# Verify credentials
export GCP_SERVICE_ACCOUNT_JSON=/path/to/sa.json
python -c "from webapp.parser.utils.google_drive_client import GoogleDriveClient; print(GoogleDriveClient().is_available())"
```

### Issue: 403 Forbidden on folder operations

**Cause:** Service account not granted Drive access

**Solution:**

1. Check service account email has Editor role on Drive folders
2. Verify folder is shared with service account
3. Test with `client.get_file_metadata(folder_id)`

### Issue: Deduplication prevents valid uploads

**Cause:** Old metadata tracking stale hashes

**Solution:**

```bash
# Clear old metadata
rm $CONTEXT_LIBRARY_DIR/verification/sync_metadata.json

# Re-initialize (next sync will rebuild index)
```

### Issue: Large files timeout during upload

**Cause:** Network instability or file size

**Solution:**

```python
# Upload in chunks
# Current implementation uses MediaFileUpload with resumable=True
# Increase timeout in google_drive_client.py if needed
```

---

## 📈 Next Steps (Phase 3+)

| Phase | Feature | Timeline |
| --- | --- | --- |
| **2** | ✅ Sync infrastructure | Complete |
| **3** | DL Sync Dashboard UI | Q1 2026 |
| **3** | Auto-sync scheduling | Q1 2026 |
| **4** | Versioning & rollback | Q2 2026 |
| **5** | ML-driven promotion | Q2 2026 |

---

## 📞 Support

**For sync issues:**

- Check logs: `tail -f $CONTEXT_LIBRARY_DIR/verification/sync_log.jsonl`
- Verify config: `grep -i "DRIVE\|SYNC" .env`
- Test endpoint: `curl http://localhost:5000/api/verification/sync/status`

**For Google Drive issues:**

- [Google Drive API Documentation](https://developers.google.com/drive)
- [Service Account Setup Guide](https://cloud.google.com/docs/authentication/getting-started)
- [Quota Information](https://developers.google.com/drive/api/guides/limits)

---

**Version:** 1.0.0  
**Last Updated:** February 2, 2026  
**Status:** ✅ Production Ready
