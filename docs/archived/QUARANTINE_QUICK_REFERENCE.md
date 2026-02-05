# Quick Reference: Quarantine Transparency System

## 🎯 Quick Start

### View Quarantine UI

```txt
https://localhost:5000/quarantine/review
(Requires: Client certificate)
```

### API Endpoints

```txt
GET  /api/quarantine/pending     → List pending items
GET  /api/quarantine/reviewed    → View decisions
GET  /api/quarantine/item/<id>   → Get details
POST /api/quarantine/review      → Submit decision
GET  /api/quarantine/stats       → Queue stats
```

### Check Audit Trail

```bash
cat LOG_DIR/quarantine/review_decisions.jsonl | jq .
```

---

## 📋 Quarantine Entry Structure

```json
{
  "id": "q_1704067200000",
  "url": "https://example.org/results",
  "reason": "LOW_TRUST_SCORE",
  "reason_description": "URL scored low on security assessment...",
  "trust_score": 42.5,
  "trust_factors": {
    "domain_reputation": "unknown",
    "ssl_valid": true,
    "rate_limit_risk": "medium"
  },
  "data_notices": [
    {
      "data_type": "trust_score",
      "description": "Computed trust score: 42.5/100",
      "usage": "Security filtering to prevent untrusted sources",
      "retention_days": 30
    }
  ],
  "created_at": "2024-01-01T12:00:00Z",
  "review_decision": "pending",
  "reviewed_by": null,
  "review_notes": null
}
```

---

## 🔑 Key Classes

### QuarantineReason

```python
LOW_TRUST_SCORE          # Score < threshold
SUSPICIOUS_HOST          # CDN/cloud storage pattern
INVALID_URL              # Format error
CLOUDFLARE_CHALLENGE     # CAPTCHA detected
SSL_VERIFICATION_FAILED  # Cert validation error
RATE_LIMITED             # Rate limit hit
MANUAL_FLAG              # User/admin flag
```

### DataCollectionNotice

```python
DataCollectionNotice(
    data_type="trust_score",                    # What
    description="Computed: 42.5/100",           # Specifics
    usage="Security filtering, prevent attacks", # Why
    retention_days=30                           # How long
)
```

### QuarantineEntry

```python
entry = QuarantineEntry(
    url="https://example.org/results",
    reason="LOW_TRUST_SCORE",
    session_id="sess_xxx",
    principal="user@org.gov",
    trust_score=42.5,
    trust_factors={...},
    data_notices=[DataCollectionNotice(...), ...],
    metadata={...}
)
```

### QuarantineQueue

```python
queue = get_quarantine_queue()
queue.enqueue(entry)                    # Add item
queue.get_pending(limit=100)            # List pending
queue.review(id, decision, principal)   # Record decision
queue.get_reviewed(limit=100)           # View history
queue.clear_old(days=30)                # Cleanup
```

---

## 🔌 Integration Points

### 1. Trust Scorer (html_election_parser.py)

```python
if should_quarantine(trust_score, target_url, privilege_tier):
    queue = get_quarantine_queue()
    queue.enqueue(
        url=target_url,
        reason=QuarantineReason.LOW_TRUST_SCORE,
        principal=principal,
        trust_score=trust_score,
        trust_factors=trust_factors,
        data_notices=[...]
    )
```

### 2. Flask App (Smart_Elections_Parser_Webapp.py)

```python
from webapp.parser.quarantine_endpoints import quarantine_bp
app.register_blueprint(quarantine_bp)
```

### 3. Authentication (Decorators)

```python
@quarantine_bp.route("/...")
@_require_quarantine_enabled
@_require_reviewer
def endpoint_name():
    principal = _get_reviewer_principal()
    # principal from client cert or SSO
```

---

## 📊 Quarantine Reasons & Descriptions

| Reason | Description |
| -------- | ------------- |
| `LOW_TRUST_SCORE` | URL scored low on security assessment. May be malicious or unreliable. |
| `SUSPICIOUS_HOST` | Host matches known third-party CDN/cloud storage patterns. Requires review. |
| `INVALID_URL` | URL format invalid or missing required components. |
| `CLOUDFLARE_CHALLENGE` | Cloudflare CAPTCHA detected. Browser interaction may be required. |
| `SSL_VERIFICATION_FAILED` | SSL certificate validation failed. Possible MITM attack. |
| `RATE_LIMITED` | URL or source IP hit rate limits. Retry after cooldown. |
| `MANUAL_FLAG` | Manually flagged by user or administrator. |

---

## 🧪 Common Operations

### Enqueue URL

```python
from webapp.parser.quarantine_endpoints import (
    get_quarantine_queue,
    QuarantineEntry,
    DataCollectionNotice,
)

queue = get_quarantine_queue()
entry = QuarantineEntry(
    url="https://example.org/results",
    reason="LOW_TRUST_SCORE",
    principal="admin@org.gov",
    trust_score=35.0,
    data_notices=[
        DataCollectionNotice(
            data_type="trust_score",
            description="Score: 35.0/100 (below 50 threshold)",
            usage="Security filtering",
            retention_days=30
        ),
    ]
)
queue.enqueue(entry)
```

### Review Decision

```python
queue = get_quarantine_queue()
queue.review(
    item_id="q_1704067200000",
    decision="approved",
    reviewed_by="reviewer@org.gov",
    notes="URL verified as legitimate"
)
```

### Get Pending Items

```python
queue = get_quarantine_queue()
pending = queue.get_pending(limit=100)
for entry in pending:
    print(f"{entry.url}: {entry.reason}")
```

### Get Statistics

```python
queue = get_quarantine_queue()
stats = queue.get_stats()
print(f"Pending: {stats['total_pending']}")
print(f"By reason: {stats['pending_by_reason']}")
```

---

## 📂 File Locations

| File | Purpose |
| -------- | --------- |
| `webapp/parser/quarantine_queue.py` | Queue implementation |
| `webapp/parser/quarantine_endpoints.py` | API + UI routes |
| `LOG_DIR/quarantine/queue.jsonl` | Persisted entries |
| `LOG_DIR/quarantine/review_decisions.jsonl` | Audit trail |

---

## 🔐 Authentication

All endpoints require:

- **Client Certificate**: CN field extracted as principal
- **Fallback**: X-OIDC-* headers (SSO principal)
- **Error**: Returns 401 if not authenticated

### Test with cURL

```bash
curl -X GET \
  --cert client.crt \
  --key client.key \
  https://localhost:5000/api/quarantine/pending
```

---

## 💾 Persistence

**Format**: JSONL (one JSON object per line)  
**Location**: `LOG_DIR/quarantine/`  
**Files**:

- `queue.jsonl` - All quarantine entries
- `review_decisions.jsonl` - Review history

**Benefits**:

- Text-based (human-readable)
- Grep-able (searchable)
- No database needed
- Easy backup/archival
- Survives restarts

---

## 🛠️ Troubleshooting

### No items showing in UI

1. Check `LOG_DIR/quarantine/queue.jsonl` exists
2. Verify authentication (need client cert)
3. Check browser console for errors
4. Try `/api/quarantine/stats` to verify queue

### Can't submit review

1. Verify principal extracted (check logs)
2. Ensure `decision` is valid (approved/rejected/modified)
3. Check `certification_reason` is provided
4. Verify item ID is correct

### Lost audit trail

1. Check `LOG_DIR/quarantine/review_decisions.jsonl`
2. If missing, may be cleaned up (default 30 days)
3. Consider archiving JSONL files before cleanup

---

## 📈 Monitoring

### Queue Size

```bash
wc -l LOG_DIR/quarantine/queue.jsonl
```

### Pending Items

```bash
curl https://localhost:5000/api/quarantine/stats \
  --cert client.crt --key client.key | jq .
```

### Review Decisions

```bash
cat LOG_DIR/quarantine/review_decisions.jsonl | jq -s 'group_by(.decision) | map({key: .[0].decision, count: length})'
```

---

## 🚀 Deployment

1. **Enable** in config: Set `ENABLE_VERIFICATION_FRAMEWORK=true`
2. **Create** quarantine dir: Auto-created on first enqueue
3. **Register** blueprint: Done in `Smart_Elections_Parser_Webapp.py`
4. **Test** endpoints: Navigate to `/quarantine/review`
5. **Monitor** audit trail: Check `review_decisions.jsonl`

---

## 📞 Support

### For Issues

1. Check logs: `LOG_DIR/quarantine/` JSONL files
2. Verify authentication: Client cert CN field
3. Review code: `quarantine_queue.py` + `quarantine_endpoints.py`
4. See docs: `QUARANTINE_TRANSPARENCY_IMPLEMENTATION.md`

### For Questions

1. Review this quick reference
2. Check session memory: `/memories/session_comprehensive_summary.md`
3. Read main implementation doc
4. Check code comments
