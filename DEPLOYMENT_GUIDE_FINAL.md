# 🚀 Quarantine Transparency System - Deployment Guide

**Version**: 1.0 - Production Ready  
**Last Updated**: Current Session  
**Status**: ✅ READY FOR DEPLOYMENT

---

## Quick Start (5 minutes)

### 1. Verify Files Are In Place

```bash
# Check core files exist
ls -l webapp/parser/quarantine_queue.py
ls -l webapp/parser/quarantine_endpoints.py

# Verify blueprint registration in main app
grep -n "quarantine_bp" webapp/Smart_Elections_Parser_Webapp.py
```

### 2. Start the Application

```bash
python -m flask run
# or with gunicorn:
gunicorn -w 4 -b 0.0.0.0:5000 'webapp.Smart_Elections_Parser_Webapp:app'
```

### 3. Access the Review Interface

1. Install client certificate (or use dev bypass if enabled)
2. Navigate to: `https://localhost:5000/quarantine/review`
3. You should see the interactive review UI

---

## Detailed Deployment Steps

### Prerequisites Checklist

- [ ] Python 3.8+ installed
- [ ] Flask application running or deployed
- [ ] Client certificate authentication configured
- [ ] `LOG_DIR` environment variable set and writable
- [ ] All dependencies installed (`requirements.txt`)
- [ ] Database (if used) initialized

### Step 1: Code Installation

```bash
# Copy files to correct locations
cp quarantine_queue.py <your-project>/webapp/parser/
cp quarantine_endpoints.py <your-project>/webapp/parser/

# Verify permissions
chmod 644 webapp/parser/quarantine_*.py
```

### Step 2: Configuration

#### Environment Variables (Optional)

```bash
# Enable/disable feature (default: true)
export ENABLE_VERIFICATION_FRAMEWORK=true

# Set quarantine data retention (days)
# Default: 30 days
export QUARANTINE_RETENTION_DAYS=30

# Enable dev bypass for local testing
export ALLOW_DEV_NO_PRINCIPAL=true  # local dev only!
```

#### Flask App Changes

Already completed - verify in `Smart_Elections_Parser_Webapp.py`:

```python
# Should see this around line 395-410:
from webapp.parser.quarantine_endpoints import quarantine_bp
app.register_blueprint(quarantine_bp)
```

### Step 3: Database & Storage

The system uses JSONL files (no database required):

```bash
# Ensure directory is writable
mkdir -p $LOG_DIR/quarantine
chmod 755 $LOG_DIR/quarantine

# The following files will be auto-created:
# - $LOG_DIR/quarantine/queue.jsonl (all entries)
# - $LOG_DIR/quarantine/review_decisions.jsonl (audit trail)
```

### Step 4: Verify Installation

```bash
# Start Flask app (ensure client cert auth is configured)
python -c "
from webapp.parser.quarantine_queue import QuarantineQueue
from webapp.parser.quarantine_endpoints import quarantine_bp
print('✅ All imports successful')
"

# Test API endpoint
curl -X GET https://localhost:5000/api/quarantine/stats \
  --cert client.crt --key client.key \
  -H "Accept: application/json"
# Should return: {"pending": 0, "reviewed": 0, "by_reason": {}}
```

---

## Testing Workflow

### Test 1: Manual UI Review

**Objective**: Verify web interface loads and displays correctly

**Steps**:

1. Navigate to `https://<your-domain>/quarantine/review` (requires auth)
2. Verify two tabs visible: "Pending Review" and "Review History"
3. If no pending items, that's normal (will appear when URLs are quarantined)
4. Data collection notices should be visible when items exist

**Expected Result**:

- UI loads without errors
- Authentication required
- Empty state shows gracefully

---

### Test 2: Trigger Quarantine Entry

**Objective**: Verify URL gets quarantined with correct metadata

**Steps**:

1. In your URL list, add a URL with suspicious characteristics (e.g., `https://docs.google.com/...`)
2. Run parser on this URL
3. Monitor logs for quarantine entry:

   ```bash
   tail -f $LOG_DIR/quarantine/queue.jsonl
   ```

4. Entry should appear with reason and data collection notices

**Expected Result**:

```json
{
  "id": "...",
  "url": "https://...",
  "reason": "SUSPICIOUS_HOST",
  "data_notices": [
    {
      "data_type": "trust_score",
      "usage": "Security filtering..."
    }
  ]
}
```

---

### Test 3: API - Get Pending Items

**Objective**: Verify API endpoint returns quarantined URLs

**Command**:

```bash
curl -X GET "https://<your-domain>/api/quarantine/pending?limit=10" \
  --cert client.crt --key client.key \
  -H "Accept: application/json" | jq .
```

**Expected Result**:

```json
{
  "pending": [
    {
      "id": "qid_123...",
      "url": "https://...",
      "reason": "SUSPICIOUS_HOST",
      "created_at": "2024-01-15T10:30:00Z",
      "data_notices": [...]
    }
  ]
}
```

---

### Test 4: Review Submission

**Objective**: Verify reviewer can submit decision and it's logged

**Command**:

```bash
curl -X POST "https://<your-domain>/api/quarantine/review" \
  --cert client.crt --key client.key \
  -H "Content-Type: application/json" \
  -d '{
    "quarantine_id": "qid_123...",
    "decision": "approve",
    "notes": "URL appears legitimate upon review"
  }' | jq .
```

**Expected Result**:

```json
{
  "success": true,
  "decision_id": "rev_456...",
  "message": "Decision recorded"
}
```

**Verify Audit Trail**:

```bash
tail -1 $LOG_DIR/quarantine/review_decisions.jsonl | jq .
```

Should show:

- `quarantine_id`: ID of quarantine entry
- `decision`: "approve" or "reject"
- `reviewed_by`: Principal from client cert
- `timestamp`: ISO datetime
- `notes`: Your review notes

---

### Test 5: Approved URL Can Be Processed

**Objective**: After approval, verify URL is no longer quarantined

**Steps**:

1. Submit approval for test URL (from Test 4)
2. Add same URL to input again
3. Run parser - should process normally
4. Check that it's NOT in quarantine queue

**Expected Result**:

- URL processes without quarantine
- New results in output folder
- No entry in `queue.jsonl`

---

## Monitoring & Maintenance

### Daily Checks

```bash
# 1. Check queue size (should not grow unbounded)
wc -l $LOG_DIR/quarantine/queue.jsonl

# 2. Monitor approval rate
grep '"decision":' $LOG_DIR/quarantine/review_decisions.jsonl | \
  grep -c '"approve"'  # Count approvals

# 3. Check for stuck entries (older than 30 days)
jq '.created_at' $LOG_DIR/quarantine/queue.jsonl | \
  head -5  # Check oldest entries
```

### Weekly Reports

```bash
# Quarantine summary
curl -s "https://<your-domain>/api/quarantine/stats" \
  --cert client.crt --key client.key | jq .

# Export this week's decisions
jq 'select(.timestamp >= "2024-01-15")' \
  $LOG_DIR/quarantine/review_decisions.jsonl > /tmp/weekly_review.jsonl
```

### Monthly Analysis

```bash
# Top quarantine reasons
jq '.reason' $LOG_DIR/quarantine/queue.jsonl | sort | uniq -c | sort -rn

# Approval rate
total=$(wc -l < $LOG_DIR/quarantine/review_decisions.jsonl)
approved=$(grep -c '"decision": "approve"' $LOG_DIR/quarantine/review_decisions.jsonl)
echo "Approval rate: $((approved * 100 / total))%"

# Which reviewers are active
jq '.reviewed_by' $LOG_DIR/quarantine/review_decisions.jsonl | sort | uniq -c
```

---

## Troubleshooting

### Issue: "401 Unauthorized" accessing `/quarantine/review`

**Cause**: Client certificate not present or not recognized

**Solution**:

```bash
# Verify cert is loaded
echo $SSL_CERT_FILE
echo $SSL_KEY_FILE

# Try dev bypass (local only!)
export ALLOW_DEV_NO_PRINCIPAL=true
# But DO NOT use in production!
```

### Issue: No quarantine entries appearing

**Cause**: Trust scorer not triggering quarantine

**Solution**:

1. Check log output for trust score:

   ```bash
   grep "trust_score" $LOG_DIR/*.log
   ```

2. Verify URL is actually suspicious enough
3. Check if `should_quarantine()` threshold is too high

### Issue: Directory permission errors

**Cause**: `$LOG_DIR/quarantine/` not writable

**Solution**:

```bash
# Fix permissions
chmod 755 $LOG_DIR/quarantine
chmod 644 $LOG_DIR/quarantine/*.jsonl

# Verify
ls -ld $LOG_DIR/quarantine
```

### Issue: Old entries not being cleaned up

**Cause**: Cleanup function not running or disabled

**Solution**:

```bash
# Manually trigger cleanup
python -c "
from webapp.parser.quarantine_queue import QuarantineQueue
q = QuarantineQueue()
q.clear_old()
print('Cleanup complete')
"
```

---

## Security Verification

### ✅ Authentication Enforced

```bash
# This should FAIL (no cert)
curl -X GET "https://<your-domain>/api/quarantine/pending"
# Expected: 401 Unauthorized

# This should SUCCEED (with cert)
curl -X GET "https://<your-domain>/api/quarantine/pending" \
  --cert client.crt --key client.key
# Expected: 200 OK with JSON data
```

### ✅ Authorization Enforced

```bash
# Verify principal is captured in decisions
jq '.reviewed_by' $LOG_DIR/quarantine/review_decisions.jsonl | head -5

# Each should show certificate CN, e.g.: "user@example.com"
```

### ✅ Audit Trail Immutable

```bash
# Audit trail is append-only
# Verify no lines are removed
wc -l $LOG_DIR/quarantine/review_decisions.jsonl
# Run again later - count should only increase
```

---

## Performance Baseline

For sizing / capacity planning:

| Metric | Baseline | Safe Limit |
| -------- | ---------- | ----------- |
| Quarantine entries | < 1000 | 10,000 |
| Daily reviews | < 100 | 1,000 |
| API response time | < 100ms | < 500ms |
| UI load time | < 1s | < 3s |
| JSONL file size | < 5MB | 100MB |

---

## Rollback Plan

If issues encountered:

### Option 1: Disable Feature

```bash
# Set feature flag to false
export ENABLE_VERIFICATION_FRAMEWORK=false

# Restart app
systemctl restart your-app

# URLs will not be quarantined, but system won't break
```

### Option 2: Remove Integration

```bash
# Comment out in html_election_parser.py:
# queue.enqueue(...)  # DISABLED

# Restart app
```

### Option 3: Clean Start

```bash
# Back up audit trail (important!)
cp $LOG_DIR/quarantine/review_decisions.jsonl /backup/

# Delete quarantine data
rm -rf $LOG_DIR/quarantine/

# Restart app
# Quarantine data will be recreated fresh
```

---

## Post-Deployment Checklist

- [ ] UI accessible at `/quarantine/review`
- [ ] Authentication working (401 without cert)
- [ ] API endpoints responding
- [ ] Quarantine entries being created
- [ ] Reviews can be submitted
- [ ] Audit trail being logged
- [ ] No errors in application logs
- [ ] Cleanup running (old entries removed)
- [ ] Stakeholders trained on UI
- [ ] Monitoring in place (check queue size daily)

---

## Support & Documentation

**Quick Links**:

- API Documentation: See `QUARANTINE_QUICK_REFERENCE.md`
- Architecture Details: See `QUARANTINE_TRANSPARENCY_IMPLEMENTATION.md`
- Implementation Checklist: See `IMPLEMENTATION_CHECKLIST_COMPLETE.md`

**Common Operations**:

- View pending quarantines: `GET /api/quarantine/pending`
- Submit review: `POST /api/quarantine/review`
- Check stats: `GET /api/quarantine/stats`
- Manual audit: `jq .` on JSONL files

---

## Success Criteria ✅

Your deployment is successful when:

1. ✅ You can navigate to `/quarantine/review` with client cert
2. ✅ At least one URL has been quarantined (appears in pending list)
3. ✅ You can click "Approve" or "Reject" on quarantined URL
4. ✅ Decision appears in review history
5. ✅ Audit trail shows reviewer principal and timestamp
6. ✅ No errors in application logs

**Estimated Time to Success**: 15-30 minutes

---

**Version**: 1.0 Production  
**Last Updated**: Current Session  
**Status**: ✅ READY FOR PRODUCTION DEPLOYMENT
