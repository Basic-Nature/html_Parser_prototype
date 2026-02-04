# Election Operations Playbook: Real-Time Monitoring & Response

**Purpose**: Step-by-step operational guide for election day (or election week) monitoring using the Smart Elections Parser with VocabLoader.

**Audience**: Election Day Coordinator, Data Operations Manager, IT Support  
**Frequency**: Daily election cycles (or continuous monitoring for rolling elections)  
**Criticality**: HIGH (CORE ELECTION INTEGRITY OPERATIONS)

---

## Pre-Election Procedures (Day Before)

### 06:00 PM – Preparation Meeting

**Attendees**: Coordinator, Data Lead, IT Support  
**Duration**: 30 minutes

**Agenda**:

1. Confirm election type (general, primary, runoff, special measure)
2. Verify data sources (county websites, FTP servers, secure upload links)
3. Review expected data formats (PDF, XLSX, CSV, JSON)
4. Identify "canary" data (small, early-arriving test batch)
5. Confirm alert contact list (escalation chain)

**Checklist**:

- [ ] All data source URLs verified (DNS resolution tested)
- [ ] VPN/network access confirmed for county FTP servers
- [ ] Backup internet connection available (failover tested)
- [ ] Database connection to PostgreSQL warehouse tested (credentials fresh)
- [ ] Slack/email alerts configured
- [ ] War room setup (multiple monitors, shared communication channel)

**Output**: `pre_election_checklist.md` filled in by Data Lead

---

### 07:00 PM – Technical Validation

**Person**: IT Support Lead

**Tasks**:

1. **Start Fresh VM / Container**

   ```bash
   docker pull smart-elections-parser:latest
   docker run -d --name election-parser \
     -e ELECTION_MODE=false \
     -e VOCAB_TRUST_THRESHOLD=0.90 \
     -v /data/elections:/app/output \
     smart-elections-parser:latest
   ```

2. **Verify VocabLoader Initialization**

   ```bash
   python -c "from webapp.parser.config import get_vocab_loader; \
              loader = get_vocab_loader(); \
              print(f'Vocab loaded: {len(loader.load_vocab_set(\"entities/offices.txt\"))} offices')"
   ```

3. **Load All Vocab Entities**

   ```bash
   # entities/offices.txt
   # entities/parties.txt
   # entities/jurisdictions.txt
   # entities/contest_types.txt
   # entities/result_terms.txt
   # validators/office_validators.txt
   # validators/party_validators.txt
   # scoring/anomaly_thresholds.txt
   ```

   Expected: All files load, hash verified, trust scores logged

4. **Database Warm-up**

   ```sql
   -- Ensure warehouse_election_results table exists
   SELECT COUNT(*) FROM warehouse_election_results;
   
   -- Verify write permissions
   INSERT INTO warehouse_election_results 
   (state, county, contest, row_count, timestamp) 
   VALUES ('CA', 'test', 'test_warmup', 0, NOW())
   RETURNING id;
   ```

5. **Test Canary Upload**
   - Upload small test CSV (5 rows, 3 offices)
   - Parse with `election_mode=false` (prep mode)
   - Verify output CSV written to `/output/results_*.csv`
   - Check audit log: `/logs/vocab_audit.jsonl` contains entries
   - Expected: Headers extracted, 5 rows parsed, 0 anomalies

6. **Enable Dry-Run Election Mode**

   ```bash
   export ELECTION_MODE=true
   export ENABLE_SNAPSHOTS=true
   ```

   - Attempt to modify vocab file (should fail with VocabSecurityError)
   - Create snapshot: `loader.create_snapshot()` → `/snapshots/snapshot_TIMESTAMP.json`
   - Verify snapshot contains: offices count, parties count, trust scores, hash

**Output**: `technical_validation_log.txt` with timestamps + result of each test

---

### 08:00 PM – Stakeholder Alignment

**Call**: Coordinator, Data Leads (county), Election Officials

**Topics**:

- Expected data arrival times (midnight vs 8 AM vs rolling)
- Data format agreement (XLSX columns must match specification)
- Escalation path for anomalies (e.g., if duplicate candidate detected, call County Clerk)
- Contact list (phone numbers for key decision-makers)

**Output**: `escalation_matrix.md`

---

## Election Day Morning (Election Mode Active)

### 06:00 AM – System Startup

**Person**: IT Support Lead

**Procedure**:

1. **Verify Database Readiness**

   ```bash
   psql -h ${POSTGRES_HOST} -d ${POSTGRES_DB} -U ${POSTGRES_USER} \
     -c "SELECT version();"
   ```

2. **Enable Election Mode**

   ```python
   from webapp.parser.config import get_vocab_loader
   
   loader = get_vocab_loader(session_id="election_2026_general_ca")
   loader.enable_election_mode()  # Blocks all modifications
   
   # Create initial snapshot
   snapshot_path = loader.create_snapshot(
       description="Election day startup snapshot",
       completion_percentage=0
   )
   print(f"Initial snapshot: {snapshot_path}")
   ```

3. **Start Parser Service**

   ```bash
   systemctl start smart-elections-parser
   # or
   python -m webapp.Smart_Elections_Parser_Webapp &
   ```

4. **Verify Web UI Accessible**
   - Navigate to: <https://electionpulse.org/ballot_lens>
   - Confirm "Election Mode Active" banner displayed
   - Verify no "Edit/Upload" buttons (read-only)

5. **Initialize Audit & Monitoring**

   ```bash
   # Create daily audit log file
   touch /logs/election_audit_2026_general_ca.jsonl
   
   # Start log tail in dedicated terminal
   tail -f /logs/election_audit_2026_general_ca.jsonl | jq '.' > /logs/live_monitor.txt
   
   # Start anomaly watch
   watch -n 5 'grep "anomaly_reason" /logs/election_audit_2026_general_ca.jsonl | tail -10'
   ```

6. **Record Startup Time**

   ```json
   {
     "event": "election_mode_enabled",
     "timestamp": "2026-11-03T06:00:00Z",
     "session_id": "election_2026_general_ca",
     "initial_snapshot_path": "/snapshots/snapshot_2026110306.json",
     "election_officials_notified": true,
     "status": "ready"
   }
   ```

**Output**: `election_day_startup_log.txt`

---

### 07:00 AM – Open War Room

**Attendees**: Coordinator, Data Lead, IT Support (all present + on-call)

**Setup**:

- 3 monitors: (1) Live log tail, (2) DB query results, (3) Email/Slack alerts
- Shared Google Doc with minute-by-minute notes
- Escalation phone lines open
- Coffee/snacks available

**Initial Briefing**:

- All systems operational
- Waiting for first data batch
- Alert thresholds reviewed
- Contact list confirmed

---

### 08:00 AM – First Data Batch Arrives

**Scenario**: County election office uploads first batch of precinct results (PDF)

**Procedure**:

1. **Receive & Log**

   ```bash
   # File arrives at: /uploads/county_results_batch_001.pdf
   # Timestamp: 08:15 AM
   
   echo "{
     \"event\": \"file_received\",
     \"filename\": \"county_results_batch_001.pdf\",
     \"timestamp\": \"2026-11-03T08:15:00Z\",
     \"size_bytes\": 524288,
     \"source\": \"county_elections_ftp\"
   }" >> /logs/election_audit_2026_general_ca.jsonl
   ```

2. **Source Verification**

   ```python
   from webapp.parser.config import get_vocab_loader
   
   loader = get_vocab_loader(session_id="election_2026_general_ca")
   
   source_url = "ftp://results.county.gov/election2026/general_results.pdf"
   source_info = loader.get_verified_source(source_url)
   
   if not source_info or source_info["trust_score"] < 0.90:
       logger.error(f"ALERT: Unverified source {source_url}")
       # Escalate to Coordinator immediately
   else:
       logger.info(f"Source verified: {source_info['authority']}")
   ```

3. **Parse with Election Mode**

   ```python
   # Parser called with election_mode=true
   # (This will fail if someone attempts to modify vocab)
   
   try:
       headers, rows, contest, metadata = parse_pdf(
           page,
           context={"election_mode": True},
           session_id="election_2026_general_ca"
       )
   except VocabSecurityError as e:
       logger.critical(f"VOCAB SECURITY ERROR: {e}")
       # DO NOT PROCEED – notify IT immediately
       # Possible tampering detected
   ```

4. **Score Headers for Anomalies**

   ```python
   loader = get_vocab_loader(session_id="election_2026_general_ca")
   
   score = loader.score_keyword_combination(
       keywords=headers,
       context="header_validation"
   )
   
   if score["confidence"] == "low":
       logger.warning(f"LOW CONFIDENCE: {score}")
       # Log reason + take screenshot
       reason = loader.get_anomaly_reason_definition("suspicious_header")
       if reason["quarantine_required"]:
           # STOP – Manual review required
   ```

5. **Detect Anomalies**

   ```python
   # Common anomalies during first batch:
   
   # Anomaly 1: Duplicate candidates in same race
   if len(candidates) != len(set(candidates)):
       logger.error(f"ANOMALY: Duplicates detected {candidates}")
       # Call county to verify
   
   # Anomaly 2: Vote totals exceed registered voters
   if total_votes > registered_voters:
       logger.warning(f"ANOMALY: Votes exceed voters")
       # Note for post-election analysis
   
   # Anomaly 3: Unrecognized office names
   for office in offices:
       if not loader.resolve_alias("office", office):
           logger.warning(f"ANOMALY: Unknown office {office}")
           # May need to add to vocab (with approval)
   ```

6. **Write to Warehouse**

   ```sql
   INSERT INTO warehouse_election_results 
   (state, county, contest, row_count, column_count, extraction_confidence, 
    source_url, principal, session_id, timestamp)
   VALUES 
   ('CA', 'Santa Clara', 'General 2026', 15250, 8, 0.98,
    'ftp://results.county.gov/general_results.pdf',
    'election_operator@county.gov',
    'election_2026_general_ca',
    NOW())
   RETURNING id;
   ```

7. **Log Success**

   ```json
   {
     "event": "batch_processed",
     "batch_id": 1,
     "filename": "county_results_batch_001.pdf",
     "rows_inserted": 15250,
     "anomalies": 0,
     "confidence": 0.98,
     "completion_percentage": 15,
     "timestamp": "2026-11-03T08:30:00Z"
   }
   ```

**Expected Output**:

- `results_2026110308.csv` in `/output/`
- Audit entries logged to JSONL
- War room notes updated

---

## Hourly Monitoring (08:00 AM – 09:00 PM)

**Frequency**: Every hour on the hour

**Person**: Data Lead (rotating shifts)

**Tasks**:

### Every Hour (00 minutes)

1. **Create Snapshot with Completion %**

   ```python
   loader = get_vocab_loader(session_id="election_2026_general_ca")
   
   # Query warehouse for completion estimate
   SELECT COUNT(*) FROM warehouse_election_results 
   WHERE session_id = 'election_2026_general_ca'
   AND timestamp >= CURRENT_DATE;
   
   completion_pct = (row_count / estimated_total_voters) * 100
   
   snapshot = loader.create_snapshot(
       description=f"Hourly snapshot - {completion_pct}% complete",
       completion_percentage=int(completion_pct)
   )
   print(f"Snapshot {completion_pct}%: {snapshot}")
   ```

2. **Check for Anomalies**

   ```bash
   # Query anomaly log
   grep "anomaly_reason" /logs/election_audit_2026_general_ca.jsonl | tail -20
   ```

   Expected: 0-2 anomalies per hour (normal variability)  
   Alert if: > 5 anomalies in one hour → **Possible data issue**

3. **Monitor Trust Scores**

   ```sql
   SELECT 
     source_url, 
     COUNT(*) as batch_count,
     AVG(extraction_confidence) as avg_confidence,
     MIN(extraction_confidence) as min_confidence
   FROM warehouse_election_results
   WHERE session_id = 'election_2026_general_ca'
   GROUP BY source_url
   ORDER BY avg_confidence ASC;
   ```

   Alert if: avg_confidence < 0.85 → **Investigate source**

4. **Verify Audit Log Integrity**

   ```bash
   # Check file size growth
   ls -lh /logs/election_audit_2026_general_ca.jsonl
   
   # Expected: ~50-100 KB per hour (normal parsing traffic)
   # Alert if: No growth for 30 minutes → Check parser status
   ```

5. **Post Status Update**

   ```txt
   Slack message:
   
   🗳️ Election Status Update - 11:00 AM
   ✅ Completion: 35% (127,500 / 365,000 voters)
   ✅ Data Quality: 0.96 avg confidence
   ✅ Anomalies: 1 (minor - duplicate candidate name, already resolved)
   ✅ Batches Processed: 12
   ⚠️ Last Update: 10:58 AM (on schedule)
   
   No action required. Continuing normal monitoring.
   ```

---

## Real-Time Anomaly Response

**Triggered When**: Anomaly log shows `"anomaly_reason": "X"` with `"quarantine_required": true`

### Scenario 1: Suspicious Header

**Trigger**:

```json
{
  "anomaly_reason": "suspicious_header",
  "severity": "high",
  "quarantine_required": true,
  "keywords": ["VoterSecureIDNumber", "CitizenshipStatus"]
}
```

**Response** (5 min):

1. **Screenshot the data**

   ```bash
   cp /output/results_XXXXXXX.csv /suspicious/suspicious_headers_TIMESTAMP.csv
   ```

2. **Log the incident**

   ```json
   {
     "event": "anomaly_investigation",
     "anomaly_reason": "suspicious_header",
     "investigation_start": "2026-11-03T10:15:00Z",
     "responsible_person": "data_lead@county.gov",
     "action": "reviewing_source_document"
   }
   ```

3. **Call county election office**

   ```txt
   "Hi, we received data with unusual column headers:
    'VoterSecureIDNumber', 'CitizenshipStatus'.
    Are these part of the official export?
    We're quarantining this batch for verification."
   ```

4. **Resolution**
   - **If legitimate**: Update `entities/result_terms.txt` to include new term

     ```txt
     # New headers approved by County Clerk
     VoterSecureIDNumber
     CitizenshipStatus
     ```

     Re-parse batch with updated vocab

   - **If error**: Reject batch, ask county to resubmit

---

### Scenario 2: High Vote Anomaly

**Trigger**:

```json
{
  "anomaly_reason": "vote_count_exceeds_registered_voters",
  "severity": "critical",
  "quarantine_required": true,
  "votes_count": 987654,
  "registered_voters": 654321
}
```

**Response** (Immediate):

1. **Stop processing this batch**

   ```bash
   # Mark batch as quarantined in warehouse
   UPDATE warehouse_election_results
   SET quarantine_reason = 'vote_count_exceeds_voters'
   WHERE id = <batch_id>;
   ```

2. **Notify coordinator immediately** (phone call, not email)

   ```txt
   "CRITICAL: Batch contains 987k votes but county only has 654k voters.
    Possible data corruption or wrong file sent.
    Batch quarantined. Standing by for your instruction."
   ```

3. **Gather evidence**
   - Screenshot vote counts
   - Take hash of original file
   - Save full file to evidence folder

   ```bash
   sha256sum /uploads/county_results_batch_XXX.pdf > \
     /evidence/hash_batch_XXX.txt
   cp /uploads/county_results_batch_XXX.pdf \
     /evidence/quarantined_batch_XXX.pdf
   ```

4. **Call county election office**

   ```txt
   "We cannot process your latest batch. 
    The vote counts are impossibly high (987k votes, but only 654k voters).
    Please verify and resubmit."
   ```

5. **Resolution**
   - **If corrected file sent**: Delete quarantined version, re-parse
   - **If data corruption**: Work with county IT to investigate source

---

### Scenario 3: Unknown Office Names

**Trigger**:

```json
{
  "anomaly_reason": "unrecognized_entity",
  "severity": "medium",
  "entity_type": "office",
  "unknown_value": "County Assessor-Recorder",
  "quarantine_required": false
}
```

**Response** (30 min):

1. **Check if legitimate office**
   - Search state election authority website
   - Verify with county official

2. **If legitimate**: Add to vocab

   ```bash
   # Append to entities/offices.txt
   echo "County Assessor-Recorder" >> \
     webapp/parser/Context_Integration/vocab/entities/offices.txt
   
   # During election mode, this requires elevation:
   # Principal must have "vocab_editor" role
   # Audit log will show: added_by, timestamp, reason
   ```

3. **Re-process batch** with updated vocab

4. **Log action**

   ```json
   {
     "event": "vocab_updated",
     "entity_type": "office",
     "new_entry": "County Assessor-Recorder",
     "source": "county_official_verification",
     "principal": "data_lead@county.gov",
     "timestamp": "2026-11-03T10:45:00Z"
   }
   ```

---

## End-of-Day Procedures (09:00 PM – 10:00 PM)

### 09:00 PM – Final Snapshot & Data Verification

**Person**: Coordinator + Data Lead

**Tasks**:

1. **Create final snapshot**

   ```python
   loader = get_vocab_loader(session_id="election_2026_general_ca")
   
   final_snapshot = loader.create_snapshot(
       description="Final snapshot - 100% complete",
       completion_percentage=100
   )
   print(f"Final snapshot: {final_snapshot}")
   ```

2. **Verify row counts**

   ```sql
   SELECT 
     COUNT(*) as total_rows,
     COUNT(DISTINCT county) as counties,
     COUNT(DISTINCT contest) as contests,
     AVG(extraction_confidence) as avg_confidence
   FROM warehouse_election_results
   WHERE session_id = 'election_2026_general_ca';
   ```

3. **Check for data gaps**

   ```txt
   Expected result:
   - 365 precincts from 58 counties
   - If missing precincts: Contact county, request resubmission
   ```

4. **Lock database** (no more writes)

   ```bash
   # Snapshot taken = data locked
   # Election mode remains active
   ```

---

### 10:00 PM – Disable Election Mode & Archive

**Person**: IT Support Lead

**Procedure**:

1. **Disable Election Mode** (allows post-election analysis updates)

   ```python
   loader = get_vocab_loader(session_id="election_2026_general_ca")
   loader.disable_election_mode()  # Allows write again
   ```

2. **Archive All Files**

   ```bash
   # Create election archive
   mkdir -p /archive/election_2026_general_ca
   
   # Copy results, logs, snapshots
   cp -r /output/* /archive/election_2026_general_ca/
   cp /logs/election_audit_2026_general_ca.jsonl \
      /archive/election_2026_general_ca/
   cp -r /snapshots/snapshot_*.json \
      /archive/election_2026_general_ca/snapshots/
   
   # Create manifest
   cat > /archive/election_2026_general_ca/MANIFEST.txt << EOF
   Election: General 2026
   State: California
   Date: 2026-11-03
   Archive Created: $(date -Iseconds)
   
   Contents:
   - results_*.csv (parsed election results)
   - election_audit_2026_general_ca.jsonl (audit trail)
   - snapshots/ (hourly vocab snapshots)
   - MANIFEST.txt (this file)
   
   Retention: 7 years (per election code)
   EOF
   ```

3. **Verify Archive Integrity**

   ```bash
   # Create checksums
   cd /archive/election_2026_general_ca
   find . -type f -exec sha256sum {} \; > CHECKSUMS.txt
   
   # Verify
   sha256sum -c CHECKSUMS.txt
   ```

4. **End-of-Day Report**

   ```template
   To: elections_director@county.gov
   Subject: Election 2026 General - Final Results Processed
   
   Results Summary:
   - Total voters counted: 365,000
   - Precincts reporting: 365 / 365 (100%)
   - Data quality average: 96.3%
   - Processing time: 15 hours
   - Anomalies detected: 3 (all resolved)
   - Final status: COMPLETE ✓
   
   All results archived and locked.
   Audit trail preserved for 7 years per state law.
   
   Questions? Contact elections_data_team@county.gov
   ```

---

## Post-Election Analysis (Days 2-5)

### Day 2 – Data Quality Review

**Person**: Data Lead + Election Official

**Tasks**:

1. **Query anomaly patterns**

   ```sql
   SELECT 
     anomaly_reason,
     COUNT(*) as count,
     AVG(severity) as avg_severity
   FROM election_anomalies
   WHERE session_id = 'election_2026_general_ca'
   GROUP BY anomaly_reason
   ORDER BY count DESC;
   ```

   Expected: Most anomalies are minor (case variations, alias mismatches)

2. **Identify systemic issues**

   ```txt
   - If 50+ anomalies of same type: May indicate format issue
     → Contact county IT, may need to adjust parser logic
   
   - If anomalies from single source: May indicate single county's 
     data entry error
     → Contact specific county for correction
   ```

3. **Document lessons learned**

   ```markdown
   # Lessons Learned - Election 2026 General
   
   ## What Went Well
   - All sources verified correctly (0 security incidents)
   - Hourly snapshots helped with completion tracking
   - Election mode prevented accidental data modification
   
   ## Issues Encountered
   1. County submitted results with extra column "Internal_ID"
      → Added to vocab post-election
      → Action: Pre-test with county before next election
   
   2. One precinct reported vote total >100% (105%)
      → Identified immediately by anomaly detection
      → County corrected within 1 hour
      → Outcome: GOOD (system caught error)
   
   ## Improvements for Next Election
   - Pre-flight checklist: Have county do test upload 1 week before
   - More frequent snapshots during peak hours (every 30 min vs hourly)
   - Train county data entry staff on expected format
   ```

---

## Troubleshooting Guide

### Problem: Parser Stops Processing (No New Batches)

**Diagnosis** (2 min):

```bash
# Check if parser process still running
ps aux | grep Smart_Elections_Parser_Webapp

# Check logs for errors
tail -100 /logs/election_audit_2026_general_ca.jsonl | grep -i error

# Check database connection
psql -h ${POSTGRES_HOST} -d ${POSTGRES_DB} -c "SELECT 1;"
```

**Common Causes & Fixes**:

1. **Network disconnection**: Restart parser after network restored
2. **Database connection lost**:

   ```bash
   # Reconnect
   export POSTGRES_PASSWORD=$(aws secretsmanager get-secret-value --secret-id postgres-password --query SecretString --output text)
   systemctl restart smart-elections-parser
   ```

3. **Disk full**:

   ```bash
   df -h /data  # Check disk usage
   # Archive old results if needed
   ```

---

### Problem: Anomalies Spike Suddenly

**Diagnosis** (1 min):

```bash
# Check for pattern
grep "anomaly_reason" /logs/election_audit_2026_general_ca.jsonl | tail -50
```

**Actions**:

1. **If all same type of anomaly**: Probable data format change from one county
   - Call county: "Are you sending data in different format?"

2. **If anomalies from multiple counties**: Possible parser bug
   - Pause processing
   - Review parser logs for exception traces
   - Contact IT support

---

### Problem: Trust Score Drops Below Threshold

**Diagnosis**:

```sql
SELECT 
  source_url, 
  extraction_confidence,
  timestamp
FROM warehouse_election_results
WHERE extraction_confidence < 0.85
ORDER BY timestamp DESC;
```

**Actions**:

1. Verify source URL is still valid
2. Check if source started sending data in different format
3. If issue persists: Escalate to Election Official for source investigation

---

## Emergency Contacts

| Role | Name | Phone | Email |
| ------ | ------ | ------- | ------- |
| Coordinator | [Name] | [X-XXX-XXXX] | <coordinator@county.gov> |
| Data Lead | [Name] | [X-XXX-XXXX] | <data_lead@county.gov> |
| IT Support | [Name] | [X-XXX-XXXX] | <itsupport@county.gov> |
| Election Director | [Name] | [X-XXX-XXXX] | <director@county.gov> |
| Vendor Support | Smart Elections | 1-800-ELECT-NOW | <support@smartelections.com> |

---

## Checkpoints Summary

✅ **Pre-Election (Day Before)**

- [ ] All systems tested
- [ ] Data sources verified
- [ ] Contact list confirmed
- [ ] Canary data successfully parsed

✅ **Election Morning (06:00 AM)**

- [ ] Database ready
- [ ] Election mode enabled
- [ ] Parser service running
- [ ] War room operational
- [ ] Initial snapshot created

✅ **Hourly Monitoring (08:00 AM – 09:00 PM)**

- [ ] Completion % tracked
- [ ] Anomalies monitored
- [ ] Trust scores verified
- [ ] Audit log growing
- [ ] Status updates posted

✅ **End of Day (09:00 PM)**

- [ ] Final snapshot created
- [ ] All data archived
- [ ] Checksums verified
- [ ] Election mode disabled
- [ ] Final report sent

✅ **Post-Election (Days 2-5)**

- [ ] Data quality reviewed
- [ ] Anomalies analyzed
- [ ] Lessons learned documented
- [ ] Improvements planned

---

**Owner**: Election Operations Team  
**Last Updated**: 2026-02-03  
**Version**: 1.0  
**Status**: READY FOR ELECTION DAY
