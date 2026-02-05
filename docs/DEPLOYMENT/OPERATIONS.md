---
layout: default
title: Operations & Runbook
---

## Operations & Runbook

Operational procedures, runbooks, monitoring guidelines, and troubleshooting for running the Smart Elections Parser in production.

> **Note**: This document consolidates content from:
>
> - [ELECTION_OPERATIONS_PLAYBOOK.md](../ELECTION_OPERATIONS_PLAYBOOK.md) - Election ops procedures
> - [troubleshooting.md](../troubleshooting.md) - Troubleshooting guide
> - [INTEGRITY_MONITORING.md](../INTEGRITY_MONITORING.md) - Data integrity monitoring
> - [WAREHOUSE_VERIFICATION_GUIDE.md](../WAREHOUSE_VERIFICATION_GUIDE.md) - QA procedures
>
> For complete details, consult the individual source documents.

## 📋 Daily Operations

### Morning Checklist (Start of Election Day)

```checklist
□ Verify system connectivity and uptime
□ Check disk space (minimum 10GB recommended)
□ Monitor error logs from previous day
□ Verify database connections
□ Test parsing with sample documents
□ Confirm QA panel authentication is working
□ Check certificate(s) expiry (alert if < 30 days)
□ Verify backup systems operational
□ Confirm team availability for issues
□ Brief team on known issues/workarounds
```

### Hourly Checks (During Election)

```checklist
□ Monitor system resource usage (CPU, memory, disk)
□ Check for parsing errors in logs
□ Verify result files are being created
□ Monitor API response times (< 2 seconds target)
□ Spot-check parsed data for obvious errors
□ Monitor upload directory growth
□ Check application error logs for patterns
```

### Evening Checklist (End of Day)

```checklist
□ Archive processed files
□ Review all errors encountered
□ Document workarounds applied
□ Verify all data backed up
□ Check logs for anomalies
□ Update status report for next shift
□ Confirm no pending notifications
```

## 🚨 Incident Response

### Critical Issues Flowchart

```tree
Issue Detected
    ↓
[STEP 1: ASSESS]
├─ What is affected? (Core parsing, QA panel, database?)
├─ How many users/results affected?
├─ Is it production or development?
└─ Severity: Critical / High / Medium / Low
    ↓
[STEP 2: COMMUNICATE]
├─ Notify team via #incidents Slack channel
├─ Update status page (if public)
└─ Provide initial ETA for resolution
    ↓
[STEP 3: MITIGATE]
├─ Apply immediate workaround (if available)
├─ Route traffic away from problematic system
└─ Preserve logs/diagnostics for post-incident analysis
    ↓
[STEP 4: ROOT CAUSE]
├─ Review logs and error messages
├─ Check recent configuration/code changes
├─ Monitor systems for patterns
└─ Consult runbooks for known issues
    ↓
[STEP 5: FIX & VERIFY]
├─ Implement permanent fix
├─ Test in staging environment
├─ Deploy to production
└─ Verify issue is resolved
    ↓
[STEP 6: POST-INCIDENT]
├─ Document incident and resolution
├─ Schedule post-mortem if severe
├─ Update runbooks with learnings
└─ Communicate final status to team
```

## 🔍 Monitoring & Alerting

### Key Metrics

| Metric | Alert Threshold | Check Frequency |
| -------- | ----------------- | ----------------- |
| CPU Usage | > 85% | Every 5 min |
| Memory Usage | > 75% | Every 5 min |
| Disk Usage | > 85% | Every 10 min |
| Error Rate | > 1% | Every minute |
| Response Time | > 5 sec | Every minute |
| Certificate Expiry | < 30 days | Daily |
| Upload Dir Size | > 500 MB | Every hour |

### Setting Up Alerts

**Azure Monitor**:

```bash
# Create alert for high CPU
az monitor metrics alert create \
  --name parser-high-cpu \
  --resource-group smart-elections-parser \
  --scopes /subscriptions/$SUBSCRIPTION/resourceGroups/smart-elections-parser \
  --condition "avg Percentage CPU > 85" \
  --window-size 5m \
  --evaluation-frequency 1m
```

### Interpreting Common Alerts

***High CPU (> 85%)***

- Normal during heavy parsing (multiple concurrent jobs)
- Check: Are jobs queuing up?
- Action: Consider horizontal scaling or rate limiting

***High Memory (> 75%)***

- Check: Large file sizes being parsed?
- Action: Check for memory leaks in logs
- Solution: Restart application if sustained

***Disk Full (> 85%)***

- Check: Upload/output directories
- Action: Archive old results or increase disk
- Prevent: Set auto-cleanup policy

## 🐛 Troubleshooting Common Issues

### Issue 1: Parsing Fails with "No Data Found"

**Diagnosis**:

```bash
# Check application logs
tail -50 /var/log/parser/application.log | grep -i "no data"

# Check file size (PDFs sometimes report size but have corruption)
ls -lah ./uploads/*.pdf
```

**Solutions** (in order):

1. Verify file integrity: `file uploaded.pdf`
2. Try alternative parsing method (HTML vs PDF vs CSV)
3. Check if file is encrypted or requires password
4. Run OCR if document is image-based PDF
5. Try manual validation: open file locally

### Issue 2: Certificate Authentication Fails (401)

**Diagnosis**:

```bash
# Check if certificate header is being forwarded
curl -v https://your-app.azurewebsites.net/qa/health 2>&1 | grep "X-ARR"

# Verify certificate details
openssl x509 -in client-cert.pem -text -noout
```

**Solutions**:

1. Verify certificate not expired: `openssl x509 -noout -dates -in cert.pem`
2. Check Azure App Service has "Client certificate" enabled
3. Restart app after configuration changes
4. Try development mode: `QA_REQUIRE_CERT_AUTH=false`
5. Check browser has certificate imported

### Issue 3: Memory Leak/Slow Performance Over Time

**Diagnosis**:

```bash
# Monitor memory growth
watch -n 5 'ps aux | grep python'

# Check for accumulating processes
ps aux | grep python | wc -l
```

**Solutions**:

1. Identify memory leak source: `python -m memory_profiler script.py`
2. Check for unclosed file handles: `lsof -p <pid>`
3. Restart application (temporary): `kill -9 <pid>`
4. Review logs for accumulating errors
5. Check database connections (not closed properly)

### Issue 4: Parsing Errors Spike

**Diagnosis**:

```bash
# Count errors in last hour
grep "ERROR" /var/log/parser/application.log | \
  grep "$(date -d '1 hour ago' +%Y-%m-%d)" | wc -l

# Find most common errors
grep "ERROR" /var/log/parser/application.log | \
  cut -d: -f5- | sort | uniq -c | sort -rn | head -10
```

**Solutions**:

1. Check for configuration issues
2. Verify database connectivity
3. Check for rate limiting (exceeding API quotas?)
4. Review recent code/config changes
5. Check if election data source changed format

## 📊 Data Integrity Monitoring

### Validation Checks

Run automated checks:

```bash
# Validate all parsed data
python scripts/verify_all_parsers.py

# Check data consistency
python health/integrity_check.py --format summary

# Generate report
python health/generate_data_report.py --output report.html
```

### Common Data Issues

| Issue | Cause | Fix |
| -------- | ----------------- | ----------------- |
| Vote totals don't match | Incomplete extraction | Re-extract with different strategy |
| Duplicate candidates | Name normalization failed | Review normalization rules |
| Missing races | Content parsing missed sections | Check source document structure |
| Percentage errors | Rounding or incomplete data | Flag for manual review |
| Inconsistent party values | Different source formats | Apply mapping/normalization |

### Integrity Report Generation

```bash
# Generate daily integrity report
python health/generate_data_report.py \
  --start-date 2024-01-01 \
  --end-date 2024-01-02 \
  --output integrity_report_2024-01-02.html

# Email report
mail -s "Daily Integrity Report" team@example.com < \
  integrity_report_2024-01-02.html
```

## 🔄 Backup & Recovery

### Backup Procedures

```bash
# Backup database
python -c "from webapp import db; db.create_backup('backup_$(date +%s).db')"

# Backup parsed results
tar -czf results_backup_$(date +%Y%m%d).tar.gz ./output/

# Archive to cloud storage
az storage blob upload-batch \
  --source ./output/ \
  --destination results \
  --account-name storageaccount
```

### Recovery Procedures

```bash
# Restore database
python -c "from webapp import db; db.restore_backup('backup_timestamp.db')"

# Restore from cloud
az storage blob download-batch \
  --source results \
  --destination ./output/ \
  --account-name storageaccount
```

## 📝 Log Analysis

### Finding Issues in Logs

```bash
# All errors
grep -i "error" application.log

# Errors in specific module
grep -i "error" application.log | grep "html_scanner"

# Errors in last N hours
grep -i "error" application.log | grep "$(date -d '3 hours ago' +%H)"

# Errors with context (10 lines after)
grep -A 10 -i "error" application.log | head -50
```

### Producing Diagnostic Bundle

```bash
# Create diagnostics archive
mkdir diagnostic_bundle
cp /var/log/parser/application.log diagnostic_bundle/
cp /var/log/parser/error.log diagnostic_bundle/
ps aux > diagnostic_bundle/processes.txt
df -h > diagnostic_bundle/disk_usage.txt
free -h > diagnostic_bundle/memory_usage.txt
netstat -an > diagnostic_bundle/network.txt

tar -czf diagnostics_$(date +%s).tar.gz diagnostic_bundle
```

## 🧹 Maintenance Tasks

### Weekly Tasks

```bash
# Clean up old temporary files
find ./uploads -type f -mtime +7 -delete

# Rotate logs
logrotate -f /etc/logrotate.d/parser

# Update threat intelligence feeds
python scripts/update_threat_feeds.py

# Verify all tests still pass
python -m pytest webapp/tests/ -q
```

### Monthly Tasks

```bash
# Full system backup
tar -czf full_backup_$(date +%Y%m%d).tar.gz \
  ./webapp ./output ./uploads

# Database maintenance
python -c "from webapp import db; db.optimize()"

# Certificate renewal check
openssl x509 -noout -dates -in cert.pem | grep notAfter

# Dependency security scan
pip audit
```

### Quarterly Tasks

```bash
# Review and update runbooks
# Review access logs for unauthorized attempts
# Test disaster recovery procedures
# Perform penetration testing (if applicable)
# Review third-party dependencies for updates
```

## 📞 Escalation Procedures

### Level 1: Self-Service (User)

- Check FAQ and documentation
- Check status page for known issues
- Attempt workarounds (clear cache, retry)

### Level 2: Support Team

- Contact support via email/chat
- Provide error messages and steps to reproduce
- Wait for investigation (typical: 1-2 hours)

### Level 3: Engineering Team

- Critical issues (system down, data loss)
- Escalation requires: impact assessment + incident description
- Response time: 30 minutes for critical

### Level 4: Vendor/External

- Azure support for infrastructure issues
- Browser vendor for compatibility issues
- Third-party library maintainers for upstream issues

## ✅ Operational Checklist

**Start of Shift**:

- [ ] Review incident log from previous shift
- [ ] Check system health dashboard
- [ ] Verify backup systems
- [ ] Confirm team availability

**During Shift**:

- [ ] Monitor dashboards hourly
- [ ] Review error logs periodically
- [ ] Respond to incidents promptly
- [ ] Document any issues encountered

**End of Shift**:

- [ ] Document handoff notes
- [ ] Archive session logs
- [ ] Alert next shift of any ongoing issues
- [ ] Verify backup completed

---

**Related Documents**:

- [Deployment Guide](./DEPLOYMENT.md) - Deployment procedures
- [Security & Authentication](./SECURITY.md) - Security operations
- [Election Operations Playbook](../ELECTION_OPERATIONS_PLAYBOOK.md) - Detailed election procedures

**Sources**:

- [ELECTION_OPERATIONS_PLAYBOOK.md](../ELECTION_OPERATIONS_PLAYBOOK.md)
- [troubleshooting.md](../troubleshooting.md)
- [INTEGRITY_MONITORING.md](../INTEGRITY_MONITORING.md)

**Last Updated**: Consolidated operations runbook
