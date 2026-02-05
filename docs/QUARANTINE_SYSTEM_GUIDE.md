# Quarantine + Trust Transparency System - Complete Guide

**Status**: ✅ PRODUCTION READY  
**Last Updated**: Current Session

> Comprehensive guide for the Smart Elections Parser transparent quarantine review pipeline with full audit logging and data collection explanations.

---

## 🚀 Quick Start (5-10 minutes)

### Version 1: Just Deploy It

**→ [Move to DEPLOYMENT_GUIDE.md section below](#deployment)**

### Version 2: Understand First, Then Deploy

1. **What is the Quarantine System?** → See [Overview](#overview)
2. **How did we build it?** → See [Architecture](#architecture)
3. **How do I actually deploy it?** → See [Deployment](#deployment)
4. **API Reference?** → See [Quick Reference](#quick-reference)

---

## Overview

The **Quarantine + Trust Transparency Pipeline** is a transparent system that:

✅ **Quarantines suspicious election URLs** based on trust scoring  
✅ **Provides data collection transparency** - explains what data is collected and why  
✅ **Creates complete audit trails** - all review decisions are logged and certified  
✅ **Enables manual review** - reviewers can inspect, annotate, and approve data  
✅ **Integrates with ML/NLP** - suspect items feed back into model retraining

### Key Principles

| Principle | Implementation |
| ----------- | ---------------- |
| **Transparency** | Every quarantine action includes explicit reason + human description |
| **Auditability** | Full audit trail: who reviewed, when, decision, certification reason |
| **Data Minimalism** | Only collect data we need; explain why collecting each field |
| **User Agency** | Reviewers make final decisions; system provides scoring context |
| **Consistency** | Standardized reason codes + messaging across UI and API |

---

## Architecture

### System Components

#### 1. **Quarantine Core** (`webapp/parser/quarantine_endpoints.py`)

```python
class QuarantineReason(enum.Enum):
    """Enumeration of standardized quarantine reasons"""
    LOW_TRUST_SCORE        # URL scored low on security assessment
    DOMAIN_FLAGGED         # Domain/registrar flagged by external service
    SSL_ANOMALY            # SSL certificate issues detected
    CONTENT_ANOMALY        # Page content differs from expected pattern
    PARSING_FAILURE        # Parser failed on this URL (before quarantine)
```

Each reason includes a human-readable description explaining:

- What triggered this reason
- What data we collected
- Why we collected it
- What happens next

#### 2. **Transparent Data Collection** (`DataCollectionNotice`)

Example: A URL with `LOW_TRUST_SCORE` triggers collection of:

```json
{
  "field": "domain_whois",
  "collected": true,
  "reason": "Registrar and registration age help assess domain legitimacy",
  "retention": "7 days",
  "legal_basis": "User consent (displayed before parsing)"
}
```

#### 3. **Quarantine Entry** (Complete Record)

```json
{
  "id": "q_1704067200000",
  "url": "https://example.org/results",
  "reason": "LOW_TRUST_SCORE",
  "reason_description": "URL scored 0.32/1.0 on trust assessment...",
  "session_id": "sess_xxx",
  "principal": "user@org.gov",
  "timestamp": "2024-01-01T12:00:00Z",
  "data_snapshot": {
    "domain": "example.org",
    "ssl_grade": "B",
    "whois_age_days": 45,
    "parsed_content_confidence": 0.78
  },
  "data_collection_notice": [
    {
      "field": "domain_whois",
      "collected": true,
      "reason": "Registrar and registration age help assess domain legitimacy",
      "legal_basis": "User consent"
    }
  ],
  "review_decision": {
    "status": "PENDING",
    "reviewer_principal": null,
    "decision": null,
    "certification_reason": null,
    "reviewed_at": null
  }
}
```

#### 4. **API Endpoints**

All endpoints require authenticated reviewer principal (client cert or SSO):

| Endpoint | Method | Purpose |
| ----------- | -------- | --------- |
| `/quarantine/review` | GET | Interactive web UI for quarantine review |
| `/api/quarantine/pending` | GET | List pending quarantine items (JSON) |
| `/api/quarantine/reviewed` | GET | View historical review decisions |
| `/api/quarantine/item/<id>` | GET | Get specific quarantine entry details |
| `/api/quarantine/review` | POST | Record review decision with certification |
| `/api/quarantine/stats` | GET | Queue statistics and metrics |

#### 5. **Review Workflow**

```txt
URL Parsed
    ↓
Trust Score Calculated (0.0-1.0)
    ↓
Score < Threshold? (default: 0.3)
    ↓ YES
Quarantine Entry Created + Persisted
    ↓
Reviewer Notified → Interactive Review UI
    ↓
Reviewer Decision
    ├─ APPROVE      → Data enters analytics pipeline
    ├─ REJECT       → Data discarded, logged
    └─ ARCHIVE      → Held for manual inspection later
    ↓
Review Logged + Certified
    ↓
Data Feedback → ML Retraining Job
```

---

## Deployment

### Prerequisites

- Python 3.11+
- Flask + SocketIO (already in webapp)
- PostgreSQL with SQLAlchemy models (already configured)
- Client certificate auth or SSO integration

### Installation Steps

1. **Ensure dependencies are installed**

   ```bash
   pip install -r requirements.txt
   # Already includes Flask, SQLAlchemy, orjson
   ```

2. **Initialize database** (if not already done)

   ```bash
   python -c "from webapp.parser.utils.models import Base, engine; Base.metadata.create_all(engine)"
   ```

3. **Create quarantine queue persistence directory**

   ```bash
   mkdir -p webapp/parser/Context_Integration/Context_Library/log/quarantine
   ```

4. **Verify endpoints are registered**
   - Check `webapp/Smart_Elections_Parser_Webapp.py` for quarantine route registration
   - Ensure `/quarantine/review` and `/api/quarantine/*` routes are active

### Testing

| Test Case | Command | Expected | Status |
| ----------- | -------- | --------- | ------- |
| Quarantine pending items | `curl http://localhost:5000/api/quarantine/pending` | JSON list | ✅ |
| Submit review decision | `curl -X POST http://localhost:5000/api/quarantine/review -d {...}` | {"status":"recorded"} | ✅ |
| View review UI | `http://localhost:5000/quarantine/review` | Interactive form | ✅ |
| Query stats | `curl http://localhost:5000/api/quarantine/stats` | {"pending":N, "reviewed":M} | ✅ |

### Azure Deployment

See [AZURE_DEPLOYMENT_GUIDE.md](AZURE_DEPLOYMENT_GUIDE.md) for container-specific setup.

---

## Quick Reference

### Quarantine Reason Codes

| Code | Meaning | Auto-Triggered | Requires Review |
| ----------- | -------- | --------- | ------- |
| `LOW_TRUST_SCORE` | Trust score < 0.3 | Yes | Yes |
| `DOMAIN_FLAGGED` | Domain on blocklist | Yes | Yes |
| `SSL_ANOMALY` | SSL cert issues | Yes | Yes |
| `CONTENT_ANOMALY` | Content pattern mismatch | Yes | Yes |
| `PARSING_FAILURE` | Parser error mid-run | Yes | Yes |
| `MANUAL_REVIEW` | User flagged manually | No | Yes |

### Review Decision Codes

| Code | Action | Consequence |
| ----------- | -------- | --------- |
| `APPROVE` | Data is legitimate | Enter analytics pipeline + ML training |
| `REJECT` | Data is invalid | Discard + log rejection |
| `ARCHIVE` | Data needs inspection | Hold for manual audit later |

### Common Operations

**Check pending quarantine items:**

```bash
curl -H "Authorization: Bearer $TOKEN" http://localhost:5000/api/quarantine/pending | jq '.'
```

**Approve a quarantine entry:**

```bash
curl -X POST http://localhost:5000/api/quarantine/review \
  -H "Content-Type: application/json" \
  -d '{
    "entry_id": "q_1704067200000",
    "decision": "APPROVE",
    "reviewer_principal": "user@org.gov",
    "certification_reason": "Domain verified via WHOIS; trust score justified"
  }'
```

**Get quarantine statistics:**

```bash
curl -H "Authorization: Bearer $TOKEN" http://localhost:5000/api/quarantine/stats | jq '.'
```

---

## File Locations

| Component | Path |
| ----------- | ------ |
| Core endpoints | `webapp/parser/quarantine_endpoints.py` |
| Data models | `webapp/parser/utils/models.py` (Alert, QuarantineEntry) |
| UI templates | `webapp/templates/quarantine_review.html` |
| API integration | `webapp/Smart_Elections_Parser_Webapp.py` |
| Persistence | `webapp/parser/Context_Integration/Context_Library/log/quarantine/` |

---

## Running the Web UI

**Start the Flask app:**

```bash
python -m webapp.Smart_Elections_Parser_Webapp
```

**Access the quarantine review dashboard:**

```txt
http://localhost:5000/quarantine/review
```

### Example Workflow

1. **Open quarantine review UI** → Shows pending items with trust scores
2. **Inspect entry** → Click to expand reason, data collected, confidence scores
3. **Review data snapshot** → See what was extracted from the URL
4. **Make decision** → APPROVE / REJECT / ARCHIVE with certification
5. **Submit** → Decision recorded with timestamp, reviewer ID, certification
6. **See history** → Historical decisions available for audit

---

## Integration with ML Retraining

Every approved/rejected quarantine decision feeds back into:

1. **Trust Score Model** - Improves domain/URL classification
2. **Pattern Recognition** - Refines content anomaly detection
3. **NER/Entity Models** - Better entity extraction from election pages
4. **Confidence Calibration** - Tune confidence thresholds based on reviewer verdicts

Retraining job runs daily/weekly depending on volume. See `health/retrain_*.py` for details.

---

## Troubleshooting

### Issue: Quarantine entries not persisting

**Solution**: Check `log/quarantine/` directory exists and is writable

```bash
ls -la webapp/parser/Context_Integration/Context_Library/log/quarantine/
```

### Issue: Review UI not loading

**Solution**: Verify Flask app is running and routes are registered

```bash
curl http://localhost:5000/quarantine/review -I
```

### Issue: API returns 401 Unauthorized

**Solution**: Ensure client certificate is provided or SSO token is valid

```bash
curl -cert client.crt -key client.key http://localhost:5000/api/quarantine/pending
```

### Issue: Reviewer decisions not being recorded

**Solution**: Verify PostgreSQL connection and Alert table exists

```bash
python -c "from webapp.parser.utils.db_utils import get_session; s = get_session(); print(s.query(Alert).count())"
```

---

## For More Information

- **Architecture Deep Dive**: See docs/architecture.md
- **System Governance**: See docs/SYSTEM_GOVERNANCE.md
- **Data Privacy**: See docs/ELECTION_OPERATIONS_PLAYBOOK.md
- **Deployment**: See docs/DEPLOYMENT_GUIDE.md

---

**Questions?** Check the Quick Reference section or review the File Locations table to find the relevant code.
