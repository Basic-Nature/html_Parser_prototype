# Quick References Index

Master reference guide for Smart Elections Parser - quick lookup for APIs, frameworks, and common operations.

---

## 📋 Quick Navigation

### Certificate Authentication (Cert Auth)

- **Status**: Phase 5 in progress (57% complete through Step 4)
- **Key Components**: Client certificate validation, caching, change detection
- **Implementation**: `webapp/parser/utils/cert_auth.py`
- **Key Files**:
  - Cache mechanism: certs cached locally with change detection
  - SSO fallback: certificate fails → SSO gateway redirect
  - Privilege tiers: BASIC, ELEVATED, ADMIN, SYSTEM

| Operation | Command/Code | Purpose |
| ----------- | -------------- | --------- |
| Generate test cert | `openssl req -x509 -newkey rsa:2048 -keyout cert.key -out cert.crt` | Testing |
| Load cert cache | `CertAuthManager.load_cache()` | Initialize from disk |
| Validate certificate | `cert_auth.validate_client_cert(cert_data)` | Verify cert present + valid |
| Check expiration | `cert_auth.is_cert_expired(cert)` | Determine if renewal needed |
| Detect change | `CertAuthManager.detect_change()` | Invalidate cache if cert changed |

---

### Confidence/Caution Framework

- **Status**: Phase A complete (100% - confidence/caution entity validation)
- **Key Components**: Entity confidence scoring, anomaly detection, decision codes
- **Implementation**: `webapp/parser/health/confidence_scoring.py`

#### Decision Codes

| Code | Meaning | Action |
 | ----------- | -------- | --------- |
| `APPROVE` | High confidence data | Enter analytics |
| `CAUTION` | Medium confidence | Flag for review |
| `QUARANTINE` | Low confidence | Hold for manual review |
| `ANOMALY_DETECTED` | Suspicious pattern | Alert + log |

#### Confidence Scoring

```txt
Base Score: 0.5
+ URL trust: +0.2 (high-quality domain)
+ Pattern match: +0.15 (matches known election data format)
+ Content confidence: +0.1 (parser confidence)
- Anomaly flags: -0.15 (suspicious patterns detected)
- SSL issues: -0.1 (certificate problems)
= Final Score: 0.0 - 1.0 (0=reject, 1=approve)
```

#### Signal Types

| Signal | Weight | Example |
| ----------- | -------- | --------- |
| `VALID_ELECTION_DOMAIN` | +0.2 | Registered `.gov` or secretary of state site |
| `UNUSUAL_DOMAIN_AGE` | -0.15 | Domain registered < 30 days old |
| `MULTIPLE_PARSING_ERRORS` | -0.25 | Parser failed 3+ times on this URL |
| `CONFLICTING_DATA` | -0.3 | Results contradict previous data for same contest |

---

### Quarantine System

- **Status**: Production ready
- **Key Components**: Quarantine queue, review UI, audit trails
- **Implementation**: `webapp/parser/quarantine_endpoints.py`

#### Quarantine Reason Codes

| Code | Trigger | Review Required |
| ----------- | -------- | --------- |
| `LOW_TRUST_SCORE` | Score < 0.3 | Yes |
| `DOMAIN_FLAGGED` | Domain on blocklist | Yes |
| `SSL_ANOMALY` | Certificate issues | Yes |
| `CONTENT_ANOMALY` | Pattern mismatch | Yes |
| `PARSING_FAILURE` | Parser error | Yes |

#### Quarantine API Endpoints

```bash
# List pending quarantine items
GET /api/quarantine/pending

# Get specific item
GET /api/quarantine/item/<id>

# Submit review decision
POST /api/quarantine/review
{
  "entry_id": "q_123",
  "decision": "APPROVE|REJECT|ARCHIVE",
  "reviewer_principal": "user@org.gov",
  "certification_reason": "..."
}

# Get statistics
GET /api/quarantine/stats
```

---

### UI Implementation

- **Status**: Complete - Modal banner + heartbeat filtering
- **Key Components**: CSS containment, heartbeat filtering, modal overlay
- **Implementation**: `static/js/run_parser.js`, `static/css/run_parser.css`

#### CSS Key Classes

| Class | Purpose | Location |
| ----------- | -------- | --------- |
| `.modal-overlay` | Overlay + container | `run_parser.css` |
| `.modal-content` | Inner box | `run_parser.css` |
| `.heartbeat` | Animated pulse | `run_parser.css` |
| `.banner-notification` | Status messages | `run_parser.css` |

#### JavaScript Key Functions

| Function | Purpose | Location |
| ----------- | -------- | --------- |
| `openModalBanner()` | Display modal overlay | `run_parser.js` |
| `closeModalBanner()` | Hide modal overlay | `run_parser.js` |
| `filterHeartbeatEvents()` | Skip noisy heartbeat logs | `run_parser.js` |
| `showNotification(msg, type)` | Show toast-style message | `run_parser.js` |

#### Heartbeat Filtering

```javascript
// Filter out verbose heartbeat logs during parsing
const isHeartbeat = (msg) => msg.includes('heartbeat') || msg.includes('ping');
if (!isHeartbeat(logMessage)) {
  displayLog(logMessage);
}
```

---

### Deployment & Azure

- **Status**: Active - Production deployments
- **Key Components**: Container setup, environment variables, health checks
- **Platforms**: Azure Web App (Linux), GitHub Actions CI/CD

#### Environment Variables

| Variable | Default | Purpose |
| ----------- | -------- | --------- |
| `POSTGRES_URL` | (local dev) | PostgreSQL connection string |
| `POPPLER_PATH` | (auto-detect) | PDF to image converter |
| `LOG_LEVEL` | INFO | Logging verbosity |
| `TIMEOUT_SECONDS` | 300 | Parser timeout |

#### Azure Deployment Commands

```bash
# Build container
docker build -t parser:latest .

# Run locally
docker run -p 5000:5000 -e POSTGRES_URL="..." parser:latest

# Deploy to Azure
az webapp up --name parser-app --resource-group my-group --runtime "PYTHON:3.12"
```

#### Health Check Endpoints

| Endpoint | Status | Response |
| ----------- | -------- | --------- |
| `GET /health` | ✅ OK | `{"status":"healthy"}` |
| `GET /health/db` | 🔴 Error | `{"status":"error","reason":"DB down"}` |
| `GET /azure_health` | ✅ Dashboard | HTML operations console |

---

### Testing & Verification

- **Status**: MyPy type checking complete, test coverage improving
- **Tools**: pytest, MyPy, markdownlint
- **Commands**:

```bash
# Run type checking
mypy webapp/ --ignore-missing-imports

# Run unit tests
pytest webapp/tests/ -v

# Run linting
pylint webapp/parser/ --load-plugins pylint_django

# Check markdown
npx markdownlint docs/**/*.md
```

---

### Data Architecture

- **Status**: Fully implemented - fixture pipeline
- **Key Components**: Fixtures (JSON), cache (short-term), log (append-only JSONL), PostgreSQL (warehouse)

#### Source Priority (Index Builder)

1. **CSVs** (local-only, if present)
2. **Fixtures JSON/JSONL** (committed to git)
3. **Cache JSON** (with `--include-cache`)
4. **Log JSONL** (with `--include-log`)

#### Confidence Filtering

```bash
# Default (permissive)
python scripts/build_election_index.py --min-confidence 0.0

# Production (strict)
python scripts/build_election_index.py --min-confidence 0.7

# With all sources
python scripts/build_election_index.py --include-cache --include-log --min-confidence 0.7
```

#### Migration to PostgreSQL

```bash
# Dry run
python scripts/migrate_fixtures_to_warehouse.py --dry-run

# Production migration
python scripts/migrate_fixtures_to_warehouse.py --min-confidence 0.7 --batch-size 500
```

---

## 🔗 Cross-References

| Topic | Documentation | Code |
| ----------- | --------------- | ------- |
| Backend Architecture | docs/architecture.md | webapp/parser/ |
| System Governance | docs/SYSTEM_GOVERNANCE.md | webapp/parser/utils/privilege_tiers.py |
| Verification Pipeline | docs/VERIFICATION_ARCHITECTURE.md | webapp/parser/Context_Integration/ |
| Database Models | docs/index.md (Data section) | webapp/parser/utils/models.py |
| Fixture Pipeline | docs/architecture.md (section F) | scripts/build_election_index.py |
| Deployment | docs/DEPLOYMENT_GUIDE.md | Dockerfile, requirements.txt |
| CLI Reference | docs/index.md (CLI section) | webapp/parser/html_election_parser.py |

---

## ⚡ Common Tasks

**I need to...**

- **Add a confidence signal**: Modify `confidence_scoring.py`, add signal type, update weights
- **Create a new quarantine reason**: Add to `QuarantineReason` enum, register endpoint
- **Deploy to production**: See DEPLOYMENT_GUIDE.md → Azure section
- **Review quarantine items**: Access `/quarantine/review` in web UI
- **Build election index**: `python scripts/build_election_index.py --src webapp/parser/fixtures`
- **Migrate data to PostgreSQL**: `python scripts/migrate_fixtures_to_warehouse.py --min-confidence 0.7`
- **Check database health**: `curl http://localhost:5000/health/db`
- **View operations dashboard**: Access `/azure_health` in web UI
- **Run tests**: `pytest webapp/tests/ -v`
- **Check type safety**: `mypy webapp/ --ignore-missing-imports`

---

**Last Updated**: February 5, 2026  
**Related Docs**: [QUARANTINE_SYSTEM_GUIDE.md](QUARANTINE_SYSTEM_GUIDE.md) | [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) | [architecture.md](architecture.md)
