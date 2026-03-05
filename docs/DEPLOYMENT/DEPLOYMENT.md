---
layout: default
title: Deployment Guide
---

## Deployment Guide

Complete guide for deploying the Smart Elections Parser to production environments, including local testing, cloud platforms (Azure), and general best practices.

## CI/CD Workflow Topology

For workflow ownership, trigger boundaries, and environment targets, see:

- [CI Topology](CI_TOPOLOGY.html)

> **Note**: This document consolidates content from:
>
> - [DEPLOYMENT_GUIDE.md](../DEPLOYMENT_GUIDE.md) - General deployment procedures
> - [AZURE_DEPLOYMENT_CHECKLIST.md](../AZURE_DEPLOYMENT_CHECKLIST.md) - Azure-specific checklist
> - [PHASE2_AZURE_DEPLOYMENT.md](../PHASE2_AZURE_DEPLOYMENT.md) - Phase 2 Azure details
> - [PHASE2_DEPLOYMENT_CHECKLIST.md](../PHASE2_DEPLOYMENT_CHECKLIST.md) - QA deployment checklist
>
> For Azure-specific details, see the individual source documents linked above.

## 🚀 Quick Start

### Local Development Deployment

```bash
# 1. Clone repository
git clone https://github.com/basic-nature/html_Parser_prototype.git
cd html_Parser_prototype

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.template .env
# Edit .env with your settings

# 5. Run parser (CLI mode)
python webapp/parser/html_election_parser.py --url <your_election_url>

# 6. Run parser (web UI mode)
python Smart_Elections_Parser_Webapp.py
# Access at http://localhost:5000
```

## 📋 Pre-Deployment Requirements

### System Requirements

- **Python**: 3.8–3.12
- **Memory**: Minimum 2GB (4GB+ recommended)
- **Disk**: Minimum 1GB (5GB+ recommended for caching)
- **Browser**: Playwright (included) or Selenium (optional)

### Dependencies

```bash
# Core
python >= 3.8
pip >= 21.0

# Optional (Windows only for PDF support)
Poppler (for pdf2image)
# Download from: https://github.com/oschwartz10612/poppler-windows/releases/
```

### Credentials & Configuration

- `.env` file with required keys (copy from `.env.template`)
- Optional: Azure credentials (for cloud deployment)
- Optional: Client certificate (for QA endpoints)
- Database connection string (if using external database)

## 📦 Installation Methods

### Method 1: Manual Installation (Recommended for Development)

```bash
# 1. Clone the repository
git clone https://github.com/basic-nature/html_Parser_prototype.git
cd html_Parser_prototype

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate

# 3. Install requirements
pip install -r requirements.txt

# 4. Verify installation
python -c "import webapp; print('Installation successful')"
```

### Method 2: Automated Installation Scripts

**Windows**:

```bash
install.bat
```

**Linux/macOS**:

```bash
chmod +x install.sh
./install.sh
```

**Python-based installer** (all platforms):

```bash
python install.py
```

### Method 3: Docker Deployment (If Available)

```bash
# Build image
docker build -t smart-elections-parser:latest .

# Run container
docker run -p 5000:5000 \
  -e PARSER_MODE=web \
  -v /path/to/output:/app/output \
  smart-elections-parser:latest
```

## 🔧 Configuration

### Environment Variables

Create `.env` file in project root:

```bash
# Application mode
PARSER_MODE=web              # CLI or web mode
PARSER_LOG_LEVEL=INFO        # DEBUG, INFO, WARNING, ERROR

# Security
QA_REQUIRE_CERT_AUTH=false   # true for production, false for dev
ALLOWED_DOMAINS=localhost    # Comma-separated for web UI

# Optional: Database
DATABASE_URL=sqlite:///parser.db

# Optional: Cloud deployment
AZURE_SUBSCRIPTION_ID=<subscription>
AZURE_RESOURCE_GROUP=<group>
AZURE_APP_SERVICE=<service>
```

### Environment Split Policy (GitHub vs Azure)

- Keep `.env.template` in GitHub as a schema-only template (no real secrets).
- Never commit runtime `.env` files; use GitHub Secrets and Azure App Settings for production values.
- Azure App Settings are the source of truth in production; local `.env` is for development only.
- Container builds exclude `.env`/`.env.*`/`.env.template` via `.dockerignore` to avoid shipping local config.
- CI deployment workflow fails if committed runtime `.env` files are detected.

### Flask Configuration

In `Smart_Elections_Parser_Webapp.py`:

```python
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['OUTPUT_FOLDER'] = './output'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['SESSION_TIMEOUT'] = 3600  # 1 hour
```

## 🌐 Azure Deployment

### Prerequisites

```bash
# Install Azure CLI
curl https://aka.ms/installazurecliwindows -o AzureCLI.msi
# Or: brew install azure-cli (macOS)

# Login to Azure
az login

# Create resource group
az group create \
  --name smart-elections-parser \
  --location eastus
```

### Deployment Steps

1. **Create App Service Plan**

   ```bash
   az appservice plan create \
     --name smart-elections-parser-plan \
     --resource-group smart-elections-parser \
     --sku B2 --is-linux
   ```

2. **Create Web App**

   ```bash
   az webapp create \
     --resource-group smart-elections-parser \
     --plan smart-elections-parser-plan \
     --name smart-elections-parser-app \
     --runtime "PYTHON:3.11"
   ```

3. **Configure Application Settings**

   ```bash
   az webapp config appsettings set \
     --resource-group smart-elections-parser \
     --name smart-elections-parser-app \
     --settings \
       WEBSITES_PORT=5000 \
       QA_REQUIRE_CERT_AUTH=true \
       PARSER_LOG_LEVEL=INFO
   ```

4. **Deploy Code**

   ```bash
   # Via Git
   az webapp up \
     --name smart-elections-parser-app \
     --resource-group smart-elections-parser
   
   # Or via ZIP file
   zip -r app.zip . -x "*.git*"
   az webapp deployment source config-zip \
     --resource-group smart-elections-parser \
     --name smart-elections-parser-app \
     --src-path app.zip
   ```

5. **Configure SSL/TLS**

   ```bash
   # Upload certificate
   az webapp config set \
     --resource-group smart-elections-parser \
     --name smart-elections-parser-app \
     --minimum-tls-version 1.2
   ```

6. **Enable Application Insights** (Optional)

   ```bash
   az monitor app-insights component create \
     --resource-group smart-elections-parser \
     --location eastus \
     --app smart-elections-insights
   ```

### Post-Deployment Verification

```bash
# Check app status
az webapp show \
  --resource-group smart-elections-parser \
  --name smart-elections-parser-app

# View logs
az webapp log tail \
  --resource-group smart-elections-parser \
  --name smart-elections-parser-app

# Test endpoint
curl https://smart-elections-parser-app.azurewebsites.net/health
```

## 🧪 Testing Deployment

### Health Checks

```bash
# Basic health check
curl http://localhost:5000/health

# With expected response:
# {"status": "healthy", "version": "x.y.z"}
```

### Functionality Tests

```bash
# Run automated tests
python -m pytest webapp/tests/

# Run smoke tests (quick validation)
python run_statement_test.py

# Full test suite
python automate.py --full
```

### Performance Validation

```bash
# Check response times
time curl http://localhost:5000/parse/test

# Monitor resource usage
# CPU: Should be < 80% average
# Memory: Should be < 1GB for typical operations
# Disk: Monitor upload directory growth
```

## 🔒 Security Checklist

### Before Going Live

- [ ] Set `QA_REQUIRE_CERT_AUTH=true` in production
- [ ] Configure HTTPS/TLS (minimum version 1.2)
- [ ] Set strong `SECRET_KEY` for session management
- [ ] Disable debug mode (`DEBUG=False` in Flask config)
- [ ] Set up authentication for admin endpoints
- [ ] Configure Content Security Policy (CSP) headers
- [ ] Enable HTTPS only (redirect HTTP to HTTPS)
- [ ] Review and rotate credentials in `.env`
- [ ] Set up firewall rules (if on-premises)
- [ ] Enable logging and monitoring

### Secrets Management

```bash
# ❌ DON'T: Commit secrets to git
.env
credentials.json
secrets.yaml

# ✅ DO: Use environment variables or vaults
# Azure Key Vault
# AWS Secrets Manager
# HashiCorp Vault
```

## 📊 Monitoring & Logging

### Application Logs

```bash
# View logs (Windows)
Get-Content -Path app.log -Tail 50

# View logs (Linux)
tail -50 app.log

# Search logs
grep -i "error" app.log | head -20
```

### Azure Monitoring

```bash
# Stream logs in real-time
az webapp log tail \
  --resource-group smart-elections-parser \
  --name smart-elections-parser-app

# View application events
az monitor activity-log list \
  --resource-group smart-elections-parser
```

### Key Metrics to Monitor

- **Response time**: < 2s for typical requests
- **Error rate**: < 1% (5xx errors)
- **Memory usage**: < 80% available
- **CPU usage**: Peak < 90% (monitor for sustained high usage)
- **Disk space**: Monitor upload/output folders

## 🆚 Rollback Procedures

### If Deployment Fails

**Git-based deployment**:

```bash
# Revert to previous commit
git revert <commit-hash>
git push azure main
```

**Azure App Service**:

```bash
# Swap deployment slots
az webapp deployment slot swap \
  --resource-group smart-elections-parser \
  --name smart-elections-parser-app \
  --slot staging
```

**Manual rollback**:

```bash
# Stop app
az webapp stop \
  --resource-group smart-elections-parser \
  --name smart-elections-parser-app

# Deploy previous version
# ... redeploy steps ...

# Start app
az webapp start \
  --resource-group smart-elections-parser \
  --name smart-elections-parser-app
```

## 📈 Scaling & Performance

### Local Development

- Single-threaded Flask development server
- Adequate for testing and development

### Production (Azure)

```bash
# Increase tier if needed
az appservice plan update \
  --name smart-elections-parser-plan \
  --sku P1V2  # Premium tier with more CPU/memory
```

### Horizontal Scaling

- Use Azure App Service auto-scaling
- Configure rules based on metrics (CPU, memory, request count)

## 🔄 Continuous Deployment

### GitHub Actions Example

```yaml
name: Deploy to Azure
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Deploy to App Service
        uses: azure/webapps-deploy@v2
        with:
          app-name: smart-elections-parser-app
          publish-profile: ${{ secrets.AZURE_PUBLISH_PROFILE }}
```

## ✅ Post-Deployment Checklist

- [ ] Verify application starts without errors
- [ ] Test core functionality (basic parsing)
- [ ] Verify security settings (`QA_REQUIRE_CERT_AUTH=true`)
- [ ] Check logging (errors and access logs)
- [ ] Monitor resource usage (CPU, memory, disk)
- [ ] Test failover/recovery procedures
- [ ] Document any non-standard configuration
- [ ] Set up alerts and monitoring
- [ ] Schedule backups (if using database)
- [ ] Communicate deployment to stakeholders

---

**Related Documents**:

- [Security & Authentication](./SECURITY.md) - Certificate-based auth
- [Operations Runbook](./OPERATIONS.md) - Operational procedures
- [Troubleshooting](../DEPLOYMENT_GUIDE.md) - Detailed troubleshooting

**Source References**:

- [DEPLOYMENT_GUIDE.md](../DEPLOYMENT_GUIDE.md)
- [AZURE_DEPLOYMENT_CHECKLIST.md](../AZURE_DEPLOYMENT_CHECKLIST.md)

**Last Updated**: Consolidated deployment guide
