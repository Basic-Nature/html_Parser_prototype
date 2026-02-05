# Deployment Guide - Smart Elections Parser

Complete deployment instructions for Smart Elections Parser, including local development, Docker containers, and Azure Web App production deployment.

---

## 🚀 Quick Start (5 minutes)

### Local Development

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create .env file
cp .env.example .env
# Edit .env with your settings

# 3. Initialize database
python -c "from webapp.parser.utils.models import Base, engine; Base.metadata.create_all(engine)"

# 4. Run Flask app
python -m webapp.Smart_Elections_Parser_Webapp

# 5. Access in browser
open http://localhost:5000
```

### Docker (Local Testing)

```bash
# Build image
docker build -t parser:latest .

# Run container
docker run -p 5000:5000 \
  -e POSTGRES_URL="postgresql://user:pass@db:5432/parser" \
  parser:latest

# Access
open http://localhost:5000
```

### Azure Web App (Production)

```bash
# Prerequisites: Azure CLI installed, authenticated

# Deploy
az webapp up --name parser-app --resource-group my-rg --runtime "PYTHON:3.12"

# View logs
az webapp log tail --name parser-app --resource-group my-rg

# Access
open https://parser-app.azurewebsites.net
```

---

## 📋 Prerequisites

### System Requirements

| Component | Minimum | Recommended | Notes |
| ----------- | --------- | ------------- | ------- |
| Python | 3.11 | 3.12 | Active support required |
| PostgreSQL | 12 | 14+ | SQLAlchemy models compatible |
| Memory | 2GB | 4GB+ | For ML/NLP models |
| Disk | 5GB | 20GB+ | For logs, cache, fixtures |

### Dependencies

```txt
# Core
Flask==2.3.0+
Flask-SocketIO==5.3.0+
SQLAlchemy==2.0.0+
orjson==3.9.0+

# Data Science
pandas==1.5.0+
numpy==1.24.0+
spacy==3.5.0+ (with model download)
scikit-learn==1.2.0+

# Security
cryptography==40.0.0+
python-dotenv==1.0.0+

# See requirements.txt for full list
```

### Optional but Recommended

| Tool | Purpose | Install |
| ----------- | -------- | --------- |
| `poppler-utils` | PDF processing | `apt-get install poppler-utils` (Linux) |
| `tesseract-ocr` | OCR support | `brew install tesseract` (macOS) |
| `Docker` | Containerization | <https://docker.com> |
| `nginx` | Reverse proxy | `apt-get install nginx` (Linux) |

---

## 📦 Local Development Setup

### 1. Clone Repository

```bash
git clone https://github.com/Basic-Nature/html_Parser_prototype.git
cd html_Parser_prototype
```

### 2. Create Virtual Environment

```bash
# Create venv
python3.12 -m venv venv

# Activate
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
# Create .env file
cat > .env << EOF
# Database
POSTGRES_URL=postgresql://postgres:postgres@localhost:5432/parser_dev

# Logging
LOG_LEVEL=DEBUG
LOG_DIR=webapp/parser/Context_Integration/Context_Library/log

# Security
CERT_AUTH_ENABLED=False  # Enable when certs ready
SSO_ENABLED=False

# Parser
TIMEOUT_SECONDS=300
MAX_RETRIES=3

# Flask
FLASK_ENV=development
FLASK_DEBUG=True
EOF
```

### 5. Initialize Database

```bash
# Create tables
python -c "from webapp.parser.utils.models import Base, engine; Base.metadata.create_all(engine)"

# Verify
python -c "from webapp.parser.utils.db_utils import get_session; s = get_session(); print(f'Session OK: {s}')"
```

### 6. Download spaCy Model

```bash
python -m spacy download en_core_web_md
python -m spacy download en_core_web_sm
```

### 7. Run Application

```bash
python -m webapp.Smart_Elections_Parser_Webapp
```

**Output:**

```txt
 * Serving Flask app 'Smart_Elections_Parser_Webapp'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
```

**Access**: <http://localhost:5000>

---

## 🐳 Docker Deployment

### Build Container

```bash
# Standard build
docker build -t parser:latest .

# With build args
docker build \
  --build-arg PYTHON_VERSION=3.12 \
  --build-arg LOG_LEVEL=INFO \
  -t parser:prod .
```

### Run Locally

```bash
# Development (with debug logs)
docker run -it \
  -p 5000:5000 \
  -e LOG_LEVEL=DEBUG \
  -e POSTGRES_URL="postgresql://postgres:postgres@host.docker.internal:5432/parser" \
  parser:latest

# Production
docker run -d \
  -p 5000:5000 \
  -e LOG_LEVEL=INFO \
  -e POSTGRES_URL="$POSTGRES_URL" \
  -e FLASK_ENV=production \
  --health-cmd="curl -f http://localhost:5000/health || exit 1" \
  --health-interval=30s \
  parser:latest
```

### Push to Registry

```bash
# Docker Hub
docker tag parser:latest myuser/parser:latest
docker push myuser/parser:latest

# Azure Container Registry
az acr login --name myregistry
docker tag parser:latest myregistry.azurecr.io/parser:latest
docker push myregistry.azurecr.io/parser:latest
```

---

## ☁️ Azure Web App Deployment

### Prerequisites

```bash
# Install Azure CLI
# https://learn.microsoft.com/en-us/cli/azure/install-azure-cli

# Login
az login

# Set defaults
az account set --subscription "MySubscription"
az config set defaults.group=my-resource-group
```

### Deploy via CLI

```bash
# Quick deploy (creates app if needed)
az webapp up \
  --name parser-app \
  --resource-group my-rg \
  --runtime "PYTHON:3.12" \
  --app-service-plan my-plan

# Stream deployment logs
az webapp log tail --name parser-app
```

### Deploy via Docker Container

```bash
# Create app service plan
az appservice plan create \
  --name parser-plan \
  --resource-group my-rg \
  --sku B2 \
  --is-linux

# Create web app
az webapp create \
  --name parser-app \
  --resource-group my-rg \
  --plan parser-plan \
  --deployment-container-image-name myregistry.azurecr.io/parser:latest

# Configure registry
az webapp config container set \
  --name parser-app \
  --resource-group my-rg \
  --docker-custom-image-name myregistry.azurecr.io/parser:latest \
  --docker-registry-server-url https://myregistry.azurecr.io \
  --docker-registry-server-user $USERNAME \
  --docker-registry-server-password $PASSWORD
```

### Configuration

```bash
# Set environment variables
az webapp config appsettings set \
  --name parser-app \
  --resource-group my-rg \
  --settings \
    POSTGRES_URL="postgresql://user:pass@server:5432/db" \
    LOG_LEVEL=INFO \
    FLASK_ENV=production \
    WEBSITES_PORT=5000

# Enable logging
az webapp log config \
  --name parser-app \
  --resource-group my-rg \
  --application-logging true \
  --level verbose

# View logs
az webapp log tail --name parser-app
```

---

## 🏥 Health Checks & Monitoring

### Endpoints

```bash
# Basic health
curl http://localhost:5000/health

# Database health
curl http://localhost:5000/health/db

# Metrics
curl http://localhost:5000/health/metrics

# Operations dashboard
open http://localhost:5000/azure_health
```

### Expected Responses

| Endpoint | Status | Response |
| ----------- | -------- | --------- |
| `/health` | ✅ | `{"status":"healthy"}` |
| `/health/db` | ✅ | `{"status":"connected","records":1000}` |
| `/health/metrics` | ✅ | `{"requests":5000,"errors":2,"uptime":"48h"}` |

### Troubleshooting Health Checks

**Issue**: `/health` returns 500

```bash
# Check logs
tail -f webapp/parser/Context_Integration/Context_Library/log/*.jsonl

# Verify database
python -c "from webapp.parser.utils.db_utils import get_session; s = get_session(); print(s.execute('SELECT 1'))"
```

**Issue**: `/health/db` returns error

```bash
# Check connection string
echo $POSTGRES_URL

# Test with psql
psql $POSTGRES_URL -c "SELECT 1"

# Verify tables
python -c "from webapp.parser.utils.models import Base; print(Base.metadata.tables.keys())"
```

---

## 🔐 Security Configuration

### Certificate Authentication

```bash
# Generate test certificate
openssl req -x509 -newkey rsa:2048 -keyout cert.key -out cert.crt -days 365

# Configure in .env
CERT_AUTH_ENABLED=True
CERT_PATH=/etc/ssl/certs
CERT_KEY_PATH=/etc/ssl/private
CERT_CACHE_TTL=3600
```

### HTTPS/TLS

```bash
# Azure: Enable HTTPS enforcement
az webapp config set \
  --name parser-app \
  --resource-group my-rg \
  --https-only true

# Local: Use nginx as reverse proxy
# /etc/nginx/sites-available/parser
upstream parser {
    server 127.0.0.1:5000;
}

server {
    listen 443 ssl;
    server_name parser.example.com;
    ssl_certificate /path/to/cert.crt;
    ssl_certificate_key /path/to/cert.key;
    location / {
        proxy_pass http://parser;
    }
}
```

### Environment Variables

```bash
# Never commit to git
echo ".env" >> .gitignore

# Use secrets manager
az keyvault create --name parser-secrets --resource-group my-rg
az keyvault secret set --vault-name parser-secrets --name postgres-url --value "$POSTGRES_URL"
```

---

## 📊 Known Issues & Fixes

### Issue: orjson Python 3.13 ABI Mismatch

**Error**: `ImportError: orjson cannot import in Python 3.13`

**Fix**:

```bash
# Downgrade Python to 3.12 or pin orjson < 4.0
pip install "orjson==3.9.5" --force-reinstall
```

### Issue: Timeout on Large Files

**Error**: `TimeoutError: Parser timeout after 300s`

**Fix**:

```bash
# Increase timeout
export TIMEOUT_SECONDS=600

# Or in .env
TIMEOUT_SECONDS=600
```

### Issue: PostgreSQL Connection Refused

**Error**: `psycopg2.OperationalError: connection refused`

**Fix**:

```bash
# Verify PostgreSQL running
pg_isready -h localhost -p 5432

# Check connection string
echo $POSTGRES_URL

# For Azure: ensure firewall allows connection
az postgres server firewall-rule create \
  --name AllowLocalhost \
  --server myserver \
  --start-ip-address 0.0.0.0 \
  --end-ip-address 255.255.255.255
```

---

## 📝 Verification Checklist

Before considering deployment complete:

- [ ] Flask app starts without errors: `python -m webapp.Smart_Elections_Parser_Webapp`
- [ ] Health endpoint responds: `curl http://localhost:5000/health`
- [ ] Database connected: `curl http://localhost:5000/health/db`
- [ ] Web UI loads: `http://localhost:5000`
- [ ] Parser accepts URLs without crashing
- [ ] Quarantine system accessible: `http://localhost:5000/quarantine/review`
- [ ] Logs being written: `ls -la webapp/parser/Context_Integration/Context_Library/log/`
- [ ] No errors in logs: `tail -f webapp/parser/Context_Integration/Context_Library/log/*.jsonl`
- [ ] Tests passing: `pytest webapp/tests/ -v`
- [ ] Type checking clean: `mypy webapp/ --ignore-missing-imports`

---

## 📞 Support & Troubleshooting

### Debug Mode

```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG

# Run with verbose output
FLASK_APP=webapp/Smart_Elections_Parser_Webapp FLASK_ENV=development python -m flask run --debug

# Capture all logs
python -m webapp.Smart_Elections_Parser_Webapp 2>&1 | tee debug.log
```

### Common Log Paths

```txt
webapp/parser/Context_Integration/Context_Library/log/*.jsonl
webapp/parser/Context_Integration/Context_Library/cache/
output/
```

### Getting Help

1. **Check logs**: `tail -f webapp/parser/Context_Integration/Context_Library/log/*.jsonl`
2. **Review docs**: See [docs/](../docs/) folder
3. **Run tests**: `pytest webapp/tests/ -v`
4. **Check issues**: <https://github.com/Basic-Nature/html_Parser_prototype/issues>

---

**Last Updated**: February 5, 2026  
**Related**: [AZURE_DEPLOYMENT_CHECKLIST.md](AZURE_DEPLOYMENT_CHECKLIST.md) | [architecture.md](architecture.md) | [QUICK_REFERENCES.md](QUICK_REFERENCES.md)
