# Azure Deployment Checklist & Known Issues

**Last Updated**: February 2, 2026  
**Status**: ✅ ACTIVE

---

## 🚨 Critical Issues

### Issue #1: orjson + Python 3.13 Binary Incompatibility

**Status**: ✅ **RESOLVED** (Feb 2, 2026)

**Problem**:

```txt
ImportError: undefined symbol: _PyDict_Contains_KnownHash
```

- Occurred when Azure deployed Python 3.13 with orjson 3.9.5
- Binary wheels have ABI mismatch with Python 3.13
- Blocks ALL worker processes during gunicorn startup

**Root Cause**:

- `.Dockerfile` used `FROM python:3.13-slim-bookworm` (too new)
- `pyproject.toml` and `requirements.txt` target Python 3.12
- orjson 3.9.5 binary was compiled for Python 3.12, not 3.13

**Fix Applied**:

1. ✅ Changed `.Dockerfile` line 2 to `FROM python:3.12-slim-bookworm`
2. ✅ Added warning in `requirements.txt` about orjson + Python version coupling
3. ✅ Added comment in `.Dockerfile` explaining the constraint

**How to Verify**:

1. Pull latest image from Azure Container Registry
2. Run container → gunicorn should boot without orjson errors
3. Check logs: `[INFO] Listening at: http://0.0.0.0:8000`

**Future Workarounds** (if Python 3.13 required):

- Option A: Wait for orjson 3.10+ with Python 3.13 support
- Option B: Build orjson from source: `pip install --no-binary orjson orjson==3.9.5`
- Option C: Switch to alternative JSON library (e.g., `ujson`, `msgpack`)

**Tracking**: <https://github.com/ijl/orjson/issues/564>

---

## ✅ Pre-Deployment Validation

### Before pushing to Azure ACR, verify

```bash
# 1. Check Python version in Dockerfile
grep "FROM python:" .Dockerfile
# Expected: python:3.12-slim-bookworm (NOT 3.13)

# 2. Verify orjson version matches pyproject.toml
grep "orjson" requirements.txt
# Expected: orjson==3.9.5

# 3. Test local build
docker build -f .Dockerfile -t ballotlens:test .
# Should complete without import errors

# 4. Run container locally
docker run -p 8000:8000 ballotlens:test
# Wait for: "[INFO] Listening at: http://0.0.0.0:8000"
```

---

## 🔍 Known Version Constraints

| Package | Version | Constraint | Reason |
| --------- | --------- | ----------- | -------- |
| **Python** | 3.12 | MUST be 3.12 | orjson ABI compatibility |
| **orjson** | 3.9.5 | MUST stay ≤3.9 | Python 3.13 not supported in 3.9.x |
| **spaCy** | 3.8.11 | Pinned | Model compatibility with 3.8.0 |
| **torch** | 2.9.1 | CPU only | Reduces image size (~4GB → 1.5GB) |
| **sentence-transformers** | ≥5.1.0 | Min 5.1 | Embedding quality improvements |

---

## 📋 Azure Deployment Steps

### 1. **Pre-Deployment** (Local)

- [ ] Run `python automate.py` (full validation)
- [ ] Verify all tests pass
- [ ] Check `.Dockerfile` Python version (must be 3.12)
- [ ] Confirm `requirements.txt` has `orjson==3.9.5`

### 2. **Build & Registry**

```bash
# Build image
docker build -f .Dockerfile -t ballotlens:latest .

# Tag for Azure Container Registry
docker tag ballotlens:latest \
  ballotlensregistry-c2dtaseferg0gchr.azurecr.io/ballotlens:latest

# Push to ACR
az acr login --name ballotlensregistry-c2dtaseferg0gchr
docker push ballotlensregistry-c2dtaseferg0gchr.azurecr.io/ballotlens:latest
```

### 3. **Azure App Service Update**

```bash
# Option A: Via Azure CLI
az webapp config container set \
  --name ballotlens \
  --resource-group ballot-lens-rg \
  --docker-custom-image-name ballotlensregistry-c2dtaseferg0gchr.azurecr.io/ballotlens:latest \
  --docker-registry-server-url https://ballotlensregistry-c2dtaseferg0gchr.azurecr.io

# Option B: Via Azure Portal
# Settings → Container Settings → Image Source → Azure Container Registry
# Select: ballotlens:latest
```

### 4. **Post-Deployment** (Azure)

- [ ] Check **Deployment Center** → Status should show "Successful"
- [ ] Wait 2-3 minutes for container to start
- [ ] Monitor **Log Stream** for gunicorn startup (should see "Listening at")
- [ ] Test `/health` endpoint → Should return `{"status": "ok"}`
- [ ] Check **App Insights** for any import errors

### 5. **Rollback Plan** (If deployment fails)

```bash
# Revert to previous working image
az webapp config container set \
  --name ballotlens \
  --resource-group ballot-lens-rg \
  --docker-custom-image-name ballotlensregistry-c2dtaseferg0gchr.azurecr.io/ballotlens:<PREVIOUS_HASH>
```

---

## 🛠️ Environment Variables (Azure App Settings)

Ensure these are set in **Azure App Service → Configuration → Application settings**:

| Variable | Value | Purpose |
| --------- | ------- | --------- |
| `WEBSITES_PORT` | `8000` | Port mapping for App Service |
| `ENABLE_OCR` | `true` | OCR support enabled |
| `PLAYWRIGHT_BROWSERS_PATH` | `0` | Use system-installed browsers |
| `TRANSFORMERS_OFFLINE` | `1` | Use local sentence-transformers model |
| `HUGGINGFACE_HUB_OFFLINE` | `1` | No HuggingFace network calls |
| `SENTENCE_TRANSFORMER_LOCAL_PATH` | `/models/sentence/all-MiniLM-L6-v2` | Baked model path |
| `DATABASE_URL` | `postgresql://...` | PostgreSQL connection string |
| `FLASK_SECRET_KEY` | `<long-random-string>` | Session encryption key |

---

## 📊 Monitoring & Alerts

### Key Metrics to Watch (Azure Monitor)

1. **Container Startup Time**: Should complete within 60s
   - Alert if > 120s (may indicate image pull issues)
2. **Worker Boot Failures**: Should be 0
   - Alert if > 0 (import errors, missing deps)
3. **HTTP 5xx Errors**: Should be < 1% of requests
   - Alert if spike (app crashing, out of memory)
4. **Memory Usage**: Monitor growth
   - Alert if > 512MB (sentence-transformers can be memory-hungry)

### Logs to Check on Failure

```txt
# Check these in real-time logs:
https://ballotlens-cubrcudretaebca9.scm.westus3-01.azurewebsites.net/api/vfs/LogFiles/

1. StandardOutput (gunicorn output)
   - Look for "ImportError: undefined symbol" → Need Python 3.12
   - Look for "[ERROR] Worker failed to boot" → Deployment blocker

2. CodeProfiler (if enabled)
   - trace file at /tmp/771c48_profiler_trace.json
   - Use for performance profiling

3. Docker events
   - "Site container ... terminated during site startup" → Check worker logs
```

---

## 🚀 Performance Tuning

### Gunicorn Workers (gunicorn.conf.py)

Current settings optimized for Azure App Service (B2 tier: 2 cores, 3.5GB RAM):

- **Workers**: 3 (default: CPU cores + 1)
- **Threads**: 4 (async support via threading)
- **Timeout**: 120s (parser operations can be slow)
- **Keep-alive**: 5s

Adjust if you upgrade app tier:

```python
# For B3 (4 cores, 7GB):
workers = 5
threads = 4

# For P1V2 (2 cores, 3.5GB):
workers = 3
threads = 4
```

---

## 🔐 Security Checklist

- [ ] No hardcoded secrets in `.Dockerfile` or `requirements.txt`
- [ ] Azure Key Vault stores database credentials
- [ ] HTTPS enforced (App Service HTTPS only)
- [ ] CSP headers configured in Flask app
- [ ] Client certificate auth enabled (if needed)
- [ ] CORS restricted to known origins

---

## 📝 Troubleshooting

### "Container start method failed"

→ Check `.Dockerfile` Python version (should be 3.12)

### "Worker failed to boot with exit code 3"

→ This is the orjson error. Verify:

- `.Dockerfile` line 2: `FROM python:3.12-slim-bookworm`
- Rebuild and push new image

### "Site is blocked" (2-minute startup delay)

→ Normal Azure behavior after deployment; wait for "Site is unblocked" message

### "Connection refused: database"

→ Verify DATABASE_URL in App Settings and database is accessible from Azure VNet

---

## ✅ Last Deployment Summary

**Date**: February 2, 2026  
**Changes**: Fixed Python 3.13 → 3.12 in `.Dockerfile`  
**Status**: ✅ Ready to redeploy  
**Build Time**: ~5 minutes (image rebuild)  
**Expected Startup**: 60-90 seconds

**Next Deployment Command**:

```bash
docker build -f .Dockerfile -t ballotlens:$(git rev-parse --short HEAD) .
docker tag ballotlens:$(git rev-parse --short HEAD) \
  ballotlensregistry-c2dtaseferg0gchr.azurecr.io/ballotlens:latest
docker push ballotlensregistry-c2dtaseferg0gchr.azurecr.io/ballotlens:latest
```

---

**Questions?** Check:

1. [.Dockerfile](.Dockerfile) lines 1-50 for base image config
2. [requirements.txt](requirements.txt) for dependency versions
3. [gunicorn.conf.py](gunicorn.conf.py) for worker settings
4. Azure App Service **Deployment Center** for deployment logs
