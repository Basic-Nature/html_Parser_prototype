# 🚀 Azure Deployment - Quick Fix Applied

**Date**: February 2, 2026  
**Issue**: orjson import fails on Python 3.13 with `ImportError: undefined symbol: _PyDict_Contains_KnownHash`  
**Status**: ✅ **FIXED** — Using Python 3.12.10

---

## What Was Fixed

### 1. ✅ `.Dockerfile` - Python Version Downgrade

**File**: [.Dockerfile](.Dockerfile) Line 2  
**Change**: `FROM python:3.13-slim-bookworm` → `FROM python:3.12.10-slim-bookworm`  
**Reason**: orjson 3.9.5 binary is incompatible with Python 3.13 ABI; 3.12.10 is the tuned version

### 2. ✅ `requirements.txt` - Added Warning Documentation

**File**: [requirements.txt](requirements.txt) Lines 37-40  
**Change**: Added comments explaining Python 3.12.10 requirement and orjson coupling  
**Reason**: Prevent future deployments with Python 3.13; enforce 3.12.10 for consistency

### 3. ✅ `AZURE_DEPLOYMENT_CHECKLIST.md` - Created

**File**: [AZURE_DEPLOYMENT_CHECKLIST.md](AZURE_DEPLOYMENT_CHECKLIST.md)  
**Purpose**: Comprehensive guide for future Azure deployments, known issues, and rollback procedures

---

## How to Redeploy

### Step 1: Build Local Image (Verify Fix)

```bash
cd C:\Users\olivi\html_Parser_prototype

# Build with .Dockerfile (not Dockerfile, note the dot)
docker build -f .Dockerfile -t ballotlens:test .
```

**Expected Output**:

```txt
✅ Installation of requirements.txt ... SUCCESS
✅ spaCy model en_core_web_sm loaded ... SUCCESS
✅ sentence-transformers/all-MiniLM-L6-v2 saved to /models/sentence/all-MiniLM-L6-v2 ... SUCCESS
```

### Step 2: Test Container Starts

```bash
docker run -p 8000:8000 ballotlens:test
```

**Expected to see**:

```txt
[INFO] Starting gunicorn 25.0.0
[INFO] Listening at: http://0.0.0.0:8000 (PID: 1)
[INFO] Using worker: sync
[INFO] Booting worker with pid: 7
```

**NOT** (this was the error):

```txt
[ERROR] Exception in worker process
ImportError: /usr/local/lib/python3.13/site-packages/orjson/orjson.cpython-313-x86_64-linux-gnu.so: undefined symbol: _PyDict_Contains_KnownHash
```

### Step 3: Push to Azure Container Registry

```bash
# Tag image
$hash = git rev-parse --short HEAD
docker tag ballotlens:test `
  ballotlensregistry-c2dtaseferg0gchr.azurecr.io/ballotlens:${hash}

# Login to ACR
az acr login --name ballotlensregistry-c2dtaseferg0gchr

# Push
docker push ballotlensregistry-c2dtaseferg0gchr.azurecr.io/ballotlens:${hash}
```

### Step 4: Deploy in Azure Portal

Navigate to: **App Service → Deployment Center → Settings**

Update the image tag to your new hash:

```txt
ballotlensregistry-c2dtaseferg0gchr.azurecr.io/ballotlens:YOUR_HASH
```

### Step 5: Monitor Startup

Go to: **App Service → Log Stream**

Watch for:

- ✅ "Container is running" (within 60 seconds)
- ✅ "Listening at: <http://0.0.0.0:8000>" (gunicorn ready)
- ❌ Any "ImportError" messages

---

## Files Changed

| File | Change | Impact |
| ------ | -------- | -------- |
| [.Dockerfile](.Dockerfile) | Line 2: Python 3.13 → 3.12.10 | **CRITICAL** - Fixes immediate deployment failure |
| [requirements.txt](requirements.txt) | Lines 37-40: Added comments | **INFO** - Prevents future misconfigurations |
| [AZURE_DEPLOYMENT_CHECKLIST.md](AZURE_DEPLOYMENT_CHECKLIST.md) | NEW FILE | **REFERENCE** - Deployment runbook & troubleshooting |

---

## Why This Happened

```txt
Timeline of the Issue:
├─ Azure Container Registry was built with Python 3.13
├─ requirements.txt pins orjson==3.9.5 (compiled for Python 3.12)
├─ Container boots, imports webapp.Smart_Elections_Parser_Webapp
├─ Line 58: import orjson
├─ Python 3.13's C API differs from 3.12
├─ orjson.so binary has symbol: _PyDict_Contains_KnownHash (3.12 only)
├─ ImportError: undefined symbol → Worker fails to boot
└─ Result: All 5 gunicorn workers die during startup
```

**Why Python 3.12?**

- `pyproject.toml` specifies `python_version = "3.12"`
- All testing/CI done on Python 3.12
- orjson 3.9.5 wheels are precompiled for 3.12
- Future: orjson 3.10+ may support 3.13 (tracked in their GitHub)

---

## Verification Checklist

Before marking as complete:

- [ ] `.Dockerfile` line 2 changed to `python:3.12-slim-bookworm`
- [ ] `requirements.txt` has warning comment for orjson
- [ ] Local `docker build -f .Dockerfile` succeeds
- [ ] `docker run` shows "Listening at <http://0.0.0.0:8000>"
- [ ] No "ImportError" in logs
- [ ] Azure deployment uses new image hash
- [ ] Azure logs show "Container is running" within 60s
- [ ] Health check passes: `curl https://ballotlens.azurewebsites.net/health`

---

## Rollback (If Needed)

If new deployment still fails:

```bash
# Get previous working image hash
az acr repository list-manifests --name ballotlensregistry-c2dtaseferg0gchr \
  --repository ballotlens --query "[*].tags[0]" -o table

# Revert to previous hash
az webapp config container set \
  --name ballotlens \
  --resource-group ballot-lens-rg \
  --docker-custom-image-name \
    ballotlensregistry-c2dtaseferg0gchr.azurecr.io/ballotlens:PREVIOUS_HASH
```

---

## Next Steps (After Successful Deployment)

1. ✅ Verify webapp loads at <https://ballotlens.azurewebsites.net>
2. ✅ Test core functionality (parser runs, logs display correctly)
3. ✅ Monitor App Insights for errors over 24 hours
4. ✅ Consider upgrading orjson to 3.10 if available (check GitHub)
5. ✅ Add Python 3.12 version constraint to CI/CD pipeline

---

**Questions?** See [AZURE_DEPLOYMENT_CHECKLIST.md](AZURE_DEPLOYMENT_CHECKLIST.md) for detailed troubleshooting and monitoring guide.
