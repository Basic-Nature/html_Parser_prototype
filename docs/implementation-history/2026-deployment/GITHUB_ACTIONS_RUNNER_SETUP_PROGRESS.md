# GitHub Actions Runner Setup Progress – Reference Guide

**Last Updated:** May 6, 2026
**Status:** In Progress (Paused for price stability monitoring)
**Priority:** Resume when ready to test connections and validate secrets

---

## 1. Current Setup Status

### ✅ Completed Tasks

- [x] Private Git repository configured for runner
- [x] GitHub Actions runner infrastructure initialized
- [x] Repository access configured
- [x] Basic runner environment structure in place

### ❌ Outstanding Tasks

- [ ] Test connections (runner ↔ GitHub, runner ↔ API endpoints)
- [ ] Verify GitHub Secrets are correctly placed and accessible
- [ ] Validate environment variables in runner context
- [ ] Link access testing (specific endpoint TBD – user working on this)
- [ ] End-to-end workflow execution validation
- [ ] Redis cache removal verification (ensure no residual cost charges)

### 🟡 Work Paused For

**Reason:** Monitoring price stability after Redis cache removal
**Duration:** TBD – user will resume when ready
**Cost Concern:** Previous Redis usage generated unexpected charges; now disabled

---

## 2. What We Haven't Done Yet

### Phase 1: Connection Verification (NEXT)

When resuming, verify:

1. Runner can authenticate to GitHub Actions
2. Runner can reach configured API endpoints (health check, data_framework, etc.)
3. Network connectivity from runner to external services is stable

### Phase 2: Secrets & Environment Variables (DEPENDS ON PHASE 1)

1. Validate all secrets are stored in GitHub Settings → Secrets
2. Confirm secrets are accessible in workflow context
3. Test credential propagation to runner environment
4. Verify no secrets leak to logs or artifacts

### Phase 3: Workflow Execution (DEPENDS ON PHASES 1-2)

1. Trigger test workflow from runner
2. Validate smoke tests execute and report correctly
3. Verify stress test results are captured
4. Confirm API comparison runs without auth failures
5. Validate artifacts are produced and stored

### Phase 4: Cost & Monitoring (ONGOING)

1. Monitor GitHub Actions minutes usage
2. Confirm no Redis-related charges appear after cache removal
3. Set up cost alerts if available
4. Document monthly spend baseline

---

## 3. Secrets & Environment Variables Checklist

### GitHub Secrets Required (Repository Settings → Secrets and variables)

**Core Authentication:**

- [ ] `GITHUB_TOKEN` – Auto-provided by GitHub Actions (verify it's not manually re-stored)
- [ ] `AZURE_CREDENTIALS` – (if Azure deployment is part of workflow) JSON credentials object
- [ ] `API_AUTH_TOKEN` – (if API requires Bearer token) Private token for authentication

**Database/External Services:**

- [ ] `DATABASE_URL` – Connection string to production/staging DB (if applicable)
- [ ] `SERVICE_ACCOUNT_JSON` – Google Service Account (if Google Sheets/Cloud used)
- [ ] `REDIS_URL` – (**SHOULD BE REMOVED** if Redis cache disabled; verify it's not present)

**Application Configuration:**

- [ ] `AZURE_SUBSCRIPTION_ID` – Azure subscription identifier
- [ ] `AZURE_RESOURCE_GROUP` – Azure resource group name
- [ ] `AZURE_APP_SERVICE_NAME` – Azure App Service name for deployment
- [ ] `DOCKER_REGISTRY_URL` – Container registry endpoint (if used)
- [ ] `DOCKER_REGISTRY_USERNAME` – Container registry username
- [ ] `DOCKER_REGISTRY_PASSWORD` – Container registry password token

**Testing & Monitoring:**

- [ ] `STRESS_TEST_CONCURRENCY` – Parallel request count (recommend: 8-16)
- [ ] `STRESS_TEST_MAX_FAILURE_RATE` – Acceptable failure threshold (e.g., 0.05 = 5%)
- [ ] `SMOKE_TEST_TIMEOUT` – Endpoint timeout in seconds (recommend: 30)

### Environment Variables in Workflow File

**Reference Pattern (add to .github/workflows/test.yml or similar):**

```yaml
env:
  API_BASE_URL: https://electionpulse.org
  LOCAL_API_BASE_URL: http://127.0.0.1:5000
  STRESS_TEST_REQUESTS_PER_ENDPOINT: 20
  STRESS_TEST_CONCURRENCY: 8
  STRESS_TEST_MAX_FAILURE_RATE: 0.10

jobs:
  api-tests:
    runs-on: self-hosted
    steps:
      - name: Run Smoke Tests
        run: |
          python tools/smoke_webapp_api.py \
            --base-url ${{ env.LOCAL_API_BASE_URL }} \
            --output-json smoke-local-results.json
```

---

## 4. Testing Procedures (Ready to Use When Resuming)

### Quick Verification Script (Run on Runner)

```bash
#!/bin/bash
# Save as: scripts/verify_runner_ready.sh

echo "=== GitHub Actions Runner Readiness Check ==="
echo ""

# 1. GitHub connectivity
echo "1. Testing GitHub API access..."
curl -s -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/user \
  | grep -q '"login"' && echo "✓ GitHub auth working" || echo "✗ GitHub auth FAILED"

# 2. API endpoint connectivity
echo ""
echo "2. Testing API endpoints..."
for endpoint in http://127.0.0.1:5000/health \
                https://electionpulse.org/health; do
  curl -s -m 5 "$endpoint" > /dev/null && \
    echo "✓ $endpoint reachable" || \
    echo "✗ $endpoint FAILED"
done

# 3. Secrets accessible
echo ""
echo "3. Checking secrets in runner context..."
[ -n "$GITHUB_TOKEN" ] && echo "✓ GITHUB_TOKEN set" || echo "✗ GITHUB_TOKEN missing"
[ -n "$API_BASE_URL" ] && echo "✓ API_BASE_URL set" || echo "✗ API_BASE_URL missing"

# 4. Python/tools ready
echo ""
echo "4. Verifying tools..."
python tools/smoke_webapp_api.py --help > /dev/null && \
  echo "✓ smoke_webapp_api.py ready" || \
  echo "✗ smoke_webapp_api.py FAILED"

echo ""
echo "=== Readiness Check Complete ==="
```

### Manual Test Sequence (When Ready)

1. **SSH into runner machine** (or access runner environment)
2. **Run verification script:**

   ```bash
   bash scripts/verify_runner_ready.sh
   ```

3. **Check for failures** – any `✗` indicates a configuration gap
4. **If all pass:** Proceed to workflow execution test

### Workflow Execution Test

```yaml
# .github/workflows/runner-validation.yml
name: Runner Validation Test

on:
  workflow_dispatch:  # Manual trigger

jobs:
  validate-runner:
    runs-on: self-hosted
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Run Smoke Tests
        run: |
          python tools/smoke_webapp_api.py \
            --base-url http://127.0.0.1:5000 \
            --output-json /tmp/smoke-results.json

      - name: Upload Smoke Results
        uses: actions/upload-artifact@v3
        with:
          name: smoke-test-results
          path: /tmp/smoke-results.json

      - name: Report Results
        run: |
          python -c "
          import json
          with open('/tmp/smoke-results.json') as f:
              data = json.load(f)
          print('Smoke Test Summary:')
          print(f'  Timestamp: {data.get(\"timestamp\")}')
          print(f'  Base URL: {data.get(\"base_url\")}')
          print(f'  Results: {data.get(\"results\", {})}')
          "
```

---

## 5. Redis Cache Removal Verification

### ✅ Completed Actions

- Identified Redis cache added cost to charges
- Disabled Redis usage in codebase

### ⚠️ Remaining Verification

**Check for residual Redis references:**

```bash
grep -r "redis" --include="*.py" webapp/
grep -r "REDIS" --include="*.py" .
grep -r "cache" --include="*.yml" .github/workflows/
```

**Confirm charges have stopped:**

- [ ] Monitor GitHub Actions billing (Settings → Billing & Plans)
- [ ] Verify no Azure Cache for Redis charges in Azure portal
- [ ] Confirm monthly spend trend is decreasing
- [ ] Check runner machine cost is not unusually high

**If charges still appear:**

1. Check runner machine logs for Redis connection attempts
2. Verify .env file doesn't contain REDIS_URL
3. Search for any lingering Redis configuration in deployment files
4. Audit recent commits for any re-introduced Redis usage

---

## 6. Quick Reference: Resume Workflow

### When Ready to Resume

1. **Verify current state:**

   ```bash
   git status
   git log --oneline -5
   ```

2. **Check for Redis charges:**
   - Log into Azure portal → Cost Management
   - Search recent charges for "Cache for Redis"

3. **Run quick diagnostic:**

   ```bash
   bash scripts/verify_runner_ready.sh
   ```

4. **Fix any issues flagged** and update checklist above

5. **Trigger test workflow:**
   - Go to GitHub repo → Actions → "Runner Validation Test" → "Run workflow"

6. **Monitor execution:**
   - Watch workflow run in GitHub UI
   - Check runner machine logs if needed: `ssh user@runner-ip` → `/var/log/github-runner/`

7. **Review results:**
   - Download artifacts from workflow run
   - Compare against baseline smoke results from local dev environment
   - Document any divergences in this file

---

## 7. Known Issues & Workarounds

### Issue: Secrets not visible in workflow

**Workaround:** Ensure workflow file has correct repo access permissions:

```yaml
permissions:
  contents: read
  packages: read
```

### Issue: Runner fails to authenticate to GitHub

**Workaround:** Regenerate runner token in GitHub UI (Settings → Actions → Runners → Re-generate token)

### Issue: API calls from runner timeout

**Workaround:** Increase timeout in stress test and smoke test: `--timeout 60` (seconds)

### Issue: Previous Redis charges still appearing

**Workaround:** Check deployment pipeline for any Azure Resource Manager templates that auto-create Redis; disable in production deployment settings

---

## 8. Next Step Overview

| Phase   | Task                   | Status         | Trigger                    |
| ------- | ---------------------- | -------------- | -------------------------- |
| Current | Monitor price stability| 🟡 In Progress | Ongoing                    |
| Next    | Test runner connections| ⏸️ Paused      | User resume                |
| Then    | Validate secrets access| ⏸️ Paused      | Phase 1 pass               |
| After   | Execute workflows      | ⏸️ Paused      | Phase 2 pass               |
| Final   | Cost monitoring        | ⏸️ Paused      | Phases 1-3 complete        |

---

## 9. Supporting Documentation

**Related files to review when resuming:**

- [docs/DEPLOYMENT/POST_DEPLOY_VERIFICATION.md](../DEPLOYMENT/POST_DEPLOY_VERIFICATION.md) – Post-deployment checks
- [.github/workflows/](../../.github/workflows/) – Existing workflow files (if any)
- [tools/smoke_webapp_api.py](../../tools/smoke_webapp_api.py) – Smoke test utility
- [tools/stress_webapp_api.py](../../tools/stress_webapp_api.py) – Stress test utility

**Key contact points:**

- [GitHub Actions documentation](https://docs.github.com/en/actions)
- [Self-hosted runners guide](https://docs.github.com/en/actions/hosting-your-own-runners)
- [Azure Billing & Cost Management](https://portal.azure.com) → Cost Management

---

## 10. Session Notes

**May 6, 2026 Decision:** User chose to pause runner setup to monitor price stability after Redis removal. Focus shifted to link access testing and other development priorities. This document serves as a comprehensive reference for resuming runner setup without losing context.

**Cost Context:** Redis cache had been added and subsequently removed. User wants to verify charges have stabilized before investing further in runner configuration.

**Action Item:** Return to this document once:

1. Price stability is confirmed
2. Link access testing is complete
3. User is ready to resume runner workflow setup
