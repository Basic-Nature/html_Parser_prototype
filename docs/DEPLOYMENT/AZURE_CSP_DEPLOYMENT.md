---
layout: default
title: "Azure Deployment - CSP Hardening Checklist"
---

## Azure Deployment - CSP Hardening Checklist

## Pre-Deployment Verification (Local)

### 1. Verify Local Resources Exist

```bash
# Check all vendor files are present
ls -la webapp/static/vendor/ | grep -E "bootstrap|chart|socket"

# Expected files:
# bootstrap.min.css (wrapper)
# bootstrap-5.3.8.min.css (versioned)
# bootstrap.bundle.min.js (wrapper) 
# bootstrap-5.3.8.bundle.min.js (versioned)
# chart.umd.js
# socket.io-4.7.5.min.js
# xlsx.full.min.js
```

### 2. Test STRICT CSP Mode Locally

```bash
# Set environment
export CSP_MODE=STRICT
export ALLOW_STYLE_ATTR=0

# Run Flask app
python -m webapp.Smart_Elections_Parser_Webapp

# Open browser: http://localhost:5000
```

### 3. Verify No CSP Violations in Browser

1. Open **Developer Tools → Console**
2. Load main page and check for messages:
   - ✅ Should NOT see "violates the following Content Security Policy"
   - ✅ Bootstrap styling should be visible (buttons, modals, etc.)
   - ✅ Charts should render normally

### 4. Test All Features

- [ ] Load ballot results (if test data available)
- [ ] Open modal dialogs (check Bootstrap modals work)
- [ ] Real-time updates via Socket.IO
- [ ] Quality dashboard (Chart.js visualization)
- [ ] Excel export functionality
- [ ] Verify no console errors

### 5. Check Response Headers

```bash
curl -I http://localhost:5000/ | grep -A 1 "Content-Security-Policy"

# Verify output contains:
# - script-src 'self' 'nonce-...'  (NO cdn.jsdelivr.net)
# - style-src-elem 'self'  (NO cdn.jsdelivr.net)
# - style-src-attr 'none'  (no inline styles)
```

---

## Azure Deployment Steps

### Step 1: Set Environment Variables

In **Azure Portal**:

1. Navigate to your **App Service**
2. Go to **Settings → Configuration**
3. Click **+ New application setting**
4. Add these settings:

| Name | Value | Purpose |
| ------ | ------- | --------- |
| `CSP_MODE` | `STRICT` | Enable strict CSP (no CDN) |
| `ALLOW_STYLE_ATTR` | `0` | Disable inline styles |
| `NODE_OPTIONS` | `--trace-deprecation` | (keep existing if present) |
| `GITHUB_PAGES` | `true` | (keep existing if present) |

1. Click **Save**
2. **Confirm restart** when prompted

### Step 2: Verify Environment Variables Applied

```bash
# SSH into Azure App Service (if available) or use Log Stream

# Check environment is set
echo $CSP_MODE              # Should output: STRICT
echo $ALLOW_STYLE_ATTR       # Should output: 0
```

### Step 3: Monitor Deployment

1. Go to **Deployments** tab
2. Wait for deployment to complete (green checkmark)
3. Open **Log Stream** tab
4. Look for startup output:

   ```txt
   * Running on https://your-app.azurewebsites.net
   * Debugger PIN: [...]
   ```

### Step 4: Post-Deployment Verification

#### Test 1: Load Application

```bash
curl -I https://your-app.azurewebsites.net/ | grep Content-Security-Policy
```

**Expected output**:

```txt
Content-Security-Policy: default-src 'self'; base-uri 'self'; frame-ancestors 'none'; form-action 'self'; object-src 'none'; script-src 'self' 'nonce-...'; style-src 'self'; style-src-elem 'self'; style-src-attr 'none'; img-src 'self' data:; font-src 'self' data:; connect-src 'self' ws: wss:;
```

- ✅ No `https://cdn.jsdelivr.net` in policy
- ✅ No `https://cdn.socket.io` in policy
- ✅ Contains `'nonce-...'` for scripts

#### Test 2: Browser Console Check

1. Navigate to <https://your-app.azurewebsites.net>
2. Open **DevTools → Console**
3. Expected state:
   - ✅ Page renders with Bootstrap styling
   - ✅ No red CSP violation messages
   - ✅ UI appears normal (modals, buttons functional)

#### Test 3: API Functionality

1. Submit a ballot/election data (via UI or API)
2. Expected:
   - ✅ Real-time updates work (Socket.IO)
   - ✅ Quality metrics visible (Chart.js)
   - ✅ No 403 CSP errors in response headers

---

## Fallback Scenario (Emergency Only)

### If Local Resources Fail

If you need to temporarily allow CDN fallback (e.g., if local vendor files become corrupted):

1. Azure Portal → **Configuration**
2. Change `CSP_MODE` to `RELAXED`
3. Click **Save** and **Confirm restart**
4. Verify CDN is being used:

   ```bash
   curl -I https://your-app.azurewebsites.net/ | grep Content-Security-Policy
   # Should now include: https://cdn.jsdelivr.net
   ```

5. **Change back to STRICT** once local resources are restored:

   ```bash
   # Download fresh Bootstrap CSS/JS to vendor/ directory
   curl -o webapp/static/vendor/bootstrap-5.3.8.min.css \
     https://cdn.jsdelivr.net/npm/bootstrap@5.3.8/dist/css/bootstrap.min.css
   ```

---

## Troubleshooting Azure Deployment

### Issue: "violates Content Security Policy" Errors in Azure

**Diagnosis**:

```bash
# Check which resources are being blocked
curl -I https://your-app.azurewebsites.net/ | grep -i csp
```

**Solution**:

1. Ensure `CSP_MODE=STRICT` environment variable is set
2. Verify all vendor files uploaded: `webapp/static/vendor/bootstrap*.*`
3. Restart App Service if environment variable changed
4. Wait 60 seconds for environment to fully apply

### Issue: Bootstrap Styling Missing

**Diagnosis**: Page loads but no Bootstrap styling

**Solutions**:

1. Check Network tab in DevTools - look for 404 on CSS files
2. Verify `webapp/static/vendor/bootstrap-5.3.8.min.css` exists in deployment
3. Check App Service logs for errors

### Issue: Real-Time Updates Don't Work

**Diagnosis**: Socket.IO connection fails, no real-time data

**Solutions**:

1. Verify `connect-src 'self' ws: wss:` is in CSP header
2. Check Socket.IO library loaded: `webapp/static/vendor/socket.io-4.7.5.min.js`
3. Check WebSocket proxy settings in Azure if using Internal Load Balancer

---

## File Verification for Deployment

Before pushing to Azure, verify these files are committed:

```bash
# Check files exist in git
git ls-files webapp/static/vendor/ | grep -E "bootstrap|chart|socket"

# Should show:
webapp/static/vendor/bootstrap.min.css
webapp/static/vendor/bootstrap-5.3.8.min.css
webapp/static/vendor/bootstrap-5.3.8.min.css.map
webapp/static/vendor/bootstrap.bundle.min.js
webapp/static/vendor/bootstrap-5.3.8.bundle.min.js
webapp/static/vendor/bootstrap-5.3.8.bundle.min.js.map
webapp/static/vendor/chart.umd.js
webapp/static/vendor/chart.umd.js.map
webapp/static/vendor/socket.io-4.7.5.min.js
webapp/static/vendor/socket.io-4.7.5.min.js.map
```

---

## Rollback Plan

If STRICT CSP causes issues:

1. **Immediate**: Set `CSP_MODE=RELAXED` in Azure Configuration
2. **Restart**: Click **Restart** button on App Service overview
3. **Wait**: Allow 30-60 seconds for restart complete
4. **Verify**: Test application functionality
5. **Diagnose**: Check logs for root cause before returning to STRICT

---

## Security Audit Checklist

For periodic security audits:

- [ ] `CSP_MODE=STRICT` is set in Azure
- [ ] No `RELAXED` mode in production
- [ ] All vendor files are current versions (check dates)
- [ ] No CSP policy headers mention external CDNs
- [ ] Nonces are regenerated per request (not cached)
- [ ] HTTPS only (verified in Azure)
- [ ] No inline styles in templates (`style-src-attr 'none'`)
- [ ] No inline scripts (all use nonces)

---

## References

- **CSP Documentation**: See [docs/DEPLOYMENT/CSP_SECURITY_MODEL.md](CSP_SECURITY_MODEL.md)
- **Azure App Service Docs**: <https://learn.microsoft.com/en-us/azure/app-service/>
- **Bootstrap v5.3.8**: <https://getbootstrap.com/docs/5.3/>
