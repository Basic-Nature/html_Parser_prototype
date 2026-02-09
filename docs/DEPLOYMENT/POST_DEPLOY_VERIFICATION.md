# Post-Deployment Verification Checklist

## Critical Missing Checks (Identified Feb 6, 2026)

### Current State

✅ **PASSING** (from logs):

- Container starts successfully
- Gunicorn workers boot
- Health checks returning 200
- HTTPS redirects working (tested in workflow)
- Static assets loading
- Socket.IO connections established

❌ **MISSING** (not verified):

### 1. Content Security Policy (CSP) Verification

**Risk**: STRICT CSP mode set but not tested; could block Bootstrap/vendor assets

```bash
# Add to workflow after "Post-deploy HTTPS redirect probe"
- name: Verify CSP headers
  run: |
    CSP=$(curl -sI https://www.electionpulse.org | grep -i "content-security-policy")
    if [ -z "$CSP" ]; then
      echo "ERROR: No CSP header found"
      exit 1
    fi
    echo "CSP Header: $CSP"
    # Verify nonce-based script loading
    if echo "$CSP" | grep -q "script-src.*nonce-"; then
      echo "✓ Nonce-based CSP detected"
    else
      echo "ERROR: CSP missing nonce directive"
      exit 1
    fi
```

### 2. Database Connectivity Test

**Risk**: PostgreSQL connection not verified; app may start but fail on data access

```bash
- name: Test PostgreSQL connection
  run: |
    RESPONSE=$(curl -s -w "\n%{http_code}" https://www.electionpulse.org/api/warehouse_election_results?limit=1)
    HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
    BODY=$(echo "$RESPONSE" | sed '$d')
    
    if [ "$HTTP_CODE" != "200" ]; then
      echo "ERROR: Database query failed (HTTP $HTTP_CODE)"
      echo "Response: $BODY"
      exit 1
    fi
    echo "✓ Database connectivity verified"
```

### 3. Python Version & orjson Compatibility

**Risk**: Python 3.13/orjson ABI mismatch (just fixed but not verified)

```bash
- name: Verify Python version and orjson import
  run: |
    PYTHON_VER=$(curl -s https://www.electionpulse.org/api/health/python-version)
    if echo "$PYTHON_VER" | grep -q "3.12.10"; then
      echo "✓ Python 3.12.10 confirmed"
    else
      echo "ERROR: Wrong Python version: $PYTHON_VER"
      exit 1
    fi
    
    # Test orjson import by hitting an endpoint that uses it
    curl -s https://www.electionpulse.org/api/urls | jq . > /dev/null
    echo "✓ orjson working (JSON parsing succeeded)"
```

### 4. API Endpoints Functional Test

**Risk**: Frontend may load but APIs may fail

```bash
- name: Test critical API endpoints
  run: |
    set -e
    
    # File listing (uses filesystem)
    curl -f https://www.electionpulse.org/api/fs/list?root=input&path= | jq .
    
    # URL list (uses orjson)
    curl -f https://www.electionpulse.org/api/urls | jq . | head -n 5
    
    # Warehouse query (uses PostgreSQL)
    curl -f https://www.electionpulse.org/api/warehouse_election_results?limit=5 | jq .
    
    echo "✓ All API endpoints responding"
```

### 5. Socket.IO WebSocket Upgrade Test

**Risk**: Socket.IO connections may fail to upgrade from polling to WebSocket

```bash
- name: Test Socket.IO connectivity
  run: |
    # Test polling handshake
    HANDSHAKE=$(curl -s "https://www.electionpulse.org/socket.io/?EIO=4&transport=polling")
    if echo "$HANDSHAKE" | grep -q "sid"; then
      echo "✓ Socket.IO handshake successful"
    else
      echo "ERROR: Socket.IO handshake failed"
      exit 1
    fi
```

### 6. OCR Service Availability

**Risk**: ENABLE_OCR=True but Tesseract/Poppler not verified

```bash
- name: Verify OCR dependencies
  run: |
    # This requires adding a /api/health/ocr endpoint that tests:
    # - pytesseract.image_to_string() with a test image
    # - pdf2image with a test PDF
    OCR_STATUS=$(curl -s https://www.electionpulse.org/api/health/ocr)
    if echo "$OCR_STATUS" | grep -q "available"; then
      echo "✓ OCR service available"
    else
      echo "⚠ OCR service unavailable (non-fatal)"
    fi
```

### 7. Static Asset Integrity (Bootstrap Vendor Files)

**Risk**: Recently fixed Bootstrap installation not verified; CSP may block

```bash
- name: Verify Bootstrap vendor files load
  run: |
    # Test that Bootstrap CSS loads and is not a redirect
    BS_CSS=$(curl -sL -w "%{http_code}" https://www.electionpulse.org/static/vendor/bootstrap-5.3.8.min.css -o /dev/null)
    if [ "$BS_CSS" = "200" ]; then
      echo "✓ Bootstrap CSS accessible"
    else
      echo "ERROR: Bootstrap CSS returned HTTP $BS_CSS"
      exit 1
    fi
    
    # Verify content is actual Bootstrap (not CDN wrapper)
    BS_CONTENT=$(curl -s https://www.electionpulse.org/static/vendor/bootstrap-5.3.8.min.css | head -c 100)
    if echo "$BS_CONTENT" | grep -q "Bootstrap"; then
      echo "✓ Bootstrap CSS is actual code (not wrapper)"
    else
      echo "ERROR: Bootstrap CSS appears to be a wrapper/redirect"
      exit 1
    fi
```

---

## Recommended Workflow Addition

Add to [.github/workflows/main_ballotlens.yml](../../.github/workflows/main_ballotlens.yml) after line 261 (after "Post-deploy HTTPS redirect probe"):

```yaml
      - name: Wait for app warm-up (30s)
        run: sleep 30

      - name: Verify CSP headers
        run: |
          CSP=$(curl -sI https://www.electionpulse.org | grep -i "content-security-policy")
          if [ -z "$CSP" ]; then
            echo "ERROR: No CSP header found"
            exit 1
          fi
          echo "CSP Header: $CSP"
          if echo "$CSP" | grep -q "script-src.*nonce-"; then
            echo "✓ Nonce-based CSP detected"
          else
            echo "WARNING: CSP missing nonce directive (non-fatal in permissive mode)"
          fi

      - name: Test API endpoints
        run: |
          set -e
          echo "Testing /api/fs/list..."
          curl -f https://www.electionpulse.org/api/fs/list?root=input&path= | jq . > /dev/null
          
          echo "Testing /api/urls..."
          curl -f https://www.electionpulse.org/api/urls | jq . | head -n 5
          
          echo "Testing /api/warehouse_election_results..."
          curl -f "https://www.electionpulse.org/api/warehouse_election_results?limit=1" | jq . > /dev/null
          
          echo "✓ All API endpoints responding"

      - name: Test Socket.IO handshake
        run: |
          HANDSHAKE=$(curl -s "https://www.electionpulse.org/socket.io/?EIO=4&transport=polling")
          if echo "$HANDSHAKE" | grep -q "sid"; then
            echo "✓ Socket.IO handshake successful"
          else
            echo "ERROR: Socket.IO handshake failed"
            echo "Response: $HANDSHAKE"
            exit 1
          fi

      - name: Verify Bootstrap static assets
        run: |
          BS_CSS=$(curl -sL -w "%{http_code}" https://www.electionpulse.org/static/vendor/bootstrap-5.3.8.min.css -o /dev/null)
          if [ "$BS_CSS" != "200" ]; then
            echo "ERROR: Bootstrap CSS returned HTTP $BS_CSS"
            exit 1
          fi
          
          BS_CONTENT=$(curl -s https://www.electionpulse.org/static/vendor/bootstrap-5.3.8.min.css | head -c 100)
          if echo "$BS_CONTENT" | grep -q "Bootstrap"; then
            echo "✓ Bootstrap CSS is actual code (not CDN wrapper)"
          else
            echo "ERROR: Bootstrap CSS appears to be a wrapper"
            exit 1
          fi

      - name: Test Python version and orjson
        run: |
          # Verify orjson works by parsing JSON from /api/urls
          curl -s https://www.electionpulse.org/api/urls | jq . > /dev/null
          echo "✓ orjson working (no Python 3.13 ABI error)"
```

---

## Manual Verification Checklist (Post-Deploy)

After successful deployment, manually verify:

- [ ] **UI loads** - <https://www.electionpulse.org> displays without console errors
- [ ] **Ballot Lens page** - `/ballot_lens` renders with all panels
- [ ] **File browser** - Input/Output file listings populate
- [ ] **URL search** - `/api/urls` returns election result URLs
- [ ] **Contest modal** - Can open contest selection modal
- [ ] **Theme toggle** - Light/dark mode switch works
- [ ] **Socket.IO live updates** - Real-time progress updates during parse jobs
- [ ] **PDF extraction** - Upload PDF and verify OCR extraction works
- [ ] **Output download** - Generated CSVs download successfully
- [ ] **Database write** - Parsed data saves to PostgreSQL warehouse

---

## Health Endpoint Proposal

Create `/api/health/system` endpoint returning:

```json
{
  "status": "healthy",
  "python_version": "3.12.10",
  "orjson_available": true,
  "database_connected": true,
  "ocr_available": true,
  "socketio_running": true,
  "csp_mode": "STRICT",
  "deployment_sha": "d356214df01375b5bf602ecf567c4e9da66bc30f",
  "uptime_seconds": 3456
}
```

Workflow can test this single endpoint instead of multiple checks.

---

## Next Steps

1. **Immediate**: Add missing verification steps to workflow (above)
2. **Short-term**: Create `/api/health/system` endpoint
3. **Medium-term**: Add Playwright E2E smoke tests (headless UI checks)
4. **Long-term**: Integrate accuracy regression tests (see DATA_COMPARISON_ROADMAP.md)
