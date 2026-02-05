# Phase 2 Deployment Checklist – Quick Reference

**Status**: ✅ Phase 2 Code Complete (UI + Socket Integration)  
**Next Step**: Azure Deployment & PostgreSQL Setup

---

## 🚀 Quick Deployment

### Step 1: Authentication Setup

**⚠️ SECURITY NOTICE**: `QA_REQUIRE_CERT_AUTH` defaults to `true` (certificate authentication required).

**Option A: Production Setup** (Recommended):

- Follow the complete guide: **[AZURE_CERTIFICATE_AUTH_SETUP.md](AZURE_CERTIFICATE_AUTH_SETUP.md)**
- Configure Azure App Service client certificates
- Generate reviewer certificates
- Full audit trail with principal attribution

**Option B: Development/Testing Only** (Not for production):

- Azure Portal → App Service → Configuration → Application settings
- Add: `QA_REQUIRE_CERT_AUTH = false`
- Click **OK** → **Save** (app restarts automatically)
- ⚠️ **WARNING**: Disables authentication - all actions logged as `system:development`

### Step 2: Create PostgreSQL Tables (5 min)

1. Azure Portal → Your PostgreSQL Server → **Query Editor** (or use psql)
2. Connect to database: `verified_data` (create if doesn't exist)
3. Open `scripts/create_verified_data_schema.sql` from this repo
4. Copy & paste entire SQL script into Azure Query Editor
5. Click **Run** → Should see "5 tables created" message
6. Verify: Run `SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';`

### Step 3: Test QA Panels on Azure (5 min)

1. Navigate to: `https://<your-app>.azurewebsites.net/ballot_lens`
2. Parse any test URL (e.g., California election results)
3. **Expected Behavior**:
   - Results appear with QA panels below each contest
   - DL1 badge shows (yellow "⚠ Unverified")
   - Confidence % displays (e.g., "85.3%")
   - Issues list shows detected problems (if any)
   - "✓ Promote to DL2" button appears
4. **Browser Console Check**: Should see `[QA Integration] Initialized successfully`

### Step 4: Test Promotion Workflow (5 min)

1. Click **"✓ Promote to DL2"** button on any result
2. Modal appears → Enter certification reason (e.g., "Manually verified source")
3. Click **Confirm**
4. **Expected**: Badge changes to green "✓ Verified" (DL2)
5. **Verify in PostgreSQL**:

   ```sql
   SELECT * FROM verification_lineage ORDER BY action_timestamp DESC LIMIT 5;
   SELECT dl_status, contest_name FROM verified_datasets ORDER BY created_at DESC LIMIT 5;
   ```

---

## 📝 What Just Got Deployed

**Phase 2 Complete** (Commit ready):

- ✅ Cert auth bypass for Azure (`QA_REQUIRE_CERT_AUTH=false`)
- ✅ QA Panel UI module (`quality_assurance_panel.js` - 502 lines)
- ✅ QA Panel styling (`quality_assurance_panel.css` - 363 lines)
- ✅ Socket integration layer (`quality_assurance_integration.js` - 450+ lines)
- ✅ TypeScript errors fixed (7 errors → 0 errors)
- ✅ PostgreSQL schema ready (`create_verified_data_schema.sql`)

**What's Working Now**:

- 🟢 Auto-classification when parsing completes (via socket events)
- 🟢 QA panels inject into result cards automatically
- 🟢 DL1/DL2 badge rendering with confidence scores
- 🟢 Issue detection (5 types: missing headers, low confidence, etc.)
- 🟢 Promotion workflow (DL1 → DL2 with audit trail)
- 🟢 Event delegation for dynamic promote buttons
- 🟢 Manual "🔍 Classify QA" trigger button (in results preview bar)

**Pending** (Your action required):

- ⏳ Set `QA_REQUIRE_CERT_AUTH=false` in Azure App Settings
- ⏳ Create PostgreSQL tables (run SQL script)
- ⏳ Test E2E workflow on live site

---

## 🔍 Troubleshooting

**Problem**: QA panels still not showing after Azure deployment

**Check**:

1. **Environment variable set?**

   ```bash
   # Azure Cloud Shell
   az webapp config appsettings list --name <app-name> --resource-group <rg-name> | grep QA_REQUIRE_CERT_AUTH
   # Should return: "name": "QA_REQUIRE_CERT_AUTH", "value": "false"
   ```

2. **App restarted?**

   ```bash
   az webapp restart --name <app-name> --resource-group <rg-name>
   ```

3. **Browser console errors?**
   - Open DevTools (F12) → Console tab
   - Look for `[QA Integration]` logs
   - Should see: `Initialized successfully`
   - Check for 401 errors in Network tab

4. **PostgreSQL tables created?**

   ```sql
   SELECT COUNT(*) FROM verified_datasets;  -- Should work (return 0 initially)
   ```

**Problem**: TypeScript errors in VS Code

**Fix**: Files are already fixed (7 errors resolved). If you see errors:

- Close and reopen VS Code
- Run: `npm run check-js` to verify
- Ensure `quality_assurance_panel.js` has latest changes from commit

**Problem**: Promote button not working

**Check**:

1. **Network tab**: POST to `/api/data-assurance/verify-and-promote` should return 200
2. **PostgreSQL connection**: Verify `VERIFIED_DATA_DB_*` env vars are set
3. **Reviewer principal**: Should see `system:development` in logs when cert auth disabled

---

## 📊 Verification Queries

After testing, run these in Azure PostgreSQL Query Editor:

```sql
-- 1. Count datasets by status
SELECT dl_status, COUNT(*) as count
FROM verified_datasets
GROUP BY dl_status;

-- 2. Recent promotions
SELECT dataset_id, action_type, reviewer_principal, action_timestamp
FROM verification_lineage
ORDER BY action_timestamp DESC
LIMIT 10;

-- 3. Unresolved QA issues
SELECT qi.issue_type, qi.severity, COUNT(*) as count
FROM quality_issues qi
WHERE qi.is_resolved = FALSE
GROUP BY qi.issue_type, qi.severity
ORDER BY count DESC;

-- 4. Datasets with high confidence
SELECT contest_name, state_abbr, extraction_confidence, trust_score
FROM verified_datasets
WHERE extraction_confidence > 90
  AND trust_score > 80
ORDER BY extracted_at DESC
LIMIT 20;
```

---

## 🎯 Success Criteria

**Phase 2 is complete when**:

- ✅ You can parse a URL on Azure and see QA panels below results
- ✅ DL1 badge appears with confidence % and issue list
- ✅ Clicking "Promote to DL2" changes badge to green "Verified"
- ✅ PostgreSQL `verification_lineage` table has an entry with `system:development` principal
- ✅ Browser console shows `[QA Integration] Initialized successfully`

---

## 🚦 Next Steps (Phase 3)

**After Phase 2 testing passes**:

1. **Data Framework Dashboard** (New page):
   - State/county search filters
   - DL1/DL2 dataset table with pagination
   - Export workflows (CSV/JSON downloads)
   - Geographic visualization (state/county maps)

2. **Production Certificate Auth** (Optional):
   - Set `QA_REQUIRE_CERT_AUTH=true`
   - Upload CA certificates to Azure App Service
   - Configure privilege tiers (county/state/federal reviewers)
   - See `docs/PHASE2_AZURE_DEPLOYMENT.md` "Production Mode" section

3. **Application Insights Monitoring**:
   - Track QA API latency
   - Monitor promotion patterns
   - Alert on low-confidence datasets
   - Dashboard for data quality trends

---

## 📚 Reference Docs

- **🔐 Certificate Setup**: `docs/AZURE_CERTIFICATE_AUTH_SETUP.md` ⭐ (production required)
- **Quick Start**: `docs/PHASE2_QUICK_FIX.md` (5-minute guide - development mode)
- **Full Deployment**: `docs/PHASE2_AZURE_DEPLOYMENT.md` (comprehensive)
- **PostgreSQL Schema**: `docs/VERIFIED_DATA_SCHEMA.md` (design docs)
- **SQL Script**: `scripts/create_verified_data_schema.sql` (ready to run)
- **Phase 1 Backend**: Git commit `0b51394`
- **Phase 2a UI + Cert Fix**: Git commit `2ca7420`
- **Phase 2b Socket Integration**: Current working state (ready to commit)

---

## ✅ Commit Phase 2b

After testing, commit the socket integration:

```bash
git add .
git commit -m "feat(qa): Phase 2b - Socket integration layer

- quality_assurance_integration.js (450+ lines)
- Hooks into run_summary socket event
- Auto-classifies results after parsing
- Event delegation for promote buttons
- Manual 'Classify QA' trigger button
- Staggered API calls (150ms delay)
- Fixed 7 TypeScript errors in quality_assurance_panel.js
- Created PostgreSQL schema SQL script

Status: Phase 2 Complete (Backend + UI + Integration)
Next: Test E2E on Azure, verify promotion workflow"

git push
```

---

**Questions?** See troubleshooting section above or review `docs/PHASE2_AZURE_DEPLOYMENT.md` for comprehensive guide.
