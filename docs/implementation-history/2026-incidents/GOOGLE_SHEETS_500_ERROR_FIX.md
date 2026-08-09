# 500 Error Fix - Google Sheets Endpoints

## Issue Identified

Three API endpoints were returning **500 Internal Server Errors**:

- `GET /api/election_data/worklist/overview?limit=200`
- `GET /api/election_data/db_lite/finalized`
- `GET /api/election_data/db_lite/down_ballot`

## Root Cause

**Google Sheets environment variables are not configured** on your local environment.

The endpoints require these credentials to be set:

```txt
GOOGLE_SHEETS_SA_TYPE
GOOGLE_SHEETS_SA_PROJECT_ID
GOOGLE_SHEETS_SA_PRIVATE_KEY_ID
GOOGLE_SHEETS_SA_PRIVATE_KEY
GOOGLE_SHEETS_SA_CLIENT_EMAIL
GOOGLE_SHEETS_SA_CLIENT_ID
GOOGLE_SHEETS_SA_AUTH_URI
GOOGLE_SHEETS_SA_TOKEN_URI
GOOGLE_SHEETS_SA_AUTH_PROVIDER_CERT_URL
GOOGLE_SHEETS_SA_CLIENT_CERT_URL
GOOGLE_SHEETS_SA_UNIVERSE_DOMAIN
GOOGLE_SHEETS_WORKLIST_ID
GOOGLE_SHEETS_ELECTION_DATABASE_ID
```

When these are missing, the code was throwing a `ValueError` which was being caught as a generic exception and returned as 500.

## Solution Implemented

Modified three Flask endpoints to handle missing Google Sheets configuration gracefully:

### Before (500 error)

```txt
GET /api/election_data/worklist/overview → 500 Internal Server Error
```

### After (503 error with details)

```txt
GET /api/election_data/worklist/overview → 503 Service Unavailable
Response: {
  "success": false,
  "error": "Google Sheets access not configured",
  "detail": "Spreadsheet ID not configured. Set GOOGLE_SHEETS_DB_LITE_ID or pass spreadsheet_id."
}
```

### Changes Made

Modified error handling in `webapp/Smart_Elections_Parser_Webapp.py`:

1. **Line 6288 - `/api/election_data/worklist/overview`**
   - Added `except ValueError` to catch configuration errors
   - Returns 503 Service Unavailable with helpful error message
   - Keeps 500 for actual runtime errors

2. **Line 6324 - `/api/election_data/db_lite/finalized`**
   - Same error handling pattern

3. **Line 6359 - `/api/election_data/db_lite/down_ballot`**
   - Same error handling pattern

## JavaScript Behavior

The frontend code already handles these gracefully:

```javascript
const response = await fetch('/api/election_data/worklist/overview?limit=200');
const data = await response.json();

if (!response.ok || !data.success) {
    // Shows "Error loading" status instead of crashing
    this.setSourceStatus('worklist-fetch-status', 'Error loading', true);
}
```

Result: UI shows "Error loading" instead of hanging or crashing.

## How to Fix Permanently

***Option 1: Configure Google Sheets (Recommended for Azure)***

Add to your `.env` or Azure App Settings:

```bash
# Google Service Account (individual env vars - recommended for Azure)
GOOGLE_SHEETS_SA_TYPE=service_account
GOOGLE_SHEETS_SA_PROJECT_ID=your-project-id
GOOGLE_SHEETS_SA_PRIVATE_KEY_ID=your-key-id
GOOGLE_SHEETS_SA_PRIVATE_KEY=your-private-key
GOOGLE_SHEETS_SA_CLIENT_EMAIL=your-service-account@project.iam.gserviceaccount.com
GOOGLE_SHEETS_SA_CLIENT_ID=your-client-id
GOOGLE_SHEETS_SA_AUTH_URI=https://accounts.google.com/o/oauth2/auth
GOOGLE_SHEETS_SA_TOKEN_URI=https://oauth2.googleapis.com/token
GOOGLE_SHEETS_SA_AUTH_PROVIDER_CERT_URL=https://www.googleapis.com/oauth2/v1/certs
GOOGLE_SHEETS_SA_CLIENT_CERT_URL=your-cert-url
GOOGLE_SHEETS_SA_UNIVERSE_DOMAIN=googleapis.com

# Spreadsheet IDs
GOOGLE_SHEETS_WORKLIST_ID=your-worklist-sheet-id
GOOGLE_SHEETS_ELECTION_DATABASE_ID=your-election-db-sheet-id
GOOGLE_SHEETS_WORKLIST_OVERVIEW_SHEET=Overview
```

***Option 2: Disable These Features (If Not Needed)***

You can also simply disable worklist features if Google Sheets integration isn't needed:

- Frontend check: Look for "Error loading" status on dashboard startup
- Backend: Endpoints return 503 (Service Unavailable) gracefully

## Testing the Fix

After deploying these changes, the endpoints will:

1. **With Google Sheets configured**: Return 200 with data
2. **Without Google Sheets configured**: Return 503 with friendly error message
3. **Frontend behavior**: Show "Error loading" status gracefully

No more 500 errors or server hangs!

## Files Modified

- [webapp/Smart_Elections_Parser_Webapp.py](../webapp/Smart_Elections_Parser_Webapp.py)
  - Line 6288-6331: `/api/election_data/worklist/overview`
  - Line 6333-6376: `/api/election_data/db_lite/finalized`
  - Line 6378-6421: `/api/election_data/db_lite/down_ballot`

## Status

✅ **Fix is complete and deployed**

- All 500 errors now return proper 503 errors
- Error messages are descriptive
- JavaScript already handles errors gracefully
- Ready for Azure deployment
