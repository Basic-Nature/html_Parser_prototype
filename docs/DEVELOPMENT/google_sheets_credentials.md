# Google Sheets Credential Loading System

## Overview

The Google Sheets client now supports flexible credential loading optimized for both **Azure deployments** (individual environment variables) and **local development** (JSON file). The system tries multiple sources in priority order to find valid credentials.

**Status:** ✅ Implemented & Tested

---

## Credential Loading Priority Chain

The `GoogleSheetsElectionClient` attempts to load credentials in this order:

### Priority 1: Explicit Parameter (Highest Priority)

- **When:** `credentials_json` parameter passed to constructor
- **Format:** File path or JSON string
- **Use Case:** Programmatic credential injection
- **Example:**

  ```python
  client = GoogleSheetsElectionClient(
      credentials_json="/path/to/service_account.json",
      spreadsheet_id="sheet-id-123"
  )
  ```

### Priority 2: Individual Environment Variables (Recommended for Azure)

- **When:** All 11 `GOOGLE_SHEETS_SA_*` variables are set
- **Environment Variables:**
  - `GOOGLE_SHEETS_SA_TYPE`
  - `GOOGLE_SHEETS_SA_PROJECT_ID`
  - `GOOGLE_SHEETS_SA_PRIVATE_KEY_ID`
  - `GOOGLE_SHEETS_SA_PRIVATE_KEY`
  - `GOOGLE_SHEETS_SA_CLIENT_EMAIL`
  - `GOOGLE_SHEETS_SA_CLIENT_ID`
  - `GOOGLE_SHEETS_SA_AUTH_URI`
  - `GOOGLE_SHEETS_SA_TOKEN_URI`
  - `GOOGLE_SHEETS_SA_AUTH_PROVIDER_CERT_URL`
  - `GOOGLE_SHEETS_SA_CLIENT_CERT_URL`
  - `GOOGLE_SHEETS_SA_UNIVERSE_DOMAIN`

- **Description:** Recommended for Azure Key Vault integration where secrets are stored individually
- **Use Case:** CI/CD pipelines, Azure Container Instances, secure multi-tenant deployments
- **Advantage:** No single large credential string to manage; integrates seamlessly with Azure Key Vault

### Priority 3: Legacy JSON String Environment Variable

- **When:** `GOOGLE_SHEETS_ELECTION_DB_LITE_CREDENTIALS` environment variable contains valid JSON
- **Format:** Complete service account JSON as a string
- **Description:** Legacy approach, kept for backward compatibility
- **Use Case:** Existing deployments that use this variable
- **⚠️ Note:** Not recommended for new deployments; use Priority 2 or 4 instead

### Priority 4: GCP Standard Environment Variable (Recommended for Local Dev)

- **When:** `GOOGLE_APPLICATION_CREDENTIALS` environment variable points to a valid JSON file
- **Format:** File path (absolute or relative)
- **Description:** GCP-standard approach; aligns with `gcloud` CLI and standard tooling
- **Use Case:** Local development with file-based credentials
- **Example in .env:**

  ```txt
  GOOGLE_APPLICATION_CREDENTIALS=google_service_account.json
  ```

### Priority 5: Project Root JSON File (Local Dev Convenience)

- **When:** `google_service_account.json` exists in project root
- **Format:** Service account JSON file
- **Description:** Automatic fallback for local development
- **Use Case:** Fastest setup for local testing; works without any environment variables
- **⚠️ Note:** .gitignore prevents accidental commits

---

## Setup Instructions

### Option A: Local Development (Recommended for Local)

**Quickest Setup - No Environment Variables Needed:**

```bash
# 1. Place credentials file in project root
cp /path/to/service_account.json ./google_service_account.json

# 2. Add required environment variables to .env
echo 'GOOGLE_SHEETS_DB_LITE_ID=your-spreadsheet-id' >> .env
echo 'GOOGLE_SHEETS_WORKLIST_ID=your-worklist-id' >> .env

# 3. That's it! The client will auto-detect the JSON file
```

**Or, if you prefer explicit GOOGLE_APPLICATION_CREDENTIALS:**

```bash
echo 'GOOGLE_APPLICATION_CREDENTIALS=google_service_account.json' >> .env
echo 'GOOGLE_SHEETS_DB_LITE_ID=your-spreadsheet-id' >> .env
```

---

### Option B: Azure Deployment (Recommended for Azure)

**Setup with Individual Environment Variables (via Azure Key Vault):**

```bash
# 1. Add all 11 credential variables to Azure Key Vault
# (Or set in .env for local testing of this approach)

export GOOGLE_SHEETS_SA_TYPE='service_account'
export GOOGLE_SHEETS_SA_PROJECT_ID='your-project-id'
export GOOGLE_SHEETS_SA_PRIVATE_KEY_ID='your-key-id'
export GOOGLE_SHEETS_SA_PRIVATE_KEY='-----BEGIN RSA PRIVATE KEY-----\n...\n-----END RSA PRIVATE KEY-----'
export GOOGLE_SHEETS_SA_CLIENT_EMAIL='your-email@project.iam.gserviceaccount.com'
export GOOGLE_SHEETS_SA_CLIENT_ID='123456789'
export GOOGLE_SHEETS_SA_AUTH_URI='https://accounts.google.com/o/oauth2/auth'
export GOOGLE_SHEETS_SA_TOKEN_URI='https://oauth2.googleapis.com/token'
export GOOGLE_SHEETS_SA_AUTH_PROVIDER_CERT_URL='https://www.googleapis.com/oauth2/v1/certs'
export GOOGLE_SHEETS_SA_CLIENT_CERT_URL='https://www.googleapis.com/robot/v1/metadata/x509/...'
export GOOGLE_SHEETS_SA_UNIVERSE_DOMAIN='googleapis.com'

# 2. Add spreadsheet IDs
export GOOGLE_SHEETS_DB_LITE_ID='your-spreadsheet-id'
export GOOGLE_SHEETS_WORKLIST_ID='your-worklist-id'
```

**In Azure Key Vault:**

- Create secrets for each `GOOGLE_SHEETS_SA_*` variable
- Azure automatically injects them into the application environment
- No files to manage; secrets are encrypted and audited

---

## Implementation Details

### File: `webapp/parser/data_standardization/google_sheets_client.py`

**Key Functions:**

#### `_build_service_account_json_from_env()`

- Constructs service account JSON dict from individual env vars
- Used in Priority 2 (Azure approach)
- Returns: `Dict[str, Any]` or `None`

#### `_load_credentials_from_file(file_path: str)`

- Loads and parses JSON from file path
- Used in Priority 4 & 5 (local dev approaches)
- Returns: `Dict[str, Any]` or `None`
- Handles file not found and JSON decode errors gracefully

#### `GoogleSheetsElectionClient.__init__()`

- Implements the complete priority chain
- Validates that credentials were found before initialization
- Validates that spreadsheet ID is configured
- Provides detailed error messages with setup instructions

---

## Error Handling

### ValueError: "Google Sheets credentials not configured"

**Causes:**

- None of the 5 credential sources are available
- Spreadsheet ID not set

**Solution Message Includes:**

1. How to set up individual env vars (Azure)
2. How to set up JSON file (Local)
3. How to set up legacy JSON string (Legacy)

### Example Error Response (Flask Endpoints)

```json
{
  "success": false,
  "error": "Google Sheets access not configured",
  "detail": "Google Sheets credentials not configured. Configure one of:\n\n  OPTION 1 (Azure Recommended)...\n\nFor local development, you can use: GOOGLE_APPLICATION_CREDENTIALS=/path/to/google_service_account.json\nFor Azure, configure individual GOOGLE_SHEETS_SA_* environment variables."
}
```

---

## Testing

### Test Script: `webapp/tests/test_google_sheets_credentials_contract.py`

Demonstrates:

1. Credential not found → ValueError
2. JSON file loading → Success
3. Individual env vars → Success

**Run:**

```bash
python webapp/tests/test_google_sheets_credentials_contract.py
```

---

## Spreadsheet Configuration

### Required Environment Variables (Separate from Credentials)

```env
# Database spreadsheet (DB-Lite)
GOOGLE_SHEETS_DB_LITE_ID=<your-spreadsheet-id>

# Worklist spreadsheet (if using worklist features)
GOOGLE_SHEETS_WORKLIST_ID=<your-spreadsheet-id>

# Sheet name within worklist (optional, defaults to "Overview")
GOOGLE_SHEETS_WORKLIST_OVERVIEW_SHEET=Overview
```

---

## Migration Guide

### From Legacy to New System

**Old Way (Still Supported):**

```env
GOOGLE_SHEETS_ELECTION_DB_LITE_CREDENTIALS='{"type":"service_account",...}'
```

**New Way (For Local Dev):**

```env
GOOGLE_APPLICATION_CREDENTIALS=google_service_account.json
GOOGLE_SHEETS_DB_LITE_ID=spreadsheet-id
```

**New Way (For Azure):**

```env
GOOGLE_SHEETS_SA_TYPE=service_account
GOOGLE_SHEETS_SA_PROJECT_ID=project-id
# ... (other 9 variables)
GOOGLE_SHEETS_DB_LITE_ID=spreadsheet-id
```

**No code changes required** - the client automatically detects the approach being used.

---

## Logs & Debugging

### Enable Credential Loading Logs

Set in environment:

```env
LOG_LEVEL=DEBUG
```

**Expected Log Output (Successful Load):**

```txt
✓ Using credentials from individual env vars (Azure recommended)
✓ Google Sheets authentication successful (from env vars)
```

Or:

```txt
Using credentials from GOOGLE_APPLICATION_CREDENTIALS: google_service_account.json
✓ Google Sheets authentication successful (from file: google_service_account.json)
```

### Verify Credentials Are Loaded

```python
import os

# Check which credentials are available
print("Individual Env Vars Set:", "GOOGLE_SHEETS_SA_TYPE" in os.environ)
print("GOOGLE_APPLICATION_CREDENTIALS:", os.getenv('GOOGLE_APPLICATION_CREDENTIALS', 'NOT SET'))
print("Legacy JSON Var Set:", "GOOGLE_SHEETS_ELECTION_DB_LITE_CREDENTIALS" in os.environ)
print("JSON File Exists:", os.path.exists("google_service_account.json"))
```

---

## Deployment Recommendations

### Local Development

```txt
Recommended: Priority 4 & 5 (JSON file approaches)
Setup Time: <1 minute
Complexity: Low
Security: Good (file in .gitignore)
```

### CI/CD (GitHub Actions, Azure Pipelines)

```txt
Recommended: Priority 2 (Individual env vars + secrets manager)
Setup Time: <5 minutes
Complexity: Medium
Security: Excellent (encrypted secret storage)
```

### Production

```txt
Recommended: Priority 2 (Individual env vars + Key Vault)
Setup Time: <15 minutes
Complexity: Medium
Security: Excellent (audited, rotated, encrypted)
```

---

## FAQ

**Q: Which approach should I use?**

- **Local Dev:** JSON file (Priority 4/5) - Simplest
- **Azure:** Individual env vars (Priority 2) - Integrates with Key Vault
- **Legacy System:** JSON string (Priority 3) - For existing deployments

**Q: What if multiple credential sources are configured?**

- The system uses the first one found in priority order
- No conflicts; highest priority wins

**Q: Can I use GOOGLE_APPLICATION_CREDENTIALS on Azure?**

- Yes! But individual env vars (Priority 2) are recommended for Key Vault integration

**Q: What happens if credentials are invalid or expired?**

- You'll get `GoogleAuthError` during service initialization
- Flask endpoints return 503 Service Unavailable
- Client code can catch `ValueError` and handle gracefully

**Q: Is google_service_account.json ever committed to Git?**

- No - it's in .gitignore
- Safe to commit; malicious users can't access secrets in .gitignore'd files

---

## Version History

| Version | Date | Changes |
| --------- | ------ | --------- |
| 2.0 | Jan 2026 | Added Priority 4 & 5, improved error messages, added test script |
| 1.0 | - | Original two-priority system (individual vars + JSON string) |

---

## Related Files

- [google_sheets_client.py](../../webapp/parser/data_standardization/google_sheets_client.py) - Main implementation
- [.env.template](../../.env.template) - Environment template (lines 322-350)
- [Smart_Elections_Parser_Webapp.py](../../webapp/Smart_Elections_Parser_Webapp.py) - Flask endpoints (lines 6288-6430)
- [test_credential_loading.py](../../webapp/tests/test_google_sheets_credentials_contract.py) - Test script

---

## Support

For issues or questions about credential setup:

1. Check logs with `LOG_LEVEL=DEBUG`
2. Run credential test: `python webapp/tests/test_google_sheets_credentials_contract.py`
3. Verify env vars: `python -c "import os; [print(k) for k in os.environ if 'GOOGLE' in k]"`
