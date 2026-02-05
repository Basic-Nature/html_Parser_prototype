# Azure Certificate Authentication Setup

**Security Notice**: As of Phase 2, `QA_REQUIRE_CERT_AUTH` defaults to `true` (certificate authentication required). Local development requires explicitly setting `QA_REQUIRE_CERT_AUTH=false`.

## Production Setup (Azure App Service)

### Prerequisites

- Azure App Service (Standard tier or higher for client certificates)
- OpenSSL or Azure Cloud Shell
- Admin access to Azure App Service

---

## Step 1: Enable Client Certificates on App Service

### Via Azure Portal

1. Navigate to **Azure Portal** → **App Services** → **Your App**
2. Go to **Configuration** → **General settings**
3. Under **Client certificate mode**, select:
   - **Require** (strict mode - best for production)
   - **Optional** (allows mixed auth - useful for migration)
4. Click **Save**

### Via Azure CLI

```bash
az webapp update \
  --resource-group <resource-group> \
  --name <app-name> \
  --set clientCertEnabled=true \
  --set clientCertMode=Required
```

---

## Step 2: Configure Certificate Header Forwarding

By default, Azure App Service forwards client certificates via the `X-ARR-ClientCert` header.

### Verify Header Forwarding

Add this test endpoint to verify the header is being forwarded:

```python
@app.route('/api/test-cert-header', methods=['GET'])
def test_cert_header():
    cert_header = request.headers.get('X-ARR-ClientCert')
    return {
        'has_cert_header': bool(cert_header),
        'header_length': len(cert_header) if cert_header else 0,
        'all_headers': dict(request.headers)
    }
```

Navigate to: `https://your-app.azurewebsites.net/api/test-cert-header`

**Expected Response** (when client cert provided):

```json
{
  "has_cert_header": true,
  "header_length": 1234,
  "all_headers": {
    "X-ARR-ClientCert": "-----BEGIN CERTIFICATE-----\nMIID...",
    ...
  }
}
```

---

## Step 3: Generate Client Certificates for Reviewers

### Option A: Self-Signed Certificates (Development)

**Generate CA Certificate** (Certificate Authority):

```bash
# Create CA private key
openssl genrsa -out ca-key.pem 4096

# Create CA certificate (valid 10 years)
openssl req -new -x509 -days 3650 -key ca-key.pem -out ca-cert.pem \
  -subj "/C=US/ST=Arizona/L=Tucson/O=Election Integrity/CN=Ballot Lens CA"
```

**Generate Client Certificate** (for each reviewer):

```bash
# Create client private key
openssl genrsa -out client-key.pem 4096

# Create certificate signing request
openssl req -new -key client-key.pem -out client-csr.pem \
  -subj "/C=US/ST=Arizona/L=Tucson/O=Election Integrity/CN=John Doe/emailAddress=john.doe@example.gov"

# Sign client certificate with CA (valid 1 year)
openssl x509 -req -days 365 -in client-csr.pem \
  -CA ca-cert.pem -CAkey ca-key.pem -CAcreateserial \
  -out client-cert.pem

# Package client cert + key into PKCS#12 (for browser import)
openssl pkcs12 -export -out client-cert.p12 \
  -inkey client-key.pem -in client-cert.pem \
  -certfile ca-cert.pem -name "John Doe - Ballot Lens Reviewer"
```

**Distribute to Reviewer**:

- Send `client-cert.p12` file securely (encrypted email/secure file share)
- Provide password used during PKCS#12 export
- Reviewer imports into browser (Chrome: Settings → Privacy → Manage certificates)

### Option B: Enterprise PKI (Production)

Use your organization's existing PKI infrastructure:

1. Request client certificates from IT/Security team
2. Specify required certificate fields:
   - **Subject CN**: Reviewer's full name
   - **Subject Email**: Official email address
   - **Extended Key Usage**: Client Authentication (1.3.6.1.5.5.7.3.2)
   - **Key Usage**: Digital Signature, Key Encipherment
3. Import issued certificates into Azure App Service (see Step 4)

---

## Step 4: Upload CA Certificate to Azure App Service

Azure needs the CA certificate to validate client certificates.

### Via Azure Portal

1. Go to **App Service** → **TLS/SSL settings** → **Private Key Certificates (.pfx)**
2. Click **+ Create App Service Managed Certificate**
3. Upload `ca-cert.pem` (convert to .pfx if needed):

   ```bash
   openssl pkcs12 -export -out ca-cert.pfx -nokeys -in ca-cert.pem
   ```

4. Note the certificate **Thumbprint** (SHA-1 hash)

### Via Azure CLI

```bash
az webapp config ssl upload \
  --resource-group <resource-group> \
  --name <app-name> \
  --certificate-file ca-cert.pfx \
  --certificate-password <password>
```

---

## Step 5: Configure App Service to Trust CA

### Add Application Setting

**Azure Portal** → **App Service** → **Configuration** → **Application settings**

Add:

```txt
Name:  WEBSITE_LOAD_CERTIFICATES
Value: *
```

This tells Azure to load all uploaded certificates into the app's trust store.

### Verify Certificate Validation

The `@_require_reviewer` decorator automatically validates certificates using the uploaded CA.

Test by making an API call with and without a valid client cert:

```bash
# Without cert (should fail with 401)
curl https://your-app.azurewebsites.net/api/data-assurance/classify

# With cert (should succeed)
curl --cert client-cert.pem --key client-key.pem \
  https://your-app.azurewebsites.net/api/data-assurance/classify
```

---

## Step 6: Configure Privilege Tiers (Optional)

If using role-based access control, configure privilege tiers in `privilege_tiers.py`:

```python
PRIVILEGE_TIERS = {
    "TIER_1_COUNTY": {
        "allowed_actions": ["classify", "flag_issues"],
        "certificate_ou": "County Clerk",
        "max_promotion_level": "DL1",
    },
    "TIER_2_STATE": {
        "allowed_actions": ["classify", "promote", "verify"],
        "certificate_ou": "Secretary of State",
        "max_promotion_level": "DL2",
    },
    "TIER_3_FEDERAL": {
        "allowed_actions": ["classify", "promote", "verify", "audit"],
        "certificate_ou": "Federal Election Commission",
        "max_promotion_level": "DL2",
    },
}
```

Map certificate attributes to privilege levels in the decorator.

---

## Local Development Setup

For local development without certificates:

### Option 1: Disable Cert Auth (Development Only)

Add to `.env`:

```bash
QA_REQUIRE_CERT_AUTH=false
```

**⚠️ WARNING**: Never deploy to production with this setting. Development mode logs all actions under `system:development` principal with no audit trail.

### Option 2: Use Self-Signed Certs Locally

1. Generate self-signed certs (see Step 3 Option A)
2. Import `client-cert.p12` into browser
3. Configure Flask to accept client certs:

   ```python
   app.run(ssl_context=('server-cert.pem', 'server-key.pem'))
   ```

4. Access via `https://localhost:5000`

---

## Troubleshooting

### Issue: "401 Unauthorized" on QA Endpoints

**Diagnostic**:

```bash
# Check if cert header is present
curl -v https://your-app.azurewebsites.net/api/test-cert-header
```

**Possible Causes**:

1. **Client cert not installed in browser**
   - Import `client-cert.p12` into browser certificate store
   - Restart browser after import

2. **App Service client cert mode not enabled**

   ```bash
   az webapp show --resource-group <rg> --name <app> --query "clientCertEnabled"
   # Should return: true
   ```

3. **CA certificate not uploaded to Azure**
   - Verify in Portal: TLS/SSL settings → Private Key Certificates
   - Upload `ca-cert.pfx` if missing

4. **Certificate expired**

   ```bash
   openssl x509 -in client-cert.pem -noout -dates
   # Check notBefore and notAfter dates
   ```

### Issue: Browser Not Prompting for Certificate

**Check**:

1. **Certificate installed correctly**
   - Chrome: `chrome://settings/certificates` → "Your certificates"
   - Firefox: Preferences → Privacy & Security → Certificates → View Certificates

2. **Certificate valid for domain**

   ```bash
   openssl x509 -in client-cert.pem -noout -text | grep "Subject Alternative Name"
   ```

3. **Browser using correct profile**
   - Certificates are profile-specific
   - Try incognito/private mode to test

### Issue: Certificate Validation Failing

**Debug** in `qa_endpoints.py`:

```python
@_require_reviewer
def my_endpoint():
    cert_pem = request.headers.get('X-ARR-ClientCert')
    logger.info(f"Cert header length: {len(cert_pem) if cert_pem else 0}")
    logger.info(f"Cert preview: {cert_pem[:100] if cert_pem else 'MISSING'}")
    logger.info(f"Reviewer principal: {g.reviewer_principal}")
    ...
```

Check logs for:

- Cert header presence and format
- Parsed certificate fields (CN, Email, OU)
- Privilege tier assignment

---

## Security Best Practices

1. **Certificate Rotation**
   - Generate new client certs every 6-12 months
   - Revoke old certs when reviewers leave organization
   - Maintain certificate revocation list (CRL) or OCSP

2. **Private Key Protection**
   - Never commit private keys to git
   - Store CA private key in Azure Key Vault or HSM
   - Encrypt client cert .p12 files with strong passwords

3. **Audit Logging**
   - All QA actions logged to `verification_lineage` table
   - Include certificate CN in audit trail
   - Monitor for unusual patterns (off-hours access, bulk promotions)

4. **Network Isolation**
   - Use Azure Private Endpoints for database connections
   - Restrict App Service to VNet (if applicable)
   - Enable Azure Front Door with WAF for DDoS protection

5. **Backup & Recovery**
   - Back up CA certificate and private key to secure offline storage
   - Document certificate issuance procedures
   - Test certificate renewal process before expiration

---

## Migration from Development to Production

**Current State** (Development):

- `QA_REQUIRE_CERT_AUTH=false` (no cert check)
- All actions attributed to `system:development`

**Production State** (Target):

- `QA_REQUIRE_CERT_AUTH=true` (cert required)
- Client certificates issued to authorized reviewers
- Full audit trail with principal attribution

**Migration Steps**:

1. **Generate certificates** (Step 3)
2. **Upload CA to Azure** (Step 4)
3. **Enable client cert mode** (Step 1)
4. **Test with one reviewer** (import cert, verify QA endpoints work)
5. **Remove `QA_REQUIRE_CERT_AUTH=false`** from Azure App Settings (defaults to true)
6. **Distribute certs to all reviewers**
7. **Update documentation/runbooks**

---

## Reference

**Files**:

- `webapp/parser/config.py` - `QA_REQUIRE_CERT_AUTH` configuration
- `webapp/parser/quality_assurance/qa_endpoints.py` - `@_require_reviewer` decorator
- `webapp/parser/quality_assurance/privilege_tiers.py` - RBAC configuration (if used)

**Azure Documentation**:

- [Configure TLS mutual authentication](https://docs.microsoft.com/en-us/azure/app-service/app-service-web-configure-tls-mutual-auth)
- [Use certificates in code](https://docs.microsoft.com/en-us/azure/app-service/configure-ssl-certificate-in-code)

**OpenSSL Documentation**:

- [OpenSSL Certificate Generation](https://www.openssl.org/docs/manmaster/man1/openssl-req.html)
- [PKCS#12 Format](https://www.openssl.org/docs/manmaster/man1/openssl-pkcs12.html)
