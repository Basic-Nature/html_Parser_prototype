---
layout: default  
title: Security & Authentication
---

# Security & Authentication

Comprehensive guide to certificate-based security, authentication mechanisms, and secure deployment practices for the Smart Elections Parser on Azure and other cloud platforms.

> **Note**: This document references and summarizes content from:
> - [AZURE_CERTIFICATE_AUTH_SETUP.md](../AZURE_CERTIFICATE_AUTH_SETUP.md) - Detailed Azure setup
> - [CERT_AUTH_IMPLEMENTATION.md](../CERT_AUTH_IMPLEMENTATION.md) - Implementation details
>
> For complete Azure-specific instructions, see [AZURE_CERTIFICATE_AUTH_SETUP.md](../AZURE_CERTIFICATE_AUTH_SETUP.md)

## 🔐 Overview: Certificate-Based Authentication

The Smart Elections Parser uses **certificate-based authentication** to securely control access to sensitive QA and debugging endpoints. This approach provides:

- **Strong encryption**: TLS/SSL certificate-based validation
- **Mutual authentication**: Server verifies client certificates
- **Audit trail**: All access events logged with certificate details
- **No password management**: Eliminates password-related vulnerabilities
- **Azure-native**: Integrates with Azure App Service native features

## 🏛️ Architecture

```
┌─────────────────────────────────────────────────┐
│ Client (QA Panel, Admin Tool, Browser)          │
│ ✓ Has client certificate (PEM/PKCS12)          │
└────────────────┬────────────────────────────────┘
                 │ TLS Handshake with Client Cert
                 ↓
┌─────────────────────────────────────────────────┐
│ Azure App Service / Load Balancer               │
│ ✓ Receives X-ARR-ClientCert header             │
│ ✓ Verifies certificate chain                    │
└────────────────┬────────────────────────────────┘
                 │ Forward cert (X-ARR-ClientCert)
                 ↓
┌─────────────────────────────────────────────────┐
│ Smart Elections Parser Application              │
│ ✓ Validates X-ARR-ClientCert header             │
│ ✓ Checks certificate against CA                 │
│ ✓ Extracts client identity                      │
│ ✓ Authorizes access to QA endpoints             │
└─────────────────────────────────────────────────┘
```

## 🔑 Configuration

### Environment Variables

```bash
# Certificate validation toggle (production default: true)
QA_REQUIRE_CERT_AUTH=true

# Optional: Allowed certificate subjects (comma-separated)
QA_ALLOWED_CERT_SUBJECTS="CN=qa.tool.local,CN=admin.tool.local"

# Optional: Certificate trust store path
QA_CA_BUNDLE_PATH=/etc/ssl/certs/ca-bundle.crt

# Optional: Fallback principal (development only, default: disabled)
QA_FALLBACK_PRINCIPAL=system:development  # Only if explicitly enabled
```

### Security Defaults

| Setting | Development | Production |
|---------|-------------|-----------|
| `QA_REQUIRE_CERT_AUTH` | `false` (optional) | `true` (required) ✓ |
| Client cert validation | Disabled | Enabled |
| Fallback auth | Allowed | Disabled |
| Audit logging | Info level | Detailed |

## 🛠️ Certificate Setup

### For Azure Deployment

1. **Generate Client Certificate** (if you don't have one):
   ```bash
   # Generate private key
   openssl genrsa -out client-key.pem 2048
   
   # Generate certificate signing request
   openssl req -new -key client-key.pem -out client.csr
   
   # Sign certificate (or use your CA)
   openssl x509 -req -in client.csr \
     -signkey client-key.pem -out client-cert.pem \
     -days 365
   ```

2. **Convert to PKCS12** (for browser import):
   ```bash
   openssl pkcs12 -export \
     -in client-cert.pem \
     -inkey client-key.pem \
     -out client-cert.p12 \
     -name "QA Parser Client"
   ```

3. **Upload to Azure**:
   - Certificate → Azure Key Vault
   - Configure App Service to present client certificate requirement
   - Set TLS version to 1.2+ (recommended: 1.3)

4. **Configure Application**:
   ```python
   # In your Flask app
   from flask import request
   
   def validate_cert():
       """Validate client certificate from Azure header."""
       cert_header = request.headers.get('X-ARR-ClientCert')
       if not cert_header and os.getenv('QA_REQUIRE_CERT_AUTH') == 'true':
           abort(401, "Client certificate required")
       # ... additional validation
   ```

## 🔐 QA Endpoint Protection

### Protected Endpoints
All endpoints under `/qa/*` and `/api/qa/*` require certificate validation:

```python
@app.route('/qa/override', methods=['POST'])
@require_cert_auth()  # Decorator validates certificate
def qa_override():
    """Override extraction results (QA only)."""
    # Implementation
```

### Validation Steps
1. Check if `X-ARR-ClientCert` header exists
2. Decode certificate from header
3. Verify certificate chain against trusted CAs
4. Extract subject CN (Common Name)  
5. Compare against allowed subjects (if configured)
6. Log access attempt with certificate details
7. Grant or deny access

## 🚀 Development Mode

For local development without certificates:

```bash
# Development (local testing)
export QA_REQUIRE_CERT_AUTH=false

# Or explicitly enable fallback auth (security review recommended)
export QA_FALLBACK_PRINCIPAL=system:development

# Start parser (certificate validation skipped)
python Smart_Elections_Parser_Webapp.py
```

⚠️ **Warning**: Development-only settings must NEVER be used in production.

## 📋 Deployment Checklist

### Pre-Deployment
- [ ] Generate or obtain client certificate(s)
- [ ] Verify certificate validity (not expired, proper chain)
- [ ] Convert to PKCS12 for browser use if needed
- [ ] Store private key securely (Azure Key Vault recommended)
- [ ] Document certificate subjects and renewal dates
- [ ] Test locally with `QA_REQUIRE_CERT_AUTH=false` first

### Azure Configuration
- [ ] Upload certificate to Azure Key Vault
- [ ] Configure App Service client certificate requirement
- [ ] Set minimum TLS version to 1.2 (1.3 recommended)
- [ ] Enable "Client certificate mode" in App Service  
- [ ] Test that `X-ARR-ClientCert` header is forwarded
- [ ] Verify certificate validation in logs

### Application Configuration
- [ ] Set `QA_REQUIRE_CERT_AUTH=true` in production
- [ ] Configure `QA_ALLOWED_CERT_SUBJECTS` if restricting access
- [ ] Set up certificate renewal process (30-60 days before expiry)
- [ ] Enable audit logging to track certificate usage
- [ ] Test QA endpoint access with certificate
- [ ] Test rejection of requests without certificate

### Post-Deployment
- [ ] Monitor logs for certificate validation errors
- [ ] Verify all QA endpoints return 401 without certificate
- [ ] Test QA panel functionality with certificate
- [ ] Document certificate usage and troubleshooting
- [ ] Set calendar reminder for certificate renewal

## 🔍 Troubleshooting

### Issue: "Client certificate required" (401) on Azure

**Cause**: Azure App Service not forwarding client certificate header

**Solution**:
1. Verify "Client certificate mode" enabled in App Service settings
2. Check TLS minimum version (must be 1.2+)
3. Restart App Service after configuration changes
4. Test with simple request: `curl -H "User-Agent: test" https://your-app.azurewebsites.net/qa/health`
5. Check Azure Activity Log for configuration changes

### Issue: "Certificate verification failed"

**Cause**: Certificate not in trusted chain or invalid format

**Solution**:
1. Verify certificate validity: `openssl x509 -in cert.pem -text -noout`
2. Check expiry date (renewal needed if < 30 days)
3. Verify certificate chain: `openssl verify -CAfile ca-bundle.crt cert.pem`
4. Ensure certificate format matches (`pem` vs `pkcs12`)
5. Check `QA_CA_BUNDLE_PATH` if using custom CA

### Issue: QA panel shows "Unauthorized"

**Cause**: Browser doesn't have correct client certificate installed

**Solution**:
1. Import PKCS12 certificate into browser (`.p12` file)
2. Verify certificate appears in browser settings: `about:certificates`
3. Check certificate subject matches `QA_ALLOWED_CERT_SUBJECTS`
4. Clear browser cache and reload
5. Test in incognito/private mode to rule out cache issues

### Issue: Certificate validation always fails

**Cause**: `QA_REQUIRE_CERT_AUTH` incorrectly set

**Solution**:
1. Check environment variable: `echo $QA_REQUIRE_CERT_AUTH`
2. Restart application after configuration change
3. Review application logs: `grep -i "cert" application.log`
4. For development only: temporarily set to `false` to verify application works
5. Reset to `true` before production

## 📊 Audit Logging

All certificate-based access is logged:

```
[INFO] Certificate Auth: CN=qa.tool.local, Subject=/CN=qa.tool.local/O=ElevaSoft, Access granted to /qa/override
[INFO] Certificate Auth: Missing X-ARR-ClientCert header, Access denied to /api/qa/custom_data
[WARNING] Certificate Auth: CN=expired.key, Certificate expired (2023-12-31), Access denied
```

### Log Analysis
```bash
# Count certificate validation failures
grep "Certificate Auth.*Access denied" app.log | wc -l

# Find all QA access attempts
grep "Certificate Auth" app.log | tail -20

# Identify certificate issues
grep -i "certificate.*failed\|expired\|invalid" app.log
```

## 🔄 Certificate Renewal

### Before Expiry
1. Generate new certificate (30-60 days before expiry)
2. Test new certificate locally first
3. Schedule update during maintenance window
4. Notify users of certificate change

### Renewal Process
1. Generate new certificate (same subject/CN)
2. Upload to Azure Key Vault (new version)
3. Update App Service to use new version
4. Restart application
5. Verify QA endpoints working with new cert
6. Document update with timestamp

### Revocation
If certificate is compromised:
1. Generate new certificate immediately
2. Update all client installs within 24 hours
3. Consider adding old cert to revocation list
4. Review logs for unauthorized access (past 30 days)
5. Implement additional monitoring

## ✅ Testing

### Manual Test
```bash
# Without certificate (should fail in production)
curl https://your-app.azurewebsites.net/qa/health
# Response: 401 Unauthorized

# With certificate (production only if certificate installed)
curl --cert client-cert.pem --key client-key.pem \
  https://your-app.azurewebsites.net/qa/health
# Response: 200 OK + health status
```

### Automated Testing
```python
import requests
from requests.auth import HTTPCertAuth

# Test with certificate
cert = ('client-cert.pem', 'client-key.pem')
response = requests.get(
    'https://your-app.azurewebsites.net/qa/health',
    cert=cert,
    verify='/path/to/ca-cert.pem'
)
assert response.status_code == 200
```

---

**Related Documents**:
- [AZURE_CERTIFICATE_AUTH_SETUP.md](../AZURE_CERTIFICATE_AUTH_SETUP.md) - Complete Azure setup guide
- [Deployment Guide](./DEPLOYMENT.md) - General deployment procedures
- [Operations Runbook](./OPERATIONS.md) - Operational procedures

**Last Updated**: Consolidated security & authentication guide
