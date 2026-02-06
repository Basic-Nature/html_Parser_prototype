---
layout: default
title: "Content Security Policy (CSP) - Security Model"
---

## Content Security Policy Security Model

## Overview

The Smart Elections Parser implements a **local-first Content Security Policy** that prioritizes security for election data while maintaining graceful fallback capabilities for emergency scenarios.

## Security Modes

### STRICT Mode (Recommended - Default)

- **CSP_MODE=STRICT**
- **Behavior**: No external CDN access allowed
- **Resource Loading**: Bootstrap, Chart.js, Socket.IO all load from local vendor files
- **Use Case**: Production deployments, election data processing, sensitive environments
- **Security**: Maximum security - no external dependencies on CDN availability or integrity

### RELAXED Mode (Emergency Only)

- **CSP_MODE=RELAXED**
- **Behavior**: Allows fallback to CDN if local resources unavailable
- **Resource Loading**:
  - Primary: Local vendor files
  - Fallback: `https://cdn.jsdelivr.net` (JavaScript/CSS)
  - Fallback: `https://cdn.socket.io` (WebSocket library)
- **Use Case**: Development, testing, emergency scenarios where local resources fail
- **Security**: Slightly reduced during fallback, but primary path is still local-first

## Fallback Strategy

### How It Works

1. **JavaScript (Bootstrap, Charts)**
   - Wrapper files in `webapp/static/vendor/` attempt local loading first
   - If local load fails and CSP permits, falls back to CDN
   - CSP blocks fallback in STRICT mode automatically

2. **CSS (Bootstrap)**
   - Primary @import loads local Bootstrap CSS
   - Fallback @import to CDN (blocked by STRICT CSP)
   - Provides graceful degradation if local CSS unavailable

3. **WebSocket (Socket.IO)**
   - Local Socket.IO 4.7.5 available in vendor/
   - Fallback to CDN allowed in RELAXED mode
   - STRICT mode enforces local-only

## Environment Configuration

### Azure App Service Deployment

Set these environment variables in **Configuration → Application Settings**:

```txt
CSP_MODE                = STRICT
ALLOW_STYLE_ATTR        = 0
CSP_EXTRA_SCRIPT        = (leave empty)
CSP_EXTRA_CONNECT       = (leave empty)
```

### Local Development

```bash
# Run with STRICT CSP (recommended)
export CSP_MODE=STRICT
python -m webapp.Smart_Elections_Parser_Webapp

# Or use RELAXED for debugging external resources
export CSP_MODE=RELAXED
python -m webapp.Smart_Elections_Parser_Webapp
```

## Local Resource Cache

All required dependencies are cached locally:

```tree
webapp/static/vendor/
├── bootstrap-5.3.8.min.css       # Bootstrap framework CSS
├── bootstrap-5.3.8.bundle.min.js  # Bootstrap framework JS
├── chart.umd.js                   # Chart.js for dashboards
├── socket.io-4.7.5.min.js         # WebSocket communication
└── xlsx.full.min.js               # Excel export library
```

### Regenerating Local Cache

If you need to update Bootstrap or other dependencies:

```bash
# Download latest Bootstrap from jsDelivr
curl -o webapp/static/vendor/bootstrap-5.3.8.min.css \
  https://cdn.jsdelivr.net/npm/bootstrap@5.3.8/dist/css/bootstrap.min.css

curl -o webapp/static/vendor/bootstrap-5.3.8.bundle.min.js \
  https://cdn.jsdelivr.net/npm/bootstrap@5.3.8/dist/js/bootstrap.bundle.min.js
```

## CSP Directives Applied

### Default Policy (STRICT Mode)

```txt
default-src 'self'
script-src 'self' [nonce-value]
style-src 'self'
style-src-elem 'self'
style-src-attr 'none'
connect-src 'self' ws: wss:
img-src 'self' data:
font-src 'self' data:
frame-ancestors 'none'
form-action 'self'
object-src 'none'
base-uri 'self'
```

### Policy with RELAXED Mode

Adds to above:

```txt
script-src ... https://cdn.jsdelivr.net https://cdn.socket.io
style-src-elem ... https://cdn.jsdelivr.net
connect-src ... https://cdn.jsdelivr.net https://cdn.socket.io
```

## Nonce Generation

All script tags loading external code use cryptographic nonces:

```html
<script nonce="{{ g.csp_nonce }}" src="/static/vendor/bootstrap.bundle.min.js"></script>
```

- **Nonce**: 16-character cryptographic token (regenerated per request)
- **Purpose**: Allows specific scripts to execute under CSP directive
- **Security**: Prevents inline script injection attacks

## Testing CSP Compliance

### Browser DevTools Check

1. Open **Developer Tools → Console**
2. Look for CSP violation warnings (should be empty)
3. Check headers: **Network tab → Response headers → Content-Security-Policy**

### Strict CSP Verification

```bash
# Verify STRICT mode is active
curl -I http://localhost:5000/ | grep Content-Security-Policy

# Expected output (STRICT):
# style-src-elem 'self'
# script-src 'self' 'nonce-...'
# (No cdn.jsdelivr.net references)
```

### Fallback Testing

```bash
# Test RELAXED mode
export CSP_MODE=RELAXED
python -m webapp.Smart_Elections_Parser_Webapp

# DevTools should show CDN allowed in Policy
```

## Troubleshooting

### "violates the following Content Security Policy directive"

**Cause**: External resource blocked by CSP
**Solution**:

1. Check environment: `CSP_MODE` should be `STRICT` for production
2. If error is for jsDelivr, ensure local vendor files exist
3. If legitimate third-party needed, use `CSP_EXTRA_SCRIPT` or `CSP_EXTRA_CONNECT`

### Missing Bootstrap Styling

**Cause**: CSS not loading (CSP blocking or missing files)
**Solution**:

1. Verify `webapp/static/vendor/bootstrap-5.3.8.min.css` exists
2. Regenerate CSS from CDN if corrupted
3. Check browser Network tab for 404 errors

### Socket.IO Connection Fails

**Cause**: WebSocket falls back to polling, blocked by CSP
**Solution**:

1. Verify `webapp/static/vendor/socket.io-4.7.5.min.js` exists
2. Check `connect-src` includes `ws: wss:` in policy
3. Set `CSP_MODE=RELAXED` temporarily to test CDN fallback

## Security Considerations

### Why Local-First?

1. **No external dependency chains**: CDN availability doesn't affect app
2. **Supply chain security**: Can audit all code before use
3. **Network isolation**: Suitable for locked-down networks
4. **Edition audit**: Specific version (5.3.8) is frozen and versioned
5. **Election data protection**: Reduces fingerprinting vectors

### Why Fallback Support?

1. **Emergency access**: Allows temporary CDN use if local fails
2. **Development flexibility**: Easier testing during development
3. **Graceful degradation**: App doesn't fail completely if local unavailable
4. **Administratively controlled**: Fallback requires explicit CSP setting

### Risks & Mitigations

| Risk | Mitigation |
| ------ | ----------- |
| CDN compromise | Default STRICT mode blocks CDN; manual override required |
| CDN unavailability | Local cache ensures always available |
| Version mismatch | Explicit versioning (5.3.8) in filenames |
| Inline script injection | Nonce-based CSP prevents inline scripts |
| Style injection | `style-src-attr 'none'` blocks inline styles by default |

## References

- [MDN: Content Security Policy](https://developer.mozilla.org/en-US/docs/Web/HTTP/CSP)
- [OWASP: CSP Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Content_Security_Policy_Cheat_Sheet.html)
- [jsDelivr: Bootstrap CDN](https://www.jsdelivr.com/package/npm/bootstrap)
