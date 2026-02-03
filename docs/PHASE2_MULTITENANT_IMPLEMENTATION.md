# Phase 2: Multi-Tenant Security Framework Implementation

## Overview

This document describes the complete implementation of a comprehensive multi-tenant security framework for the Smart Elections Parser. The framework enforces privilege-tier-aware URL trust scoring, session branching isolation, and audit trails for all security decisions.

## Architecture Summary

### Core Components

#### 1. **Session Branching Module** (`session_branching.py`)

- **Purpose**: Per-principal URL isolation enforcement to prevent cross-tenant data leakage
- **Key Class**: `SessionBranch` maintains quarantine/reject URL lists per principal
- **Global Map**: `_BRANCH_ISOLATION_MAP` with RLock for thread-safe access
- **Key Functions**:
  - `validate_url_access(principal, url, access_type, principal_source)` → `(bool, reason)`
  - `add_url_to_isolation(principal, url, status)` → Adds URLs to quarantine/reject
  - `get_isolation_summary(principal)` → Returns audit summary
  - `cleanup_principal_isolation(principal)` → Removes branch on logout

- **Security Features**:
  - ROOT_ADMIN can bypass with audit logging
  - Tracks all access attempts with timestamps
  - Logs isolation breaches as WARNING level
  - Thread-safe RLock protection

#### 2. **Session Manager Extensions** (`session_manager.py`)

- **Purpose**: Bridge session metadata to isolation enforcement
- **New Methods** (4):
  - `validate_principal_url_access(session_id, url, access_type)` - Delegates to session_branching
  - `add_url_to_principal_isolation(session_id, url, status)` - Adds URL to isolation
  - `get_principal_isolation_summary(session_id)` - Returns audit data
  - `cleanup_principal_isolation(session_id)` - Removes isolation branch

- **Integration Pattern**:
  - Extracts principal from session metadata
  - Calls corresponding session_branching function
  - Returns result to caller

#### 3. **Trust Scorer Enhancement** (`url_trust_scorer.py`)

- **Enhanced Functions**:
  - `should_quarantine(trust_score, url, privilege_tier=None)` with tier logic:
    - ROOT_ADMIN: returns False (bypass)
    - ADMIN_FULL_TRUST: stricter range (40-50 instead of 30-50)
    - Others: standard (30-50 range)

  - `should_reject(trust_score, url, privilege_tier=None)` with tier logic:
    - ROOT_ADMIN: returns False with WARNING log (bypasses but audited)
    - ADMIN_FULL_TRUST: stricter threshold (< 20)
    - Others: standard (< 30)

- **Logging**: All tier-specific decisions logged with privilege_tier in payload

#### 4. **Integrity Check Enhancement** (`Integrity_check.py`)

- **Enhanced Return Value**: `analyze_contests()` now includes `tier_summary` dict:

  ```python
  "tier_summary": {
    "privilege_tier": tier.value if tier else None,
    "tier_name": tier.name if tier else "UNKNOWN",
    "trust_factors_present": bool(trust_factors),
    "verified_domain": trust_factors.get("verified_domain"),
    "admin_boost_applied": trust_factors.get("admin_boost_applied"),
    "anomaly_strategy": "strict_verified" | "all_anomalies_reviewed" | "standard"
  }
  ```

- **ML Integration**: Trust factors (9 dimensions) now features in anomaly detection:
  - verified_domain, gov_domain, ssl_valid, suspicious_tld
  - phishing_count, historical_success, admin_boost_applied
  - domain_mimicry, allowlist_match

- **Tier-Specific Anomaly Detection**:
  - ADMIN_FULL_TRUST + verified_domain: contamination=0.01 (only severe anomalies)
  - Standard: contamination=0.05 (more permissive)

#### 5. **Orchestration Wiring** (`html_election_parser.py`)

- **Central Hub**: `orchestrate_url()` extracts and passes privilege information through entire pipeline
- **Key Changes**:
  - Extracts `principal` and `principal_source` from kwargs
  - Retrieves `privilege_tier` via `get_principal_tier(principal, principal_source)`
  - Passes `privilege_tier` to:
    - `should_reject()` checks
    - `should_quarantine()` checks
    - `ai_analyze_results()` function
    - snapshot_context for DOM snapshot mode

  - **ai_analyze_results() Function**:
    - Extended signature: `ai_analyze_results(..., trust_factors=None, privilege_tier=None)`
    - Calls `analyze_contests(contests, trust_factors=trust_factors, privilege_tier=privilege_tier)`
    - Extracts `tier_summary` from results
    - Includes tier_summary in error/info logging payloads

  - **Call Sites**: Updated 3 calls to ai_analyze_results():
    - Line ~1363: snapshot_context path
    - Line ~1543: Selenium fallback path
    - Line ~1839: full navigation path

#### 6. **Web Pipeline Integration** (`web_pipeline.py`)

- **Purpose**: Enforce multi-tenant isolation at web entry point
- **Key Additions**:
  - **Initialization** (lines ~120-140):
    - Import `get_isolated_branch` and `get_principal_tier` from session_branching
    - Initialize isolation branch for principal
    - Log tier information

  - **Pre-Processing Validation** (2 locations):
    - **Before first main() call** (lines ~310-355):
      - For URLs from file (interactive path)
      - Validates each URL via `validate_url_access()`
      - Filters out blocked URLs with WARNING logging
      - Returns if all URLs are blocked

    - **Before second main() call** (lines ~375-415):
      - For explicit URLs provided
      - Same validation and filtering logic
      - Returns if all URLs are blocked

  - **Session Cleanup** (lines ~600-615 in finally block):
    - Calls `cleanup_principal_isolation(principal)`
    - Logs cleanup INFO event
    - Handles cleanup exceptions gracefully

## Privilege Tier System

### Tier Hierarchy

1. **ROOT_ADMIN** - Highest privilege
   - Bypasses all URL rejection/quarantine checks
   - Actions logged as WARNING for audit trail
   - Stricter contamination in ML (0.01)

2. **ADMIN_FULL_TRUST** - Full administrative access
   - Gets stricter thresholds: reject < 20, quarantine 40-50
   - Stricter contamination in ML (0.01) for verified domains
   - Can review all anomalies

3. **DATA_STEWARD** - Data management
   - Standard thresholds apply
   - Can create/modify analysis
   - Limited anomaly review

4. **REVIEWER** - Read/review access
   - Standard thresholds
   - Can view analysis results
   - Cannot modify

5. **USER** - Basic user
   - Standard thresholds
   - Limited access to own analyses
   - View-only for shared results

### Trust Score Scale

- **0-29**: Rejected (immediately blocked)
- **30-49**: Quarantined (requires manual review)
- **50-79**: Direct processing allowed
- **80-100**: High confidence processing

Thresholds adjusted per tier via privilege-specific logic.

## Audit Logging

All security decisions and privilege operations logged with consistent structure:

### Security Decision Logs

```json
{
  "level": "WARNING|ERROR|INFO",
  "type": "trust_scorer",
  "message": "URL rejected due to low trust score...",
  "session_id": "sess_...",
  "url": "https://...",
  "trust_score": 25,
  "trust_factors": {...},
  "privilege_tier": "ADMIN_FULL_TRUST",
  "principal": "user@example.com",
  "principal_source": "sso_oid"
}
```

### Isolation Breach Logs

```json
{
  "level": "WARNING",
  "type": "isolation",
  "message": "[MultiTenant] URL blocked due to isolation...",
  "session_id": "sess_...",
  "principal": "user@example.com",
  "url": "https://...",
  "block_reason": "quarantined"
}
```

### Privilege Bypass Logs

```json
{
  "level": "INFO",
  "type": "trust_scorer",
  "message": "ROOT_ADMIN bypass applied to trust check",
  "session_id": "sess_...",
  "privilege_tier": "ROOT_ADMIN",
  "url": "https://...",
  "trust_score": 15,
  "bypass_reason": "admin_authority"
}
```

## Data Flow

### URL Processing Pipeline

```txt
1. URL Entry (web_pipeline.py)
   ├─ Initialize isolation branch
   ├─ Pre-process validation against principal isolation
   │  └─ validate_url_access() → filter blocked URLs
   ├─ Call main() with URLs
   │
2. URL Processing (html_election_parser.py)
   ├─ orchestrate_url()
   ├─ Extract principal/principal_source
   ├─ compute_trust_score()
   │  └─ Returns trust_score and trust_factors
   ├─ Check trust score
   │  ├─ should_reject(trust_score, url, privilege_tier)
   │  │  └─ Tier-aware threshold check
   │  └─ should_quarantine(trust_score, url, privilege_tier)
   │     └─ Tier-aware threshold check
   ├─ Parse URL content (if not rejected)
   │
3. Anomaly Detection
   ├─ ai_analyze_results(headers, data, contest, metadata, 
   │                     trust_factors=trust_factors, 
   │                     privilege_tier=privilege_tier)
   ├─ analyze_contests(contests, trust_factors, privilege_tier)
   ├─ detect_anomalies_with_ml()
   │  └─ Uses trust_factors + tier-specific contamination
   ├─ Returns tier_summary
   │
4. Output
   ├─ Log results with tier_summary
   ├─ Download access checked against isolation
   │
5. Cleanup (finally block in web_pipeline.py)
   └─ cleanup_principal_isolation(principal)
      └─ Remove isolation branch, log completion
```

## Security Guarantees

### Multi-Tenant Isolation

✅ Per-principal quarantine/reject lists prevent cross-tenant access
✅ Isolation validated before URL processing begins
✅ All access attempts logged with timestamps
✅ Cleanup on session end removes isolation branch

### Privilege Enforcement

✅ ROOT_ADMIN bypasses logged for audit trail
✅ ADMIN_FULL_TRUST gets stricter thresholds for safety
✅ Tier information flows through entire pipeline
✅ ML anomaly detection considers privilege tier

### Audit Trail

✅ All trust decisions logged with decision factors
✅ Privilege bypasses logged at INFO level
✅ Isolation breaches logged at WARNING level
✅ Session cleanup logged for accountability

## Implementation Status

### Completed ✅

- Session branching module with isolation enforcement
- Session manager integration with 4 new methods
- Trust scorer tier-aware functions (should_reject/should_quarantine)
- Integrity check tier_summary and trust factors in ML
- Privilege tier propagation through orchestrate_url()
- Web pipeline isolation validation and cleanup

### Files Modified

1. **Created**: `webapp/parser/health/session_branching.py` (230 lines)
2. **Modified**: `webapp/parser/health/session_manager.py` (+ 103 lines)
3. **Modified**: `webapp/parser/utils/url_trust_scorer.py` (2 functions)
4. **Modified**: `webapp/parser/Context_Integration/Integrity_check.py` (analyze_contests enhancement)
5. **Modified**: `webapp/parser/html_election_parser.py` (orchestrate_url + ai_analyze_results)
6. **Modified**: `webapp/parser/web_pipeline.py` (process_urls_for_web + ~150 lines)

## Next Steps (Future Enhancements)

1. **Quarantine Review Workflow**: Interactive manual review of quarantined URLs with approval workflow
2. **Isolation Metrics Dashboard**: Real-time visualization of per-principal isolation status
3. **Privilege Tier Adjustment**: Dynamic tier adjustment based on verified actions
4. **Cross-Tenant Audit Reports**: Aggregate isolation and privilege reports for compliance
5. **Breach Detection**: Automated detection of isolation breach patterns

## Troubleshooting

### URL Blocked Unexpectedly

- Check isolation status: `session_manager.get_principal_isolation_summary(session_id)`
- Verify principal has access tier for URL domain
- Check logs for "URL blocked due to isolation" messages

### Privilege Tier Not Applied

- Verify principal/principal_source passed to pipeline
- Check `get_principal_tier()` returns expected tier
- Confirm tier information in orchestrate_url() logs

### Isolation Cleanup Failed

- Check for exceptions in finally block logs
- Verify cleanup_principal_isolation() idempotent behavior
- Sessions will eventually expire if cleanup fails

---

**Framework Version**: 1.0  
**Last Updated**: 2025-01-19  
**Status**: Production Ready ✅
