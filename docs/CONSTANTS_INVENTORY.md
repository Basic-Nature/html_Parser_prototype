# Constants Inventory: Current State & Migration Plan

**Last Updated**: 2026-02-03  
**Status**: Active Enumeration  
**Authoritative Source**: `webapp/parser/config.py` and related configuration files

---

## Overview

This document catalogs all constants currently in the Smart Elections Parser codebase, maps them to target vocab files, assigns trust levels, and identifies migration priorities.

**Legend**:

- **Trust Level**: high (0.95+), medium (0.75-0.94), low (0.50-0.74), deprecated (< 0.50)
- **Vocab File**: Target compartmentalized file after migration
- **Status**: existing (already in code), unmapped (needs definition), deprecated (can be removed)
- **Migration Priority**: 1 (critical, first wave), 2 (important, second wave), 3 (optional, third wave)

---

## 1. Jurisdictional Constants

| Constant | Current Value | Trust Level | Vocab File | Migration Priority | Notes |
| ---------- | --------------- | ------------- | ------------ | ------------------- | ------- |
| `STATES` | List of 50 US state names | high (0.99) | `entities/jurisdictions.txt` | 1 | Official FIPS standard |
| `STATE_ABBREVIATIONS` | {"CA": "California", ...} | high (0.99) | `validators/state_aliases.txt` | 1 | FIPS standard |
| `COUNTIES_BY_STATE` | {"CA": ["Alameda", "Alpine", ...], ...} | high (0.98) | `entities/jurisdictions.txt` | 1 | Official state registries |
| `COUNTY_ALIASES` | {"LA": "Los Angeles", ...} | medium (0.85) | `validators/county_aliases.txt` | 1 | Manual curation from sources |
| `DISTRICTS` | List of congressional districts | high (0.95) | `entities/jurisdictions.txt` | 2 | Official redistricting data |
| `PRECINCTS_BY_COUNTY` | Dict of precinct IDs per county | medium (0.80) | `entities/jurisdictions.txt` | 3 | Dynamic per-county data (update weekly) |
| `TIMEZONE_MAP` | {"CA": "PT", "NY": "ET", ...} | high (0.98) | `validators/state_aliases.txt` | 3 | Official US timezone standard |

**Source Authority**:

- US Census Bureau (FIPS)
- Secretary of State offices (official county listings)
- Congressional redistricting data (FEC/Census)
- Precinct data from county clerks (refreshed weekly during election season)

---

## 2. Political Constants

| Constant | Current Value | Trust Level | Vocab File | Migration Priority | Notes |
| ---------- | --------------- | ------------- | ------------ | ------------------- | ------- |
| `VALID_OFFICES` | ["President", "Senator", "Representative", ...] | high (0.99) | `entities/offices.txt` | 1 | State election codes |
| `OFFICE_ALIASES` | {"Pres": "President", ...} | medium (0.85) | `validators/office_aliases.txt` | 1 | Common abbreviations |
| `VALID_PARTIES` | ["Democratic", "Republican", "Green", ...] | high (0.98) | `entities/parties.txt` | 1 | Official state registries (FEC) |
| `PARTY_ALIASES` | {"Dem": "Democratic", "GOP": "Republican", ...} | medium (0.80) | `validators/party_aliases.txt` | 1 | Common colloquialisms |
| `VALID_DIVISIONS` | ["General", "State", "Local", ...] | medium (0.85) | `entities/contest_types.txt` | 2 | State-specific classification |
| `SUPPORTED_OFFICES_BY_STATE` | {"CA": ["Governor", "Senator", ...], ...} | high (0.95) | `entities/jurisdictions.txt` | 2 | Official SoS office lists per state |
| `PARTY_CODES` | {"D": "Democratic", "R": "Republican", ...} | high (0.98) | `validators/party_aliases.txt` | 1 | FEC official abbreviations |
| `PARTY_COLORS` | {"Democratic": "#0015BC", ...} | medium (0.75) | `scoring/trust_signals.txt` | 3 | Visualization only (low criticality) |

**Source Authority**:

- State election codes (official offices)
- Federal Election Commission registry (party names + codes)
- Manual curation from news media (common abbreviations)

---

## 3. Contest & Election Type Constants

| Constant | Current Value | Trust Level | Vocab File | Migration Priority | Notes |
| ---------- | --------------- | ------------- | ------------ | ------------------- | ------- |
| `CONTEST_TYPES` | ["General Election", "Primary", "Runoff", ...] | high (0.99) | `entities/contest_types.txt` | 1 | State election code definitions |
| `ELECTION_TYPES` | ["General", "Primary", "Special", ...] | high (0.99) | `entities/contest_types.txt` | 1 | Common classification |
| `MEASURE_TYPES` | ["Proposition", "Bond", "Recall", ...] | high (0.95) | `entities/contest_types.txt` | 2 | State-specific measure types |
| `VALID_BALLOT_MEASURES` | Dynamically populated from state data | high (0.90) | Dynamic entity source | 3 | Updated per election cycle |

**Source Authority**:

- State election codes (official contest types)
- State Secretary of State guidance (measure classifications)

---

## 4. Result & Data Format Constants

| Constant | Current Value | Trust Level | Vocab File | Migration Priority | Notes |
| ---------- | --------------- | ------------- | ------------ | ------------------- | ------- |
| `RESULT_COLUMN_HEADERS` | ["Candidate", "Votes", "Percentage", ...] | high (0.95) | `snapshots/table_headers.txt` | 1 | Observed in real exports |
| `COMMON_HEADER_VARIATIONS` | {"Vote Count": "Votes", ...} | medium (0.85) | `snapshots/table_headers.txt` | 1 | Normalization mapping |
| `NUMERIC_COLUMNS` | ["Votes", "Percentage", "Margin", ...] | high (0.98) | `snapshots/result_terms.txt` | 2 | Expected numeric columns |
| `TEXT_COLUMNS` | ["Candidate", "Party", "Office", ...] | high (0.98) | `snapshots/result_terms.txt` | 2 | Expected text columns |
| `SUPPORTED_EXTENSIONS` | [".pdf", ".csv", ".xlsx", ".html", ...] | high (0.99) | `config.py` (keep here—technical, not vocab) | N/A | File type support (technical, not political) |
| `MIME_TYPES` | {"pdf": "application/pdf", ...} | high (0.99) | `config.py` (keep here) | N/A | Technical specification |

**Source Authority**:

- Observed in official SoS result exports (column header variations)
- Federal/state file format standards (technical specs)

---

## 5. Safety & Trust Constants

| Constant | Current Value | Trust Level | Vocab File | Migration Priority | Notes |
| ---------- | --------------- | ------------- | ------------ | ------------------- | ------- |
| `URL_ALLOWLIST_HOSTS` | ["results.sos.ca.gov", ...] | high (0.99) | `sources/verified_sources.txt` | 1 | Official election authority domains |
| `URL_ALLOWLIST_SUFFIXES` | [".gov", ".edu", ...] | high (0.98) | `config.py` (keep here—policy) | N/A | Trust policy (not vocab-domain) |
| `URL_BLOCK_PRIVATE_IPS` | True | high (0.99) | `config.py` (keep here—policy) | N/A | Security policy |
| `URL_ENFORCE_ALLOWLIST` | True | high (0.99) | `config.py` (keep here—policy) | N/A | Security policy |
| `VERIFIED_SOURCES` | List of trusted source URLs w/ metadata | high (0.99) | `sources/verified_sources.txt` | 1 | Official SoS URLs + metadata |
| `TRUST_SCORE_THRESHOLDS` | {"high": 0.85, "medium": 0.60, ...} | medium (0.80) | `scoring/trust_signals.txt` | 2 | Confidence calculation thresholds |
| `ANOMALY_QUARANTINE_RULES` | Rules for flagging suspicious data | high (0.90) | `scoring/anomaly_reasons.txt` | 1 | Election integrity requirements |
| `CONFIDENCE_THRESHOLD_DEFAULT` | 0.70 | medium (0.75) | `config.py` (keep here—tuning) | N/A | Operational parameter |

**Source Authority**:

- Official election authority websites (verified source URLs)
- Election Integrity Team (trust scoring policy)
- State election code (anomaly quarantine rules)

---

## 6. Handler & Processing Constants

| Constant | Current Value | Trust Level | Vocab File | Migration Priority | Notes |
| ---------- | --------------- | ------------- | ------------ | ------------------- | ------- |
| `SUPPORTED_HANDLERS` | ["html", "pdf", "csv", "xlsx", "json"] | high (0.99) | `config.py` (keep here—technical) | N/A | Handler registry (technical) |
| `HANDLER_TIMEOUTS_MS` | {"html": 30000, "pdf": 60000, ...} | high (0.95) | `config.py` (keep here—tuning) | N/A | Performance tuning |
| `MAX_PDF_PAGES` | 500 | high (0.95) | `config.py` (keep here—limit) | N/A | Resource limit policy |
| `MAX_CSV_ROWS` | 100000 | high (0.95) | `config.py` (keep here—limit) | N/A | Resource limit policy |
| `MAX_XLSX_BYTES` | 52428800 (50 MB) | high (0.95) | `config.py` (keep here—limit) | N/A | Resource limit policy |
| `NAV_MAX_ATTEMPTS` | 3 | high (0.90) | `config.py` (keep here—tuning) | N/A | Retry policy |
| `NAV_TIMEOUT_PLAYWRIGHT_MS` | 30000 | high (0.90) | `config.py` (keep here—tuning) | N/A | Performance tuning |

**Note**: Handler/processing constants are **technical** (not election-integrity-domain) and should **remain in config.py**. Only vocab-domain constants migrate to compartmentalized files.

---

## 7. Database & Storage Constants

| Constant | Current Value | Trust Level | Vocab File | Migration Priority | Notes |
| ---------- | --------------- | ------------- | ------------ | ------------------- | ------- |
| `POSTGRES_HOST` | env var | N/A | `.env` (secrets, NOT vocab) | N/A | Database configuration (keep secure) |
| `POSTGRES_DB` | "election_results" | high (0.99) | `config.py` (keep here—technical) | N/A | Database name (not vocab) |
| `POSTGRES_USER_RAW` | env var | N/A | `.env` (secrets, NOT vocab) | N/A | Credentials (keep secure) |
| `POSTGRES_PASSWORD_RAW` | env var | N/A | `.env` (secrets, NOT vocab) | N/A | Credentials (keep secure) |
| `INPUT_DIR` | "input/" | high (0.99) | `config.py` (keep here—technical) | N/A | Folder path |
| `OUTPUT_DIR` | "output/" | high (0.99) | `config.py` (keep here—technical) | N/A | Folder path |
| `UPLOADS_DIR` | "uploads/" | high (0.99) | `config.py` (keep here—technical) | N/A | Folder path |
| `LOG_DIR` | "log/" | high (0.99) | `config.py` (keep here—technical) | N/A | Folder path |

**Note**: Database credentials **MUST stay in `.env`** (secrets management). Folder paths are technical configuration—keep in `config.py`.

---

## 8. Feature Flags & Operational Constants

| Constant | Current Value | Trust Level | Vocab File | Migration Priority | Notes |
| ---------- | --------------- | ------------- | ------------ | ------------------- | ------- |
| `ENABLE_AI_ANALYSIS` | True/False | medium (0.75) | `config.py` (keep here—feature flag) | N/A | Feature toggle |
| `ENABLE_PARALLEL` | True/False | high (0.90) | `config.py` (keep here—tuning) | N/A | Performance flag |
| `ENABLE_REALTIME_STREAM` | True/False | medium (0.80) | `config.py` (keep here—feature flag) | N/A | Feature toggle |
| `ENABLE_SELENIUM_FALLBACK` | True/False | high (0.90) | `config.py` (keep here—tuning) | N/A | Fallback strategy |
| `ENABLE_HEALTH_TASKS` | True/False | high (0.95) | `config.py` (keep here—feature flag) | N/A | Operational toggle |
| `HEARTBEAT_ENABLED` | True | high (0.95) | `config.py` (keep here—tuning) | N/A | Monitoring flag |
| `ALLOW_GOOGLE_DOCS` | True/False | medium (0.70) | `config.py` (keep here—policy) | N/A | Cloud service policy |
| `ALLOW_LEGACY_OUTPUT_DOWNLOAD` | True/False | high (0.95) | `config.py` (keep here—backward compat) | N/A | Backward compatibility flag |

**Note**: Feature flags and operational toggles stay in `config.py`—they're runtime configuration, not election-integrity-domain vocab.

---

## 9. Deprecated/Candidates for Removal

| Constant | Current Value | Trust Level | Vocab File | Reason for Deprecation | Status |
| ---------- | --------------- | ------------- | ------------ | ---------------------- | -------- |
| `LEGACY_OFFICE_NAMES` | Outdated office list | low (0.40) | N/A (remove) | Replaced by official SoS registry | Deprecated |
| `HARDCODED_COUNTY_MAP` | Old CSV-based county mapping | low (0.35) | N/A (remove) | Replaced by dynamic sources | Deprecated |
| `OLD_RESULT_FORMATS` | Unsupported file type specs | low (0.20) | N/A (remove) | No longer used | Deprecated |

**Recommendation**: Schedule removal in Q2 2026 after handler migration complete.

---

## 10. Migration Summary Table

### By Priority Level

***Priority 1 (Immediate—Week 1-2)***

- State/county jurisdictions
- Valid offices
- Valid parties
- Contest types
- State aliases
- County aliases
- Verified sources registry
- Anomaly quarantine rules

**Count**: 13 constants

***Priority 2 (Second Wave—Week 3-4)***

- Supported offices per state
- Measure types
- Divisions
- Trust score thresholds
- Result column headers (snapshots)
- Common header variations

**Count**: 10 constants

***Priority 3 (Optional—Week 5+)***

- Party colors
- Precincts by county
- Ballot measures (dynamic)
- Timezone map

**Count**: 4 constants

***Keep in config.py (Not Vocab)***

- File extensions + MIME types (technical)
- Handler registry (technical)
- Handler timeouts (tuning)
- Resource limits (policy)
- Database configuration (secrets)
- Folder paths (technical)
- Feature flags (operational)
- URL trust policy (security policy)

**Count**: 20+ constants remain in config.py

---

## 11. Vocab File Mapping Matrix

| Vocab File | Source Constants | Record Count | Update Frequency | Trust Level |
| ------------ | ----------------- | -------------- | ------------------ | ------------- |
| `entities/offices.txt` | VALID_OFFICES, OFFICE_ALIASES (canonical) | ~150 | Annually (per SoS updates) | high (0.99) |
| `entities/parties.txt` | VALID_PARTIES | ~10 | Annually (FEC) | high (0.98) |
| `entities/jurisdictions.txt` | STATES, COUNTIES_BY_STATE, DISTRICTS, PRECINCTS_BY_COUNTY, SUPPORTED_OFFICES_BY_STATE | ~3,200 | Weekly (precinct updates) | high (0.98) |
| `entities/contest_types.txt` | CONTEST_TYPES, ELECTION_TYPES, MEASURE_TYPES | ~20 | Annually (per SoS) | high (0.99) |
| `entities/result_terms.txt` | NUMERIC_COLUMNS, TEXT_COLUMNS (terms for header matching) | ~30 | As-needed (new formats) | high (0.95) |
| `validators/state_aliases.txt` | STATE_ABBREVIATIONS, TIMEZONE_MAP | ~200 | Annually | high (0.98) |
| `validators/county_aliases.txt` | COUNTY_ALIASES | ~3,000 | Weekly (precinct updates) | medium (0.85) |
| `validators/office_aliases.txt` | OFFICE_ALIASES | ~200 | Annually | medium (0.85) |
| `validators/party_aliases.txt` | PARTY_ALIASES, PARTY_CODES | ~50 | Annually | high (0.98) |
| `sources/verified_sources.txt` | URL_ALLOWLIST_HOSTS, VERIFIED_SOURCES | ~100-200 | Weekly (during election season) | high (0.99) |
| `scoring/trust_signals.txt` | TRUST_SCORE_THRESHOLDS (confidence signals) | ~15-20 | Quarterly (tuning) | medium (0.80) |
| `scoring/anomaly_reasons.txt` | ANOMALY_QUARANTINE_RULES | ~12-15 | Quarterly (policy updates) | high (0.90) |
| `snapshots/table_headers.txt` | RESULT_COLUMN_HEADERS, COMMON_HEADER_VARIATIONS | ~100 | Bi-weekly (pattern observation) | high (0.95) |

---

## 12. Data Collection & Verification Strategy

### For each vocab file, establish authority

**High Trust (0.95+)**:

- Official government agencies (SoS, FEC)
- Authoritative public registries (FIPS, Census)
- Federal election standards (VVSG)

**Medium Trust (0.75-0.94)**:

- Manual curation from multiple sources
- Academic research (redistricting, election analysis)
- Professional election organizations

**Low Trust (0.50-0.74)**:

- Single-source data
- Crowdsourced information
- Legacy/unmaintained sources

**Very Low Trust (< 0.50)**:

- Unverified sources
- Deprecated formats
- Candidates for removal

---

## 13. Implementation Phases

### Phase 1: Extract & Catalog (Week 1)

- ✅ Enumerate all constants (this document)
- [ ] Classify by domain (jurisdictional, political, technical, etc.)
- [ ] Assign trust levels based on source authority
- [ ] Map to target vocab files
- [ ] Identify data source URLs

### Phase 2: Collect & Verify (Week 2)

- [ ] Gather data from authoritative sources
- [ ] Cross-validate across multiple sources
- [ ] Document source + verification method
- [ ] Compute SHA-256 hashes
- [ ] Create manifest.md

### Phase 3: Populate Vocab Files (Week 2-3)

- [ ] Create .txt files in compartmentalized structure
- [ ] Populate with canonical data
- [ ] Add header comments (source, trust score, update frequency)
- [ ] Verify hash checksums

### Phase 4: Implement VocabLoader (Week 3-4)

- [ ] Build vocab_loader.py class
- [ ] Add hash verification
- [ ] Add audit logging
- [ ] Comprehensive test suite (95%+ coverage)

### Phase 5: Migrate Handlers (Week 4-5)

- [ ] Update imports in all handlers
- [ ] Test backward compatibility
- [ ] Verify no regressions
- [ ] Phase out old API

### Phase 6: Production (Week 6)

- [ ] Staging deployment
- [ ] Production rollout
- [ ] Monitoring & post-election analysis

---

## 14. Frequently Asked Questions

**Q: Why move constants out of config.py?**  
A: Compartmentalization enables:

- Independent updates (update offices without redeploying code)
- Trust verification (verify source before accepting data)
- Election integrity (immutability during live scanning)
- Audit trails (provenance tracking for forensics)

**Q: What constants stay in config.py?**  
A: Technical/operational constants:

- File extensions + MIME types
- Handler registry
- Timeouts & resource limits
- Database configuration (via .env secrets)
- Feature flags
- Security policies
- Folder paths

**Q: What if a vocab file is corrupted?**  
A: VocabLoader detects corruption via SHA-256 hash verification. On mismatch:

1. Log anomaly (tampering suspected)
2. Quarantine data for manual review
3. Fall back to previous snapshot (if available)
4. Alert operations team

**Q: How are vocab files kept in sync across replicas?**  
A: Git-tracked + CI/CD deployment:

1. Vocab files committed to git (manifest.md includes hashes)
2. CI/CD verifies hashes on every deployment
3. Deployed to all replicas in lockstep
4. Election mode prevents out-of-sync modifications

**Q: Can vocab files be modified during elections?**  
A: No. During election_mode=True:

- All write operations blocked
- All reads logged (audit trail)
- Point-in-time snapshots created every 15 minutes
- Ensures reproducibility & forensics

---

## 15. Success Criteria

✅ **Inventory Complete** when:

- [ ] All constants cataloged
- [ ] Trust levels assigned
- [ ] Vocab files identified
- [ ] Data sources documented
- [ ] Migration priorities set
- [ ] Dependencies resolved (which constants depend on which?)

---

## Appendix: Constant Dependency Graph

```tree
STATES
  ├── STATE_ABBREVIATIONS
  ├── SUPPORTED_OFFICES_BY_STATE (depends on VALID_OFFICES)
  ├── COUNTIES_BY_STATE
  └── TIMEZONE_MAP

COUNTIES_BY_STATE
  ├── COUNTY_ALIASES
  └── PRECINCTS_BY_COUNTY

VALID_OFFICES
  ├── OFFICE_ALIASES
  └── SUPPORTED_OFFICES_BY_STATE

VALID_PARTIES
  ├── PARTY_ALIASES
  └── PARTY_CODES

VERIFIED_SOURCES
  ├── URL_ALLOWLIST_HOSTS
  └── TRUST_SCORE_THRESHOLDS

CONTEST_TYPES
  ├── ELECTION_TYPES
  └── MEASURE_TYPES

RESULT_COLUMN_HEADERS
  ├── COMMON_HEADER_VARIATIONS
  ├── NUMERIC_COLUMNS
  └── TEXT_COLUMNS
```

---

**Owner**: Election Integrity Team  
**Last Updated**: 2026-02-03  
**Next Review**: 2026-02-10 (after Phase 1 complete)  
**Questions**: Contact <election-integrity@example.org>
