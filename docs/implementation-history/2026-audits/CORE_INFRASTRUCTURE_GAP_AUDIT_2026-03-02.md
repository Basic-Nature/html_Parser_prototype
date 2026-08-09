# Core Infrastructure Gap Audit (2026-03-02)

<!-- markdownlint-disable-file -->

## Scope of this pass

This audit summarizes unfinished core infrastructure work after the current modularization wave (route wrapper extraction + Socket orchestration split).

Reviewed sources:
- docs/DEVELOPMENT/todos*.md
- docs/CORE/ARCHITECTURE.md
- docs/QUALITY/DATA_COMPARISON_ROADMAP.md
- docs/QUALITY/GOOGLE_SHEETS_MIGRATION.md
- docs/STATE_HANDLER_INTEGRATION.md
- docs/SMART_ELECTIONS_IMPLEMENTATION_PHASE3_COMPLETE.md
- key code TODO hotspots under `webapp/parser/*` and `webapp/Smart_Elections_Parser_Webapp.py`

---

## Executive summary

### What is now structurally strong

1. Monolith route decorators are fully wrapped through blueprint delegation maps.
2. Ballot Lens socket orchestration is extracted and staged internally.
3. Artifact-first completion signaling (CSV/XLSX/metadata awareness) is in place.

### What remains unfinished (core)

1. **Verification authorization depth is incomplete**
   - Tier enforcement is placeholder and currently accepts any authenticated principal.
2. **Data assurance quality metrics are partial**
   - Rejected-count and richer QA metrics are not fully wired.
3. **NER training path is placeholder-level**
   - Entity extraction/alignment in QA + BERT fine-tuning still uses placeholder mapping.
4. **Ground-truth comparison system is still planning-stage**
   - DL1/DL2 comparator + regression detection not implemented.
5. **State handler expansion + vendor base infrastructure is incomplete**
   - Coverage expansion and vendor base classes remain open technical debt.
6. **Workflow UI phases remain partially documented as pending**
   - Worklist/QC modal ecosystem and full integration path still listed as pending.
7. **Frontend test depth is thin in critical Ballot Lens areas**
   - Placeholder test scaffold exists without behavioral jsdom coverage.

### Pass A update (2026-03-02)

1. ✅ **Verifier tier enforcement implemented** in `verification_endpoints.py`:
  - Removed placeholder TODO bypass.
  - Added real `get_principal_tier(...)` authorization checks.
  - Added explicit 401 (unauthenticated) and 403 (under-tier) responses.
  - Added access-denied audit logging with required vs actual tier context.

2. ✅ **Verification tier tests added**:
  - `webapp/tests/test_verification_tier_enforcement.py`
  - Covers unauthenticated, under-tier denied, and sufficient-tier allowed paths.

3. ✅ **QA promotion authority hardened + rejected-count wired**:
  - `qa_endpoints.verify-and-promote` now requires `ADMIN_REVIEWER` minimum tier.
  - QA stats now return real `rejected_count` from `verified_data.verified_datasets`.
  - Monolith fallback `api_data_assurance_promote` now requires authenticated principal + `ADMIN_REVIEWER` tier and records reviewer principal from auth context.

4. ✅ **QA authority/stats tests added**:
  - `webapp/tests/test_qa_authority_and_stats.py`
  - Covers under-tier deny, admin reviewer allow, and rejected-count stats response wiring.

### Pass B update (2026-03-02)

1. ✅ **DataComparator MVP scaffolded**:
  - `webapp/parser/utils/data_comparator.py`
  - Implements candidate normalization + DL1/DL2 comparison (`exact`, `near`, `mismatch`, `missing`, `extra`) and vote diff summary.

2. ✅ **First regression-report contract added**:
  - `DataComparator.build_regression_report(...)` emits contract schema `1.0` with `summary`, `gate`, `mismatches`, and `context`.
  - `scripts/data_comparison_report.py` generates `output/reports/data_comparison_latest.json` and fails CI-style on gate failure unless `--soft` is used.

3. ✅ **Comparator tests added**:
  - `webapp/tests/test_data_comparator.py`
  - Covers exact/near/mismatch behavior and regression gate contract.

### Automation hardening update (2026-03-02)

1. ✅ **`automate.py` now supports optional DL comparison stage**:
  - Added `--compare-dl1-dl2`, `--dl1-path`, `--dl2-path`, threshold flags, and strict/soft controls.

2. ✅ **Structured automation run manifest added**:
  - `output/reports/automation_run_latest.json` now records stage results, stage details, strict mode, and critical failures.

3. ✅ **Per-stage stdout/stderr log artifacts added**:
  - `output/reports/logs/<stage>.stdout.log`
  - `output/reports/logs/<stage>.stderr.log`

4. ✅ **Subprocess stage execution unified**:
  - Shared execution path records command/cwd/duration/exit code for web checks, tests, self-check, ballot-lens-check, pipeline-check, and dl-compare.

5. ✅ **Webapp startup validation improved**:
  - Startup validation now performs real import check for `webapp.Smart_Elections_Parser_Webapp` instead of no-op success.

---

## Priority backlog (recommended)

## P0 — Security / governance completion

### 1) Enforce verifier privilege tiers (no placeholder auth)
- Code hotspot:
  - `webapp/parser/verification_endpoints.py`
- Current gap:
  - `_require_verifier_tier` includes TODO and bypasses tier checks.
- Expand with:
  - Principal-to-tier resolver integration.
  - Explicit deny logs + audit event schema for unauthorized attempts.
  - Endpoint-by-endpoint minimum tier matrix validation tests.

### 2) Harden QA promotion/review authority contracts
- Code hotspots:
  - `webapp/Smart_Elections_Parser_Webapp.py` (data assurance endpoints)
  - `webapp/parser/quality_assurance/qa_endpoints.py`
- Current gap:
  - Reviewer identity and rejection statistics are partially implemented.
- Expand with:
  - Rejected count query path.
  - Reviewer principal normalization and immutable chain-of-custody fields.
  - Negative-path tests for unauthorized promote actions.

---

## P1 — Data integrity / correctness infrastructure

### 3) Implement DL1 vs DL2 comparison engine
- Docs signaling unfinished:
  - `docs/QUALITY/DATA_COMPARISON_ROADMAP.md` (planning status)
  - `docs/QUALITY/GOOGLE_SHEETS_MIGRATION.md` (DataComparator + load script TODO)
- Expand with:
  - `webapp/parser/utils/data_comparator.py` baseline.
  - Structured diff report schema (exact/near/mismatch/missing/extra).
  - Regression threshold policy and CI gate mode.

### 4) Close migration-to-warehouse tooling gaps
- Current gap:
  - Migration docs reference scripts not yet implemented (`load_dl1_to_postgres.py`, parity tooling maturity).
- Expand with:
  - Canonical DL1 loader script with dry-run and checksum manifest.
  - Side-by-side parity verifier with deterministic output artifact.
  - Rollback command path codified into scripts.

---

## P1 — ML/NLP training quality completion

### 5) Replace placeholder NER extraction in QA ingestion
- Code hotspot:
  - `webapp/Smart_Elections_Parser_Webapp.py` (QA classify path writes `entities = []`)
- Expand with:
  - spaCy entity extraction path for sampled text rows.
  - Entity schema validation before DB write.
  - Sampling telemetry (how many rows/entities persisted).

### 6) Fix token/span alignment in BERT NER pipeline
- Code hotspot:
  - `webapp/parser/health/fine_tune_bert_ner.py`
- Current gap:
  - TODO notes indicate placeholder `ner_tags[0]` alignment.
- Expand with:
  - Character-offset to token-index aligner.
  - Unit tests for overlap/partial-token/edge offsets.
  - Training data quality report (invalid span ratio).

---

## P2 — Coverage / operability expansion

### 7) State handler and vendor coverage expansion
- Docs signaling unfinished:
  - `docs/STATE_HANDLER_INTEGRATION.md` (target 20/56, vendor base classes not implemented)
- Expand with:
  - Vendor base class abstractions for `--vendor` templates.
  - Generator-backed rollout plan for highest-volume states/counties.
  - Fallback crash-hardening in extraction retry paths.

### 8) Complete workflow UI phases and DB integration
- Docs signaling unfinished:
  - `docs/SMART_ELECTIONS_IMPLEMENTATION_PHASE3_COMPLETE.md` (Phase 4/5/6 pending)
- Expand with:
  - Worklist grid + QC modals with current backend contracts.
  - DB integration and migration verification scripts.
  - End-to-end workflow tests: assign -> preqc -> qc1 -> qc2 -> production.

### 9) Deepen frontend test coverage for Ballot Lens
- Code hotspot:
  - `webapp/static/js/__tests__/ballot_lens_modern.placeholder.test.js`
- Expand with:
  - JSDOM-based behavioral tests for placeholder/session helpers.
  - run-summary artifact rendering assertions.
  - socket event simulation for parser output/progress/download-ready.

---

## Suggested execution passes (next 3 passes)

### Pass A (security contracts)
1. Implement verifier tier enforcement.
2. Add unauthorized-path integration tests.
3. Wire rejected-count and reviewer identity guarantees.

### Pass B (data correctness)
1. Deliver `DataComparator` MVP.
2. Add DL1 load + parity verification scripts.
3. Add CI comparison report artifact.

### Pass C (ML + UX reliability)
1. Replace QA entity placeholders with spaCy extraction.
2. Fix NER token alignment and tests.
3. Upgrade Ballot Lens frontend tests from scaffold to behavioral coverage.

---

## Exit criteria for “core infrastructure complete”

1. No placeholder auth/tier enforcement in verification/QA promotion paths.
2. DL1 vs DL2 automated comparison runs in CI with thresholded pass/fail policy.
3. NER ingestion/training uses real span alignment, not placeholder tags.
4. Workflow UI phases marked pending are implemented or explicitly deferred with owners/timeline.
5. Ballot Lens critical UI/session behaviors have deterministic automated tests.
