# Infrastructure Plan (Draft)

## Goals

- Reliable local/dev/prod environments with reproducible builds and consistent dependencies.
- Safer deployments with clear rollout/rollback and health verification.
- Observable runtime (logs/metrics/alerts) and secure secrets handling.
- Faster developer feedback via automated checks.

## Scope

- Webapp runtime (Flask + Socket.IO + parser pipeline)
- Background jobs / health tasks
- Storage (input/output/uploads, logs, metadata)
- Database (Postgres) optional/guarded
- CI/CD and deployment environments

## Current Signals

- Local startup via `python -m webapp.Smart_Elections_Parser_Webapp`
- Tests run via `python -m pytest webapp/tests/`
- Config driven by `.env` + `webapp/parser/config.py`
- Logs written to `LOG_DIR` and JSONL files

## Phased Plan

Relationship mapping (single-paragraph framing): Treat each phase as a vector that supports downstream branches, with embedded relationships surfaced through safe_ helpers and contextual categorization; favor named vector labels in a 2:3 ratio to thematic labels (thematic at 1:3) to keep the field navigable while still conveying intent. Observability and security are upstream stabilizers for deployment gates; performance baselines anchor scaling; and ML/knowledge linking consumes verified signals from logs, metrics, and trusted sources, reinforcing the same map of dependencies.

### Phase 0 — Inventory & Baseline (1–2 days)

- Document current runtime inputs/outputs and required env vars.
- Capture minimum supported Python/Node versions and dependencies.
- Verify app startup + tests on clean machine.
- Deliverable: baseline checklist + “known-good” setup notes.
- Relationship mapping: establishes the dependency graph and vector labels used by later phases (observability, security, performance, deployment, ML).

### Phase 1 — Environment Hardening (2–4 days)

- Standardize environments:
  - Python: pin versions + lockfiles (pip-tools or uv).
  - Node: lockfile + consistent build scripts.
- Provide `make`/scripts for:
  - install
  - test
  - run
  - lint/type-check
- Deliverable: repeatable setup across dev machines.
- Relationship mapping: reproducible environments become the stability backbone for deployment gates and performance baselines.

### Phase 2 — Observability (2–4 days)

- Logs: ensure structured JSON logging is consistent and bounded.
- Metrics: enable Prometheus under feature flag; document counters.
- Health endpoints: `/health` and `/azure_health` documented with expected outputs.
- Deliverable: runbook + dashboard checklist.
- Relationship mapping: observability signals feed security monitoring, anomaly workflows, and deployment readiness checks.

### Phase 2b — Decision Gate Integration (1–2 days)

**Purpose:** Establish the nonpartisan confidence/caution decision framework and observability hooks for election result entity validation (offices, parties, jurisdictions, sources).

**Scope:**

- Decision engine: weighted confidence/caution calculation (2:1 signal to anomaly ratio).
- Parallel validation API: four safe_decide_* functions (jurisdiction, office, party, source).
- Vocabulary management: secure, audited loading of canonical entities + aliases.
- Decision logging: JSONL event streams with deduplication and session tracking.
- Metrics & observability: Prometheus counters for decision outcomes (PROCEED / CAUTION / STOP).
- Integration tests: 26+ pytest cases covering all Phase 2b modules.

**Deliverables:**

1. **Core Modules (Production-Ready)**
   - `entity_confidence_map.py` (510 lines): Weighted confidence engine with signal/anomaly types and override triggers.
   - `safe_decide.py` (250 lines): Parallel decision API returning DecisionTuple with 12-field audit trail.
   - `VocabLoader` (434 lines): Secure vocabulary file loading with integrity checks, rate limiting, and JSONL audit logs.
   - `DecisionTuple` (shared_logic.py): TypedDict for standardized decision output.

2. **Vocabulary Files (Entities + Validators + Sources)**
   - `entities/offices.txt`: 60+ canonical office titles (President, Governor, Senator, etc.).
   - `entities/parties.txt`: 10+ canonical party names (Democratic Party, Republican Party, etc.).
   - `entities/jurisdictions.txt`: 50 states + DC + sample counties.
   - `entities/contest_types.txt`: Additional contest type definitions.
   - `entities/result_terms.txt`: Election result terminology.
   - `validators/office_aliases.txt`: 25+ mappings (Pres → President, Gov → Governor).
   - `validators/party_aliases.txt`: 40+ mappings (Dem → Democratic Party, Rep → Republican Party).
   - `sources/verified_sources.txt`: 60+ verified election authority domains (sos.gov, fec.gov, elections.ny.gov, etc.).
   - `scoring/coefficients.txt`: Signal/anomaly/override weighting reference.

3. **Observability & Logging**
   - **Decision Filtering** (shared_logger.py): Deduplicates decision events (same entity + decision_code) within 5-minute window, preventing log spam during batch processing.
   - **Prometheus Metrics** (metrics_prom.py): Three decision counters with entity_type/reason/state labels:
     - `smart_decision_proceed_total`: Entities passed confidence checks (decision: PROCEED).
     - `smart_decision_caution_total`: Entities with mixed signals requiring manual review (decision: CAUTION).
     - `smart_decision_stop_total`: Entities failed confidence checks (decision: STOP).
   - **JSONL Audit Trail**: Every decision logged with session_id, timestamp, confidence, caution, signals, anomalies, and reasoning.

4. **Security Hardening**
   - Path traversal prevention: Safe path resolution via `.resolve().relative_to()`.
   - File validation: `.txt`-only filter; whitelist subdirs (entities, validators, sources, scoring).
   - Integrity checks: SHA256 file hashing; detection of file mutations.
   - Rate limiting: 60-second cooldown per vocab file reload to prevent brute-force attacks.
   - Session tracking: All decisions linked to session_id for audit and replay.

5. **Integration Tests** (test_phase_a_integration.py, 580+ lines)
   - 6 pytest fixtures for deterministic setup.
   - 6 test classes with 26+ test cases.
   - Coverage:
     - EntityConfidenceMap: singleton, signal/anomaly types, calculation formula.
     - SafeDecideAPI: decision routing, DecisionTuple structure, decision codes.
     - VocabLoader: canonical loading, cache hits, path security, integrity checks.
     - LoggerDecisionFiltering: deduplication within 5-minute window, cleanup.
     - PrometheusMetrics: counter increments for all three decision types.
     - PhaseAIntegration: end-to-end flows combining all modules.
   - Run: `pytest webapp/tests/test_phase_a_integration.py -v`

**Relationship Mapping:**

- Decision gate integration builds on observability from Phase 2, adding entity-level confidence scoring.
- Phase 2b outputs (decision metrics + audit logs) inform Phase 3 deployment gates.
- Vocabulary files and anomaly mappings provide semantic anchors for Phase 6 ML/knowledge linking.
- Parallel safe_decide API enables Phase 3 deployment gates without blocking existing result import paths.

**Rollout (Phase 2b → Phase 3 Gates):**

- Phase B Week 1 (pending approval): Decision logging only; no enforcement. Measurement period to collect baseline decision distributions.
- Phase B Week 2: Soft gate deployment; CAUTION decisions show UI badges but don't block user actions.
- Phase B Week 3+: Enforcement gates for high-risk scenarios (low-confidence offices, unverified sources, anomalies).

---

### Phase 3 — Deployment Workflow (3–6 days)

- Define dev/staging/prod parity (env vars and resource config).
- CI: add steps for tests + MyPy + JS checks.
- CD: add safe rollout (blue/green or slot-based) and rollback script.
- Deliverable: deploy checklist + scripted release flow.
- Relationship mapping: deployment gates consume observability + security inputs and surface performance thresholds for promotion/rollback.

### Phase 4 — Security & Secrets (2–5 days)

- Centralize secrets (env manager / Azure Key Vault / GitHub Actions secrets).
- Validate CSP + security headers in prod.
- Review file uploads and URL allowlist rules.
- Deliverable: security checklist + incident response notes.
- Relationship mapping: security signals are first-class events that enrich observability and protect ML/knowledge ingestion.

### Phase 5 — Performance & Scaling (2–5 days)

- Profile high-load endpoints and parser runs.
- Tune thread/worker counts and memory limits.
- Add caching strategy for downloads and output verification.
- Deliverable: performance baseline and scaling guidance.
- Relationship mapping: performance baselines close the loop on deployment readiness and inform ML pipeline capacity.

### Phase 6 — ML/NLP Advancement & Knowledge Linking (3–7 days)

- Contextual logging & anomaly mapping:
  - Define a normalized event schema for extraction, validation, and anomaly signals.
  - Persist anomaly traces with links to source URL, handler, contest, and table hashes.
  - Add anomaly “reason codes” to support targeted remediation and model feedback loops.
- Librarian + constants integration:
  - Move static constants into a curated text-based knowledge folder (e.g., tagged TXT files).
  - Treat each file as a versioned entity for database-backed memory and lookup.
  - Build a mapping index (entity → canonical label → validators) used by parsers and ML.
- Verified source anchoring:
  - Maintain a “verified sources” registry with link-level trust metadata.
  - Ingest and persist confirmed results to reduce re-processing and improve confidence scoring.
  - Use verified data to seed contextual comparisons and guard model drift.
- Direct URL search safeguards:
  - Gate unknown URLs behind enhanced trust scoring + stricter session-level controls.
  - Require contextual matches (state/county/contest) before accepting new data.
  - Quarantine uncertain outputs to prevent negative backscatter into memory.
- Deliverables:
  - Knowledge folder structure + indexing rules
  - Anomaly taxonomy + logging schema
  - Verified sources registry + ingestion flow
  - Direct URL safety gate policy
- Relationship mapping: ML/knowledge linking consumes verified sources plus observability and security signals to strengthen downstream trust.

#### Knowledge Base Folder Layout (TXT Entities)

Leverage the existing folder: `webapp/parser/Context_Integration/vocab`

Proposed layout (by context + persistence tier):

```text
webapp/parser/Context_Integration/vocab/
  README.md
  entities/
    offices.txt
    parties.txt
    jurisdictions.txt
    contest_types.txt
    result_terms.txt
  validators/
    county_aliases.txt
    state_aliases.txt
    candidate_name_suffixes.txt
    ballot_terms.txt
  sources/
    verified_sources.txt
    disallowed_hosts.txt
  scoring/
    trust_signals.txt
    anomaly_reasons.txt
  snapshots/
    schema_tokens.txt
    table_headers.txt
```

Notes:

- Each file is a newline-separated list of canonical tokens (optionally `alias -> canonical`).
- Keep a lightweight manifest in `README.md` describing each file’s purpose and schema.
- Transition `constants.py` into:
  - `constants.py` → thin helper module that loads from `vocab/*` and exposes typed accessors.
  - Consolidate keyword sets into files under `entities/` and `validators/`.
  - Store probability/relationship hints in `scoring/` for librarian retrieval.

#### Anomaly Mapping Schema (Minimum Fields)

Recommended JSON fields for anomaly events (logged + persisted):

```json
{
  "event_type": "anomaly",
  "reason_code": "mismatched_totals|missing_candidate|suspicious_header|...",
  "severity": "low|medium|high",
  "source_url": "...",
  "handler": "...",
  "contest": "...",
  "state": "...",
  "county": "...",
  "table_hash": "...",
  "row_count": 0,
  "column_count": 0,
  "observed_headers": ["..."],
  "expected_headers": ["..."],
  "confidence": 0.0,
  "session_id": "...",
  "timestamp": "ISO8601"
}
```

Integration notes:

- Librarian can use `reason_code` + `observed_headers` to retrieve context rules.
- `table_hash` links the anomaly to snapshot-derived signatures.
- Persist in JSONL for quick audit; promote to DB for long-term trend analysis.

## Risks & Mitigations

- **Runtime config drift** → enforce env templates + startup validation.
- **Dependency conflicts** → lockfiles + CI validation.
- **Log growth** → retention policy + rotation tooling.
- **Partial deploys** → health gate + rollback automation.

## Deliverables

- Environment setup guide
- Deployment runbook
- Health/metrics dashboard notes
- Security checklist
- Performance profile summary

## Next Steps (If approved)

1. Confirm target deployment platform (Azure App Service, VM, etc.).
2. Choose lockfile tooling for Python (pip-tools/uv/poetry).
3. Define CI pipeline requirements and environment matrix.
