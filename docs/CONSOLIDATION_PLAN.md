# Documentation Consolidation Plan

**Status**: Phase 2b Complete  
**Date**: 2026-02-05  
**Objective**: Consolidate 104 markdown files into categorical structure for GitHub Pages

---

## Executive Summary

**Current State**:

- 104 markdown files scattered across `docs/` with subdirectories
- Multiple duplicate/overlapping deployment guides
- Archived files mixed with active documentation
- Todo files in main directory (todos.md, todos_high.md, todos_medium.md, todos_low.md, todos_clean.md, todos_fixed.md)
- Status/completion files creating noise

**Target State**:

- Organized documentation by category (5-6 main categories)
- Single source of truth for each topic
- Clean GitHub Pages navigation
- Archived files properly separated
- Active development todos tracked in code comments + Issues, not MD files

---

## Phase 1: Documentation Audit & Categorization

### Category 1: Core Architecture & Design

**Purpose**: System design, data models, architecture patterns  
**Files to Consolidate Into**:

1. **`ARCHITECTURE.md`** (Master reference)
   - `architecture.md` ✓ Keep (rename to ARCHITECTURE.md for consistency)
   - `handlers.md` → Merge into ARCHITECTURE.md section
   - `pipeline_map.md` → Merge into ARCHITECTURE.md section
   - `VERIFICATION_ARCHITECTURE.md` → Merge into ARCHITECTURE.md (QA section)
   - `CONSTANTS_INVENTORY.md` → Keep as reference, link from ARCHITECTURE.md

2. **`DATA_MODELS.md`** (New - Consolidate data/schema docs)
   - `VERIFIED_DATA_SCHEMA.md` → Merge with PostgreSQL schema section
   - `SCHEMA_UNIFICATION_PROGRESS.md` → Archive (outdated progress notes)
   - `VERIFICATION_FRAMEWORK.md` → Merge (data model for verification)

---

### Category 2: Deployment & Operations

**Purpose**: Production deployment, operations runbooks, troubleshooting  
**Files to Consolidate Into**:

1. **`DEPLOYMENT.md`** (Single deployment guide)
   - `DEPLOYMENT_GUIDE.md` → Base content ✓
   - `PHASE2_AZURE_DEPLOYMENT.md` → Merge (Azure-specific section)
   - `AZURE_DEPLOYMENT_CHECKLIST.md` → Merge (checklist section, add to troubleshooting)
   - `PHASE2_DEPLOYMENT_CHECKLIST.md` → Merge (QA deployment section)
   - Delete: `PHASE2_QUICK_FIX.md` (covered in DEPLOYMENT.md quick start)

2. **`SECURITY.md`** (Security & authentication)
   - `AZURE_CERTIFICATE_AUTH_SETUP.md` → Full content ✓
   - `CERT_AUTH_IMPLEMENTATION.md` → Archive (superseded by new setup guide)
   - `CERT_AUTH_STEP5_CHECKLIST.md` → Archive (outdated)
   - Delete: `CERT_AUTH_*.md` (old step files)

3. **`OPERATIONS.md`** (Runbooks & monitoring)
   - `ELECTION_OPERATIONS_PLAYBOOK.md` → Keep ✓
   - `INTEGRITY_MONITORING.md` → Keep ✓
   - `WAREHOUSE_VERIFICATION_GUIDE.md` → Keep ✓
   - `troubleshooting.md` → Merge troubleshooting section

---

### Category 3: Quality Assurance & Verification

**Purpose**: DL1/DL2 verification framework, testing, validation  
**Files to Consolidate Into**:

1. **`VERIFICATION.md`** (QA verification framework)
   - `VERIFICATION_FRAMEWORK.md` → Keep ✓
   - `VERIFICATION_IMPLEMENTATION_COMPLETE.md` → Archive (status note)
   - `VERIFICATION_SYNC_IMPLEMENTATION.md` → Merge (implementation details)
   - `VERIFICATION_TESTING_GUIDE.md` → Merge (testing section)
   - `VERIFICATION_DOCUMENTATION_INDEX.md` → Archive (index no longer needed)

2. **`QUARANTINE_SYSTEM.md`** (Quarantine workflow)
   - `QUARANTINE_SYSTEM_GUIDE.md` → Keep ✓
   - `QUARANTINE_INDEX.md` → Merge (quick reference)
   - Delete: `*QUARANTINE*.md` archived files (old versions)

---

### Category 4: Quality Metrics & ML

**Purpose**: ML models, quality metrics, optimization  
**Files to Consolidate Into**:

1. **`ML_FRAMEWORK.md`** (ML & quality metrics)
   - `ML_QUICKSTART.md` → Keep ✓
   - `ML_OPTIMIZATION_METRICS.md` → Merge
   - `ML_OPTIMIZATION_SUMMARY.md` → Merge
   - `ML_QUALITY_METRICS_SUMMARY.md` → Merge
   - `ML_DEPLOYMENT_CHECKLIST.md` → Merge (as testing section)
   - Delete: duplicate optimizer files

2. **`CONFIDENCE_FRAMEWORK.md`** (Confidence & trust scoring)
   - `CONFIDENCE_CAUTION_FRAMEWORK.md` → Keep ✓
   - Archive: old quick reference files

---

### Category 5: Features & Implementation

**Purpose**: Feature documentation, how-to guides  
**Files to Consolidate Into**:

1. **`FEATURES.md`** (Active features)
   - `MODERN_UI_FEATURES.md` → Merge (UI features section)
   - `PARSER_UX_OPTIMIZATION_IDEAS.md` → Merge (UX improvements)
   - `UI_ENHANCEMENT_ROADMAP.md` → Archive (roadmap, move to issues)

2. **`GUIDES.md`** (How-to guides)
   - `HANDLER_MIGRATION_GUIDE.md` → Move/rename `HANDLER_DEVELOPMENT.md`
   - `fec_fuzzy.md` → Rename to `FEC_FUZZY_MATCHING.md` ✓
   - `VOCAB_MIGRATION_CHECKLIST.md` → Archive (completed task)
   - `VOCAB_LOADER_API_SPECIFICATION.md` → Merge into API docs

---

### Category 6: Guidelines & Governance

**Purpose**: Principles, governance, integrity guidelines  
**Files to Consolidate Into**:

1. **`GOVERNANCE.md`** (System governance)
   - `SYSTEM_GOVERNANCE.md` → Keep ✓
   - `Election_Integrity_Guidelines.md` → Keep ✓

---

## Phase 2: Redundant Files to Delete

### Status/Completion Files (Remove - tracked in git commits)

```txt
DELETE:
- IMPLEMENTATION_COMPLETE.md
- IMPLEMENTATION_COMPLETE_UI.md
- STEP1_TRUST_SCORER_COMPLETE.md
- STEP2_DOM_SNAPSHOT_COMPLETE.md
- MYPY_RESOLUTION_COMPLETE.md
- MYPY_CODE_CHANGES_REFERENCE.md
- TYPE_HINTS_AND_LEARNING_SYSTEM_FIX.md
- VERIFICATION_IMPLEMENTATION_COMPLETE.md
- PHASE2_QUICK_FIX.md (superseded by DEPLOYMENT.md)
- IMPLEMENTATION_STATUS.md
- MODAL_BANNER_AND_LOGGING_FIXES.md
- PDF_RESOURCE_CLEANUP.md
```

### Todo Files (Remove - use GitHub Issues + code comments)

```txt
DELETE:
- todos.md
- todos_high.md
- todos_medium.md
- todos_low.md
- todos_clean.md
- todos_fixed.md

REPLACE WITH:
- GitHub Issues board (organized by milestone)
- TODO comments in code (tracked by scripts/generate_todo_index.py)
```

### Old Phase/Implementation Reports

```txt
MOVE TO archived/:
- implementation-phases/PHASE_A_FINAL_REPORT.md
- implementation-phases/PHASE_A_IMPLEMENTATION_ROADMAP.md
- implementation-phases/PHASE_A_IMPLEMENTATION_SUMMARY.md
- implementation-phases/PHASE_12_COMPLETION_REPORT.md
- implementation-history/* (all files)
- session-logs/* (all files except latest summary)
```

### Outdated Deployment Guides

```txt
DELETE/ARCHIVE:
- PHASE2_MULTITENANT_IMPLEMENTATION.md (not in current roadmap)
- INFRASTRUCTURE_PLAN.md (outdated)
- INFRASTRUCTURE_PLAN.md (old version)
```

---

## Phase 3: Directory Structure

### Current Structure

```tree
docs/
├── (root - 50+ files)
├── archived/ (20+ old files)
├── implementation-history/ (5 files)
├── implementation-phases/ (4 files)
├── session-logs/ (5 files)
└── quick_reference.html
```

### New Structure

```tree
docs/
├── _index.md ⭐ NEW - Main navigation hub
├── README.md (GitHub Pages landing page)
│
├── CORE/
│   ├── ARCHITECTURE.md (system design, handlers, pipeline)
│   ├── DATA_MODELS.md (schemas, verification framework)
│   └── CONSTANTS.md (quick reference)
│
├── DEPLOYMENT/
│   ├── DEPLOYMENT.md (all deployment guides, checklists)
│   ├── SECURITY.md (certificate auth, authentication)
│   └── OPERATIONS.md (runbooks, monitoring, troubleshooting)
│
├── QUALITY/
│   ├── VERIFICATION.md (QA framework, testing)
│   ├── QUARANTINE_SYSTEM.md (quarantine workflow)
│   └── ML_FRAMEWORK.md (ML models, quality metrics)
│
├── FEATURES/
│   ├── GUIDES.md (how-to guides, handler development)
│   ├── FEC_FUZZY_MATCHING.md
│   ├── CONFIDENCE_FRAMEWORK.md
│   ├── ELECTION_OPERATIONS.md
│   └── INTEGRITY_GUIDELINES.md
│
├── GOVERNANCE/
│   └── GOVERNANCE.md (system governance, principles)
│
├── archived/ (OLD FILES)
│   ├── implementation-history/
│   ├── implementation-phases/
│   ├── session-logs/
│   └── OLD_*.md (renamed old files)
│
└── _github_pages/ (for static site generation)
    ├── _config.yml
    ├── index.html
    └── navigation.json
```

---

## Phase 4: Migration Checklist

### Step 1: Create New Directory Structure

- [ ] Create `docs/CORE/` directory
- [ ] Create `docs/DEPLOYMENT/` directory
- [ ] Create `docs/QUALITY/` directory
- [ ] Create `docs/FEATURES/` directory
- [ ] Create `docs/GOVERNANCE/` directory

### Step 2: Consolidate Files

- [ ] Merge `handlers.md`, `pipeline_map.md` → `ARCHITECTURE.md`
- [ ] Merge `VERIFIED_DATA_SCHEMA.md` → `DATA_MODELS.md`
- [ ] Merge `VERIFICATION_FRAMEWORK.md` → `DATA_MODELS.md`
- [ ] Merge deployment checklists → `DEPLOYMENT.md`
- [ ] Merge QuickStart sections → `DEPLOYMENT.md` quick start
- [ ] Merge `VERIFICATION_SYNC_IMPLEMENTATION.md` → `VERIFICATION.md`
- [ ] Merge `VERIFICATION_TESTING_GUIDE.md` → `VERIFICATION.md`
- [ ] Merge ML files → `ML_FRAMEWORK.md`

### Step 3: Archive Old Files

- [ ] Copy `implementation-history/` to `archived/implementation-history/`
- [ ] Copy `implementation-phases/` to `archived/implementation-phases/`
- [ ] Copy `session-logs/` to `archived/session-logs/`
- [ ] Rename cert auth old files → `archived/OLD_CERT_AUTH_*.md`
- [ ] Rename phase files → `archived/OLD_PHASE*.md`

### Step 4: Delete Redundant Files

- [ ] Delete `todos*.md` (5 files)
- [ ] Delete `*COMPLETE.md` status files (10 files)
- [ ] Delete `PHASE2_QUICK_FIX.md` (merged into DEPLOYMENT.md)
- [ ] Delete old implementation history in root

### Step 5: Create Index & Navigation

- [ ] Create `_index.md` (main documentation hub)
- [ ] Update `README.md` to point to docs
- [ ] Create GitHub Pages `navigation.json`
- [ ] Create `.github/pages/_config.yml` with sidebar navigation

### Step 6: Commit & Verify

- [ ] Run linting check on all MD files
- [ ] Build GitHub Pages locally to verify navigation
- [ ] Commit: "refactor(docs): Consolidate into categorical structure"
- [ ] Verify GitHub Pages builds successfully

---

## Key Benefits of Consolidation

1. **Clarity**: Single source of truth per topic
2. **Maintainability**: Easier to keep docs in sync with code
3. **Discovery**: Better navigation via GitHub Pages
4. **Scalability**: Clear structure for new documentation
5. **Reduced Noise**: Removed 60+ status/completion files
6. **Versioning**: Old docs safely archived

---

## File Mapping Reference

| Old File | New Location | Action |
| ---------- | -------------- | -------- |
| architecture.md | CORE/ARCHITECTURE.md | Rename |
| handlers.md | CORE/ARCHITECTURE.md | Merge into section |
| pipeline_map.md | CORE/ARCHITECTURE.md | Merge into section |
| VERIFIED_DATA_SCHEMA.md | CORE/DATA_MODELS.md | Move & merge |
| VERIFICATION_FRAMEWORK.md | QUALITY/VERIFICATION.md | Move & keep title |
| DEPLOYMENT_GUIDE.md | DEPLOYMENT/DEPLOYMENT.md | Move & rename |
| AZURE_DEPLOYMENT_CHECKLIST.md | DEPLOYMENT/DEPLOYMENT.md | Merge into checklist section |
| PHASE2_AZURE_DEPLOYMENT.md | DEPLOYMENT/DEPLOYMENT.md | Merge into Azure section |
| PHASE2_DEPLOYMENT_CHECKLIST.md | DEPLOYMENT/DEPLOYMENT.md | Merge into QA section |
| AZURE_CERTIFICATE_AUTH_SETUP.md | DEPLOYMENT/SECURITY.md | Move & rename |
| ELECTION_OPERATIONS_PLAYBOOK.md | FEATURES/ELECTION_OPERATIONS.md | Move & keep |
| Election_Integrity_Guidelines.md | FEATURES/INTEGRITY_GUIDELINES.md | Move & keep |
| QUARANTINE_SYSTEM_GUIDE.md | QUALITY/QUARANTINE_SYSTEM.md | Move & keep |
| ML_QUICKSTART.md | QUALITY/ML_FRAMEWORK.md | Keep heading |
| CONFIDENCE_CAUTION_FRAMEWORK.md | FEATURES/CONFIDENCE_FRAMEWORK.md | Keep & rename |
| SYSTEM_GOVERNANCE.md | GOVERNANCE/GOVERNANCE.md | Move & keep |
| HANDLER_MIGRATION_GUIDE.md | FEATURES/GUIDES.md | Move & rename section |
| fec_fuzzy.md | FEATURES/FEC_FUZZY_MATCHING.md | Rename & move |
| MODERN_UI_FEATURES.md | FEATURES/GUIDES.md | Merge as section |
| troubleshooting.md | DEPLOYMENT/OPERATIONS.md | Merge |
| INTEGRITY_MONITORING.md | DEPLOYMENT/OPERATIONS.md | Merge |
| WAREHOUSE_VERIFICATION_GUIDE.md | DEPLOYMENT/OPERATIONS.md | Merge |

---

## Next Steps

1. Review this plan for approval
2. Execute Phase 1-3 migrations
3. Test GitHub Pages locally
4. Commit consolidated structure
5. Delete archived directory from main branch (keep in git history)

**Estimated Time**:

- Planning: 30 min ✓
- Execution: 2-3 hours
- Testing: 30 min
- **Total**: ~4 hours
