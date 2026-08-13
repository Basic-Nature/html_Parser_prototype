# Contributing to Election Pulse

Thank you for contributing to Election Pulse.

Election Pulse is an evidence-backed election data platform for acquiring, parsing, normalizing, validating, and analyzing election information across heterogeneous public sources.

Contributions are welcome across parsing, data modeling, validation, testing, documentation, security, accessibility, visualization, deployment, and election-data research.

Because election data is sensitive to interpretation and small transformations can materially affect downstream analysis, Election Pulse places particular emphasis on **traceability, reproducibility, and preserving source evidence**.

---

## Before You Contribute

Please become familiar with the project's core principles:

1. Preserve source evidence.
2. Never invent missing election data.
3. Distinguish zero values from missing values.
4. Prefer structured official sources when available.
5. Normalize without destroying the original representation.
6. Keep source-specific behavior modular.
7. Centralize reusable parsing and validation logic.
8. Do not promote parser observations directly into trusted knowledge.
9. Surface discrepancies rather than silently correcting them.
10. Keep important transformations reproducible and auditable.

These principles take priority over making a parser merely "work" for one source.

---

## Repository Direction

Election Pulse is undergoing architectural consolidation.

Some parts of the repository predate the current domain-oriented architecture and may contain:

* legacy interfaces
* duplicated parsing logic
* experimental utilities
* historical context-library behavior
* transitional storage patterns
* source-specific assumptions
* deprecated documentation
* temporary debugging tools

Finding one of these patterns does not necessarily mean it should be copied into new code.

When modifying an older component, first determine whether an existing shared abstraction should now own that behavior.

The authoritative architecture is progressively being documented under:

```text
docs/ARCHITECTURE/
```

Historical implementation documents belong under:

```text
docs/implementation-history/
```

Historical documentation should not be treated as the current architectural contract unless explicitly identified as such.

---

## Architectural Domains

Changes should have a clear architectural owner.

The major Election Pulse domains are:

```text
Acquisition
    ↓
Parsing
    ↓
Normalization
    ↓
Canonical Election Model
    ↓
Evidence / Validation / Context
    ↓
Analysis / APIs / Presentation
```

Supporting domains include:

```text
Integrity
Storage
Automation
Security
Observability
Deployment
```

Before introducing a new module or abstraction, ask:

> Which domain owns this responsibility?

If the answer is unclear, the design should usually be clarified before additional implementation is added.

---

## Parser Contributions

Election sources vary substantially between jurisdictions and vendors.

A parser contribution may involve:

* HTML or DOM extraction
* structured JSON or API parsing
* CSV or spreadsheet parsing
* PDF extraction
* OCR
* vendor-specific behavior
* state-specific behavior
* county-specific behavior
* dynamic fallback behavior

Source-specific handlers should identify differences in the source while relying on shared components for behavior that is not jurisdiction-specific.

### Prefer Shared Logic

Do not duplicate generic behavior inside a state or county handler when it belongs in shared parser infrastructure.

Examples include:

* candidate normalization
* vote-method normalization
* table interpretation
* evidence creation
* totals calculation
* metadata cleaning
* output construction
* discrepancy detection
* common DOM operations

A handler should primarily describe what makes its source different.

---

## Prefer Structured Sources

When the same official election information is available through multiple representations, prefer the most structured reliable source.

A typical preference order is:

```text
Official API / JSON
        ↓
CSV / Spreadsheet
        ↓
Structured HTML
        ↓
PDF with selectable data
        ↓
OCR / image extraction
```

This is a guideline rather than an absolute rule.

Source authority, completeness, provenance, and reporting structure must also be considered.

Do not replace a higher-quality official source with a visually convenient but less reliable representation.

---

## Canonical Election Data

Parser implementations should increasingly produce or feed a common canonical election representation.

Conceptually:

```text
Election
├── Jurisdictions
├── Precincts
├── Contests
│   ├── Candidates / Choices
│   ├── Vote Methods
│   └── Results
├── Evidence
├── Provenance
└── Validation
```

Downstream systems should not independently reinterpret raw parser structures when the canonical layer can provide that interpretation once.

When introducing a new field, consider whether it represents:

* source evidence
* canonical election data
* derived analysis
* validation state
* runtime state
* learned knowledge
* presentation metadata

These categories should not be mixed casually.

---

## Precinct Output Requirements

Precinct-level output must preserve cross-precinct comparability.

Each precinct should retain every known candidate and applicable vote method, including legitimate zero values.

Typical vote methods include:

```text
Election Day
Early Voting
Absentee Mail
Provisional
```

Additional methods may be introduced when the source reports them.

For each candidate, the output should preserve the available method breakdown and candidate total.

Never remove a candidate or method solely because its value is zero.

### Zero Is Not Missing

These states are semantically different:

```text
0
missing
unavailable
not reported
not applicable
parse failure
```

Do not silently convert one into another.

---

## Evidence and Provenance

Parser evidence must remain distinguishable from canonical knowledge.

For example:

```text
Raw source value
        ↓
Parser observation
        ↓
Normalization / resolution
        ↓
Canonical value
```

Where appropriate, evidence should preserve information such as:

* source URL
* source file
* raw value
* extraction method
* DOM location
* table relationship
* OCR region
* jurisdiction
* parser or rule version
* timestamp
* confidence
* normalization rule
* review status

Do not discard useful raw evidence after normalization merely because the normalized value appears correct.

---

## Context and Learned Knowledge

Election Pulse distinguishes among:

```text
Canonical Knowledge
Learned Knowledge
Runtime State
Parser Evidence
Generated Indexes
Telemetry
```

These categories must not be treated as interchangeable storage.

### Parser Evidence Is Not Knowledge

A parser discovering a pattern during one run does not automatically make that pattern authoritative.

### Learned Knowledge Requires Promotion

Learned observations should be:

* attributable
* confidence-aware
* reviewable
* reproducible
* explicitly approved where required

before becoming trusted reusable context.

### Runtime State Does Not Belong in Canonical Context

Temporary information such as:

* migration state
* telemetry
* diagnostic output
* temporary parser evidence
* cache state
* process locks

must not silently become canonical knowledge.

---

## Election Integrity and Anomaly Detection

Election Pulse may identify unusual patterns, reconciliation failures, or statistical anomalies.

Contributors must preserve the distinction between:

```text
Observation
    ↓
Evidence
    ↓
Validation
    ↓
Interpretation
    ↓
Conclusion
```

An anomaly does not by itself establish misconduct, fraud, error, or any other cause.

Code and documentation should describe what the data demonstrates without assigning unsupported explanations.

Prefer language such as:

```text
discrepancy detected
reconciliation failed
unexpected ratio
requires review
source mismatch
insufficient evidence
```

over unsupported causal conclusions.

---

## Validation Requirements

Election data transformations should be validated wherever practical.

Important checks include:

* candidate totals equal method totals
* contest totals reconcile
* precinct totals reconcile
* reported ballot-method totals reconcile
* duplicate precincts are detected
* candidates are not silently omitted
* vote methods are not silently omitted
* missing values remain distinguishable from zero
* parser failures remain visible
* source metadata is preserved

A failed reconciliation should normally produce a validation signal rather than being silently repaired.

---

## PDF and OCR Contributions

OCR should be treated as an evidence-generation process.

OCR output is not automatically authoritative election data.

Contributions involving PDF or OCR processing should account for conditions such as:

* scanned pages
* selectable text
* handwritten marks
* rotated pages
* degraded images
* page-spanning tables
* mixed layouts
* ambiguous characters
* incomplete extraction

When confidence is insufficient, preserve the ambiguity for review.

Do not manufacture a clean value merely because downstream code expects one.

---

## Testing

New behavior should normally include tests.

Tests may cover:

* unit behavior
* parser contracts
* normalization
* jurisdiction handlers
* structured-source parsing
* fallback behavior
* reconciliation
* evidence preservation
* security
* frontend behavior
* OCR behavior
* integration pathways

Run the Python test suite with:

```bash
python -m pytest
```

For targeted development, run the smallest relevant suite first and expand validation before merging.

Example:

```bash
python -m pytest webapp/tests/test_context_write_policy.py -q
```

Also verify changed Python modules compile:

```bash
python -m compileall webapp
```

Before committing, check the diff for whitespace errors:

```bash
git diff --check
```

Not every local environment currently contains every optional testing dependency. If a test cannot run because of an environment dependency, document that limitation rather than reporting the suite as passing.

---

## Test Data

Election test fixtures should use real or explicitly controlled source data.

Do not fabricate election totals for parser previews or validation demonstrations unless a test explicitly requires synthetic data and clearly identifies it as synthetic.

Regression fixtures should remain:

* small when possible
* deterministic
* attributable
* reviewable
* free of secrets

Large generated datasets should not automatically be committed to Git.

---

## Git LFS

Git LFS should be reserved for files that genuinely benefit from large-file storage.

Good candidates may include:

* large generated election indexes
* model weights
* large binary datasets
* checkpoints
* large immutable artifacts

Ordinary files such as these generally belong in normal Git:

```text
small JSON
JSONL configuration
Python
JavaScript
CSS
Markdown
vocabularies
schemas
small fixtures
```

Avoid broad rules such as:

```text
*.json
*.jsonl
```

because they make ordinary configuration and reference data difficult to review.

Prefer explicit paths for known large artifacts.

---

## Runtime and Generated Files

Do not commit runtime state unless it is intentionally maintained as a fixture or reference artifact.

Examples that generally should remain untracked include:

* process locks
* PID files
* telemetry
* temporary parser output
* OCR diagnostics
* generated caches
* local databases
* local environment files
* temporary migration state
* debug output

If a generated artifact must be committed, document why it belongs in source control.

---

## Secrets and Credentials

Never commit:

* passwords
* private keys
* production certificates
* API secrets
* access tokens
* database credentials
* populated local `.env` files

Use:

```text
.env.template
```

to document required environment variables without including sensitive values.

If you discover a committed credential, treat it as compromised and follow the project's security process.

See:

```text
SECURITY.md
```

for security reporting guidance.

---

## Code Organization

Reusable logic should have one clear owner.

Before adding a helper, search for an existing implementation.

Avoid creating parallel versions such as:

```text
normalize_candidate()
normalize_candidate_name()
clean_candidate()
canonicalize_candidate()
fix_candidate_name()
```

unless those functions intentionally represent different stages.

Prefer one documented contract over several subtly different implementations.

---

## Logging

Use the project's shared logging infrastructure rather than ad hoc debugging output.

Avoid committing temporary statements such as:

```python
print(...)
```

or:

```javascript
console.log(...)
```

unless they intentionally belong to supported CLI or frontend behavior.

Runtime verbosity should respect the project's configured logging behavior and environment settings.

Do not log secrets or sensitive authentication material.

---

## Configuration

Behavior that differs between environments should normally be configurable rather than hardcoded.

Examples include:

* logging level
* output generation
* authentication behavior
* deployment environment
* diagnostics
* optional parser behavior

Use existing configuration mechanisms before introducing another configuration source.

---

## Documentation Contributions

Documentation should describe the current system unless it is explicitly historical.

### Current Architecture

Place authoritative architecture documentation under:

```text
docs/ARCHITECTURE/
```

### Core Contracts

Stable implemented contracts and reference material belong under:

```text
docs/CORE/
```

### Historical Documentation

Superseded designs, implementation summaries, and migration history belong under:

```text
docs/implementation-history/
```

Do not leave an obsolete design beside the current architecture without clearly identifying which one is authoritative.

### Documentation Should Explain Why

Useful documentation should explain:

* responsibility
* inputs
* outputs
* contracts
* boundaries
* invariants
* failure behavior
* interaction with other domains

Avoid documentation that merely reproduces the current function list.

---

## Markdown

Markdown in the repository should pass the configured Markdown linting rules.

Keep:

* heading levels consistent
* fenced code blocks labeled
* lists formatted consistently
* links relative where appropriate
* trailing whitespace removed
* excessively implementation-specific documentation out of root files

The root Markdown files should remain concise entry points.

Detailed implementation documentation belongs under `docs/`.

---

## Branch and Commit Practices

Keep commits focused.

Prefer:

```text
one architectural concern
one bug fix
one refactor
one documentation consolidation
```

over mixing unrelated changes into a single commit.

Examples:

```text
fix: preserve missing vote methods during normalization

refactor: separate parser evidence from learned context persistence

docs: consolidate canonical election architecture

test: add reconciliation coverage for precinct totals

chore: narrow LFS tracking for parser fixtures
```

Before committing:

```bash
git status --short
git diff --check
git diff --cached --check
```

Review what is actually staged:

```bash
git diff --cached --name-status
git diff --cached --stat
```

Do not rely on a broad "stage all" operation when unrelated runtime, generated, or experimental files are present.

---

## Pull Requests

A pull request should make it possible for another contributor to understand:

1. What changed?
2. Why was the change necessary?
3. Which architectural domain owns the behavior?
4. What evidence or source exposed the problem?
5. How was the change validated?
6. Does it alter a canonical contract?
7. Does it affect stored or learned context?
8. Are there known limitations?

For parser changes, include the jurisdiction, vendor, source format, or representative source involved when appropriate.

For validation fixes, explain the discrepancy being detected or resolved.

For architecture changes, explain which responsibility is moving and why.

---

## Adding Jurisdiction Support

Do not begin by copying an entire existing county handler.

First determine:

```text
What is already generic?
        ↓
What is vendor-specific?
        ↓
What is state-specific?
        ↓
What is county-specific?
```

Only the genuinely jurisdiction-specific behavior should live in the narrowest handler.

If multiple jurisdictions require the same workaround, that is usually evidence that the behavior belongs in a shared or vendor-level component.

---

## Adding a New Vote Method

If a source introduces an additional legitimate vote method, preserve it.

Do not force every source into only:

```text
Election Day
Early Voting
Absentee Mail
Provisional
```

Those are common methods, not an exhaustive ontology.

New methods should be normalized through the canonical vote-method system while preserving the source terminology and evidence.

---

## Adding a New Candidate or Contest Pattern

Do not hardcode a new alias into an unrelated parser merely to make one page work.

Determine whether the information belongs in:

```text
canonical vocabulary
jurisdiction-specific rule
vendor rule
learned context
parser heuristic
```

and preserve provenance for learned additions.

This distinction is important for preventing source-specific observations from becoming global assumptions.

---

## Definition of Done

A contribution is generally ready when:

* the responsibility belongs in the correct domain
* shared behavior has not been unnecessarily duplicated
* source evidence is preserved
* missing values remain distinct from zero
* canonical structures remain compatible or intentionally versioned
* relevant validation exists
* relevant tests pass
* documentation is updated when contracts change
* runtime artifacts are not accidentally committed
* secrets are absent
* the staged diff contains only intended changes
* the change can be explained and reproduced by another contributor

---

## Questions and Architectural Changes

For substantial changes, prefer discussing the architecture before building a second competing implementation.

This is particularly important for changes involving:

* the Canonical Election Model
* evidence structures
* context persistence
* parser contracts
* storage
* authentication
* validation semantics
* ML/NLP promotion
* jurisdiction routing
* output contracts

The objective is not to prevent experimentation.

It is to prevent successful experiments from becoming permanent architectural fragmentation.

---

## Final Principle

When uncertain about how Election Pulse should handle election information, prefer the design that preserves the most trustworthy path back to the original evidence.

**Acquire the source. Preserve the evidence. Normalize the data. Validate the result. Make it auditable.**
